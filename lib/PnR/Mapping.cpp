#include "PnR/Mapping.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>

using namespace loom::pnr;

namespace {

enum class ResourceKind {
  FabricOp,
  MemLoad,
  MemStore,
};

struct SoftwareNode {
  std::string id;
  std::string operation;
  ResourceKind resourceKind;
  mlir::Operation *op = nullptr;
};

struct HardwareResource {
  std::string id;
  ResourceKind kind;
  std::string schedule;
  std::map<std::string, std::string> swConfigs;
  llvm::StringSet<> supportedOps;
  bool used = false;
};

std::string csvEscape(llvm::StringRef value) {
  if (value.find_first_of(",\"\n\r") == llvm::StringRef::npos)
    return value.str();
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"')
      escaped.push_back('"');
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

llvm::Error createParentDirectories(llvm::StringRef outputPath) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (parent.empty())
    return llvm::Error::success();
  if (std::error_code ec = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(ec, "could not create %s", parent.c_str());
  return llvm::Error::success();
}

std::string mappingId(llvm::StringRef workload, llvm::StringRef hardware) {
  std::string id = workload.str();
  id += "__";
  for (char ch : hardware) {
    if (llvm::isAlnum(ch) || ch == '_')
      id.push_back(ch);
    else
      id.push_back('_');
  }
  return id;
}

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef path) {
  return mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
}

std::optional<std::string> symbolName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::Operation *findSymbolOp(mlir::ModuleOp module, llvm::StringRef opName,
                              llvm::StringRef symbol) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != opName)
      return;
    std::optional<std::string> name = symbolName(op);
    if (name && *name == symbol)
      found = op;
  });
  return found;
}

bool isIgnoredOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "dataflow.graph.return";
}

bool isAdapterOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "builtin.unrealized_conversion_cast" ||
         name == "arith.index_cast";
}

std::optional<ResourceKind> resourceKindForSoftwareOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "dataflow.load")
    return ResourceKind::MemLoad;
  if (name == "dataflow.store")
    return ResourceKind::MemStore;
  if (fabric::isFabricOpSupported(name))
    return ResourceKind::FabricOp;
  return std::nullopt;
}

llvm::StringRef resourceKindName(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return "fabric.op";
  case ResourceKind::MemLoad:
    return "fabric.mem.load";
  case ResourceKind::MemStore:
    return "fabric.mem.store";
  }
  llvm_unreachable("unknown resource kind");
}

llvm::Expected<llvm::SmallVector<SoftwareNode>>
collectSoftwareNodes(mlir::Operation *graph) {
  llvm::SmallVector<SoftwareNode> nodes;
  llvm::StringMap<unsigned> counts;
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    if (isIgnoredOp(&op) || isAdapterOp(&op))
      continue;
    std::optional<ResourceKind> kind = resourceKindForSoftwareOp(&op);
    if (!kind) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "graph contains unsupported operation for PnR mapping: %s",
          op.getName().getStringRef().str().c_str());
    }
    std::string opName = op.getName().getStringRef().str();
    unsigned index = counts[opName]++;
    nodes.push_back(
        SoftwareNode{opName + "#" + std::to_string(index), opName, *kind, &op});
  }
  return nodes;
}

std::uint64_t integerAttrValue(mlir::Attribute attr) {
  if (auto intAttr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return static_cast<std::uint64_t>(intAttr.getInt());
  return 0;
}

std::string configValue(mlir::Attribute attr) {
  if (auto stringAttr = llvm::dyn_cast_if_present<mlir::StringAttr>(attr))
    return stringAttr.getValue().str();
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

std::string scheduleName(fabric::Schedule schedule) {
  return fabric::stringifySchedule(schedule).str();
}

std::string nearestSchedule(mlir::Operation *op) {
  for (mlir::Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    if (auto attr = cursor->getAttrOfType<fabric::ScheduleAttr>("schedule"))
      return scheduleName(attr.getValue());
  }
  return "spatial";
}

void appendMemResources(mlir::Operation *op, llvm::StringRef hardwareName,
                        llvm::SmallVectorImpl<HardwareResource> &resources) {
  std::uint64_t loadPorts = 0;
  std::uint64_t storePorts = 0;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (hwParams && !hwParams.empty()) {
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0])) {
      loadPorts = integerAttrValue(dict.get("load_group_size"));
      storePorts = integerAttrValue(dict.get("store_group_size"));
    }
  }
  std::string schedule = nearestSchedule(op);
  for (std::uint64_t i = 0; i < loadPorts; ++i) {
    resources.push_back(
        HardwareResource{(hardwareName + "::mem.load#" + llvm::Twine(i)).str(),
                         ResourceKind::MemLoad,
                         schedule,
                         {},
                         {},
                         false});
  }
  for (std::uint64_t i = 0; i < storePorts; ++i) {
    resources.push_back(
        HardwareResource{(hardwareName + "::mem.store#" + llvm::Twine(i)).str(),
                         ResourceKind::MemStore,
                         schedule,
                         {},
                         {},
                         false});
  }
}

void appendFabricOpResource(
    mlir::Operation *op, llvm::StringRef hardwareName, unsigned index,
    llvm::SmallVectorImpl<HardwareResource> &resources) {
  auto opList = op->getAttrOfType<mlir::ArrayAttr>("op_list");
  if (!opList)
    return;
  HardwareResource resource;
  resource.id = (hardwareName + "::fabric.op#" + llvm::Twine(index)).str();
  resource.kind = ResourceKind::FabricOp;
  resource.schedule = nearestSchedule(op);
  if (auto swConfigs = op->getAttrOfType<mlir::DictionaryAttr>("sw_configs")) {
    for (mlir::NamedAttribute namedAttr : swConfigs) {
      resource.swConfigs[namedAttr.getName().getValue().str()] =
          configValue(namedAttr.getValue());
    }
  }
  for (mlir::Attribute attr : opList) {
    if (auto sym = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      resource.supportedOps.insert(sym.getValue());
  }
  resources.push_back(std::move(resource));
}

llvm::Expected<llvm::SmallVector<HardwareResource>>
collectHardwareResources(mlir::Operation *hardware, llvm::StringRef name) {
  llvm::SmallVector<HardwareResource> resources;
  unsigned fabricOpIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.op") {
      appendFabricOpResource(op, name, fabricOpIndex++, resources);
      return;
    }
    if (opName == "fabric.mem")
      appendMemResources(op, name, resources);
  });
  if (resources.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "hardware has no mappable resources");
  return resources;
}

HardwareResource *
claimResource(SoftwareNode &node,
              llvm::MutableArrayRef<HardwareResource> resources) {
  for (HardwareResource &resource : resources) {
    if (resource.used || resource.kind != node.resourceKind)
      continue;
    if (resource.kind == ResourceKind::FabricOp &&
        !resource.supportedOps.contains(node.operation))
      continue;
    resource.used = true;
    return &resource;
  }
  return nullptr;
}

std::optional<std::string> configFor(const HardwareResource &resource,
                                     llvm::StringRef key) {
  auto it = resource.swConfigs.find(key.str());
  if (it == resource.swConfigs.end())
    return std::nullopt;
  return it->second;
}

llvm::DenseMap<mlir::Operation *, std::string>
indexNodeIds(llvm::ArrayRef<SoftwareNode> nodes) {
  llvm::DenseMap<mlir::Operation *, std::string> byOperation;
  for (const SoftwareNode &node : nodes)
    byOperation.try_emplace(node.op, node.id);
  return byOperation;
}

struct RouteBuilder {
  struct ProducerRef {
    std::string softwareId;
    unsigned resultIndex = 0;
  };

  struct EdgeKey {
    std::string producerSoftwareId;
    unsigned producerResultIndex = 0;
    std::string consumerSoftwareId;
    unsigned consumerOperandIndex = 0;

    bool operator<(const EdgeKey &other) const {
      return std::tie(producerSoftwareId, producerResultIndex,
                      consumerSoftwareId, consumerOperandIndex) <
             std::tie(other.producerSoftwareId, other.producerResultIndex,
                      other.consumerSoftwareId, other.consumerOperandIndex);
    }
  };

  llvm::DenseMap<mlir::Value, ProducerRef> producer;
  llvm::DenseMap<mlir::Value, mlir::Value> adapterForward;
  std::map<EdgeKey, std::string> payloadKindByEdge;

  std::optional<ProducerRef> resolve(mlir::Value value) {
    auto direct = producer.find(value);
    if (direct != producer.end())
      return direct->second;
    auto adapter = adapterForward.find(value);
    if (adapter == adapterForward.end())
      return std::nullopt;
    return resolve(adapter->second);
  }
};

std::string payloadKind(mlir::Value value) {
  if (mlir::isa<mlir::NoneType>(value.getType()))
    return "control";
  return "data";
}

void collectValueProducers(
    mlir::Operation *graph,
    const llvm::DenseMap<mlir::Operation *, std::string> &nodeIds,
    RouteBuilder &builder) {
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    auto nodeIt = nodeIds.find(&op);
    if (nodeIt != nodeIds.end()) {
      unsigned resultIndex = 0;
      for (mlir::Value result : op.getResults())
        builder.producer.try_emplace(
            result, RouteBuilder::ProducerRef{nodeIt->second, resultIndex++});
      continue;
    }
    if (!isAdapterOp(&op) || op.getNumOperands() == 0)
      continue;
    mlir::Value source = op.getOperand(0);
    for (mlir::Value result : op.getResults())
      builder.adapterForward.try_emplace(result, source);
  }
}

llvm::SmallVector<RouteRecord, 0>
collectRoutes(llvm::ArrayRef<SoftwareNode> nodes, mlir::Operation *graph,
              llvm::ArrayRef<PlacementRecord> placements,
              llvm::StringRef hardwareName) {
  RouteBuilder builder;
  llvm::DenseMap<mlir::Operation *, std::string> nodeIds = indexNodeIds(nodes);
  collectValueProducers(graph, nodeIds, builder);

  llvm::StringMap<std::string> hardwareBySoftware;
  for (const PlacementRecord &placement : placements)
    hardwareBySoftware.try_emplace(placement.softwareId, placement.hardwareId);

  for (const SoftwareNode &node : nodes) {
    unsigned operandIndex = 0;
    for (mlir::Value operand : node.op->getOperands()) {
      std::optional<RouteBuilder::ProducerRef> source =
          builder.resolve(operand);
      if (!source || source->softwareId == node.id) {
        ++operandIndex;
        continue;
      }
      RouteBuilder::EdgeKey key{source->softwareId, source->resultIndex,
                                node.id, operandIndex++};
      builder.payloadKindByEdge.try_emplace(std::move(key),
                                            payloadKind(operand));
    }
  }

  llvm::SmallVector<RouteRecord, 0> routes;
  std::size_t index = 0;
  for (const auto &[edge, kind] : builder.payloadKindByEdge) {
    const std::string &from = edge.producerSoftwareId;
    const std::string &to = edge.consumerSoftwareId;
    auto fromHw = hardwareBySoftware.find(from);
    auto toHw = hardwareBySoftware.find(to);
    if (fromHw == hardwareBySoftware.end() || toHw == hardwareBySoftware.end())
      continue;
    std::string recordId = "route#" + std::to_string(index++);
    std::string edgeRef =
        from + ".result" + std::to_string(edge.producerResultIndex) + "->" +
        to + ".operand" + std::to_string(edge.consumerOperandIndex);
    RouteSegment segment;
    segment.segmentId = "seg0";
    segment.segmentKind = "module_path";
    segment.sourceEndpoint = fromHw->second + ".out";
    segment.sinkEndpoint = toHw->second + ".in";
    segment.hardwareRef = hardwareName.str();
    RouteRecord route;
    route.recordId = recordId;
    route.edgeRef = edgeRef;
    route.producerBinding = "placement:" + from;
    route.consumerBinding = "placement:" + to;
    route.payloadKind = kind;
    route.fromSoftwareId = from;
    route.toSoftwareId = to;
    route.segments.push_back(std::move(segment));
    routes.push_back(std::move(route));
  }
  return routes;
}

llvm::json::Object placementJson(const PlacementRecord &placement) {
  return llvm::json::Object{
      {"software", placement.softwareId},
      {"operation", placement.operation},
      {"resource_kind", placement.resourceKind},
      {"hardware", placement.hardwareId},
      {"schedule", placement.schedule},
  };
}

llvm::json::Object routeJson(const RouteRecord &route) {
  llvm::json::Array segments;
  for (const RouteSegment &segment : route.segments) {
    llvm::json::Object segmentObject{
        {"segment_id", segment.segmentId},
        {"segment_kind", segment.segmentKind},
        {"source_endpoint", segment.sourceEndpoint},
        {"sink_endpoint", segment.sinkEndpoint},
    };
    if (!segment.hardwareRef.empty())
      segmentObject.try_emplace("hardware_ref", segment.hardwareRef);
    segments.push_back(std::move(segmentObject));
  }
  return llvm::json::Object{
      {"record_id", route.recordId},
      {"edge_ref", route.edgeRef},
      {"producer_binding", route.producerBinding},
      {"consumer_binding", route.consumerBinding},
      {"payload_kind", route.payloadKind},
      {"from", route.fromSoftwareId},
      {"to", route.toSoftwareId},
      {"status", "routed"},
      {"segments", std::move(segments)},
  };
}

void addConfig(llvm::SmallVectorImpl<ConfigEntry> &entries,
               llvm::StringRef target, llvm::StringRef registerName,
               llvm::StringRef value, llvm::StringRef source) {
  entries.push_back(
      ConfigEntry{target.str(), registerName.str(), value.str(), source.str()});
}

llvm::Error appendPlacementConfig(MappingSummary &summary,
                                  const SoftwareNode &node,
                                  const HardwareResource &resource) {
  if (resource.kind == ResourceKind::FabricOp) {
    if (std::optional<std::string> opSel = configFor(resource, "op_sel")) {
      if (*opSel != node.operation)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "hardware resource %s is configured for %s but software op is %s",
            resource.id.c_str(), opSel->c_str(), node.operation.c_str());
    }
  }

  std::string source = "placement:" + node.id;
  addConfig(summary.configEntries, resource.id, "software_id", node.id, source);
  addConfig(summary.configEntries, resource.id, "operation", node.operation,
            source);
  addConfig(summary.configEntries, resource.id, "resource_kind",
            resourceKindName(node.resourceKind), source);
  addConfig(summary.configEntries, resource.id, "schedule", resource.schedule,
            source);

  if (resource.kind == ResourceKind::FabricOp &&
      resource.supportedOps.size() > 1 && !configFor(resource, "op_sel")) {
    addConfig(summary.configEntries, resource.id, "sw_configs.op_sel",
              node.operation, source);
  }
  for (const auto &[key, value] : resource.swConfigs) {
    addConfig(summary.configEntries, resource.id, "sw_configs." + key, value,
              source);
  }
  return llvm::Error::success();
}

std::string routeTarget(const MappingSummary &summary,
                        llvm::StringRef recordId) {
  return summary.mappingId + "::" + recordId.str();
}

std::string routeSource(const RouteRecord &route) {
  return "route:" + route.recordId;
}

void appendRouteConfig(MappingSummary &summary) {
  for (const RouteRecord &route : summary.routes) {
    std::string source = routeSource(route);
    std::string target = routeTarget(summary, route.recordId);
    addConfig(summary.configEntries, target, "from_software_id",
              route.fromSoftwareId, source);
    addConfig(summary.configEntries, target, "to_software_id",
              route.toSoftwareId, source);
    addConfig(summary.configEntries, target, "segment_count",
              std::to_string(route.segments.size()), source);
    for (std::size_t segmentIndex = 0; segmentIndex < route.segments.size();
         ++segmentIndex) {
      const RouteSegment &segment = route.segments[segmentIndex];
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      addConfig(summary.configEntries, target, prefix + "kind",
                segment.segmentKind, source);
      addConfig(summary.configEntries, target, prefix + "source_endpoint",
                segment.sourceEndpoint, source);
      addConfig(summary.configEntries, target, prefix + "sink_endpoint",
                segment.sinkEndpoint, source);
      if (!segment.hardwareRef.empty())
        addConfig(summary.configEntries, target, prefix + "hardware_ref",
                  segment.hardwareRef, source);
    }
  }
}

std::string configKey(llvm::StringRef target, llvm::StringRef registerName,
                      llvm::StringRef source) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << target << '\x1f' << registerName << '\x1f' << source;
  return key;
}

std::string registerKey(llvm::StringRef target, llvm::StringRef registerName) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << target << '\x1f' << registerName;
  return key;
}

llvm::Error validateConfigBitstream(const MappingSummary &summary) {
  if (summary.status != "pass")
    return llvm::Error::success();

  llvm::StringSet<> seen;
  llvm::StringSet<> writtenRegisters;
  for (const ConfigEntry &entry : summary.configEntries) {
    if (entry.target.empty() || entry.registerName.empty() ||
        entry.source.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream contains an incomplete "
                                     "register assignment");
    if (entry.registerName == "schedule" && entry.value != "spatial" &&
        entry.value != "temporal")
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream contains invalid "
                                     "schedule value %s",
                                     entry.value.c_str());
    std::string key = configKey(entry.target, entry.registerName, entry.source);
    if (!seen.insert(key).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream contains duplicate assignment for %s",
          entry.target.c_str());
    std::string regKey = registerKey(entry.target, entry.registerName);
    if (!writtenRegisters.insert(regKey).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream writes register %s on %s more than once",
          entry.registerName.c_str(), entry.target.c_str());
  }

  for (const PlacementRecord &placement : summary.placements) {
    std::string source = "placement:" + placement.softwareId;
    for (llvm::StringRef reg :
         {"software_id", "operation", "resource_kind", "schedule"}) {
      if (!seen.contains(configKey(placement.hardwareId, reg, source)))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "config bitstream is missing placement register %s for %s",
            reg.str().c_str(), placement.hardwareId.c_str());
    }
  }

  for (const RouteRecord &route : summary.routes) {
    std::string source = routeSource(route);
    std::string target = routeTarget(summary, route.recordId);
    if (!seen.contains(configKey(target, "from_software_id", source)) ||
        !seen.contains(configKey(target, "to_software_id", source)) ||
        !seen.contains(configKey(target, "segment_count", source)))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream is missing route endpoint registers for %s",
          target.c_str());
    for (std::size_t segmentIndex = 0; segmentIndex < route.segments.size();
         ++segmentIndex) {
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      for (llvm::StringRef reg : {"kind", "source_endpoint", "sink_endpoint"}) {
        std::string registerName = prefix + reg.str();
        if (!seen.contains(configKey(target, registerName, source)))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "config bitstream is missing route segment register %s for %s",
              registerName.c_str(), target.c_str());
      }
      if (!route.segments[segmentIndex].hardwareRef.empty()) {
        std::string registerName = prefix + "hardware_ref";
        if (!seen.contains(configKey(target, registerName, source)))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "config bitstream is missing route segment register %s for %s",
              registerName.c_str(), target.c_str());
      }
    }
  }
  return llvm::Error::success();
}

llvm::json::Object configJson(const ConfigEntry &entry) {
  return llvm::json::Object{
      {"target", entry.target},
      {"register", entry.registerName},
      {"value", entry.value},
      {"source", entry.source},
  };
}

} // namespace

llvm::Expected<MappingSummary>
loom::pnr::createMapping(const MappingOptions &options) {
  mlir::DialectRegistry registry = makeRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> dfg =
      parseModule(context, options.dfgMlirPath);
  if (!dfg)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse DFG MLIR");
  mlir::OwningOpRef<mlir::ModuleOp> hardware =
      parseModule(context, options.hardwareMlirPath);
  if (!hardware)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse hardware MLIR");

  mlir::Operation *graph =
      findSymbolOp(*dfg, "dataflow.graph.func", options.graphName);
  if (!graph)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find dataflow graph %s",
                                   options.graphName.c_str());
  mlir::Operation *hardwareOp =
      findSymbolOp(*hardware, "fabric.module", options.hardwareName);
  if (!hardwareOp)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find fabric hardware %s",
                                   options.hardwareName.c_str());

  auto nodesOrErr = collectSoftwareNodes(graph);
  if (!nodesOrErr)
    return nodesOrErr.takeError();
  auto resourcesOrErr =
      collectHardwareResources(hardwareOp, options.hardwareName);
  if (!resourcesOrErr)
    return resourcesOrErr.takeError();

  MappingSummary summary;
  summary.workload =
      options.workload.empty() ? options.graphName : options.workload;
  summary.hardware = options.hardwareName;
  summary.graph = options.graphName;
  summary.mappingId = mappingId(summary.workload, summary.hardware);
  summary.status = "pass";

  for (SoftwareNode &node : *nodesOrErr) {
    HardwareResource *resource = claimResource(node, *resourcesOrErr);
    if (!resource) {
      summary.status = "fail";
      summary.diagnostic =
          "missing hardware resource for software op " + node.operation;
      ++summary.unplacedRecords;
      continue;
    }
    summary.placements.push_back(PlacementRecord{
        node.id, node.operation, resourceKindName(node.resourceKind).str(),
        resource->id, resource->schedule});
    if (llvm::Error err = appendPlacementConfig(summary, node, *resource))
      return std::move(err);
  }

  summary.routes =
      collectRoutes(*nodesOrErr, graph, summary.placements, summary.hardware);
  if (summary.status == "pass") {
    appendRouteConfig(summary);
    if (llvm::Error err = validateConfigBitstream(summary))
      return std::move(err);
    summary.diagnostic = "mapped software graph to fabric resources";
  } else {
    summary.routes.clear();
    summary.configEntries.clear();
  }
  return summary;
}

llvm::Error
loom::pnr::writeMappingCsv(llvm::StringRef outputPath,
                           llvm::ArrayRef<MappingSummary> summaries) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());

  out << "workload,hardware,mapping_id,placed_records,routed_edges,"
         "unrouted_edges,unplaced_records,status,diagnostic\n";
  for (const MappingSummary &summary : summaries) {
    out << csvEscape(summary.workload) << ',' << csvEscape(summary.hardware)
        << ',' << csvEscape(summary.mappingId) << ','
        << summary.placements.size() << ',' << summary.routes.size() << ','
        << summary.unroutedEdges << ',' << summary.unplacedRecords << ','
        << csvEscape(summary.status) << ',' << csvEscape(summary.diagnostic)
        << '\n';
  }
  return llvm::Error::success();
}

llvm::Error loom::pnr::writeMappingJson(llvm::StringRef outputPath,
                                        const MappingSummary &summary) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;

  llvm::json::Array placements;
  for (const PlacementRecord &placement : summary.placements)
    placements.push_back(placementJson(placement));

  llvm::json::Array routes;
  for (const RouteRecord &route : summary.routes)
    routes.push_back(routeJson(route));

  llvm::json::Array configEntries;
  for (const ConfigEntry &entry : summary.configEntries)
    configEntries.push_back(configJson(entry));

  llvm::json::Object root{
      {"schema_version", 1},
      {"kind", "pnr_mapping"},
      {"workload", summary.workload},
      {"hardware", summary.hardware},
      {"graph", summary.graph},
      {"mapping_id", summary.mappingId},
      {"status", summary.status},
      {"placed_records", static_cast<int64_t>(summary.placements.size())},
      {"routed_edges", static_cast<int64_t>(summary.routes.size())},
      {"unrouted_edges", static_cast<int64_t>(summary.unroutedEdges)},
      {"unplaced_records", static_cast<int64_t>(summary.unplacedRecords)},
      {"config_records", static_cast<int64_t>(summary.configEntries.size())},
      {"placements", std::move(placements)},
      {"routes", std::move(routes)},
      {"config_bitstream", std::move(configEntries)},
  };
  if (!summary.diagnostic.empty()) {
    llvm::json::Array diagnostics;
    diagnostics.push_back(summary.diagnostic);
    root.try_emplace("diagnostics", std::move(diagnostics));
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}
