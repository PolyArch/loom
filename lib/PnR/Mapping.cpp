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
#include <set>
#include <string>
#include <system_error>

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
  for (std::uint64_t i = 0; i < loadPorts; ++i) {
    resources.push_back(HardwareResource{
        (hardwareName + "::mem.load#" + llvm::Twine(i)).str(),
        ResourceKind::MemLoad, {}, false});
  }
  for (std::uint64_t i = 0; i < storePorts; ++i) {
    resources.push_back(HardwareResource{
        (hardwareName + "::mem.store#" + llvm::Twine(i)).str(),
        ResourceKind::MemStore, {}, false});
  }
}

void appendFabricOpResource(mlir::Operation *op, llvm::StringRef hardwareName,
                            unsigned index,
                            llvm::SmallVectorImpl<HardwareResource> &resources) {
  auto opList = op->getAttrOfType<mlir::ArrayAttr>("op_list");
  if (!opList)
    return;
  HardwareResource resource;
  resource.id = (hardwareName + "::fabric.op#" + llvm::Twine(index)).str();
  resource.kind = ResourceKind::FabricOp;
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

HardwareResource *claimResource(SoftwareNode &node,
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

llvm::DenseMap<mlir::Operation *, std::string>
indexNodeIds(llvm::ArrayRef<SoftwareNode> nodes) {
  llvm::DenseMap<mlir::Operation *, std::string> byOperation;
  for (const SoftwareNode &node : nodes)
    byOperation.try_emplace(node.op, node.id);
  return byOperation;
}

struct RouteBuilder {
  llvm::DenseMap<mlir::Value, std::string> producer;
  llvm::DenseMap<mlir::Value, mlir::Value> adapterForward;
  std::set<std::pair<std::string, std::string>> routes;

  std::string resolve(mlir::Value value) {
    auto direct = producer.find(value);
    if (direct != producer.end())
      return direct->second;
    auto adapter = adapterForward.find(value);
    if (adapter == adapterForward.end())
      return "";
    return resolve(adapter->second);
  }
};

void collectValueProducers(mlir::Operation *graph,
                           const llvm::DenseMap<mlir::Operation *, std::string>
                               &nodeIds,
                           RouteBuilder &builder) {
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    auto nodeIt = nodeIds.find(&op);
    if (nodeIt != nodeIds.end()) {
      for (mlir::Value result : op.getResults())
        builder.producer.try_emplace(result, nodeIt->second);
      continue;
    }
    if (!isAdapterOp(&op) || op.getNumOperands() == 0)
      continue;
    mlir::Value source = op.getOperand(0);
    for (mlir::Value result : op.getResults())
      builder.adapterForward.try_emplace(result, source);
  }
}

llvm::SmallVector<RouteRecord>
collectRoutes(llvm::ArrayRef<SoftwareNode> nodes, mlir::Operation *graph) {
  RouteBuilder builder;
  llvm::DenseMap<mlir::Operation *, std::string> nodeIds = indexNodeIds(nodes);
  collectValueProducers(graph, nodeIds, builder);

  for (const SoftwareNode &node : nodes) {
    for (mlir::Value operand : node.op->getOperands()) {
      std::string source = builder.resolve(operand);
      if (source.empty() || source == node.id)
        continue;
      builder.routes.insert({std::move(source), node.id});
    }
  }

  llvm::SmallVector<RouteRecord> routes;
  for (const auto &[from, to] : builder.routes)
    routes.push_back(RouteRecord{from, to});
  return routes;
}

llvm::json::Object placementJson(const PlacementRecord &placement) {
  return llvm::json::Object{
      {"software", placement.softwareId},
      {"operation", placement.operation},
      {"resource_kind", placement.resourceKind},
      {"hardware", placement.hardwareId},
  };
}

llvm::json::Object routeJson(const RouteRecord &route) {
  return llvm::json::Object{
      {"from", route.fromSoftwareId},
      {"to", route.toSoftwareId},
      {"status", "routed"},
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
  summary.workload = options.workload.empty() ? options.graphName : options.workload;
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
      ++summary.unroutedEdges;
      continue;
    }
    summary.placements.push_back(PlacementRecord{
        node.id, node.operation, resourceKindName(node.resourceKind).str(),
        resource->id});
  }

  summary.routes = collectRoutes(*nodesOrErr, graph);
  if (summary.status == "pass") {
    summary.diagnostic = "mapped software graph to fabric resources";
  } else {
    summary.routes.clear();
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
         "unrouted_edges,status,diagnostic\n";
  for (const MappingSummary &summary : summaries) {
    out << csvEscape(summary.workload) << ',' << csvEscape(summary.hardware)
        << ',' << csvEscape(summary.mappingId) << ','
        << summary.placements.size() << ',' << summary.routes.size() << ','
        << summary.unroutedEdges << ',' << csvEscape(summary.status) << ','
        << csvEscape(summary.diagnostic) << '\n';
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
      {"placements", std::move(placements)},
      {"routes", std::move(routes)},
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
