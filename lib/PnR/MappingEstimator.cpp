#include "PnR/MappingEstimator.h"
#include "MappingHardware.h"

#include "Common/ArtifactText.h"
#include "Common/ResolvedConfig.h"
#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

using namespace loom::pnr;
using loom::pnr::detail::ConcreteMemOccurrence;
using loom::pnr::detail::MemAccessKind;
using loom::pnr::detail::MemOccurrenceIdentity;

namespace {

constexpr std::uint64_t kRouteSegmentWeight = 1;
constexpr std::uint64_t kMemoryAccessWeight = 4;
constexpr std::uint64_t kStoreCommitWeight = 3;
constexpr std::uint64_t kWidthAdapterWeight = 1;
constexpr std::uint64_t kFunctionalUnitWeight = 1;
constexpr std::uint64_t kLoadAddressWeight = 1;
constexpr std::uint64_t kStoreAddressWeight = 2;
constexpr std::uint64_t kConfigRecordsPerScoreUnit = 128;
constexpr llvm::StringLiteral kMappingSchemaVersion = "3.0";

struct RouteStats {
  std::uint64_t routeCount = 0;
  std::uint64_t segmentCount = 0;
  std::uint64_t computedLoadAddressRoutes = 0;
  std::uint64_t computedStoreAddressRoutes = 0;
};

struct ConfigEntries {
  llvm::StringMap<std::string> valuesByFullKey;
  llvm::StringSet<> writtenRegisters;
};

struct HardwareArtifactResource {
  std::string resourceKind;
  std::string schedule;
  llvm::StringSet<> supportedOps;
  mlir::Operation *op = nullptr;
};

struct HardwareRouteSegment {
  std::string segmentKind;
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  std::string hardwareRef;
};

struct HardwareTopology {
  std::set<std::string> endpoints;
  std::set<std::string> edgeKeys;
  std::map<std::string, HardwareRouteSegment> segmentByEdge;
};

struct EndpointKey {
  mlir::Operation *op = nullptr;
  unsigned index = 0;

  bool operator<(const EndpointKey &other) const {
    return std::tie(op, index) < std::tie(other.op, other.index);
  }
};

struct PlacementInfo {
  std::string hardware;
  std::string resourceKind;
  std::string operation;
};

struct HardwareSelection {
  mlir::Operation *module = nullptr;
  std::string moduleName;
};

std::string nearestHardwareSchedule(mlir::Operation *op) {
  for (mlir::Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    if (auto attr = cursor->getAttrOfType<fabric::ScheduleAttr>("schedule"))
      return fabric::stringifySchedule(attr.getValue()).str();
  }
  return "spatial";
}

bool operationMatchesResourceKind(llvm::StringRef resourceKind,
                                  llvm::StringRef operation) {
  if (resourceKind == "fabric.mem.load")
    return operation == "dataflow.load" || operation == "llvm.load";
  if (resourceKind == "fabric.mem.store")
    return operation == "dataflow.store" || operation == "llvm.store";
  if (resourceKind == "fabric.op")
    return fabric::isFabricOpSupported(operation);
  return false;
}

llvm::Expected<std::string>
requireObjectString(const llvm::json::Object &object, llvm::StringRef key,
                    llvm::StringRef diagnosticContext);

std::string systemCoreHardwareIdentity(llvm::StringRef systemName,
                                       llvm::StringRef accCoreName) {
  return (systemName + "::" + accCoreName).str();
}

std::optional<std::string> symbolName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::Operation *findFabricModule(mlir::ModuleOp module,
                                  llvm::StringRef symbol) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != "fabric.module")
      return;
    std::optional<std::string> name = symbolName(op);
    if (name && *name == symbol)
      found = op;
  });
  return found;
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

llvm::Expected<HardwareSelection>
selectHardwareForMapping(mlir::ModuleOp module, llvm::StringRef hardwareName,
                         const llvm::json::Object &mapping) {
  std::optional<llvm::StringRef> rootKind =
      mapping.getString("hardware_root_kind");
  if (!rootKind || rootKind->empty() || *rootKind == "fabric.module") {
    mlir::Operation *hardware = findFabricModule(module, hardwareName);
    if (!hardware)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware artifact does not contain fabric.module %s",
          hardwareName.str().c_str());
    return HardwareSelection{hardware, hardwareName.str()};
  }

  if (*rootKind != "fabric.system")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping artifact has unsupported hardware_root_kind %s",
        rootKind->str().c_str());

  auto systemOrErr =
      requireObjectString(mapping, "hardware_system", "mapping artifact");
  if (!systemOrErr)
    return systemOrErr.takeError();
  auto accCoreOrErr =
      requireObjectString(mapping, "selected_acc_core", "mapping artifact");
  if (!accCoreOrErr)
    return accCoreOrErr.takeError();
  auto spatialOrErr =
      requireObjectString(mapping, "spatialcore_template", "mapping artifact");
  if (!spatialOrErr)
    return spatialOrErr.takeError();
  std::string expectedHardware =
      systemCoreHardwareIdentity(*systemOrErr, *accCoreOrErr);
  if (expectedHardware != hardwareName)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping hardware %s does not match selected system core %s",
        hardwareName.str().c_str(), expectedHardware.c_str());

  mlir::Operation *system = findSymbolOp(module, "fabric.system", *systemOrErr);
  if (!system)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware artifact does not contain fabric.system %s",
        systemOrErr->c_str());

  fabric::NodeOp selectedNode;
  system->walk([&](fabric::NodeOp node) {
    if (selectedNode)
      return;
    if (node.getSymName() == *accCoreOrErr)
      selectedNode = node;
  });
  if (!selectedNode || selectedNode.getKind() != "acc_core")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s does not contain acc_core %s", systemOrErr->c_str(),
        accCoreOrErr->c_str());
  mlir::FlatSymbolRefAttr spatial = selectedNode.getSpatialAttr();
  if (!spatial || spatial.getValue() != *spatialOrErr)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s acc_core %s does not reference spatialcore %s",
        systemOrErr->c_str(), accCoreOrErr->c_str(), spatialOrErr->c_str());

  mlir::Operation *hardware = findFabricModule(module, *spatialOrErr);
  if (!hardware)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware artifact does not contain fabric.module %s",
        spatialOrErr->c_str());
  return HardwareSelection{hardware, *spatialOrErr};
}

void addHardwareResource(llvm::StringMap<HardwareArtifactResource> &resources,
                         llvm::StringRef resourceId,
                         llvm::StringRef resourceKind,
                         mlir::Operation *op = nullptr) {
  HardwareArtifactResource resource;
  resource.resourceKind = resourceKind.str();
  resource.schedule = nearestHardwareSchedule(op);
  resource.op = op;
  resources.try_emplace(resourceId, std::move(resource));
}

void appendHardwareMemResources(
    mlir::Operation *op, llvm::StringRef hardwareName,
    const MemOccurrenceIdentity &identity,
    llvm::StringMap<HardwareArtifactResource> &resources) {
  for (std::uint64_t i = 0; i < identity.loadCount; ++i)
    addHardwareResource(resources,
                        loom::pnr::detail::memResourceId(
                            hardwareName, MemAccessKind::Load, identity, i),
                        "fabric.mem.load", op);
  for (std::uint64_t i = 0; i < identity.storeCount; ++i)
    addHardwareResource(resources,
                        loom::pnr::detail::memResourceId(
                            hardwareName, MemAccessKind::Store, identity, i),
                        "fabric.mem.store", op);
}

void appendHardwareOpResource(
    mlir::Operation *op, llvm::StringRef hardwareName, unsigned index,
    llvm::StringMap<HardwareArtifactResource> &resources) {
  auto opList = op->getAttrOfType<mlir::ArrayAttr>("op_list");
  if (!opList)
    return;
  HardwareArtifactResource resource;
  resource.resourceKind = "fabric.op";
  resource.schedule = nearestHardwareSchedule(op);
  resource.op = op;
  for (mlir::Attribute attr : opList) {
    if (auto sym = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      resource.supportedOps.insert(sym.getValue());
  }
  resources.try_emplace(
      (hardwareName + "::fabric.op#" + llvm::Twine(index)).str(),
      std::move(resource));
}

llvm::StringMap<HardwareArtifactResource> collectHardwareArtifactResources(
    mlir::Operation *hardware, llvm::StringRef hardwareName,
    llvm::ArrayRef<ConcreteMemOccurrence> memOccurrences) {
  llvm::StringMap<HardwareArtifactResource> resources;
  unsigned fabricOpIndex = 0;
  unsigned memOccurrenceIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!loom::pnr::detail::isConcreteHardwareOperation(op, hardware))
      return;
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.op") {
      appendHardwareOpResource(op, hardwareName, fabricOpIndex++, resources);
      return;
    }
    if (opName == "fabric.mem") {
      assert(memOccurrenceIndex < memOccurrences.size());
      const ConcreteMemOccurrence &occurrence =
          memOccurrences[memOccurrenceIndex++];
      assert(occurrence.op == op);
      appendHardwareMemResources(op, hardwareName, occurrence.identity,
                                 resources);
    }
  });
  return resources;
}

llvm::Expected<std::string>
requireObjectString(const llvm::json::Object &object, llvm::StringRef key,
                    llvm::StringRef diagnosticContext);

std::string endpointFor(llvm::StringRef resourceId, llvm::StringRef endpoint,
                        unsigned index) {
  return (resourceId + "." + endpoint + llvm::Twine(index)).str();
}

std::string topologyEdgeKey(llvm::StringRef sourceEndpoint,
                            llvm::StringRef sinkEndpoint) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << sourceEndpoint << '\x1f' << sinkEndpoint;
  return key;
}

bool isPlaceholderRouteEndpoint(llvm::StringRef endpoint) {
  return endpoint.ends_with(".out") || endpoint.ends_with(".in");
}

bool isRouteResourceOp(llvm::StringRef opName) {
  return opName == "fabric.switch" || opName == "fabric.fifo" ||
         opName == "fabric.boundary";
}

std::string routeResourceKind(llvm::StringRef opName) {
  if (opName == "fabric.switch")
    return "fabric.switch";
  if (opName == "fabric.fifo")
    return "fabric.fifo";
  return "fabric.boundary";
}

std::string internalRouteSegmentKind(llvm::StringRef opName) {
  if (opName == "fabric.fifo")
    return "buffer";
  if (opName == "fabric.boundary")
    return "boundary_crossing";
  return "module_path";
}

llvm::SmallVector<std::string, 4> switchConnectivityRows(mlir::Operation *op) {
  llvm::SmallVector<std::string, 4> rows;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (!hwParams || hwParams.empty())
    return rows;
  auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0]);
  if (!dict)
    return rows;
  auto table = llvm::dyn_cast_if_present<mlir::ArrayAttr>(
      dict.get("connectivity_table"));
  if (!table)
    return rows;
  for (mlir::Attribute attr : table) {
    auto row = llvm::dyn_cast<mlir::StringAttr>(attr);
    if (!row)
      return {};
    rows.push_back(row.getValue().str());
  }
  return rows;
}

bool switchConnectsInputToOutput(llvm::ArrayRef<std::string> rows,
                                 unsigned inputIndex, unsigned outputIndex,
                                 unsigned inputCount) {
  if (outputIndex >= rows.size())
    return false;
  llvm::StringRef row = rows[outputIndex];
  if (row.size() != inputCount || inputIndex >= inputCount)
    return false;
  return row[inputCount - 1 - inputIndex] == '1';
}

void addTopologySegment(HardwareTopology &topology,
                        HardwareRouteSegment segment) {
  topology.endpoints.insert(segment.sourceEndpoint);
  topology.endpoints.insert(segment.sinkEndpoint);
  std::string edgeKey =
      topologyEdgeKey(segment.sourceEndpoint, segment.sinkEndpoint);
  topology.edgeKeys.insert(edgeKey);
  topology.segmentByEdge.try_emplace(edgeKey, segment);
}

unsigned memResultPortBase(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return 0;
  if (mlir::isa<mlir::MemRefType>(op->getResult(0).getType()))
    return 1;
  return 0;
}

std::optional<unsigned>
hardwareOperandIndexForResourceKind(llvm::StringRef resourceKind,
                                    llvm::StringRef operation,
                                    unsigned softwareOperandIndex) {
  if (resourceKind == "fabric.op")
    return softwareOperandIndex;
  if (resourceKind == "fabric.mem.load") {
    if (softwareOperandIndex == 1)
      return 0;
    if (softwareOperandIndex == 2)
      return 1;
    return std::nullopt;
  }
  if (resourceKind == "fabric.mem.store") {
    if (operation == "llvm.store") {
      if (softwareOperandIndex == 0)
        return 1;
      if (softwareOperandIndex == 1)
        return 0;
      return std::nullopt;
    }
    if (softwareOperandIndex >= 1 && softwareOperandIndex <= 3)
      return softwareOperandIndex - 1;
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<unsigned>
hardwareResultIndexForResourceKind(llvm::StringRef resourceKind,
                                   unsigned softwareResultIndex) {
  if (resourceKind == "fabric.op")
    return softwareResultIndex;
  if (resourceKind == "fabric.mem.load") {
    if (softwareResultIndex <= 1)
      return softwareResultIndex;
    return std::nullopt;
  }
  if (resourceKind == "fabric.mem.store") {
    if (softwareResultIndex == 0)
      return 0;
    return std::nullopt;
  }
  return std::nullopt;
}

llvm::Expected<unsigned> parseRouteIndex(llvm::StringRef text,
                                         llvm::StringRef context) {
  unsigned value = 0;
  if (text.empty() || text.getAsInteger(10, value))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping route edge_ref has invalid %s",
                                   context.str().c_str());
  return value;
}

llvm::Expected<std::pair<unsigned, unsigned>>
parseRouteEndpointIndices(const llvm::json::Object &route,
                          llvm::StringRef fromSoftware,
                          llvm::StringRef toSoftware) {
  auto edgeRefOrErr = requireObjectString(route, "edge_ref", "mapping route");
  if (!edgeRefOrErr)
    return edgeRefOrErr.takeError();
  std::string producerPrefix = (fromSoftware + ".result").str();
  std::string consumerPrefix = ("->" + toSoftware + ".operand").str();
  llvm::StringRef rest(*edgeRefOrErr);
  if (!rest.consume_front(producerPrefix))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route edge_ref does not match producer software id");
  std::pair<llvm::StringRef, llvm::StringRef> split =
      rest.split(consumerPrefix);
  if (split.second.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route edge_ref does not match consumer software id");
  auto resultIndexOrErr = parseRouteIndex(split.first, "producer result index");
  if (!resultIndexOrErr)
    return resultIndexOrErr.takeError();
  auto operandIndexOrErr =
      parseRouteIndex(split.second, "consumer operand index");
  if (!operandIndexOrErr)
    return operandIndexOrErr.takeError();
  return std::pair<unsigned, unsigned>{*resultIndexOrErr, *operandIndexOrErr};
}

llvm::Expected<std::pair<std::string, std::string>> expectedRouteEndpoints(
    const llvm::json::Object &route,
    const std::map<std::string, PlacementInfo> &placementBySoftware) {
  auto fromOrErr = requireObjectString(route, "from", "mapping route");
  if (!fromOrErr)
    return fromOrErr.takeError();
  auto toOrErr = requireObjectString(route, "to", "mapping route");
  if (!toOrErr)
    return toOrErr.takeError();
  auto producerBindingOrErr =
      requireObjectString(route, "producer_binding", "mapping route");
  if (!producerBindingOrErr)
    return producerBindingOrErr.takeError();
  auto consumerBindingOrErr =
      requireObjectString(route, "consumer_binding", "mapping route");
  if (!consumerBindingOrErr)
    return consumerBindingOrErr.takeError();
  std::string expectedProducerBinding = "placement:" + *fromOrErr;
  if (*producerBindingOrErr != expectedProducerBinding)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route producer_binding does not match from software id");
  std::string expectedConsumerBinding = "placement:" + *toOrErr;
  if (*consumerBindingOrErr != expectedConsumerBinding)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route consumer_binding does not match to software id");

  auto indicesOrErr = parseRouteEndpointIndices(route, *fromOrErr, *toOrErr);
  if (!indicesOrErr)
    return indicesOrErr.takeError();
  auto producerPlacement = placementBySoftware.find(*fromOrErr);
  if (producerPlacement == placementBySoftware.end())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping route producer lacks placement %s",
                                   fromOrErr->c_str());
  auto consumerPlacement = placementBySoftware.find(*toOrErr);
  if (consumerPlacement == placementBySoftware.end())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping route consumer lacks placement %s",
                                   toOrErr->c_str());

  std::optional<unsigned> producerResultIndex =
      hardwareResultIndexForResourceKind(producerPlacement->second.resourceKind,
                                         indicesOrErr->first);
  if (!producerResultIndex)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route producer endpoint is not representable on hardware");
  std::optional<unsigned> consumerOperandIndex =
      hardwareOperandIndexForResourceKind(
          consumerPlacement->second.resourceKind,
          consumerPlacement->second.operation, indicesOrErr->second);
  if (!consumerOperandIndex)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping route consumer endpoint is not representable on hardware");

  return std::pair<std::string, std::string>{
      endpointFor(producerPlacement->second.hardware, "result",
                  *producerResultIndex),
      endpointFor(consumerPlacement->second.hardware, "operand",
                  *consumerOperandIndex)};
}

std::optional<std::string> forwardedBlockArgumentEndpoint(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints);

std::optional<std::string> sourceEndpointForValue(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints) {
  if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
    auto it = resultEndpoints.find(
        EndpointKey{opResult.getOwner(), opResult.getResultNumber()});
    if (it != resultEndpoints.end())
      return it->second;
    return std::nullopt;
  }
  return forwardedBlockArgumentEndpoint(value, resultEndpoints);
}

std::optional<std::string> forwardedBlockArgumentEndpoint(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints) {
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  mlir::Operation *parent = blockArg.getOwner()->getParentOp();
  if (!parent)
    return std::nullopt;
  llvm::StringRef parentName = parent->getName().getStringRef();
  if (parentName != "fabric.fu" && parentName != "fabric.pe")
    return std::nullopt;
  unsigned index = blockArg.getArgNumber();
  if (index >= parent->getNumOperands())
    return std::nullopt;
  return sourceEndpointForValue(parent->getOperand(index), resultEndpoints);
}

void addGenericEndpointMaps(
    mlir::Operation *op, llvm::StringRef resourceId,
    std::map<EndpointKey, std::string> &operandEndpoints,
    std::map<EndpointKey, std::string> &resultEndpoints) {
  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex)
    operandEndpoints.try_emplace(
        EndpointKey{op, operandIndex},
        endpointFor(resourceId, "operand", operandIndex));
  for (unsigned resultIndex = 0; resultIndex < op->getNumResults();
       ++resultIndex)
    resultEndpoints.try_emplace(EndpointKey{op, resultIndex},
                                endpointFor(resourceId, "result", resultIndex));
}

bool isFabricBoundaryOp(llvm::StringRef opName) {
  return opName == "fabric.fu" || opName == "fabric.pe";
}

void addMemEndpointMaps(mlir::Operation *op, llvm::StringRef hardwareName,
                        const MemOccurrenceIdentity &identity,
                        std::map<EndpointKey, std::string> &operandEndpoints,
                        std::map<EndpointKey, std::string> &resultEndpoints) {
  unsigned operandBase = 1;
  unsigned resultBase = memResultPortBase(op);
  for (std::uint64_t i = 0; i < identity.loadCount; ++i) {
    std::string resourceId = loom::pnr::detail::memResourceId(
        hardwareName, MemAccessKind::Load, identity, i);
    operandEndpoints.try_emplace(EndpointKey{op, operandBase},
                                 endpointFor(resourceId, "operand", 0));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 1},
                                 endpointFor(resourceId, "operand", 1));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase},
                                endpointFor(resourceId, "result", 0));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase + 1},
                                endpointFor(resourceId, "result", 1));
    operandBase += 2;
    resultBase += 2;
  }
  for (std::uint64_t i = 0; i < identity.storeCount; ++i) {
    std::string resourceId = loom::pnr::detail::memResourceId(
        hardwareName, MemAccessKind::Store, identity, i);
    operandEndpoints.try_emplace(EndpointKey{op, operandBase},
                                 endpointFor(resourceId, "operand", 0));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 1},
                                 endpointFor(resourceId, "operand", 1));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 2},
                                 endpointFor(resourceId, "operand", 2));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase},
                                endpointFor(resourceId, "result", 0));
    operandBase += 3;
    resultBase += 1;
  }
}

void recordEndpointMapValues(
    HardwareTopology &topology,
    const std::map<EndpointKey, std::string> &operandEndpoints,
    const std::map<EndpointKey, std::string> &resultEndpoints) {
  for (const auto &[_, endpoint] : operandEndpoints)
    topology.endpoints.insert(endpoint);
  for (const auto &[_, endpoint] : resultEndpoints)
    topology.endpoints.insert(endpoint);
}

void addYieldBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const std::map<EndpointKey, std::string> &resultEndpoints,
    HardwareTopology &topology) {
  if (op->getName().getStringRef() != "fabric.yield")
    return;
  mlir::Operation *parent = op->getParentOp();
  if (!parent)
    return;
  if (!isFabricBoundaryOp(parent->getName().getStringRef()))
    return;
  auto parentId = fabricBoundaryIds.find(parent);
  if (parentId == fabricBoundaryIds.end())
    return;
  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex) {
    std::optional<std::string> sourceEndpoint =
        sourceEndpointForValue(op->getOperand(operandIndex), resultEndpoints);
    if (!sourceEndpoint)
      continue;
    auto sinkIt = resultEndpoints.find(EndpointKey{parent, operandIndex});
    if (sinkIt == resultEndpoints.end())
      continue;
    addTopologySegment(topology,
                       HardwareRouteSegment{"module_path", *sourceEndpoint,
                                            sinkIt->second, parentId->second});
  }
}

void addFuToPeBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const std::map<EndpointKey, std::string> &resultEndpoints,
    HardwareTopology &topology) {
  if (op->getName().getStringRef() != "fabric.fu")
    return;
  mlir::Operation *parent = op->getParentOp();
  if (!parent || parent->getName().getStringRef() != "fabric.pe")
    return;
  auto peId = fabricBoundaryIds.find(parent);
  if (peId == fabricBoundaryIds.end())
    return;
  unsigned resultCount = std::min(op->getNumResults(), parent->getNumResults());
  for (unsigned resultIndex = 0; resultIndex < resultCount; ++resultIndex) {
    auto sourceIt = resultEndpoints.find(EndpointKey{op, resultIndex});
    auto sinkIt = resultEndpoints.find(EndpointKey{parent, resultIndex});
    if (sourceIt == resultEndpoints.end() || sinkIt == resultEndpoints.end())
      continue;
    addTopologySegment(topology,
                       HardwareRouteSegment{"module_path", sourceIt->second,
                                            sinkIt->second, peId->second});
  }
}

HardwareTopology buildHardwareTopology(
    mlir::Operation *hardware, llvm::StringRef hardwareName,
    const llvm::StringMap<HardwareArtifactResource> &resources,
    llvm::ArrayRef<ConcreteMemOccurrence> memOccurrences) {
  HardwareTopology topology;
  std::map<EndpointKey, std::string> operandEndpoints;
  std::map<EndpointKey, std::string> resultEndpoints;
  llvm::DenseMap<mlir::Operation *, std::string> fabricBoundaryIds;
  for (const auto &entry : resources) {
    const HardwareArtifactResource &resource = entry.getValue();
    if (resource.resourceKind == "fabric.op" && resource.op)
      addGenericEndpointMaps(resource.op, entry.getKey(), operandEndpoints,
                             resultEndpoints);
  }

  llvm::StringMap<unsigned> fabricBoundaryCounts;
  llvm::StringMap<unsigned> routeResourceCounts;
  unsigned memOccurrenceIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!loom::pnr::detail::isConcreteHardwareOperation(op, hardware))
      return;
    llvm::StringRef opName = op->getName().getStringRef();
    if (isFabricBoundaryOp(opName)) {
      unsigned index = fabricBoundaryCounts[opName]++;
      std::string resourceId =
          (hardwareName + "::" + opName + "#" + llvm::Twine(index)).str();
      fabricBoundaryIds.try_emplace(op, resourceId);
      addGenericEndpointMaps(op, resourceId, operandEndpoints, resultEndpoints);
      return;
    }
    if (opName == "fabric.mem") {
      assert(memOccurrenceIndex < memOccurrences.size());
      const ConcreteMemOccurrence &occurrence =
          memOccurrences[memOccurrenceIndex++];
      assert(occurrence.op == op);
      addMemEndpointMaps(op, hardwareName, occurrence.identity,
                         operandEndpoints, resultEndpoints);
      return;
    }
    if (!isRouteResourceOp(opName))
      return;
    std::string kind = routeResourceKind(opName);
    unsigned index = routeResourceCounts[kind]++;
    addGenericEndpointMaps(
        op, (hardwareName + "::" + kind + "#" + llvm::Twine(index)).str(),
        operandEndpoints, resultEndpoints);
  });
  recordEndpointMapValues(topology, operandEndpoints, resultEndpoints);

  unsigned ssaEdgeIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!loom::pnr::detail::isConcreteHardwareOperation(op, hardware))
      return;
    llvm::StringRef opName = op->getName().getStringRef();
    for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
         ++operandIndex) {
      std::optional<std::string> sourceEndpoint =
          sourceEndpointForValue(op->getOperand(operandIndex), resultEndpoints);
      if (!sourceEndpoint)
        continue;
      auto destIt = operandEndpoints.find(EndpointKey{op, operandIndex});
      if (destIt == operandEndpoints.end())
        continue;
      addTopologySegment(
          topology,
          HardwareRouteSegment{
              "resource_edge", *sourceEndpoint, destIt->second,
              (hardwareName + "::ssa_edge#" + llvm::Twine(ssaEdgeIndex++))
                  .str()});
    }

    addYieldBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);
    addFuToPeBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);

    if (!isRouteResourceOp(opName))
      return;
    std::optional<std::string> destId;
    auto firstOperand = operandEndpoints.find(EndpointKey{op, 0});
    if (firstOperand != operandEndpoints.end()) {
      llvm::StringRef endpoint = firstOperand->second;
      std::size_t dot = endpoint.rfind(".operand");
      if (dot != std::string::npos)
        destId = endpoint.take_front(dot).str();
    }
    if (!destId)
      return;
    llvm::SmallVector<std::string, 4> switchRows;
    if (opName == "fabric.switch")
      switchRows = switchConnectivityRows(op);
    for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
         ++operandIndex) {
      for (unsigned resultIndex = 0; resultIndex < op->getNumResults();
           ++resultIndex) {
        if (opName == "fabric.switch" &&
            !switchConnectsInputToOutput(switchRows, operandIndex, resultIndex,
                                         op->getNumOperands()))
          continue;
        addTopologySegment(
            topology,
            HardwareRouteSegment{internalRouteSegmentKind(opName),
                                 endpointFor(*destId, "operand", operandIndex),
                                 endpointFor(*destId, "result", resultIndex),
                                 *destId});
      }
    }
  });
  return topology;
}

llvm::Error validateHardwareArtifact(llvm::StringRef hardwareMlirPath,
                                     llvm::StringRef hardwareName,
                                     const llvm::json::Object &mapping) {
  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  for (const llvm::json::Value &routeValue : *routes) {
    const llvm::json::Object *route = routeValue.getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    for (const llvm::json::Value &segmentValue : *segments) {
      const llvm::json::Object *segment = segmentValue.getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      for (llvm::StringRef key :
           {"segment_id", "segment_kind", "source_endpoint", "sink_endpoint"}) {
        if (!segment->getString(key))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment lacks string field %s", key.str().c_str());
      }
    }
  }
  if (hardwareMlirPath.empty()) {
    if (!routes->empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware MLIR is required to verify routed mapping");
    return llvm::Error::success();
  }

  mlir::DialectRegistry registry;
  registry.insert<fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(hardwareMlirPath, &context);
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse hardware artifact %s",
                                   hardwareMlirPath.str().c_str());
  auto selectionOrErr =
      selectHardwareForMapping(*module, hardwareName, mapping);
  if (!selectionOrErr)
    return selectionOrErr.takeError();
  if (mlir::failed(fabric::elaborateInstances(
          mlir::cast<fabric::ModuleOp>(selectionOrErr->module))))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping estimator could not elaborate selected fabric.module @%s",
        selectionOrErr->moduleName.c_str());
  llvm::SmallVector<ConcreteMemOccurrence, 2> memOccurrences =
      loom::pnr::detail::collectConcreteMemOccurrences(selectionOrErr->module);
  llvm::StringMap<HardwareArtifactResource> resources =
      collectHardwareArtifactResources(
          selectionOrErr->module, selectionOrErr->moduleName, memOccurrences);
  HardwareTopology topology =
      buildHardwareTopology(selectionOrErr->module, selectionOrErr->moduleName,
                            resources, memOccurrences);
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  std::map<std::string, PlacementInfo> placementBySoftware;
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    auto softwareOrErr =
        requireObjectString(*placement, "software", "mapping placement");
    if (!softwareOrErr)
      return softwareOrErr.takeError();
    auto hardwareOrErr =
        requireObjectString(*placement, "hardware", "mapping placement");
    if (!hardwareOrErr)
      return hardwareOrErr.takeError();
    auto resourceKindOrErr =
        requireObjectString(*placement, "resource_kind", "mapping placement");
    if (!resourceKindOrErr)
      return resourceKindOrErr.takeError();
    auto operationOrErr =
        requireObjectString(*placement, "operation", "mapping placement");
    if (!operationOrErr)
      return operationOrErr.takeError();
    auto scheduleOrErr =
        requireObjectString(*placement, "schedule", "mapping placement");
    if (!scheduleOrErr)
      return scheduleOrErr.takeError();

    auto resourceIt = resources.find(*hardwareOrErr);
    if (resourceIt == resources.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware artifact does not contain resource %s",
          hardwareOrErr->c_str());
    if (resourceIt->second.resourceKind != *resourceKindOrErr)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s has kind %s but mapping requires %s",
          hardwareOrErr->c_str(), resourceIt->second.resourceKind.c_str(),
          resourceKindOrErr->c_str());
    if (resourceIt->second.schedule != *scheduleOrErr)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s has schedule %s but mapping requires %s",
          hardwareOrErr->c_str(), resourceIt->second.schedule.c_str(),
          scheduleOrErr->c_str());
    if (!operationMatchesResourceKind(*resourceKindOrErr, *operationOrErr))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "operation %s is incompatible with resource kind %s",
          operationOrErr->c_str(), resourceKindOrErr->c_str());
    if (*resourceKindOrErr == "fabric.op" &&
        !resourceIt->second.supportedOps.contains(*operationOrErr))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s does not support operation %s",
          hardwareOrErr->c_str(), operationOrErr->c_str());
    if (!placementBySoftware
             .try_emplace(*softwareOrErr,
                          PlacementInfo{*hardwareOrErr, *resourceKindOrErr,
                                        *operationOrErr})
             .second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping contains duplicate placement for software %s",
          softwareOrErr->c_str());
  }
  for (const llvm::json::Value &routeValue : *routes) {
    const llvm::json::Object *route = routeValue.getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    auto expectedEndpointsOrErr =
        expectedRouteEndpoints(*route, placementBySoftware);
    if (!expectedEndpointsOrErr)
      return expectedEndpointsOrErr.takeError();
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    std::optional<std::string> previousSink;
    for (auto [segmentIndex, segmentValue] : llvm::enumerate(*segments)) {
      const llvm::json::Object *segment = segmentValue.getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      auto sourceEndpointOrErr = requireObjectString(
          *segment, "source_endpoint", "mapping route segment");
      if (!sourceEndpointOrErr)
        return sourceEndpointOrErr.takeError();
      auto sinkEndpointOrErr = requireObjectString(*segment, "sink_endpoint",
                                                   "mapping route segment");
      if (!sinkEndpointOrErr)
        return sinkEndpointOrErr.takeError();
      if (isPlaceholderRouteEndpoint(*sourceEndpointOrErr) ||
          isPlaceholderRouteEndpoint(*sinkEndpointOrErr))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment endpoint uses placeholder suffix");
      if (segmentIndex == 0 &&
          *sourceEndpointOrErr != expectedEndpointsOrErr->first)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route source endpoint does not match mapped producer");
      if (topology.endpoints.find(*sourceEndpointOrErr) ==
          topology.endpoints.end())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "hardware artifact does not contain route endpoint %s",
            sourceEndpointOrErr->c_str());
      if (topology.endpoints.find(*sinkEndpointOrErr) ==
          topology.endpoints.end())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "hardware artifact does not contain route endpoint %s",
            sinkEndpointOrErr->c_str());
      if (previousSink && *previousSink != *sourceEndpointOrErr)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not contiguous");
      std::string edgeKey =
          topologyEdgeKey(*sourceEndpointOrErr, *sinkEndpointOrErr);
      if (topology.edgeKeys.find(edgeKey) == topology.edgeKeys.end())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not present in hardware topology");
      auto topologySegment = topology.segmentByEdge.find(edgeKey);
      if (std::optional<llvm::StringRef> segmentKind =
              segment->getString("segment_kind")) {
        if (topologySegment != topology.segmentByEdge.end() &&
            topologySegment->second.segmentKind != *segmentKind)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment kind %s does not match hardware topology "
              "kind %s",
              segmentKind->str().c_str(),
              topologySegment->second.segmentKind.c_str());
      }
      previousSink = *sinkEndpointOrErr;
      if (const llvm::json::Value *hardwareRefValue =
              segment->get("hardware_ref")) {
        std::optional<llvm::StringRef> hardwareRef =
            hardwareRefValue->getAsString();
        if (!hardwareRef)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment hardware_ref is not a string");
        if (topologySegment != topology.segmentByEdge.end() &&
            topologySegment->second.hardwareRef != *hardwareRef)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment hardware_ref %s does not match hardware "
              "topology ref %s",
              hardwareRef->str().c_str(),
              topologySegment->second.hardwareRef.c_str());
      }
    }
    if (previousSink && *previousSink != expectedEndpointsOrErr->second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping route sink endpoint does not match mapped consumer");
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::json::Object> parseJsonObject(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError())
    return llvm::createStringError(ec, "could not read %s", path.str().c_str());
  auto parsedOrErr = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsedOrErr)
    return parsedOrErr.takeError();
  const llvm::json::Object *object = parsedOrErr->getAsObject();
  if (!object)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s is not a JSON object",
                                   path.str().c_str());
  return *object;
}

llvm::Expected<std::uint64_t>
requireNonNegativeInteger(const llvm::json::Object &object, llvm::StringRef key,
                          llvm::StringRef path) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks non-negative integer field %s",
                                   path.str().c_str(), key.str().c_str());
  return static_cast<std::uint64_t>(*value);
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef key,
                                          llvm::StringRef path) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks string field %s",
                                   path.str().c_str(), key.str().c_str());
  return value->str();
}

llvm::Expected<std::string>
requireObjectString(const llvm::json::Object &object, llvm::StringRef key,
                    llvm::StringRef diagnosticContext) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return llvm::createStringError(
        std::errc::invalid_argument, "%s lacks string field %s",
        diagnosticContext.str().c_str(), key.str().c_str());
  return value->str();
}

llvm::Expected<std::string>
requireMappingStatus(const llvm::json::Object &object, llvm::StringRef path) {
  std::optional<llvm::StringRef> schemaVersion =
      object.getString("schema_version");
  if (!schemaVersion || *schemaVersion != kMappingSchemaVersion)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s has unsupported schema_version; expected string \"%s\"",
        path.str().c_str(), kMappingSchemaVersion.data());
  std::optional<llvm::StringRef> kind = object.getString("kind");
  if (!kind || *kind != "pnr_mapping")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s has wrong kind", path.str().c_str());
  std::optional<llvm::StringRef> status = object.getString("status");
  if (!status || status->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks string field status",
                                   path.str().c_str());
  if (*status != "pass" && *status != "fail" && *status != "unsupported" &&
      *status != "skipped" && *status != "blocked" && *status != "not_run")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping artifact status %s is not supported", status->str().c_str());
  return status->str();
}

llvm::Error validateMappingCounts(const llvm::json::Object &mapping,
                                  llvm::StringRef mappingPath,
                                  llvm::StringRef mappingStatus) {
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  auto placedOrErr =
      requireNonNegativeInteger(mapping, "placed_records", mappingPath);
  if (!placedOrErr)
    return placedOrErr.takeError();
  if (*placedOrErr != placements->size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping placed_records field %llu does not match placements array "
        "size %llu",
        static_cast<unsigned long long>(*placedOrErr),
        static_cast<unsigned long long>(placements->size()));

  auto unplacedOrErr =
      requireNonNegativeInteger(mapping, "unplaced_records", mappingPath);
  if (!unplacedOrErr)
    return unplacedOrErr.takeError();
  auto unroutedOrErr =
      requireNonNegativeInteger(mapping, "unrouted_edges", mappingPath);
  if (!unroutedOrErr)
    return unroutedOrErr.takeError();
  const llvm::json::Array *configBitstream =
      mapping.getArray("config_bitstream");
  if (!configBitstream)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks config_bitstream");
  auto configRecordsOrErr =
      requireNonNegativeInteger(mapping, "config_records", mappingPath);
  if (!configRecordsOrErr)
    return configRecordsOrErr.takeError();
  if (*configRecordsOrErr != configBitstream->size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config_records field %llu does not match config_bitstream "
        "size %llu",
        static_cast<unsigned long long>(*configRecordsOrErr),
        static_cast<unsigned long long>(configBitstream->size()));
  if (mappingStatus == "pass" && *unplacedOrErr != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "passing mapping artifact has non-zero unplaced_records");
  if (mappingStatus == "pass" && *unroutedOrErr != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "passing mapping artifact has non-zero unrouted_edges");
  return llvm::Error::success();
}

llvm::Error validateMappingConfigIdentity(const llvm::json::Object &mapping,
                                          llvm::StringRef mappingPath,
                                          const loom::ResolvedConfig &config) {
  std::optional<llvm::StringRef> identitySpelling =
      mapping.getString("resolved_config_identity");
  if (!identitySpelling)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "config_missing_required_profile: %s lacks resolved_config_identity",
        mappingPath.str().c_str());

  std::optional<llvm::StringRef> configId = mapping.getString("config_id");
  if (!configId || configId->empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "config_missing_required_profile: %s lacks config_id",
        mappingPath.str().c_str());
  if (*configId != config.configId)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "config_identity_mismatch: mapping config_id %s does not match "
        "resolved config_id %s",
        configId->str().c_str(), config.configId.c_str());

  auto parsedIdentity = loom::parseArtifactIdentityHex(*identitySpelling);
  if (!parsedIdentity) {
    llvm::consumeError(parsedIdentity.takeError());
    return llvm::createStringError(
        std::errc::invalid_argument,
        "config_identity_mismatch: mapping resolved_config_identity is not a "
        "valid ArtifactIdentity");
  }

  const loom::ArtifactIdentity expected = loom::resolvedConfigIdentity(config);
  if (*parsedIdentity != expected)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "config_identity_mismatch: mapping resolved_config_identity %s does "
        "not match resolved configuration %s",
        identitySpelling->str().c_str(),
        loom::formatArtifactIdentityHex(expected).c_str());

  return llvm::Error::success();
}

bool isSupportedResourceKind(llvm::StringRef resourceKind) {
  return resourceKind == "fabric.op" || resourceKind == "fabric.mem.load" ||
         resourceKind == "fabric.mem.store";
}

bool isSupportedSchedule(llvm::StringRef schedule) {
  return schedule == "spatial" || schedule == "temporal";
}

bool isWidthAdapterOperation(llvm::StringRef operation) {
  return operation == "llvm.trunc" || operation == "llvm.zext";
}

bool isComputedStoreAddressEdge(llvm::StringRef edgeRef) {
  return edgeRef.contains("->dataflow.store#") && edgeRef.contains(".operand1");
}

bool isComputedLoadAddressEdge(llvm::StringRef edgeRef) {
  return edgeRef.contains("->dataflow.load#") &&
         edgeRef.contains(".operand1") &&
         !edgeRef.starts_with("dataflow.stream#");
}

bool hasExpensiveFunctionalUnit(llvm::StringRef operation) {
  return operation == "arith.muli" || operation == "arith.mulf" ||
         operation == "arith.divsi" || operation == "arith.divui" ||
         operation == "arith.divf" || operation == "arith.remsi" ||
         operation == "arith.remui" || operation == "arith.remf" ||
         operation == "llvm.intr.fmuladd";
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

llvm::Error expectConfig(const ConfigEntries &entries, llvm::StringRef target,
                         llvm::StringRef registerName, llvm::StringRef source,
                         llvm::StringRef expectedValue) {
  std::string key = configKey(target, registerName, source);
  auto it = entries.valuesByFullKey.find(key);
  if (it == entries.valuesByFullKey.end())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config bitstream is missing %s for %s",
        registerName.str().c_str(), target.str().c_str());
  if (it->second != expectedValue)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config bitstream value for %s on %s is %s, expected %s",
        registerName.str().c_str(), target.str().c_str(), it->second.c_str(),
        expectedValue.str().c_str());
  return llvm::Error::success();
}

llvm::Error collectConfigEntries(const llvm::json::Object &mapping,
                                 llvm::StringRef mappingArtifactPath,
                                 MappingEstimateReport &report,
                                 ConfigEntries &entries) {
  const llvm::json::Array *configArray = mapping.getArray("config_bitstream");
  if (!configArray)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks config_bitstream");
  auto declaredRecordsOrErr =
      requireNonNegativeInteger(mapping, "config_records", mappingArtifactPath);
  if (!declaredRecordsOrErr)
    return declaredRecordsOrErr.takeError();
  if (*declaredRecordsOrErr != configArray->size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config_records field %llu does not match config_bitstream "
        "size %llu",
        static_cast<unsigned long long>(*declaredRecordsOrErr),
        static_cast<unsigned long long>(configArray->size()));
  report.configRecords = configArray->size();
  report.configLoadScore = report.configRecords / kConfigRecordsPerScoreUnit;
  for (const llvm::json::Value &value : *configArray) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream entry is not an object");
    auto targetOrErr =
        requireObjectString(*entry, "target", "config bitstream entry");
    if (!targetOrErr)
      return targetOrErr.takeError();
    auto registerOrErr =
        requireObjectString(*entry, "register", "config bitstream entry");
    if (!registerOrErr)
      return registerOrErr.takeError();
    auto sourceOrErr =
        requireObjectString(*entry, "source", "config bitstream entry");
    if (!sourceOrErr)
      return sourceOrErr.takeError();
    auto valueOrErr =
        requireObjectString(*entry, "value", "config bitstream entry");
    if (!valueOrErr)
      return valueOrErr.takeError();

    std::string regKey = registerKey(*targetOrErr, *registerOrErr);
    if (!entries.writtenRegisters.insert(regKey).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping config bitstream writes register %s on %s more than once",
          registerOrErr->c_str(), targetOrErr->c_str());
    std::string key = configKey(*targetOrErr, *registerOrErr, *sourceOrErr);
    if (!entries.valuesByFullKey.try_emplace(key, *valueOrErr).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping config bitstream contains duplicate assignment for %s",
          targetOrErr->c_str());
  }
  return llvm::Error::success();
}

llvm::Error collectPlacementStats(const llvm::json::Object &mapping,
                                  MappingEstimateReport &report) {
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  report.placedRecords = placements->size();
  std::uint64_t loadPlacements = 0;
  std::uint64_t storePlacements = 0;
  std::uint64_t widthAdapterPlacements = 0;
  std::uint64_t functionalUnitPlacements = 0;
  std::set<std::string> placedOperationKinds;
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    std::optional<llvm::StringRef> schedule = placement->getString("schedule");
    if (!schedule || !isSupportedSchedule(*schedule))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement schedule %s is not supported",
          schedule ? schedule->str().c_str() : "<missing>");
    if (*schedule == "temporal")
      ++report.temporalPlacements;
    else
      ++report.spatialPlacements;

    std::optional<llvm::StringRef> resourceKind =
        placement->getString("resource_kind");
    if (!resourceKind || !isSupportedResourceKind(*resourceKind))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement resource_kind %s is not supported",
          resourceKind ? resourceKind->str().c_str() : "<missing>");
    std::optional<llvm::StringRef> operation =
        placement->getString("operation");
    if (!operation || !operationMatchesResourceKind(*resourceKind, *operation))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "operation %s is incompatible with "
                                     "resource kind %s",
                                     operation ? operation->str().c_str()
                                               : "<missing>",
                                     resourceKind->str().c_str());
    if (*resourceKind == "fabric.op" && operation &&
        isWidthAdapterOperation(*operation))
      ++widthAdapterPlacements;
    if (*resourceKind == "fabric.op" && operation &&
        hasExpensiveFunctionalUnit(*operation))
      ++functionalUnitPlacements;
    if (operation && !operation->empty())
      placedOperationKinds.insert(operation->str());
    if (*resourceKind == "fabric.mem.load")
      ++loadPlacements;
    if (*resourceKind == "fabric.mem.store")
      ++storePlacements;
  }
  report.memoryAccessScore =
      (loadPlacements + storePlacements) * kMemoryAccessWeight +
      storePlacements * kStoreCommitWeight;
  report.widthAdapterScore = widthAdapterPlacements * kWidthAdapterWeight;
  report.functionalUnitScore = functionalUnitPlacements * kFunctionalUnitWeight;
  report.resourceMixScore = placedOperationKinds.size();
  report.temporalConflictScore =
      report.temporalPlacements == 0
          ? 0
          : report.temporalPlacements * (1 + report.routedEdges);
  return llvm::Error::success();
}

llvm::Expected<RouteStats>
collectRouteStats(const llvm::json::Object &mapping,
                  llvm::StringRef mappingArtifactPath) {
  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  auto routedEdgesOrErr =
      requireNonNegativeInteger(mapping, "routed_edges", mappingArtifactPath);
  if (!routedEdgesOrErr)
    return routedEdgesOrErr.takeError();
  RouteStats stats;
  stats.routeCount = routes->size();
  if (*routedEdgesOrErr != stats.routeCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping routed_edges field %llu does not match routes array size %llu",
        static_cast<unsigned long long>(*routedEdgesOrErr),
        static_cast<unsigned long long>(stats.routeCount));
  for (const llvm::json::Value &value : *routes) {
    const llvm::json::Object *route = value.getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    auto edgeRefOrErr =
        requireObjectString(*route, "edge_ref", "mapping route");
    if (!edgeRefOrErr)
      return edgeRefOrErr.takeError();
    auto statusOrErr = requireObjectString(*route, "status", "mapping route");
    if (!statusOrErr)
      return statusOrErr.takeError();
    if (*statusOrErr != "routed")
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route status %s is not routed",
                                     statusOrErr->c_str());
    if (isComputedLoadAddressEdge(*edgeRefOrErr))
      ++stats.computedLoadAddressRoutes;
    if (isComputedStoreAddressEdge(*edgeRefOrErr))
      ++stats.computedStoreAddressRoutes;
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    for (const llvm::json::Value &segmentValue : *segments) {
      const llvm::json::Object *segment = segmentValue.getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      for (llvm::StringRef key :
           {"segment_id", "segment_kind", "source_endpoint", "sink_endpoint"}) {
        if (!segment->getString(key))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment lacks string field %s", key.str().c_str());
      }
      ++stats.segmentCount;
    }
  }
  return stats;
}

llvm::Error validateConfigCoverage(const llvm::json::Object &mapping,
                                   const MappingEstimateReport &report,
                                   const ConfigEntries &configEntries) {
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    auto softwareOrErr =
        requireObjectString(*placement, "software", "mapping placement");
    if (!softwareOrErr)
      return softwareOrErr.takeError();
    auto hardwareOrErr =
        requireObjectString(*placement, "hardware", "mapping placement");
    if (!hardwareOrErr)
      return hardwareOrErr.takeError();
    auto operationOrErr =
        requireObjectString(*placement, "operation", "mapping placement");
    if (!operationOrErr)
      return operationOrErr.takeError();
    auto resourceKindOrErr =
        requireObjectString(*placement, "resource_kind", "mapping placement");
    if (!resourceKindOrErr)
      return resourceKindOrErr.takeError();
    auto scheduleOrErr =
        requireObjectString(*placement, "schedule", "mapping placement");
    if (!scheduleOrErr)
      return scheduleOrErr.takeError();
    std::string source = "placement:" + *softwareOrErr;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "software_id", source, *softwareOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "operation", source, *operationOrErr))
      return err;
    if (llvm::Error err =
            expectConfig(configEntries, *hardwareOrErr, "resource_kind", source,
                         *resourceKindOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "schedule", source, *scheduleOrErr))
      return err;
  }

  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  for (std::size_t i = 0; i < routes->size(); ++i) {
    const llvm::json::Object *route = (*routes)[i].getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    auto recordOrErr =
        requireObjectString(*route, "record_id", "mapping route");
    if (!recordOrErr)
      return recordOrErr.takeError();
    auto fromOrErr = requireObjectString(*route, "from", "mapping route");
    if (!fromOrErr)
      return fromOrErr.takeError();
    auto toOrErr = requireObjectString(*route, "to", "mapping route");
    if (!toOrErr)
      return toOrErr.takeError();
    std::string source = "route:" + *recordOrErr;
    std::string target = report.mappingId + "::" + *recordOrErr;
    if (llvm::Error err = expectConfig(configEntries, target,
                                       "from_software_id", source, *fromOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, target, "to_software_id",
                                       source, *toOrErr))
      return err;
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    if (llvm::Error err =
            expectConfig(configEntries, target, "segment_count", source,
                         std::to_string(segments->size())))
      return err;
    for (std::size_t segmentIndex = 0; segmentIndex < segments->size();
         ++segmentIndex) {
      const llvm::json::Object *segment =
          (*segments)[segmentIndex].getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      for (auto [jsonKey, registerName] :
           {std::pair<llvm::StringRef, llvm::StringRef>{"segment_kind", "kind"},
            {"source_endpoint", "source_endpoint"},
            {"sink_endpoint", "sink_endpoint"}}) {
        std::optional<llvm::StringRef> value = segment->getString(jsonKey);
        if (!value)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment lacks string field %s",
              jsonKey.str().c_str());
        std::string segmentRegister = prefix + registerName.str();
        if (llvm::Error err = expectConfig(configEntries, target,
                                           segmentRegister, source, *value))
          return err;
      }
      if (std::optional<llvm::StringRef> value =
              segment->getString("hardware_ref")) {
        std::string segmentRegister = prefix + "hardware_ref";
        if (llvm::Error err = expectConfig(configEntries, target,
                                           segmentRegister, source, *value))
          return err;
      }
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<MappingEstimateReport>
loom::pnr::estimateMapping(const MappingEstimateOptions &options) {
  auto mappingOrErr = parseJsonObject(options.mappingArtifactPath);
  if (!mappingOrErr)
    return mappingOrErr.takeError();

  auto mappingStatusOrErr =
      requireMappingStatus(*mappingOrErr, options.mappingArtifactPath);
  if (!mappingStatusOrErr)
    return mappingStatusOrErr.takeError();
  if (llvm::Error err = validateMappingCounts(
          *mappingOrErr, options.mappingArtifactPath, *mappingStatusOrErr))
    return std::move(err);

  loom::ResolvedConfig resolvedConfig = loom::defaultResolvedConfig();
  if (llvm::Error err = validateMappingConfigIdentity(
          *mappingOrErr, options.mappingArtifactPath, resolvedConfig))
    return std::move(err);

  MappingEstimateReport report(loom::resolvedConfigIdentity(resolvedConfig));
  report.configId = resolvedConfig.configId;
  auto workloadOrErr =
      requireString(*mappingOrErr, "workload", options.mappingArtifactPath);
  if (!workloadOrErr)
    return workloadOrErr.takeError();
  report.workload = *workloadOrErr;

  auto hardwareOrErr =
      requireString(*mappingOrErr, "hardware", options.mappingArtifactPath);
  if (!hardwareOrErr)
    return hardwareOrErr.takeError();
  report.hardware = *hardwareOrErr;
  report.hardwareArtifact = options.hardwareMlirPath;
  auto mappingIdOrErr =
      requireString(*mappingOrErr, "mapping_id", options.mappingArtifactPath);
  if (!mappingIdOrErr)
    return mappingIdOrErr.takeError();
  report.mappingId = *mappingIdOrErr;
  if (llvm::Error err = validateHardwareArtifact(
          options.hardwareMlirPath, report.hardware, *mappingOrErr))
    return std::move(err);

  if (*mappingStatusOrErr != "pass") {
    auto routeStatsOrErr =
        collectRouteStats(*mappingOrErr, options.mappingArtifactPath);
    if (!routeStatsOrErr)
      return routeStatsOrErr.takeError();
    report.routedEdges = routeStatsOrErr->routeCount;
    report.routeSegments = routeStatsOrErr->segmentCount;
    auto configRecordsOrErr = requireNonNegativeInteger(
        *mappingOrErr, "config_records", options.mappingArtifactPath);
    if (!configRecordsOrErr)
      return configRecordsOrErr.takeError();
    report.configRecords = *configRecordsOrErr;
    if (llvm::Error err = collectPlacementStats(*mappingOrErr, report))
      return std::move(err);
    report.memoryAccessScore = 0;
    report.widthAdapterScore = 0;
    report.functionalUnitScore = 0;
    report.resourceMixScore = 0;
    report.loadAddressScore = 0;
    report.storeAddressScore = 0;
    report.configLoadScore = 0;
    report.temporalConflictScore = 0;
    report.totalCostScore = 0;
    report.status = "blocked";
    report.diagnostic = "mapping artifact status " + *mappingStatusOrErr +
                        " prevents a complete mapping estimate";
    if (const llvm::json::Array *diagnostics =
            mappingOrErr->getArray("diagnostics")) {
      if (!diagnostics->empty()) {
        if (std::optional<llvm::StringRef> detail =
                (*diagnostics)[0].getAsString())
          report.diagnostic += ": " + detail->str();
      }
    }
    return report;
  }

  auto routeStatsOrErr =
      collectRouteStats(*mappingOrErr, options.mappingArtifactPath);
  if (!routeStatsOrErr)
    return routeStatsOrErr.takeError();
  report.routedEdges = routeStatsOrErr->routeCount;
  report.routeSegments = routeStatsOrErr->segmentCount;
  report.routeSegmentScore = report.routeSegments * kRouteSegmentWeight;
  report.loadAddressScore =
      routeStatsOrErr->computedLoadAddressRoutes * kLoadAddressWeight;
  report.storeAddressScore =
      routeStatsOrErr->computedStoreAddressRoutes * kStoreAddressWeight;

  ConfigEntries configEntries;
  if (llvm::Error err = collectConfigEntries(
          *mappingOrErr, options.mappingArtifactPath, report, configEntries))
    return std::move(err);
  if (llvm::Error err =
          validateConfigCoverage(*mappingOrErr, report, configEntries))
    return std::move(err);

  if (llvm::Error err = collectPlacementStats(*mappingOrErr, report))
    return std::move(err);

  report.totalCostScore = report.routeSegmentScore + report.memoryAccessScore +
                          report.widthAdapterScore +
                          report.functionalUnitScore + report.resourceMixScore +
                          report.loadAddressScore + report.storeAddressScore +
                          report.configLoadScore + report.temporalConflictScore;
  report.status = "pass";
  return report;
}
