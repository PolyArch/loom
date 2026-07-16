#include "MappingHardware.h"
#include "MappingInternal.h"

#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace loom::pnr::detail {

namespace {

struct EndpointKey {
  mlir::Operation *op = nullptr;
  unsigned index = 0;

  bool operator<(const EndpointKey &other) const {
    return std::tie(op, index) < std::tie(other.op, other.index);
  }
};

constexpr unsigned kExhaustivePlacementNodeLimit = 20;

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

bool requiresSemanticEncodingAwarePnr(fabric::OpOp op) {
  auto fu = op->getParentOfType<fabric::FuOp>();
  if (!fu)
    return false;
  return fu->getAttrOfType<mlir::ArrayAttr>("valid_encodings") ||
         fabric::classifyFabricOpModes(op).kind !=
             fabric::FabricOpModeKind::Legacy;
}

void appendHwParamOptions(mlir::Operation *op, HardwareResource &resource) {
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  auto fabricOp = mlir::cast<fabric::OpOp>(op);
  if (!hwParams || requiresSemanticEncodingAwarePnr(fabricOp))
    return;
  for (mlir::Attribute paramSet : hwParams) {
    auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(paramSet);
    if (!dict)
      continue;
    for (mlir::NamedAttribute namedAttr : dict) {
      std::set<std::string> &values =
          resource.hwParamOptions[namedAttr.getName().getValue().str()];
      if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(namedAttr.getValue())) {
        for (mlir::Attribute value : array)
          values.insert(configValue(value));
        continue;
      }
      values.insert(configValue(namedAttr.getValue()));
    }
  }
}

void appendMemResources(mlir::Operation *op, llvm::StringRef hardwareName,
                        const MemOccurrenceIdentity &identity,
                        llvm::SmallVectorImpl<HardwareResource> &resources) {
  std::string schedule = nearestSchedule(op);
  for (std::uint64_t i = 0; i < identity.loadCount; ++i) {
    resources.push_back(HardwareResource{
        memResourceId(hardwareName, MemAccessKind::Load, identity, i),
        ResourceKind::MemLoad,
        schedule,
        op,
        {},
        {},
        {},
        false});
  }
  for (std::uint64_t i = 0; i < identity.storeCount; ++i) {
    resources.push_back(HardwareResource{
        memResourceId(hardwareName, MemAccessKind::Store, identity, i),
        ResourceKind::MemStore,
        schedule,
        op,
        {},
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
  resource.op = op;
  appendHwParamOptions(op, resource);
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

std::string endpointFor(llvm::StringRef resourceId, llvm::StringRef endpoint,
                        unsigned index) {
  return (resourceId + "." + endpoint + llvm::Twine(index)).str();
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

bool boundaryCarriesSoftwarePayload(fabric::BoundaryOp boundary,
                                    unsigned inputIndex, unsigned outputIndex) {
  if (inputIndex != 0 || outputIndex != 0)
    return false;
  switch (boundary.getDirection()) {
  case fabric::BoundaryDirection::S2t:
  case fabric::BoundaryDirection::T2t:
  case fabric::BoundaryDirection::T2s:
    return true;
  }
  llvm_unreachable("unknown fabric boundary direction");
}

std::tuple<std::string, std::string, std::string, std::string>
routeSegmentTieKey(const HardwareRouteSegment &segment) {
  return {segment.sinkEndpoint, segment.segmentKind, segment.hardwareRef,
          segment.sourceEndpoint};
}

void addTopologySegment(HardwareTopology &topology,
                        HardwareRouteSegment segment) {
  HardwareRouteSegment reverseIndexSegment = segment;
  topology.segmentsBySink[reverseIndexSegment.sinkEndpoint].push_back(
      std::move(reverseIndexSegment));
  topology.segmentsBySource[segment.sourceEndpoint].push_back(
      std::move(segment));
}

void normalizeTopologyRoutes(HardwareTopology &topology) {
  for (auto &[sourceEndpoint, segments] : topology.segmentsBySource) {
    (void)sourceEndpoint;
    llvm::sort(segments, [](const HardwareRouteSegment &lhs,
                            const HardwareRouteSegment &rhs) {
      return routeSegmentTieKey(lhs) < routeSegmentTieKey(rhs);
    });
  }
  for (auto &[sinkEndpoint, segments] : topology.segmentsBySink) {
    (void)sinkEndpoint;
    llvm::sort(segments, [](const HardwareRouteSegment &lhs,
                            const HardwareRouteSegment &rhs) {
      return routeSegmentTieKey(lhs) < routeSegmentTieKey(rhs);
    });
  }
}

unsigned memResultPortBase(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return 0;
  if (mlir::isa<mlir::MemRefType>(op->getResult(0).getType()))
    return 1;
  return 0;
}

std::optional<TransportShape> transportShape(mlir::Type type) {
  if (auto bits = mlir::dyn_cast<fabric::BitsType>(type))
    return TransportShape{TransportKind::Bits, bits.getWidth()};
  if (auto bitsTag = mlir::dyn_cast<fabric::BitsTagType>(type))
    return TransportShape{TransportKind::BitsTag, bitsTag.getWidth()};
  return std::nullopt;
}

mlir::Type physicalOperandType(mlir::Operation *op, unsigned index) {
  if (auto fifo = mlir::dyn_cast<fabric::FifoOp>(op))
    return fifo.getOutput().getType();

  llvm::ArrayRef<mlir::Type> innerTypes;
  if (auto boundary = mlir::dyn_cast<fabric::BoundaryOp>(op))
    innerTypes = boundary.getInnerInputTypes();
  else if (auto switchOp = mlir::dyn_cast<fabric::SwitchOp>(op))
    innerTypes = switchOp.getInnerInputTypes();
  else if (auto mem = mlir::dyn_cast<fabric::MemOp>(op))
    innerTypes = mem.getInnerInputTypes();

  if (!innerTypes.empty() && index < innerTypes.size())
    return innerTypes[index];
  return op->getOperand(index).getType();
}

using HardwareEndpointMap = std::map<EndpointKey, HardwareEndpoint>;

struct SourceEndpointResolution {
  HardwareEndpoint endpoint;
  TransportShape valueShape;
  unsigned payloadCapacity = 0;
  bool explicitKindTransition = false;
};

std::optional<SourceEndpointResolution>
forwardedBlockArgumentEndpoint(mlir::Value value,
                               const HardwareEndpointMap &resultEndpoints);

std::optional<SourceEndpointResolution>
sourceEndpointForValue(mlir::Value value,
                       const HardwareEndpointMap &resultEndpoints) {
  if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
    auto it = resultEndpoints.find(
        EndpointKey{opResult.getOwner(), opResult.getResultNumber()});
    if (it != resultEndpoints.end())
      return SourceEndpointResolution{it->second, it->second.shape,
                                      it->second.shape.payloadWidth, false};
    return std::nullopt;
  }
  return forwardedBlockArgumentEndpoint(value, resultEndpoints);
}

std::optional<SourceEndpointResolution>
forwardedBlockArgumentEndpoint(mlir::Value value,
                               const HardwareEndpointMap &resultEndpoints) {
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
  std::optional<SourceEndpointResolution> source =
      sourceEndpointForValue(parent->getOperand(index), resultEndpoints);
  std::optional<TransportShape> blockShape = transportShape(blockArg.getType());
  if (!source || !blockShape)
    return std::nullopt;

  if (source->valueShape.kind != blockShape->kind) {
    auto pe = mlir::dyn_cast<fabric::PeOp>(parent);
    if (!pe || pe.getSchedule() != fabric::Schedule::Temporal)
      return std::nullopt;
    source->explicitKindTransition = true;
  }
  source->payloadCapacity =
      std::min(source->payloadCapacity, blockShape->payloadWidth);
  source->valueShape = *blockShape;
  return source;
}

void addGenericEndpointMaps(mlir::Operation *op, llvm::StringRef resourceId,
                            HardwareEndpointMap &operandEndpoints,
                            HardwareEndpointMap &resultEndpoints) {
  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex) {
    std::optional<TransportShape> shape =
        transportShape(physicalOperandType(op, operandIndex));
    if (!shape)
      continue;
    operandEndpoints.try_emplace(
        EndpointKey{op, operandIndex},
        HardwareEndpoint{endpointFor(resourceId, "operand", operandIndex),
                         *shape});
  }
  for (unsigned resultIndex = 0; resultIndex < op->getNumResults();
       ++resultIndex) {
    std::optional<TransportShape> shape =
        transportShape(op->getResult(resultIndex).getType());
    if (!shape)
      continue;
    resultEndpoints.try_emplace(
        EndpointKey{op, resultIndex},
        HardwareEndpoint{endpointFor(resourceId, "result", resultIndex),
                         *shape});
  }
}

bool isFabricBoundaryOp(llvm::StringRef opName) {
  return opName == "fabric.fu" || opName == "fabric.pe";
}

void addMemEndpointMaps(mlir::Operation *op, llvm::StringRef hardwareName,
                        const MemOccurrenceIdentity &identity,
                        HardwareEndpointMap &operandEndpoints,
                        HardwareEndpointMap &resultEndpoints) {
  auto addOperand = [&](unsigned opIndex, llvm::StringRef resourceId,
                        unsigned portIndex) {
    std::optional<TransportShape> shape =
        transportShape(physicalOperandType(op, opIndex));
    if (!shape)
      return;
    operandEndpoints.try_emplace(
        EndpointKey{op, opIndex},
        HardwareEndpoint{endpointFor(resourceId, "operand", portIndex),
                         *shape});
  };
  auto addResult = [&](unsigned opIndex, llvm::StringRef resourceId,
                       unsigned portIndex) {
    std::optional<TransportShape> shape =
        transportShape(op->getResult(opIndex).getType());
    if (!shape)
      return;
    resultEndpoints.try_emplace(
        EndpointKey{op, opIndex},
        HardwareEndpoint{endpointFor(resourceId, "result", portIndex), *shape});
  };

  unsigned operandBase = 1;
  unsigned resultBase = memResultPortBase(op);
  for (std::uint64_t i = 0; i < identity.loadCount; ++i) {
    std::string resourceId =
        memResourceId(hardwareName, MemAccessKind::Load, identity, i);
    addOperand(operandBase, resourceId, 0);
    addOperand(operandBase + 1, resourceId, 1);
    addResult(resultBase, resourceId, 0);
    addResult(resultBase + 1, resourceId, 1);
    operandBase += 2;
    resultBase += 2;
  }
  for (std::uint64_t i = 0; i < identity.storeCount; ++i) {
    std::string resourceId =
        memResourceId(hardwareName, MemAccessKind::Store, identity, i);
    addOperand(operandBase, resourceId, 0);
    addOperand(operandBase + 1, resourceId, 1);
    addOperand(operandBase + 2, resourceId, 2);
    addResult(resultBase, resourceId, 0);
    operandBase += 3;
    resultBase += 1;
  }
}

std::optional<HardwareRouteSegment>
makeTopologySegment(llvm::StringRef segmentKind,
                    const SourceEndpointResolution &source,
                    const HardwareEndpoint &sink, llvm::StringRef hardwareRef,
                    bool ownerAllowsKindTransition = false) {
  bool explicitKindTransition =
      source.explicitKindTransition || ownerAllowsKindTransition;
  if (source.endpoint.shape.kind != sink.shape.kind && !explicitKindTransition)
    return std::nullopt;
  return HardwareRouteSegment{
      segmentKind.str(), source.endpoint.id, sink.id, hardwareRef.str(),
      std::min(source.payloadCapacity, sink.shape.payloadWidth)};
}

std::optional<HardwareRouteSegment>
makeTopologySegment(llvm::StringRef segmentKind, const HardwareEndpoint &source,
                    const HardwareEndpoint &sink, llvm::StringRef hardwareRef,
                    bool ownerAllowsKindTransition = false) {
  return makeTopologySegment(segmentKind,
                             SourceEndpointResolution{source, source.shape,
                                                      source.shape.payloadWidth,
                                                      false},
                             sink, hardwareRef, ownerAllowsKindTransition);
}

void addYieldBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const HardwareEndpointMap &resultEndpoints, HardwareTopology &topology) {
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
    std::optional<SourceEndpointResolution> sourceEndpoint =
        sourceEndpointForValue(op->getOperand(operandIndex), resultEndpoints);
    if (!sourceEndpoint)
      continue;
    auto sinkIt = resultEndpoints.find(EndpointKey{parent, operandIndex});
    if (sinkIt == resultEndpoints.end())
      continue;
    auto pe = mlir::dyn_cast<fabric::PeOp>(parent);
    bool temporalPe = pe && pe.getSchedule() == fabric::Schedule::Temporal;
    std::optional<HardwareRouteSegment> segment =
        makeTopologySegment("module_path", *sourceEndpoint, sinkIt->second,
                            parentId->second, temporalPe);
    if (segment)
      addTopologySegment(topology, std::move(*segment));
  }
}

void addFuToPeBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const HardwareEndpointMap &resultEndpoints, HardwareTopology &topology) {
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
    auto pe = mlir::cast<fabric::PeOp>(parent);
    std::optional<HardwareRouteSegment> segment = makeTopologySegment(
        "module_path", sourceIt->second, sinkIt->second, peId->second,
        pe.getSchedule() == fabric::Schedule::Temporal);
    if (segment)
      addTopologySegment(topology, std::move(*segment));
  }
}

std::optional<unsigned>
hardwareOperandIndexForSoftwareEndpoint(ResourceKind kind, mlir::Operation *op,
                                        unsigned softwareOperandIndex) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return softwareOperandIndex;
  case ResourceKind::MemLoad:
    if (softwareOperandIndex == 1)
      return 0;
    if (softwareOperandIndex == 2)
      return 1;
    return std::nullopt;
  case ResourceKind::MemStore:
    if (op && op->getName().getStringRef() == "llvm.store") {
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
  llvm_unreachable("unknown resource kind");
}

std::optional<unsigned>
hardwareResultIndexForSoftwareEndpoint(ResourceKind kind,
                                       unsigned softwareResultIndex) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return softwareResultIndex;
  case ResourceKind::MemLoad:
    if (softwareResultIndex <= 1)
      return softwareResultIndex;
    return std::nullopt;
  case ResourceKind::MemStore:
    if (softwareResultIndex == 0)
      return 0;
    return std::nullopt;
  }
  llvm_unreachable("unknown resource kind");
}

HardwareTopology
buildHardwareTopology(mlir::Operation *hardware, llvm::StringRef hardwareName,
                      const llvm::SmallVectorImpl<HardwareResource> &resources,
                      llvm::ArrayRef<ConcreteMemOccurrence> memOccurrences) {
  HardwareTopology topology;
  HardwareEndpointMap operandEndpoints;
  HardwareEndpointMap resultEndpoints;
  llvm::DenseMap<mlir::Operation *, std::string> fabricBoundaryIds;
  for (const HardwareResource &resource : resources) {
    if (resource.kind == ResourceKind::FabricOp && resource.op)
      addGenericEndpointMaps(resource.op, resource.id, operandEndpoints,
                             resultEndpoints);
  }

  llvm::StringMap<unsigned> fabricBoundaryCounts;
  llvm::StringMap<unsigned> routeResourceCounts;
  unsigned memOccurrenceIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!isConcreteHardwareOperation(op, hardware))
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

  unsigned ssaEdgeIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!isConcreteHardwareOperation(op, hardware))
      return;
    llvm::StringRef opName = op->getName().getStringRef();
    for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
         ++operandIndex) {
      mlir::Value operand = op->getOperand(operandIndex);
      std::optional<SourceEndpointResolution> sourceEndpoint =
          sourceEndpointForValue(operand, resultEndpoints);
      if (!sourceEndpoint)
        continue;
      auto destIt = operandEndpoints.find(EndpointKey{op, operandIndex});
      if (destIt == operandEndpoints.end())
        continue;
      std::string hardwareRef =
          (hardwareName + "::ssa_edge#" + llvm::Twine(ssaEdgeIndex++)).str();
      std::optional<HardwareRouteSegment> segment = makeTopologySegment(
          "resource_edge", *sourceEndpoint, destIt->second, hardwareRef);
      if (segment)
        addTopologySegment(topology, std::move(*segment));
    }

    addYieldBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);
    addFuToPeBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);

    if (!isRouteResourceOp(opName))
      return;
    std::optional<std::string> destId;
    auto firstOperand = operandEndpoints.find(EndpointKey{op, 0});
    if (firstOperand != operandEndpoints.end()) {
      llvm::StringRef endpoint = firstOperand->second.id;
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
        if (auto boundary = mlir::dyn_cast<fabric::BoundaryOp>(op))
          if (!boundaryCarriesSoftwarePayload(boundary, operandIndex,
                                              resultIndex))
            continue;
        auto sourceIt = operandEndpoints.find(EndpointKey{op, operandIndex});
        auto sinkIt = resultEndpoints.find(EndpointKey{op, resultIndex});
        if (sourceIt == operandEndpoints.end() ||
            sinkIt == resultEndpoints.end())
          continue;
        std::optional<HardwareRouteSegment> segment = makeTopologySegment(
            internalRouteSegmentKind(opName), sourceIt->second, sinkIt->second,
            *destId, opName == "fabric.boundary");
        if (segment)
          addTopologySegment(topology, std::move(*segment));
      }
    }
  });
  normalizeTopologyRoutes(topology);
  return topology;
}

} // namespace

llvm::Expected<HardwareModel> collectHardwareModel(mlir::Operation *hardware,
                                                   llvm::StringRef name) {
  bool hasUnsupportedFuConfiguration = false;
  std::optional<std::string> malformedFuConfiguration;
  hardware->walk([&](fabric::OpOp op) {
    if (!isConcreteHardwareOperation(op, hardware))
      return;
    fabric::FabricOpModeClassification classification =
        fabric::classifyFabricOpModes(op);
    if (classification.kind == fabric::FabricOpModeKind::Malformed) {
      if (!malformedFuConfiguration)
        malformedFuConfiguration = classification.diagnostic;
      return;
    }
    if (!hasUnsupportedFuConfiguration && requiresSemanticEncodingAwarePnr(op))
      hasUnsupportedFuConfiguration = true;
  });
  if (malformedFuConfiguration)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "legacy PnR cannot consume malformed fabric.op hw_params in @%s: %s",
        name.str().c_str(), malformedFuConfiguration->c_str());
  if (hasUnsupportedFuConfiguration)
    return llvm::createStringError(
        std::errc::not_supported,
        "legacy PnR cannot consume normalized fabric.op hw_params in @%s; "
        "a selected fabric.fu semantic encoding is required",
        name.str().c_str());

  HardwareModel model;
  unsigned fabricOpIndex = 0;
  llvm::SmallVector<ConcreteMemOccurrence, 2> memOccurrences =
      collectConcreteMemOccurrences(hardware);
  unsigned memOccurrenceIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    if (!isConcreteHardwareOperation(op, hardware))
      return;
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.op") {
      appendFabricOpResource(op, name, fabricOpIndex++, model.resources);
      return;
    }
    if (opName == "fabric.mem") {
      assert(memOccurrenceIndex < memOccurrences.size());
      const ConcreteMemOccurrence &occurrence =
          memOccurrences[memOccurrenceIndex++];
      assert(occurrence.op == op);
      appendMemResources(op, name, occurrence.identity, model.resources);
    }
  });
  if (model.resources.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "hardware has no mappable resources");
  model.topology =
      buildHardwareTopology(hardware, name, model.resources, memOccurrences);
  return model;
}

namespace {

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

    bool operator==(const ProducerRef &other) const {
      return softwareId == other.softwareId && resultIndex == other.resultIndex;
    }
  };

  llvm::DenseMap<mlir::Value, llvm::SmallVector<ProducerRef, 2>> producer;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> adapterForward;
  struct EdgeTransport {
    std::string payloadKind;
    unsigned requiredPayloadWidth = 0;
  };
  std::map<RouteEdgeKey, EdgeTransport> transportByEdge;

  void addProducer(mlir::Value value, ProducerRef ref) {
    auto &refs = producer[value];
    if (llvm::is_contained(refs, ref))
      return;
    refs.push_back(std::move(ref));
  }

  void addForward(mlir::Value value, mlir::Value source) {
    auto &sources = adapterForward[value];
    if (llvm::is_contained(sources, source))
      return;
    sources.push_back(source);
  }

  void resolveInto(mlir::Value value,
                   llvm::SmallVectorImpl<ProducerRef> &resolved,
                   llvm::DenseSet<mlir::Value> &visiting) {
    if (!visiting.insert(value).second)
      return;
    auto direct = producer.find(value);
    if (direct != producer.end()) {
      for (const ProducerRef &ref : direct->second)
        if (!llvm::is_contained(resolved, ref))
          resolved.push_back(ref);
    }
    auto adapter = adapterForward.find(value);
    if (adapter != adapterForward.end())
      for (mlir::Value source : adapter->second)
        resolveInto(source, resolved, visiting);
    visiting.erase(value);
  }

  llvm::SmallVector<ProducerRef, 2> resolve(mlir::Value value) {
    llvm::SmallVector<ProducerRef, 2> resolved;
    llvm::DenseSet<mlir::Value> visiting;
    resolveInto(value, resolved, visiting);
    return resolved;
  }
};

std::string payloadKind(mlir::Value value) {
  if (mlir::isa<mlir::NoneType>(value.getType()))
    return "control";
  return "data";
}

void collectValueProducers(
    mlir::Block &block,
    const llvm::DenseMap<mlir::Operation *, std::string> &nodeIds,
    RouteBuilder &builder) {
  for (mlir::Operation &op : block) {
    if (isStructuredContainerOp(&op)) {
      for (mlir::Region &region : op.getRegions())
        for (mlir::Block &nested : region)
          collectValueProducers(nested, nodeIds, builder);
      if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
        auto condition = whileOp.getConditionOp();
        if (condition && condition.getArgs().size() == op.getNumResults()) {
          for (auto [result, source] :
               llvm::zip(op.getResults(), condition.getArgs()))
            builder.addForward(result, source);
        }
        continue;
      }
      for (mlir::Region &region : op.getRegions()) {
        for (mlir::Block &nested : region) {
          if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(
                  nested.getTerminator())) {
            if (condition.getArgs().size() != op.getNumResults())
              continue;
            for (auto [result, source] :
                 llvm::zip(op.getResults(), condition.getArgs()))
              builder.addForward(result, source);
            continue;
          }
          auto yield =
              mlir::dyn_cast<mlir::scf::YieldOp>(nested.getTerminator());
          if (!yield || yield.getNumOperands() != op.getNumResults())
            continue;
          for (auto [result, source] :
               llvm::zip(op.getResults(), yield.getOperands()))
            builder.addForward(result, source);
        }
      }
      continue;
    }
    auto nodeIt = nodeIds.find(&op);
    if (nodeIt != nodeIds.end()) {
      unsigned resultIndex = 0;
      for (mlir::Value result : op.getResults())
        builder.addProducer(
            result, RouteBuilder::ProducerRef{nodeIt->second, resultIndex++});
      continue;
    }
    if (isPointerBookkeepingOp(&op))
      continue;
    if (!isAdapterOp(&op) || shouldMaterializeAdapterOp(&op) ||
        op.getNumOperands() == 0)
      continue;
    mlir::Value source = op.getOperand(0);
    for (mlir::Value result : op.getResults())
      builder.addForward(result, source);
  }
}

void collectValueProducers(
    mlir::Operation *graph,
    const llvm::DenseMap<mlir::Operation *, std::string> &nodeIds,
    RouteBuilder &builder) {
  collectValueProducers(graph->getRegion(0).front(), nodeIds, builder);
}

void sortNodesByPlacementPriority(llvm::MutableArrayRef<SoftwareNode> nodes,
                                  llvm::ArrayRef<HardwareResource> resources) {
  std::stable_sort(nodes.begin(), nodes.end(),
                   [&](const SoftwareNode &lhs, const SoftwareNode &rhs) {
                     unsigned lhsCount =
                         compatibleResourceCount(lhs, resources);
                     unsigned rhsCount =
                         compatibleResourceCount(rhs, resources);
                     if (lhsCount != rhsCount)
                       return lhsCount < rhsCount;
                     if (lhs.operation != rhs.operation)
                       return lhs.operation < rhs.operation;
                     return false;
                   });
}

std::optional<llvm::SmallVector<RouteSegment, 2>>
findRoute(const HardwareTopology &topology, const std::string &sourceEndpoint,
          const std::string &sinkEndpoint, unsigned requiredPayloadWidth) {
  struct ReverseStep {
    std::string nextEndpoint;
    HardwareRouteSegment segment;
  };

  std::deque<std::string> worklist;
  std::set<std::string> visited;
  std::map<std::string, ReverseStep> nextByEndpoint;

  visited.insert(sinkEndpoint);
  worklist.push_back(sinkEndpoint);

  while (!worklist.empty()) {
    std::string currentEndpoint = std::move(worklist.front());
    worklist.pop_front();
    if (currentEndpoint == sourceEndpoint)
      break;

    auto incomingIt = topology.segmentsBySink.find(currentEndpoint);
    if (incomingIt == topology.segmentsBySink.end())
      continue;
    for (const HardwareRouteSegment &incoming : incomingIt->second) {
      if (incoming.payloadCapacity < requiredPayloadWidth)
        continue;
      if (!visited.insert(incoming.sourceEndpoint).second)
        continue;
      nextByEndpoint[incoming.sourceEndpoint] =
          ReverseStep{currentEndpoint, incoming};
      if (incoming.sourceEndpoint == sourceEndpoint) {
        worklist.clear();
        break;
      }
      worklist.push_back(incoming.sourceEndpoint);
    }
  }

  if (!visited.count(sourceEndpoint))
    return std::nullopt;

  llvm::SmallVector<RouteSegment, 2> path;
  std::string currentEndpoint = sinkEndpoint;
  currentEndpoint = sourceEndpoint;
  while (currentEndpoint != sinkEndpoint) {
    auto stepIt = nextByEndpoint.find(currentEndpoint);
    if (stepIt == nextByEndpoint.end())
      return std::nullopt;
    const HardwareRouteSegment &candidate = stepIt->second.segment;
    RouteSegment segment;
    segment.segmentKind = candidate.segmentKind;
    segment.sourceEndpoint = candidate.sourceEndpoint;
    segment.sinkEndpoint = candidate.sinkEndpoint;
    segment.hardwareRef = candidate.hardwareRef;
    path.push_back(std::move(segment));
    currentEndpoint = stepIt->second.nextEndpoint;
  }
  for (auto [index, segment] : llvm::enumerate(path))
    segment.segmentId = "seg" + std::to_string(index);
  return path;
}

std::string edgeRefFor(const RouteEdgeKey &edge) {
  return edge.producerSoftwareId + ".result" +
         std::to_string(edge.producerResultIndex) + "->" +
         edge.consumerSoftwareId + ".operand" +
         std::to_string(edge.consumerOperandIndex);
}

void addUnroutedEdge(RouteCollection &collection, const RouteEdgeKey &edge,
                     llvm::StringRef payloadKind,
                     llvm::StringRef sourceEndpoint,
                     llvm::StringRef sinkEndpoint, llvm::StringRef diagnostic) {
  UnroutedEdgeRecord record;
  record.edgeRef = edgeRefFor(edge);
  record.producerBinding = "placement:" + edge.producerSoftwareId;
  record.consumerBinding = "placement:" + edge.consumerSoftwareId;
  record.payloadKind = payloadKind.str();
  record.fromSoftwareId = edge.producerSoftwareId;
  record.toSoftwareId = edge.consumerSoftwareId;
  record.sourceEndpoint = sourceEndpoint.str();
  record.sinkEndpoint = sinkEndpoint.str();
  record.diagnostic = diagnostic.str();
  collection.unroutedEdgeDetails.push_back(std::move(record));
  ++collection.unroutedEdges;
}

std::optional<llvm::SmallVector<RouteSegment, 2>>
cachedFindRoute(const HardwareTopology &topology, RouteCache &cache,
                const std::string &sourceEndpoint,
                const std::string &sinkEndpoint,
                unsigned requiredPayloadWidth) {
  RouteCacheKey key{sourceEndpoint, sinkEndpoint, requiredPayloadWidth};
  auto cached = cache.routes.find(key);
  if (cached != cache.routes.end())
    return cached->second;
  std::optional<llvm::SmallVector<RouteSegment, 2>> path =
      findRoute(topology, sourceEndpoint, sinkEndpoint, requiredPayloadWidth);
  auto [it, inserted] = cache.routes.try_emplace(std::move(key), path);
  (void)inserted;
  return it->second;
}

} // namespace

llvm::Expected<RoutingProblem>
buildRoutingProblem(llvm::ArrayRef<SoftwareNode> nodes,
                    mlir::Operation *graph) {
  RouteBuilder builder;
  llvm::DenseMap<mlir::Operation *, std::string> nodeIds = indexNodeIds(nodes);
  collectValueProducers(graph, nodeIds, builder);

  std::map<std::string, ResourceKind> kindBySoftware;
  std::map<std::string, mlir::Operation *> opBySoftware;
  for (const SoftwareNode &node : nodes)
    kindBySoftware.try_emplace(node.id, node.resourceKind);
  for (const SoftwareNode &node : nodes)
    opBySoftware.try_emplace(node.id, node.op);

  for (const SoftwareNode &node : nodes) {
    unsigned operandIndex = 0;
    for (mlir::Value operand : node.op->getOperands()) {
      if (!hardwareOperandIndexForSoftwareEndpoint(node.resourceKind, node.op,
                                                   operandIndex)) {
        ++operandIndex;
        continue;
      }
      llvm::SmallVector<RouteBuilder::ProducerRef, 2> sources =
          builder.resolve(operand);
      if (sources.empty()) {
        ++operandIndex;
        continue;
      }
      unsigned consumerOperandIndex = operandIndex++;
      std::optional<unsigned> requiredPayloadWidth =
          softwareBitWidth(operand.getType());
      if (!requiredPayloadWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "software edge into %s operand %u has unsupported transport type",
            node.id.c_str(), consumerOperandIndex);
      for (const RouteBuilder::ProducerRef &source : sources) {
        if (source.softwareId == node.id)
          continue;
        RouteEdgeKey key{source.softwareId, source.resultIndex, node.id,
                         consumerOperandIndex};
        builder.transportByEdge.try_emplace(
            std::move(key), RouteBuilder::EdgeTransport{payloadKind(operand),
                                                        *requiredPayloadWidth});
      }
    }
  }

  RoutingProblem problem;
  for (const auto &[edge, transport] : builder.transportByEdge) {
    auto fromKind = kindBySoftware.find(edge.producerSoftwareId);
    auto toKind = kindBySoftware.find(edge.consumerSoftwareId);
    if (fromKind == kindBySoftware.end() || toKind == kindBySoftware.end())
      continue;
    auto toOp = opBySoftware.find(edge.consumerSoftwareId);
    std::optional<unsigned> producerResultIndex =
        hardwareResultIndexForSoftwareEndpoint(fromKind->second,
                                               edge.producerResultIndex);
    std::optional<unsigned> consumerOperandIndex =
        hardwareOperandIndexForSoftwareEndpoint(
            toKind->second, toOp == opBySoftware.end() ? nullptr : toOp->second,
            edge.consumerOperandIndex);
    problem.edges.push_back(SoftwareRouteEdge{
        edge, transport.payloadKind, transport.requiredPayloadWidth,
        producerResultIndex, consumerOperandIndex});
  }
  return problem;
}

RouteCollection collectRoutes(const RoutingProblem &problem,
                              llvm::ArrayRef<PlacementRecord> placements,
                              const HardwareTopology &topology,
                              RouteCache &routeCache) {
  llvm::StringMap<std::string> hardwareBySoftware;
  for (const PlacementRecord &placement : placements)
    hardwareBySoftware.try_emplace(placement.softwareId, placement.hardwareId);

  RouteCollection collection;
  std::size_t index = 0;
  for (const SoftwareRouteEdge &routeEdge : problem.edges) {
    const RouteEdgeKey &edge = routeEdge.key;
    llvm::StringRef kind = routeEdge.payloadKind;
    const std::string &from = edge.producerSoftwareId;
    const std::string &to = edge.consumerSoftwareId;
    auto fromHw = hardwareBySoftware.find(from);
    auto toHw = hardwareBySoftware.find(to);
    if (fromHw == hardwareBySoftware.end() || toHw == hardwareBySoftware.end())
      continue;
    if (!routeEdge.producerResultIndex || !routeEdge.consumerOperandIndex) {
      addUnroutedEdge(collection, edge, kind, "", "",
                      "software endpoint has no hardware endpoint");
      continue;
    }
    std::string sourceEndpoint =
        endpointFor(fromHw->second, "result", *routeEdge.producerResultIndex);
    std::string sinkEndpoint =
        endpointFor(toHw->second, "operand", *routeEdge.consumerOperandIndex);
    std::optional<llvm::SmallVector<RouteSegment, 2>> path =
        cachedFindRoute(topology, routeCache, sourceEndpoint, sinkEndpoint,
                        routeEdge.requiredPayloadWidth);
    if (!path) {
      addUnroutedEdge(collection, edge, kind, sourceEndpoint, sinkEndpoint,
                      "no Fabric ADG route connects source to sink");
      continue;
    }
    std::string recordId = "route#" + std::to_string(index++);
    std::string edgeRef = edgeRefFor(edge);
    RouteRecord route;
    route.recordId = recordId;
    route.edgeRef = edgeRef;
    route.producerBinding = "placement:" + from;
    route.consumerBinding = "placement:" + to;
    route.payloadKind = kind.str();
    route.fromSoftwareId = from;
    route.toSoftwareId = to;
    route.segments = std::move(*path);
    collection.routes.push_back(std::move(route));
  }
  return collection;
}

namespace {

bool partialPlacementRoutes(const RoutingProblem &routingProblem,
                            llvm::ArrayRef<PlacementRecord> placements,
                            const HardwareTopology &topology,
                            RouteCache &routeCache) {
  RouteCollection partial =
      collectRoutes(routingProblem, placements, topology, routeCache);
  return partial.unroutedEdges == 0;
}

const SoftwareNode *findNodeById(llvm::ArrayRef<SoftwareNode> nodes,
                                 llvm::StringRef softwareId) {
  for (const SoftwareNode &node : nodes)
    if (node.id == softwareId)
      return &node;
  return nullptr;
}

const HardwareResource *
findResourceById(llvm::ArrayRef<HardwareResource> resources,
                 llvm::StringRef hardwareId) {
  for (const HardwareResource &resource : resources)
    if (resource.id == hardwareId)
      return &resource;
  return nullptr;
}

std::optional<std::size_t>
findPlacementIndex(llvm::ArrayRef<PlacementRecord> placements,
                   llvm::StringRef softwareId) {
  for (auto [index, placement] : llvm::enumerate(placements))
    if (placement.softwareId == softwareId)
      return index;
  return std::nullopt;
}

llvm::StringSet<>
usedHardwareIds(llvm::ArrayRef<PlacementRecord> placements,
                std::optional<std::size_t> exceptIndex = std::nullopt) {
  llvm::StringSet<> used;
  for (auto [index, placement] : llvm::enumerate(placements)) {
    if (exceptIndex && index == *exceptIndex)
      continue;
    used.insert(placement.hardwareId);
  }
  return used;
}

std::uint64_t countUnroutedEdges(const RoutingProblem &routingProblem,
                                 llvm::ArrayRef<PlacementRecord> placements,
                                 const HardwareTopology &topology,
                                 RouteCache &routeCache) {
  RouteCollection routes =
      collectRoutes(routingProblem, placements, topology, routeCache);
  return routes.unroutedEdges;
}

std::optional<llvm::SmallVector<PlacementRecord, 32>>
replacementCandidate(llvm::ArrayRef<SoftwareNode> nodes,
                     llvm::ArrayRef<PlacementRecord> placements,
                     std::size_t index, const HardwareResource &resource) {
  const SoftwareNode *node = findNodeById(nodes, placements[index].softwareId);
  if (!node || !resourceIsCompatible(*node, resource))
    return std::nullopt;

  llvm::SmallVector<PlacementRecord, 32> candidate(placements.begin(),
                                                   placements.end());
  candidate[index] = makePlacementRecord(*node, resource);
  return candidate;
}

std::optional<llvm::SmallVector<PlacementRecord, 32>>
swapCandidate(llvm::ArrayRef<SoftwareNode> nodes,
              llvm::ArrayRef<HardwareResource> resources,
              llvm::ArrayRef<PlacementRecord> placements, std::size_t lhsIndex,
              std::size_t rhsIndex) {
  if (lhsIndex == rhsIndex)
    return std::nullopt;
  const SoftwareNode *lhsNode =
      findNodeById(nodes, placements[lhsIndex].softwareId);
  const SoftwareNode *rhsNode =
      findNodeById(nodes, placements[rhsIndex].softwareId);
  const HardwareResource *lhsResource =
      findResourceById(resources, placements[lhsIndex].hardwareId);
  const HardwareResource *rhsResource =
      findResourceById(resources, placements[rhsIndex].hardwareId);
  if (!lhsNode || !rhsNode || !lhsResource || !rhsResource)
    return std::nullopt;
  if (!resourceIsCompatible(*lhsNode, *rhsResource) ||
      !resourceIsCompatible(*rhsNode, *lhsResource))
    return std::nullopt;

  llvm::SmallVector<PlacementRecord, 32> candidate(placements.begin(),
                                                   placements.end());
  candidate[lhsIndex] = makePlacementRecord(*lhsNode, *rhsResource);
  candidate[rhsIndex] = makePlacementRecord(*rhsNode, *lhsResource);
  return candidate;
}

bool repairUnroutedGreedyPlacements(
    llvm::ArrayRef<SoftwareNode> nodes, const RoutingProblem &routingProblem,
    llvm::ArrayRef<HardwareResource> resources,
    const HardwareTopology &topology,
    llvm::SmallVectorImpl<PlacementRecord> &placements,
    RouteCache &routeCache) {
  std::uint64_t currentUnrouted =
      countUnroutedEdges(routingProblem, placements, topology, routeCache);
  for (std::size_t iteration = 0; iteration < placements.size(); ++iteration) {
    if (currentUnrouted == 0)
      return true;
    RouteCollection routes =
        collectRoutes(routingProblem, placements, topology, routeCache);
    llvm::SmallVector<std::size_t, 16> repairIndices;
    for (const UnroutedEdgeRecord &edge : routes.unroutedEdgeDetails) {
      if (std::optional<std::size_t> producer =
              findPlacementIndex(placements, edge.fromSoftwareId))
        if (!llvm::is_contained(repairIndices, *producer))
          repairIndices.push_back(*producer);
      if (std::optional<std::size_t> consumer =
              findPlacementIndex(placements, edge.toSoftwareId))
        if (!llvm::is_contained(repairIndices, *consumer))
          repairIndices.push_back(*consumer);
    }

    std::uint64_t bestUnrouted = currentUnrouted;
    llvm::SmallVector<PlacementRecord, 32> bestPlacements;
    for (std::size_t index : repairIndices) {
      for (std::size_t otherIndex = 0; otherIndex < placements.size();
           ++otherIndex) {
        std::optional<llvm::SmallVector<PlacementRecord, 32>> candidate =
            swapCandidate(nodes, resources, placements, index, otherIndex);
        if (!candidate)
          continue;
        std::uint64_t candidateUnrouted = countUnroutedEdges(
            routingProblem, *candidate, topology, routeCache);
        if (candidateUnrouted >= bestUnrouted)
          continue;
        bestUnrouted = candidateUnrouted;
        bestPlacements.assign(candidate->begin(), candidate->end());
        if (bestUnrouted == 0)
          break;
      }
      if (bestUnrouted == 0)
        break;

      llvm::StringSet<> used = usedHardwareIds(placements, index);
      for (const HardwareResource &resource : resources) {
        if (used.contains(resource.id))
          continue;
        std::optional<llvm::SmallVector<PlacementRecord, 32>> candidate =
            replacementCandidate(nodes, placements, index, resource);
        if (!candidate)
          continue;
        std::uint64_t candidateUnrouted = countUnroutedEdges(
            routingProblem, *candidate, topology, routeCache);
        if (candidateUnrouted >= bestUnrouted)
          continue;
        bestUnrouted = candidateUnrouted;
        bestPlacements.assign(candidate->begin(), candidate->end());
        if (bestUnrouted == 0)
          break;
      }
      if (bestUnrouted == 0)
        break;
    }
    if (bestUnrouted >= currentUnrouted)
      return false;
    placements.assign(bestPlacements.begin(), bestPlacements.end());
    currentUnrouted = bestUnrouted;
  }
  return currentUnrouted == 0;
}

void clearPlacementState(llvm::MutableArrayRef<HardwareResource> resources,
                         llvm::SmallVectorImpl<PlacementRecord> &placements) {
  for (HardwareResource &resource : resources)
    resource.used = false;
  placements.clear();
}

bool chooseRouteFeasiblePlacements(
    llvm::MutableArrayRef<SoftwareNode> nodes,
    const RoutingProblem &routingProblem,
    llvm::MutableArrayRef<HardwareResource> resources,
    const HardwareTopology &topology,
    llvm::SmallVectorImpl<PlacementRecord> &placements, unsigned nodeIndex,
    RouteCache &routeCache) {
  if (nodeIndex == nodes.size())
    return true;

  SoftwareNode &node = nodes[nodeIndex];
  for (HardwareResource &resource : resources) {
    if (resource.used || !resourceIsCompatible(node, resource))
      continue;
    resource.used = true;
    placements.push_back(makePlacementRecord(node, resource));

    if (partialPlacementRoutes(routingProblem, placements, topology,
                               routeCache) &&
        chooseRouteFeasiblePlacements(nodes, routingProblem, resources,
                                      topology, placements, nodeIndex + 1,
                                      routeCache))
      return true;

    placements.pop_back();
    resource.used = false;
  }
  return false;
}

} // namespace

bool placeRouteFeasible(llvm::MutableArrayRef<SoftwareNode> nodes,
                        const RoutingProblem &routingProblem,
                        llvm::MutableArrayRef<HardwareResource> resources,
                        const HardwareTopology &topology,
                        llvm::SmallVectorImpl<PlacementRecord> &placements,
                        RouteCache &routeCache) {
  sortNodesByPlacementPriority(nodes, resources);
  clearPlacementState(resources, placements);
  bool greedyComplete = true;
  for (SoftwareNode &node : nodes) {
    HardwareResource *resource = claimResource(node, resources);
    if (!resource) {
      greedyComplete = false;
      break;
    }
    placements.push_back(makePlacementRecord(node, *resource));
  }
  if (greedyComplete &&
      partialPlacementRoutes(routingProblem, placements, topology, routeCache))
    return true;
  if (greedyComplete &&
      repairUnroutedGreedyPlacements(nodes, routingProblem, resources, topology,
                                     placements, routeCache))
    return true;

  clearPlacementState(resources, placements);

  if (nodes.size() > kExhaustivePlacementNodeLimit)
    return false;

  if (chooseRouteFeasiblePlacements(nodes, routingProblem, resources, topology,
                                    placements, 0, routeCache))
    return true;
  clearPlacementState(resources, placements);
  return false;
}

} // namespace loom::pnr::detail
