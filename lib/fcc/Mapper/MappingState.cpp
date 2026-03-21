#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/TechMapper.h"
#include "fcc/Mapper/TypeCompat.h"
#include "MapperRoutingInternal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"

#include <algorithm>
#include <cassert>
#include <tuple>

namespace fcc {

namespace {

bool isSpatialPEFunctionalNode(const Node *node) {
  return node && getNodeAttrStr(node, "resource_class") == "functional" &&
         getNodeAttrStr(node, "pe_kind") == "spatial_pe";
}

bool isFunctionalNode(const Node *node) {
  return node && getNodeAttrStr(node, "resource_class") == "functional";
}

bool isDirectMemoryHardwarePort(const Port *hwPort, const Graph &adg) {
  if (!hwPort || hwPort->parentNode == INVALID_ID)
    return false;
  const Node *hwNode = adg.getNode(hwPort->parentNode);
  return hwNode && getNodeAttrStr(hwNode, "resource_class") == "memory";
}

bool directMemoryHardwarePortAllowsSharing(const Port *hwPort,
                                           const Graph &adg) {
  if (!isDirectMemoryHardwarePort(hwPort, adg))
    return false;
  if (mlir::isa<mlir::MemRefType>(hwPort->type))
    return true;
  auto info = detail::getPortTypeInfo(hwPort->type);
  return info && info->isTagged;
}

bool canMapSoftwareTypeToDirectMemoryHardware(mlir::Type swType,
                                              mlir::Type hwType) {
  if (mlir::isa<mlir::MemRefType>(swType) || mlir::isa<mlir::MemRefType>(hwType))
    return canMapSoftwareTypeToHardware(swType, hwType);
  return canMapSoftwareTypeToBridgeHardware(swType, hwType);
}

bool isDirectBindingPath(llvm::ArrayRef<IdIndex> path) {
  return path.size() == 2 && path[0] == path[1];
}

#ifndef NDEBUG
constexpr uint64_t kDebugCacheVerifyInterval = 257;

template <typename T>
llvm::SmallVector<T, 8> sortedCopy(llvm::ArrayRef<T> values) {
  llvm::SmallVector<T, 8> copy(values.begin(), values.end());
  llvm::sort(copy);
  return copy;
}

bool equalStringCountMaps(const llvm::StringMap<unsigned> &lhs,
                          const llvm::StringMap<unsigned> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &entry : lhs) {
    auto it = rhs.find(entry.getKey());
    if (it == rhs.end() || it->second != entry.getValue())
      return false;
  }
  return true;
}
#endif

} // namespace

namespace mapper_detail {
std::vector<double> buildEdgePlacementWeightCache(const Graph &dfg);
}

void MappingState::init(const Graph &dfg, const Graph &adg,
                       const ADGFlattener *flattener) {
  size_t swNodes = dfg.nodes.size();
  size_t swPorts = dfg.ports.size();
  size_t swEdges = dfg.edges.size();
  size_t hwNodes = adg.nodes.size();
  size_t hwPorts = adg.ports.size();
  size_t hwEdges = adg.edges.size();

  swNodeToHwNode.assign(swNodes, INVALID_ID);
  swPortToHwPort.assign(swPorts, INVALID_ID);
  swEdgeToHwPaths.assign(swEdges, {});

  hwNodeToSwNodes.assign(hwNodes, {});
  hwPortToSwPorts.assign(hwPorts, {});
  hwEdgeToSwEdges.assign(hwEdges, {});

  portToUsingEdges.assign(hwPorts, {});
  spatialPEOccupancyCount.clear();
  hwEdgeByPortPair.clear();
  hwEdgeByPortPair.resize(hwPorts);
  swEdgeTaggedObservationKeys.assign(swEdges, {});
  swEdgeTemporalRouteRecords.assign(swEdges, {});
  taggedObservationIndex.clear();
  temporalRouteIndex.clear();
  routeStatsInitialized = false;
  routeStatsFixedInternalMask.clear();
  routeStatsEdgeWeights.clear();
  routeStatsEdgeModes.clear();
  routeStatsPenaltyActive.clear();
  routeStatsUnroutedEdgeSet.clear();
  routeStatsCounters = {};
  routeStatsUnroutedPenalty = 0.0;

  hasTaggedResources = false;
  for (IdIndex portId = 0; portId < static_cast<IdIndex>(adg.ports.size());
       ++portId) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    if (auto typeInfo = detail::getPortTypeInfo(port->type);
        typeInfo && typeInfo->isTagged) {
      hasTaggedResources = true;
      break;
    }
  }
  if (!hasTaggedResources) {
    for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
         ++nodeId) {
      const Node *node = adg.getNode(nodeId);
      if (!node)
        continue;
      llvm::StringRef opKind = getNodeAttrStr(node, "op_kind");
      if (opKind == "add_tag" || opKind == "del_tag" || opKind == "map_tag" ||
          opKind == "temporal_sw") {
        hasTaggedResources = true;
        break;
      }
    }
  }

  hwNodeRows.assign(hwNodes, -1);
  hwNodeCols.assign(hwNodes, -1);
  occupancyGridCols = 0;
  functionalRowOccupancy.clear();
  functionalColOccupancy.clear();
  functionalCellOccupancy.clear();
  if (flattener) {
    int maxRow = -1;
    int maxCol = -1;
    for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
         ++hwId) {
      auto [row, col] = flattener->getNodeGridPos(hwId);
      hwNodeRows[hwId] = row;
      hwNodeCols[hwId] = col;
      if (row >= 0 && col >= 0) {
        maxRow = std::max(maxRow, row);
        maxCol = std::max(maxCol, col);
      }
    }
    if (maxRow >= 0 && maxCol >= 0) {
      occupancyGridCols = static_cast<unsigned>(maxCol + 1);
      functionalRowOccupancy.assign(static_cast<size_t>(maxRow) + 1, 0);
      functionalColOccupancy.assign(static_cast<size_t>(maxCol) + 1, 0);
      functionalCellOccupancy.assign(
          (static_cast<size_t>(maxRow) + 1) *
              (static_cast<size_t>(maxCol) + 1),
          0);
    }
  }
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(adg.edges.size());
       ++edgeId) {
    const Edge *edge = adg.getEdge(edgeId);
    if (!edge || edge->srcPort == INVALID_ID || edge->dstPort == INVALID_ID ||
        edge->srcPort >= hwEdgeByPortPair.size())
      continue;
    hwEdgeByPortPair[edge->srcPort][edge->dstPort] = edgeId;
  }

  undoLog.clear();
  savepointMarkers.clear();
#ifndef NDEBUG
  debugMutationCount = 0;
  debugSavepointCheckpoints.clear();
#endif

  totalCost = 0.0;
}

void MappingState::initializeRouteStats(const Graph &dfg,
                                        llvm::ArrayRef<uint8_t> fixedInternalMask) {
  routeStatsInitialized = true;
  routeStatsFixedInternalMask.assign(fixedInternalMask.begin(),
                                     fixedInternalMask.end());
  if (routeStatsFixedInternalMask.size() < dfg.edges.size())
    routeStatsFixedInternalMask.resize(dfg.edges.size(), 0);
  routeStatsEdgeWeights = mapper_detail::buildEdgePlacementWeightCache(dfg);
  routeStatsEdgeModes.assign(dfg.edges.size(), RouteStatsEdgeMode::Missing);
  routeStatsPenaltyActive.assign(dfg.edges.size(), 0);
  routeStatsUnroutedEdgeSet.clear();
  routeStatsCounters = {};
  routeStatsUnroutedPenalty = 0.0;

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    ++routeStatsCounters.overallEdges;
    RouteStatsEdgeMode mode = computeRouteStatsEdgeMode(edgeId);
    routeStatsEdgeModes[edgeId] = mode;
    switch (mode) {
    case RouteStatsEdgeMode::FixedInternal:
      ++routeStatsCounters.fixedInternalEdges;
      ++routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::DirectBinding:
      ++routeStatsCounters.directBindingEdges;
      ++routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::RoutedRouter:
      ++routeStatsCounters.routerEdges;
      ++routeStatsCounters.routedOverallEdges;
      ++routeStatsCounters.routedRouterEdges;
      break;
    case RouteStatsEdgeMode::UnroutedRouter:
      ++routeStatsCounters.routerEdges;
      ++routeStatsCounters.unroutedRouterEdges;
      routeStatsUnroutedEdgeSet.insert(edgeId);
      if (computeRouteStatsPenaltyActive(edgeId, dfg)) {
        routeStatsPenaltyActive[edgeId] = 1;
        if (edgeId < static_cast<IdIndex>(routeStatsEdgeWeights.size()))
          routeStatsUnroutedPenalty += routeStatsEdgeWeights[edgeId];
      }
      break;
    case RouteStatsEdgeMode::Missing:
      break;
    }
  }
}

MappingState::Savepoint MappingState::beginSavepoint() {
  savepointMarkers.push_back(undoLog.size());
#ifndef NDEBUG
  debugSavepointCheckpoints.push_back(save());
#endif
  return Savepoint{savepointMarkers.back()};
}

void MappingState::rollbackSavepoint(Savepoint savepoint) {
  if (savepointMarkers.empty() || savepointMarkers.back() != savepoint.marker)
    return;
  while (undoLog.size() > savepoint.marker) {
    applyUndoRecord(undoLog.back());
    undoLog.pop_back();
  }
  savepointMarkers.pop_back();
#ifndef NDEBUG
  assert(!debugSavepointCheckpoints.empty());
  debugVerifyCheckpointEqualsCurrent(debugSavepointCheckpoints.back());
  debugSavepointCheckpoints.pop_back();
#endif
}

void MappingState::commitSavepoint(Savepoint savepoint) {
  if (savepointMarkers.empty() || savepointMarkers.back() != savepoint.marker)
    return;
  savepointMarkers.pop_back();
#ifndef NDEBUG
  assert(!debugSavepointCheckpoints.empty());
  debugSavepointCheckpoints.pop_back();
#endif
  if (savepointMarkers.empty())
    undoLog.clear();
}

bool MappingState::hasActiveSavepoint() const {
  return !savepointMarkers.empty();
}

void MappingState::recordSwNodeToHwNode(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SwNodeToHwNode;
  record.index = index;
  record.idValue = swNodeToHwNode[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordSwPortToHwPort(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SwPortToHwPort;
  record.index = index;
  record.idValue = swPortToHwPort[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordSwEdgeToHwPath(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SwEdgeToHwPath;
  record.index = index;
  record.pathValue = swEdgeToHwPaths[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordHwNodeToSwNodes(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::HwNodeToSwNodes;
  record.index = index;
  record.nodeVecValue = hwNodeToSwNodes[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordHwPortToSwPorts(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::HwPortToSwPorts;
  record.index = index;
  record.nodeVecValue = hwPortToSwPorts[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordHwEdgeToSwEdges(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::HwEdgeToSwEdges;
  record.index = index;
  record.edgeVecValue = hwEdgeToSwEdges[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordPortToUsingEdges(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::PortToUsingEdges;
  record.index = index;
  record.edgeVecValue = portToUsingEdges[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordSpatialPEOccupancy(llvm::StringRef peName) {
  if (!hasActiveSavepoint() || peName.empty())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SpatialPEOccupancy;
  record.stringKey = peName.str();
  auto it = spatialPEOccupancyCount.find(peName);
  record.countExisted = it != spatialPEOccupancyCount.end();
  record.countValue = record.countExisted ? it->second : 0;
  undoLog.push_back(std::move(record));
}

void MappingState::recordSwEdgeTaggedObservationKeys(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SwEdgeTaggedObservationKeys;
  record.index = index;
  record.taggedObservationVecValue = swEdgeTaggedObservationKeys[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordSwEdgeTemporalRouteRecords(size_t index) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::SwEdgeTemporalRouteRecords;
  record.index = index;
  record.temporalRouteRecordVecValue = swEdgeTemporalRouteRecords[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordTaggedObservationIndex(
    const TaggedObservationKey &key) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::TaggedObservationIndex;
  record.taggedObservationKey = key;
  if (auto it = taggedObservationIndex.find(key); it != taggedObservationIndex.end())
    record.edgeVecValue = it->second;
  undoLog.push_back(std::move(record));
}

void MappingState::recordTemporalRouteIndex(const TemporalRouteGroupKey &key) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::TemporalRouteIndex;
  record.temporalRouteGroupKey = key;
  if (auto it = temporalRouteIndex.find(key); it != temporalRouteIndex.end())
    record.temporalRouteUseVecValue = it->second;
  undoLog.push_back(std::move(record));
}

void MappingState::recordFunctionalRowOccupancy(size_t index) {
  if (!hasActiveSavepoint() || index >= functionalRowOccupancy.size())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::FunctionalRowOccupancy;
  record.index = index;
  record.countValue = functionalRowOccupancy[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordFunctionalColOccupancy(size_t index) {
  if (!hasActiveSavepoint() || index >= functionalColOccupancy.size())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::FunctionalColOccupancy;
  record.index = index;
  record.countValue = functionalColOccupancy[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordFunctionalCellOccupancy(size_t index) {
  if (!hasActiveSavepoint() || index >= functionalCellOccupancy.size())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::FunctionalCellOccupancy;
  record.index = index;
  record.countValue = functionalCellOccupancy[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordRouteStatsEdgeMode(size_t index) {
  if (!hasActiveSavepoint() || index >= routeStatsEdgeModes.size())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::RouteStatsEdgeMode;
  record.index = index;
  record.routeStatsEdgeModeValue = routeStatsEdgeModes[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordRouteStatsPenaltyActive(size_t index) {
  if (!hasActiveSavepoint() || index >= routeStatsPenaltyActive.size())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::RouteStatsPenaltyActive;
  record.index = index;
  record.countValue = routeStatsPenaltyActive[index];
  undoLog.push_back(std::move(record));
}

void MappingState::recordRouteStatsUnroutedEdgeMembership(IdIndex edgeId) {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::RouteStatsUnroutedEdgeMembership;
  record.idValue = edgeId;
  record.countExisted = routeStatsUnroutedEdgeSet.count(edgeId) > 0;
  undoLog.push_back(std::move(record));
}

void MappingState::recordRouteStatsCounters() {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::RouteStatsCounters;
  record.routeStatsCountersValue = routeStatsCounters;
  undoLog.push_back(std::move(record));
}

void MappingState::recordRouteStatsUnroutedPenalty() {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::RouteStatsUnroutedPenalty;
  record.doubleValue = routeStatsUnroutedPenalty;
  undoLog.push_back(std::move(record));
}

void MappingState::recordTotalCost() {
  if (!hasActiveSavepoint())
    return;
  UndoRecord record;
  record.kind = UndoRecord::Kind::TotalCost;
  record.doubleValue = totalCost;
  undoLog.push_back(std::move(record));
}

void MappingState::applyUndoRecord(const UndoRecord &record) {
  switch (record.kind) {
  case UndoRecord::Kind::SwNodeToHwNode:
    swNodeToHwNode[record.index] = record.idValue;
    break;
  case UndoRecord::Kind::SwPortToHwPort:
    swPortToHwPort[record.index] = record.idValue;
    break;
  case UndoRecord::Kind::SwEdgeToHwPath:
    swEdgeToHwPaths[record.index] = record.pathValue;
    break;
  case UndoRecord::Kind::HwNodeToSwNodes:
    hwNodeToSwNodes[record.index] = record.nodeVecValue;
    break;
  case UndoRecord::Kind::HwPortToSwPorts:
    hwPortToSwPorts[record.index] = record.nodeVecValue;
    break;
  case UndoRecord::Kind::HwEdgeToSwEdges:
    hwEdgeToSwEdges[record.index] = record.edgeVecValue;
    break;
  case UndoRecord::Kind::PortToUsingEdges:
    portToUsingEdges[record.index] = record.edgeVecValue;
    break;
  case UndoRecord::Kind::SpatialPEOccupancy:
    if (record.countExisted)
      spatialPEOccupancyCount[record.stringKey] = record.countValue;
    else
      spatialPEOccupancyCount.erase(record.stringKey);
    break;
  case UndoRecord::Kind::SwEdgeTaggedObservationKeys:
    swEdgeTaggedObservationKeys[record.index] = record.taggedObservationVecValue;
    break;
  case UndoRecord::Kind::SwEdgeTemporalRouteRecords:
    swEdgeTemporalRouteRecords[record.index] =
        record.temporalRouteRecordVecValue;
    break;
  case UndoRecord::Kind::TaggedObservationIndex:
    if (record.edgeVecValue.empty())
      taggedObservationIndex.erase(record.taggedObservationKey);
    else
      taggedObservationIndex[record.taggedObservationKey] = record.edgeVecValue;
    break;
  case UndoRecord::Kind::TemporalRouteIndex:
    if (record.temporalRouteUseVecValue.empty())
      temporalRouteIndex.erase(record.temporalRouteGroupKey);
    else
      temporalRouteIndex[record.temporalRouteGroupKey] =
          record.temporalRouteUseVecValue;
    break;
  case UndoRecord::Kind::FunctionalRowOccupancy:
    functionalRowOccupancy[record.index] = record.countValue;
    break;
  case UndoRecord::Kind::FunctionalColOccupancy:
    functionalColOccupancy[record.index] = record.countValue;
    break;
  case UndoRecord::Kind::FunctionalCellOccupancy:
    functionalCellOccupancy[record.index] = record.countValue;
    break;
  case UndoRecord::Kind::RouteStatsEdgeMode:
    routeStatsEdgeModes[record.index] = record.routeStatsEdgeModeValue;
    break;
  case UndoRecord::Kind::RouteStatsPenaltyActive:
    routeStatsPenaltyActive[record.index] =
        static_cast<uint8_t>(record.countValue);
    break;
  case UndoRecord::Kind::RouteStatsUnroutedEdgeMembership:
    if (record.countExisted)
      routeStatsUnroutedEdgeSet.insert(record.idValue);
    else
      routeStatsUnroutedEdgeSet.erase(record.idValue);
    break;
  case UndoRecord::Kind::RouteStatsCounters:
    routeStatsCounters = record.routeStatsCountersValue;
    break;
  case UndoRecord::Kind::RouteStatsUnroutedPenalty:
    routeStatsUnroutedPenalty = record.doubleValue;
    break;
  case UndoRecord::Kind::TotalCost:
    totalCost = record.doubleValue;
    break;
  }
}

bool MappingState::isSpatialPEOccupied(llvm::StringRef peName, const Graph &adg,
                                       IdIndex ignoreHwNode) const {
  (void)adg;
  (void)ignoreHwNode;
  if (peName.empty())
    return false;
  auto it = spatialPEOccupancyCount.find(peName);
  return it != spatialPEOccupancyCount.end() && it->second > 0;
}

IdIndex MappingState::lookupHwEdge(IdIndex outPort, IdIndex inPort) const {
  if (outPort >= hwEdgeByPortPair.size())
    return INVALID_ID;
  auto it = hwEdgeByPortPair[outPort].find(inPort);
  if (it == hwEdgeByPortPair[outPort].end())
    return INVALID_ID;
  return it->second;
}

unsigned MappingState::getFunctionalRowOccupancy(int row) const {
  if (row < 0 || row >= static_cast<int>(functionalRowOccupancy.size()))
    return 0;
  return functionalRowOccupancy[static_cast<size_t>(row)];
}

unsigned MappingState::getFunctionalColOccupancy(int col) const {
  if (col < 0 || col >= static_cast<int>(functionalColOccupancy.size()))
    return 0;
  return functionalColOccupancy[static_cast<size_t>(col)];
}

size_t MappingState::computeCellIndex(int row, int col) const {
  if (row < 0 || col < 0 || occupancyGridCols == 0)
    return static_cast<size_t>(-1);
  return static_cast<size_t>(row) * occupancyGridCols +
         static_cast<size_t>(col);
}

unsigned MappingState::getFunctionalCellOccupancy(int row, int col) const {
  size_t index = computeCellIndex(row, col);
  if (index == static_cast<size_t>(-1) || index >= functionalCellOccupancy.size())
    return 0;
  return functionalCellOccupancy[index];
}

void MappingState::applyFunctionalNodeOccupancyDelta(IdIndex hwNode, int delta,
                                                     const Graph &adg) {
  if (delta == 0 || hwNode >= adg.nodes.size())
    return;
  const Node *hwN = adg.getNode(hwNode);
  if (!isFunctionalNode(hwN))
    return;
  if (hwNode >= hwNodeRows.size() || hwNode >= hwNodeCols.size())
    return;
  int row = hwNodeRows[hwNode];
  int col = hwNodeCols[hwNode];
  if (row < 0 || col < 0)
    return;

  if (row < static_cast<int>(functionalRowOccupancy.size())) {
    recordFunctionalRowOccupancy(static_cast<size_t>(row));
    functionalRowOccupancy[static_cast<size_t>(row)] =
        static_cast<unsigned>(static_cast<int>(
                                  functionalRowOccupancy[static_cast<size_t>(row)]) +
                              delta);
  }
  if (col < static_cast<int>(functionalColOccupancy.size())) {
    recordFunctionalColOccupancy(static_cast<size_t>(col));
    functionalColOccupancy[static_cast<size_t>(col)] =
        static_cast<unsigned>(static_cast<int>(
                                  functionalColOccupancy[static_cast<size_t>(col)]) +
                              delta);
  }
  size_t cellIndex = computeCellIndex(row, col);
  if (cellIndex != static_cast<size_t>(-1) &&
      cellIndex < functionalCellOccupancy.size()) {
    recordFunctionalCellOccupancy(cellIndex);
    functionalCellOccupancy[cellIndex] = static_cast<unsigned>(
        static_cast<int>(functionalCellOccupancy[cellIndex]) + delta);
  }
}

MappingState::RouteStatsEdgeMode
MappingState::computeRouteStatsEdgeMode(IdIndex swEdge) const {
  if (swEdge >= swEdgeToHwPaths.size() ||
      swEdge >= routeStatsFixedInternalMask.size())
    return RouteStatsEdgeMode::Missing;
  if (routeStatsFixedInternalMask[swEdge])
    return RouteStatsEdgeMode::FixedInternal;
  const auto &path = swEdgeToHwPaths[swEdge];
  if (path.empty())
    return RouteStatsEdgeMode::UnroutedRouter;
  if (isDirectBindingPath(path))
    return RouteStatsEdgeMode::DirectBinding;
  return RouteStatsEdgeMode::RoutedRouter;
}

bool MappingState::computeRouteStatsPenaltyActive(IdIndex swEdge,
                                                  const Graph &dfg) const {
  if (computeRouteStatsEdgeMode(swEdge) != RouteStatsEdgeMode::UnroutedRouter)
    return false;
  const Edge *edge = dfg.getEdge(swEdge);
  if (!edge)
    return false;
  const Port *srcPort = dfg.getPort(edge->srcPort);
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
      dstPort->parentNode == INVALID_ID)
    return false;
  bool srcMapped = srcPort->parentNode < swNodeToHwNode.size() &&
                   swNodeToHwNode[srcPort->parentNode] != INVALID_ID;
  bool dstMapped = dstPort->parentNode < swNodeToHwNode.size() &&
                   swNodeToHwNode[dstPort->parentNode] != INVALID_ID;
  return srcMapped && dstMapped;
}

void MappingState::refreshRouteStatsForEdge(IdIndex swEdge, const Graph &dfg) {
  if (!routeStatsInitialized || swEdge >= routeStatsEdgeModes.size())
    return;

  RouteStatsEdgeMode oldMode = routeStatsEdgeModes[swEdge];
  bool oldPenaltyActive =
      swEdge < routeStatsPenaltyActive.size() && routeStatsPenaltyActive[swEdge];
  RouteStatsEdgeMode newMode = computeRouteStatsEdgeMode(swEdge);
  bool newPenaltyActive = computeRouteStatsPenaltyActive(swEdge, dfg);
  if (oldMode == newMode && oldPenaltyActive == newPenaltyActive)
    return;

  recordRouteStatsCounters();
  recordRouteStatsEdgeMode(swEdge);
  recordRouteStatsPenaltyActive(swEdge);
  recordRouteStatsUnroutedEdgeMembership(swEdge);
  recordRouteStatsUnroutedPenalty();

  auto removeMode = [&](RouteStatsEdgeMode mode) {
    switch (mode) {
    case RouteStatsEdgeMode::FixedInternal:
      --routeStatsCounters.fixedInternalEdges;
      --routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::DirectBinding:
      --routeStatsCounters.directBindingEdges;
      --routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::RoutedRouter:
      --routeStatsCounters.routerEdges;
      --routeStatsCounters.routedOverallEdges;
      --routeStatsCounters.routedRouterEdges;
      break;
    case RouteStatsEdgeMode::UnroutedRouter:
      --routeStatsCounters.routerEdges;
      --routeStatsCounters.unroutedRouterEdges;
      routeStatsUnroutedEdgeSet.erase(swEdge);
      break;
    case RouteStatsEdgeMode::Missing:
      break;
    }
  };
  auto addMode = [&](RouteStatsEdgeMode mode) {
    switch (mode) {
    case RouteStatsEdgeMode::FixedInternal:
      ++routeStatsCounters.fixedInternalEdges;
      ++routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::DirectBinding:
      ++routeStatsCounters.directBindingEdges;
      ++routeStatsCounters.routedOverallEdges;
      break;
    case RouteStatsEdgeMode::RoutedRouter:
      ++routeStatsCounters.routerEdges;
      ++routeStatsCounters.routedOverallEdges;
      ++routeStatsCounters.routedRouterEdges;
      break;
    case RouteStatsEdgeMode::UnroutedRouter:
      ++routeStatsCounters.routerEdges;
      ++routeStatsCounters.unroutedRouterEdges;
      routeStatsUnroutedEdgeSet.insert(swEdge);
      break;
    case RouteStatsEdgeMode::Missing:
      break;
    }
  };

  removeMode(oldMode);
  if (oldPenaltyActive && swEdge < static_cast<IdIndex>(routeStatsEdgeWeights.size()))
    routeStatsUnroutedPenalty -= routeStatsEdgeWeights[swEdge];

  routeStatsEdgeModes[swEdge] = newMode;
  routeStatsPenaltyActive[swEdge] = newPenaltyActive ? 1u : 0u;

  addMode(newMode);
  if (newPenaltyActive && swEdge < static_cast<IdIndex>(routeStatsEdgeWeights.size()))
    routeStatsUnroutedPenalty += routeStatsEdgeWeights[swEdge];
}

void MappingState::refreshRouteStatsForIncidentEdges(IdIndex swNode,
                                                     const Graph &dfg) {
  if (!routeStatsInitialized)
    return;
  const Node *swN = dfg.getNode(swNode);
  if (!swN)
    return;
  llvm::DenseSet<IdIndex> seenIncidentEdges;
  auto refreshPorts = [&](llvm::ArrayRef<IdIndex> ports) {
    for (IdIndex pid : ports) {
      const Port *port = dfg.getPort(pid);
      if (!port)
        continue;
      for (IdIndex edgeId : port->connectedEdges) {
        if (seenIncidentEdges.insert(edgeId).second)
          refreshRouteStatsForEdge(edgeId, dfg);
      }
    }
  };
  refreshPorts(swN->inputPorts);
  refreshPorts(swN->outputPorts);
}

void MappingState::clearTaggedObservationsForEdge(IdIndex swEdge) {
  if (!hasTaggedResources || swEdge >= swEdgeTaggedObservationKeys.size() ||
      swEdge >= swEdgeTemporalRouteRecords.size())
    return;

  if (!swEdgeTaggedObservationKeys[swEdge].empty()) {
    recordSwEdgeTaggedObservationKeys(swEdge);
    for (const TaggedObservationKey &key : swEdgeTaggedObservationKeys[swEdge]) {
      auto it = taggedObservationIndex.find(key);
      if (it == taggedObservationIndex.end())
        continue;
      recordTaggedObservationIndex(key);
      auto &vec = it->second;
      vec.erase(std::remove(vec.begin(), vec.end(), swEdge), vec.end());
      if (vec.empty())
        taggedObservationIndex.erase(it);
    }
    swEdgeTaggedObservationKeys[swEdge].clear();
  }

  if (!swEdgeTemporalRouteRecords[swEdge].empty()) {
    recordSwEdgeTemporalRouteRecords(swEdge);
    for (const TemporalRouteRecord &record : swEdgeTemporalRouteRecords[swEdge]) {
      auto it = temporalRouteIndex.find(record.key);
      if (it == temporalRouteIndex.end())
        continue;
      recordTemporalRouteIndex(record.key);
      auto &vec = it->second;
      vec.erase(std::remove_if(vec.begin(), vec.end(),
                               [&](const TemporalRouteUse &use) {
                                 return use.edgeId == swEdge &&
                                        use.inPortId == record.inPortId &&
                                        use.outPortId == record.outPortId;
                               }),
                vec.end());
      if (vec.empty())
        temporalRouteIndex.erase(it);
    }
    swEdgeTemporalRouteRecords[swEdge].clear();
  }
}

void MappingState::refreshTaggedObservationsForEdge(
    IdIndex swEdge, llvm::ArrayRef<IdIndex> path, const Graph &dfg,
    const Graph &adg) {
  if (!hasTaggedResources || swEdge >= swEdgeTaggedObservationKeys.size() ||
      swEdge >= swEdgeTemporalRouteRecords.size())
    return;

  clearTaggedObservationsForEdge(swEdge);
  if (path.empty())
    return;

  auto fullPath = routing_detail::buildExportPathForEdge(swEdge, path, *this, dfg,
                                                         adg);
  llvm::SmallVector<routing_detail::TaggedPathObservation, 8> taggedObs;
  llvm::SmallVector<routing_detail::TemporalSwitchTagRouteObservation, 4>
      temporalObs;
  routing_detail::appendTaggedPathObservations(swEdge, fullPath, *this, dfg,
                                               adg, taggedObs);
  routing_detail::appendTemporalSwitchTagRouteObservations(swEdge, fullPath,
                                                           *this, dfg, adg,
                                                           temporalObs);

  if (!taggedObs.empty()) {
    recordSwEdgeTaggedObservationKeys(swEdge);
    llvm::DenseSet<TaggedObservationKey> seenTagged;
    auto &records = swEdgeTaggedObservationKeys[swEdge];
    for (const auto &obs : taggedObs) {
      TaggedObservationKey key{obs.kind, obs.first, obs.second, obs.tag};
      if (!seenTagged.insert(key).second)
        continue;
      records.push_back(key);
      recordTaggedObservationIndex(key);
      taggedObservationIndex[key].push_back(swEdge);
    }
  }

  if (!temporalObs.empty()) {
    recordSwEdgeTemporalRouteRecords(swEdge);
    llvm::DenseSet<uint64_t> seenTemporal;
    auto &records = swEdgeTemporalRouteRecords[swEdge];
    for (const auto &obs : temporalObs) {
      uint64_t dedupKey = llvm::hash_combine(obs.nodeId, obs.inPortId,
                                             obs.outPortId, obs.tag);
      if (!seenTemporal.insert(dedupKey).second)
        continue;
      TemporalRouteRecord record{{obs.nodeId, obs.tag}, obs.inPortId,
                                 obs.outPortId};
      records.push_back(record);
      recordTemporalRouteIndex(record.key);
      temporalRouteIndex[record.key].push_back(
          TemporalRouteUse{swEdge, obs.inPortId, obs.outPortId});
    }
  }
}

ActionResult MappingState::mapNode(IdIndex swNode, IdIndex hwNode,
                                    const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size() || hwNode >= hwNodeToSwNodes.size())
    return ActionResult::FailedInternalError;

  // Most hardware nodes are exclusive. Memory nodes may host multiple
  // software memories, but mapNode must still enforce numRegion capacity so
  // every placement path, including local repair and CP-SAT, stays legal.
  const Node *hwN = adg.getNode(hwNode);
  bool allowMultiOccupancy =
      hwN && getNodeAttrStr(hwN, "resource_class") == "memory";
  if (allowMultiOccupancy) {
    int64_t numRegion = std::max<int64_t>(1, getNodeAttrInt(hwN, "numRegion", 1));
    if (static_cast<int64_t>(hwNodeToSwNodes[hwNode].size()) >= numRegion)
      return ActionResult::FailedResourceUnavailable;
  } else if (!hwNodeToSwNodes[hwNode].empty()) {
    return ActionResult::FailedResourceUnavailable;
  }

  if (isSpatialPEFunctionalNode(hwN) &&
      isSpatialPEOccupied(getNodeAttrStr(hwN, "pe_name"), adg, hwNode)) {
    return ActionResult::FailedHardConstraint;
  }

  recordSwNodeToHwNode(swNode);
  swNodeToHwNode[swNode] = hwNode;
  recordHwNodeToSwNodes(hwNode);
  hwNodeToSwNodes[hwNode].push_back(swNode);

  // Track spatial PE occupancy.
  if (isSpatialPEFunctionalNode(hwN)) {
    llvm::StringRef peName = getNodeAttrStr(hwN, "pe_name");
    if (!peName.empty()) {
      recordSpatialPEOccupancy(peName);
      ++spatialPEOccupancyCount[peName];
    }
  }
  applyFunctionalNodeOccupancyDelta(hwNode, +1, adg);
  refreshRouteStatsForIncidentEdges(swNode, dfg);

  // Auto-map ports: match by position.
  const Node *swN = dfg.getNode(swNode);
  bool skipAutoPorts = false;
  if (hwN && getNodeAttrStr(hwN, "resource_class") == "memory")
    skipAutoPorts = true;

  if (swN && hwN && !skipAutoPorts) {
    if (swN->inputPorts.size() > hwN->inputPorts.size() ||
        swN->outputPorts.size() > hwN->outputPorts.size()) {
      unmapNode(swNode, dfg, adg);
      return ActionResult::FailedTypeMismatch;
    }
    for (unsigned i = 0;
         i < swN->inputPorts.size() && i < hwN->inputPorts.size(); ++i) {
      ActionResult result =
          mapPort(swN->inputPorts[i], hwN->inputPorts[i], dfg, adg);
      if (result != ActionResult::Success) {
        unmapNode(swNode, dfg, adg);
        return result;
      }
    }
    for (unsigned i = 0;
         i < swN->outputPorts.size() && i < hwN->outputPorts.size(); ++i) {
      ActionResult result =
          mapPort(swN->outputPorts[i], hwN->outputPorts[i], dfg, adg);
      if (result != ActionResult::Success) {
        unmapNode(swNode, dfg, adg);
        return result;
      }
    }
  }

#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

ActionResult MappingState::unmapNode(IdIndex swNode,
                                      const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size())
    return ActionResult::FailedInternalError;

  IdIndex hwNode = swNodeToHwNode[swNode];
  if (hwNode == INVALID_ID)
    return ActionResult::Success;

  // Remove reverse mapping.
  if (hwNode < hwNodeToSwNodes.size()) {
    recordHwNodeToSwNodes(hwNode);
    auto &vec = hwNodeToSwNodes[hwNode];
    vec.erase(std::remove(vec.begin(), vec.end(), swNode), vec.end());
  }

  // Any routed software edge incident to this node becomes invalid once the
  // node or its ports move, including synthetic direct bindings for extmemory.
  const Node *swN = dfg.getNode(swNode);
  if (swN) {
    llvm::SmallVector<IdIndex, 16> incidentEdges;
    llvm::DenseSet<IdIndex> seenIncidentEdges;
    auto collectIncidentEdges = [&](llvm::ArrayRef<IdIndex> ports) {
      for (IdIndex pid : ports) {
        const Port *port = dfg.getPort(pid);
        if (!port)
          continue;
        for (IdIndex edgeId : port->connectedEdges) {
          if (seenIncidentEdges.insert(edgeId).second) {
            incidentEdges.push_back(edgeId);
          }
        }
      }
    };
    collectIncidentEdges(swN->inputPorts);
    collectIncidentEdges(swN->outputPorts);
    for (IdIndex edgeId : incidentEdges)
      unmapEdge(edgeId, dfg, adg);
  }

  // Unmap ports.
  if (swN) {
    for (IdIndex pid : swN->inputPorts) {
      if (pid < swPortToHwPort.size()) {
        IdIndex hwPort = swPortToHwPort[pid];
        if (hwPort != INVALID_ID && hwPort < hwPortToSwPorts.size()) {
          recordHwPortToSwPorts(hwPort);
          auto &vec = hwPortToSwPorts[hwPort];
          vec.erase(std::remove(vec.begin(), vec.end(), pid), vec.end());
        }
        recordSwPortToHwPort(pid);
        swPortToHwPort[pid] = INVALID_ID;
      }
    }
    for (IdIndex pid : swN->outputPorts) {
      if (pid < swPortToHwPort.size()) {
        IdIndex hwPort = swPortToHwPort[pid];
        if (hwPort != INVALID_ID && hwPort < hwPortToSwPorts.size()) {
          recordHwPortToSwPorts(hwPort);
          auto &vec = hwPortToSwPorts[hwPort];
          vec.erase(std::remove(vec.begin(), vec.end(), pid), vec.end());
        }
        recordSwPortToHwPort(pid);
        swPortToHwPort[pid] = INVALID_ID;
      }
    }
  }

  // Update spatial PE occupancy tracking.
  const Node *hwN = adg.getNode(hwNode);
  if (isSpatialPEFunctionalNode(hwN)) {
    llvm::StringRef peName = getNodeAttrStr(hwN, "pe_name");
    if (!peName.empty()) {
      recordSpatialPEOccupancy(peName);
      auto it = spatialPEOccupancyCount.find(peName);
      if (it != spatialPEOccupancyCount.end()) {
        if (it->second <= 1)
          spatialPEOccupancyCount.erase(it);
        else
          --it->second;
      }
    }
  }
  applyFunctionalNodeOccupancyDelta(hwNode, -1, adg);

  recordSwNodeToHwNode(swNode);
  swNodeToHwNode[swNode] = INVALID_ID;
  refreshRouteStatsForIncidentEdges(swNode, dfg);
#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

ActionResult MappingState::mapPort(IdIndex swPort, IdIndex hwPort,
                                    const Graph &dfg, const Graph &adg) {
  if (swPort >= swPortToHwPort.size() || hwPort >= hwPortToSwPorts.size())
    return ActionResult::FailedInternalError;

  if (swPortToHwPort[swPort] == hwPort) {
#ifndef NDEBUG
    debugMaybeVerifyCachedState(dfg, adg);
#endif
    return ActionResult::Success;
  }

  if (swPortToHwPort[swPort] != INVALID_ID)
    return ActionResult::FailedResourceUnavailable;
  const Port *swP = dfg.getPort(swPort);
  const Port *hwP = adg.getPort(hwPort);
  if (!swP || !hwP)
    return ActionResult::FailedInternalError;

  bool allowShared = directMemoryHardwarePortAllowsSharing(hwP, adg);
  if (!hwPortToSwPorts[hwPort].empty() && !allowShared)
    return ActionResult::FailedResourceUnavailable;

  bool typeCompatible =
      allowShared ? canMapSoftwareTypeToDirectMemoryHardware(swP->type, hwP->type)
                  : canMapSoftwareTypeToHardware(swP->type, hwP->type);
  if (!typeCompatible)
    return ActionResult::FailedTypeMismatch;

  recordSwPortToHwPort(swPort);
  swPortToHwPort[swPort] = hwPort;
  recordHwPortToSwPorts(hwPort);
  hwPortToSwPorts[hwPort].push_back(swPort);
#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

ActionResult MappingState::mapPortBridgeAware(IdIndex swPort, IdIndex hwPort,
                                              const Graph &dfg,
                                              const Graph &adg) {
  if (swPort >= swPortToHwPort.size() || hwPort >= hwPortToSwPorts.size())
    return ActionResult::FailedInternalError;

  if (swPortToHwPort[swPort] == hwPort) {
#ifndef NDEBUG
    debugMaybeVerifyCachedState(dfg, adg);
#endif
    return ActionResult::Success;
  }

  if (swPortToHwPort[swPort] != INVALID_ID)
    return ActionResult::FailedResourceUnavailable;
  const Port *swP = dfg.getPort(swPort);
  const Port *hwP = adg.getPort(hwPort);
  if (!swP || !hwP)
    return ActionResult::FailedInternalError;

  if (!hwPortToSwPorts[hwPort].empty())
    return ActionResult::FailedResourceUnavailable;
  if (!canMapSoftwareTypeToBridgeHardware(swP->type, hwP->type))
    return ActionResult::FailedTypeMismatch;

  recordSwPortToHwPort(swPort);
  swPortToHwPort[swPort] = hwPort;
  recordHwPortToSwPorts(hwPort);
  hwPortToSwPorts[hwPort].push_back(swPort);
#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

ActionResult MappingState::mapEdge(IdIndex swEdge,
                                    llvm::ArrayRef<IdIndex> path,
                                    const Graph &dfg, const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  if (!swEdgeToHwPaths[swEdge].empty())
    unmapEdge(swEdge, dfg, adg);

  recordSwEdgeToHwPath(swEdge);
  swEdgeToHwPaths[swEdge].assign(path.begin(), path.end());

  // Track HW edge usage.
  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPort = path[i];
    IdIndex inPort = path[i + 1];
    IdIndex edgeId = lookupHwEdge(outPort, inPort);
    if (edgeId == INVALID_ID || edgeId >= hwEdgeToSwEdges.size())
      continue;
    recordHwEdgeToSwEdges(edgeId);
    hwEdgeToSwEdges[edgeId].push_back(swEdge);
  }

  // Maintain portToUsingEdges index (skip synthetic direct bindings).
  if (!(path.size() == 2 && path[0] == path[1])) {
    for (IdIndex portId : path) {
      if (portId < portToUsingEdges.size()) {
        recordPortToUsingEdges(portId);
        portToUsingEdges[portId].push_back(swEdge);
      }
    }
  }

  refreshTaggedObservationsForEdge(swEdge, path, dfg, adg);
  refreshRouteStatsForEdge(swEdge, dfg);

#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

ActionResult MappingState::unmapEdge(IdIndex swEdge, const Graph &dfg,
                                     const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  auto &path = swEdgeToHwPaths[swEdge];
  if (path.empty())
    return ActionResult::Success;

  clearTaggedObservationsForEdge(swEdge);

  // Remove from portToUsingEdges index (skip synthetic direct bindings).
  if (!(path.size() == 2 && path[0] == path[1])) {
    for (IdIndex portId : path) {
      if (portId < portToUsingEdges.size()) {
        recordPortToUsingEdges(portId);
        auto &vec = portToUsingEdges[portId];
        vec.erase(std::remove(vec.begin(), vec.end(), swEdge), vec.end());
      }
    }
  }

  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPort = path[i];
    IdIndex inPort = path[i + 1];
    IdIndex edgeId = lookupHwEdge(outPort, inPort);
    if (edgeId == INVALID_ID || edgeId >= hwEdgeToSwEdges.size())
      continue;
    recordHwEdgeToSwEdges(edgeId);
    auto &mappedEdges = hwEdgeToSwEdges[edgeId];
    mappedEdges.erase(
        std::remove(mappedEdges.begin(), mappedEdges.end(), swEdge),
        mappedEdges.end());
  }

  recordSwEdgeToHwPath(swEdge);
  path.clear();
  refreshRouteStatsForEdge(swEdge, dfg);
#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
  return ActionResult::Success;
}

void MappingState::clearRoutes(const Graph &dfg, const Graph &adg,
                              bool preserveDirectBindings) {
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(swEdgeToHwPaths.size());
       ++edgeId) {
    const auto &path = swEdgeToHwPaths[edgeId];
    if (path.empty())
      continue;
    if (preserveDirectBindings && path.size() == 2 && path[0] == path[1])
      continue;
    unmapEdge(edgeId, dfg, adg);
  }
#ifndef NDEBUG
  debugMaybeVerifyCachedState(dfg, adg);
#endif
}

MappingState::Checkpoint MappingState::save() const {
  Checkpoint cp;
  cp.swNodeToHwNode = swNodeToHwNode;
  cp.swPortToHwPort = swPortToHwPort;
  cp.swEdgeToHwPaths = swEdgeToHwPaths;
  cp.hwNodeToSwNodes = hwNodeToSwNodes;
  cp.hwPortToSwPorts = hwPortToSwPorts;
  cp.hwEdgeToSwEdges = hwEdgeToSwEdges;
  cp.portToUsingEdges = portToUsingEdges;
  cp.spatialPEOccupancyCount = spatialPEOccupancyCount;
  cp.swEdgeTaggedObservationKeys = swEdgeTaggedObservationKeys;
  cp.swEdgeTemporalRouteRecords = swEdgeTemporalRouteRecords;
  cp.taggedObservationIndex = taggedObservationIndex;
  cp.temporalRouteIndex = temporalRouteIndex;
  cp.functionalRowOccupancy = functionalRowOccupancy;
  cp.functionalColOccupancy = functionalColOccupancy;
  cp.functionalCellOccupancy = functionalCellOccupancy;
  cp.routeStatsInitialized = routeStatsInitialized;
  cp.routeStatsFixedInternalMask = routeStatsFixedInternalMask;
  cp.routeStatsEdgeWeights = routeStatsEdgeWeights;
  cp.routeStatsEdgeModes = routeStatsEdgeModes;
  cp.routeStatsPenaltyActive = routeStatsPenaltyActive;
  cp.routeStatsUnroutedEdgeSet = routeStatsUnroutedEdgeSet;
  cp.routeStatsCounters = routeStatsCounters;
  cp.routeStatsUnroutedPenalty = routeStatsUnroutedPenalty;
  cp.totalCost = totalCost;
  return cp;
}

void MappingState::restore(const Checkpoint &cp) {
  swNodeToHwNode = cp.swNodeToHwNode;
  swPortToHwPort = cp.swPortToHwPort;
  swEdgeToHwPaths = cp.swEdgeToHwPaths;
  hwNodeToSwNodes = cp.hwNodeToSwNodes;
  hwPortToSwPorts = cp.hwPortToSwPorts;
  hwEdgeToSwEdges = cp.hwEdgeToSwEdges;
  portToUsingEdges = cp.portToUsingEdges;
  spatialPEOccupancyCount = cp.spatialPEOccupancyCount;
  swEdgeTaggedObservationKeys = cp.swEdgeTaggedObservationKeys;
  swEdgeTemporalRouteRecords = cp.swEdgeTemporalRouteRecords;
  taggedObservationIndex = cp.taggedObservationIndex;
  temporalRouteIndex = cp.temporalRouteIndex;
  functionalRowOccupancy = cp.functionalRowOccupancy;
  functionalColOccupancy = cp.functionalColOccupancy;
  functionalCellOccupancy = cp.functionalCellOccupancy;
  routeStatsInitialized = cp.routeStatsInitialized;
  routeStatsFixedInternalMask = cp.routeStatsFixedInternalMask;
  routeStatsEdgeWeights = cp.routeStatsEdgeWeights;
  routeStatsEdgeModes = cp.routeStatsEdgeModes;
  routeStatsPenaltyActive = cp.routeStatsPenaltyActive;
  routeStatsUnroutedEdgeSet = cp.routeStatsUnroutedEdgeSet;
  routeStatsCounters = cp.routeStatsCounters;
  routeStatsUnroutedPenalty = cp.routeStatsUnroutedPenalty;
  totalCost = cp.totalCost;
}

#ifndef NDEBUG
void MappingState::debugVerifyCheckpointEqualsCurrent(
    const Checkpoint &cp) const {
  assert(swNodeToHwNode == cp.swNodeToHwNode);
  assert(swPortToHwPort == cp.swPortToHwPort);
  assert(swEdgeToHwPaths == cp.swEdgeToHwPaths);
  assert(hwNodeToSwNodes == cp.hwNodeToSwNodes);
  assert(hwPortToSwPorts == cp.hwPortToSwPorts);
  assert(hwEdgeToSwEdges == cp.hwEdgeToSwEdges);
  assert(portToUsingEdges == cp.portToUsingEdges);
  assert(equalStringCountMaps(spatialPEOccupancyCount, cp.spatialPEOccupancyCount));
  assert(swEdgeTaggedObservationKeys == cp.swEdgeTaggedObservationKeys);
  assert(swEdgeTemporalRouteRecords == cp.swEdgeTemporalRouteRecords);
  assert(taggedObservationIndex.size() == cp.taggedObservationIndex.size());
  for (const auto &entry : cp.taggedObservationIndex) {
    auto it = taggedObservationIndex.find(entry.getFirst());
    assert(it != taggedObservationIndex.end());
    auto expectedEdges = entry.getSecond();
    auto actualEdges = it->second;
    llvm::sort(expectedEdges);
    llvm::sort(actualEdges);
    assert(expectedEdges == actualEdges);
  }
  assert(temporalRouteIndex.size() == cp.temporalRouteIndex.size());
  for (const auto &entry : cp.temporalRouteIndex) {
    auto it = temporalRouteIndex.find(entry.getFirst());
    assert(it != temporalRouteIndex.end());
    auto expectedUses = entry.getSecond();
    auto actualUses = it->second;
    llvm::sort(expectedUses, [](const TemporalRouteUse &lhs,
                                const TemporalRouteUse &rhs) {
      return std::tie(lhs.edgeId, lhs.inPortId, lhs.outPortId) <
             std::tie(rhs.edgeId, rhs.inPortId, rhs.outPortId);
    });
    llvm::sort(actualUses, [](const TemporalRouteUse &lhs,
                              const TemporalRouteUse &rhs) {
      return std::tie(lhs.edgeId, lhs.inPortId, lhs.outPortId) <
             std::tie(rhs.edgeId, rhs.inPortId, rhs.outPortId);
    });
    assert(expectedUses == actualUses);
  }
  assert(functionalRowOccupancy == cp.functionalRowOccupancy);
  assert(functionalColOccupancy == cp.functionalColOccupancy);
  assert(functionalCellOccupancy == cp.functionalCellOccupancy);
  assert(routeStatsInitialized == cp.routeStatsInitialized);
  assert(routeStatsFixedInternalMask == cp.routeStatsFixedInternalMask);
  assert(routeStatsEdgeWeights == cp.routeStatsEdgeWeights);
  assert(routeStatsEdgeModes == cp.routeStatsEdgeModes);
  assert(routeStatsPenaltyActive == cp.routeStatsPenaltyActive);
  assert(routeStatsUnroutedEdgeSet == cp.routeStatsUnroutedEdgeSet);
  assert(routeStatsCounters.overallEdges == cp.routeStatsCounters.overallEdges);
  assert(routeStatsCounters.fixedInternalEdges ==
         cp.routeStatsCounters.fixedInternalEdges);
  assert(routeStatsCounters.directBindingEdges ==
         cp.routeStatsCounters.directBindingEdges);
  assert(routeStatsCounters.routerEdges == cp.routeStatsCounters.routerEdges);
  assert(routeStatsCounters.routedOverallEdges ==
         cp.routeStatsCounters.routedOverallEdges);
  assert(routeStatsCounters.routedRouterEdges ==
         cp.routeStatsCounters.routedRouterEdges);
  assert(routeStatsCounters.unroutedRouterEdges ==
         cp.routeStatsCounters.unroutedRouterEdges);
  assert(std::abs(routeStatsUnroutedPenalty - cp.routeStatsUnroutedPenalty) <=
         1e-9);
  assert(std::abs(totalCost - cp.totalCost) <= 1e-9);
}

void MappingState::debugVerifyCachedState(const Graph &dfg,
                                          const Graph &adg) const {
  llvm::StringMap<unsigned> expectedPEOccupancy;
  std::vector<unsigned> expectedRow(functionalRowOccupancy.size(), 0);
  std::vector<unsigned> expectedCol(functionalColOccupancy.size(), 0);
  std::vector<unsigned> expectedCell(functionalCellOccupancy.size(), 0);

  for (IdIndex swNode = 0; swNode < static_cast<IdIndex>(swNodeToHwNode.size());
       ++swNode) {
    IdIndex hwNode = swNodeToHwNode[swNode];
    if (hwNode == INVALID_ID || hwNode >= static_cast<IdIndex>(adg.nodes.size()))
      continue;
    const Node *hwN = adg.getNode(hwNode);
    if (isSpatialPEFunctionalNode(hwN)) {
      llvm::StringRef peName = getNodeAttrStr(hwN, "pe_name");
      if (!peName.empty())
        ++expectedPEOccupancy[peName];
    }
    if (isFunctionalNode(hwN) && hwNode < static_cast<IdIndex>(hwNodeRows.size()) &&
        hwNode < static_cast<IdIndex>(hwNodeCols.size())) {
      int row = hwNodeRows[hwNode];
      int col = hwNodeCols[hwNode];
      if (row >= 0 && row < static_cast<int>(expectedRow.size()))
        ++expectedRow[static_cast<size_t>(row)];
      if (col >= 0 && col < static_cast<int>(expectedCol.size()))
        ++expectedCol[static_cast<size_t>(col)];
      size_t cellIndex = computeCellIndex(row, col);
      if (cellIndex != static_cast<size_t>(-1) && cellIndex < expectedCell.size())
        ++expectedCell[cellIndex];
    }
  }

  assert(equalStringCountMaps(spatialPEOccupancyCount, expectedPEOccupancy));
  assert(functionalRowOccupancy == expectedRow);
  assert(functionalColOccupancy == expectedCol);
  assert(functionalCellOccupancy == expectedCell);

  if (hasTaggedResources) {
    std::vector<llvm::SmallVector<TaggedObservationKey, 8>>
        expectedEdgeTagged(swEdgeTaggedObservationKeys.size());
    std::vector<llvm::SmallVector<TemporalRouteRecord, 4>> expectedEdgeTemporal(
        swEdgeTemporalRouteRecords.size());
    llvm::DenseMap<TaggedObservationKey, EdgeReverseMap> expectedTaggedIndex;
    llvm::DenseMap<TemporalRouteGroupKey, llvm::SmallVector<TemporalRouteUse, 2>>
        expectedTemporalIndex;

    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(swEdgeToHwPaths.size()); ++edgeId) {
      const auto &path = swEdgeToHwPaths[edgeId];
      if (path.empty())
        continue;
      auto fullPath = routing_detail::buildExportPathForEdge(edgeId, path, *this,
                                                             dfg, adg);
      llvm::SmallVector<routing_detail::TaggedPathObservation, 8> taggedObs;
      llvm::SmallVector<routing_detail::TemporalSwitchTagRouteObservation, 4>
          temporalObs;
      routing_detail::appendTaggedPathObservations(edgeId, fullPath, *this, dfg,
                                                   adg, taggedObs);
      routing_detail::appendTemporalSwitchTagRouteObservations(
          edgeId, fullPath, *this, dfg, adg, temporalObs);

      llvm::DenseSet<TaggedObservationKey> seenTagged;
      for (const auto &obs : taggedObs) {
        TaggedObservationKey key{obs.kind, obs.first, obs.second, obs.tag};
        if (!seenTagged.insert(key).second)
          continue;
        expectedEdgeTagged[edgeId].push_back(key);
        expectedTaggedIndex[key].push_back(edgeId);
      }

      llvm::DenseSet<uint64_t> seenTemporal;
      for (const auto &obs : temporalObs) {
        uint64_t dedupKey =
            llvm::hash_combine(obs.nodeId, obs.inPortId, obs.outPortId, obs.tag);
        if (!seenTemporal.insert(dedupKey).second)
          continue;
        TemporalRouteRecord record{{obs.nodeId, obs.tag}, obs.inPortId,
                                   obs.outPortId};
        expectedEdgeTemporal[edgeId].push_back(record);
        expectedTemporalIndex[record.key].push_back(
            TemporalRouteUse{edgeId, obs.inPortId, obs.outPortId});
      }
    }

    assert(swEdgeTaggedObservationKeys == expectedEdgeTagged);
    assert(swEdgeTemporalRouteRecords == expectedEdgeTemporal);
    assert(taggedObservationIndex.size() == expectedTaggedIndex.size());
    for (const auto &entry : expectedTaggedIndex) {
      auto it = taggedObservationIndex.find(entry.getFirst());
      assert(it != taggedObservationIndex.end());
      auto expectedEdges = entry.getSecond();
      auto actualEdges = it->second;
      llvm::sort(expectedEdges);
      llvm::sort(actualEdges);
      assert(expectedEdges == actualEdges);
    }
    assert(temporalRouteIndex.size() == expectedTemporalIndex.size());
    for (const auto &entry : expectedTemporalIndex) {
      auto it = temporalRouteIndex.find(entry.getFirst());
      assert(it != temporalRouteIndex.end());
      auto expectedUses = entry.getSecond();
      auto actualUses = it->second;
      llvm::sort(expectedUses, [](const TemporalRouteUse &lhs,
                                  const TemporalRouteUse &rhs) {
        return std::tie(lhs.edgeId, lhs.inPortId, lhs.outPortId) <
               std::tie(rhs.edgeId, rhs.inPortId, rhs.outPortId);
      });
      llvm::sort(actualUses, [](const TemporalRouteUse &lhs,
                                const TemporalRouteUse &rhs) {
        return std::tie(lhs.edgeId, lhs.inPortId, lhs.outPortId) <
               std::tie(rhs.edgeId, rhs.inPortId, rhs.outPortId);
      });
      assert(expectedUses == actualUses);
    }
  }

  if (routeStatsInitialized) {
    std::vector<RouteStatsEdgeMode> expectedModes(routeStatsEdgeModes.size(),
                                                  RouteStatsEdgeMode::Missing);
    std::vector<uint8_t> expectedPenalty(routeStatsPenaltyActive.size(), 0);
    std::set<IdIndex> expectedUnroutedSet;
    RouteStatsCounters expectedCounters;
    double expectedPenaltySum = 0.0;

    for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
         ++edgeId) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      ++expectedCounters.overallEdges;
      RouteStatsEdgeMode mode = computeRouteStatsEdgeMode(edgeId);
      if (edgeId < static_cast<IdIndex>(expectedModes.size()))
        expectedModes[edgeId] = mode;
      switch (mode) {
      case RouteStatsEdgeMode::FixedInternal:
        ++expectedCounters.fixedInternalEdges;
        ++expectedCounters.routedOverallEdges;
        break;
      case RouteStatsEdgeMode::DirectBinding:
        ++expectedCounters.directBindingEdges;
        ++expectedCounters.routedOverallEdges;
        break;
      case RouteStatsEdgeMode::RoutedRouter:
        ++expectedCounters.routerEdges;
        ++expectedCounters.routedOverallEdges;
        ++expectedCounters.routedRouterEdges;
        break;
      case RouteStatsEdgeMode::UnroutedRouter:
        ++expectedCounters.routerEdges;
        ++expectedCounters.unroutedRouterEdges;
        expectedUnroutedSet.insert(edgeId);
        if (computeRouteStatsPenaltyActive(edgeId, dfg)) {
          if (edgeId < static_cast<IdIndex>(expectedPenalty.size()))
            expectedPenalty[edgeId] = 1;
          if (edgeId < static_cast<IdIndex>(routeStatsEdgeWeights.size()))
            expectedPenaltySum += routeStatsEdgeWeights[edgeId];
        }
        break;
      case RouteStatsEdgeMode::Missing:
        break;
      }
    }

    assert(routeStatsEdgeModes == expectedModes);
    assert(routeStatsPenaltyActive == expectedPenalty);
    assert(routeStatsUnroutedEdgeSet == expectedUnroutedSet);
    assert(routeStatsCounters.overallEdges == expectedCounters.overallEdges);
    assert(routeStatsCounters.fixedInternalEdges ==
           expectedCounters.fixedInternalEdges);
    assert(routeStatsCounters.directBindingEdges ==
           expectedCounters.directBindingEdges);
    assert(routeStatsCounters.routerEdges == expectedCounters.routerEdges);
    assert(routeStatsCounters.routedOverallEdges ==
           expectedCounters.routedOverallEdges);
    assert(routeStatsCounters.routedRouterEdges ==
           expectedCounters.routedRouterEdges);
    assert(routeStatsCounters.unroutedRouterEdges ==
           expectedCounters.unroutedRouterEdges);
    assert(std::abs(routeStatsUnroutedPenalty - expectedPenaltySum) <= 1e-9);
  }
}

void MappingState::debugMaybeVerifyCachedState(const Graph &dfg,
                                               const Graph &adg) {
  ++debugMutationCount;
  if (debugMutationCount % kDebugCacheVerifyInterval != 0)
    return;
  debugVerifyCachedState(dfg, adg);
}
#endif

} // namespace fcc
