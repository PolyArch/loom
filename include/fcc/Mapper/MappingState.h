#ifndef FCC_MAPPER_MAPPINGSTATE_H
#define FCC_MAPPER_MAPPINGSTATE_H

#include "fcc/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>
#include <set>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace fcc {

class Graph;
class ADGFlattener;

enum class ActionResult {
  Success,
  FailedHardConstraint,
  FailedResourceUnavailable,
  FailedTypeMismatch,
  FailedInternalError,
};

class MappingState {
public:
  using EdgePath = llvm::SmallVector<IdIndex, 8>;
  using NodeReverseMap = llvm::SmallVector<IdIndex, 2>;
  using EdgeReverseMap = llvm::SmallVector<IdIndex, 4>;

  enum class TaggedObservationKind : uint8_t {
    RoutingOutput = 0,
    HardwareEdge = 1,
  };

  struct TaggedObservationKey {
    TaggedObservationKind kind = TaggedObservationKind::RoutingOutput;
    IdIndex first = INVALID_ID;
    IdIndex second = INVALID_ID;
    uint64_t tag = 0;

    bool operator==(const TaggedObservationKey &other) const {
      return kind == other.kind && first == other.first &&
             second == other.second && tag == other.tag;
    }
  };

  struct TemporalRouteGroupKey {
    IdIndex nodeId = INVALID_ID;
    uint64_t tag = 0;

    bool operator==(const TemporalRouteGroupKey &other) const {
      return nodeId == other.nodeId && tag == other.tag;
    }
  };

  struct TemporalRouteUse {
    IdIndex edgeId = INVALID_ID;
    IdIndex inPortId = INVALID_ID;
    IdIndex outPortId = INVALID_ID;

    bool operator==(const TemporalRouteUse &other) const {
      return edgeId == other.edgeId && inPortId == other.inPortId &&
             outPortId == other.outPortId;
    }
  };

  struct TemporalRouteRecord {
    TemporalRouteGroupKey key;
    IdIndex inPortId = INVALID_ID;
    IdIndex outPortId = INVALID_ID;

    bool operator==(const TemporalRouteRecord &other) const {
      return key == other.key && inPortId == other.inPortId &&
             outPortId == other.outPortId;
    }
  };

  enum class RouteStatsEdgeMode : uint8_t {
    Missing = 0,
    FixedInternal = 1,
    DirectBinding = 2,
    RoutedRouter = 3,
    UnroutedRouter = 4,
  };

  struct RouteStatsCounters {
    unsigned overallEdges = 0;
    unsigned fixedInternalEdges = 0;
    unsigned directBindingEdges = 0;
    unsigned routerEdges = 0;
    unsigned routedOverallEdges = 0;
    unsigned routedRouterEdges = 0;
    unsigned unroutedRouterEdges = 0;
  };

  MappingState() = default;

  /// Initialize state vectors for the given DFG and ADG sizes.
  void init(const Graph &dfg, const Graph &adg,
            const ADGFlattener *flattener = nullptr);
  void initializeRouteStats(const Graph &dfg,
                            llvm::ArrayRef<uint8_t> fixedInternalMask);

  // Forward mappings: SW entity -> HW entity.
  std::vector<IdIndex> swNodeToHwNode;
  std::vector<IdIndex> swPortToHwPort;
  std::vector<EdgePath> swEdgeToHwPaths;

  // Reverse mappings: HW entity -> SW entities.
  std::vector<NodeReverseMap> hwNodeToSwNodes;
  std::vector<NodeReverseMap> hwPortToSwPorts;
  std::vector<EdgeReverseMap> hwEdgeToSwEdges;

  // Resource usage index: HW port ID -> SW edge IDs whose paths use that port.
  // Dense vector indexed by port ID for O(1) lookup in hot paths.
  // Excludes synthetic direct-binding routes (path[0] == path[1]).
  std::vector<EdgeReverseMap> portToUsingEdges;

  // Runtime FIFO bypass overrides keyed by hardware node id.
  // -1 means use the ADG default, 0 means force buffered, 1 means force bypassed.
  std::vector<int8_t> hwNodeFifoBypassedOverride;

  // Spatial PE occupancy count keyed by pe_name.
  llvm::StringMap<unsigned> spatialPEOccupancyCount;

  // Immutable hardware edge lookup indexed by source output port and
  // destination input port.
  std::vector<llvm::DenseMap<IdIndex, IdIndex>> hwEdgeByPortPair;

  // Tagged-path observation indexes for conflict checks.
  std::vector<llvm::SmallVector<TaggedObservationKey, 8>>
      swEdgeTaggedObservationKeys;
  std::vector<llvm::SmallVector<TemporalRouteRecord, 4>>
      swEdgeTemporalRouteRecords;
  llvm::DenseMap<TaggedObservationKey, EdgeReverseMap> taggedObservationIndex;
  llvm::DenseMap<TemporalRouteGroupKey, llvm::SmallVector<TemporalRouteUse, 2>>
      temporalRouteIndex;

  // Immutable grid metadata and dynamic functional occupancy summaries.
  std::vector<int> hwNodeRows;
  std::vector<int> hwNodeCols;
  unsigned occupancyGridCols = 0;
  std::vector<unsigned> functionalRowOccupancy;
  std::vector<unsigned> functionalColOccupancy;
  std::vector<unsigned> functionalCellOccupancy;

  bool routeStatsInitialized = false;
  std::vector<uint8_t> routeStatsFixedInternalMask;
  std::vector<double> routeStatsEdgeWeights;
  std::vector<RouteStatsEdgeMode> routeStatsEdgeModes;
  std::vector<uint8_t> routeStatsPenaltyActive;
  std::set<IdIndex> routeStatsUnroutedEdgeSet;
  RouteStatsCounters routeStatsCounters;
  double routeStatsUnroutedPenalty = 0.0;

  bool hasTaggedResources = false;

  // Cost metrics.
  double totalCost = 0.0;

  struct Savepoint {
    size_t marker = 0;
  };

  // Action primitives.
  ActionResult mapNode(IdIndex swNode, IdIndex hwNode,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapNode(IdIndex swNode,
                         const Graph &dfg, const Graph &adg);
  ActionResult mapPort(IdIndex swPort, IdIndex hwPort,
                       const Graph &dfg, const Graph &adg);
  ActionResult mapPortBridgeAware(IdIndex swPort, IdIndex hwPort,
                                  const Graph &dfg, const Graph &adg);
  ActionResult mapEdge(IdIndex swEdge, llvm::ArrayRef<IdIndex> path,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapEdge(IdIndex swEdge, const Graph &dfg, const Graph &adg);
  void clearRoutes(const Graph &dfg, const Graph &adg,
                   bool preserveDirectBindings = true);

  Savepoint beginSavepoint();
  void rollbackSavepoint(Savepoint savepoint);
  void commitSavepoint(Savepoint savepoint);

  /// Checkpoint/restore for SA.
  struct Checkpoint {
    std::vector<IdIndex> swNodeToHwNode;
    std::vector<IdIndex> swPortToHwPort;
    std::vector<EdgePath> swEdgeToHwPaths;
    std::vector<NodeReverseMap> hwNodeToSwNodes;
    std::vector<NodeReverseMap> hwPortToSwPorts;
    std::vector<EdgeReverseMap> hwEdgeToSwEdges;
    std::vector<EdgeReverseMap> portToUsingEdges;
    std::vector<int8_t> hwNodeFifoBypassedOverride;
    llvm::StringMap<unsigned> spatialPEOccupancyCount;
    std::vector<llvm::SmallVector<TaggedObservationKey, 8>>
        swEdgeTaggedObservationKeys;
    std::vector<llvm::SmallVector<TemporalRouteRecord, 4>>
        swEdgeTemporalRouteRecords;
    llvm::DenseMap<TaggedObservationKey, EdgeReverseMap> taggedObservationIndex;
    llvm::DenseMap<TemporalRouteGroupKey,
                   llvm::SmallVector<TemporalRouteUse, 2>>
        temporalRouteIndex;
    std::vector<unsigned> functionalRowOccupancy;
    std::vector<unsigned> functionalColOccupancy;
    std::vector<unsigned> functionalCellOccupancy;
    bool routeStatsInitialized;
    std::vector<uint8_t> routeStatsFixedInternalMask;
    std::vector<double> routeStatsEdgeWeights;
    std::vector<RouteStatsEdgeMode> routeStatsEdgeModes;
    std::vector<uint8_t> routeStatsPenaltyActive;
    std::set<IdIndex> routeStatsUnroutedEdgeSet;
    RouteStatsCounters routeStatsCounters;
    double routeStatsUnroutedPenalty;
    double totalCost;
  };

  Checkpoint save() const;
  void restore(const Checkpoint &cp);

  bool isSpatialPEOccupied(llvm::StringRef peName,
                           const Graph &adg,
                           IdIndex ignoreHwNode = INVALID_ID) const;

  IdIndex lookupHwEdge(IdIndex outPort, IdIndex inPort) const;
  unsigned getFunctionalRowOccupancy(int row) const;
  unsigned getFunctionalColOccupancy(int col) const;
  unsigned getFunctionalCellOccupancy(int row, int col) const;

private:
  struct UndoRecord {
    enum class Kind {
      SwNodeToHwNode,
      SwPortToHwPort,
      SwEdgeToHwPath,
      HwNodeToSwNodes,
      HwPortToSwPorts,
      HwEdgeToSwEdges,
      PortToUsingEdges,
      SpatialPEOccupancy,
      SwEdgeTaggedObservationKeys,
      SwEdgeTemporalRouteRecords,
      TaggedObservationIndex,
      TemporalRouteIndex,
      FunctionalRowOccupancy,
      FunctionalColOccupancy,
      FunctionalCellOccupancy,
      RouteStatsEdgeMode,
      RouteStatsPenaltyActive,
      RouteStatsUnroutedEdgeMembership,
      RouteStatsCounters,
      RouteStatsUnroutedPenalty,
      TotalCost,
    };

    Kind kind = Kind::SwNodeToHwNode;
    size_t index = 0;
    IdIndex idValue = INVALID_ID;
    EdgePath pathValue;
    NodeReverseMap nodeVecValue;
    EdgeReverseMap edgeVecValue;
    llvm::SmallVector<TaggedObservationKey, 8> taggedObservationVecValue;
    llvm::SmallVector<TemporalRouteRecord, 4> temporalRouteRecordVecValue;
    TaggedObservationKey taggedObservationKey;
    TemporalRouteGroupKey temporalRouteGroupKey;
    llvm::SmallVector<TemporalRouteUse, 2> temporalRouteUseVecValue;
    std::string stringKey;
    unsigned countValue = 0;
    bool countExisted = false;
    RouteStatsEdgeMode routeStatsEdgeModeValue = RouteStatsEdgeMode::Missing;
    RouteStatsCounters routeStatsCountersValue;
    double doubleValue = 0.0;
  };

  bool hasActiveSavepoint() const;
  void recordSwNodeToHwNode(size_t index);
  void recordSwPortToHwPort(size_t index);
  void recordSwEdgeToHwPath(size_t index);
  void recordHwNodeToSwNodes(size_t index);
  void recordHwPortToSwPorts(size_t index);
  void recordHwEdgeToSwEdges(size_t index);
  void recordPortToUsingEdges(size_t index);
  void recordSpatialPEOccupancy(llvm::StringRef peName);
  void recordSwEdgeTaggedObservationKeys(size_t index);
  void recordSwEdgeTemporalRouteRecords(size_t index);
  void recordTaggedObservationIndex(const TaggedObservationKey &key);
  void recordTemporalRouteIndex(const TemporalRouteGroupKey &key);
  void recordFunctionalRowOccupancy(size_t index);
  void recordFunctionalColOccupancy(size_t index);
  void recordFunctionalCellOccupancy(size_t index);
  void recordRouteStatsEdgeMode(size_t index);
  void recordRouteStatsPenaltyActive(size_t index);
  void recordRouteStatsUnroutedEdgeMembership(IdIndex edgeId);
  void recordRouteStatsCounters();
  void recordRouteStatsUnroutedPenalty();
  void recordTotalCost();
  void applyUndoRecord(const UndoRecord &record);
  size_t computeCellIndex(int row, int col) const;
  void applyFunctionalNodeOccupancyDelta(IdIndex hwNode, int delta,
                                         const Graph &adg);
  RouteStatsEdgeMode computeRouteStatsEdgeMode(IdIndex swEdge) const;
  bool computeRouteStatsPenaltyActive(IdIndex swEdge, const Graph &dfg) const;
  void refreshRouteStatsForEdge(IdIndex swEdge, const Graph &dfg);
  void refreshRouteStatsForIncidentEdges(IdIndex swNode, const Graph &dfg);
  void refreshTaggedObservationsForEdge(IdIndex swEdge, llvm::ArrayRef<IdIndex> path,
                                        const Graph &dfg, const Graph &adg);
  void clearTaggedObservationsForEdge(IdIndex swEdge);
#ifndef NDEBUG
  void debugVerifyCheckpointEqualsCurrent(const Checkpoint &cp) const;
  void debugVerifyCachedState(const Graph &dfg, const Graph &adg) const;
  void debugMaybeVerifyCachedState(const Graph &dfg, const Graph &adg);
  uint64_t debugMutationCount = 0;
  std::vector<Checkpoint> debugSavepointCheckpoints;
#endif

  std::vector<UndoRecord> undoLog;
  std::vector<size_t> savepointMarkers;
};

} // namespace fcc

namespace llvm {

template <>
struct DenseMapInfo<fcc::MappingState::TaggedObservationKey> {
  static inline fcc::MappingState::TaggedObservationKey getEmptyKey() {
    return {fcc::MappingState::TaggedObservationKind::RoutingOutput,
            fcc::INVALID_ID, fcc::INVALID_ID, 0};
  }

  static inline fcc::MappingState::TaggedObservationKey getTombstoneKey() {
    return {fcc::MappingState::TaggedObservationKind::HardwareEdge,
            fcc::INVALID_ID - 1, fcc::INVALID_ID - 1, 0};
  }

  static unsigned getHashValue(
      const fcc::MappingState::TaggedObservationKey &key) {
    return hash_combine(static_cast<unsigned>(key.kind), key.first, key.second,
                        key.tag);
  }

  static bool isEqual(const fcc::MappingState::TaggedObservationKey &lhs,
                      const fcc::MappingState::TaggedObservationKey &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<fcc::MappingState::TemporalRouteGroupKey> {
  static inline fcc::MappingState::TemporalRouteGroupKey getEmptyKey() {
    return {fcc::INVALID_ID, 0};
  }

  static inline fcc::MappingState::TemporalRouteGroupKey getTombstoneKey() {
    return {fcc::INVALID_ID - 1, 0};
  }

  static unsigned getHashValue(
      const fcc::MappingState::TemporalRouteGroupKey &key) {
    return hash_combine(key.nodeId, key.tag);
  }

  static bool isEqual(const fcc::MappingState::TemporalRouteGroupKey &lhs,
                      const fcc::MappingState::TemporalRouteGroupKey &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // FCC_MAPPER_MAPPINGSTATE_H
