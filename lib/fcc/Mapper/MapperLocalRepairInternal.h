#ifndef FCC_MAPPER_MAPPERLOCALREPAIRINTERNAL_H
#define FCC_MAPPER_MAPPERLOCALREPAIRINTERNAL_H

#include "MapperInternal.h"
#include "MapperRoutingInternal.h"
#include "fcc/Mapper/Mapper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>
#include <vector>

namespace fcc {

bool isRoutingCrossbarOutputPortForRepair(IdIndex portId, const Graph &adg);
double estimatePortDistance(IdIndex lhsPortId, IdIndex rhsPortId,
                            const Graph &adg,
                            const ADGFlattener &flattener);

class LocalRepairDriver {
public:
  LocalRepairDriver(
      Mapper &mapper, MappingState &state,
      const MappingState::Checkpoint &baseCheckpoint,
      llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      std::vector<TechMappedEdgeKind> &edgeKinds, const Mapper::Options &opts,
      const CongestionState *congestion, unsigned recursionDepth);

  bool run();

private:
  using CandidateMap =
      llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>;

  double edgeWeight(IdIndex edgeId) const;
  std::pair<unsigned, double>
  computePriorityMetrics(const MappingState &candidateState) const;
  double computeRepairabilityScore(const MappingState &candidateState) const;
  bool shouldEscalateToCPSat() const;
  bool rerouteRepairState(MappingState &repairState) const;
  bool updateBest(bool allRouted);
  double evaluateFailedEdgeDelta(
      llvm::ArrayRef<std::pair<IdIndex, IdIndex>> moves) const;
  llvm::SmallVector<IdIndex, 24>
  buildFocusedRepairNeighborhood(llvm::ArrayRef<IdIndex> seedEdges,
                                 unsigned maxEdges) const;
  llvm::SmallVector<IdIndex, 24>
  buildConflictNeighborhood(llvm::ArrayRef<IdIndex> seedEdges,
                            unsigned maxEdges) const;
  void expandConflictNeighborhood(
      llvm::SmallVectorImpl<IdIndex> &repairEdges, unsigned maxEdges) const;

  bool runHotspotRepairAndEarlyCPSat();
  bool runMemoryExactRepairs();
  bool runLateRepairStages();

  Mapper &mapper;
  MappingState &state;
  const MappingState::Checkpoint &baseCheckpoint;
  llvm::ArrayRef<IdIndex> failedEdges;
  const Graph &dfg;
  const Graph &adg;
  const ADGFlattener &flattener;
  const CandidateMap &candidates;
  std::vector<TechMappedEdgeKind> &edgeKinds;
  const Mapper::Options &opts;
  const CongestionState *congestion;
  unsigned recursionDepth;

  const MapperLocalRepairOptions &repairOpts;
  std::vector<double> edgeWeights;
  mapper_detail::CandidateSetMap candidateSets;
  unsigned maxRepairRecursionDepth = 0;

  llvm::DenseMap<IdIndex, double> hotspotWeights;
  llvm::DenseMap<IdIndex, double> failedEdgeWeights;
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> nodeToFailedEdges;
  std::vector<IdIndex> hotspots;

  MappingState::Checkpoint bestCheckpoint;
  MappingState::Checkpoint bestPlacementCheckpoint;
  unsigned bestRouted = 0;
  double bestUnroutedPenalty = 0.0;
  size_t bestPathLen = 0;
  double bestPlacementCost = 0.0;
  bool bestAllRouted = false;
  std::vector<IdIndex> bestFailedEdges;
  llvm::DenseSet<IdIndex> repairPriorityEdges;
  unsigned bestPriorityRouted = 0;
  double bestPriorityPenalty = 0.0;
  double bestRepairabilityScore = 0.0;
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPERLOCALREPAIRINTERNAL_H
