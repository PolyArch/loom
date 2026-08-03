#ifndef LOOM_PNR_SPATIALROUTECOSTSTATE_H
#define LOOM_PNR_SPATIALROUTECOSTSTATE_H

#include "Common/ResolvedPnrPolicy.h"
#include "PnR/RoutingNegotiation.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

/// Worker-local PathFinder cost projection over one exact Spatial candidate.
/// Persistent references never enter this state. The candidate remains the
/// sole owner of per-net active claims; this derived overlay borrows that view.
/// Selecting a logical net removes exactly its old atomic route claims from
/// the working occupancy and incrementally updates only claims and traversals
/// reachable through the frozen reverse CSR indices.
class SpatialRouteCostState final {
public:
  static llvm::Expected<SpatialRouteCostState>
  create(const SpatialCandidateState &candidate);

  llvm::Error selectLogicalNet(std::optional<PnrIndex> logicalNet);
  llvm::Error selectLogicalNet(PnrIndex logicalNet,
                               llvm::ArrayRef<std::uint64_t> activeClaimBits);
  llvm::Error
  updateSelectedLogicalNetClaims(llvm::ArrayRef<std::uint64_t> claimBits);
  llvm::Error acceptSelectedLogicalNet();
  llvm::Error
  synchronizeCandidateTraversals(llvm::ArrayRef<PnrIndex> traversals);
  llvm::Error resetFromCandidate();
  llvm::Error advancePathFinderIteration();

  std::optional<PnrIndex> selectedLogicalNet() const {
    return selectedLogicalNet_;
  }
  std::uint64_t presentPressure() const { return presentPressure_; }
  std::uint64_t historyPressure(PnrIndex capacityDimension) const;
  std::uint64_t workingCapacityUsageRaw(PnrIndex capacityDimension) const;
  RouteCost capacityOveruseCost(PnrIndex capacityDimension) const;
  bool hasCapacityOveruse() const;
  bool isBoundTo(const SpatialCandidateState &candidate) const {
    return candidate_ == &candidate;
  }
  llvm::ArrayRef<RouteCost> lowerBoundArcCosts() const {
    return lowerBoundArcCosts_;
  }
  llvm::ArrayRef<RouteCost> currentArcCosts() const { return currentArcCosts_; }
  std::size_t retainedStorageBytes() const;

private:
  SpatialRouteCostState() = default;

  void beginUpdate();
  llvm::Error stageLogicalNet(PnrIndex logicalNet, bool restore);
  llvm::Error stageClaimBits(llvm::ArrayRef<std::uint64_t> claimBits,
                             bool restore);
  llvm::Error stageClaim(PnrIndex claim, bool restore);
  llvm::Error finishUpdate();
  llvm::Expected<RouteCost> computeTraversalCost(PnrIndex traversal,
                                                 bool dynamicCost,
                                                 bool stagedClaims) const;
  llvm::Expected<RouteCost>
  computeTraversalCost(PnrIndex traversal, std::uint64_t presentPressure,
                       llvm::ArrayRef<std::uint64_t> historyPressure) const;
  llvm::Expected<RouteCost>
  computeTraversalCostImpl(PnrIndex traversal, bool dynamicCost,
                           bool stagedClaims, std::uint64_t presentPressure,
                           llvm::ArrayRef<std::uint64_t> historyPressure) const;
  std::uint64_t capacityUsageForCost(PnrIndex capacityDimension,
                                     bool stagedUsage) const;
  RouteCost claimOveruseForCost(PnrIndex claim, bool stagedClaims) const;
  llvm::ArrayRef<std::uint64_t> logicalNetClaimBits(PnrIndex logicalNet) const;

  const SpatialCandidateState *candidate_ = nullptr;
  const FrozenSpatialPnrProblem *problem_ = nullptr;
  ResolvedPathFinderPolicy policy_;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex routeClaimCount_ = 0;
  std::size_t routeClaimWordCount_ = 0;
  std::optional<PnrIndex> selectedLogicalNet_;
  std::uint64_t presentPressure_ = 0;

  std::vector<std::uint64_t> workingCapacityUsageRaw_;
  std::vector<std::uint64_t> historyPressure_;
  std::vector<RouteCost> capacityOveruseCosts_;
  std::vector<RouteCost> currentClaimOveruseCosts_;
  std::vector<RouteCost> lowerBoundTraversalCosts_;
  std::vector<RouteCost> currentTraversalCosts_;
  std::vector<RouteCost> lowerBoundArcCosts_;
  std::vector<RouteCost> currentArcCosts_;
  std::vector<std::uint64_t> selectedLogicalNetClaimBits_;

  std::vector<std::uint64_t> capacityUpdateEpochs_;
  std::vector<std::uint64_t> claimUpdateEpochs_;
  std::vector<std::uint64_t> traversalUpdateEpochs_;
  std::vector<std::uint64_t> stagedCapacityUsageRaw_;
  std::vector<std::uint64_t> stagedHistoryPressure_;
  std::vector<RouteCost> stagedCapacityOveruseCosts_;
  std::vector<RouteCost> stagedClaimOveruseCosts_;
  std::vector<RouteCost> stagedTraversalCosts_;
  std::vector<PnrIndex> affectedCapacities_;
  std::vector<PnrIndex> affectedClaims_;
  std::vector<PnrIndex> affectedTraversals_;
  std::uint64_t updateEpoch_ = 0;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALROUTECOSTSTATE_H
