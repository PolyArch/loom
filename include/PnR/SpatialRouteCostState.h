#ifndef LOOM_PNR_SPATIALROUTECOSTSTATE_H
#define LOOM_PNR_SPATIALROUTECOSTSTATE_H

#include "Common/ResolvedPnrPolicy.h"
#include "PnR/RoutingNegotiation.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::pnr {

namespace detail {
struct SpatialRouteCostSwitchRowState;
}

struct SpatialTagDomainUse final {
  PnrIndex domain = 0;
  std::uint64_t marginalResidentCount = 0;

  friend bool operator==(SpatialTagDomainUse lhs, SpatialTagDomainUse rhs) {
    return lhs.domain == rhs.domain &&
           lhs.marginalResidentCount == rhs.marginalResidentCount;
  }
};

/// Worker-local PathFinder cost projection over one exact Spatial candidate.
/// Persistent references never enter this state. The candidate remains the
/// sole owner of per-net active claims; this derived overlay borrows that view.
/// Selecting a logical net removes exactly its old atomic route claims from
/// the working occupancy and incrementally updates only claims and traversals
/// reachable through the frozen reverse CSR indices.
class SpatialRouteCostState final {
public:
  SpatialRouteCostState(SpatialRouteCostState &&) noexcept;
  SpatialRouteCostState(const SpatialRouteCostState &) = delete;
  SpatialRouteCostState &operator=(const SpatialRouteCostState &) = delete;
  SpatialRouteCostState &operator=(SpatialRouteCostState &&) = delete;
  ~SpatialRouteCostState();

  static llvm::Expected<SpatialRouteCostState>
  create(const SpatialCandidateState &candidate);

  llvm::Error selectLogicalNet(std::optional<PnrIndex> logicalNet);
  llvm::Error selectLogicalNet(PnrIndex logicalNet,
                               llvm::ArrayRef<std::uint64_t> activeClaimBits);
  llvm::Error
  updateSelectedLogicalNetClaims(llvm::ArrayRef<std::uint64_t> claimBits);
  llvm::Error updateSelectedLogicalNetTagUses(
      const RouteTreeState &route,
      const SpatialTagContinuityProjection &continuity);
  llvm::Error acceptSelectedLogicalNet();
  llvm::Error
  synchronizeTagProjection(const SpatialTagAssignmentSummary &summary,
                           llvm::ArrayRef<PnrIndex> changedLogicalNets = {});
  llvm::Error synchronizeTagProjection(const SpatialTagAssignmentDelta &delta);
  llvm::Error commitTagProjectionDelta();
  llvm::Error rollbackTagProjectionDelta();
  bool hasActiveTagProjectionDelta() const {
    return inverseTagDelta_.has_value();
  }
  llvm::Error synchronizeCandidateTags();
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
  bool hasTagPressureViolation() const;
  bool arcHasTagPressure(PnrIndex arc) const;
  bool logicalNetHasTagPressure(PnrIndex logicalNet) const;
  RouteCost logicalNetTagPressure(PnrIndex logicalNet) const;
  std::uint64_t logicalNetTagUnassignedCount(PnrIndex logicalNet) const;
  llvm::ArrayRef<SpatialTagDomainUse>
  logicalNetTagDomainUses(PnrIndex logicalNet) const;
  std::uint64_t workingTagDomainUsage(PnrIndex domain) const;
  std::uint64_t tagDomainEncodingCapacity(PnrIndex domain) const;
  std::optional<std::uint64_t> tagDomainResidentCapacity(PnrIndex domain) const;
  std::uint64_t tagDomainResidentOveruse(PnrIndex domain) const;
  std::uint64_t tagDomainConflictCount(PnrIndex domain) const;
  bool isBoundTo(const SpatialCandidateState &candidate) const {
    return candidate_ == &candidate;
  }
  llvm::ArrayRef<RouteCost> lowerBoundArcCosts() const {
    return lowerBoundArcCosts_;
  }
  std::uint64_t lowerBoundCostRevision() const {
    return lowerBoundCostRevision_;
  }
  llvm::ArrayRef<RouteCost> currentArcCosts() const { return currentArcCosts_; }
  std::size_t retainedStorageBytes() const;

private:
  SpatialRouteCostState() = default;

  llvm::Error resetFromVerifiedCandidate();

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
  llvm::Expected<RouteCost> computeArcCost(PnrIndex arc, bool dynamicCost,
                                           bool stagedClaims,
                                           bool stagedTags) const;
  llvm::Expected<RouteCost>
  computeArcCost(PnrIndex arc, std::uint64_t presentPressure,
                 llvm::ArrayRef<std::uint64_t> routeHistoryPressure,
                 llvm::ArrayRef<std::uint64_t> residentHistoryPressure,
                 llvm::ArrayRef<std::uint64_t> encodingHistoryPressure) const;
  llvm::Expected<RouteCost>
  computeTagDomainCost(PnrIndex domain, bool resident, bool dynamicCost,
                       bool stagedTags, std::uint64_t presentPressure,
                       std::uint64_t historyPressure) const;
  llvm::Error
  replaceSelectedTagUses(llvm::ArrayRef<SpatialTagDomainUse> replacement);
  llvm::Error stageTagUses(llvm::ArrayRef<SpatialTagDomainUse> uses,
                           bool restore);
  llvm::Error rebuildTagProjectionFromCandidate(bool resetHistory);
  llvm::Error rebuildSwitchRowProjectionFromCandidate();
  llvm::Error
  synchronizeCandidateSwitchRows(llvm::ArrayRef<PnrIndex> changedLogicalNets);
  llvm::Error recomputeAllArcCosts(bool resetTagHistory);
  std::uint64_t tagUsageForCost(PnrIndex domain, bool stagedTags) const;
  std::uint64_t encodingPressureRaw(PnrIndex domain, bool stagedTags) const;
  llvm::ArrayRef<std::uint64_t> logicalNetClaimBits(PnrIndex logicalNet) const;

  const SpatialCandidateState *candidate_ = nullptr;
  const FrozenSpatialPnrProblem *problem_ = nullptr;
  ResolvedPathFinderPolicy policy_;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex routeClaimCount_ = 0;
  std::size_t routeClaimWordCount_ = 0;
  std::optional<PnrIndex> selectedLogicalNet_;
  std::uint64_t presentPressure_ = 0;
  std::uint64_t lowerBoundCostRevision_ = 0;

  std::vector<std::uint64_t> workingCapacityUsageRaw_;
  std::vector<std::uint64_t> historyPressure_;
  std::vector<RouteCost> capacityOveruseCosts_;
  std::vector<RouteCost> currentClaimOveruseCosts_;
  std::vector<RouteCost> lowerBoundTraversalCosts_;
  std::vector<RouteCost> currentTraversalCosts_;
  std::vector<RouteCost> lowerBoundArcCosts_;
  std::vector<RouteCost> currentArcCosts_;
  std::vector<std::uint64_t> selectedLogicalNetClaimBits_;

  std::vector<std::vector<SpatialTagDomainUse>> logicalNetTagUses_;
  std::vector<std::uint64_t> logicalNetTagUnassignedCounts_;
  std::uint64_t tagUnassignedCount_ = 0;
  std::vector<std::vector<std::optional<llvm::APInt>>> logicalNetTagValues_;
  std::vector<SpatialTagDomainUse> selectedLogicalNetTagUses_;
  std::vector<std::uint64_t> workingTagDomainUsage_;
  std::vector<std::uint64_t> tagDomainConflictCounts_;
  std::vector<std::uint64_t> tagResidentHistoryPressure_;
  std::vector<std::uint64_t> tagEncodingHistoryPressure_;
  std::vector<RouteCost> tagResidentOveruseCosts_;
  std::vector<RouteCost> tagEncodingPressureCosts_;
  std::vector<PnrIndex> tagDomainArcOffsets_;
  std::vector<PnrIndex> tagDomainArcs_;

  std::vector<std::uint64_t> capacityUpdateEpochs_;
  std::vector<std::uint64_t> claimUpdateEpochs_;
  std::vector<std::uint64_t> traversalUpdateEpochs_;
  std::vector<std::uint64_t> arcUpdateEpochs_;
  std::vector<std::uint64_t> stagedCapacityUsageRaw_;
  std::vector<std::uint64_t> stagedHistoryPressure_;
  std::vector<std::uint64_t> stagedTagResidentHistoryPressure_;
  std::vector<std::uint64_t> stagedTagEncodingHistoryPressure_;
  std::vector<RouteCost> stagedCapacityOveruseCosts_;
  std::vector<RouteCost> stagedClaimOveruseCosts_;
  std::vector<RouteCost> stagedTraversalCosts_;
  std::vector<std::uint64_t> tagDomainUpdateEpochs_;
  std::vector<std::uint64_t> stagedTagDomainUsage_;
  std::vector<RouteCost> stagedTagResidentOveruseCosts_;
  std::vector<RouteCost> stagedTagEncodingPressureCosts_;
  std::vector<RouteCost> stagedArcCosts_;
  std::vector<PnrIndex> affectedCapacities_;
  std::vector<PnrIndex> affectedClaims_;
  std::vector<PnrIndex> affectedTraversals_;
  std::vector<PnrIndex> affectedTagDomains_;
  std::vector<PnrIndex> affectedTagArcs_;
  std::unique_ptr<detail::SpatialRouteCostSwitchRowState> switchRows_;
  std::optional<SpatialTagAssignmentDelta> inverseTagDelta_;
  std::uint64_t updateEpoch_ = 0;

  friend class SpatialActionExecutorScratch;
  friend class SpatialActionProbe;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALROUTECOSTSTATE_H
