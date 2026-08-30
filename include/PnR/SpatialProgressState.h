#ifndef LOOM_PNR_SPATIALPROGRESSSTATE_H
#define LOOM_PNR_SPATIALPROGRESSSTATE_H

#include "Fabric/Identity/FabricRefs.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <map>
#include <string>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class RouteTreeState;
class SpatialCandidateState;

enum class SpatialProgressRouteAnchorKind : std::uint8_t {
  RouteTreeArc,
  SourceAttachment,
  SinkAttachment,
};

struct SpatialProgressRouteAnchor final {
  SpatialProgressRouteAnchorKind kind =
      SpatialProgressRouteAnchorKind::RouteTreeArc;
  PnrIndex logicalNet = getInvalidPnrIndex();
  PnrIndex traversal = getInvalidPnrIndex();
  PnrIndex endpoint = getInvalidPnrIndex();
  PnrIndex sinkObligation = getInvalidPnrIndex();
};

struct SpatialFiniteBufferConflictWitness final {
  PnrIndex ownerOrdinal = getInvalidPnrIndex();
  ::loom::fabric::FabricFifoOccurrenceRef owner;
  std::vector<PnrIndex> competingLogicalNets;
  std::vector<SpatialProgressRouteAnchor> routeAnchors;
};

struct SpatialFifoCapacityShortfall final {
  ::loom::fabric::FabricFifoOccurrenceRef owner;
  std::uint64_t selectedCapacity = 0;
  std::uint64_t minimumLegalCapacity = 0;
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> logicalNets;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> routeAnchors;
};

llvm::Expected<std::optional<SpatialFifoCapacityShortfall>>
projectSpatialFifoCapacityShortfall(const SpatialCandidateState &candidate);

/// One FIFO shared-pool use of one logical net, rebuilt by walking every
/// producer-to-sink channel of that net. Channel count is distinct from route
/// traversal count: a multicast prefix participates in each sink channel.
struct SpatialProgressOwnerCapacityUse final {
  PnrIndex owner = getInvalidPnrIndex();
  std::uint64_t channelCount = 0;
  std::uint64_t initializedFeedbackChannelCount = 0;
  bool repeatedWithinChannel = false;
  bool queueClassIndeterminate = false;
  std::vector<llvm::APInt> queueClasses;

  friend bool operator==(const SpatialProgressOwnerCapacityUse &lhs,
                         const SpatialProgressOwnerCapacityUse &rhs) {
    return lhs.owner == rhs.owner && lhs.channelCount == rhs.channelCount &&
           lhs.initializedFeedbackChannelCount ==
               rhs.initializedFeedbackChannelCount &&
           lhs.repeatedWithinChannel == rhs.repeatedWithinChannel &&
           lhs.queueClassIndeterminate == rhs.queueClassIndeterminate &&
           lhs.queueClasses == rhs.queueClasses;
  }
};

struct SpatialProgressNetCapacityProjection final {
  std::vector<SpatialProgressOwnerCapacityUse> owners;

  friend bool operator==(const SpatialProgressNetCapacityProjection &lhs,
                         const SpatialProgressNetCapacityProjection &rhs) {
    return lhs.owners == rhs.owners;
  }
};

struct SpatialProgressStatistics final {
  std::uint64_t incrementalUpdateCount = 0;
  std::uint64_t incrementalUpdateWallTimeNanoseconds = 0;
  std::uint64_t cachedVerificationCount = 0;
  std::uint64_t coldVerificationCount = 0;
  std::uint64_t coldVerificationWallTimeNanoseconds = 0;
  std::uint64_t coldProgressScanCount = 0;
  std::uint64_t coldProgressScanWallTimeNanoseconds = 0;
};

/// Rebuildable exact projection of route-selected durable progress facts.
/// FrozenSpatialPnrProblem owns the immutable owner/dependency indices. This
/// state contains only dense ordinals, refcounts, bitsets, and cached totals.
class SpatialProgressState final {
public:
  static llvm::Expected<SpatialProgressState>
  create(const SpatialCandidateState &candidate);

  std::uint64_t sharedFiniteBufferConflictCount() const {
    return sharedFiniteBufferConflictCount_;
  }
  std::uint64_t routeDependencyViolationCount() const {
    return routeDependencyViolationCount_;
  }
  std::uint64_t capacityProofDebtWitnessCount() const {
    return capacityProofDebtWitnessCount_;
  }
  std::uint64_t capacityShortfallOwnerCount() const {
    return capacityShortfallOwnerCount_;
  }
  std::uint64_t capacityShortfall() const { return capacityShortfall_; }
  std::uint64_t capacityShortfall(PnrIndex owner) const {
    return owner < ownerCount_ ? ownerCapacityShortfall(owner) : 0;
  }
  std::uint64_t capacityObligationRouteAnchorCount() const {
    return capacityObligationRouteAnchorCount_;
  }
  bool capacityProofDebtOwner(PnrIndex owner) const {
    return owner < ownerCount_ &&
           (capacityProofDebtOwnerBits_[owner / 64] &
            (std::uint64_t{1} << (owner % 64))) != 0;
  }
  bool capacityShortfallOwner(PnrIndex owner) const {
    return owner < ownerCount_ && ownerCapacityShortfall(owner) != 0;
  }
  /// Shared finite-buffer residency is a typed unproven-recurrence fact, not
  /// a Spatial violation: the ordinary Mapping remains importable while the
  /// unestablished recurrence withholds Dataflow spectrum qualification, and
  /// the router prices foreign residency as negotiable congestion. Only route
  /// dependency violations remain hard progress violations.
  std::uint64_t hardProgressViolation() const {
    return routeDependencyViolationCount_ + capacityShortfallOwnerCount_;
  }
  PnrIndex finiteBufferOwnerLogicalNetCount(PnrIndex owner) const;
  bool finiteBufferOwnerConflicts(PnrIndex owner) const;
  std::optional<PnrIndex> firstFiniteBufferConflictOwner() const;
  llvm::Error
  enumerateFiniteBufferConflictOwners(std::vector<PnrIndex> &owners) const;
  std::optional<PnrIndex> firstCapacityProofDebtOwner() const;
  llvm::Error
  enumerateCapacityProofDebtOwners(std::vector<PnrIndex> &owners) const;
  std::optional<PnrIndex> firstCapacityShortfallOwner() const;
  llvm::Error
  enumerateCapacityShortfallOwners(std::vector<PnrIndex> &owners) const;
  std::uint64_t logicalNetRouteDependencyViolationCount(
      PnrIndex logicalNet) const;
  std::size_t retainedStorageBytes() const;
  const SpatialProgressStatistics &statistics() const { return statistics_; }

  llvm::Error applyTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                                  PnrIndex removed, PnrIndex added);
  void revertTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                            PnrIndex removed, PnrIndex added) noexcept;
  llvm::Error refreshLogicalNetRouteDependencies(
      const SpatialCandidateState &candidate, PnrIndex logicalNet);
  void restoreLogicalNetRouteDependencyCount(PnrIndex logicalNet,
                                             std::uint64_t count) noexcept;
  llvm::Expected<SpatialProgressNetCapacityProjection>
  replaceLogicalNetCapacityProjection(const SpatialCandidateState &candidate,
                                      PnrIndex logicalNet,
                                      const RouteTreeState *route = nullptr);
  void restoreLogicalNetCapacityProjection(
      PnrIndex logicalNet,
      SpatialProgressNetCapacityProjection projection) noexcept;

  llvm::Expected<std::vector<SpatialFiniteBufferConflictWitness>>
  finiteBufferConflictWitnesses(const SpatialCandidateState &candidate) const;
  llvm::Error rebuildFiniteBufferConflictWitness(
      const SpatialCandidateState &candidate, PnrIndex owner,
      SpatialFiniteBufferConflictWitness &witness) const;
  llvm::Error rebuildCapacityProofDebtWitness(
      const SpatialCandidateState &candidate, PnrIndex owner,
      SpatialFiniteBufferConflictWitness &witness) const;
  llvm::Error rebuildCapacityShortfallWitness(
      const SpatialCandidateState &candidate, PnrIndex owner,
      SpatialFiniteBufferConflictWitness &witness) const;
  /// Checks the dense incremental representation without reading RouteTrees.
  llvm::Error
  verifyCachedState(const SpatialCandidateState &candidate) const;
  /// Reconstructs progress facts from RouteTrees and runs the independent
  /// closed-wait verifier after checking the cached representation.
  llvm::Error verify(const SpatialCandidateState &candidate) const;

private:
  SpatialProgressState() = default;
  SpatialProgressState(
      const FrozenSpatialPnrProblem &problem, PnrIndex logicalNetCount,
      PnrIndex ownerCount, std::size_t logicalNetWordCount,
      std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netOwnerRefcounts,
      std::vector<PnrIndex> ownerLogicalNetCounts,
      std::vector<std::uint64_t> ownerLogicalNetBits,
      std::vector<std::uint64_t> conflictingOwnerBits,
      std::vector<std::uint64_t> netRouteDependencyViolationCounts)
      : problem_(&problem), logicalNetCount_(logicalNetCount),
        ownerCount_(ownerCount), logicalNetWordCount_(logicalNetWordCount),
        netOwnerRefcounts_(std::move(netOwnerRefcounts)),
        ownerLogicalNetCounts_(std::move(ownerLogicalNetCounts)),
        ownerLogicalNetBits_(std::move(ownerLogicalNetBits)),
        conflictingOwnerBits_(std::move(conflictingOwnerBits)),
        netRouteDependencyViolationCounts_(
            std::move(netRouteDependencyViolationCounts)) {}

  llvm::Expected<std::uint64_t> projectLogicalNetRouteDependencies(
      const SpatialCandidateState &candidate, PnrIndex logicalNet) const;
  llvm::Error applyNetCapacityProjection(
      const SpatialProgressNetCapacityProjection &projection, bool add);
  llvm::Error rebuildCapacityOwnerWitness(
      const SpatialCandidateState &candidate, PnrIndex owner,
      SpatialFiniteBufferConflictWitness &witness) const;
  bool ownerHasCapacityProofDebt(PnrIndex owner) const;
  std::uint64_t ownerCapacityShortfall(PnrIndex owner) const;
  void refreshOwnerCapacityObligation(PnrIndex owner, bool oldDebt,
                                      std::uint64_t oldShortfall,
                                      PnrIndex oldRouteAnchorCount) noexcept;

  const FrozenSpatialPnrProblem *problem_ = nullptr;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex ownerCount_ = 0;
  std::size_t logicalNetWordCount_ = 0;
  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netOwnerRefcounts_;
  std::vector<PnrIndex> ownerLogicalNetCounts_;
  std::vector<std::uint64_t> ownerLogicalNetBits_;
  std::vector<std::uint64_t> conflictingOwnerBits_;
  std::vector<std::uint64_t> netRouteDependencyViolationCounts_;
  std::vector<SpatialProgressNetCapacityProjection> netCapacityProjections_;
  std::vector<std::uint64_t> ownerChannelCounts_;
  std::vector<std::uint64_t> ownerInitializedFeedbackChannelCounts_;
  std::vector<std::uint64_t> ownerRepeatedChannelNetCounts_;
  std::vector<std::uint64_t> ownerIndeterminateQueueClassNetCounts_;
  std::vector<std::map<std::string, PnrIndex>> ownerQueueClassRefcounts_;
  std::vector<PnrIndex> traversalSelectionCounts_;
  std::vector<PnrIndex> ownerRouteAnchorCounts_;
  std::vector<std::uint64_t> capacityProofDebtOwnerBits_;
  std::uint64_t sharedFiniteBufferConflictCount_ = 0;
  std::uint64_t routeDependencyViolationCount_ = 0;
  std::uint64_t capacityProofDebtWitnessCount_ = 0;
  std::uint64_t capacityShortfallOwnerCount_ = 0;
  std::uint64_t capacityShortfall_ = 0;
  std::uint64_t capacityObligationRouteAnchorCount_ = 0;
  bool statisticsEnabled_ = false;
  mutable SpatialProgressStatistics statistics_;

  friend class SpatialCandidateState;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPROGRESSSTATE_H
