#ifndef LOOM_PNR_SPATIALPROGRESSSTATE_H
#define LOOM_PNR_SPATIALPROGRESSSTATE_H

#include "Fabric/Identity/FabricRefs.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
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
  /// Shared finite-buffer residency is a typed unproven-recurrence fact, not
  /// a Spatial violation: the ordinary Mapping remains importable while the
  /// unestablished recurrence withholds Dataflow spectrum qualification, and
  /// the router prices foreign residency as negotiable congestion. Only route
  /// dependency violations remain hard progress violations.
  std::uint64_t hardProgressViolation() const {
    return routeDependencyViolationCount_;
  }
  PnrIndex finiteBufferOwnerLogicalNetCount(PnrIndex owner) const;
  bool finiteBufferOwnerConflicts(PnrIndex owner) const;
  std::optional<PnrIndex> firstFiniteBufferConflictOwner() const;
  llvm::Error
  enumerateFiniteBufferConflictOwners(std::vector<PnrIndex> &owners) const;
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

  llvm::Expected<std::vector<SpatialFiniteBufferConflictWitness>>
  finiteBufferConflictWitnesses(const SpatialCandidateState &candidate) const;
  llvm::Error rebuildFiniteBufferConflictWitness(
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

  const FrozenSpatialPnrProblem *problem_ = nullptr;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex ownerCount_ = 0;
  std::size_t logicalNetWordCount_ = 0;
  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netOwnerRefcounts_;
  std::vector<PnrIndex> ownerLogicalNetCounts_;
  std::vector<std::uint64_t> ownerLogicalNetBits_;
  std::vector<std::uint64_t> conflictingOwnerBits_;
  std::vector<std::uint64_t> netRouteDependencyViolationCounts_;
  std::uint64_t sharedFiniteBufferConflictCount_ = 0;
  std::uint64_t routeDependencyViolationCount_ = 0;
  bool statisticsEnabled_ = false;
  mutable SpatialProgressStatistics statistics_;

  friend class SpatialCandidateState;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPROGRESSSTATE_H
