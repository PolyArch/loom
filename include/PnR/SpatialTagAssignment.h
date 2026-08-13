#ifndef LOOM_PNR_SPATIALTAGASSIGNMENT_H
#define LOOM_PNR_SPATIALTAGASSIGNMENT_H

#include "Fabric/IR/PhysicalTag.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::pnr {

class SpatialCandidateState;
class SpatialMoveTransaction;

namespace detail {
struct SpatialTagAssignmentScratchStorage;
struct SpatialTagAssignmentStateStorage;
} // namespace detail

/// Canonical, removable Physical Tag assignment for exact selected routes.
/// Segment ordinals are grouped by canonical logical-net order. The two CSR
/// relations retain local Fabric match domains without materializing pairwise
/// interference edges or a dense 2^tag_width namespace.
class SpatialTagAssignmentProjection final {
public:
  llvm::ArrayRef<PnrIndex> netSegmentOffsets() const {
    return netSegmentOffsets_;
  }
  llvm::ArrayRef<SpatialTagContinuitySegment> segments() const {
    return segments_;
  }
  llvm::ArrayRef<std::optional<llvm::APInt>> values() const { return values_; }
  llvm::ArrayRef<PnrIndex> segmentDomainOffsets() const {
    return segmentDomainOffsets_;
  }
  llvm::ArrayRef<PnrIndex> segmentDomains() const { return segmentDomains_; }
  llvm::ArrayRef<PnrIndex> domainSegmentOffsets() const {
    return domainSegmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> domainSegments() const { return domainSegments_; }
  llvm::ArrayRef<PnrIndex> domainSegments(PnrIndex domain) const;
  std::uint64_t unassignedCount() const { return unassignedCount_; }
  std::uint64_t conflictCount() const { return conflictCount_; }
  std::uint64_t residentCapacityOveruse() const {
    return residentCapacityOveruse_;
  }

private:
  std::vector<PnrIndex> netSegmentOffsets_;
  std::vector<SpatialTagContinuitySegment> segments_;
  std::vector<std::optional<llvm::APInt>> values_;
  std::vector<PnrIndex> segmentDomainOffsets_;
  std::vector<PnrIndex> segmentDomains_;
  std::vector<PnrIndex> domainSegmentOffsets_;
  std::vector<PnrIndex> domainSegments_;
  std::uint64_t unassignedCount_ = 0;
  std::uint64_t conflictCount_ = 0;
  std::uint64_t residentCapacityOveruse_ = 0;

  friend llvm::Expected<SpatialTagAssignmentProjection>
  deriveCanonicalSpatialTagAssignments(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes);
};

llvm::Expected<SpatialTagAssignmentProjection>
deriveCanonicalSpatialTagAssignments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes);

struct SpatialTagAssignmentSummary final {
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
  std::uint64_t residentCapacityOveruse = 0;
};

/// Reusable transaction storage for route-local Physical Tag updates. The
/// storage is prepared once per Frozen model and retains prior net buffers so
/// repeated route moves can reuse capacity.
class SpatialTagAssignmentScratch final {
public:
  SpatialTagAssignmentScratch();
  SpatialTagAssignmentScratch(const SpatialTagAssignmentScratch &) = delete;
  SpatialTagAssignmentScratch &
  operator=(const SpatialTagAssignmentScratch &) = delete;
  SpatialTagAssignmentScratch(SpatialTagAssignmentScratch &&) = delete;
  SpatialTagAssignmentScratch &
  operator=(SpatialTagAssignmentScratch &&) = delete;
  ~SpatialTagAssignmentScratch();

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);
  std::size_t retainedStorageBytes() const;

private:
  std::unique_ptr<detail::SpatialTagAssignmentScratchStorage> storage_;

  friend class SpatialTagAssignmentState;
};

/// Candidate-owned Physical Tag decisions plus rebuildable route-continuity
/// and local-domain occupancy caches. Values are decisions; segment and
/// occupancy tables are exact removable projections of routes and values.
class SpatialTagAssignmentState final {
public:
  SpatialTagAssignmentState(SpatialTagAssignmentState &&) noexcept;
  SpatialTagAssignmentState(const SpatialTagAssignmentState &) = delete;
  SpatialTagAssignmentState &
  operator=(const SpatialTagAssignmentState &) = delete;
  SpatialTagAssignmentState &operator=(SpatialTagAssignmentState &&) = delete;
  ~SpatialTagAssignmentState();

  llvm::ArrayRef<SpatialTagContinuitySegment>
  segments(PnrIndex logicalNet) const;
  llvm::ArrayRef<std::optional<llvm::APInt>> values(PnrIndex logicalNet) const;
  llvm::ArrayRef<PnrIndex> segmentDomains(PnrIndex logicalNet,
                                          PnrIndex segment) const;
  std::uint64_t unassignedCount() const;
  std::uint64_t conflictCount() const;
  std::uint64_t domainConflictCount(PnrIndex domain) const;
  std::uint64_t residentCapacityOveruse() const;
  std::uint64_t domainResidentCapacityOveruse(PnrIndex domain) const;
  bool domainValueConflicts(PnrIndex domain, const llvm::APInt &value) const;

private:
  explicit SpatialTagAssignmentState(
      std::unique_ptr<detail::SpatialTagAssignmentStateStorage> storage);

  static llvm::Expected<SpatialTagAssignmentState>
  create(const FrozenSpatialPnrProblem &problem,
         llvm::ArrayRef<RouteTreeStateHandle> routes);
  llvm::Expected<SpatialTagAssignmentSummary>
  projectVerifiedRoutes(llvm::ArrayRef<const RouteTreeState *> routes) const;
  llvm::Error verify(llvm::ArrayRef<RouteTreeStateHandle> routes) const;
  llvm::Error stageRouteUpdates(
      llvm::ArrayRef<RouteTreeStateHandle> routes,
      llvm::ArrayRef<std::optional<RouteTreeTransaction>> routeTransactions,
      llvm::ArrayRef<PnrIndex> touchedRoutes,
      SpatialTagAssignmentScratch &scratch);
  void commit(SpatialTagAssignmentScratch &scratch) noexcept;
  void rollback(SpatialTagAssignmentScratch &scratch) noexcept;

  std::unique_ptr<detail::SpatialTagAssignmentStateStorage> storage_;

  friend class SpatialCandidateState;
  friend class SpatialMoveTransaction;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALTAGASSIGNMENT_H
