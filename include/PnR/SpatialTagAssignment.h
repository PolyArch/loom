#ifndef LOOM_PNR_SPATIALTAGASSIGNMENT_H
#define LOOM_PNR_SPATIALTAGASSIGNMENT_H

#include "Fabric/IR/PhysicalTag.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

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

  friend llvm::Expected<SpatialTagAssignmentProjection>
  deriveCanonicalSpatialTagAssignments(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes);
};

llvm::Expected<SpatialTagAssignmentProjection>
deriveCanonicalSpatialTagAssignments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALTAGASSIGNMENT_H
