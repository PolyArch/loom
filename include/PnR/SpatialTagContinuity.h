#ifndef LOOM_PNR_SPATIALTAGCONTINUITY_H
#define LOOM_PNR_SPATIALTAGCONTINUITY_H

#include "PnR/RouteTreeState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::pnr {

class SpatialTagContinuityProjection;
class SpatialTagContinuityScratch;

namespace detail {
llvm::Error
rebuildSpatialTagContinuityUnchecked(const RouteTreeState &route,
                                     SpatialTagContinuityProjection &result,
                                     SpatialTagContinuityScratch &scratch);
/// Extends a projection that `rebuildSpatialTagContinuityUnchecked` (or a
/// previous successful extension) produced for this route immediately before
/// exactly one branch of `branchArcs` was attached at `attachmentEndpoint`,
/// with no other route change in between. Returns false without a usable
/// result when the branch starts a new continuity segment or the projection
/// predates the route's first branch; the caller must then rebuild. On
/// success the projection and scratch are byte-identical to a full rebuild.
llvm::Expected<bool> extendSpatialTagContinuityForBranchUnchecked(
    const RouteTreeState &route, PnrIndex attachmentEndpoint,
    llvm::ArrayRef<PnrIndex> branchArcs, SpatialTagContinuityProjection &result,
    SpatialTagContinuityScratch &scratch);
} // namespace detail

enum class SpatialTagContinuityOriginKind : std::uint8_t {
  RouteSource,
  BoundaryPoint,
};

/// One maximal tagged segment of an exact selected RouteTree. `origin` is a
/// routing-endpoint ordinal for RouteSource and a frozen continuity-point
/// ordinal for BoundaryPoint. Both mechanically recover their exact Fabric
/// owner from the route's FrozenSpatialRoutingGraph.
struct SpatialTagContinuitySegment final {
  SpatialTagContinuityOriginKind originKind =
      SpatialTagContinuityOriginKind::RouteSource;
  PnrIndex origin = getInvalidPnrIndex();
  std::uint32_t tagWidthBits = 0;

  friend bool operator==(const SpatialTagContinuitySegment &lhs,
                         const SpatialTagContinuitySegment &rhs) {
    return lhs.originKind == rhs.originKind && lhs.origin == rhs.origin &&
           lhs.tagWidthBits == rhs.tagWidthBits;
  }
};

/// Reusable work storage for deriving exact route-local tag continuity. One
/// instance may be shared sequentially across nets in a worker-local move.
class SpatialTagContinuityScratch final {
public:
  SpatialTagContinuityScratch() = default;
  SpatialTagContinuityScratch(const SpatialTagContinuityScratch &) = delete;
  SpatialTagContinuityScratch &
  operator=(const SpatialTagContinuityScratch &) = delete;
  SpatialTagContinuityScratch(SpatialTagContinuityScratch &&) = delete;
  SpatialTagContinuityScratch &
  operator=(SpatialTagContinuityScratch &&) = delete;
  ~SpatialTagContinuityScratch() = default;

  std::size_t retainedStorageBytes() const;

private:
  std::vector<PnrIndex> worklist_;
  std::vector<PnrIndex> order_;
  std::vector<PnrIndex> remap_;
  std::vector<SpatialTagContinuitySegment> canonicalSegments_;
  std::vector<std::pair<PnrIndex, PnrIndex>> incidence_;

  friend llvm::Error detail::rebuildSpatialTagContinuityUnchecked(
      const RouteTreeState &route, SpatialTagContinuityProjection &result,
      SpatialTagContinuityScratch &scratch);
  friend llvm::Expected<bool> detail::extendSpatialTagContinuityForBranchUnchecked(
      const RouteTreeState &route, PnrIndex attachmentEndpoint,
      llvm::ArrayRef<PnrIndex> branchArcs,
      SpatialTagContinuityProjection &result,
      SpatialTagContinuityScratch &scratch);
};

/// Cold, removable projection of one selected route. The node array is dense
/// in RouteTreeState::nodeStorage(); inactive and untagged nodes carry
/// getInvalidPnrIndex(). The two CSR relations describe the same deduplicated
/// segment/domain incidence in both directions. Search-time tag state may
/// cache this relation but may not reinterpret its Fabric-owned boundaries or
/// local match domains.
class SpatialTagContinuityProjection final {
public:
  llvm::ArrayRef<SpatialTagContinuitySegment> segments() const {
    return segments_;
  }
  llvm::ArrayRef<PnrIndex> nodeSegments() const { return nodeSegments_; }
  llvm::ArrayRef<PnrIndex> segmentDomainOffsets() const {
    return segmentDomainOffsets_;
  }
  llvm::ArrayRef<PnrIndex> segmentDomains() const { return segmentDomains_; }
  llvm::ArrayRef<PnrIndex> domainSegmentOffsets() const {
    return domainSegmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> domainSegments() const { return domainSegments_; }
  std::size_t retainedStorageBytes() const {
    return segments_.capacity() * sizeof(SpatialTagContinuitySegment) +
           nodeSegments_.capacity() * sizeof(PnrIndex) +
           segmentDomainOffsets_.capacity() * sizeof(PnrIndex) +
           segmentDomains_.capacity() * sizeof(PnrIndex) +
           domainSegmentOffsets_.capacity() * sizeof(PnrIndex) +
           domainSegments_.capacity() * sizeof(PnrIndex);
  }

private:
  std::vector<SpatialTagContinuitySegment> segments_;
  std::vector<PnrIndex> nodeSegments_;
  std::vector<PnrIndex> segmentDomainOffsets_;
  std::vector<PnrIndex> segmentDomains_;
  std::vector<PnrIndex> domainSegmentOffsets_;
  std::vector<PnrIndex> domainSegments_;

  friend llvm::Error detail::rebuildSpatialTagContinuityUnchecked(
      const RouteTreeState &route, SpatialTagContinuityProjection &result,
      SpatialTagContinuityScratch &scratch);
  friend llvm::Expected<bool> detail::extendSpatialTagContinuityForBranchUnchecked(
      const RouteTreeState &route, PnrIndex attachmentEndpoint,
      llvm::ArrayRef<PnrIndex> branchArcs,
      SpatialTagContinuityProjection &result,
      SpatialTagContinuityScratch &scratch);
};

llvm::Expected<SpatialTagContinuityProjection>
deriveSpatialTagContinuity(const RouteTreeState &route);
llvm::Expected<SpatialTagContinuityProjection>
deriveSpatialTagContinuity(const RouteTreeTransaction &route);
llvm::Error rebuildSpatialTagContinuity(const RouteTreeTransaction &route,
                                        SpatialTagContinuityProjection &result,
                                        SpatialTagContinuityScratch &scratch);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALTAGCONTINUITY_H
