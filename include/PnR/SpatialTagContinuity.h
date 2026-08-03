#ifndef LOOM_PNR_SPATIALTAGCONTINUITY_H
#define LOOM_PNR_SPATIALTAGCONTINUITY_H

#include "PnR/RouteTreeState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr {

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

/// Cold, removable projection of one selected route. The node array is dense
/// in RouteTreeState::nodeStorage(); inactive and untagged nodes carry
/// getInvalidPnrIndex(). Search-time tag state may cache this relation but may
/// not reinterpret its Fabric-owned boundary transitions.
class SpatialTagContinuityProjection final {
public:
  llvm::ArrayRef<SpatialTagContinuitySegment> segments() const {
    return segments_;
  }
  llvm::ArrayRef<PnrIndex> nodeSegments() const { return nodeSegments_; }

private:
  std::vector<SpatialTagContinuitySegment> segments_;
  std::vector<PnrIndex> nodeSegments_;

  friend llvm::Expected<SpatialTagContinuityProjection>
  deriveSpatialTagContinuity(const RouteTreeState &route);
};

llvm::Expected<SpatialTagContinuityProjection>
deriveSpatialTagContinuity(const RouteTreeState &route);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALTAGCONTINUITY_H
