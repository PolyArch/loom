#include "PnR/SpatialTagContinuity.h"

#include "Fabric/IR/BoundaryDataPath.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::fabric;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral selectedContinuity =
    "SpatialTagContinuityProjection";
constexpr PnrCapacityContext nodeCountContext{selectedContinuity,
                                              "node_segments", "route_nodes",
                                              PnrCapacityMeasure::Count};
constexpr PnrCapacityContext segmentCountContext{selectedContinuity, "segments",
                                                 "tag_continuity_segments",
                                                 PnrCapacityMeasure::Count};
constexpr PnrCapacityContext segmentIndexContext{selectedContinuity, "segments",
                                                 "tag_continuity_segments",
                                                 PnrCapacityMeasure::Index};
constexpr PnrCapacityContext domainCountContext{
    selectedContinuity, "domain_segment_offsets", "tag_match_domains",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext incidenceCountContext{
    selectedContinuity, "segment_domains", "segment_domain_incidence",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial tag-continuity projection: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

std::optional<std::uint32_t> tagWidth(const ::fabric::DataPathType &path) {
  if (path.kind == ::fabric::DataPathKind::BitsTag)
    return path.tagWidthBits;
  return std::nullopt;
}

/// Shared classification of how one physical arc carries tag continuity from
/// its source endpoint to its target endpoint. All structural validation of
/// the arc against the projection lives here so the whole-tree rebuild and
/// the single-branch extension cannot diverge.
struct TagArcTransition final {
  enum class Kind : std::uint8_t { Inherit, StartSegment, RemoveTag };
  Kind kind = Kind::Inherit;
  PnrIndex pointOrdinal = getInvalidPnrIndex();
  std::uint32_t widthBits = 0;
};

llvm::Expected<TagArcTransition>
classifyTagArcTransition(const FrozenSpatialRoutingGraph &routing,
                         PnrIndex parentEndpoint, PnrIndex childEndpoint,
                         PnrIndex traversal, PnrIndex parentSegment,
                         llvm::ArrayRef<SpatialTagContinuitySegment> segments) {
  const auto endpoints = routing.routingEndpoints();
  const auto traversalPoints = routing.tagContinuity().traversalPointOrdinals();
  const auto points = routing.tagContinuity().points();
  if (parentEndpoint >= endpoints.size() || childEndpoint >= endpoints.size())
    return invalid("a route node names an absent physical endpoint");
  if (traversal >= traversalPoints.size())
    return invalid("a route child disagrees with its physical arc");
  const auto sourceWidth = tagWidth(endpoints[parentEndpoint].dataPath);
  const auto destinationWidth = tagWidth(endpoints[childEndpoint].dataPath);
  if (sourceWidth.has_value() != (parentSegment != getInvalidPnrIndex()))
    return invalid("a tagged route source has no continuity segment");
  if (sourceWidth && (parentSegment >= segments.size() ||
                      segments[parentSegment].tagWidthBits != *sourceWidth))
    return invalid("a route source disagrees with its segment width");

  const PnrIndex pointOrdinal = traversalPoints[traversal];
  TagArcTransition transition;
  if (pointOrdinal == getInvalidPnrIndex()) {
    if (sourceWidth.has_value() != destinationWidth.has_value() ||
        (sourceWidth && *sourceWidth != *destinationWidth))
      return invalid("a non-boundary traversal changes Physical Tag shape");
    transition.kind = TagArcTransition::Kind::Inherit;
    return transition;
  }
  if (pointOrdinal >= points.size())
    return invalid("a route traversal names an absent boundary point");
  const FrozenSpatialTagContinuityPoint &point = points[pointOrdinal];
  switch (point.kind) {
  case FabricBoundaryTagContinuityKind::TokenWriter:
  case FabricBoundaryTagContinuityKind::ConfigurableWriter:
    if (sourceWidth || !destinationWidth ||
        parentSegment != getInvalidPnrIndex() || point.inputTagWidthBits != 0 ||
        point.outputTagWidthBits != *destinationWidth)
      return invalid("a tag writer has inconsistent route endpoints");
    transition.kind = TagArcTransition::Kind::StartSegment;
    transition.pointOrdinal = pointOrdinal;
    transition.widthBits = *destinationWidth;
    return transition;
  case FabricBoundaryTagContinuityKind::Rewriter:
    if (!sourceWidth || !destinationWidth ||
        parentSegment == getInvalidPnrIndex() ||
        point.inputTagWidthBits != *sourceWidth ||
        point.outputTagWidthBits != *destinationWidth)
      return invalid("a tag rewriter has inconsistent route endpoints");
    transition.kind = TagArcTransition::Kind::StartSegment;
    transition.pointOrdinal = pointOrdinal;
    transition.widthBits = *destinationWidth;
    return transition;
  case FabricBoundaryTagContinuityKind::Remover:
    if (!sourceWidth || destinationWidth ||
        parentSegment == getInvalidPnrIndex() ||
        point.inputTagWidthBits != *sourceWidth ||
        point.outputTagWidthBits != 0)
      return invalid("a tag remover has inconsistent route endpoints");
    transition.kind = TagArcTransition::Kind::RemoveTag;
    return transition;
  }
  return invalid("a route traversal names an unknown boundary point kind");
}

} // namespace

/// Sorts and deduplicates the (segment, domain) incidence and derives both
/// CSR directions from it, leaving the incidence in (domain, segment) order.
static llvm::Error rebuildTagIncidenceIndexes(
    std::size_t segmentCount, std::vector<PnrIndex> &segmentDomainOffsets,
    std::vector<PnrIndex> &segmentDomains,
    std::vector<PnrIndex> &domainSegmentOffsets,
    std::vector<PnrIndex> &domainSegments,
    std::vector<std::pair<PnrIndex, PnrIndex>> &incidence,
    std::size_t domainCount) {
  llvm::sort(incidence);
  incidence.erase(std::unique(incidence.begin(), incidence.end()),
                  incidence.end());
  if (llvm::Error error =
          preflightPnrIndexCapacity(incidenceCountContext, incidence.size()))
    return error;

  segmentDomainOffsets.clear();
  segmentDomains.clear();
  segmentDomainOffsets.reserve(segmentCount + 1);
  segmentDomains.reserve(incidence.size());
  auto incidenceIt = incidence.begin();
  for (PnrIndex segment = 0; segment < segmentCount; ++segment) {
    segmentDomainOffsets.push_back(segmentDomains.size());
    while (incidenceIt != incidence.end() && incidenceIt->first == segment) {
      segmentDomains.push_back(incidenceIt->second);
      ++incidenceIt;
    }
  }
  segmentDomainOffsets.push_back(segmentDomains.size());
  if (incidenceIt != incidence.end())
    return invalid("segment/domain incidence is not segment ordered");

  llvm::sort(incidence, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
  });
  domainSegmentOffsets.clear();
  domainSegments.clear();
  domainSegmentOffsets.reserve(domainCount + 1);
  domainSegments.reserve(incidence.size());
  incidenceIt = incidence.begin();
  for (PnrIndex domain = 0; domain < domainCount; ++domain) {
    domainSegmentOffsets.push_back(domainSegments.size());
    while (incidenceIt != incidence.end() && incidenceIt->second == domain) {
      domainSegments.push_back(incidenceIt->first);
      ++incidenceIt;
    }
  }
  domainSegmentOffsets.push_back(domainSegments.size());
  if (incidenceIt != incidence.end())
    return invalid("segment/domain incidence is not domain ordered");
  return llvm::Error::success();
}

llvm::Error loom::pnr::detail::rebuildSpatialTagContinuityUnchecked(
    const RouteTreeState &route, SpatialTagContinuityProjection &result,
    SpatialTagContinuityScratch &scratch) {
  const auto nodes = route.nodeStorage();
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, nodes.size()))
    return error;
  result.segments_.clear();
  result.nodeSegments_.clear();
  result.segmentDomainOffsets_.clear();
  result.segmentDomains_.clear();
  result.domainSegmentOffsets_.clear();
  result.domainSegments_.clear();
  const auto appendSegment =
      [&](SpatialTagContinuityOriginKind originKind, PnrIndex origin,
          std::uint32_t tagWidthBits) -> llvm::Expected<PnrIndex> {
    auto ordinal =
        checkedPnrIndex(segmentIndexContext, result.segments_.size());
    if (!ordinal)
      return ordinal.takeError();
    result.segments_.push_back({originKind, origin, tagWidthBits});
    return *ordinal;
  };
  const FrozenSpatialRoutingGraph &routing = route.routingGraph();
  if (route.isUnrouted()) {
    const auto matchDomains = routing.tagContinuity().matchDomains();
    if (llvm::Error error =
            preflightPnrIndexCapacity(domainCountContext, matchDomains.size()))
      return error;
    result.segmentDomainOffsets_.push_back(0);
    result.domainSegmentOffsets_.assign(matchDomains.size() + 1, 0);
    return llvm::Error::success();
  }

  result.nodeSegments_.assign(nodes.size(), getInvalidPnrIndex());

  const auto endpoints = routing.routingEndpoints();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto traversalPoints = routing.tagContinuity().traversalPointOrdinals();
  const auto points = routing.tagContinuity().points();
  const auto sourceEndpoint = route.sourceEndpoint();
  if (!sourceEndpoint || *sourceEndpoint >= endpoints.size())
    return invalid("a routed tree has no source endpoint");
  const auto root = route.findNode(*sourceEndpoint);
  if (!root || *root >= nodes.size())
    return invalid("a routed tree has no source node");

  if (const auto width = tagWidth(endpoints[*sourceEndpoint].dataPath)) {
    auto segment = appendSegment(SpatialTagContinuityOriginKind::RouteSource,
                                 *sourceEndpoint, *width);
    if (!segment)
      return segment.takeError();
    result.nodeSegments_[*root] = *segment;
  }

  auto &worklist = scratch.worklist_;
  worklist.clear();
  worklist.reserve(route.activeNodeCount());
  worklist.push_back(*root);
  std::size_t visited = 0;
  while (visited != worklist.size()) {
    const PnrIndex parentSlot = worklist[visited++];
    if (parentSlot >= nodes.size() || !nodes[parentSlot].isActive())
      return invalid("the route worklist contains an inactive node");
    const RouteTreeNode &parent = nodes[parentSlot];
    for (PnrIndex childSlot = parent.firstChild;
         childSlot != getInvalidPnrIndex();
         childSlot = nodes[childSlot].nextSibling) {
      if (childSlot >= nodes.size() || !nodes[childSlot].isActive())
        return invalid("a route child is absent from node storage");
      const RouteTreeNode &child = nodes[childSlot];
      if (child.parentArc >= arcs.size() ||
          child.parentArc >= arcSources.size())
        return invalid("a route child has no physical arc");
      const EndpointRoutingArc &arc = arcs[child.parentArc];
      if (arcSources[child.parentArc] != parent.endpoint ||
          arc.target != child.endpoint ||
          arc.traversal >= traversalPoints.size())
        return invalid("a route child disagrees with its physical arc");
      if (parent.endpoint >= endpoints.size() ||
          child.endpoint >= endpoints.size())
        return invalid("a route node names an absent physical endpoint");

      const PnrIndex parentSegment = result.nodeSegments_[parentSlot];
      auto transition = classifyTagArcTransition(
          routing, parent.endpoint, child.endpoint, arc.traversal,
          parentSegment, result.segments_);
      if (!transition)
        return transition.takeError();
      PnrIndex childSegment = getInvalidPnrIndex();
      switch (transition->kind) {
      case TagArcTransition::Kind::Inherit:
        childSegment = parentSegment;
        break;
      case TagArcTransition::Kind::StartSegment: {
        auto segment =
            appendSegment(SpatialTagContinuityOriginKind::BoundaryPoint,
                          transition->pointOrdinal, transition->widthBits);
        if (!segment)
          return segment.takeError();
        childSegment = *segment;
        break;
      }
      case TagArcTransition::Kind::RemoveTag:
        break;
      }
      result.nodeSegments_[childSlot] = childSegment;
      worklist.push_back(childSlot);
    }
  }
  if (visited != route.activeNodeCount())
    return invalid("route traversal did not reach every active node");
  if (llvm::Error error = preflightPnrIndexCapacity(segmentCountContext,
                                                    result.segments_.size()))
    return error;
  auto &order = scratch.order_;
  order.resize(result.segments_.size());
  std::iota(order.begin(), order.end(), PnrIndex{0});
  llvm::sort(order, [&](PnrIndex lhs, PnrIndex rhs) {
    const auto &left = result.segments_[lhs];
    const auto &right = result.segments_[rhs];
    return std::tie(left.originKind, left.origin) <
           std::tie(right.originKind, right.origin);
  });
  auto &remap = scratch.remap_;
  remap.assign(order.size(), getInvalidPnrIndex());
  auto &canonical = scratch.canonicalSegments_;
  canonical.clear();
  canonical.reserve(order.size());
  for (PnrIndex oldOrdinal : order) {
    const auto &segment = result.segments_[oldOrdinal];
    if (!canonical.empty() &&
        canonical.back().originKind == segment.originKind &&
        canonical.back().origin == segment.origin)
      return invalid("one route starts a continuity origin more than once");
    auto newOrdinal = checkedPnrIndex(segmentIndexContext, canonical.size());
    if (!newOrdinal)
      return newOrdinal.takeError();
    remap[oldOrdinal] = *newOrdinal;
    canonical.push_back(segment);
  }
  for (PnrIndex &segment : result.nodeSegments_)
    if (segment != getInvalidPnrIndex()) {
      if (segment >= remap.size() || remap[segment] == getInvalidPnrIndex())
        return invalid("a route node names an absent continuity segment");
      segment = remap[segment];
    }
  result.segments_.assign(canonical.begin(), canonical.end());

  const auto matchDomains = routing.tagContinuity().matchDomains();
  const auto endpointDomains =
      routing.tagContinuity().endpointMatchDomainOrdinals();
  if (endpointDomains.size() != endpoints.size())
    return invalid("the frozen tag match-domain index is not endpoint dense");
  if (llvm::Error error =
          preflightPnrIndexCapacity(domainCountContext, matchDomains.size()))
    return error;

  auto &incidence = scratch.incidence_;
  incidence.clear();
  incidence.reserve(route.activeNodeCount());
  for (auto [slot, node] : llvm::enumerate(nodes)) {
    if (!node.isActive())
      continue;
    if (node.endpoint >= endpointDomains.size())
      return invalid("a route node has no frozen tag match-domain entry");
    const PnrIndex domain = endpointDomains[node.endpoint];
    if (domain == getInvalidPnrIndex())
      continue;
    const PnrIndex segment = result.nodeSegments_[slot];
    if (domain >= matchDomains.size() || segment >= result.segments_.size())
      return invalid("tag match-domain incidence names an absent record");
    if (matchDomains[domain].tagWidthBits !=
        result.segments_[segment].tagWidthBits)
      return invalid("a tag segment intersects a domain of another width");
    incidence.emplace_back(segment, domain);
  }
  return rebuildTagIncidenceIndexes(
      result.segments_.size(), result.segmentDomainOffsets_,
      result.segmentDomains_, result.domainSegmentOffsets_,
      result.domainSegments_, incidence, matchDomains.size());
}

llvm::Expected<bool>
loom::pnr::detail::extendSpatialTagContinuityForBranchUnchecked(
    const RouteTreeState &route, PnrIndex attachmentEndpoint,
    llvm::ArrayRef<PnrIndex> branchArcs, SpatialTagContinuityProjection &result,
    SpatialTagContinuityScratch &scratch) {
  if (branchArcs.empty())
    return true;
  const auto nodes = route.nodeStorage();
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, nodes.size()))
    return error;
  // A projection captured before the route's first branch never assigned the
  // source node a segment; only a whole-tree rebuild can seed it.
  if (result.nodeSegments_.empty())
    return false;

  const FrozenSpatialRoutingGraph &routing = route.routingGraph();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto matchDomains = routing.tagContinuity().matchDomains();
  const auto endpointDomains =
      routing.tagContinuity().endpointMatchDomainOrdinals();
  if (endpointDomains.size() != routing.routingEndpoints().size())
    return invalid("the frozen tag match-domain index is not endpoint dense");
  if (result.nodeSegments_.size() < nodes.size())
    result.nodeSegments_.resize(nodes.size(), getInvalidPnrIndex());

  const auto attachmentSlot = route.findNode(attachmentEndpoint);
  if (!attachmentSlot || *attachmentSlot >= nodes.size())
    return invalid("a branch attachment is absent from its RouteTree");
  PnrIndex parentSlot = *attachmentSlot;
  bool incidenceChanged = false;
  for (PnrIndex arcOrdinal : branchArcs) {
    if (arcOrdinal >= arcs.size() || arcOrdinal >= arcSources.size())
      return invalid("a route child has no physical arc");
    const EndpointRoutingArc &arc = arcs[arcOrdinal];
    if (arcSources[arcOrdinal] != nodes[parentSlot].endpoint)
      return invalid("a route child disagrees with its physical arc");
    const auto childSlot = route.findNode(arc.target);
    if (!childSlot || *childSlot >= nodes.size() ||
        !nodes[*childSlot].isActive())
      return invalid("a branch arc target is absent from its RouteTree");
    const PnrIndex parentSegment = result.nodeSegments_[parentSlot];
    auto transition = classifyTagArcTransition(
        routing, nodes[parentSlot].endpoint, arc.target, arc.traversal,
        parentSegment, result.segments_);
    if (!transition)
      return transition.takeError();
    if (transition->kind == TagArcTransition::Kind::StartSegment)
      return false;
    const PnrIndex childSegment =
        transition->kind == TagArcTransition::Kind::Inherit
            ? parentSegment
            : getInvalidPnrIndex();
    result.nodeSegments_[*childSlot] = childSegment;

    const PnrIndex domain = endpointDomains[arc.target];
    if (domain != getInvalidPnrIndex()) {
      if (domain >= matchDomains.size() ||
          childSegment >= result.segments_.size())
        return invalid("tag match-domain incidence names an absent record");
      if (matchDomains[domain].tagWidthBits !=
          result.segments_[childSegment].tagWidthBits)
        return invalid("a tag segment intersects a domain of another width");
      scratch.incidence_.emplace_back(childSegment, domain);
      incidenceChanged = true;
    }
    parentSlot = *childSlot;
  }
  if (!incidenceChanged)
    return true;
  if (llvm::Error error = rebuildTagIncidenceIndexes(
          result.segments_.size(), result.segmentDomainOffsets_,
          result.segmentDomains_, result.domainSegmentOffsets_,
          result.domainSegments_, scratch.incidence_, matchDomains.size()))
    return std::move(error);
  return true;
}

std::size_t SpatialTagContinuityScratch::retainedStorageBytes() const {
  return worklist_.capacity() * sizeof(PnrIndex) +
         order_.capacity() * sizeof(PnrIndex) +
         remap_.capacity() * sizeof(PnrIndex) +
         canonicalSegments_.capacity() * sizeof(SpatialTagContinuitySegment) +
         incidence_.capacity() * sizeof(std::pair<PnrIndex, PnrIndex>);
}

llvm::Expected<SpatialTagContinuityProjection>
loom::pnr::deriveSpatialTagContinuity(const RouteTreeState &route) {
  if (llvm::Error error = route.verify())
    return error;
  SpatialTagContinuityProjection result;
  SpatialTagContinuityScratch scratch;
  if (llvm::Error error =
          detail::rebuildSpatialTagContinuityUnchecked(route, result, scratch))
    return error;
  return result;
}

llvm::Expected<SpatialTagContinuityProjection>
loom::pnr::deriveSpatialTagContinuity(const RouteTreeTransaction &route) {
  SpatialTagContinuityProjection result;
  SpatialTagContinuityScratch scratch;
  if (llvm::Error error = rebuildSpatialTagContinuity(route, result, scratch))
    return error;
  return result;
}

llvm::Error
loom::pnr::rebuildSpatialTagContinuity(const RouteTreeTransaction &route,
                                       SpatialTagContinuityProjection &result,
                                       SpatialTagContinuityScratch &scratch) {
  auto prepared = route.preparedState();
  if (!prepared)
    return prepared.takeError();
  return detail::rebuildSpatialTagContinuityUnchecked(**prepared, result,
                                                      scratch);
}
