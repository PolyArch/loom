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

} // namespace

llvm::Expected<SpatialTagContinuityProjection>
loom::pnr::deriveSpatialTagContinuity(const RouteTreeState &route) {
  if (llvm::Error error = route.verify())
    return std::move(error);
  const auto nodes = route.nodeStorage();
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, nodes.size()))
    return std::move(error);
  SpatialTagContinuityProjection result;
  result.nodeSegments_.assign(nodes.size(), getInvalidPnrIndex());
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
      return std::move(error);
    result.segmentDomainOffsets_.push_back(0);
    result.domainSegmentOffsets_.assign(matchDomains.size() + 1, 0);
    return result;
  }

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

  std::vector<PnrIndex> worklist;
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
      const FrozenSpatialRoutingArc &arc = arcs[child.parentArc];
      if (arcSources[child.parentArc] != parent.endpoint ||
          arc.target != child.endpoint ||
          arc.traversal >= traversalPoints.size())
        return invalid("a route child disagrees with its physical arc");
      if (parent.endpoint >= endpoints.size() ||
          child.endpoint >= endpoints.size())
        return invalid("a route node names an absent physical endpoint");

      const auto sourceWidth = tagWidth(endpoints[parent.endpoint].dataPath);
      const auto destinationWidth =
          tagWidth(endpoints[child.endpoint].dataPath);
      const PnrIndex parentSegment = result.nodeSegments_[parentSlot];
      if (sourceWidth.has_value() != (parentSegment != getInvalidPnrIndex()))
        return invalid("a tagged route source has no continuity segment");
      if (sourceWidth &&
          (parentSegment >= result.segments_.size() ||
           result.segments_[parentSegment].tagWidthBits != *sourceWidth))
        return invalid("a route source disagrees with its segment width");

      const PnrIndex pointOrdinal = traversalPoints[arc.traversal];
      PnrIndex childSegment = getInvalidPnrIndex();
      if (pointOrdinal == getInvalidPnrIndex()) {
        if (sourceWidth.has_value() != destinationWidth.has_value() ||
            (sourceWidth && *sourceWidth != *destinationWidth))
          return invalid("a non-boundary traversal changes Physical Tag shape");
        childSegment = parentSegment;
      } else {
        if (pointOrdinal >= points.size())
          return invalid("a route traversal names an absent boundary point");
        const FrozenSpatialTagContinuityPoint &point = points[pointOrdinal];
        switch (point.kind) {
        case FabricBoundaryTagContinuityKind::TokenWriter:
        case FabricBoundaryTagContinuityKind::ConfigurableWriter: {
          if (sourceWidth || !destinationWidth ||
              parentSegment != getInvalidPnrIndex() ||
              point.inputTagWidthBits != 0 ||
              point.outputTagWidthBits != *destinationWidth)
            return invalid("a tag writer has inconsistent route endpoints");
          auto segment =
              appendSegment(SpatialTagContinuityOriginKind::BoundaryPoint,
                            pointOrdinal, *destinationWidth);
          if (!segment)
            return segment.takeError();
          childSegment = *segment;
          break;
        }
        case FabricBoundaryTagContinuityKind::Rewriter: {
          if (!sourceWidth || !destinationWidth ||
              parentSegment == getInvalidPnrIndex() ||
              point.inputTagWidthBits != *sourceWidth ||
              point.outputTagWidthBits != *destinationWidth)
            return invalid("a tag rewriter has inconsistent route endpoints");
          auto segment =
              appendSegment(SpatialTagContinuityOriginKind::BoundaryPoint,
                            pointOrdinal, *destinationWidth);
          if (!segment)
            return segment.takeError();
          childSegment = *segment;
          break;
        }
        case FabricBoundaryTagContinuityKind::Remover:
          if (!sourceWidth || destinationWidth ||
              parentSegment == getInvalidPnrIndex() ||
              point.inputTagWidthBits != *sourceWidth ||
              point.outputTagWidthBits != 0)
            return invalid("a tag remover has inconsistent route endpoints");
          break;
        }
      }
      result.nodeSegments_[childSlot] = childSegment;
      worklist.push_back(childSlot);
    }
  }
  if (visited != route.activeNodeCount())
    return invalid("route traversal did not reach every active node");
  if (llvm::Error error = preflightPnrIndexCapacity(segmentCountContext,
                                                    result.segments_.size()))
    return std::move(error);
  std::vector<PnrIndex> order(result.segments_.size());
  std::iota(order.begin(), order.end(), PnrIndex{0});
  llvm::sort(order, [&](PnrIndex lhs, PnrIndex rhs) {
    const auto &left = result.segments_[lhs];
    const auto &right = result.segments_[rhs];
    return std::tie(left.originKind, left.origin) <
           std::tie(right.originKind, right.origin);
  });
  std::vector<PnrIndex> remap(order.size(), getInvalidPnrIndex());
  std::vector<SpatialTagContinuitySegment> canonical;
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
  result.segments_ = std::move(canonical);

  const auto matchDomains = routing.tagContinuity().matchDomains();
  const auto endpointDomains =
      routing.tagContinuity().endpointMatchDomainOrdinals();
  if (endpointDomains.size() != endpoints.size())
    return invalid("the frozen tag match-domain index is not endpoint dense");
  if (llvm::Error error =
          preflightPnrIndexCapacity(domainCountContext, matchDomains.size()))
    return std::move(error);

  std::vector<std::pair<PnrIndex, PnrIndex>> incidence;
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
  llvm::sort(incidence);
  incidence.erase(std::unique(incidence.begin(), incidence.end()),
                  incidence.end());
  if (llvm::Error error =
          preflightPnrIndexCapacity(incidenceCountContext, incidence.size()))
    return std::move(error);

  result.segmentDomainOffsets_.reserve(result.segments_.size() + 1);
  result.segmentDomains_.reserve(incidence.size());
  auto incidenceIt = incidence.begin();
  for (PnrIndex segment = 0; segment < result.segments_.size(); ++segment) {
    result.segmentDomainOffsets_.push_back(result.segmentDomains_.size());
    while (incidenceIt != incidence.end() && incidenceIt->first == segment) {
      result.segmentDomains_.push_back(incidenceIt->second);
      ++incidenceIt;
    }
  }
  result.segmentDomainOffsets_.push_back(result.segmentDomains_.size());
  if (incidenceIt != incidence.end())
    return invalid("segment/domain incidence is not segment ordered");

  llvm::sort(incidence, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
  });
  result.domainSegmentOffsets_.reserve(matchDomains.size() + 1);
  result.domainSegments_.reserve(incidence.size());
  incidenceIt = incidence.begin();
  for (PnrIndex domain = 0; domain < matchDomains.size(); ++domain) {
    result.domainSegmentOffsets_.push_back(result.domainSegments_.size());
    while (incidenceIt != incidence.end() && incidenceIt->second == domain) {
      result.domainSegments_.push_back(incidenceIt->first);
      ++incidenceIt;
    }
  }
  result.domainSegmentOffsets_.push_back(result.domainSegments_.size());
  if (incidenceIt != incidence.end())
    return invalid("segment/domain incidence is not domain ordered");
  return result;
}
