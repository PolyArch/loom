#include "SpatialSwitchRowPacking.h"

#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial Temporal switch row demand: " + message);
}

struct DemandKey final {
  PnrIndex domain = 0;
  PnrIndex logicalNet = 0;
  PnrIndex segment = 0;

  friend bool operator<(DemandKey lhs, DemandKey rhs) {
    return std::tie(lhs.domain, lhs.logicalNet, lhs.segment) <
           std::tie(rhs.domain, rhs.logicalNet, rhs.segment);
  }
};

struct SelectedCrosspoint final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  ::loom::fabric::FabricOrdinal output = 0;
  PnrIndex traversal = 0;
};

using InputOutputs =
    std::map<::loom::fabric::FabricOrdinal, std::vector<SelectedCrosspoint>>;

llvm::Expected<PnrIndex> checkedIndex(std::size_t value,
                                      llvm::StringRef description) {
  if (value >= getInvalidPnrIndex())
    return invalid(description + " exceeds PnrIndex");
  return static_cast<PnrIndex>(value);
}

llvm::Error appendRouteDemands(const FrozenSpatialPnrProblem &problem,
                               PnrIndex logicalNet, const RouteTreeState &route,
                               const SpatialTagContinuityProjection &continuity,
                               std::map<DemandKey, InputOutputs> &selected) {
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  if (&route.routingGraph() != &routing)
    return invalid("route belongs to another frozen problem");
  if (route.isUnrouted())
    return llvm::Error::success();
  const auto nodes = route.nodeStorage();
  const auto nodeSegments = continuity.nodeSegments();
  if (nodeSegments.size() != nodes.size())
    return invalid("route and continuity node inventories disagree");
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto traversals = routing.traversals();
  const auto endpoints = routing.routingEndpoints();
  const auto endpointDomains =
      routing.tagContinuity().endpointMatchDomainOrdinals();
  const auto domains = routing.tagContinuity().matchDomains();
  if (endpointDomains.size() != endpoints.size())
    return invalid("endpoint match-domain index has the wrong width");
  for (auto [slot, node] : llvm::enumerate(nodes)) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= arcs.size() || node.parentArc >= arcSources.size())
      return invalid("route node names an absent physical arc");
    const PnrIndex traversal = arcs[node.parentArc].traversal;
    if (traversal >= traversals.size())
      return invalid("route node names an absent physical traversal");
    const auto *payload =
        std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
            &traversals[traversal].reference.payload);
    if (!payload)
      continue;
    const PnrIndex source = arcSources[node.parentArc];
    if (source >= endpoints.size() || source >= endpointDomains.size())
      return invalid("switch traversal source is out of range");
    const PnrIndex domain = endpointDomains[source];
    if (domain == getInvalidPnrIndex())
      continue;
    if (domain >= domains.size() ||
        domains[domain].kind !=
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable ||
        endpoints[source].reference.owner !=
            ::loom::fabric::FabricTransportEndpointOwnerRef::of(
                payload->owner) ||
        endpoints[source].reference.ordinal != payload->input)
      return invalid("switch traversal disagrees with its match domain");
    const PnrIndex segment = nodeSegments[slot];
    if (segment == getInvalidPnrIndex() ||
        segment >= continuity.segments().size())
      return invalid("Temporal switch traversal has no tag segment");
    selected[{domain, logicalNet, segment}][payload->input].push_back(
        {payload->owner, payload->output, traversal});
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
materializeDemands(std::map<DemandKey, InputOutputs> selected) {
  std::vector<SpatialTemporalSwitchSegmentDemand> result;
  result.reserve(selected.size());
  for (auto &[key, inputs] : selected) {
    SpatialTemporalSwitchSegmentDemand demand{
        key.domain, key.logicalNet, key.segment, {}};
    demand.signatures.reserve(inputs.size());
    for (auto &[input, crosspoints] : inputs) {
      llvm::sort(crosspoints, [](const auto &lhs, const auto &rhs) {
        return std::tie(lhs.output, lhs.traversal) <
               std::tie(rhs.output, rhs.traversal);
      });
      if (std::adjacent_find(crosspoints.begin(), crosspoints.end(),
                             [](const auto &lhs, const auto &rhs) {
                               return lhs.output == rhs.output;
                             }) != crosspoints.end())
        return invalid("one segment repeats a switch crosspoint");
      SpatialTemporalSwitchInputSignature signature{
          crosspoints.front().occurrence, input, {}, {}};
      signature.outputs.reserve(crosspoints.size());
      signature.traversals.reserve(crosspoints.size());
      for (const SelectedCrosspoint &crosspoint : crosspoints) {
        if (crosspoint.occurrence != signature.occurrence)
          return invalid("one input signature crosses switch occurrences");
        signature.outputs.push_back(crosspoint.output);
        signature.traversals.push_back(crosspoint.traversal);
      }
      demand.signatures.push_back(std::move(signature));
    }
    std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView> views;
    views.reserve(demand.signatures.size());
    for (const SpatialTemporalSwitchInputSignature &signature :
         demand.signatures)
      views.push_back(
          {signature.occurrence, signature.input, signature.outputs});
    if (llvm::Error error =
            ::loom::fabric::validateFabricTemporalSwitchRouteDemand({views}))
      return std::move(error);
    result.push_back(std::move(demand));
  }
  return result;
}

llvm::Error
verifyDemandCoverage(const FrozenSpatialPnrProblem &problem,
                     PnrIndex logicalNet,
                     const SpatialTagContinuityProjection &continuity,
                     const std::map<DemandKey, InputOutputs> &selected) {
  const auto domains = problem.routing().tagContinuity().matchDomains();
  const auto offsets = continuity.segmentDomainOffsets();
  const auto segmentDomains = continuity.segmentDomains();
  std::size_t observed = 0;
  for (PnrIndex segment = 0; segment < continuity.segments().size(); ++segment)
    for (PnrIndex incidence = offsets[segment];
         incidence < offsets[segment + 1]; ++incidence) {
      const PnrIndex domain = segmentDomains[incidence];
      if (domain >= domains.size())
        return invalid("continuity segment names an absent match domain");
      if (domains[domain].kind !=
          ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
        continue;
      if (selected.find({domain, logicalNet, segment}) == selected.end())
        return invalid("switch-domain incidence has no route signature");
      ++observed;
    }
  const std::size_t selectedForNet =
      llvm::count_if(selected, [&](const auto &entry) {
        return entry.first.logicalNet == logicalNet;
      });
  if (observed != selectedForNet)
    return invalid("switch row demand inventory is not canonical");
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<PnrIndex>
SpatialTagInterferenceProjection::conflicts(PnrIndex vertex) const {
  assert(vertex + 1 < conflictOffsets_.size());
  return llvm::ArrayRef(conflicts_)
      .slice(conflictOffsets_[vertex],
             conflictOffsets_[vertex + 1] - conflictOffsets_[vertex]);
}

bool SpatialTagInterferenceProjection::interferes(PnrIndex lhs,
                                                  PnrIndex rhs) const {
  if (lhs == rhs || lhs + 1 >= conflictOffsets_.size() ||
      rhs + 1 >= conflictOffsets_.size())
    return false;
  return std::binary_search(conflicts(lhs).begin(), conflicts(lhs).end(), rhs);
}

bool SpatialTagInterferenceProjection::interferes(PnrIndex domain, PnrIndex lhs,
                                                  PnrIndex rhs) const {
  if (lhs == rhs || domain >= domainConflicts_.size() ||
      lhs >= vertexRefs_.size() || rhs >= vertexRefs_.size())
    return false;
  return interferes(domain, vertexRefs_[lhs], vertexRefs_[rhs]);
}

bool SpatialTagInterferenceProjection::interferes(
    SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) const {
  if (lhs == rhs)
    return false;
  SpatialTagConflictPair pair{lhs, rhs};
  if (pair.rhs < pair.lhs)
    std::swap(pair.lhs, pair.rhs);
  return std::binary_search(globalConflicts_.begin(), globalConflicts_.end(),
                            pair);
}

bool SpatialTagInterferenceProjection::interferes(
    PnrIndex domain, SpatialTagVertexRef lhs, SpatialTagVertexRef rhs) const {
  if (lhs == rhs || domain >= domainConflicts_.size())
    return false;
  SpatialTagConflictPair pair{lhs, rhs};
  if (pair.rhs < pair.lhs)
    std::swap(pair.lhs, pair.rhs);
  return std::binary_search(domainConflicts_[domain].begin(),
                            domainConflicts_[domain].end(), pair);
}

SpatialTagVertexRef
SpatialTagInterferenceProjection::vertexRef(PnrIndex vertex) const {
  assert(vertex < vertexRefs_.size());
  return vertexRefs_[vertex];
}

PnrIndex SpatialTagInterferenceProjection::vertexOrdinal(
    SpatialTagVertexRef vertex) const {
  const auto found = llvm::lower_bound(vertexRefs_, vertex);
  if (found == vertexRefs_.end() || !(*found == vertex))
    return getInvalidPnrIndex();
  return static_cast<PnrIndex>(found - vertexRefs_.begin());
}

bool SpatialTagInterferenceProjection::equivalentDerivedState(
    const SpatialTagInterferenceProjection &other) const {
  return domainVertices_ == other.domainVertices_ &&
         domainConflicts_ == other.domainConflicts_ &&
         netDomains_ == other.netDomains_ &&
         netSwitchDemands_ == other.netSwitchDemands_;
}

std::size_t SpatialTagInterferenceProjection::retainedStorageBytes() const {
  std::size_t bytes =
      netSegmentOffsets_.capacity() * sizeof(PnrIndex) +
      vertexRefs_.capacity() * sizeof(SpatialTagVertexRef) +
      conflictOffsets_.capacity() * sizeof(PnrIndex) +
      conflicts_.capacity() * sizeof(PnrIndex) +
      globalConflicts_.capacity() * sizeof(SpatialTagConflictPair) +
      domainVertices_.capacity() * sizeof(std::vector<SpatialTagVertexRef>) +
      domainConflicts_.capacity() *
          sizeof(std::vector<SpatialTagConflictPair>) +
      netDomains_.capacity() * sizeof(std::vector<PnrIndex>) +
      netSwitchDemands_.capacity() *
          sizeof(std::vector<SpatialTemporalSwitchSegmentDemand>);
  for (const auto &vertices : domainVertices_)
    bytes += vertices.capacity() * sizeof(SpatialTagVertexRef);
  for (const auto &conflicts : domainConflicts_)
    bytes += conflicts.capacity() * sizeof(SpatialTagConflictPair);
  for (const auto &domains : netDomains_)
    bytes += domains.capacity() * sizeof(PnrIndex);
  for (const auto &demands : netSwitchDemands_) {
    bytes += demands.capacity() * sizeof(SpatialTemporalSwitchSegmentDemand);
    for (const SpatialTemporalSwitchSegmentDemand &demand : demands) {
      bytes += demand.signatures.capacity() *
               sizeof(SpatialTemporalSwitchInputSignature);
      for (const SpatialTemporalSwitchInputSignature &signature :
           demand.signatures)
        bytes += signature.outputs.capacity() *
                     sizeof(::loom::fabric::FabricOrdinal) +
                 signature.traversals.capacity() * sizeof(PnrIndex);
    }
  }
  return bytes;
}

std::size_t SpatialTagInterferenceUpdateScratch::retainedStorageBytes() const {
  std::size_t bytes =
      previousNetSegmentOffsets_.capacity() * sizeof(PnrIndex) +
      previousVertexRefs_.capacity() * sizeof(SpatialTagVertexRef) +
      previousConflictOffsets_.capacity() * sizeof(PnrIndex) +
      previousConflicts_.capacity() * sizeof(PnrIndex) +
      previousGlobalConflicts_.capacity() * sizeof(SpatialTagConflictPair) +
      affectedDomains_.capacity() * sizeof(PnrIndex) +
      domainDeltas_.capacity() * sizeof(DomainDelta) +
      netDemandDeltas_.capacity() * sizeof(NetDemandDelta);
  for (const DomainDelta &delta : domainDeltas_)
    bytes += delta.vertices.capacity() * sizeof(SpatialTagVertexRef) +
             delta.conflicts.capacity() * sizeof(SpatialTagConflictPair);
  for (const NetDemandDelta &delta : netDemandDeltas_) {
    bytes +=
        delta.domains.capacity() * sizeof(PnrIndex) +
        delta.demands.capacity() * sizeof(SpatialTemporalSwitchSegmentDemand);
    for (const SpatialTemporalSwitchSegmentDemand &demand : delta.demands) {
      bytes += demand.signatures.capacity() *
               sizeof(SpatialTemporalSwitchInputSignature);
      for (const SpatialTemporalSwitchInputSignature &signature :
           demand.signatures)
        bytes += signature.outputs.capacity() *
                     sizeof(::loom::fabric::FabricOrdinal) +
                 signature.traversals.capacity() * sizeof(PnrIndex);
    }
  }
  return bytes;
}

bool loom::pnr::detail::compatibleSpatialTemporalSwitchDemands(
    const SpatialTemporalSwitchSegmentDemand &lhs,
    const SpatialTemporalSwitchSegmentDemand &rhs) {
  if (lhs.domain != rhs.domain)
    return false;
  std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>
      leftSignatures;
  std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>
      rightSignatures;
  leftSignatures.reserve(lhs.signatures.size());
  rightSignatures.reserve(rhs.signatures.size());
  for (const auto &signature : lhs.signatures)
    leftSignatures.push_back(
        {signature.occurrence, signature.input, signature.outputs});
  for (const auto &signature : rhs.signatures)
    rightSignatures.push_back(
        {signature.occurrence, signature.input, signature.outputs});
  return ::loom::fabric::compatibleFabricTemporalSwitchRouteDemands(
      {leftSignatures}, {rightSignatures});
}

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
loom::pnr::detail::deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity) {
  if (routes.size() != problem.transfers().logicalNets().size() ||
      continuity.size() != routes.size())
    return invalid("route and continuity inventories disagree");
  std::map<DemandKey, InputOutputs> selected;
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    const RouteTreeState *route = routes[logicalNet];
    if (!route || !continuity[logicalNet])
      return invalid("route has no tag-continuity projection");
    if (llvm::Error error = appendRouteDemands(
            problem, logicalNet, *route, *continuity[logicalNet], selected))
      return std::move(error);
  }
  for (PnrIndex logicalNet = 0; logicalNet < continuity.size(); ++logicalNet) {
    if (llvm::Error error = verifyDemandCoverage(
            problem, logicalNet, *continuity[logicalNet], selected))
      return std::move(error);
  }
  return materializeDemands(std::move(selected));
}

namespace {

using FlatCrosspoint = ::loom::pnr::detail::SpatialTemporalSwitchCrosspoint;

/// Route-node walk shared with appendRouteDemands, collecting into one flat
/// vector instead of nested maps; one sort then linear grouping replaces the
/// per-crosspoint node allocations of the map-based path.
llvm::Error
collectRouteCrosspoints(const FrozenSpatialPnrProblem &problem,
                        const RouteTreeState &route,
                        const SpatialTagContinuityProjection &continuity,
                        std::vector<FlatCrosspoint> &crosspoints) {
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  if (&route.routingGraph() != &routing)
    return invalid("route belongs to another frozen problem");
  if (route.isUnrouted())
    return llvm::Error::success();
  const auto nodes = route.nodeStorage();
  const auto nodeSegments = continuity.nodeSegments();
  if (nodeSegments.size() != nodes.size())
    return invalid("route and continuity node inventories disagree");
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto traversals = routing.traversals();
  const auto endpoints = routing.routingEndpoints();
  const auto endpointDomains =
      routing.tagContinuity().endpointMatchDomainOrdinals();
  const auto domains = routing.tagContinuity().matchDomains();
  if (endpointDomains.size() != endpoints.size())
    return invalid("endpoint match-domain index has the wrong width");
  for (auto [slot, node] : llvm::enumerate(nodes)) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= arcs.size() || node.parentArc >= arcSources.size())
      return invalid("route node names an absent physical arc");
    const PnrIndex traversal = arcs[node.parentArc].traversal;
    if (traversal >= traversals.size())
      return invalid("route node names an absent physical traversal");
    const auto *payload =
        std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
            &traversals[traversal].reference.payload);
    if (!payload)
      continue;
    const PnrIndex source = arcSources[node.parentArc];
    if (source >= endpoints.size() || source >= endpointDomains.size())
      return invalid("switch traversal source is out of range");
    const PnrIndex domain = endpointDomains[source];
    if (domain == getInvalidPnrIndex())
      continue;
    if (domain >= domains.size() ||
        domains[domain].kind !=
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable ||
        endpoints[source].reference.owner !=
            ::loom::fabric::FabricTransportEndpointOwnerRef::of(
                payload->owner) ||
        endpoints[source].reference.ordinal != payload->input)
      return invalid("switch traversal disagrees with its match domain");
    const PnrIndex segment = nodeSegments[slot];
    if (segment == getInvalidPnrIndex() ||
        segment >= continuity.segments().size())
      return invalid("Temporal switch traversal has no tag segment");
    crosspoints.push_back({domain, segment, payload->input, payload->output,
                           traversal, payload->owner});
  }
  return llvm::Error::success();
}

} // namespace

void loom::pnr::detail::SpatialTemporalSwitchDemandScratch::recycle(
    std::vector<SpatialTemporalSwitchSegmentDemand> &&demands) {
  for (SpatialTemporalSwitchSegmentDemand &demand : demands) {
    for (SpatialTemporalSwitchInputSignature &signature : demand.signatures) {
      signature.outputs.clear();
      signature.traversals.clear();
      signaturePool_.push_back(std::move(signature));
    }
    demand.signatures.clear();
    demandPool_.push_back(std::move(demand));
  }
  demands.clear();
  vectorPool_.push_back(std::move(demands));
}

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
loom::pnr::detail::deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity,
    SpatialTemporalSwitchDemandScratch &scratch) {
  std::vector<FlatCrosspoint> &crosspoints = scratch.crosspoints_;
  crosspoints.clear();
  if (llvm::Error error =
          collectRouteCrosspoints(problem, route, continuity, crosspoints))
    return std::move(error);
  llvm::sort(crosspoints,
             [](const FlatCrosspoint &lhs, const FlatCrosspoint &rhs) {
               return std::tie(lhs.domain, lhs.segment, lhs.input, lhs.output,
                               lhs.traversal) <
                      std::tie(rhs.domain, rhs.segment, rhs.input, rhs.output,
                               rhs.traversal);
             });
  std::vector<SpatialTemporalSwitchSegmentDemand> result;
  if (!scratch.vectorPool_.empty()) {
    result = std::move(scratch.vectorPool_.back());
    scratch.vectorPool_.pop_back();
    result.clear();
  }
  const auto acquireDemand = [&]() {
    if (scratch.demandPool_.empty())
      return SpatialTemporalSwitchSegmentDemand{};
    SpatialTemporalSwitchSegmentDemand demand =
        std::move(scratch.demandPool_.back());
    scratch.demandPool_.pop_back();
    demand.signatures.clear();
    return demand;
  };
  const auto acquireSignature = [&]() {
    if (scratch.signaturePool_.empty())
      return SpatialTemporalSwitchInputSignature{};
    SpatialTemporalSwitchInputSignature signature =
        std::move(scratch.signaturePool_.back());
    scratch.signaturePool_.pop_back();
    signature.outputs.clear();
    signature.traversals.clear();
    return signature;
  };
  std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView> &views =
      scratch.views_;
  for (std::size_t begin = 0; begin < crosspoints.size();) {
    const PnrIndex domain = crosspoints[begin].domain;
    const PnrIndex segment = crosspoints[begin].segment;
    SpatialTemporalSwitchSegmentDemand demand = acquireDemand();
    demand.domain = domain;
    demand.logicalNet = logicalNet;
    demand.segment = segment;
    std::size_t end = begin;
    while (end < crosspoints.size() && crosspoints[end].domain == domain &&
           crosspoints[end].segment == segment) {
      const ::loom::fabric::FabricOrdinal input = crosspoints[end].input;
      SpatialTemporalSwitchInputSignature signature = acquireSignature();
      signature.occurrence = crosspoints[end].occurrence;
      signature.input = input;
      for (; end < crosspoints.size() && crosspoints[end].domain == domain &&
             crosspoints[end].segment == segment &&
             crosspoints[end].input == input;
           ++end) {
        if (!signature.outputs.empty() &&
            signature.outputs.back() == crosspoints[end].output)
          return invalid("one segment repeats a switch crosspoint");
        if (crosspoints[end].occurrence != signature.occurrence)
          return invalid("one input signature crosses switch occurrences");
        signature.outputs.push_back(crosspoints[end].output);
        signature.traversals.push_back(crosspoints[end].traversal);
      }
      demand.signatures.push_back(std::move(signature));
    }
    views.clear();
    views.reserve(demand.signatures.size());
    for (const SpatialTemporalSwitchInputSignature &signature :
         demand.signatures)
      views.push_back(
          {signature.occurrence, signature.input, signature.outputs});
    if (llvm::Error error =
            ::loom::fabric::validateFabricTemporalSwitchRouteDemand({views}))
      return std::move(error);
    result.push_back(std::move(demand));
    begin = end;
  }
  return result;
}

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
loom::pnr::detail::deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity) {
  SpatialTemporalSwitchDemandScratch scratch;
  return deriveSpatialTemporalSwitchSegmentDemands(problem, logicalNet, route,
                                                   continuity, scratch);
}

struct loom::pnr::detail::SpatialTagInterferenceBuilder final {
  static llvm::Expected<PnrIndex> segmentOrdinal(
      SpatialTagVertexRef vertex,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity) {
    if (vertex.logicalNet >= continuity.size() ||
        !continuity[vertex.logicalNet])
      return invalid("tag vertex names an absent logical net");
    const auto segments = continuity[vertex.logicalNet]->segments();
    const auto found = llvm::lower_bound(
        segments, vertex,
        [](const SpatialTagContinuitySegment &segment,
           SpatialTagVertexRef target) {
          return std::tie(segment.originKind, segment.origin) <
                 std::tie(target.originKind, target.origin);
        });
    if (found == segments.end() || found->originKind != vertex.originKind ||
        found->origin != vertex.origin)
      return invalid("tag vertex names an absent continuity segment");
    return static_cast<PnrIndex>(found - segments.begin());
  }

  static llvm::Error rebuildCanonicalInventory(
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
      SpatialTagInterferenceProjection &result) {
    result.netSegmentOffsets_.clear();
    result.vertexRefs_.clear();
    result.netSegmentOffsets_.reserve(continuity.size() + 1);
    result.netSegmentOffsets_.push_back(0);
    for (PnrIndex logicalNet = 0; logicalNet < continuity.size();
         ++logicalNet) {
      const SpatialTagContinuityProjection *net = continuity[logicalNet];
      if (!net)
        return invalid("interference has a null continuity projection");
      const std::size_t end =
          result.vertexRefs_.size() + net->segments().size();
      auto offset = checkedIndex(end, "tag segment inventory");
      if (!offset)
        return offset.takeError();
      for (const SpatialTagContinuitySegment &segment : net->segments())
        result.vertexRefs_.push_back(
            {logicalNet, segment.originKind, segment.origin});
      result.netSegmentOffsets_.push_back(*offset);
    }
    if (!llvm::is_sorted(result.vertexRefs_))
      return invalid("tag vertex inventory is not canonical");
    return llvm::Error::success();
  }

  static const SpatialTemporalSwitchSegmentDemand *findDemand(
      const SpatialTagInterferenceProjection &projection,
      SpatialTagVertexRef vertex, PnrIndex domain,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity) {
    auto segment = segmentOrdinal(vertex, continuity);
    if (!segment || vertex.logicalNet >= projection.netSwitchDemands_.size()) {
      if (!segment)
        llvm::consumeError(segment.takeError());
      return nullptr;
    }
    const auto &demands = projection.netSwitchDemands_[vertex.logicalNet];
    const auto key = std::make_pair(domain, *segment);
    const auto found = llvm::lower_bound(
        demands, key, [](const auto &demand, const auto target) {
          return std::make_pair(demand.domain, demand.segment) < target;
        });
    return found == demands.end() || found->domain != domain ||
                   found->segment != *segment
               ? nullptr
               : &*found;
  }

  static llvm::Error rebuildDomain(
      const FrozenSpatialPnrProblem &problem, PnrIndex domain,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
      SpatialTagInterferenceProjection &result) {
    const auto domains = problem.routing().tagContinuity().matchDomains();
    if (domain >= domains.size() || domain >= result.domainVertices_.size() ||
        domain >= result.domainConflicts_.size())
      return invalid("rebuilt tag match domain is out of range");
    auto &members = result.domainVertices_[domain];
    llvm::sort(members);
    members.erase(std::unique(members.begin(), members.end()), members.end());
    auto &conflicts = result.domainConflicts_[domain];
    conflicts.clear();
    std::vector<const SpatialTemporalSwitchSegmentDemand *> switchDemands;
    if (domains[domain].kind ==
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable) {
      switchDemands.reserve(members.size());
      for (SpatialTagVertexRef member : members) {
        const auto *demand = findDemand(result, member, domain, continuity);
        if (!demand)
          return invalid("switch match-domain member has no row demand");
        switchDemands.push_back(demand);
      }
    }
    for (std::size_t left = 0; left != members.size(); ++left)
      for (std::size_t right = left + 1; right != members.size(); ++right) {
        bool conflict = true;
        if (!switchDemands.empty())
          conflict = !compatibleSpatialTemporalSwitchDemands(
              *switchDemands[left], *switchDemands[right]);
        if (conflict)
          conflicts.push_back({members[left], members[right]});
      }
    return llvm::Error::success();
  }

  static llvm::Error rebuildGlobal(SpatialTagInterferenceProjection &result,
                                   std::vector<PnrIndex> &cursorScratch) {
    result.globalConflicts_.clear();
    for (const auto &domain : result.domainConflicts_)
      result.globalConflicts_.insert(result.globalConflicts_.end(),
                                     domain.begin(), domain.end());
    llvm::sort(result.globalConflicts_);
    result.globalConflicts_.erase(std::unique(result.globalConflicts_.begin(),
                                              result.globalConflicts_.end()),
                                  result.globalConflicts_.end());
    // Two counting passes fill the incidence in place: unique canonical pairs
    // cannot produce duplicate neighbors, so per-vertex slices only need an
    // in-place sort and no per-vertex storage.
    cursorScratch.assign(result.vertexRefs_.size(), 0);
    for (const SpatialTagConflictPair &pair : result.globalConflicts_) {
      const PnrIndex lhs = result.vertexOrdinal(pair.lhs);
      const PnrIndex rhs = result.vertexOrdinal(pair.rhs);
      if (lhs == getInvalidPnrIndex() || rhs == getInvalidPnrIndex() ||
          lhs == rhs)
        return invalid("tag conflict names an absent canonical vertex");
      ++cursorScratch[lhs];
      ++cursorScratch[rhs];
    }
    result.conflictOffsets_.clear();
    result.conflictOffsets_.reserve(result.vertexRefs_.size() + 1);
    result.conflictOffsets_.push_back(0);
    std::size_t total = 0;
    for (std::size_t vertex = 0; vertex < cursorScratch.size(); ++vertex) {
      const PnrIndex degree = cursorScratch[vertex];
      if (degree > std::numeric_limits<std::size_t>::max() - total)
        return invalid("tag interference incidence size overflows");
      total += degree;
      auto offset = checkedIndex(total, "tag interference incidence inventory");
      if (!offset)
        return offset.takeError();
      cursorScratch[vertex] = result.conflictOffsets_.back();
      result.conflictOffsets_.push_back(*offset);
    }
    result.conflicts_.resize(total);
    for (const SpatialTagConflictPair &pair : result.globalConflicts_) {
      const PnrIndex lhs = result.vertexOrdinal(pair.lhs);
      const PnrIndex rhs = result.vertexOrdinal(pair.rhs);
      result.conflicts_[cursorScratch[lhs]++] = rhs;
      result.conflicts_[cursorScratch[rhs]++] = lhs;
    }
    for (std::size_t vertex = 0; vertex + 1 < result.conflictOffsets_.size();
         ++vertex)
      std::sort(result.conflicts_.begin() + result.conflictOffsets_[vertex],
                result.conflicts_.begin() +
                    result.conflictOffsets_[vertex + 1]);
    return llvm::Error::success();
  }
};

llvm::Expected<SpatialTagInterferenceProjection>
loom::pnr::detail::deriveSpatialTagInterference(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity) {
  if (routes.size() != continuity.size())
    return invalid("interference route and continuity inventories disagree");
  SpatialTagInterferenceProjection result;
  if (llvm::Error error =
          SpatialTagInterferenceBuilder::rebuildCanonicalInventory(continuity,
                                                                   result))
    return std::move(error);
  const auto domains = problem.routing().tagContinuity().matchDomains();
  result.domainVertices_.resize(domains.size());
  result.domainConflicts_.resize(domains.size());
  result.netDomains_.resize(continuity.size());
  result.netSwitchDemands_.resize(continuity.size());
  for (PnrIndex logicalNet = 0; logicalNet < continuity.size(); ++logicalNet) {
    if (!routes[logicalNet])
      return invalid("interference has a null route");
    auto demands = deriveSpatialTemporalSwitchSegmentDemands(
        problem, logicalNet, *routes[logicalNet], *continuity[logicalNet]);
    if (!demands)
      return demands.takeError();
    result.netSwitchDemands_[logicalNet] = std::move(*demands);
    const auto offsets = continuity[logicalNet]->segmentDomainOffsets();
    const auto localDomains = continuity[logicalNet]->segmentDomains();
    result.netDomains_[logicalNet].assign(localDomains.begin(),
                                          localDomains.end());
    llvm::sort(result.netDomains_[logicalNet]);
    result.netDomains_[logicalNet].erase(
        std::unique(result.netDomains_[logicalNet].begin(),
                    result.netDomains_[logicalNet].end()),
        result.netDomains_[logicalNet].end());
    for (PnrIndex segment = 0;
         segment < continuity[logicalNet]->segments().size(); ++segment) {
      const SpatialTagVertexRef vertex =
          result.vertexRefs_[result.netSegmentOffsets_[logicalNet] + segment];
      for (PnrIndex incidence = offsets[segment];
           incidence < offsets[segment + 1]; ++incidence) {
        const PnrIndex domain = localDomains[incidence];
        if (domain >= result.domainVertices_.size())
          return invalid("tag segment names an absent match domain");
        result.domainVertices_[domain].push_back(vertex);
      }
    }
  }
  for (PnrIndex domain = 0; domain < domains.size(); ++domain)
    if (llvm::Error error = SpatialTagInterferenceBuilder::rebuildDomain(
            problem, domain, continuity, result))
      return std::move(error);
  std::vector<PnrIndex> cursorScratch;
  if (llvm::Error error =
          SpatialTagInterferenceBuilder::rebuildGlobal(result, cursorScratch))
    return std::move(error);
  return result;
}

llvm::Error loom::pnr::detail::stageSpatialTagInterferenceUpdate(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity,
    llvm::ArrayRef<PnrIndex> touchedLogicalNets,
    SpatialTagInterferenceProjection &projection,
    SpatialTagInterferenceUpdateScratch &scratch) {
  if (scratch.active_)
    return invalid("tag interference update is already active");
  if (routes.size() != continuity.size() ||
      projection.netDomains_.size() != continuity.size() ||
      projection.netSwitchDemands_.size() != continuity.size())
    return invalid("tag interference update inventories disagree");
  std::vector<PnrIndex> touched(touchedLogicalNets.begin(),
                                touchedLogicalNets.end());
  llvm::sort(touched);
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
  for (PnrIndex logicalNet : touched)
    if (logicalNet >= routes.size() || !routes[logicalNet] ||
        !continuity[logicalNet])
      return invalid("tag interference update names an absent logical net");

  scratch.affectedDomains_.clear();
  for (PnrIndex logicalNet : touched) {
    scratch.affectedDomains_.insert(scratch.affectedDomains_.end(),
                                    projection.netDomains_[logicalNet].begin(),
                                    projection.netDomains_[logicalNet].end());
    for (PnrIndex domain : continuity[logicalNet]->segmentDomains())
      scratch.affectedDomains_.push_back(domain);
  }
  llvm::sort(scratch.affectedDomains_);
  scratch.affectedDomains_.erase(std::unique(scratch.affectedDomains_.begin(),
                                             scratch.affectedDomains_.end()),
                                 scratch.affectedDomains_.end());

  scratch.active_ = true;
  std::swap(scratch.previousNetSegmentOffsets_, projection.netSegmentOffsets_);
  std::swap(scratch.previousVertexRefs_, projection.vertexRefs_);
  std::swap(scratch.previousConflictOffsets_, projection.conflictOffsets_);
  std::swap(scratch.previousConflicts_, projection.conflicts_);
  std::swap(scratch.previousGlobalConflicts_, projection.globalConflicts_);
  const auto fail = [&](llvm::Error error) {
    rollbackSpatialTagInterferenceUpdate(projection, scratch);
    return error;
  };

  scratch.netDemandDeltas_.clear();
  scratch.netDemandDeltas_.reserve(touched.size());
  for (PnrIndex logicalNet : touched) {
    scratch.netDemandDeltas_.push_back({logicalNet, {}, {}});
    auto &delta = scratch.netDemandDeltas_.back();
    std::swap(delta.domains, projection.netDomains_[logicalNet]);
    projection.netDomains_[logicalNet].assign(
        continuity[logicalNet]->segmentDomains().begin(),
        continuity[logicalNet]->segmentDomains().end());
    llvm::sort(projection.netDomains_[logicalNet]);
    projection.netDomains_[logicalNet].erase(
        std::unique(projection.netDomains_[logicalNet].begin(),
                    projection.netDomains_[logicalNet].end()),
        projection.netDomains_[logicalNet].end());
    std::swap(delta.demands, projection.netSwitchDemands_[logicalNet]);
    auto demands = deriveSpatialTemporalSwitchSegmentDemands(
        problem, logicalNet, *routes[logicalNet], *continuity[logicalNet],
        scratch.demandScratch_);
    if (!demands)
      return fail(demands.takeError());
    projection.netSwitchDemands_[logicalNet] = std::move(*demands);
  }

  scratch.domainDeltas_.clear();
  scratch.domainDeltas_.reserve(scratch.affectedDomains_.size());
  for (PnrIndex domain : scratch.affectedDomains_) {
    if (domain >= projection.domainVertices_.size() ||
        domain >= projection.domainConflicts_.size())
      return fail(invalid("tag interference update domain is out of range"));
    scratch.domainDeltas_.push_back({domain, {}, {}});
    auto &delta = scratch.domainDeltas_.back();
    std::swap(delta.vertices, projection.domainVertices_[domain]);
    std::swap(delta.conflicts, projection.domainConflicts_[domain]);
    auto &members = projection.domainVertices_[domain];
    for (SpatialTagVertexRef vertex : delta.vertices)
      if (!std::binary_search(touched.begin(), touched.end(),
                              vertex.logicalNet))
        members.push_back(vertex);
    for (PnrIndex logicalNet : touched) {
      const auto offsets = continuity[logicalNet]->segmentDomainOffsets();
      const auto localDomains = continuity[logicalNet]->segmentDomains();
      const auto segments = continuity[logicalNet]->segments();
      for (PnrIndex segment = 0; segment < segments.size(); ++segment)
        if (std::binary_search(localDomains.begin() + offsets[segment],
                               localDomains.begin() + offsets[segment + 1],
                               domain))
          members.push_back({logicalNet, segments[segment].originKind,
                             segments[segment].origin});
    }
  }
  if (llvm::Error error =
          SpatialTagInterferenceBuilder::rebuildCanonicalInventory(continuity,
                                                                   projection))
    return fail(std::move(error));
  for (PnrIndex domain : scratch.affectedDomains_)
    if (llvm::Error error = SpatialTagInterferenceBuilder::rebuildDomain(
            problem, domain, continuity, projection))
      return fail(std::move(error));
  if (llvm::Error error = SpatialTagInterferenceBuilder::rebuildGlobal(
          projection, scratch.cursorScratch_))
    return fail(std::move(error));
  return llvm::Error::success();
}

void loom::pnr::detail::commitSpatialTagInterferenceUpdate(
    SpatialTagInterferenceUpdateScratch &scratch) noexcept {
  if (!scratch.active_)
    return;
  scratch.active_ = false;
  scratch.previousNetSegmentOffsets_.clear();
  scratch.previousVertexRefs_.clear();
  scratch.previousConflictOffsets_.clear();
  scratch.previousConflicts_.clear();
  scratch.previousGlobalConflicts_.clear();
  scratch.affectedDomains_.clear();
  scratch.domainDeltas_.clear();
  for (auto &delta : scratch.netDemandDeltas_)
    scratch.demandScratch_.recycle(std::move(delta.demands));
  scratch.netDemandDeltas_.clear();
}

void loom::pnr::detail::rollbackSpatialTagInterferenceUpdate(
    SpatialTagInterferenceProjection &projection,
    SpatialTagInterferenceUpdateScratch &scratch) noexcept {
  if (!scratch.active_)
    return;
  for (auto &delta : scratch.netDemandDeltas_)
    std::swap(delta.domains, projection.netDomains_[delta.logicalNet]);
  for (auto &delta : scratch.netDemandDeltas_)
    std::swap(delta.demands, projection.netSwitchDemands_[delta.logicalNet]);
  for (auto &delta : scratch.domainDeltas_) {
    std::swap(delta.vertices, projection.domainVertices_[delta.domain]);
    std::swap(delta.conflicts, projection.domainConflicts_[delta.domain]);
  }
  std::swap(scratch.previousNetSegmentOffsets_, projection.netSegmentOffsets_);
  std::swap(scratch.previousVertexRefs_, projection.vertexRefs_);
  std::swap(scratch.previousConflictOffsets_, projection.conflictOffsets_);
  std::swap(scratch.previousConflicts_, projection.conflicts_);
  std::swap(scratch.previousGlobalConflicts_, projection.globalConflicts_);
  scratch.active_ = false;
  scratch.affectedDomains_.clear();
  scratch.domainDeltas_.clear();
  scratch.netDemandDeltas_.clear();
}
