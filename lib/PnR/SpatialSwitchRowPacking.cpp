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
  if (lhs == rhs || domain >= temporalSwitchDomains_.size())
    return false;
  if (!temporalSwitchDomains_[domain])
    return true;
  if (lhs > rhs)
    std::swap(lhs, rhs);
  return compatibleSwitchPairs_.find({domain, lhs, rhs}) ==
         compatibleSwitchPairs_.end();
}

std::size_t SpatialTagInterferenceProjection::retainedStorageBytes() const {
  return netSegmentOffsets_.capacity() * sizeof(PnrIndex) +
         conflictOffsets_.capacity() * sizeof(PnrIndex) +
         conflicts_.capacity() * sizeof(PnrIndex) +
         temporalSwitchDomains_.capacity() * sizeof(std::uint8_t) +
         compatibleSwitchPairs_.size() *
             sizeof(std::tuple<PnrIndex, PnrIndex, PnrIndex>);
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

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
loom::pnr::detail::deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity) {
  if (logicalNet >= problem.transfers().logicalNets().size())
    return invalid("logical net is out of range");
  std::map<DemandKey, InputOutputs> selected;
  if (llvm::Error error =
          appendRouteDemands(problem, logicalNet, route, continuity, selected))
    return std::move(error);
  if (llvm::Error error =
          verifyDemandCoverage(problem, logicalNet, continuity, selected))
    return std::move(error);
  return materializeDemands(std::move(selected));
}

llvm::Expected<SpatialTagInterferenceProjection>
loom::pnr::detail::deriveSpatialTagInterference(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity) {
  if (routes.size() != continuity.size())
    return invalid("interference route and continuity inventories disagree");
  SpatialTagInterferenceProjection result;
  result.netSegmentOffsets_.reserve(continuity.size() + 1);
  result.netSegmentOffsets_.push_back(0);
  for (const SpatialTagContinuityProjection *net : continuity) {
    if (!net)
      return invalid("interference has a null continuity projection");
    const std::size_t end =
        static_cast<std::size_t>(result.netSegmentOffsets_.back()) +
        net->segments().size();
    auto offset = checkedIndex(end, "tag segment inventory");
    if (!offset)
      return offset.takeError();
    result.netSegmentOffsets_.push_back(*offset);
  }
  const PnrIndex vertexCount = result.netSegmentOffsets_.back();
  const auto domains = problem.routing().tagContinuity().matchDomains();
  result.temporalSwitchDomains_.reserve(domains.size());
  for (const auto &domain : domains)
    result.temporalSwitchDomains_.push_back(
        domain.kind ==
        ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable);
  std::vector<std::vector<PnrIndex>> domainVertices(domains.size());
  for (PnrIndex logicalNet = 0; logicalNet < continuity.size(); ++logicalNet) {
    const auto offsets = continuity[logicalNet]->segmentDomainOffsets();
    const auto localDomains = continuity[logicalNet]->segmentDomains();
    for (PnrIndex segment = 0;
         segment < continuity[logicalNet]->segments().size(); ++segment) {
      const PnrIndex vertex = result.netSegmentOffsets_[logicalNet] + segment;
      for (PnrIndex incidence = offsets[segment];
           incidence < offsets[segment + 1]; ++incidence) {
        const PnrIndex domain = localDomains[incidence];
        if (domain >= domainVertices.size())
          return invalid("tag segment names an absent match domain");
        domainVertices[domain].push_back(vertex);
      }
    }
  }

  auto switchDemands =
      deriveSpatialTemporalSwitchSegmentDemands(problem, routes, continuity);
  if (!switchDemands)
    return switchDemands.takeError();
  std::map<std::pair<PnrIndex, PnrIndex>,
           const SpatialTemporalSwitchSegmentDemand *>
      demandByDomainVertex;
  for (const SpatialTemporalSwitchSegmentDemand &demand : *switchDemands) {
    if (demand.logicalNet + 1 >= result.netSegmentOffsets_.size() ||
        demand.segment >= continuity[demand.logicalNet]->segments().size())
      return invalid("switch demand names an absent tag segment");
    const PnrIndex vertex =
        result.netSegmentOffsets_[demand.logicalNet] + demand.segment;
    if (!demandByDomainVertex
             .emplace(std::make_pair(demand.domain, vertex), &demand)
             .second)
      return invalid("switch demand repeats one domain segment");
  }

  std::vector<std::vector<PnrIndex>> adjacency(vertexCount);
  for (PnrIndex domain = 0; domain < domainVertices.size(); ++domain) {
    auto &members = domainVertices[domain];
    llvm::sort(members);
    members.erase(std::unique(members.begin(), members.end()), members.end());
    for (std::size_t left = 0; left != members.size(); ++left)
      for (std::size_t right = left + 1; right != members.size(); ++right) {
        bool conflict = true;
        if (domains[domain].kind ==
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable) {
          const auto lhs = demandByDomainVertex.find({domain, members[left]});
          const auto rhs = demandByDomainVertex.find({domain, members[right]});
          if (lhs == demandByDomainVertex.end() ||
              rhs == demandByDomainVertex.end())
            return invalid("switch match-domain member has no row demand");
          conflict = !compatibleSpatialTemporalSwitchDemands(*lhs->second,
                                                             *rhs->second);
          if (!conflict)
            result.compatibleSwitchPairs_.emplace(domain, members[left],
                                                  members[right]);
        }
        if (!conflict)
          continue;
        adjacency[members[left]].push_back(members[right]);
        adjacency[members[right]].push_back(members[left]);
      }
  }
  result.conflictOffsets_.reserve(static_cast<std::size_t>(vertexCount) + 1);
  result.conflictOffsets_.push_back(0);
  for (auto &neighbors : adjacency) {
    llvm::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
    if (neighbors.size() >
        std::numeric_limits<std::size_t>::max() - result.conflicts_.size())
      return invalid("tag interference incidence size overflows");
    result.conflicts_.insert(result.conflicts_.end(), neighbors.begin(),
                             neighbors.end());
    auto offset = checkedIndex(result.conflicts_.size(),
                               "tag interference incidence inventory");
    if (!offset)
      return offset.takeError();
    result.conflictOffsets_.push_back(*offset);
  }
  return result;
}
