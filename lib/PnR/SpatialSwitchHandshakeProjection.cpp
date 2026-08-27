#include "SpatialSwitchHandshakeProjection.h"

#include "SpatialSwitchRowPacking.h"
#include "SpatialTagAssignmentState.h"

#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <system_error>
#include <tuple>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial Temporal switch handshake projection: " + message);
}

const FrozenSpatialSwitchHandshakeActivation *
findActivation(const FrozenSpatialHandshakeIndex &handshake, PnrIndex domain,
               ::loom::fabric::FabricOrdinal row,
               ::loom::fabric::FabricOrdinal input) {
  const auto activations = handshake.switchActivations();
  const auto found = llvm::lower_bound(
      activations, std::tie(domain, row, input),
      [](const FrozenSpatialSwitchHandshakeActivation &candidate,
         const auto &key) {
        return std::tie(candidate.matchDomain, candidate.row, candidate.input) <
               key;
      });
  if (found == activations.end() || found->matchDomain != domain ||
      found->row != row || found->input != input)
    return nullptr;
  return &*found;
}

const FrozenSpatialSwitchHandshakeTraversalSelection *
findTraversal(const FrozenSpatialHandshakeIndex &handshake,
              const FrozenSpatialSwitchHandshakeActivation &activation,
              PnrIndex traversal) {
  const auto selections = handshake.switchTraversalSelections().slice(
      activation.traversalSelectionOffset, activation.traversalSelectionCount);
  const auto found = llvm::lower_bound(
      selections, traversal,
      [](const FrozenSpatialSwitchHandshakeTraversalSelection &candidate,
         PnrIndex key) { return candidate.traversal < key; });
  return found == selections.end() || found->traversal != traversal ? nullptr
                                                                    : &*found;
}

using SwitchDemand = SpatialTemporalSwitchSegmentDemand;

template <typename TagLookup>
llvm::Expected<std::vector<PnrIndex>>
projectDomainFragments(const FrozenSpatialPnrProblem &problem, PnrIndex domain,
                       llvm::ArrayRef<const SwitchDemand *> demands,
                       TagLookup lookupTag) {
  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  if (domain >= matchDomains.size())
    return invalid("switch demand names an absent match domain");
  if (matchDomains[domain].kind !=
      ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
    return invalid("switch demand names a non-switch match domain");

  std::vector<
      std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>>
      signatureStorage;
  signatureStorage.reserve(demands.size());
  for (const SwitchDemand *demand : demands) {
    if (!demand || demand->domain != domain)
      return invalid("switch demand grouping is inconsistent");
    signatureStorage.emplace_back();
    auto &signatures = signatureStorage.back();
    signatures.reserve(demand->signatures.size());
    for (const SpatialTemporalSwitchInputSignature &signature :
         demand->signatures)
      signatures.push_back(
          {signature.occurrence, signature.input, signature.outputs});
  }

  std::vector<::loom::fabric::FabricTemporalSwitchTaggedRouteDemandView>
      taggedDemandViews;
  std::vector<::loom::fabric::FabricTemporalSwitchCandidateRouteDemandView>
      candidateDemandViews;
  taggedDemandViews.reserve(signatureStorage.size());
  candidateDemandViews.reserve(signatureStorage.size());
  bool allTagsAssigned = true;
  for (auto [ordinal, signatures] : llvm::enumerate(signatureStorage)) {
    const SwitchDemand &demand = *demands[ordinal];
    auto tag = lookupTag(demand);
    if (!tag)
      return tag.takeError();
    if (!*tag) {
      allTagsAssigned = false;
      candidateDemandViews.push_back({{signatures}, std::nullopt});
      continue;
    }
    if (!::fabric::isRepresentablePhysicalTagValue(
            matchDomains[domain].tagWidthBits, **tag))
      return invalid("switch route demand has an out-of-range Physical Tag");
    llvm::APInt normalized =
        (**tag).zextOrTrunc(matchDomains[domain].tagWidthBits);
    taggedDemandViews.push_back({{signatures}, normalized});
    candidateDemandViews.push_back({{signatures}, std::move(normalized)});
  }

  std::vector<std::vector<std::uint64_t>> rows;
  if (allTagsAssigned) {
    auto projected =
        ::loom::fabric::projectFabricTemporalSwitchRouteRows(taggedDemandViews);
    if (!projected)
      return projected.takeError();
    rows.reserve(projected->size());
    for (auto &row : *projected)
      rows.push_back(std::move(row.demandOrdinals));
  } else {
    auto projected =
        ::loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(
            candidateDemandViews);
    if (!projected)
      return projected.takeError();
    rows.reserve(projected->size());
    for (auto &row : *projected)
      rows.push_back(std::move(row.demandOrdinals));
  }

  const FrozenSpatialHandshakeIndex &handshake = problem.handshake();
  std::vector<PnrIndex> fragments;
  for (auto [rowOrdinal, rowDemands] : llvm::enumerate(rows)) {
    const std::optional<std::uint64_t> residentCapacity =
        matchDomains[domain].residentEntryCapacity;
    if (residentCapacity && rowOrdinal >= *residentCapacity)
      continue;
    std::map<::loom::fabric::FabricOrdinal, std::vector<PnrIndex>> byInput;
    for (std::uint64_t demandOrdinal : rowDemands) {
      if (demandOrdinal >= demands.size())
        return invalid("Fabric switch row names an absent route demand");
      for (const SpatialTemporalSwitchInputSignature &signature :
           demands[demandOrdinal]->signatures)
        byInput[signature.input].insert(byInput[signature.input].end(),
                                        signature.traversals.begin(),
                                        signature.traversals.end());
    }
    for (auto &[input, traversals] : byInput) {
      llvm::sort(traversals);
      traversals.erase(std::unique(traversals.begin(), traversals.end()),
                       traversals.end());
      const auto *activation = findActivation(
          handshake, domain,
          static_cast<::loom::fabric::FabricOrdinal>(rowOrdinal), input);
      if (!activation)
        return invalid(llvm::Twine("resident switch domain ") +
                       llvm::Twine(domain) + " row " + llvm::Twine(rowOrdinal) +
                       " input " + llvm::Twine(input) +
                       " has no frozen activation");
      const auto base = handshake.switchActivationBaseFragments().slice(
          activation->baseFragmentOffset, activation->baseFragmentCount);
      fragments.insert(fragments.end(), base.begin(), base.end());
      for (PnrIndex traversal : traversals) {
        const auto *selection =
            findTraversal(handshake, *activation, traversal);
        if (!selection)
          return invalid("route crosspoint has no frozen switch fragment");
        const auto selectedFragments =
            handshake.switchTraversalFragments().slice(
                selection->fragmentOffset, selection->fragmentCount);
        fragments.insert(fragments.end(), selectedFragments.begin(),
                         selectedFragments.end());
      }
    }
  }
  llvm::sort(fragments);
  fragments.erase(std::unique(fragments.begin(), fragments.end()),
                  fragments.end());
  return fragments;
}

llvm::Expected<std::vector<std::vector<const SwitchDemand *>>>
groupAssignmentDemands(const FrozenSpatialPnrProblem &problem,
                       const SpatialTagAssignmentStateStorage &assignments) {
  if (assignments.problem != &problem ||
      assignments.interference.netSegmentOffsets().size() !=
          assignments.nets.size() + 1)
    return invalid("Tag-assignment demand cache belongs to another problem");
  std::vector<std::vector<const SwitchDemand *>> grouped(
      problem.routing().tagContinuity().matchDomains().size());
  for (PnrIndex logicalNet = 0; logicalNet < assignments.nets.size();
       ++logicalNet)
    for (const SwitchDemand &demand :
         assignments.interference.switchDemands(logicalNet)) {
      if (demand.logicalNet != logicalNet || demand.domain >= grouped.size())
        return invalid("Tag-assignment switch demand is inconsistent");
      grouped[demand.domain].push_back(&demand);
    }
  return grouped;
}

llvm::Expected<const llvm::APInt *>
assignmentTag(const SpatialTagAssignmentStateStorage &assignments,
              const SwitchDemand &demand) {
  if (demand.logicalNet >= assignments.nets.size() ||
      demand.segment >= assignments.nets[demand.logicalNet].values.size())
    return invalid("switch demand names an absent Physical Tag segment");
  const auto &value =
      assignments.nets[demand.logicalNet].values[demand.segment];
  return value ? &*value : nullptr;
}

} // namespace

bool loom::pnr::detail::hasSpatialTemporalSwitchHandshakeDomain(
    const FrozenSpatialPnrProblem &problem) {
  return llvm::any_of(
      problem.routing().tagContinuity().matchDomains(), [](const auto &domain) {
        return domain.kind == ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                                  TemporalSwitchTable;
      });
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeFragments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>> tagValues) {
  if (routes.size() != problem.transfers().logicalNets().size() ||
      tagValues.size() != routes.size())
    return invalid("route or Physical Tag inventory has the wrong size");

  std::vector<SpatialTagContinuityProjection> continuityStorage;
  std::vector<const SpatialTagContinuityProjection *> continuity;
  continuityStorage.reserve(routes.size());
  continuity.reserve(routes.size());
  SpatialTagContinuityScratch continuityScratch;
  for (const RouteTreeState *route : routes) {
    if (!route || &route->routingGraph() != &problem.routing())
      return invalid("route belongs to another frozen problem");
    continuityStorage.emplace_back();
    if (llvm::Error error = rebuildSpatialTagContinuityUnchecked(
            *route, continuityStorage.back(), continuityScratch))
      return std::move(error);
    continuity.push_back(&continuityStorage.back());
  }
  auto demands =
      deriveSpatialTemporalSwitchSegmentDemands(problem, routes, continuity);
  if (!demands)
    return demands.takeError();
  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  std::vector<std::vector<const SwitchDemand *>> demandsByDomain(
      matchDomains.size());
  for (const SwitchDemand &demand : *demands) {
    if (demand.domain >= demandsByDomain.size())
      return invalid("switch demand names an absent match domain");
    demandsByDomain[demand.domain].push_back(&demand);
  }

  std::vector<PnrIndex> fragments;
  for (PnrIndex domain = 0; domain < demandsByDomain.size(); ++domain) {
    if (demandsByDomain[domain].empty())
      continue;
    auto projected = projectDomainFragments(
        problem, domain, demandsByDomain[domain],
        [&](const SwitchDemand &demand) -> llvm::Expected<const llvm::APInt *> {
          if (demand.logicalNet >= tagValues.size() ||
              demand.segment >= tagValues[demand.logicalNet].size())
            return invalid(
                "switch demand names an absent Physical Tag segment");
          const auto &value = tagValues[demand.logicalNet][demand.segment];
          return value ? &*value : nullptr;
        });
    if (!projected)
      return projected.takeError();
    fragments.insert(fragments.end(), projected->begin(), projected->end());
  }
  llvm::sort(fragments);
  fragments.erase(std::unique(fragments.begin(), fragments.end()),
                  fragments.end());
  return fragments;
}

llvm::Expected<std::vector<std::vector<PnrIndex>>>
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeFragmentsByDomain(
    const FrozenSpatialPnrProblem &problem,
    const SpatialTagAssignmentStateStorage &assignments) {
  auto grouped = groupAssignmentDemands(problem, assignments);
  if (!grouped)
    return grouped.takeError();
  std::vector<std::vector<PnrIndex>> fragments(grouped->size());
  for (PnrIndex domain = 0; domain < grouped->size(); ++domain) {
    if ((*grouped)[domain].empty())
      continue;
    auto projected = projectDomainFragments(
        problem, domain, (*grouped)[domain], [&](const SwitchDemand &demand) {
          return assignmentTag(assignments, demand);
        });
    if (!projected)
      return projected.takeError();
    fragments[domain] = std::move(*projected);
  }
  return fragments;
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeDomainFragments(
    const FrozenSpatialPnrProblem &problem, PnrIndex domain,
    const SpatialTagAssignmentStateStorage &assignments) {
  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  if (domain >= matchDomains.size())
    return invalid("updated switch domain is out of range");
  if (matchDomains[domain].kind !=
      ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
    return invalid("updated match domain is not a Temporal switch");
  if (assignments.problem != &problem ||
      assignments.interference.netSegmentOffsets().size() !=
          assignments.nets.size() + 1)
    return invalid("Tag-assignment demand cache belongs to another problem");
  std::vector<const SwitchDemand *> demands;
  demands.reserve(assignments.interference.domainVertices(domain).size());
  for (PnrIndex logicalNet = 0; logicalNet < assignments.nets.size();
       ++logicalNet)
    for (const SwitchDemand &demand :
         assignments.interference.switchDemands(logicalNet)) {
      if (demand.logicalNet != logicalNet ||
          demand.domain >= matchDomains.size())
        return invalid("Tag-assignment switch demand is inconsistent");
      if (demand.domain == domain)
        demands.push_back(&demand);
    }
  return projectDomainFragments(problem, domain, demands,
                                [&](const SwitchDemand &demand) {
                                  return assignmentTag(assignments, demand);
                                });
}
