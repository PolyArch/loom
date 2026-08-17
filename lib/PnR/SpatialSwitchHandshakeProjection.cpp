#include "SpatialSwitchHandshakeProjection.h"

#include "SpatialSwitchRowPacking.h"

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

  std::map<PnrIndex, std::vector<const SpatialTemporalSwitchSegmentDemand *>>
      demandsByDomain;
  for (const SpatialTemporalSwitchSegmentDemand &demand : *demands)
    demandsByDomain[demand.domain].push_back(&demand);
  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  std::map<PnrIndex, std::vector<std::vector<std::uint64_t>>> rows;
  for (const auto &[domain, domainDemands] : demandsByDomain) {
    if (domain >= matchDomains.size())
      return invalid("switch demand names an absent match domain");
    std::vector<
        std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>>
        signatureStorage;
    signatureStorage.reserve(domainDemands.size());
    for (const SpatialTemporalSwitchSegmentDemand *demand : domainDemands) {
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
      const SpatialTemporalSwitchSegmentDemand &demand =
          *domainDemands[ordinal];
      if (demand.logicalNet >= tagValues.size() ||
          demand.segment >= tagValues[demand.logicalNet].size() ||
          !tagValues[demand.logicalNet][demand.segment]) {
        allTagsAssigned = false;
        candidateDemandViews.push_back({{signatures}, std::nullopt});
        continue;
      }
      const llvm::APInt &tag = *tagValues[demand.logicalNet][demand.segment];
      if (!::fabric::isRepresentablePhysicalTagValue(
              matchDomains[domain].tagWidthBits, tag))
        return invalid("switch route demand has an out-of-range Physical Tag");
      llvm::APInt normalized =
          tag.zextOrTrunc(matchDomains[domain].tagWidthBits);
      taggedDemandViews.push_back({{signatures}, normalized});
      candidateDemandViews.push_back({{signatures}, std::move(normalized)});
    }
    auto &domainRows = rows[domain];
    if (allTagsAssigned) {
      auto projected = ::loom::fabric::projectFabricTemporalSwitchRouteRows(
          taggedDemandViews);
      if (!projected)
        return projected.takeError();
      domainRows.reserve(projected->size());
      for (auto &row : *projected)
        domainRows.push_back(std::move(row.demandOrdinals));
    } else {
      auto projected =
          ::loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(
              candidateDemandViews);
      if (!projected)
        return projected.takeError();
      domainRows.reserve(projected->size());
      for (auto &row : *projected)
        domainRows.push_back(std::move(row.demandOrdinals));
    }
  }

  const FrozenSpatialHandshakeIndex &handshake = problem.handshake();
  std::vector<PnrIndex> fragments;
  for (const auto &[domain, domainRows] : rows) {
    const auto &domainDemands = demandsByDomain.find(domain)->second;
    for (auto [rowOrdinal, rowDemands] : llvm::enumerate(domainRows)) {
      const std::optional<std::uint64_t> residentCapacity =
          matchDomains[domain].residentEntryCapacity;
      if (residentCapacity && rowOrdinal >= *residentCapacity)
        continue;
      std::map<::loom::fabric::FabricOrdinal, std::vector<PnrIndex>> byInput;
      for (std::uint64_t demandOrdinal : rowDemands) {
        if (demandOrdinal >= domainDemands.size())
          return invalid("Fabric switch row names an absent route demand");
        const SpatialTemporalSwitchSegmentDemand *demand =
            domainDemands[demandOrdinal];
        for (const SpatialTemporalSwitchInputSignature &signature :
             demand->signatures)
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
                         llvm::Twine(domain) + " row " +
                         llvm::Twine(rowOrdinal) + " input " +
                         llvm::Twine(input) + " has no frozen activation");
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
  }
  llvm::sort(fragments);
  fragments.erase(std::unique(fragments.begin(), fragments.end()),
                  fragments.end());
  return fragments;
}
