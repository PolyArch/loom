#include "SpatialSwitchHandshakeProjection.h"

#include "SpatialSwitchRowPacking.h"
#include "SpatialTagAssignmentState.h"

#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <system_error>
#include <tuple>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

struct loom::pnr::detail::SpatialSwitchHandshakeProjectionScratch::
    Storage final {
  std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>
      signatureViews;
  std::vector<std::size_t> signatureOffsets;
  std::vector<::loom::fabric::FabricTemporalSwitchCandidateRouteDemandView>
      demandViews;
  ::loom::fabric::FabricTemporalSwitchCandidateRouteProjectionScratch
      routeProjectionScratch;
  ::loom::fabric::FabricTemporalSwitchRouteRowMemberSpans rowSpans;
  std::vector<const ::loom::pnr::detail::SpatialTemporalSwitchSegmentDemand *>
      demands;
  std::vector<PnrIndex> participatingNets;
  std::vector<std::pair<::loom::fabric::FabricOrdinal, PnrIndex>>
      inputTraversals;
};

loom::pnr::detail::SpatialSwitchHandshakeProjectionScratch::
    SpatialSwitchHandshakeProjectionScratch()
    : storage_(std::make_unique<Storage>()) {}
loom::pnr::detail::SpatialSwitchHandshakeProjectionScratch::
    ~SpatialSwitchHandshakeProjectionScratch() = default;

namespace {
using ProjectionStorage =
    ::loom::pnr::detail::SpatialSwitchHandshakeProjectionScratch::Storage;
} // namespace

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
llvm::Error
projectDomainFragments(const FrozenSpatialPnrProblem &problem, PnrIndex domain,
                       llvm::ArrayRef<const SwitchDemand *> demands,
                       TagLookup lookupTag, ProjectionStorage &storage,
                       std::vector<PnrIndex> &fragments) {
  fragments.clear();
  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  if (domain >= matchDomains.size())
    return invalid("switch demand names an absent match domain");
  if (matchDomains[domain].kind !=
      ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
    return invalid("switch demand names a non-switch match domain");

  std::size_t signatureCount = 0;
  for (const SwitchDemand *demand : demands) {
    if (!demand || demand->domain != domain)
      return invalid("switch demand grouping is inconsistent");
    signatureCount += demand->signatures.size();
  }
  storage.signatureViews.clear();
  storage.signatureViews.reserve(signatureCount);
  storage.signatureOffsets.clear();
  storage.signatureOffsets.reserve(demands.size() + 1);
  storage.signatureOffsets.push_back(0);
  for (const SwitchDemand *demand : demands) {
    for (const SpatialTemporalSwitchInputSignature &signature :
         demand->signatures)
      storage.signatureViews.push_back(
          {signature.occurrence, signature.input, signature.outputs});
    storage.signatureOffsets.push_back(storage.signatureViews.size());
  }

  storage.demandViews.clear();
  storage.demandViews.reserve(demands.size());
  for (auto [ordinal, demand] : llvm::enumerate(demands)) {
    auto tag = lookupTag(*demand);
    if (!tag)
      return tag.takeError();
    if (*tag) {
      if (!::fabric::isRepresentablePhysicalTagValue(
              matchDomains[domain].tagWidthBits, **tag))
        return invalid("switch route demand has an out-of-range Physical Tag");
      storage.demandViews.push_back(
          {{llvm::ArrayRef(storage.signatureViews)
                .slice(storage.signatureOffsets[ordinal],
                       storage.signatureOffsets[ordinal + 1] -
                           storage.signatureOffsets[ordinal])},
           *tag});
    } else {
      storage.demandViews.push_back(
          {{llvm::ArrayRef(storage.signatureViews)
                .slice(storage.signatureOffsets[ordinal],
                       storage.signatureOffsets[ordinal + 1] -
                           storage.signatureOffsets[ordinal])},
           nullptr});
    }
  }
  if (llvm::Error error = ::loom::fabric::
          projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
              storage.demandViews, storage.rowSpans,
              storage.routeProjectionScratch))
    return error;
  const std::size_t rowCount = storage.rowSpans.rowOffsets.empty()
                                   ? 0
                                   : storage.rowSpans.rowOffsets.size() - 1;

  const FrozenSpatialHandshakeIndex &handshake = problem.handshake();
  for (std::size_t rowOrdinal = 0; rowOrdinal < rowCount; ++rowOrdinal) {
    const std::optional<std::uint64_t> residentCapacity =
        matchDomains[domain].residentEntryCapacity;
    if (residentCapacity && rowOrdinal >= *residentCapacity)
      continue;
    const llvm::ArrayRef<std::uint64_t> rowDemands =
        llvm::ArrayRef<std::uint64_t>(storage.rowSpans.demandOrdinals)
            .slice(storage.rowSpans.rowOffsets[rowOrdinal],
                   storage.rowSpans.rowOffsets[rowOrdinal + 1] -
                       storage.rowSpans.rowOffsets[rowOrdinal]);
    auto &inputTraversals = storage.inputTraversals;
    inputTraversals.clear();
    for (std::uint64_t demandOrdinal : rowDemands) {
      if (demandOrdinal >= demands.size())
        return invalid("Fabric switch row names an absent route demand");
      for (const SpatialTemporalSwitchInputSignature &signature :
           demands[demandOrdinal]->signatures)
        for (PnrIndex traversal : signature.traversals)
          inputTraversals.push_back({signature.input, traversal});
    }
    llvm::sort(inputTraversals);
    inputTraversals.erase(
        std::unique(inputTraversals.begin(), inputTraversals.end()),
        inputTraversals.end());
    for (std::size_t begin = 0; begin < inputTraversals.size();) {
      const ::loom::fabric::FabricOrdinal input = inputTraversals[begin].first;
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
      for (; begin < inputTraversals.size() &&
             inputTraversals[begin].first == input;
           ++begin) {
        const auto *selection = findTraversal(handshake, *activation,
                                              inputTraversals[begin].second);
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
  return llvm::Error::success();
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
  std::vector<PnrIndex> domainFragments;
  SpatialSwitchHandshakeProjectionScratch localScratch;
  for (PnrIndex domain = 0; domain < demandsByDomain.size(); ++domain) {
    if (demandsByDomain[domain].empty())
      continue;
    if (llvm::Error error = projectDomainFragments(
            problem, domain, demandsByDomain[domain],
            [&](const SwitchDemand &demand)
                -> llvm::Expected<const llvm::APInt *> {
              if (demand.logicalNet >= tagValues.size() ||
                  demand.segment >= tagValues[demand.logicalNet].size())
                return invalid(
                    "switch demand names an absent Physical Tag segment");
              const auto &value = tagValues[demand.logicalNet][demand.segment];
              return value ? &*value : nullptr;
            },
            localScratch.storage(), domainFragments))
      return std::move(error);
    fragments.insert(fragments.end(), domainFragments.begin(),
                     domainFragments.end());
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
  SpatialSwitchHandshakeProjectionScratch localScratch;
  for (PnrIndex domain = 0; domain < grouped->size(); ++domain) {
    if ((*grouped)[domain].empty())
      continue;
    if (llvm::Error error = projectDomainFragments(
            problem, domain, (*grouped)[domain],
            [&](const SwitchDemand &demand) {
              return assignmentTag(assignments, demand);
            },
            localScratch.storage(), fragments[domain]))
      return std::move(error);
  }
  return fragments;
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeDomainFragments(
    const FrozenSpatialPnrProblem &problem, PnrIndex domain,
    const SpatialTagAssignmentStateStorage &assignments) {
  SpatialSwitchHandshakeProjectionScratch scratch;
  return deriveSpatialTemporalSwitchHandshakeDomainFragments(
      problem, domain, assignments, scratch);
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeDomainFragments(
    const FrozenSpatialPnrProblem &problem, PnrIndex domain,
    const SpatialTagAssignmentStateStorage &assignments,
    SpatialSwitchHandshakeProjectionScratch &scratch) {
  std::vector<PnrIndex> fragments;
  if (llvm::Error error = deriveSpatialTemporalSwitchHandshakeDomainFragments(
          problem, domain, assignments, scratch, fragments))
    return std::move(error);
  return fragments;
}

llvm::Error
loom::pnr::detail::deriveSpatialTemporalSwitchHandshakeDomainFragments(
    const FrozenSpatialPnrProblem &problem, PnrIndex domain,
    const SpatialTagAssignmentStateStorage &assignments,
    SpatialSwitchHandshakeProjectionScratch &scratch,
    std::vector<PnrIndex> &fragments) {
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
  // Every demand in this domain belongs to a segment whose vertex the
  // interference projection lists for the same domain, so the domain's
  // vertex inventory bounds the participating logical nets. Ascending net
  // order preserves the canonical demand order of a full scan.
  const auto vertices = assignments.interference.domainVertices(domain);
  std::vector<PnrIndex> &participatingNets =
      scratch.storage().participatingNets;
  participatingNets.clear();
  participatingNets.reserve(vertices.size());
  for (const SpatialTagVertexRef &vertex : vertices)
    participatingNets.push_back(vertex.logicalNet);
  llvm::sort(participatingNets);
  participatingNets.erase(
      std::unique(participatingNets.begin(), participatingNets.end()),
      participatingNets.end());
  std::vector<const SwitchDemand *> &demands = scratch.storage().demands;
  demands.clear();
  demands.reserve(vertices.size());
  for (PnrIndex logicalNet : participatingNets) {
    if (logicalNet >= assignments.nets.size())
      return invalid("Tag-assignment switch demand is inconsistent");
    for (const SwitchDemand &demand :
         assignments.interference.switchDemands(logicalNet)) {
      if (demand.logicalNet != logicalNet ||
          demand.domain >= matchDomains.size())
        return invalid("Tag-assignment switch demand is inconsistent");
      if (demand.domain == domain)
        demands.push_back(&demand);
    }
  }
  return projectDomainFragments(
      problem, domain, demands,
      [&](const SwitchDemand &demand) {
        return assignmentTag(assignments, demand);
      },
      scratch.storage(), fragments);
}
