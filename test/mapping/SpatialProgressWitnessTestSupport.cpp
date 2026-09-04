#include "TechMappingCandidateTestSupport.h"

#include "PnR/EndpointRouter.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialCandidateInitializer.h"

#include "SpatialProgressIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "spatial progress witness test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

struct RoutedNet final {
  loom::pnr::PnrIndex logicalNet = loom::pnr::getInvalidPnrIndex();
  loom::pnr::PnrIndex source = loom::pnr::getInvalidPnrIndex();
  loom::pnr::PnrIndex target = loom::pnr::getInvalidPnrIndex();
  std::vector<loom::pnr::PnrIndex> arcs;
};

std::optional<RoutedNet>
findOwnerRoute(loom::pnr::EndpointRouteSearchScratch &search,
               const loom::pnr::FrozenSpatialPnrProblem &problem,
               const loom::pnr::SpatialCandidateState &candidate,
               loom::pnr::PnrIndex logicalNet, loom::pnr::PnrIndex owner,
               llvm::ArrayRef<loom::pnr::RouteCost> routeCosts) {
  using namespace loom::pnr;
  const FrozenSpatialLogicalNet &net =
      problem.transfers().logicalNets()[logicalNet];
  if (net.sinkCount != 1)
    return std::nullopt;
  const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
  const PnrIndex target = candidate.logicalNetSinkEndpoint(logicalNet, 0);
  if (source == target)
    return std::nullopt;

  std::vector<std::uint64_t> requiredTraversals(
      (problem.routing().traversals().size() + 63) / 64, 0);
  for (PnrIndex traversal : problem.progressIndex().traversalsForOwner(owner))
    requiredTraversals[traversal / 64] |= std::uint64_t{1} << (traversal % 64);
  if (llvm::all_of(requiredTraversals,
                   [](std::uint64_t bits) { return bits == 0; }))
    return std::nullopt;

  const PnrIndex unrestrictedReplicationGroup = getInvalidPnrIndex();
  const PnrIndex firstTargetRank = 0;
  const std::uint8_t targetRequiresTraversal = 1;
  EndpointRouteSearchRequest request;
  request.sourceEndpoints = {&source, 1};
  request.sourceReplicationGroups = {&unrestrictedReplicationGroup, 1};
  request.targetEndpoints = {&target, 1};
  request.targetPreferenceRanks = {&firstTargetRank, 1};
  request.lowerBoundArcCosts = routeCosts;
  request.currentArcCosts = routeCosts;
  request.requiredPayloadWidthBits =
      candidate.logicalNetPayloadWidth(logicalNet);
  request.endpointExpansionLimit = UINT64_C(262144);
  request.requiredTraversalBits = requiredTraversals;
  request.targetRequiresTraversal = {&targetRequiresTraversal, 1};
  auto routed = search.search(request);
  if (!routed) {
    llvm::consumeError(routed.takeError());
    return std::nullopt;
  }
  return RoutedNet{logicalNet, source, target,
                   std::vector<PnrIndex>(routed->forwardArcs.begin(),
                                         routed->forwardArcs.end())};
}

bool commitRoute(loom::pnr::SpatialCandidateState &candidate,
                 loom::pnr::SpatialCandidateScratch &scratch,
                 const RoutedNet &route) {
  auto move = candidate.beginMove(scratch);
  if (!move) {
    llvm::consumeError(move.takeError());
    return false;
  }
  const auto reject = [&](llvm::Error error) {
    llvm::consumeError(std::move(error));
    move->rollback();
    return false;
  };
  if (llvm::Error error = move->bindRouteSource(route.logicalNet, route.source))
    return reject(std::move(error));
  if (llvm::Error error =
          move->bindRouteSink(route.logicalNet, 0, route.target))
    return reject(std::move(error));
  if (llvm::Error error =
          move->attachRoutePath(route.logicalNet, route.source, route.arcs, 0))
    return reject(std::move(error));
  auto closed = move->close();
  if (!closed)
    return reject(closed.takeError());
  if (!*closed) {
    move->rollback();
    return false;
  }
  if (llvm::Error error = move->commit()) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

struct ConflictFixture final {
  loom::pnr::SpatialCandidateStateHandle candidate;
  loom::pnr::PnrIndex owner = loom::pnr::getInvalidPnrIndex();
  std::array<loom::pnr::PnrIndex, 2> logicalNets{};
};

ConflictFixture
buildConflictFixture(const loom::pnr::FrozenSpatialPnrProblemHandle &problem,
                     ::fabric::FifoQueueDiscipline discipline) {
  using namespace loom::pnr;
  auto probeCandidate = take(createCanonicalSpatialCandidate(problem));
  EndpointRouteSearchScratch search;
  requireSuccess(
      search.prepare(endpointRoutingGraphView(problem->routing().topology())));
  const std::vector<RouteCost> routeCosts(
      problem->routing().routingArcs().size(), 1);

  for (PnrIndex owner = 0;
       owner < problem->progressIndex().finiteBufferOwners().size(); ++owner) {
    if (problem->progressIndex().ownerQueueDisciplines()[owner] != discipline)
      continue;
    std::vector<RoutedNet> routes;
    for (PnrIndex logicalNet = 0;
         logicalNet < problem->transfers().logicalNets().size(); ++logicalNet) {
      auto route = findOwnerRoute(search, *problem, *probeCandidate, logicalNet,
                                  owner, routeCosts);
      if (route)
        routes.push_back(std::move(*route));
    }
    for (std::size_t first = 0; first < routes.size(); ++first)
      for (std::size_t second = first + 1; second < routes.size(); ++second) {
        auto candidate = take(createCanonicalSpatialCandidate(problem));
        SpatialCandidateScratch scratch;
        requireSuccess(scratch.prepare(*problem));
        if (!commitRoute(*candidate, scratch, routes[first]) ||
            !commitRoute(*candidate, scratch, routes[second]))
          continue;
        if (!candidate->progress().finiteBufferOwnerConflicts(owner))
          fail("two owner-routed nets did not create a progress conflict");
        requireSuccess(candidate->verify());
        return {std::move(candidate),
                owner,
                {routes[first].logicalNet, routes[second].logicalNet}};
      }
  }
  fail("fixture has no two routable nets sharing a Buffered FIFO owner");
}

} // namespace

void loom::test::exerciseSpatialProgressWitnessClosure(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    ::fabric::FifoQueueDiscipline discipline) {
  using namespace pnr;
  ConflictFixture fixture = buildConflictFixture(problem, discipline);
  SpatialFiniteBufferConflictWitness witness;
  requireSuccess(fixture.candidate->rebuildFiniteBufferConflictWitness(
      fixture.owner, witness));
  if (witness.ownerOrdinal != fixture.owner ||
      witness.owner !=
          problem->progressIndex().finiteBufferOwners()[fixture.owner] ||
      witness.competingLogicalNets.size() != fixture.logicalNets.size())
    fail("typed progress witness changed its owner or competing nets");
  for (PnrIndex logicalNet : fixture.logicalNets)
    if (!llvm::is_contained(witness.competingLogicalNets, logicalNet) ||
        !llvm::any_of(witness.routeAnchors, [&](const auto &anchor) {
          return anchor.logicalNet == logicalNet &&
                 problem->progressIndex().traversalOwner(anchor.traversal) ==
                     fixture.owner;
        }))
      fail("typed progress witness omitted a net route anchor");

  std::vector<PnrIndex> conflictOwners;
  requireSuccess(
      fixture.candidate->progress().enumerateFiniteBufferConflictOwners(
          conflictOwners));
  if (!llvm::is_contained(conflictOwners, fixture.owner) ||
      fixture.candidate->progress().firstFiniteBufferConflictOwner() !=
          conflictOwners.front())
    fail("conflicting-owner bitset enumeration is not canonical");

  SpatialActionDomainScratch domain;
  requireSuccess(domain.prepare(*problem));
  requireSuccess(domain.rebuild(*fixture.candidate));
  const bool unknownBound =
      discipline == ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
  const auto depth =
      problem->progressIndex().ownerSharedSlotCapacities()[fixture.owner];
  const auto expectedGap = unknownBound || depth >= fixture.logicalNets.size()
                               ? 0
                               : fixture.logicalNets.size() - depth;
  const bool hasDebt = unknownBound || expectedGap != 0;
  if (fixture.candidate->progress().capacityProofDebtOwner(fixture.owner) !=
          unknownBound ||
      fixture.candidate->progress().capacityShortfall(fixture.owner) !=
          expectedGap)
    fail("capacity owner lost the known-bound versus unknown-bound distinction");
  const SpatialMappingAction progressAction =
      SpatialTransportRoutingAction{SpatialWitnessRegionRoutingAction{
          ResolvedPnrViolationKind::ProgressProofDebt, fixture.owner}};
  const SpatialActionKey progressKey = spatialActionKey(progressAction);
  const bool hasAction =
      llvm::any_of(domain.view().transportChoices, [&](const auto &action) {
        return spatialActionKey(SpatialMappingAction{
                   SpatialTransportRoutingAction{action}}) == progressKey;
      });
  if (hasAction != hasDebt)
    fail("capacity proof debt and its Action domain disagree");
  for (const auto &action : domain.view().transportChoices)
    if (const auto *region =
            std::get_if<SpatialWitnessRegionRoutingAction>(&action))
      if (region->witnessKind == ResolvedPnrViolationKind::HardProgressViolation)
        fail("a capacity bound became a hard progress routing Action");
  SpatialFiniteBufferConflictWitness debtWitness;
  auto reconstructed = fixture.candidate->rebuildCapacityProofDebtWitness(
      fixture.owner, debtWitness);
  if (hasDebt) {
    requireSuccess(std::move(reconstructed));
    if (debtWitness.owner != witness.owner ||
        debtWitness.competingLogicalNets != witness.competingLogicalNets ||
        debtWitness.routeAnchors.size() != witness.routeAnchors.size())
      fail("capacity debt reconstruction lost its exact owner route closure");
  } else {
    if (!reconstructed)
      fail("a sufficient capacity accepted a live proof-debt witness");
    llvm::consumeError(std::move(reconstructed));
  }
  requireSuccess(fixture.candidate->verify());

  const SpatialProgressStatistics &statistics =
      fixture.candidate->progress().statistics();
  if (statistics.incrementalUpdateCount == 0 ||
      statistics.coldVerificationCount == 0 ||
      statistics.coldProgressScanCount == 0)
    fail("progress summary counters did not observe incremental and cold work");
}
