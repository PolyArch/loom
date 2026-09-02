#include "PnR/EndpointRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>

using namespace loom::pnr;

namespace {

void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return *value;
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

void requirePath(const char *test, const EndpointRouteSearchResult &result,
                 PnrIndex source, PnrIndex target, RouteCost cost,
                 llvm::ArrayRef<PnrIndex> arcs) {
  if (result.source != source || result.target != target ||
      result.cost != cost || result.forwardArcs != arcs)
    fail(test, "route result does not match the canonical shortest path");
}

void expectFailure(const char *test,
                   llvm::Expected<EndpointRouteSearchResult> result,
                   EndpointRouteSearchFailureKind expected) {
  if (result)
    fail(test, "expected route search failure");
  bool matched = false;
  llvm::handleAllErrors(result.takeError(),
                        [&](const EndpointRouteSearchFailure &error) {
                          matched = error.kind() == expected;
                        });
  if (!matched)
    fail(test, "route search returned the wrong failure kind");
}

struct Fixture final {
  static constexpr PnrIndex noReplicationGroup = getInvalidPnrIndex();

  std::array<EndpointRoutingArc, 7> arcs{{
      {1, 0, 64, 8},
      {2, 1, 64, 8},
      {4, 2, 64, 8},
      {3, 3, 16, 8},
      {5, 4, 64, 8},
      {4, 5, 64, 8},
      {4, 6, 64, 8},
  }};
  std::array<PnrIndex, 7> arcSources{{0, 0, 1, 2, 2, 3, 5}};
  std::array<PnrIndex, 7> adjacencyOffsets{{0, 2, 3, 5, 6, 6, 7}};
  std::array<PnrIndex, 7> reverseAdjacencyOffsets{{0, 0, 1, 2, 3, 6, 7}};
  std::array<PnrIndex, 7> reverseArcOrdinals{{0, 1, 3, 2, 5, 6, 4}};
  std::array<PnrIndex, 7> traversalReplicationGroups{{
      noReplicationGroup,
      noReplicationGroup,
      noReplicationGroup,
      7,
      8,
      noReplicationGroup,
      noReplicationGroup,
  }};
  std::array<RouteCost, 7> lowerCosts{{10, 1, 1, 1, 1, 1, 1}};
  std::array<RouteCost, 7> currentCosts{{10, 1, 1, 1, 1, 1, 1}};

  EndpointRoutingGraphView graph() const {
    return {6,
            arcs,
            arcSources,
            adjacencyOffsets,
            reverseAdjacencyOffsets,
            reverseArcOrdinals,
            traversalReplicationGroups};
  }
};

EndpointRouteSearchRequest request(
    const Fixture &fixture, llvm::ArrayRef<PnrIndex> sources,
    llvm::ArrayRef<PnrIndex> sourceGroups, llvm::ArrayRef<PnrIndex> targets,
    llvm::ArrayRef<PnrIndex> targetRanks, std::uint32_t payloadWidth,
    std::uint64_t expansionLimit,
    const EndpointRouteInputRevisionOwner *lowerBoundRevisionOwner = nullptr,
    const EndpointRouteInputRevisionOwner *currentRevisionOwner = nullptr) {
  EndpointRouteSearchRequest result;
  result.sourceEndpoints = sources;
  result.sourceReplicationGroups = sourceGroups;
  result.targetEndpoints = targets;
  result.targetPreferenceRanks = targetRanks;
  result.lowerBoundArcCosts = fixture.lowerCosts;
  result.currentArcCosts = fixture.currentCosts;
  result.requiredPayloadWidthBits = payloadWidth;
  result.endpointExpansionLimit = expansionLimit;
  if (lowerBoundRevisionOwner)
    result.lowerBoundArcCostRevision = lowerBoundRevisionOwner->revision();
  if (currentRevisionOwner)
    result.currentArcCostRevision = currentRevisionOwner->revision();
  return result;
}

void arbitraryTopologyAndCanonicalTieBreak() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  if (scratch.endpointExpansionCount() != 0)
    fail(__func__, "fresh route scratch retained endpoint work");
  const std::size_t retained = scratch.retainedStorageBytes();

  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> unrestricted{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targetFour{{4}};
  const std::array<PnrIndex, 1> rankZero{{0}};
  const std::array<PnrIndex, 3> expected{{1, 3, 5}};
  const auto result =
      take(__func__, scratch.search(request(fixture, sources, unrestricted,
                                            targetFour, rankZero, 1, 64)));
  requirePath(__func__, result, 0, 4, 3, expected);

  const std::array<PnrIndex, 2> equalTargets{{3, 5}};
  const std::array<PnrIndex, 2> targetRanks{{1, 0}};
  const std::array<PnrIndex, 2> preferredTargetPath{{1, 4}};
  const auto tied =
      take(__func__, scratch.search(request(fixture, sources, unrestricted,
                                            equalTargets, targetRanks, 1, 64)));
  requirePath(__func__, tied, 0, 5, 2, preferredTargetPath);

  const std::array<PnrIndex, 1> branchSource{{2}};
  const std::array<PnrIndex, 1> requiredGroup{{8}};
  const std::array<PnrIndex, 2> branchRanks{{0, 1}};
  const std::array<PnrIndex, 1> groupEightPath{{4}};
  const auto groupFiltered = take(
      __func__, scratch.search(request(fixture, branchSource, requiredGroup,
                                       equalTargets, branchRanks, 1, 64)));
  requirePath(__func__, groupFiltered, 2, 5, 1, groupEightPath);
  if (scratch.retainedStorageBytes() != retained)
    fail(__func__, "warm route search changed retained scratch storage");
}

void widthFilteringAndWorkLimit() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};

  const std::array<PnrIndex, 3> widePath{{1, 4, 6}};
  const auto wide =
      take(__func__, scratch.search(request(fixture, sources, sourceGroups,
                                            targets, targetRanks, 32, 64)));
  requirePath(__func__, wide, 0, 4, 3, widePath);
  if (scratch.endpointExpansionCount() == 0)
    fail(__func__, "successful route did not report endpoint expansions");

  expectFailure(__func__,
                scratch.search(request(fixture, sources, sourceGroups, targets,
                                       targetRanks, 128, 64)),
                EndpointRouteSearchFailureKind::Unreachable);
  const std::uint64_t beforeLimitedSearch = scratch.endpointExpansionCount();
  expectFailure(__func__,
                scratch.search(request(fixture, sources, sourceGroups, targets,
                                       targetRanks, 1, 1)),
                EndpointRouteSearchFailureKind::WorkLimit);
  if (scratch.endpointExpansionCount() != beforeLimitedSearch + 1)
    fail(__func__, "work-limited route did not report consumed expansion");
}

void requiredTraversalProductState() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};
  const std::array<std::uint64_t, 1> required{{std::uint64_t{1} << 4}};
  const std::array<PnrIndex, 3> expected{{1, 4, 6}};

  auto constrained =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 64);
  const std::array<std::uint8_t, 1> targetRequiresTraversal{{1}};
  constrained.targetRequiresTraversal = targetRequiresTraversal;
  constrained.requiredTraversalBits = required;
  const auto result = take(__func__, scratch.search(constrained));
  requirePath(__func__, result, 0, 4, 3, expected);

  const std::array<std::uint64_t, 1> excludesRequired{{
      ((std::uint64_t{1} << fixture.arcs.size()) - 1) &
          ~(std::uint64_t{1} << 4),
  }};
  constrained.eligibleTraversalBits = excludesRequired;
  expectFailure(__func__, scratch.search(constrained),
                EndpointRouteSearchFailureKind::Unreachable);

  const std::array<PnrIndex, 2> mixedTargets{{3, 4}};
  const std::array<PnrIndex, 2> mixedRanks{{0, 1}};
  const std::array<std::uint8_t, 2> mixedRequirements{{1, 0}};
  auto mixed =
      request(fixture, sources, sourceGroups, mixedTargets, mixedRanks, 1, 64);
  mixed.targetRequiresTraversal = mixedRequirements;
  mixed.requiredTraversalBits = required;
  const std::array<PnrIndex, 3> unrestrictedTargetPath{{1, 3, 5}};
  requirePath(__func__, take(__func__, scratch.search(mixed)), 0, 4, 3,
              unrestrictedTargetPath);
}

void checkedCostAndAdmissibility() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};

  fixture.currentCosts[1] = 0;
  expectFailure(__func__,
                scratch.search(request(fixture, sources, sourceGroups, targets,
                                       targetRanks, 1, 64)),
                EndpointRouteSearchFailureKind::Invalid);

  fixture.currentCosts = fixture.lowerCosts;
  fixture.lowerCosts[5] = maxFiniteRouteCost;
  fixture.currentCosts[5] = maxFiniteRouteCost;
  expectFailure(__func__,
                scratch.search(request(fixture, sources, sourceGroups, targets,
                                       targetRanks, 1, 64)),
                EndpointRouteSearchFailureKind::ArithmeticOverflow);
}

void timingAwareArrivalAndBoundary() {
  Fixture fixture;
  fixture.lowerCosts.fill(1);
  fixture.currentCosts.fill(1);
  EndpointRouteSearchScratch scratch;
  EndpointRouteInputRevisionOwner lowerBoundRevisionOwner;
  EndpointRouteInputRevisionOwner currentRevisionOwner;
  EndpointRouteInputRevisionOwner timingRevisionOwner;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};
  const std::array<std::uint64_t, 7> delays{{8, 5, 8, 5, 1, 5, 1}};
  std::array<std::uint8_t, 7> boundaries{};
  const std::array<std::uint64_t, 1> sourceArrivals{{0}};
  const std::array<std::uint64_t, 1> targetDelays{{0}};
  auto combinational =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 128,
              &lowerBoundRevisionOwner, &currentRevisionOwner);
  combinational.physicalTimingEnabled = true;
  combinational.physicalTimingRevision = timingRevisionOwner.revision();
  combinational.arcTimingDelayQuanta = delays;
  combinational.arcTimingRegisteredDestination = boundaries;
  combinational.sourceTimingArrivalQuanta = sourceArrivals;
  combinational.targetTimingDelayQuanta = targetDelays;
  combinational.requiredTimingQuanta = 8;
  combinational.timingCriticality = 2;
  const std::array<PnrIndex, 3> timingPreferredPath{{1, 4, 6}};
  const auto combinationalResult =
      take(__func__, scratch.search(combinational));
  if (combinationalResult.forwardArcs !=
      llvm::ArrayRef<PnrIndex>(timingPreferredPath))
    fail(__func__, "candidate path arrival did not affect route selection");
  const auto repeatedCombinationalResult =
      take(__func__, scratch.search(combinational));
  if (repeatedCombinationalResult.forwardArcs !=
      combinationalResult.forwardArcs)
    fail(__func__, "stable timing projection changed its canonical path");

  const std::array<std::uint64_t, 7> recharacterizedDelays{
      {1, 5, 1, 5, 8, 5, 8}};
  requireSuccess(__func__, timingRevisionOwner.advance());
  auto recharacterized = combinational;
  recharacterized.physicalTimingRevision = timingRevisionOwner.revision();
  recharacterized.arcTimingDelayQuanta = recharacterizedDelays;
  const std::array<PnrIndex, 2> recharacterizedPath{{0, 2}};
  const auto recharacterizedResult =
      take(__func__, scratch.search(recharacterized));
  if (recharacterizedResult.forwardArcs !=
      llvm::ArrayRef<PnrIndex>(recharacterizedPath))
    fail(__func__,
         "recharacterized provider delays did not replay a changed route");

  const std::array<std::uint64_t, 1> lateRegisteredSourceArrival{{8}};
  requireSuccess(__func__, timingRevisionOwner.advance());
  auto lateCombinational = combinational;
  lateCombinational.physicalTimingRevision = timingRevisionOwner.revision();
  lateCombinational.sourceTimingArrivalQuanta = lateRegisteredSourceArrival;
  const auto lateCombinationalResult =
      take(__func__, scratch.search(lateCombinational));
  requireSuccess(__func__, timingRevisionOwner.advance());
  boundaries[1] = 1;
  auto registered = lateCombinational;
  registered.physicalTimingRevision = timingRevisionOwner.revision();
  registered.arcTimingRegisteredDestination = boundaries;
  const auto registeredResult = take(__func__, scratch.search(registered));
  if (registeredResult.forwardArcs !=
          llvm::ArrayRef<PnrIndex>(timingPreferredPath) ||
      registeredResult.cost >= lateCombinationalResult.cost)
    fail(__func__, "registered boundary did not reduce candidate path slack");

  const std::array<PnrIndex, 1> overlappingTarget{{0}};
  const std::array<std::uint64_t, 1> lateSourceArrival{{10}};
  const std::array<std::uint64_t, 1> localTargetDelay{{2}};
  auto terminal =
      request(fixture, sources, sourceGroups, overlappingTarget, targetRanks, 1,
              128, &lowerBoundRevisionOwner, &currentRevisionOwner);
  terminal.physicalTimingEnabled = true;
  terminal.physicalTimingRevision = timingRevisionOwner.revision();
  terminal.arcTimingDelayQuanta = delays;
  terminal.arcTimingRegisteredDestination = boundaries;
  terminal.sourceTimingArrivalQuanta = lateSourceArrival;
  terminal.targetTimingDelayQuanta = localTargetDelay;
  terminal.requiredTimingQuanta = 8;
  terminal.timingCriticality = 0;
  const auto terminalResult = take(__func__, scratch.search(terminal));
  const RouteCost expectedTerminalCost =
      (RouteCost{4} * routeCostScale + 7) / 8;
  if (!terminalResult.forwardArcs.empty() ||
      terminalResult.cost != expectedTerminalCost)
    fail(__func__,
         "source arrival and sink-local traversal slack were not counted "
         "exactly once");
  if (scratch.heuristicBuildCount() != 5 ||
      scratch.heuristicCacheHitCount() != 1)
    fail(__func__,
         "timing-aware search did not reuse its admissible route heuristic");
}

void failureKindSpellings() {
  const std::array<std::pair<EndpointRouteSearchFailureKind, llvm::StringRef>,
                   4>
      expected{{
          {EndpointRouteSearchFailureKind::Invalid, "invalid"},
          {EndpointRouteSearchFailureKind::ArithmeticOverflow,
           "arithmetic_overflow"},
          {EndpointRouteSearchFailureKind::Unreachable, "unreachable"},
          {EndpointRouteSearchFailureKind::WorkLimit, "work_limit"},
      }};
  for (auto [kind, spelling] : expected)
    if (stringifyEndpointRouteSearchFailureKind(kind) != spelling)
      fail(__func__, "failure kind spelling is not canonical");
}

void exactHeuristicCacheInvalidation() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  EndpointRouteInputRevisionOwner lowerBoundRevisionOwner;
  EndpointRouteInputRevisionOwner currentRevisionOwner;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};
  const std::array<PnrIndex, 3> initialPath{{1, 3, 5}};

  auto cachedRequest =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 64,
              &lowerBoundRevisionOwner, &currentRevisionOwner);
  requirePath(__func__, take(__func__, scratch.search(cachedRequest)), 0, 4, 3,
              initialPath);
  if (scratch.heuristicBuildCount() != 1 ||
      scratch.heuristicCacheHitCount() != 0)
    fail(__func__, "cold exact heuristic query did not build once");
  const std::size_t warmStorage = scratch.retainedStorageBytes();
  requirePath(__func__, take(__func__, scratch.search(cachedRequest)), 0, 4, 3,
              initialPath);
  if (scratch.heuristicBuildCount() != 1 ||
      scratch.heuristicCacheHitCount() != 1 ||
      scratch.retainedStorageBytes() != warmStorage)
    fail(__func__, "warm exact heuristic query did not reuse stable storage");

  std::array<RouteCost, 7> reboundLowerCosts = fixture.lowerCosts;
  std::array<RouteCost, 7> reboundCurrentCosts = fixture.currentCosts;
  reboundLowerCosts[3] = 20;
  reboundCurrentCosts[3] = 20;
  requireSuccess(__func__, lowerBoundRevisionOwner.advance());
  requireSuccess(__func__, currentRevisionOwner.advance());
  auto reboundRequest = cachedRequest;
  reboundRequest.lowerBoundArcCostRevision = lowerBoundRevisionOwner.revision();
  reboundRequest.currentArcCostRevision = currentRevisionOwner.revision();
  reboundRequest.lowerBoundArcCosts = reboundLowerCosts;
  reboundRequest.currentArcCosts = reboundCurrentCosts;
  const std::array<PnrIndex, 3> revisedPath{{1, 4, 6}};
  requirePath(__func__, take(__func__, scratch.search(reboundRequest)), 0, 4, 3,
              revisedPath);
  if (scratch.heuristicBuildCount() != 2 ||
      scratch.heuristicCacheHitCount() != 1)
    fail(__func__, "a distinct cost view reused an unrelated heuristic");

  requireSuccess(__func__, lowerBoundRevisionOwner.advance());
  requireSuccess(__func__, currentRevisionOwner.advance());
  fixture.lowerCosts[3] = 20;
  fixture.currentCosts[3] = 20;
  auto revisedRequest =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 64,
              &lowerBoundRevisionOwner, &currentRevisionOwner);
  requirePath(__func__, take(__func__, scratch.search(revisedRequest)), 0, 4, 3,
              revisedPath);
  if (scratch.heuristicBuildCount() != 3 ||
      scratch.heuristicCacheHitCount() != 1)
    fail(__func__, "cost revision reused a stale exact heuristic");
}

void exactHeuristicCachePreservesWideCosts() {
  constexpr RouteCost unit = RouteCost{1} << 40;
  Fixture fixture;
  fixture.lowerCosts = {unit, unit, 100 * unit, unit, unit, unit, unit};
  fixture.currentCosts = fixture.lowerCosts;
  EndpointRouteSearchScratch scratch;
  EndpointRouteInputRevisionOwner lowerBoundRevisionOwner;
  EndpointRouteInputRevisionOwner currentRevisionOwner;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};
  const std::array<PnrIndex, 3> expected{{1, 3, 5}};
  auto cachedRequest =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 64,
              &lowerBoundRevisionOwner, &currentRevisionOwner);

  const std::uint64_t coldBegin = scratch.endpointExpansionCount();
  requirePath(__func__, take(__func__, scratch.search(cachedRequest)), 0, 4,
              3 * unit, expected);
  const std::uint64_t coldExpansions =
      scratch.endpointExpansionCount() - coldBegin;
  const std::uint64_t warmBegin = scratch.endpointExpansionCount();
  requirePath(__func__, take(__func__, scratch.search(cachedRequest)), 0, 4,
              3 * unit, expected);
  const std::uint64_t warmExpansions =
      scratch.endpointExpansionCount() - warmBegin;
  if (coldExpansions != warmExpansions || scratch.heuristicBuildCount() != 1 ||
      scratch.heuristicCacheHitCount() != 1)
    fail(__func__, "wide-cost cache hit changed deterministic route work");
}

void validatedInputRevisionInvalidation() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  EndpointRouteInputRevisionOwner lowerBoundRevisionOwner;
  EndpointRouteInputRevisionOwner currentRevisionOwner;
  EndpointRouteInputRevisionOwner timingRevisionOwner;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};

  auto versioned =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 128,
              &lowerBoundRevisionOwner, &currentRevisionOwner);
  (void)take(__func__, scratch.search(versioned));
  (void)take(__func__, scratch.search(versioned));
  if (scratch.arcCostValidationScanCount() != 1)
    fail(__func__, "stable cost revisions repeated full validation");

  EndpointRouteInputRevisionOwner movedCurrentRevisionOwner(
      std::move(currentRevisionOwner));
  (void)take(__func__, scratch.search(versioned));
  if (scratch.arcCostValidationScanCount() != 1)
    fail(__func__, "owner move invalidated its stable revision state");

  requireSuccess(__func__, movedCurrentRevisionOwner.advance());
  fixture.currentCosts[0] = routeCostInfinity;
  versioned.currentArcCostRevision = movedCurrentRevisionOwner.revision();
  expectFailure(__func__, scratch.search(versioned),
                EndpointRouteSearchFailureKind::Invalid);
  if (scratch.arcCostValidationScanCount() != 2)
    fail(__func__, "a current-cost revision reused prior validation");

  requireSuccess(__func__, movedCurrentRevisionOwner.advance());
  fixture.currentCosts[0] = fixture.lowerCosts[0];
  versioned.currentArcCostRevision = movedCurrentRevisionOwner.revision();
  (void)take(__func__, scratch.search(versioned));
  auto staleCurrent = versioned;
  requireSuccess(__func__, movedCurrentRevisionOwner.advance());
  fixture.currentCosts[0] = routeCostInfinity;
  expectFailure(__func__, scratch.search(staleCurrent),
                EndpointRouteSearchFailureKind::Invalid);
  if (scratch.arcCostValidationScanCount() != 4)
    fail(__func__, "a stale current-cost token reused prior validation");

  requireSuccess(__func__, movedCurrentRevisionOwner.advance());
  fixture.currentCosts[0] = fixture.lowerCosts[0];
  versioned.currentArcCostRevision = movedCurrentRevisionOwner.revision();
  (void)take(__func__, scratch.search(versioned));
  requireSuccess(__func__, lowerBoundRevisionOwner.advance());
  fixture.lowerCosts[0] = fixture.currentCosts[0] + 1;
  versioned.lowerBoundArcCostRevision = lowerBoundRevisionOwner.revision();
  expectFailure(__func__, scratch.search(versioned),
                EndpointRouteSearchFailureKind::Invalid);
  if (scratch.arcCostValidationScanCount() != 6)
    fail(__func__, "a lower-bound revision reused prior validation");

  requireSuccess(__func__, lowerBoundRevisionOwner.advance());
  fixture.lowerCosts[0] = fixture.currentCosts[0];
  versioned.lowerBoundArcCostRevision = lowerBoundRevisionOwner.revision();
  const std::array<std::uint64_t, 7> validDelays{{1, 1, 1, 1, 1, 1, 1}};
  std::array<std::uint64_t, 7> delays = validDelays;
  std::array<std::uint8_t, 7> boundaries{};
  const std::array<std::uint64_t, 1> sourceArrivals{{0}};
  const std::array<std::uint64_t, 1> targetDelays{{0}};
  versioned.physicalTimingEnabled = true;
  versioned.physicalTimingRevision = timingRevisionOwner.revision();
  versioned.arcTimingDelayQuanta = delays;
  versioned.arcTimingRegisteredDestination = boundaries;
  versioned.sourceTimingArrivalQuanta = sourceArrivals;
  versioned.targetTimingDelayQuanta = targetDelays;
  versioned.requiredTimingQuanta = 8;
  (void)take(__func__, scratch.search(versioned));
  (void)take(__func__, scratch.search(versioned));
  if (scratch.physicalTimingValidationScanCount() != 1)
    fail(__func__, "stable timing revision repeated full validation");

  auto staleTiming = versioned;
  requireSuccess(__func__, timingRevisionOwner.advance());
  delays[0] = 0;
  expectFailure(__func__, scratch.search(staleTiming),
                EndpointRouteSearchFailureKind::Invalid);
  requireSuccess(__func__, timingRevisionOwner.advance());
  delays = validDelays;
  boundaries[0] = 2;
  versioned.physicalTimingRevision = timingRevisionOwner.revision();
  expectFailure(__func__, scratch.search(versioned),
                EndpointRouteSearchFailureKind::Invalid);
  if (scratch.physicalTimingValidationScanCount() != 3)
    fail(__func__, "timing input changes reused prior validation");

  requireSuccess(__func__, timingRevisionOwner.advance());
  boundaries[0] = 0;
  versioned.physicalTimingRevision = timingRevisionOwner.revision();
  (void)take(__func__, scratch.search(versioned));

  auto unversionedTiming = versioned;
  unversionedTiming.physicalTimingRevision.reset();
  auto unversionedDelays = delays;
  auto unversionedBoundaries = boundaries;
  unversionedTiming.arcTimingDelayQuanta = unversionedDelays;
  unversionedTiming.arcTimingRegisteredDestination = unversionedBoundaries;
  const std::uint64_t coldTimingValidationBegin =
      scratch.physicalTimingValidationScanCount();
  const std::uint64_t uncachedTimingHeuristicBegin =
      scratch.heuristicBuildCount();
  (void)take(__func__, scratch.search(unversionedTiming));
  (void)take(__func__, scratch.search(unversionedTiming));
  if (scratch.physicalTimingValidationScanCount() !=
      coldTimingValidationBegin + 2)
    fail(__func__, "unversioned timing did not retain cold validation");
  if (scratch.heuristicBuildCount() != uncachedTimingHeuristicBegin + 2)
    fail(__func__, "unversioned timing reused a route heuristic");
  unversionedDelays[0] = 0;
  expectFailure(__func__, scratch.search(unversionedTiming),
                EndpointRouteSearchFailureKind::Invalid);

  auto unversioned =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 128);
  auto unversionedLowerCosts = fixture.lowerCosts;
  auto unversionedCurrentCosts = fixture.currentCosts;
  unversioned.lowerBoundArcCosts = unversionedLowerCosts;
  unversioned.currentArcCosts = unversionedCurrentCosts;
  const std::uint64_t coldValidationBegin =
      scratch.arcCostValidationScanCount();
  (void)take(__func__, scratch.search(unversioned));
  (void)take(__func__, scratch.search(unversioned));
  if (scratch.arcCostValidationScanCount() != coldValidationBegin + 2)
    fail(__func__, "unversioned costs did not retain cold validation");
  unversionedCurrentCosts[0] = routeCostInfinity;
  expectFailure(__func__, scratch.search(unversioned),
                EndpointRouteSearchFailureKind::Invalid);

  auto orphanedRevision =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 128);
  orphanedRevision.currentArcCostRevision =
      movedCurrentRevisionOwner.revision();
  {
    EndpointRouteInputRevisionOwner temporaryOwner;
    orphanedRevision.lowerBoundArcCostRevision = temporaryOwner.revision();
  }
  const std::uint64_t orphanedValidationBegin =
      scratch.arcCostValidationScanCount();
  const std::uint64_t orphanedHeuristicBuildBegin =
      scratch.heuristicBuildCount();
  const std::uint64_t orphanedHeuristicHitBegin =
      scratch.heuristicCacheHitCount();
  (void)take(__func__, scratch.search(orphanedRevision));
  (void)take(__func__, scratch.search(orphanedRevision));
  if (scratch.arcCostValidationScanCount() != orphanedValidationBegin + 2)
    fail(__func__, "an ownerless revision did not retain cold validation");
  if (scratch.heuristicBuildCount() != orphanedHeuristicBuildBegin + 2 ||
      scratch.heuristicCacheHitCount() != orphanedHeuristicHitBegin)
    fail(__func__, "an ownerless revision reused a retained heuristic");
}

void certifiedInputRevisionSkipsValidation() {
  Fixture fixture;
  EndpointRouteSearchScratch scratch;
  EndpointRouteInputRevisionOwner lowerBoundRevisionOwner;
  EndpointRouteInputRevisionOwner currentRevisionOwner;
  requireSuccess(__func__, scratch.prepare(fixture.graph()));
  const std::array<PnrIndex, 1> sources{{0}};
  const std::array<PnrIndex, 1> sourceGroups{{Fixture::noReplicationGroup}};
  const std::array<PnrIndex, 1> targets{{4}};
  const std::array<PnrIndex, 1> targetRanks{{0}};

  requireSuccess(__func__, lowerBoundRevisionOwner.certify());
  requireSuccess(__func__, currentRevisionOwner.certify());
  auto certified =
      request(fixture, sources, sourceGroups, targets, targetRanks, 1, 128,
              &lowerBoundRevisionOwner, &currentRevisionOwner);
  (void)take(__func__, scratch.search(certified));
  if (scratch.arcCostValidationScanCount() != 0)
    fail(__func__, "certified cost revisions were scanned by the search");

  // An advance withdraws the certification: the scan resumes and rejects the
  // input the owner has not certified since.
  requireSuccess(__func__, currentRevisionOwner.advance());
  fixture.currentCosts[0] = routeCostInfinity;
  certified.currentArcCostRevision = currentRevisionOwner.revision();
  expectFailure(__func__, scratch.search(certified),
                EndpointRouteSearchFailureKind::Invalid);
  if (scratch.arcCostValidationScanCount() != 1)
    fail(__func__, "an advanced revision kept its certification");

  fixture.currentCosts[0] = fixture.lowerCosts[0];
  requireSuccess(__func__, currentRevisionOwner.certify());
  (void)take(__func__, scratch.search(certified));
  if (scratch.arcCostValidationScanCount() != 1)
    fail(__func__, "a re-certified revision was scanned again");
}

} // namespace

int main() {
  arbitraryTopologyAndCanonicalTieBreak();
  widthFilteringAndWorkLimit();
  requiredTraversalProductState();
  checkedCostAndAdmissibility();
  timingAwareArrivalAndBoundary();
  failureKindSpellings();
  exactHeuristicCacheInvalidation();
  exactHeuristicCachePreservesWideCosts();
  validatedInputRevisionInvalidation();
  certifiedInputRevisionSkipsValidation();
  return 0;
}
