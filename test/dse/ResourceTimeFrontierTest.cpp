#include "DSE/ResourceTimeFrontier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "resource-time frontier test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactRootReference reference(std::uint8_t fill) {
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return {"loom.test.resource_time", loom::SchemaVersion{1, 0},
          take(loom::ArtifactIdentity::fromBytes(bytes))};
}

dataflow::RootThreadLaunchRef root(std::uint64_t ordinal) {
  return {reference(2).artifact, dataflow::RootThreadLaunchId(ordinal)};
}

loom::ComponentViewDigest digest(std::uint8_t fill) {
  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize> bytes{};
  bytes.fill(fill);
  return take(loom::ComponentViewDigest::fromBytes(bytes));
}

loom::dse::ResourceTimeInvocationKey invocation() {
  return {reference(1), reference(2), reference(3), reference(4), reference(5),
          digest(6),    digest(7),    "main",       std::nullopt};
}

loom::dse::ResourceTimeSpeedupPoint
point(std::uint64_t resources, std::uint64_t execution,
      std::uint64_t transition = 0,
      std::optional<std::uint64_t> firstToken = std::nullopt) {
  return {{resources}, execution,
          firstToken,  std::nullopt,
          0,           transition,
          0,           loom::dse::ResourceTimeEstimateSupport::Calibrated};
}

std::vector<loom::dse::ResourceTimeRegionFeature>
fiveRegionFeatures(std::uint64_t expandedTransitionCost,
                   bool fifoConsumer = false) {
  const auto r1 = root(11);
  const auto r2 = root(12);
  const auto r3 = root(13);
  const auto r4 = root(14);
  const auto r5 = root(15);
  return {
      {r1, {}, {point(1, 10)}, 0, false, {}},
      {r2, {}, {point(1, 20, 0, 5)}, 0, false, {}},
      {r3, {}, {point(1, 15)}, 0, false, {}},
      {r4,
       {{r1, loom::pnr::ResourceTimeReadinessKind::Completion}},
       {point(1, 30), point(2, 15, expandedTransitionCost)},
       0,
       false,
       {}},
      {r5,
       fifoConsumer
           ? std::vector<
                 loom::dse::
                     ResourceTimeDependencyFeature>{{r2,
                                                     loom::pnr::
                                                         ResourceTimeReadinessKind::
                                                             FifoToken}}
           : std::vector<
                 loom::dse::
                     ResourceTimeDependencyFeature>{{r2,
                                                     loom::pnr::
                                                         ResourceTimeReadinessKind::
                                                             Completion},
                                                    {r4,
                                                     loom::pnr::
                                                         ResourceTimeReadinessKind::
                                                             Completion}},
       {point(1, 10)},
       0,
       false,
       {}}};
}

loom::dse::ResourceTimeFrontierPolicy policy() {
  loom::dse::ResourceTimeFrontierPolicy result;
  result.availableResourceUnits = {4};
  result.maximumStatesGenerated = 10000;
  result.maximumActionsGenerated = 40000;
  result.maximumStateCacheEntries = 10000;
  result.maximumRetainedBytes = 256ULL * 1024ULL * 1024ULL;
  result.beamWidth = 256;
  result.maximumFinalists = 32;
  return result;
}

const loom::dse::ResourceTimeScheduleHint &
hintUsingR4Point(const loom::dse::CompletedResourceTimeFrontier &completed,
                 std::uint64_t pointOrdinal) {
  const auto r4 = root(14);
  const loom::dse::ResourceTimeScheduleHint *best = nullptr;
  for (const auto &hint : completed.finalists) {
    const bool usesPoint = llvm::any_of(hint.actions, [&](const auto &action) {
      return action.admittedRegion == r4 &&
             action.speedupPointOrdinal == pointOrdinal;
    });
    if (usesPoint && (!best || hint.estimatedMakespanPicoseconds <
                                   best->estimatedMakespanPicoseconds))
      best = &hint;
  }
  if (!best)
    fail("frontier omitted an R4 resource allocation");
  return *best;
}

void fiveRegionCostAndReadinessAreEventDriven() {
  const std::array resourceClasses = {reference(20)};
  const auto lowOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), policy()));
  const auto *low =
      std::get_if<loom::dse::CompletedResourceTimeFrontier>(&lowOutcome);
  require(low, "low-cost frontier did not complete");
  const auto &lowExpanded = hintUsingR4Point(*low, 1);
  const auto &lowCompact = hintUsingR4Point(*low, 0);
  require(lowExpanded.estimatedMakespanPicoseconds <
              lowCompact.estimatedMakespanPicoseconds,
          "low transition cost did not favor expanded R4");
  require(low->concurrencyBounds &&
              low->concurrencyBounds->support ==
                  loom::dse::ResourceTimeEstimateSupport::Exact &&
              low->concurrencyBounds->minimumPeakConcurrentRegions == 1 &&
              low->concurrencyBounds->maximumPeakConcurrentRegions == 3 &&
              llvm::any_of(low->finalists,
                           [](const auto &hint) {
                             return hint.peakConcurrentRegions == 3;
                           }),
          "five-region witness lost its exact concurrency extrema");
  require(low->domainExhaustive,
          "unpruned five-region search was not marked exhaustive");

  const auto highOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(20), policy()));
  const auto *high =
      std::get_if<loom::dse::CompletedResourceTimeFrontier>(&highOutcome);
  require(high, "high-cost frontier did not complete");
  const auto &highExpanded = hintUsingR4Point(*high, 1);
  const auto &highCompact = hintUsingR4Point(*high, 0);
  require(highExpanded.estimatedMakespanPicoseconds >
              highCompact.estimatedMakespanPicoseconds,
          "high transition cost did not favor leaving R4 compact");

  const auto r5 = root(15);
  for (const auto &action : lowExpanded.actions)
    if (action.afterTimePicoseconds == 20)
      require(!llvm::is_contained(action.newlyReadyRegions, r5),
              "R5 became ready after R2 but before R4 completed");

  const auto fifoOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2, true), policy()));
  const auto *fifo =
      std::get_if<loom::dse::CompletedResourceTimeFrontier>(&fifoOutcome);
  require(fifo, "FIFO frontier did not complete");
  bool sawEarlyR5 = false;
  for (const auto &hint : fifo->finalists)
    for (const auto &action : hint.actions)
      if (action.admittedRegion == r5 && action.beforeTimePicoseconds == 5)
        sawEarlyR5 = true;
  require(sawEarlyR5,
          "FIFO token event did not permit R5 before R2 completion");

  require(!loom::dse::validateResourceTimeFrontierAccounting(low->accounting),
          "low-cost work accounting is not closed");
  require(low->accounting.stateMemoHits != 0,
          "event frontier did not deduplicate equivalent admission orders");
  require(low->accounting.stateMemoParetoInsertions != 0,
          "event frontier did not retain non-dominated paths to one future "
          "state");
  require(low->accounting.incrementalLowerBoundUpdates != 0 &&
              low->accounting.estimates.consumed <
                  low->accounting.states.consumed,
          "event frontier recomputed every lower bound instead of applying "
          "incremental updates");
  require(low->accounting.estimates.consumed == 1 &&
              low->accounting.incrementalLowerBoundUpdates ==
                  low->accounting.actions.consumed &&
              low->accounting.terminalHintsGenerated >=
                  low->accounting.terminalHintsRetained &&
              low->accounting.terminalHintsGenerated ==
                  low->accounting.terminalHintsRetained +
                      low->accounting.terminalHintsPruned,
          "resource-time funnel did not bound lower-bound and terminal-hint "
          "work at generation time");
}

void budgetsAndExactRejectionsRemainTyped() {
  const std::array resourceClasses = {reference(20)};
  auto singleFinalistPolicy = policy();
  singleFinalistPolicy.maximumFinalists = 1;
  const auto singleFinalistOutcome =
      take(loom::dse::exploreResourceTimeFrontier(invocation(), resourceClasses,
                                                  fiveRegionFeatures(2),
                                                  singleFinalistPolicy));
  const auto *singleFinalist =
      std::get_if<loom::dse::CompletedResourceTimeFrontier>(
          &singleFinalistOutcome);
  require(singleFinalist && singleFinalist->finalists.size() == 1 &&
              singleFinalist->concurrencyBounds &&
              singleFinalist->concurrencyBounds->support ==
                  loom::dse::ResourceTimeEstimateSupport::Exact &&
              singleFinalist->concurrencyBounds->minimumPeakConcurrentRegions ==
                  1 &&
              singleFinalist->concurrencyBounds->maximumPeakConcurrentRegions ==
                  3 &&
              !loom::dse::validateResourceTimeFrontierAccounting(
                  singleFinalist->accounting),
          "single-finalist terminal inventory exceeded its hard bound");
  auto bounded = policy();
  bounded.maximumStatesGenerated = 1;
  const auto boundedOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), bounded));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteResourceTimeFrontier>(&boundedOutcome);
  require(
      incomplete &&
          incomplete->reason ==
              loom::dse::ResourceTimeFrontierIncompleteReason::BudgetExhausted,
      "state budget was not reported as typed incomplete");
  require(!loom::dse::validateResourceTimeFrontierAccounting(
              incomplete->accounting),
          "incomplete work accounting is not closed");
  auto inconsistentMemoAccounting = incomplete->accounting;
  ++inconsistentMemoAccounting.stateMemoParetoInsertions;
  llvm::Error inconsistentMemoError =
      loom::dse::validateResourceTimeFrontierAccounting(
          inconsistentMemoAccounting);
  require(static_cast<bool>(inconsistentMemoError),
          "state memo ledger accepted an unaccounted envelope update");
  llvm::consumeError(std::move(inconsistentMemoError));

  auto impossible = fiveRegionFeatures(2);
  impossible.front().speedupCurve = {point(5, 10)};
  impossible.front().allocationDomainExhaustive = true;
  const auto impossibleOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, impossible, policy()));
  const auto *proven =
      std::get_if<loom::dse::ProvenInfeasibleResourceTimeFrontier>(
          &impossibleOutcome);
  require(
      proven &&
          proven->reason ==
              loom::dse::ResourceTimeFrontierInfeasibleReason::ResourceCapacity,
      "fixed resource-capacity failure was not proven infeasible");

  auto unknownCapacity = fiveRegionFeatures(2);
  unknownCapacity.front().speedupCurve = {point(5, 10)};
  const auto unknownOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, unknownCapacity, policy()));
  const auto *unknown =
      std::get_if<loom::dse::IncompleteResourceTimeFrontier>(&unknownOutcome);
  require(unknown && unknown->reason ==
                         loom::dse::ResourceTimeFrontierIncompleteReason::
                             ProofNotEstablished,
          "incomplete allocation domain was misclassified as infeasible");

  const auto alwaysStop = [](const void *) { return true; };
  const loom::ExecutionControlView stopped(nullptr, alwaysStop);
  const auto stoppedOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), policy(), stopped));
  const auto *cancelled =
      std::get_if<loom::dse::IncompleteResourceTimeFrontier>(&stoppedOutcome);
  require(cancelled && cancelled->reason ==
                           loom::dse::ResourceTimeFrontierIncompleteReason::
                               CancelledOrTimeout,
          "deadline stop was not reported as typed incomplete");

  auto pruned = policy();
  pruned.beamWidth = 1;
  const auto prunedOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), pruned));
  const auto *boundedResult =
      std::get_if<loom::dse::CompletedResourceTimeFrontier>(&prunedOutcome);
  require(boundedResult && !boundedResult->domainExhaustive &&
              boundedResult->accounting.statesPrunedByBeam != 0,
          "beam truncation was not exposed as bounded completion");
}

void dependencyCyclesRemainTyped() {
  const std::array resourceClasses = {reference(20)};
  const auto r1 = root(11);
  const auto r2 = root(12);
  std::vector<loom::dse::ResourceTimeRegionFeature> completionCycle{
      {r1,
       {{r2, loom::pnr::ResourceTimeReadinessKind::Completion}},
       {point(1, 10)},
       0,
       false,
       {}},
      {r2,
       {{r1, loom::pnr::ResourceTimeReadinessKind::Completion}},
       {point(1, 10)},
       0,
       false,
       {}}};
  const auto completionOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, completionCycle, policy()));
  const auto *proven =
      std::get_if<loom::dse::ProvenInfeasibleResourceTimeFrontier>(
          &completionOutcome);
  require(proven && proven->reason ==
                        loom::dse::ResourceTimeFrontierInfeasibleReason::
                            CompletionDependencyCycle,
          "completion-only dependency deadlock was not proven infeasible");

  completionCycle.front().dependencies.front().readiness =
      loom::pnr::ResourceTimeReadinessKind::FifoToken;
  completionCycle[1].speedupCurve.front().firstTokenLatencyPicoseconds = 2;
  const auto fifoOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, completionCycle, policy()));
  const auto *unsupported =
      std::get_if<loom::dse::IncompleteResourceTimeFrontier>(&fifoOutcome);
  require(unsupported &&
              unsupported->reason ==
                  loom::dse::ResourceTimeFrontierIncompleteReason::Unsupported,
          "FIFO recurrence was misclassified as exact infeasibility");
}

void replayIsDeterministic() {
  const std::array resourceClasses = {reference(20)};
  const auto firstOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), policy()));
  const auto secondOutcome = take(loom::dse::exploreResourceTimeFrontier(
      invocation(), resourceClasses, fiveRegionFeatures(2), policy()));
  const auto &first =
      std::get<loom::dse::CompletedResourceTimeFrontier>(firstOutcome);
  const auto &second =
      std::get<loom::dse::CompletedResourceTimeFrontier>(secondOutcome);
  require(first.finalists.size() == second.finalists.size(),
          "deterministic replay changed finalist count");
  for (std::size_t index = 0; index != first.finalists.size(); ++index)
    require(first.finalists[index].actions == second.finalists[index].actions &&
                first.finalists[index].estimatedMakespanPicoseconds ==
                    second.finalists[index].estimatedMakespanPicoseconds,
            "deterministic replay changed schedule selection");
  require(
      first.accounting.actions.planned == second.accounting.actions.planned &&
          first.accounting.states.planned == second.accounting.states.planned &&
          first.accounting.stateMemoHits == second.accounting.stateMemoHits,
      "deterministic replay changed formal work");
}

void mappingFunnelAdmitsOnlyBoundedFinalists() {
  auto bounded = policy();
  bounded.maximumMappingFinalists = 3;
  std::vector<loom::dse::ResourceTimeMappingCandidateInput> candidates;
  for (std::uint8_t ordinal = 0; ordinal != 6; ++ordinal) {
    auto key = invocation();
    key.sourceLineage = reference(static_cast<std::uint8_t>(30 + ordinal));
    candidates.push_back({digest(static_cast<std::uint8_t>(40 + ordinal)),
                          ordinal,
                          static_cast<std::uint64_t>((ordinal % 3) + 1),
                          static_cast<std::uint64_t>(ordinal + 1),
                          static_cast<std::uint64_t>((ordinal + 1) * 10),
                          static_cast<std::uint64_t>(ordinal + 1),
                          key,
                          {reference(20)},
                          fiveRegionFeatures(ordinal + 1)});
  }
  const auto selected =
      take(loom::dse::selectResourceTimeMappingFinalists(candidates, bounded));
  require(selected.accounting.generatedCandidates == candidates.size() &&
              selected.accounting.screenedCandidates == candidates.size() &&
              selected.accounting.detailedFrontierCandidates == 3 &&
              selected.accounting.successiveHalvingDeferredCandidates == 3 &&
              selected.accounting.mappingFinalists == 3 &&
              selected.accounting.mappingEligibleScheduleHints >
                  selected.accounting.mappingFinalists &&
              selected.accounting.mappingCallsDeferredByModel ==
                  selected.accounting.mappingEligibleScheduleHints - 3 &&
              selected.finalists.size() == 3 && selected.truncated,
          "resource-time funnel did not bound real Mapping finalists");
  require(
      selected.accounting.frontierAccounting.states.consumed != 0 &&
          selected.accounting.screeningCalibration.comparedCandidates != 0 &&
          selected.accounting.screeningCalibration.lowerBoundViolations == 0 &&
          selected.accounting.screeningCalibration.feasibleIntersection <=
              selected.accounting.screeningCalibration
                  .exactFeasibleCandidates &&
          selected.accounting.screeningCalibration.errorSamples ==
              selected.accounting.screeningCalibration
                  .exactFeasibleCandidates &&
          selected.accounting.screeningCalibration.feasibilityRecallPermille ==
              1000 &&
          selected.accounting.screeningCalibration
                  .feasibilityPrecisionPermille == 1000 &&
          selected.accounting.screeningCalibration.bestRecallPermille == 1000 &&
              selected.accounting.frontierAccounting.stateMemoMisses -
                      selected.accounting.frontierAccounting
                          .stateMemoMissCapacityRejections +
                      selected.accounting.frontierAccounting
                          .stateMemoParetoInsertions ==
                  selected.accounting.frontierAccounting.states.consumed &&
              !loom::dse::validateResourceTimeMappingFunnelAccounting(
                  selected.accounting),
          "resource-time funnel aggregate work ledger is not closed");

  auto noHintPolicy = policy();
  noHintPolicy.maximumStatesGenerated = 1;
  const auto noHint = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidates.front()}, noHintPolicy));
  require(noHint.accounting.incompleteCandidates == 1 &&
              noHint.incompleteReason ==
                  loom::dse::ResourceTimeFrontierIncompleteReason::
                      BudgetExhausted &&
              noHint.finalists.empty() &&
              noHint.accounting.mappingCallsDeferredByModel == 0 &&
              noHint.accounting.mappingCallsWithheldByIncomplete == 1,
          "incomplete resource-time candidate without a hint reached Mapping");
  auto noHintDuplicate = candidates.front();
  noHintDuplicate.candidateIdentity = digest(62);
  noHintDuplicate.inputPreferenceRank = 62;
  const auto noHintAgain = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidates.front(), noHintDuplicate}, noHintPolicy));
  require(noHintAgain.accounting.exactInvocationMemoMisses == 2 &&
              noHintAgain.accounting.exactInvocationMemoHits == 0,
          "incomplete resource-time frontier was memoized");

  auto impossible = candidates.front();
  impossible.candidateIdentity = digest(60);
  impossible.inputPreferenceRank = candidates.size();
  impossible.invocation.sourceLineage = reference(61);
  impossible.regions.front().speedupCurve = {point(5, 10)};
  impossible.regions.front().allocationDomainExhaustive = true;
  candidates.push_back(std::move(impossible));
  const auto gated =
      take(loom::dse::selectResourceTimeMappingFinalists(candidates, bounded));
  require(gated.accounting.soundGateRejectedCandidates == 1 &&
              gated.accounting.mappingFinalists == 3 &&
              !loom::dse::validateResourceTimeMappingFunnelAccounting(
                  gated.accounting),
          "resource-time sound gate did not remain separate from model "
          "deferral");
  const auto alwaysStop = [](const void *) { return true; };
  const auto cancelled = take(loom::dse::selectResourceTimeMappingFinalists(
      candidates, bounded, loom::ExecutionControlView(nullptr, alwaysStop)));
  require(cancelled.incompleteReason ==
                  loom::dse::ResourceTimeFrontierIncompleteReason::
                      CancelledOrTimeout &&
              cancelled.finalists.empty() && cancelled.evaluations.empty(),
          "resource-time funnel dispatched work after cancellation");

  auto slow = candidates.front();
  slow.candidateIdentity = digest(70);
  slow.inputPreferenceRank = 0;
  slow.invocation.sourceLineage = reference(70);
  const auto slowRoot = root(71);
  slow.regions = {{slowRoot,
                   {},
                   {{{1},
                     100,
                     std::nullopt,
                     std::nullopt,
                     0,
                     0,
                     0,
                     loom::dse::ResourceTimeEstimateSupport::Exact}},
                   0,
                   false,
                   {}}};
  auto fast = slow;
  fast.candidateIdentity = digest(71);
  fast.inputPreferenceRank = 1;
  fast.invocation.sourceLineage = reference(71);
  fast.regions.front().speedupCurve.front().executionTimePicoseconds = 1;
  const auto ranked = take(
      loom::dse::selectResourceTimeMappingFinalists({slow, fast}, bounded));
  require(ranked.finalists.size() == 2 &&
              ranked.finalists.front().candidateIdentity ==
                  fast.candidateIdentity,
          "analytic resource-time order was overwritten by input rank");
  const auto reordered = take(
      loom::dse::selectResourceTimeMappingFinalists({fast, slow}, bounded));
  require(reordered.finalists == ranked.finalists &&
              reordered.accounting.mappingFinalists ==
                  ranked.accounting.mappingFinalists,
          "resource-time provider order changed the promoted inventory");

  auto duplicate = candidates.front();
  duplicate.candidateIdentity = digest(90);
  duplicate.inputPreferenceRank = 90;
  const auto memoized = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidates.front(), duplicate}, bounded));
  const auto single = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidates.front()}, bounded));
  std::uint64_t minimumMappedConcurrency =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t maximumMappedConcurrency = 0;
  std::vector<loom::ComponentViewDigest> mappedScheduleDigests;
  for (const auto &finalist : single.finalists) {
    require(finalist.candidateIdentity ==
                    candidates.front().candidateIdentity &&
                !llvm::is_contained(mappedScheduleDigests,
                                    finalist.scheduleHintDigest),
            "one software candidate lost distinct schedule provenance");
    mappedScheduleDigests.push_back(finalist.scheduleHintDigest);
    const loom::dse::ResourceTimeScheduleHint *matched = nullptr;
    for (const auto &hint : single.evaluations.front().retainedHints) {
      const auto digest =
          take(loom::dse::deriveResourceTimeScheduleHintDigest(hint));
      if (digest == finalist.scheduleHintDigest)
        matched = &hint;
    }
    require(matched, "Mapping finalist lost its retained schedule hint");
    minimumMappedConcurrency =
        std::min(minimumMappedConcurrency, matched->peakConcurrentRegions);
    maximumMappedConcurrency =
        std::max(maximumMappedConcurrency, matched->peakConcurrentRegions);
  }
  require(single.finalists.size() == 3 && minimumMappedConcurrency == 1 &&
              maximumMappedConcurrency == 3,
          "one software candidate did not retain bounded temporal, spatial, "
          "and objective Mapping schedules");
  require(memoized.accounting.exactInvocationMemoMisses == 1 &&
              memoized.accounting.exactInvocationMemoHits == 1 &&
              memoized.accounting.frontierAccounting.states.consumed ==
                  single.accounting.frontierAccounting.states.consumed &&
              memoized.accounting.frontierAccounting.actions.consumed ==
                  single.accounting.frontierAccounting.actions.consumed &&
              !loom::dse::validateResourceTimeMappingFunnelAccounting(
                  memoized.accounting),
          "identical resource-time frontier input was not memoized exactly");

  auto invalidated = duplicate;
  invalidated.candidateIdentity = digest(91);
  invalidated.inputPreferenceRank = 91;
  invalidated.invocation.modelSnapshotDigest = digest(92);
  const auto invalidatedMemo =
      take(loom::dse::selectResourceTimeMappingFinalists(
          {candidates.front(), invalidated}, bounded));
  require(invalidatedMemo.accounting.exactInvocationMemoMisses == 2 &&
              invalidatedMemo.accounting.exactInvocationMemoHits == 0,
          "resource-time model snapshot change did not invalidate exact memo");

  auto noCache = bounded;
  noCache.maximumInvocationMemoBytes = 1;
  const auto noCacheResult = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidates.front(), duplicate}, noCache));
  require(noCacheResult.accounting.exactInvocationMemoMisses == 2 &&
              noCacheResult.accounting.exactInvocationMemoHits == 0 &&
              noCacheResult.accounting.exactInvocationMemoCapacityBypasses ==
                  2 &&
              noCacheResult.accounting.exactInvocationMemoRetainedBytes == 0,
          "resource-time exact memo exceeded its byte budget");

  auto temporalEpochGate = candidates.front();
  temporalEpochGate.candidateIdentity = digest(93);
  temporalEpochGate.inputPreferenceRank = 93;
  temporalEpochGate.regions.front().logicalEpochCount = 2;
  auto temporalEndpointPolicy = bounded;
  temporalEndpointPolicy.spectrumEndpoint =
      loom::dse::PreMappingSpectrumEndpoint::MaxTemporal;
  temporalEndpointPolicy.maximumMappingFinalists = 1;
  const auto temporalGated = take(loom::dse::selectResourceTimeMappingFinalists(
      {temporalEpochGate}, temporalEndpointPolicy));
  require(
      temporalGated.accounting.detailedFrontierCandidates != 0 &&
          temporalGated.accounting.incompleteCandidates == 0 &&
          temporalGated.accounting.successiveHalvingDeferredCandidates == 0 &&
          !temporalGated.finalists.empty() &&
          temporalGated.incompleteReason !=
              loom::dse::ResourceTimeFrontierIncompleteReason::Unsupported &&
          !loom::dse::validateResourceTimeMappingFunnelAccounting(
              temporalGated.accounting),
      "partitioned temporal epoch was incorrectly rejected before Mapping");
}

void outOfDomainScreeningRemainsMeasuredButInadmissible() {
  auto regions = fiveRegionFeatures(2);
  for (auto &region : regions)
    for (auto &candidate : region.speedupCurve)
      candidate.support = loom::dse::ResourceTimeEstimateSupport::OutOfDomain;
  loom::dse::ResourceTimeMappingCandidateInput candidate{
      digest(102),       0, 5, 5, 50, 2, invocation(), {reference(20)},
      std::move(regions)};
  auto bounded = policy();
  bounded.maximumMappingFinalists = 1;
  const auto selected =
      take(loom::dse::selectResourceTimeMappingFinalists({candidate}, bounded));
  const auto &calibration = selected.accounting.screeningCalibration;
  require(calibration.comparedCandidates == 1 &&
              calibration.exactFeasibleCandidates == 1 &&
              calibration.outOfDomainCandidates == 1 &&
              calibration.outOfDomainConfidenceCandidates == 1 &&
              calibration.screeningAdmissibleCandidates == 0 &&
              calibration.feasibleIntersection == 0 &&
              calibration.feasibilityRecallPermille == 0 &&
              calibration.outOfDomainPermille == 1000 &&
              !loom::dse::validateResourceTimeMappingFunnelAccounting(
                  selected.accounting),
          "out-of-domain screening was presented as admissible evidence");
}

void exactMemoSupportsWarmAndConcurrentReuse() {
  auto bounded = policy();
  bounded.maximumMappingFinalists = 1;
  loom::dse::ResourceTimeMappingCandidateInput candidate{
      digest(100),          0, 5, 5, 50, 2, invocation(), {reference(20)},
      fiveRegionFeatures(2)};
  loom::dse::ResourceTimeFrontierSession warmSession(
      bounded.maximumInvocationMemoEntries, bounded.maximumInvocationMemoBytes);
  const auto cold = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidate}, bounded, {}, &warmSession));
  const auto warm = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidate}, bounded, {}, &warmSession));
  require(cold.accounting.exactInvocationMemoMisses == 1 &&
              cold.accounting.exactInvocationMemoHits == 0 &&
              cold.accounting.frontierAccounting.states.consumed != 0 &&
              warm.accounting.exactInvocationMemoHits == 1 &&
              warm.accounting.exactInvocationMemoMisses == 0 &&
              warm.accounting.frontierAccounting.states.consumed == 0 &&
              warm.finalists == cold.finalists,
          "warm exact memo changed selection or repeated frontier work");
  const auto warmStatistics = warmSession.statistics();
  require(warmStatistics.requests == 2 && warmStatistics.cacheMisses == 1 &&
              warmStatistics.cacheHits == 1 && warmStatistics.entryCount == 1 &&
              warmStatistics.retainedBytes != 0,
          "warm exact memo session accounting is not closed");
  auto endpointPolicy = bounded;
  endpointPolicy.spectrumEndpoint =
      loom::dse::PreMappingSpectrumEndpoint::MaxSpatial;
  const auto focused = take(loom::dse::selectResourceTimeMappingFinalists(
      {candidate}, endpointPolicy, {}, &warmSession));
  require(focused.accounting.exactInvocationMemoHits == 1 &&
              focused.accounting.exactInvocationMemoMisses == 0 &&
              focused.accounting.frontierAccounting.states.consumed == 0,
          "endpoint ranking changed the exact analytic frontier cache key");

  std::vector<loom::dse::ResourceTimeRegionFeature> concurrentRegions;
  for (std::uint64_t ordinal = 0; ordinal != 7; ++ordinal)
    concurrentRegions.push_back(
        {root(100 + ordinal),
         {},
         {point(1, 10 + ordinal), point(2, 6 + ordinal)},
         0,
         false,
         {}});
  auto concurrentPolicy = policy();
  concurrentPolicy.maximumStatesGenerated = 50000;
  concurrentPolicy.maximumActionsGenerated = 200000;
  concurrentPolicy.maximumStateCacheEntries = 50000;
  concurrentPolicy.beamWidth = 1024;
  concurrentPolicy.maximumMappingFinalists = 1;
  loom::dse::ResourceTimeMappingCandidateInput concurrentCandidate{
      digest(101),
      0,
      7,
      7,
      70,
      2,
      invocation(),
      {reference(20)},
      std::move(concurrentRegions)};
  loom::dse::ResourceTimeFrontierSession concurrentSession(
      concurrentPolicy.maximumInvocationMemoEntries,
      concurrentPolicy.maximumInvocationMemoBytes);
  constexpr std::size_t workerCount = 4;
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> start{false};
  std::mutex errorMutex;
  std::string error;
  std::vector<std::optional<loom::dse::ResourceTimeMappingFunnel>> outcomes(
      workerCount);
  std::vector<std::thread> workers;
  workers.reserve(workerCount);
  for (std::size_t worker = 0; worker != workerCount; ++worker)
    workers.emplace_back([&, worker] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      auto outcome = loom::dse::selectResourceTimeMappingFinalists(
          {concurrentCandidate}, concurrentPolicy, {}, &concurrentSession);
      if (!outcome) {
        std::lock_guard<std::mutex> lock(errorMutex);
        error = llvm::toString(outcome.takeError());
        return;
      }
      outcomes[worker] = std::move(*outcome);
    });
  while (ready.load(std::memory_order_acquire) != workerCount)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (std::thread &worker : workers)
    worker.join();
  require(error.empty(), error);
  const auto concurrentStatistics = concurrentSession.statistics();
  require(concurrentStatistics.requests == workerCount &&
              concurrentStatistics.cacheMisses == 1 &&
              concurrentStatistics.singleFlightWaits != 0 &&
              concurrentStatistics.cacheHits +
                      concurrentStatistics.coalescedUncachedResults ==
                  workerCount - 1,
          "concurrent equal misses were not single-flight");
  std::uint64_t consumedStates = 0;
  std::uint64_t workOwners = 0;
  for (const auto &outcome : outcomes) {
    require(outcome.has_value(), "concurrent frontier lost an outcome");
    const std::uint64_t states =
        outcome->accounting.frontierAccounting.states.consumed;
    consumedStates += states;
    workOwners += states != 0;
  }
  require(consumedStates != 0 && workOwners == 1,
          "concurrent exact reuse repeated formal frontier work");
}

} // namespace

int main() {
  fiveRegionCostAndReadinessAreEventDriven();
  budgetsAndExactRejectionsRemainTyped();
  dependencyCyclesRemainTyped();
  replayIsDeterministic();
  mappingFunnelAdmitsOnlyBoundedFinalists();
  outOfDomainScreeningRemainsMeasuredButInadmissible();
  exactMemoSupportsWarmAndConcurrentReuse();
  return 0;
}
