#include "DSE/ResourceTimeFrontier.h"

#include "ResourceTimeFrontierInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

namespace loom::dse {
using namespace detail;
namespace {

using MonotonicClock = std::chrono::steady_clock;

constexpr llvm::StringLiteral resourceTimeExactMemoDescriptor{
    "loom.dse.resource_time_exact_frontier_memo.1"};
constexpr llvm::StringLiteral resourceTimeScheduleHintDescriptor{
    "loom.dse.resource_time_schedule_hint.1"};

llvm::Error invalid(const llvm::Twine &message) {
  return invalidResourceTimeFrontier(message);
}

std::uint64_t permille(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0)
    return 0;
  return static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(numerator) * 1000) / denominator);
}

void appendDataflowRoots(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  appendU64(bytes, roots.size());
  for (const auto root : roots)
    appendDataflowRoot(bytes, root);
}

void appendScheduleHint(std::vector<std::uint8_t> &bytes,
                        const ResourceTimeScheduleHint &hint) {
  appendU64(bytes, hint.actions.size());
  for (const ResourceTimeActionDelta &action : hint.actions) {
    appendU64(bytes, static_cast<std::uint64_t>(action.kind));
    bytes.push_back(action.admittedRegion ? 1 : 0);
    if (action.admittedRegion)
      appendDataflowRoot(bytes, *action.admittedRegion);
    appendOptionalU64(bytes, action.speedupPointOrdinal);
    appendU64(bytes, action.beforeTimePicoseconds);
    appendU64(bytes, action.afterTimePicoseconds);
    appendDataflowRoots(bytes, action.completedRegions);
    appendDataflowRoots(bytes, action.tokenReadyProducers);
    appendDataflowRoots(bytes, action.newlyReadyRegions);
  }
  appendU64(bytes, hint.states.size());
  for (const ResourceTimeHintState &state : hint.states) {
    appendU64(bytes, state.timePicoseconds);
    appendU64(bytes, state.active.size());
    for (const ResourceTimeHintAllocation &allocation : state.active) {
      appendDataflowRoot(bytes, allocation.region);
      appendU64(bytes, allocation.speedupPointOrdinal);
      appendU64(bytes, allocation.resourceUnits.size());
      for (std::uint64_t units : allocation.resourceUnits)
        appendU64(bytes, units);
      appendU64(bytes, allocation.completionTimePicoseconds);
    }
    appendDataflowRoots(bytes, state.ready);
    appendDataflowRoots(bytes, state.completed);
    appendU64(bytes, state.optimisticMakespanLowerBoundPicoseconds);
  }
  appendU64(bytes, hint.estimatedMakespanPicoseconds);
  appendU64(bytes, hint.optimisticMakespanLowerBoundPicoseconds);
  appendU64(bytes, hint.peakConcurrentRegions);
  appendU64(bytes, hint.totalAllocatedResourceTime);
  appendU64(bytes, static_cast<std::uint64_t>(hint.support));
}

std::string
exactFrontierMemoKey(const ResourceTimeInvocationKey &invocation,
                     llvm::ArrayRef<ArtifactRootReference> resourceClasses,
                     llvm::ArrayRef<ResourceTimeRegionFeature> regions,
                     const ResourceTimeFrontierPolicy &policy) {
  std::vector<std::uint8_t> bytes;
  appendString(bytes, resourceTimeExactMemoDescriptor);
  appendResourceTimeInvocationKey(bytes, invocation);
  appendResourceTimeFeatures(bytes, resourceClasses, regions);
  appendResourceTimePolicy(bytes, policy);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

struct ResourceTimeCandidateScreening final {
  std::uint64_t lowerBoundPicoseconds = 0;
  std::uint64_t featureScore = 0;
  ResourceTimeEstimateSupport support =
      ResourceTimeEstimateSupport::Unsupported;
  bool exactCapacityFailure = false;
};

llvm::Expected<ResourceTimeCandidateScreening>
screenCandidate(const ResourceTimeMappingCandidateInput &candidate,
                const ResourceTimeFrontierPolicy &policy) {
  if (candidate.resourceClasses.empty() || candidate.regions.empty() ||
      candidate.resourceClasses.size() != policy.availableResourceUnits.size())
    return invalid("resource-time screening inputs are not aligned");
  std::uint64_t totalCapacity = 0;
  for (std::uint64_t units : policy.availableResourceUnits) {
    const auto sum = llvm::checkedAddUnsigned(totalCapacity, units);
    if (!sum)
      return invalid("resource-time screening capacity overflows");
    totalCapacity = *sum;
  }
  if (totalCapacity == 0)
    return invalid("resource-time screening has no capacity");

  ResourceTimeCandidateScreening result;
  result.support = ResourceTimeEstimateSupport::Exact;
  std::uint64_t totalResourceWork = 0;
  std::uint64_t featureScore = 0;
  for (const ResourceTimeRegionFeature &region : candidate.regions) {
    if (region.speedupCurve.empty())
      return invalid("resource-time screening region has no speedup curve");
    bool hasFittingPoint = false;
    std::uint64_t minimumDuration = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t minimumWork = std::numeric_limits<std::uint64_t>::max();
    ResourceTimeEstimateSupport minimumDurationSupport =
        ResourceTimeEstimateSupport::Unsupported;
    ResourceTimeEstimateSupport minimumWorkSupport =
        ResourceTimeEstimateSupport::Unsupported;
    for (const ResourceTimeSpeedupPoint &point : region.speedupCurve) {
      if (point.resourceUnits.size() != candidate.resourceClasses.size() ||
          allocationMagnitude(point.resourceUnits) == 0)
        return invalid("resource-time screening point is malformed");
      std::vector<std::uint64_t> unused(point.resourceUnits.size(), 0);
      hasFittingPoint |=
          fits(unused, point.resourceUnits, policy.availableResourceUnits);
      const std::uint64_t duration = pointDuration(point);
      const auto work = llvm::checkedMulUnsigned(
          duration, allocationMagnitude(point.resourceUnits));
      if (!work)
        return invalid("resource-time screening work overflows");
      if (duration < minimumDuration ||
          (duration == minimumDuration &&
           estimateSupportRank(point.support) <
               estimateSupportRank(minimumDurationSupport))) {
        minimumDuration = duration;
        minimumDurationSupport = point.support;
      }
      if (*work < minimumWork ||
          (*work == minimumWork &&
           estimateSupportRank(point.support) <
               estimateSupportRank(minimumWorkSupport))) {
        minimumWork = *work;
        minimumWorkSupport = point.support;
      }
    }
    if (!hasFittingPoint && region.allocationDomainExhaustive)
      result.exactCapacityFailure = true;
    if (!hasFittingPoint && !region.allocationDomainExhaustive)
      minimumDurationSupport = ResourceTimeEstimateSupport::Unsupported;
    result.support =
        combineSupport(result.support, combineSupport(minimumDurationSupport,
                                                      minimumWorkSupport));
    result.lowerBoundPicoseconds =
        std::max(result.lowerBoundPicoseconds, minimumDuration);
    const auto total = llvm::checkedAddUnsigned(totalResourceWork, minimumWork);
    if (!total)
      return invalid("resource-time screening aggregate work overflows");
    totalResourceWork = *total;
    const auto feature = llvm::checkedAddUnsigned(
        region.analyticFeatures.launchSynchronizationCost,
        region.analyticFeatures.topologyCongestionProxy);
    if (!feature)
      return invalid("resource-time screening feature score overflows");
    const auto featureWithParallelism = llvm::checkedAddUnsigned(
        *feature, region.analyticFeatures.parallelismLowerBound);
    if (!featureWithParallelism)
      return invalid("resource-time screening feature score overflows");
    const auto aggregateFeature =
        llvm::checkedAddUnsigned(featureScore, *featureWithParallelism);
    if (!aggregateFeature)
      return invalid("resource-time screening feature score overflows");
    featureScore = *aggregateFeature;
  }
  const std::uint64_t resourceBound = totalResourceWork / totalCapacity +
                                      (totalResourceWork % totalCapacity != 0);
  result.lowerBoundPicoseconds =
      std::max(result.lowerBoundPicoseconds, resourceBound);
  result.featureScore = featureScore;
  return result;
}

} // namespace

llvm::Expected<ComponentViewDigest>
deriveResourceTimeScheduleHintDigest(const ResourceTimeScheduleHint &hint) {
  std::vector<std::uint8_t> bytes;
  appendScheduleHint(bytes, hint);
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(
           resourceTimeScheduleHintDescriptor.data()),
       resourceTimeScheduleHintDescriptor.size()},
      bytes);
}

int incompleteReasonPriority(ResourceTimeFrontierIncompleteReason reason) {
  switch (reason) {
  case ResourceTimeFrontierIncompleteReason::CancelledOrTimeout:
    return 4;
  case ResourceTimeFrontierIncompleteReason::BudgetExhausted:
    return 3;
  case ResourceTimeFrontierIncompleteReason::Unsupported:
    return 2;
  case ResourceTimeFrontierIncompleteReason::ProofNotEstablished:
    return 1;
  }
  llvm_unreachable("unknown resource-time incomplete reason");
}

llvm::Error validateResourceTimeFrontierAccounting(
    const ResourceTimeFrontierAccounting &accounting) {
  for (const auto *counter :
       {&accounting.sourceProjections, &accounting.actions, &accounting.states,
        &accounting.estimates, &accounting.finalists}) {
    if (counter->planned != counter->reserved)
      return invalid("planned and reserved work disagree");
    if (counter->consumed > counter->reserved ||
        counter->rejected > counter->reserved - counter->consumed ||
        counter->cancelled >
            counter->reserved - counter->consumed - counter->rejected ||
        counter->consumed + counter->rejected + counter->cancelled !=
            counter->reserved)
      return invalid("work ledger is not additively closed");
    if (counter->planned > counter->limit)
      return invalid("work ledger exceeds its limit");
  }
  if (accounting.stateMemoMissCapacityRejections > accounting.stateMemoMisses)
    return invalid("state memo miss-capacity rejections exceed misses");
  auto admitted = llvm::checkedAddUnsigned(
      accounting.stateMemoMisses - accounting.stateMemoMissCapacityRejections,
      accounting.stateMemoParetoInsertions);
  if (!admitted || *admitted != accounting.states.consumed)
    return invalid("state memo admissions differ from consumed states");
  auto memoHits = llvm::checkedAddUnsigned(accounting.stateMemoParetoInsertions,
                                           accounting.stateMemoDominatedStates);
  if (memoHits)
    memoHits = llvm::checkedAddUnsigned(
        *memoHits, accounting.stateMemoHitCapacityRejections);
  if (!memoHits || accounting.stateMemoHits != *memoHits)
    return invalid("state memo hit accounting is not closed");
  if (accounting.estimates.consumed > accounting.sourceProjections.consumed)
    return invalid("resource-time full lower-bound accounting is not closed");
  if (accounting.incrementalLowerBoundUpdates != accounting.actions.consumed)
    return invalid("resource-time incremental lower-bound accounting is not "
                   "closed");
  if (accounting.terminalHintsRetained > accounting.terminalHintsGenerated ||
      accounting.terminalHintsPruned !=
          accounting.terminalHintsGenerated - accounting.terminalHintsRetained)
    return invalid("resource-time terminal-hint accounting is not closed");
  return llvm::Error::success();
}

llvm::Error validateResourceTimeMappingFunnelAccounting(
    const ResourceTimeMappingFunnelAccounting &accounting) {
  if (llvm::Error error =
          validateResourceTimeFrontierAccounting(accounting.frontierAccounting))
    return error;
  if (accounting.soundGateRejectedCandidates > accounting.generatedCandidates ||
      accounting.estimatedCandidates > accounting.generatedCandidates ||
      accounting.incompleteCandidates > accounting.generatedCandidates)
    return invalid("resource-time funnel candidate counts exceed generation");
  if (accounting.screenedCandidates > accounting.generatedCandidates ||
      accounting.detailedFrontierCandidates > accounting.screenedCandidates ||
      accounting.successiveHalvingDeferredCandidates >
          accounting.screenedCandidates)
    return invalid("resource-time screening counts exceed their parent bound");
  auto evaluated = llvm::checkedAddUnsigned(accounting.estimatedCandidates,
                                            accounting.incompleteCandidates);
  if (evaluated)
    evaluated = llvm::checkedAddUnsigned(
        *evaluated, accounting.soundGateRejectedCandidates);
  auto promotedAndDeferred = llvm::checkedAddUnsigned(
      accounting.mappingFinalists, accounting.mappingCallsDeferredByModel);
  if (!evaluated || !promotedAndDeferred ||
      *promotedAndDeferred != accounting.mappingEligibleScheduleHints)
    return invalid("resource-time schedule promotion accounting is not "
                   "closed");
  if (accounting.mappingCallsWithheldByIncomplete >
      accounting.incompleteCandidates)
    return invalid("resource-time Mapping withholding accounting is not "
                   "closed");
  if (accounting.mappingPlanConstructionsAvoidedByExactMemo >
      accounting.mappingFinalists)
    return invalid("resource-time exact Mapping reuse exceeds finalists");
  auto memoAttempts = llvm::checkedAddUnsigned(
      accounting.exactInvocationMemoHits, accounting.exactInvocationMemoMisses);
  if (memoAttempts)
    memoAttempts = llvm::checkedAddUnsigned(
        *memoAttempts, accounting.exactInvocationMemoCoalescedUncachedResults);
  if (!memoAttempts || *memoAttempts != accounting.detailedFrontierCandidates)
    return invalid("resource-time exact memo attempts do not cover evaluated "
                   "detailed frontiers");
  auto detailedAndDeferred =
      llvm::checkedAddUnsigned(accounting.detailedFrontierCandidates,
                               accounting.successiveHalvingDeferredCandidates);
  if (!detailedAndDeferred || *detailedAndDeferred != *evaluated)
    return invalid("resource-time successive-halving accounting is not "
                   "closed");
  if (accounting.exactInvocationMemoSingleFlightWaits <
          accounting.exactInvocationMemoCoalescedUncachedResults ||
      accounting.exactInvocationMemoSingleFlightWaits <
          accounting.exactInvocationMemoCancelledWaits)
    return invalid("resource-time exact memo wait accounting is inconsistent");
  auto projectionRequests =
      llvm::checkedAddUnsigned(accounting.dataflowProjectionCacheHits,
                               accounting.dataflowProjectionCacheMisses);
  if (!projectionRequests ||
      *projectionRequests != accounting.dataflowProjectionRequests)
    return invalid("resource-time projection cache requests are not closed");
  if (accounting.dataflowProjectionCacheCapacityBypasses >
      accounting.dataflowProjectionCacheMisses)
    return invalid("resource-time projection cache bypasses exceed misses");
  if (accounting.dataflowProjectionCacheEntries >
      accounting.dataflowProjectionCacheMisses)
    return invalid("resource-time projection cache entries exceed misses");
  if (accounting.mappingPlanCandidates > accounting.mappingFinalists ||
      accounting.unsupportedBeforeMappingScheduleHints >
          accounting.mappingFinalists ||
      accounting.unsupportedBeforeMappingCandidates >
          accounting.generatedCandidates)
    return invalid("resource-time application promotion counts exceed their "
                   "bounded domains");
  const ResourceTimeScreeningCalibration &calibration =
      accounting.screeningCalibration;
  auto exactClassified =
      llvm::checkedAddUnsigned(calibration.exactFeasibleCandidates,
                               calibration.exactInfeasibleCandidates);
  auto confidenceClassified = llvm::checkedAddUnsigned(
      calibration.noConfidenceCandidates, calibration.lowConfidenceCandidates);
  if (confidenceClassified)
    confidenceClassified = llvm::checkedAddUnsigned(
        *confidenceClassified, calibration.calibratedConfidenceCandidates);
  if (confidenceClassified)
    confidenceClassified = llvm::checkedAddUnsigned(
        *confidenceClassified, calibration.outOfDomainConfidenceCandidates);
  if (!exactClassified || *exactClassified != calibration.comparedCandidates ||
      !confidenceClassified ||
      *confidenceClassified != calibration.comparedCandidates ||
      calibration.screeningAdmissibleCandidates >
          calibration.comparedCandidates ||
      calibration.feasibleIntersection > calibration.exactFeasibleCandidates ||
      calibration.feasibleIntersection >
          calibration.screeningAdmissibleCandidates ||
      calibration.bestRankMatches > calibration.bestComparisonCandidates ||
      calibration.bestComparisonCandidates > 1 ||
      calibration.outOfDomainCandidates > calibration.comparedCandidates ||
      calibration.outOfDomainCandidates !=
          calibration.outOfDomainConfidenceCandidates ||
      calibration.errorSamples != calibration.exactFeasibleCandidates ||
      calibration.meanAbsoluteErrorPicoseconds !=
          (calibration.errorSamples == 0
               ? 0
               : calibration.totalAbsoluteErrorPicoseconds /
                     calibration.errorSamples) ||
      calibration.meanRelativeErrorPermille !=
          (calibration.errorSamples == 0
               ? 0
               : calibration.totalRelativeErrorPermille /
                     calibration.errorSamples) ||
      calibration.feasibilityRecallPermille !=
          permille(calibration.feasibleIntersection,
                   calibration.exactFeasibleCandidates) ||
      calibration.feasibilityPrecisionPermille !=
          permille(calibration.feasibleIntersection,
                   calibration.screeningAdmissibleCandidates) ||
      calibration.bestRecallPermille !=
          permille(calibration.bestRankMatches,
                   calibration.bestComparisonCandidates) ||
      calibration.outOfDomainPermille !=
          permille(calibration.outOfDomainCandidates,
                   calibration.comparedCandidates) ||
      calibration.lowerBoundViolations != 0)
    return invalid("resource-time screening calibration is not closed");
  if (accounting.applicationPromotionAccountingComplete &&
      accounting.mappingPlanCandidates +
              accounting.mappingPlanConstructionsAvoidedByExactMemo +
              accounting.unsupportedBeforeMappingScheduleHints !=
          accounting.mappingFinalists)
    return invalid("resource-time Mapping finalist disposition is not closed");
  return llvm::Error::success();
}

llvm::Error
accumulateResourceTimeWorkCounter(ResourceTimeWorkCounter &destination,
                                  const ResourceTimeWorkCounter &source) {
  const auto add = [](std::uint64_t &value, std::uint64_t increment,
                      llvm::StringRef name) -> llvm::Error {
    auto result = llvm::checkedAddUnsigned(value, increment);
    if (!result)
      return invalid("resource-time " + name + " work counter overflowed");
    value = *result;
    return llvm::Error::success();
  };
  if (llvm::Error error = add(destination.limit, source.limit, "limit"))
    return error;
  if (llvm::Error error = add(destination.planned, source.planned, "planned"))
    return error;
  if (llvm::Error error =
          add(destination.reserved, source.reserved, "reserved"))
    return error;
  if (llvm::Error error =
          add(destination.consumed, source.consumed, "consumed"))
    return error;
  if (llvm::Error error =
          add(destination.rejected, source.rejected, "rejected"))
    return error;
  if (llvm::Error error =
          add(destination.cancelled, source.cancelled, "cancelled"))
    return error;
  if (llvm::Error error = add(destination.elapsedNanoseconds,
                              source.elapsedNanoseconds, "elapsed"))
    return error;
  return llvm::Error::success();
}

llvm::Error accumulateResourceTimeFrontierAccounting(
    ResourceTimeFrontierAccounting &destination,
    const ResourceTimeFrontierAccounting &source) {
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.sourceProjections, source.sourceProjections))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(destination.actions,
                                                            source.actions))
    return error;
  if (llvm::Error error =
          accumulateResourceTimeWorkCounter(destination.states, source.states))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.estimates, source.estimates))
    return error;
  if (llvm::Error error = accumulateResourceTimeWorkCounter(
          destination.finalists, source.finalists))
    return error;
  auto add = [](std::uint64_t &value, std::uint64_t increment,
                llvm::StringRef name) -> llvm::Error {
    auto result = llvm::checkedAddUnsigned(value, increment);
    if (!result)
      return invalid("resource-time " + name + " accounting overflowed");
    value = *result;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          add(destination.stateMemoHits, source.stateMemoHits, "memo hits"))
    return error;
  if (llvm::Error error = add(destination.stateMemoMisses,
                              source.stateMemoMisses, "memo misses"))
    return error;
  if (llvm::Error error =
          add(destination.stateMemoParetoInsertions,
              source.stateMemoParetoInsertions, "memo Pareto insertions"))
    return error;
  if (llvm::Error error =
          add(destination.stateMemoDominatedStates,
              source.stateMemoDominatedStates, "memo dominated states"))
    return error;
  if (llvm::Error error = add(destination.stateMemoHitCapacityRejections,
                              source.stateMemoHitCapacityRejections,
                              "memo hit capacity rejections"))
    return error;
  if (llvm::Error error = add(destination.stateMemoMissCapacityRejections,
                              source.stateMemoMissCapacityRejections,
                              "memo miss capacity rejections"))
    return error;
  if (llvm::Error error = add(destination.statesPrunedByBeam,
                              source.statesPrunedByBeam, "beam pruning"))
    return error;
  if (llvm::Error error =
          add(destination.terminalHintsGenerated, source.terminalHintsGenerated,
              "terminal hints generated"))
    return error;
  if (llvm::Error error =
          add(destination.terminalHintsRetained, source.terminalHintsRetained,
              "terminal hints retained"))
    return error;
  if (llvm::Error error =
          add(destination.terminalHintsPruned, source.terminalHintsPruned,
              "terminal hints pruned"))
    return error;
  if (llvm::Error error = add(destination.incrementalLowerBoundUpdates,
                              source.incrementalLowerBoundUpdates,
                              "incremental lower-bound updates"))
    return error;
  destination.maximumRetainedBytes =
      std::max(destination.maximumRetainedBytes, source.maximumRetainedBytes);
  return llvm::Error::success();
}

llvm::Expected<ResourceTimeMappingFunnel> selectResourceTimeMappingFinalists(
    llvm::ArrayRef<ResourceTimeMappingCandidateInput> candidates,
    const ResourceTimeFrontierPolicy &policy,
    ExecutionControlView executionControl,
    ResourceTimeFrontierSession *session) {
  if (candidates.empty() || policy.maximumMappingFinalists == 0 ||
      policy.maximumInvocationMemoEntries == 0 ||
      policy.maximumInvocationMemoBytes == 0)
    return invalid("resource-time Mapping funnel bounds must be positive");
  for (std::size_t index = 0; index != candidates.size(); ++index)
    for (std::size_t prior = 0; prior != index; ++prior) {
      if (candidates[prior].candidateIdentity ==
          candidates[index].candidateIdentity)
        return invalid("resource-time Mapping funnel has a duplicate semantic "
                       "candidate identity");
      if (candidates[prior].inputPreferenceRank ==
          candidates[index].inputPreferenceRank)
        return invalid("resource-time Mapping funnel has a duplicate input "
                       "preference rank");
    }

  const auto begin = MonotonicClock::now();
  ResourceTimeMappingFunnel result;
  result.accounting.generatedCandidates = candidates.size();
  result.evaluations.reserve(candidates.size());
  std::unique_ptr<ResourceTimeFrontierSession> localSession;
  if (!session) {
    localSession = std::make_unique<ResourceTimeFrontierSession>(
        policy.maximumInvocationMemoEntries, policy.maximumInvocationMemoBytes);
    session = localSession.get();
  }
  struct ScreenedCandidate final {
    std::size_t index = 0;
    ResourceTimeCandidateScreening screening;
  };
  std::vector<ScreenedCandidate> screened;
  screened.reserve(candidates.size());
  for (auto indexed : llvm::enumerate(candidates)) {
    if (executionControl.stopRequested()) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      break;
    }
    auto screening = screenCandidate(indexed.value(), policy);
    if (!screening)
      return screening.takeError();
    screened.push_back({indexed.index(), std::move(*screening)});
    ++result.accounting.screenedCandidates;
  }

  const auto screenedLess = [&](std::size_t lhs, std::size_t rhs) {
    const auto &left = screened[lhs];
    const auto &right = screened[rhs];
    const auto &leftCandidate = candidates[left.index];
    const auto &rightCandidate = candidates[right.index];
    const auto leftKey = std::tuple(estimateSupportRank(left.screening.support),
                                    left.screening.lowerBoundPicoseconds,
                                    left.screening.featureScore,
                                    leftCandidate.maximumUsefulResourceUnits,
                                    leftCandidate.candidateIdentity.bytes());
    const auto rightKey = std::tuple(
        estimateSupportRank(right.screening.support),
        right.screening.lowerBoundPicoseconds, right.screening.featureScore,
        rightCandidate.maximumUsefulResourceUnits,
        rightCandidate.candidateIdentity.bytes());
    return leftKey < rightKey;
  };
  std::vector<std::size_t> ranked(screened.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  llvm::sort(ranked, screenedLess);
  std::vector<std::size_t> promotionOrder;
  promotionOrder.reserve(ranked.size());
  const auto appendPromotion = [&](std::size_t screenedOrdinal) {
    if (screenedOrdinal >= screened.size() ||
        llvm::is_contained(promotionOrder, screenedOrdinal))
      return;
    promotionOrder.push_back(screenedOrdinal);
  };
  if (!ranked.empty())
    appendPromotion(ranked.front());
  if (!ranked.empty()) {
    const auto minimumCoverage = *std::min_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.acceleratedRegionCount,
                            left.acceleratedGraphCount,
                            left.acceleratedActorCount,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.acceleratedRegionCount,
                            right.acceleratedGraphCount,
                            right.acceleratedActorCount,
                            right.candidateIdentity.bytes());
        });
    const auto maximumCoverage = *std::max_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.acceleratedRegionCount,
                            left.acceleratedGraphCount,
                            left.acceleratedActorCount,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.acceleratedRegionCount,
                            right.acceleratedGraphCount,
                            right.acceleratedActorCount,
                            right.candidateIdentity.bytes());
        });
    appendPromotion(minimumCoverage);
    appendPromotion(maximumCoverage);
    const auto maximumConcentration = *std::max_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          const auto &left = candidates[screened[lhs].index];
          const auto &right = candidates[screened[rhs].index];
          return std::tuple(left.maximumUsefulResourceUnits,
                            left.candidateIdentity.bytes()) <
                 std::tuple(right.maximumUsefulResourceUnits,
                            right.candidateIdentity.bytes());
        });
    appendPromotion(maximumConcentration);
    const auto canonical = *std::min_element(
        ranked.begin(), ranked.end(), [&](std::size_t lhs, std::size_t rhs) {
          return candidates[screened[lhs].index].candidateIdentity.bytes() <
                 candidates[screened[rhs].index].candidateIdentity.bytes();
        });
    appendPromotion(canonical);
  }
  for (std::size_t screenedOrdinal : ranked)
    appendPromotion(screenedOrdinal);

  std::vector<std::optional<ResourceTimeCandidateFunnelEvaluation>> evaluations(
      candidates.size());
  std::uint64_t detailedWithHint = 0;
  const auto evaluateDetailed =
      [&](std::size_t screenedOrdinal) -> llvm::Expected<bool> {
    const ScreenedCandidate &screenedCandidate = screened[screenedOrdinal];
    const ResourceTimeMappingCandidateInput &candidate =
        candidates[screenedCandidate.index];
    const std::string memoKey =
        exactFrontierMemoKey(candidate.invocation, candidate.resourceClasses,
                             candidate.regions, policy);
    auto lookup = session->lookupOrCompute(
        memoKey,
        [&]() {
          return exploreResourceTimeFrontier(
              candidate.invocation, candidate.resourceClasses,
              candidate.regions, policy, executionControl);
        },
        executionControl);
    if (!lookup)
      return lookup.takeError();
    result.accounting.exactInvocationMemoHits += lookup->cacheHit;
    result.accounting.exactInvocationMemoMisses += lookup->cacheMiss;
    result.accounting.exactInvocationMemoSingleFlightWaits += lookup->waited;
    result.accounting.exactInvocationMemoCoalescedUncachedResults +=
        lookup->coalescedUncachedResult;
    result.accounting.exactInvocationMemoCancelledWaits +=
        lookup->cancelledWait;
    result.accounting.exactInvocationMemoCapacityBypasses +=
        lookup->capacityBypass;
    if (lookup->cancelledWait) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      return false;
    }
    const ResourceTimeFrontierOutcome *outcome = lookup->outcome.get();
    if (!outcome)
      return invalid("resource-time frontier memo produced no outcome");
    ResourceTimeCandidateFunnelEvaluation evaluation{
        candidate.candidateIdentity,
        candidate.inputPreferenceRank,
        candidate.acceleratedRegionCount,
        candidate.acceleratedGraphCount,
        candidate.acceleratedActorCount,
        candidate.maximumUsefulResourceUnits,
        ResourceTimeCandidateFunnelDisposition::Incomplete,
        screenedCandidate.screening.lowerBoundPicoseconds,
        screenedCandidate.screening.featureScore,
        screenedCandidate.screening.support,
        confidenceForSupport(screenedCandidate.screening.support),
        screenedCandidate.screening.exactCapacityFailure,
        true,
        false,
        std::nullopt,
        std::nullopt,
        {},
        std::nullopt,
        std::nullopt,
        {}};
    if (auto *completed = std::get_if<CompletedResourceTimeFrontier>(outcome)) {
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::Estimated;
      evaluation.concurrencyBounds = completed->concurrencyBounds;
      evaluation.detailedDomainExhaustive = completed->domainExhaustive;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = completed->accounting;
      evaluation.retainedHints = completed->finalists;
      if (!evaluation.retainedHints.empty())
        evaluation.bestHint = completed->finalists.front();
      ++result.accounting.estimatedCandidates;
    } else if (auto *incomplete =
                   std::get_if<IncompleteResourceTimeFrontier>(outcome)) {
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::Incomplete;
      evaluation.incompleteReason = incomplete->reason;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = incomplete->accounting;
      evaluation.retainedHints = incomplete->retainedFinalists;
      if (!evaluation.retainedHints.empty())
        evaluation.bestHint = incomplete->retainedFinalists.front();
      ++result.accounting.incompleteCandidates;
      if (!result.incompleteReason ||
          incompleteReasonPriority(incomplete->reason) >
              incompleteReasonPriority(*result.incompleteReason))
        result.incompleteReason = incomplete->reason;
    } else {
      const auto &infeasible =
          std::get<ProvenInfeasibleResourceTimeFrontier>(*outcome);
      evaluation.disposition =
          ResourceTimeCandidateFunnelDisposition::SoundGateRejected;
      evaluation.infeasibleReason = infeasible.reason;
      if (lookup->cacheMiss)
        evaluation.frontierAccounting = infeasible.accounting;
      ++result.accounting.soundGateRejectedCandidates;
    }
    ++result.accounting.detailedFrontierCandidates;
    detailedWithHint += evaluation.bestHint.has_value();
    evaluations[screenedCandidate.index] = std::move(evaluation);
    if (llvm::Error error = accumulateResourceTimeFrontierAccounting(
            result.accounting.frontierAccounting,
            evaluations[screenedCandidate.index]->frontierAccounting))
      return std::move(error);
    if (result.incompleteReason ==
        ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
      return false;
    return true;
  };

  // Exact no-fit candidates are cheap necessary-condition checks and do not
  // consume a detailed-survivor slot. Every other candidate advances in the
  // deterministic screening/diversity order until enough real Mapping
  // finalists have a schedule hint. Remaining candidates stay analytic
  // estimates and are never called infeasible.
  for (std::size_t screenedOrdinal : promotionOrder) {
    if (executionControl.stopRequested()) {
      result.incompleteReason =
          ResourceTimeFrontierIncompleteReason::CancelledOrTimeout;
      break;
    }
    const ScreenedCandidate &screenedCandidate = screened[screenedOrdinal];
    if (evaluations[screenedCandidate.index])
      continue;
    if (!screenedCandidate.screening.exactCapacityFailure &&
        detailedWithHint >= policy.maximumMappingFinalists)
      continue;
    auto keepGoing = evaluateDetailed(screenedOrdinal);
    if (!keepGoing)
      return keepGoing.takeError();
    if (!*keepGoing)
      break;
  }

  if (result.incompleteReason !=
      ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
    for (const ScreenedCandidate &screenedCandidate : screened) {
      if (evaluations[screenedCandidate.index])
        continue;
      const auto &candidate = candidates[screenedCandidate.index];
      evaluations[screenedCandidate.index] =
          ResourceTimeCandidateFunnelEvaluation{
              candidate.candidateIdentity,
              candidate.inputPreferenceRank,
              candidate.acceleratedRegionCount,
              candidate.acceleratedGraphCount,
              candidate.acceleratedActorCount,
              candidate.maximumUsefulResourceUnits,
              ResourceTimeCandidateFunnelDisposition::Estimated,
              screenedCandidate.screening.lowerBoundPicoseconds,
              screenedCandidate.screening.featureScore,
              screenedCandidate.screening.support,
              confidenceForSupport(screenedCandidate.screening.support),
              screenedCandidate.screening.exactCapacityFailure,
              false,
              false,
              std::nullopt,
              std::nullopt,
              {},
              std::nullopt,
              std::nullopt,
              {}};
      ++result.accounting.estimatedCandidates;
      ++result.accounting.successiveHalvingDeferredCandidates;
    }
  for (auto &evaluation : evaluations)
    if (evaluation)
      result.evaluations.push_back(std::move(*evaluation));

  ResourceTimeScreeningCalibration &calibration =
      result.accounting.screeningCalibration;
  const ResourceTimeCandidateFunnelEvaluation *detailedBest = nullptr;
  const ResourceTimeCandidateFunnelEvaluation *screeningBest = nullptr;
  for (const ResourceTimeCandidateFunnelEvaluation &evaluation :
       result.evaluations) {
    if (!evaluation.detailedFrontierEvaluated)
      continue;
    const bool exactFeasible =
        evaluation.disposition ==
            ResourceTimeCandidateFunnelDisposition::Estimated &&
        evaluation.bestHint && evaluation.detailedDomainExhaustive;
    const bool exactInfeasible =
        evaluation.disposition ==
        ResourceTimeCandidateFunnelDisposition::SoundGateRejected;
    if (!exactFeasible && !exactInfeasible)
      continue;
    ++calibration.comparedCandidates;
    calibration.exactFeasibleCandidates += exactFeasible;
    calibration.exactInfeasibleCandidates += exactInfeasible;
    const bool outOfDomain =
        evaluation.screeningSupport == ResourceTimeEstimateSupport::OutOfDomain;
    if (outOfDomain)
      ++calibration.outOfDomainCandidates;
    switch (evaluation.screeningConfidence) {
    case ResourceTimeEstimateConfidence::None:
      ++calibration.noConfidenceCandidates;
      break;
    case ResourceTimeEstimateConfidence::Low:
      ++calibration.lowConfidenceCandidates;
      break;
    case ResourceTimeEstimateConfidence::Calibrated:
      ++calibration.calibratedConfidenceCandidates;
      break;
    case ResourceTimeEstimateConfidence::OutOfDomain:
      ++calibration.outOfDomainConfidenceCandidates;
      break;
    }
    const bool admissible = !evaluation.screeningExactCapacityFailure &&
                            evaluation.screeningSupport !=
                                ResourceTimeEstimateSupport::Unsupported &&
        !outOfDomain;
    if (admissible)
      ++calibration.screeningAdmissibleCandidates;
    if (admissible && exactFeasible)
      ++calibration.feasibleIntersection;
    if (admissible &&
        (!screeningBest ||
         std::tuple(evaluation.screeningLowerBoundPicoseconds,
                    evaluation.candidateIdentity.bytes()) <
             std::tuple(screeningBest->screeningLowerBoundPicoseconds,
                        screeningBest->candidateIdentity.bytes())))
      screeningBest = &evaluation;
    if (exactFeasible &&
        (!detailedBest ||
        std::tuple(evaluation.bestHint->estimatedMakespanPicoseconds,
                   evaluation.candidateIdentity.bytes()) <
            std::tuple(detailedBest->bestHint->estimatedMakespanPicoseconds,
                        detailedBest->candidateIdentity.bytes())))
      detailedBest = &evaluation;
    if (!exactFeasible)
      continue;
    ++calibration.errorSamples;
    const std::uint64_t exactMakespan =
        evaluation.bestHint->estimatedMakespanPicoseconds;
    if (exactMakespan == 0)
      return invalid("resource-time exact calibration makespan is zero");
    if (evaluation.screeningLowerBoundPicoseconds > exactMakespan)
      ++calibration.lowerBoundViolations;
    const std::uint64_t error =
        evaluation.screeningLowerBoundPicoseconds > exactMakespan
            ? evaluation.screeningLowerBoundPicoseconds - exactMakespan
            : exactMakespan - evaluation.screeningLowerBoundPicoseconds;
    const auto totalAbsolute = llvm::checkedAddUnsigned(
        calibration.totalAbsoluteErrorPicoseconds, error);
    const std::uint64_t relativeError = permille(error, exactMakespan);
    const auto totalRelative = llvm::checkedAddUnsigned(
        calibration.totalRelativeErrorPermille, relativeError);
    if (!totalAbsolute || !totalRelative)
      return invalid("resource-time screening calibration error overflows");
    calibration.totalAbsoluteErrorPicoseconds = *totalAbsolute;
    calibration.maximumAbsoluteErrorPicoseconds =
        std::max(calibration.maximumAbsoluteErrorPicoseconds, error);
    calibration.totalRelativeErrorPermille = *totalRelative;
    calibration.maximumRelativeErrorPermille =
        std::max(calibration.maximumRelativeErrorPermille, relativeError);
    }
    calibration.meanAbsoluteErrorPicoseconds =
        calibration.errorSamples == 0
            ? 0
            : calibration.totalAbsoluteErrorPicoseconds /
                  calibration.errorSamples;
    calibration.meanRelativeErrorPermille =
        calibration.errorSamples == 0
            ? 0
            : calibration.totalRelativeErrorPermille / calibration.errorSamples;
    if (detailedBest && screeningBest) {
      calibration.bestComparisonCandidates = 1;
      calibration.bestRankMatches =
          detailedBest->candidateIdentity == screeningBest->candidateIdentity;
  }
    calibration.feasibilityRecallPermille = permille(
        calibration.feasibleIntersection, calibration.exactFeasibleCandidates);
    calibration.feasibilityPrecisionPermille =
        permille(calibration.feasibleIntersection,
                 calibration.screeningAdmissibleCandidates);
    calibration.bestRecallPermille = permille(
        calibration.bestRankMatches, calibration.bestComparisonCandidates);
    calibration.outOfDomainPermille = permille(
        calibration.outOfDomainCandidates, calibration.comparedCandidates);

  if (result.incompleteReason ==
      ResourceTimeFrontierIncompleteReason::CancelledOrTimeout) {
    result.truncated = true;
  }

  struct EligibleHint final {
    const ResourceTimeCandidateFunnelEvaluation *evaluation = nullptr;
    const ResourceTimeScheduleHint *hint = nullptr;
    ComponentViewDigest digest;
  };
  std::vector<EligibleHint> eligible;
  std::uint64_t withheldByIncomplete = 0;
  for (const auto &evaluation : result.evaluations) {
    if (evaluation.disposition ==
        ResourceTimeCandidateFunnelDisposition::SoundGateRejected)
      continue;
    if (!evaluation.detailedFrontierEvaluated)
      continue;
    // A timeout is a terminal incomplete checkpoint for this invocation. A
    // budget-bounded candidate may still promote a retained hint, but no
    // candidate without an explicit hint may trigger real Mapping work.
    if (evaluation.retainedHints.empty() ||
        (evaluation.incompleteReason &&
         *evaluation.incompleteReason ==
             ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)) {
      ++withheldByIncomplete;
      continue;
    }
    for (const ResourceTimeScheduleHint &hint : evaluation.retainedHints) {
      auto digest = deriveResourceTimeScheduleHintDigest(hint);
      if (!digest)
        return digest.takeError();
      const bool duplicate = llvm::any_of(eligible, [&](const auto &existing) {
        return existing.evaluation->candidateIdentity ==
                   evaluation.candidateIdentity &&
               existing.digest == *digest;
      });
      if (duplicate)
        return invalid("resource-time candidate has duplicate schedule "
                       "provenance");
      eligible.push_back({&evaluation, &hint, *digest});
    }
  }
  const auto entryLess = [&](const EligibleHint *lhs, const EligibleHint *rhs) {
    const auto endpointLess = [&](const ResourceTimeScheduleHint &left,
                                  const ResourceTimeScheduleHint &right) {
      switch (policy.spectrumEndpoint) {
      case PreMappingSpectrumEndpoint::MaxSpatial:
        return std::tuple(-left.peakConcurrentRegions,
                          left.estimatedMakespanPicoseconds,
                          left.totalAllocatedResourceTime) <
               std::tuple(-right.peakConcurrentRegions,
                          right.estimatedMakespanPicoseconds,
                          right.totalAllocatedResourceTime);
      case PreMappingSpectrumEndpoint::MaxTemporal:
        return std::tie(left.peakConcurrentRegions,
                        left.estimatedMakespanPicoseconds,
                        left.totalAllocatedResourceTime) <
               std::tie(right.peakConcurrentRegions,
                        right.estimatedMakespanPicoseconds,
                        right.totalAllocatedResourceTime);
      case PreMappingSpectrumEndpoint::Intermediate:
      case PreMappingSpectrumEndpoint::Automatic:
        return hintLess(left, right);
      }
      llvm_unreachable("unknown resource-time spectrum endpoint");
    };
    if (endpointLess(*lhs->hint, *rhs->hint))
      return true;
    if (endpointLess(*rhs->hint, *lhs->hint))
      return false;
    if (lhs->evaluation->candidateIdentity !=
        rhs->evaluation->candidateIdentity)
      return lhs->evaluation->candidateIdentity.bytes() <
             rhs->evaluation->candidateIdentity.bytes();
    return lhs->digest.bytes() < rhs->digest.bytes();
  };
  std::vector<const EligibleHint *> rankedHints;
  rankedHints.reserve(eligible.size());
  for (const EligibleHint &entry : eligible)
    rankedHints.push_back(&entry);
  llvm::sort(rankedHints, entryLess);
  const std::uint64_t limit = std::min<std::uint64_t>(
      policy.maximumMappingFinalists, rankedHints.size());
  std::vector<const EligibleHint *> selected;
  selected.reserve(limit);
  const auto append = [&](const EligibleHint *value) {
    if (!value || selected.size() == limit ||
        llvm::is_contained(selected, value))
      return;
    selected.push_back(value);
  };
  if (!rankedHints.empty())
    append(rankedHints.front());
  const EligibleHint *minimumConcurrency = nullptr;
  const EligibleHint *maximumConcurrency = nullptr;
  for (const EligibleHint *candidate : rankedHints) {
    if (!minimumConcurrency ||
        std::tie(candidate->hint->peakConcurrentRegions,
                 candidate->hint->estimatedMakespanPicoseconds) <
            std::tie(minimumConcurrency->hint->peakConcurrentRegions,
                     minimumConcurrency->hint->estimatedMakespanPicoseconds))
      minimumConcurrency = candidate;
    if (!maximumConcurrency ||
        candidate->hint->peakConcurrentRegions >
            maximumConcurrency->hint->peakConcurrentRegions ||
        (candidate->hint->peakConcurrentRegions ==
             maximumConcurrency->hint->peakConcurrentRegions &&
         candidate->hint->estimatedMakespanPicoseconds <
             maximumConcurrency->hint->estimatedMakespanPicoseconds))
      maximumConcurrency = candidate;
  }
  append(minimumConcurrency);
  append(maximumConcurrency);
  const EligibleHint *minimumCoverage = nullptr;
  const EligibleHint *maximumCoverage = nullptr;
  for (const EligibleHint *candidate : rankedHints) {
    const auto *evaluation = candidate->evaluation;
    if (!minimumCoverage ||
        std::tie(evaluation->acceleratedRegionCount,
                 evaluation->acceleratedGraphCount,
                 evaluation->acceleratedActorCount,
                 evaluation->inputPreferenceRank) <
            std::tie(minimumCoverage->evaluation->acceleratedRegionCount,
                     minimumCoverage->evaluation->acceleratedGraphCount,
                     minimumCoverage->evaluation->acceleratedActorCount,
                     minimumCoverage->evaluation->inputPreferenceRank))
      minimumCoverage = candidate;
    if (!maximumCoverage ||
        evaluation->acceleratedRegionCount >
            maximumCoverage->evaluation->acceleratedRegionCount ||
        (evaluation->acceleratedRegionCount ==
             maximumCoverage->evaluation->acceleratedRegionCount &&
         evaluation->acceleratedGraphCount >
             maximumCoverage->evaluation->acceleratedGraphCount) ||
        (evaluation->acceleratedRegionCount ==
             maximumCoverage->evaluation->acceleratedRegionCount &&
         evaluation->acceleratedGraphCount ==
             maximumCoverage->evaluation->acceleratedGraphCount &&
         evaluation->acceleratedActorCount >
             maximumCoverage->evaluation->acceleratedActorCount))
      maximumCoverage = candidate;
  }
  append(minimumCoverage);
  append(maximumCoverage);
  const EligibleHint *maximumConcentration = nullptr;
  for (const EligibleHint *candidate : rankedHints)
    if (!maximumConcentration ||
        candidate->evaluation->maximumUsefulResourceUnits >
            maximumConcentration->evaluation->maximumUsefulResourceUnits ||
        (candidate->evaluation->maximumUsefulResourceUnits ==
             maximumConcentration->evaluation->maximumUsefulResourceUnits &&
         entryLess(candidate, maximumConcentration)))
      maximumConcentration = candidate;
  append(maximumConcentration);
  if (!rankedHints.empty()) {
    const auto canonical = *std::min_element(
        rankedHints.begin(), rankedHints.end(),
        [](const auto *lhs, const auto *rhs) {
          if (lhs->evaluation->candidateIdentity !=
              rhs->evaluation->candidateIdentity)
            return lhs->evaluation->candidateIdentity.bytes() <
                   rhs->evaluation->candidateIdentity.bytes();
          return lhs->digest.bytes() < rhs->digest.bytes();
        });
    append(canonical);
  }
  for (const EligibleHint *candidate : rankedHints)
    append(candidate);
  llvm::sort(selected, entryLess);
  for (const EligibleHint *candidate : selected)
    result.finalists.push_back(
        {candidate->evaluation->candidateIdentity, candidate->digest});
  result.accounting.mappingEligibleScheduleHints = eligible.size();
  result.accounting.mappingFinalists = result.finalists.size();
  result.accounting.mappingCallsDeferredByModel =
      eligible.size() - result.finalists.size();
  result.accounting.mappingCallsWithheldByIncomplete = withheldByIncomplete;
  const ResourceTimeFrontierSessionStatistics sessionStatistics =
      session->statistics();
  result.accounting.exactInvocationMemoEntries = sessionStatistics.entryCount;
  result.accounting.exactInvocationMemoRetainedBytes =
      sessionStatistics.retainedBytes;
  result.truncated = result.truncated ||
                     result.finalists.size() < eligible.size() ||
                     result.evaluations.size() < candidates.size() ||
                     result.accounting.successiveHalvingDeferredCandidates != 0;
  result.accounting.elapsedNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          MonotonicClock::now() - begin)
          .count();
  if (llvm::Error error =
          validateResourceTimeMappingFunnelAccounting(result.accounting))
    return std::move(error);
  return result;
}

} // namespace loom::dse
