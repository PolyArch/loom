#include "Fabric/IR/FifoResourceContract.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <system_error>
#include <utility>
#include <vector>

using namespace fabric;

namespace {

constexpr StateKey bufferedQueueState{0};
constexpr StateKey bypassTransferState{1};
constexpr CapacityDimensionKey queueSlotCapacity{0};
constexpr CapacityDimensionKey enqueueServiceCapacity{1};
constexpr CapacityDimensionKey dequeueServiceCapacity{2};
constexpr CapacityDimensionKey bypassServiceCapacity{0};
constexpr ResourceTransitionKey appendTransition{0};
constexpr ResourceTransitionKey removeTransition{1};
constexpr ResourceTransitionKey replaceHeadTransition{2};
constexpr RequesterKey fifoRequester{0};
constexpr EligibilityKey enqueueEligible{0};
constexpr EligibilityKey dequeueEligible{1};
constexpr EligibilityKey simultaneousEligible{2};
constexpr EligibilityKey bypassEligible{3};
constexpr EventKey acquireEvent{0};
constexpr EventKey commitEvent{1};
constexpr EventKey nextClockBoundary{2};
constexpr EventKey bypassTransferEvent{3};
constexpr TimingContractKey bufferedTiming{0};
constexpr TimingContractKey bypassTiming{1};

ClaimDeclaration claim(std::uint32_t key, StateKey state,
                       CapacityDimensionKey dimension) {
  return ClaimDeclaration{ClaimKey(key), state, dimension, CapacityUnits(1)};
}

UsePatternDeclaration bufferedPattern(FifoUsePattern key,
                                      EligibilityKey eligibility,
                                      ResourceTransitionKey transition,
                                      std::vector<ClaimDeclaration> claims) {
  return UsePatternDeclaration{fifoUsePattern(key),
                               fifoRequester,
                               eligibility,
                               acquireEvent,
                               nextClockBoundary,
                               CommitDeclaration{commitEvent, transition},
                               bufferedTiming,
                               std::move(claims),
                               {}};
}

} // namespace

ResourceContractDeclaration
fabric::declareFifoResourceContract(std::uint32_t maxDepth, bool bypassable) {
  ResourceContractDeclaration declaration;
  if (maxDepth == 0)
    return declaration;

  declaration.states = {ResourceStateDeclaration{
      bufferedQueueState,
      {CapacityDimensionDeclaration{queueSlotCapacity, CapacityUnits(maxDepth),
                                    CapacityUnits(0)},
       CapacityDimensionDeclaration{enqueueServiceCapacity, CapacityUnits(1),
                                    CapacityUnits(0)},
       CapacityDimensionDeclaration{dequeueServiceCapacity, CapacityUnits(1),
                                    CapacityUnits(0)}}}};
  if (bypassable)
    declaration.states.push_back(ResourceStateDeclaration{
        bypassTransferState,
        {CapacityDimensionDeclaration{bypassServiceCapacity, CapacityUnits(1),
                                      CapacityUnits(0)}}});

  declaration.resourceTransitions = {appendTransition, removeTransition,
                                     replaceHeadTransition};
  declaration.timingContracts = {
      TimingContractDeclaration{bufferedTiming, {0, 1, 2, 0}}};
  if (bypassable)
    declaration.timingContracts.push_back(
        TimingContractDeclaration{bypassTiming, {0, 0, 0, 0}});
  declaration.requesters = {fifoRequester};
  declaration.eligibilityCount = bypassable ? 4 : 3;
  declaration.eventCount = 4;
  declaration.usePatterns = {
      bufferedPattern(FifoUsePattern::Enqueue, enqueueEligible,
                      appendTransition,
                      {claim(0, bufferedQueueState, queueSlotCapacity),
                       claim(1, bufferedQueueState, enqueueServiceCapacity)}),
      bufferedPattern(FifoUsePattern::Dequeue, dequeueEligible,
                      removeTransition,
                      {claim(0, bufferedQueueState, dequeueServiceCapacity)}),
      bufferedPattern(FifoUsePattern::SimultaneousDequeueEnqueue,
                      simultaneousEligible, replaceHeadTransition,
                      {claim(0, bufferedQueueState, queueSlotCapacity),
                       claim(1, bufferedQueueState, enqueueServiceCapacity),
                       claim(2, bufferedQueueState, dequeueServiceCapacity)}),
  };
  if (bypassable)
    declaration.usePatterns.push_back(UsePatternDeclaration{
        fifoUsePattern(FifoUsePattern::BypassTransfer),
        fifoRequester,
        bypassEligible,
        bypassTransferEvent,
        bypassTransferEvent,
        std::nullopt,
        bypassTiming,
        {claim(0, bypassTransferState, bypassServiceCapacity)},
        {}});
  return declaration;
}

llvm::Expected<ResourceContract>
fabric::createFifoResourceContract(std::uint32_t maxDepth, bool bypassable) {
  if (maxDepth == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "FIFO max depth must be positive");
  return ResourceContract::create(
      declareFifoResourceContract(maxDepth, bypassable));
}
