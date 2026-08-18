#include "CGRAPhysicalActionRuntime.h"

#include "Evaluation/NumericValue.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA physical action runtime test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::sim::SpatialEventCoordinate coordinate(std::uint64_t cycle,
                                             std::uint32_t delta = 0) {
  return {take(loom::evaluation::ExactRatio::get(cycle, 1)), delta};
}

fabric::ResourceContract createContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {
      {StateKey(0),
       {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)}}}};
  declaration.resourceTransitions = {ResourceTransitionKey(0)};
  declaration.timingContracts = {{TimingContractKey(0), {0, 1, 2}}};
  declaration.requesters = {RequesterKey(0), RequesterKey(1)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 3;
  for (std::uint32_t requester = 0; requester != 2; ++requester)
    declaration.usePatterns.push_back(
        {UsePatternKey(requester),
         RequesterKey(requester),
         EligibilityKey(0),
         EventKey(0),
         EventKey(2),
         CommitDeclaration{EventKey(1), ResourceTransitionKey(0)},
         TimingContractKey(0),
         {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0),
           CapacityUnits(1)}},
         {}});
  declaration.grantPolicy =
      FixedPriorityDeclaration{{RequesterKey(0), RequesterKey(1)}};
  return take(ResourceContract::create(declaration));
}

fabric::ResourceContract createAtomicContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {
      {StateKey(0),
       {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)}}}};
  declaration.resourceTransitions = {ResourceTransitionKey(0)};
  declaration.timingContracts = {{TimingContractKey(0), {0}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 1;
  declaration.usePatterns = {
      {UsePatternKey(0),
       RequesterKey(0),
       EligibilityKey(0),
       EventKey(0),
       EventKey(0),
       CommitDeclaration{EventKey(0), ResourceTransitionKey(0)},
       TimingContractKey(0),
       {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)}},
       {}}};
  return take(ResourceContract::create(declaration));
}

void contentionProducesExactLifecycleAndStall() {
  const fabric::ResourceContract contract = createContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}, {0, fabric::UsePatternKey(1)}};
  const auto resources = take(
      loom::sim::detail::freezeCgraResourceRuntimePlan(contracts, selections));
  const loom::sim::detail::CgraPhysicalUseTiming uses[] = {
      {0, 0, 1, 2, 0, 2, 1}, {1, 0, 1, 2, 0, 2, 1}};
  auto runtime = take(
      loom::sim::detail::CgraPhysicalActionRuntime::create(resources, uses));

  const loom::sim::detail::CgraPhysicalActionRequest duplicateRequests[] = {
      {0, 7}, {0, 7}};
  auto duplicate = runtime.requestBatch(duplicateRequests, coordinate(4));
  if (duplicate)
    fail("duplicate request batch was accepted");
  llvm::consumeError(duplicate.takeError());
  if (runtime.hasPendingActions() || runtime.nextCoordinate())
    fail("rejected request batch changed physical runtime state");

  const loom::sim::detail::CgraPhysicalActionRequest requests[] = {{0, 0},
                                                                   {1, 0}};
  const auto requested = take(runtime.requestBatch(requests, coordinate(5)));
  if (requested.size() != 2 ||
      requested[0].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Requested ||
      requested[1].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Requested)
    fail("request did not expose the physical lifecycle origin");

  auto frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(5, 1)) ||
      frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Granted ||
      frame->events.front().actionOrdinal != 0)
    fail("fixed-priority acquisition did not grant the first action");

  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(6, 1)) ||
      frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Committed ||
      frame->events.front().actionOrdinal != 0)
    fail("owner commit did not occur at its exact relative rank");

  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(7, 1)) ||
      frame->events.size() != 2 ||
      frame->events[0].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Retired ||
      frame->events[0].actionOrdinal != 0 ||
      frame->events[1].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Granted ||
      frame->events[1].actionOrdinal != 1)
    fail("release did not make the blocked request grantable at the boundary");

  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(8, 1)) ||
      frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Committed ||
      frame->events.front().actionOrdinal != 1)
    fail("stalled action commit ignored its grant-relative timing");

  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(9, 1)) ||
      frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Retired ||
      frame->events.front().actionOrdinal != 1 || runtime.hasPendingActions())
    fail("whole claim-envelope retirement did not close the lifecycle");
}

void equalOwnerEventCommitsAndReleasesAtomically() {
  const fabric::ResourceContract contract = createAtomicContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}};
  const auto resources = take(
      loom::sim::detail::freezeCgraResourceRuntimePlan(contracts, selections));
  const loom::sim::detail::CgraPhysicalUseTiming uses[] = {
      {0, 0, 0, 0, 0, 0, 0}};
  auto runtime = take(
      loom::sim::detail::CgraPhysicalActionRuntime::create(resources, uses));
  (void)take(runtime.request(0, 3, coordinate(11)));

  auto frame = take(runtime.advance());
  if (!frame || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Granted)
    fail("atomic owner event did not acquire exactly once");
  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(11, 1)) ||
      frame->events.size() != 2 ||
      frame->events[0].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Committed ||
      frame->events[1].kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Retired ||
      runtime.hasPendingActions())
    fail("equal owner event split commit from whole-envelope release");
}

void causalReleaseConjoinsWithIntrinsicRelease() {
  const fabric::ResourceContract contract = createContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}};
  const auto resources = take(
      loom::sim::detail::freezeCgraResourceRuntimePlan(contracts, selections));
  const loom::sim::detail::CgraPhysicalUseTiming uses[] = {
      {0, 0, 1, 2, 0, 2, 1, true}};
  auto runtime = take(
      loom::sim::detail::CgraPhysicalActionRuntime::create(resources, uses));
  (void)take(runtime.request(0, 4, coordinate(5)));

  auto frame = take(runtime.advance());
  if (!frame || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Granted)
    fail("causally released action did not acquire");
  frame = take(runtime.advance());
  if (!frame || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Committed)
    fail("causally released action did not commit");
  frame = take(runtime.advance());
  if (!frame || !frame->events.empty() || !runtime.hasPendingActions() ||
      runtime.nextCoordinate())
    fail("intrinsic release bypassed the causal release condition");

  if (llvm::Error error = runtime.satisfyCausalRelease(0, 4, coordinate(9)))
    fail(llvm::toString(std::move(error)));
  frame = take(runtime.advance());
  if (!frame ||
      frame->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(9, 1)) ||
      frame->coordinate.delta != 1 || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Retired ||
      runtime.hasPendingActions())
    fail("causal release did not retire the complete claim envelope");
}

void equalOwnerEventStillWaitsForCausalRelease() {
  const fabric::ResourceContract contract = createAtomicContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}};
  const auto resources = take(
      loom::sim::detail::freezeCgraResourceRuntimePlan(contracts, selections));
  const loom::sim::detail::CgraPhysicalUseTiming uses[] = {
      {0, 0, 0, 0, 0, 0, 0, true}};
  auto runtime = take(
      loom::sim::detail::CgraPhysicalActionRuntime::create(resources, uses));
  (void)take(runtime.request(0, 5, coordinate(11)));

  auto frame = take(runtime.advance());
  if (!frame || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Granted)
    fail("combined causal owner event did not acquire");
  frame = take(runtime.advance());
  if (!frame || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Committed ||
      !runtime.hasPendingActions())
    fail("combined owner event released before its causal condition");

  if (llvm::Error error = runtime.satisfyCausalRelease(0, 5, coordinate(12)))
    fail(llvm::toString(std::move(error)));
  frame = take(runtime.advance());
  if (!frame || frame->coordinate.delta != 1 || frame->events.size() != 1 ||
      frame->events.front().kind !=
          loom::sim::detail::CgraPhysicalLifecycleKind::Retired ||
      runtime.hasPendingActions())
    fail("combined owner event did not retire after causal release");
}

} // namespace

int main() {
  contentionProducesExactLifecycleAndStall();
  equalOwnerEventCommitsAndReleasesAtomically();
  causalReleaseConjoinsWithIntrinsicRelease();
  equalOwnerEventStillWaitsForCausalRelease();
  return EXIT_SUCCESS;
}
