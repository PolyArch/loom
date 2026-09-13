#include "CGRAResourceRuntime.h"

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA resource runtime test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

fabric::ResourceContract createContendedContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {
      {StateKey(0),
       {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)},
        {CapacityDimensionKey(1), CapacityUnits(1), CapacityUnits(0)}}}};
  declaration.timingContracts = {{TimingContractKey(0), {0, 1}}};
  declaration.requesters = {RequesterKey(0), RequesterKey(1)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  for (std::uint32_t requester = 0; requester != 2; ++requester)
    declaration.usePatterns.push_back(
        {UsePatternKey(requester),
         RequesterKey(requester),
         EligibilityKey(0),
         EventKey(0),
         EventKey(1),
         std::nullopt,
         TimingContractKey(0),
         {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)},
          {ClaimKey(1), StateKey(0), CapacityDimensionKey(1),
           CapacityUnits(1)}},
         {}});
  declaration.grantPolicy = RoundRobinDeclaration{
      {RequesterKey(0), RequesterKey(1)}, RequesterKey(1)};
  return take(ResourceContract::create(declaration));
}

fabric::ResourceContract createBroadcastContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {
      {StateKey(0),
       {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)},
        {CapacityDimensionKey(1), CapacityUnits(1), CapacityUnits(0)},
        {CapacityDimensionKey(2), CapacityUnits(1), CapacityUnits(0)}}}};
  declaration.timingContracts = {{TimingContractKey(0), {0, 1}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.usePatterns = {
      {UsePatternKey(0),
       RequesterKey(0),
       EligibilityKey(0),
       EventKey(0),
       EventKey(1),
       std::nullopt,
       TimingContractKey(0),
       {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)},
        {ClaimKey(1), StateKey(0), CapacityDimensionKey(1), CapacityUnits(1)}},
       {}},
      {UsePatternKey(1),
       RequesterKey(0),
       EligibilityKey(1),
       EventKey(0),
       EventKey(1),
       std::nullopt,
       TimingContractKey(0),
       {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)},
        {ClaimKey(1), StateKey(0), CapacityDimensionKey(2), CapacityUnits(1)}},
       {}}};
  return take(ResourceContract::create(declaration));
}

void atomicClaimsAndRoundRobinAreExecutedExactly() {
  const fabric::ResourceContract contract = createContendedContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}, {0, fabric::UsePatternKey(1)}};
  const auto plan = take(
      loom::sim::detail::freezeCgraResourceRuntimePlan(contracts, selections));
  auto runtime = take(loom::sim::detail::CgraResourceRuntime::create(plan));

  const loom::sim::detail::CgraResourceRequest firstRequests[] = {{0, 0},
                                                                  {1, 0}};
  llvm::SmallVector<loom::sim::detail::CgraResourceGrant, 4> grants;
  if (llvm::Error error = runtime.grant(firstRequests, grants))
    fail(llvm::toString(std::move(error)));
  const auto first = grants;
  if (first.size() != 1 || first.front().selectedUseOrdinal != 1 ||
      runtime.occupancy(0) != 1 || runtime.occupancy(1) != 1)
    fail("round-robin reset or atomic claim envelope changed");

  const auto blockers = runtime.capacityBlockers(0);
  if (blockers.size() != 2)
    fail("blocked dimensions lost their exact envelope holder");
  for (const auto &blocker : blockers)
    if (blocker.holder.slot != first.front().claimEnvelope.slot ||
        blocker.holder.generation != first.front().claimEnvelope.generation ||
        blocker.capacity != 1 || blocker.occupancy != 1 ||
        blocker.requestedAmount != 1 || blocker.heldAmount != 1)
      fail("capacity blocker does not quote the actual live claim");

  const loom::sim::detail::CgraResourceRequest blockedRequest{0, 0};
  if (llvm::Error error = runtime.grant({blockedRequest}, grants))
    fail(llvm::toString(std::move(error)));
  if (!grants.empty())
    fail("an unavailable claim envelope was partially granted");

  if (llvm::Error error = runtime.release(first.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));
  if (!runtime.capacityBlockers(0).empty())
    fail("released envelope remained a capacity blocker");
  if (runtime.occupancy(0) != 0 || runtime.occupancy(1) != 0)
    fail("release did not return the complete claim envelope");

  if (llvm::Error error = runtime.grant({blockedRequest}, grants))
    fail(llvm::toString(std::move(error)));
  const auto second = grants;
  if (second.size() != 1 || second.front().selectedUseOrdinal != 0)
    fail("round-robin did not advance after the successful grant");
  if (llvm::Error error = runtime.release(second.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));

  // The previous coordinate had one requester, and a work-conserving cursor
  // stays with it rather than stepping onto an idle successor.
  const loom::sim::detail::CgraResourceRequest nextRequests[] = {{0, 1},
                                                                 {1, 1}};
  if (llvm::Error error = runtime.grant(nextRequests, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 0)
    fail("a lone grant moved the round-robin cursor off its requester");
  if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));

  const loom::sim::detail::CgraResourceRequest pairedRequests[] = {{0, 2},
                                                                   {1, 2}};
  if (llvm::Error error = runtime.grant(pairedRequests, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 1)
    fail("round-robin did not alternate once a second requester appeared");
}

/// Three requesters of one merge component, so the cursor's advance rule is
/// what decides throughput. Every Temporal PE input port sits behind such a
/// component, and a cursor that stepped onto an idle successor after each
/// grant would halve a lone stream.
void roundRobinArbitrationIsWorkConserving() {
  using namespace fabric;
  using namespace loom::sim::detail;
  const SwitchResourceContract switchContract =
      take(SwitchResourceContract::create(
          {Schedule::Temporal,
           3,
           1,
           {{0, 1, 2}},
           TemporalSwitchGrantPolicy(TemporalSwitchRoundRobin{{0, 1, 2}, 0})}));
  const ResourceContract *contracts[] = {&switchContract.resourceContract()};
  const CgraResourcePatternSelection selections[] = {
      {0, take(switchContract.traversalPattern(0, 0))},
      {0, take(switchContract.traversalPattern(1, 0))},
      {0, take(switchContract.traversalPattern(2, 0))}};
  const auto plan = take(freezeCgraResourceRuntimePlan(contracts, selections));

  llvm::SmallVector<CgraResourceGrant, 4> grants;
  const auto oneCoordinate = [&](CgraResourceRuntime &runtime,
                                 llvm::ArrayRef<CgraResourceRequest> requests) {
    if (llvm::Error error = runtime.grant(requests, grants))
      fail(llvm::toString(std::move(error)));
    if (grants.size() > 1)
      fail("an arbitration component granted more than one requester");
    if (grants.empty())
      return std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t granted = grants.front().selectedUseOrdinal;
    if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
      fail(llvm::toString(std::move(error)));
    return granted;
  };

  // A lone requester that requests at every coordinate is granted at every
  // coordinate, after the one coordinate the cursor needs to reach it.
  auto lone = take(CgraResourceRuntime::create(plan));
  const CgraResourceRequest onlyLast[] = {{2, 0}};
  if (oneCoordinate(lone, onlyLast) != std::numeric_limits<std::uint64_t>::max())
    fail("the cursor reached a fresh requester without its one coordinate");
  for (std::uint64_t coordinate = 0; coordinate != 4; ++coordinate)
    if (oneCoordinate(lone, onlyLast) != 2)
      fail("a lone continuous requester was not granted every coordinate");

  // Two continuous requesters alternate from the first grant.
  auto paired = take(CgraResourceRuntime::create(plan));
  const CgraResourceRequest firstTwo[] = {{0, 0}, {1, 0}};
  for (std::uint64_t coordinate = 0; coordinate != 6; ++coordinate)
    if (oneCoordinate(paired, firstTwo) != coordinate % 2)
      fail("two continuous requesters did not alternate");
}

void derivedActivationAcquiresSharedClaimsOnce() {
  const fabric::ResourceContract contract = createBroadcastContract();
  const fabric::ResourceContract *contracts[] = {&contract};
  const loom::sim::detail::CgraResourcePatternSelection selections[] = {
      {0, fabric::UsePatternKey(0)}, {0, fabric::UsePatternKey(1)}};
  const loom::sim::detail::CgraResourceActivationSelection activations[] = {
      {0, 2}};
  const auto plan = take(loom::sim::detail::freezeCgraResourceRuntimePlan(
      contracts, selections, activations));
  if (plan.selectedUses.size() != 1 ||
      plan.selectedUses.front().claimCount != 3)
    fail("derived activation did not union its exact claim envelope");

  auto runtime = take(loom::sim::detail::CgraResourceRuntime::create(plan));
  llvm::SmallVector<loom::sim::detail::CgraResourceGrant, 4> grants;
  if (llvm::Error error = runtime.grant({{0, 0}}, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || runtime.occupancy(0) != 1 ||
      runtime.occupancy(1) != 1 || runtime.occupancy(2) != 1)
    fail("derived activation was partially acquired");
  if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));
  if (runtime.occupancy(0) != 0 || runtime.occupancy(1) != 0 ||
      runtime.occupancy(2) != 0)
    fail("derived activation did not release one whole envelope");
}

void fixedPriorityDecidesOverRegisteredRequests() {
  using namespace fabric;
  using namespace loom::sim::detail;
  const SwitchResourceContract switchContract =
      take(SwitchResourceContract::create(
          {Schedule::Temporal,
           3,
           2,
           {{0, 1, 2}, {0, 1, 2}},
           TemporalSwitchGrantPolicy(TemporalSwitchFixedPriority{{1, 0, 2}})}));
  const ResourceContract *contracts[] = {&switchContract.resourceContract()};
  const CgraResourcePatternSelection selections[] = {
      {0, take(switchContract.traversalPattern(0, 0))},
      {0, take(switchContract.traversalPattern(1, 1))}};
  const auto plan = take(freezeCgraResourceRuntimePlan(contracts, selections));
  auto runtime = take(CgraResourceRuntime::create(plan));

  // One pointer names one requester, so output-disjoint requesters no longer
  // share a coordinate even when both claim envelopes are available.
  llvm::SmallVector<CgraResourceGrant, 2> grants;
  const CgraResourceRequest disjoint[] = {{0, 0}, {1, 0}};
  if (llvm::Error error = runtime.grant(disjoint, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 1)
    fail("fixed priority granted more than its highest-priority requester");
  if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));

  // The pointer entering a coordinate was chosen from the previous
  // coordinate's requests, so the lower-priority requester is reached one
  // coordinate after the higher-priority one falls silent.
  const CgraResourceRequest lower[] = {{0, 0}};
  if (llvm::Error error = runtime.grant(lower, grants))
    fail(llvm::toString(std::move(error)));
  if (!grants.empty())
    fail("fixed priority decided over the current coordinate's requests");
  if (llvm::Error error = runtime.grant(lower, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 0)
    fail("fixed priority never reached its lower-priority requester");
}

void roundRobinAdvancesToTheNextRequesterAfterEveryGrant() {
  using namespace fabric;
  using namespace loom::sim::detail;
  const SwitchResourceContract switchContract =
      take(SwitchResourceContract::create(
          {Schedule::Temporal,
           3,
           2,
           {{0, 1, 2}, {0, 1, 2}},
           TemporalSwitchGrantPolicy(TemporalSwitchRoundRobin{{0, 1, 2}, 0})}));
  const ResourceContract *contracts[] = {&switchContract.resourceContract()};
  const CgraResourcePatternSelection selections[] = {
      {0, take(switchContract.traversalPattern(0, 0))},
      {0, take(switchContract.traversalPattern(1, 1))},
      {0, take(switchContract.traversalPattern(1, 0))},
      {0, take(switchContract.traversalPattern(2, 0))}};
  const auto plan = take(freezeCgraResourceRuntimePlan(contracts, selections));
  auto runtime = take(CgraResourceRuntime::create(plan));

  llvm::SmallVector<CgraResourceGrant, 4> grants;
  const CgraResourceRequest disjoint[] = {{0, 0}, {1, 0}};
  if (llvm::Error error = runtime.grant(disjoint, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 0)
    fail("round-robin granted more than the requester its cursor named");
  if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));

  const CgraResourceRequest contended[] = {{2, 0}, {3, 0}};
  if (llvm::Error error = runtime.grant(contended, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 2)
    fail("round-robin cursor did not advance to its next requester");
  if (llvm::Error error = runtime.release(grants.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));

  if (llvm::Error error = runtime.grant(contended, grants))
    fail(llvm::toString(std::move(error)));
  if (grants.size() != 1 || grants.front().selectedUseOrdinal != 3)
    fail("round-robin cursor did not reach its last component requester");
}

} // namespace

int main() {
  atomicClaimsAndRoundRobinAreExecutedExactly();
  derivedActivationAcquiresSharedClaimsOnce();
  fixedPriorityDecidesOverRegisteredRequests();
  roundRobinAdvancesToTheNextRequesterAfterEveryGrant();
  roundRobinArbitrationIsWorkConserving();
  return EXIT_SUCCESS;
}
