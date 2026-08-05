#include "CGRAResourceRuntime.h"

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
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

  const loom::sim::detail::CgraResourceRequest blockedRequest{0, 0};
  const std::size_t reusableCapacity = grants.capacity();
  if (llvm::Error error = runtime.grant({blockedRequest}, grants))
    fail(llvm::toString(std::move(error)));
  if (!grants.empty())
    fail("an unavailable claim envelope was partially granted");
  if (grants.capacity() != reusableCapacity)
    fail("grant discarded caller-owned result storage");

  if (llvm::Error error = runtime.release(first.front().claimEnvelope))
    fail(llvm::toString(std::move(error)));
  if (runtime.occupancy(0) != 0 || runtime.occupancy(1) != 0)
    fail("release did not return the complete claim envelope");

  if (llvm::Error error = runtime.grant({blockedRequest}, grants))
    fail(llvm::toString(std::move(error)));
  const auto second = grants;
  if (second.size() != 1 || second.front().selectedUseOrdinal != 0)
    fail("round-robin did not advance after the successful grant");
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

} // namespace

int main() {
  atomicClaimsAndRoundRobinAreExecutedExactly();
  derivedActivationAcquiresSharedClaimsOnce();
  return EXIT_SUCCESS;
}
