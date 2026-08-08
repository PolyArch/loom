#include "Fabric/IR/OperationResourceContract.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "operation resource contract test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

void oneCycleElasticContractOwnsOnePublishedResultSlot() {
  using namespace fabric;
  const ResourceContract &contract =
      oneCycleElasticOperationResourceContract();

  require(contract.stateCount() == 1,
          "one-cycle operation lost its result-slot state");
  const auto dimensions = contract.capacityDimensions(StateKey(0));
  require(dimensions.size() == 1 &&
              dimensions.front().capacity == CapacityUnits(1) &&
              dimensions.front().initialOccupancy == CapacityUnits(0),
          "one-cycle operation result slot is not empty capacity one");
  require(contract.resourceTransitionCount() == 1,
          "one-cycle operation lost its Publish transition");
  require(contract.requesterCount() == 1 &&
              contract.eligibilityCount() == 1 && contract.eventCount() == 3 &&
              contract.timingContractCount() == 1 &&
              contract.usePatternCount() == 1,
          "one-cycle operation owner domains changed");

  const auto ranks = contract.eventOrder(TimingContractKey(0));
  require(ranks == llvm::ArrayRef<std::uint32_t>({0, 1, 1}),
          "Accept, Publish, and Release do not have ranks {0, 1, 1}");

  const UsePattern active = contract.usePattern(UsePatternKey(0));
  require(active.requester == RequesterKey(0) &&
              active.eligibility == EligibilityKey(0) &&
              active.acquire == EventKey(0) && active.release == EventKey(2) &&
              active.commit && active.commit->event == EventKey(1) &&
              active.commit->transition == ResourceTransitionKey(0) &&
              active.timingAndProgress == TimingContractKey(0) &&
              active.internalTransactionCount == 0 &&
              active.parameters.empty() && active.sharingAssignments.empty(),
          "active operation use does not own Accept, Publish, and Release");
  require(active.claims.size() == 1 &&
              active.claims.front().state == StateKey(0) &&
              active.claims.front().dimension == CapacityDimensionKey(0) &&
              active.claims.front().amount == CapacityUnits(1),
          "active operation use does not claim the complete result slot");
}

} // namespace

int main() {
  oneCycleElasticContractOwnsOnePublishedResultSlot();
  return EXIT_SUCCESS;
}
