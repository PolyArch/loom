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
  const ResourceContract &contract = oneCycleElasticOperationResourceContract();

  require(contract.stateCount() == 1,
          "one-cycle operation lost its result-slot state");
  const auto dimensions = contract.capacityDimensions(StateKey(0));
  require(dimensions.size() == 1 &&
              dimensions.front().capacity == CapacityUnits(1) &&
              dimensions.front().initialOccupancy == CapacityUnits(0),
          "one-cycle operation result slot is not empty capacity one");
  require(contract.resourceTransitionCount() == 1,
          "one-cycle operation lost its Publish transition");
  require(contract.requesterCount() == 1 && contract.eligibilityCount() == 1 &&
              contract.eventCount() == 3 &&
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

void activeResultHandoffIsOwnedByExactContracts() {
  using namespace fabric;

  auto check = [](const ResourceContract &contract, bool expected,
                  llvm::StringRef message) {
    auto actual = requiresActiveResultHandoff(contract);
    if (!actual)
      fail(llvm::toString(actual.takeError()));
    require(*actual == expected, message);
  };

  check(oneCycleElasticOperationResourceContract(), true,
        "one-cycle result slot lost its active-result handoff");
  check(loopStreamOperationResourceContract(), true,
        "registered LoopStream lost its active-result handoff");
  check(loopCarryOperationResourceContract(), false,
        "transparent LoopCarry acquired a result-holding handoff");
  check(loopInvariantOperationResourceContract(), false,
        "transparent LoopInvariant acquired a result-holding handoff");
  check(loopGateOperationResourceContract(), false,
        "transparent LoopGate acquired a result-holding handoff");
}

void orderedCardinalityContractOwnsOneClaimEnvelope() {
  using namespace fabric;
  using Schema = dataflow::OperationSchemaId;

  const ResourceContract parallelize =
      llvm::cantFail(createOrderedCardinalityOperationResourceContract(
          Schema::DataflowParallelize, 8));
  require(parallelize.stateCount() == 2 &&
              parallelize.resourceTransitionCount() == 4 &&
              parallelize.requesterCount() == 1 &&
              parallelize.eligibilityCount() == 4 &&
              parallelize.eventCount() == 3 &&
              parallelize.timingContractCount() == 1 &&
              parallelize.usePatternCount() == 4,
          "parallelize ordered-cardinality owner domains changed");
  for (std::uint32_t state = 0; state != 2; ++state) {
    const auto dimensions = parallelize.capacityDimensions(StateKey(state));
    require(dimensions.size() == 1 &&
                dimensions.front().capacity == CapacityUnits(1) &&
                dimensions.front().initialOccupancy == CapacityUnits(0),
            "ordered-cardinality state is not empty capacity one");
  }
  require(parallelize.eventOrder(TimingContractKey(0)) ==
              llvm::ArrayRef<std::uint32_t>({0, 1, 1}),
          "ordered-cardinality event ranks changed");
  constexpr std::uint32_t parallelTransactions[] = {0, 1, 1, 2};
  for (std::uint32_t ordinal = 0; ordinal != 4; ++ordinal) {
    const UsePattern pattern = parallelize.usePattern(UsePatternKey(ordinal));
    require(
        pattern.requester == RequesterKey(0) &&
            pattern.eligibility == EligibilityKey(ordinal) &&
            pattern.acquire == EventKey(0) && pattern.release == EventKey(2) &&
            pattern.commit && pattern.commit->event == EventKey(1) &&
            pattern.commit->transition == ResourceTransitionKey(ordinal) &&
            pattern.timingAndProgress == TimingContractKey(0) &&
            pattern.internalTransactionCount == parallelTransactions[ordinal] &&
            pattern.parameters.empty() && pattern.sharingAssignments.empty(),
        "parallelize use pattern lost its exact claim lifetime");
    require(pattern.claims.size() ==
                (parallelTransactions[ordinal] == 0 ? 1U : 2U),
            "parallelize use pattern has the wrong claim envelope");
    for (std::uint32_t transaction = 0;
         transaction != pattern.internalTransactionCount; ++transaction)
      require(parallelize.internalTransaction(UsePatternKey(ordinal),
                                              transaction) ==
                  llvm::ArrayRef<ClaimKey>({ClaimKey(1)}),
              "parallelize transaction does not select the group slot");
  }

  const ResourceContract serialize =
      llvm::cantFail(createOrderedCardinalityOperationResourceContract(
          Schema::DataflowSerialize, 7));
  require(serialize.usePatternCount() == 2 &&
              serialize.usePattern(UsePatternKey(0)).internalTransactionCount ==
                  7 &&
              serialize.usePattern(UsePatternKey(1)).internalTransactionCount ==
                  1,
          "serialize transaction inventory is not M/1");

  auto exact = isOrderedCardinalityOperationResourceContract(
      serialize, Schema::DataflowSerialize, 7);
  require(exact && *exact, "exact serialize contract was not recognized");
  auto wrongLaneCount = isOrderedCardinalityOperationResourceContract(
      serialize, Schema::DataflowSerialize, 6);
  require(wrongLaneCount && !*wrongLaneCount,
          "serialize contract ignored maximum lane count");
  auto legacy = isOrderedCardinalityOperationResourceContract(
      oneCycleElasticOperationResourceContract(), Schema::DataflowSerialize, 7);
  require(legacy && !*legacy,
          "legacy one-cycle contract was accepted as ordered-cardinality");
}

} // namespace

int main() {
  oneCycleElasticContractOwnsOnePublishedResultSlot();
  activeResultHandoffIsOwnedByExactContracts();
  orderedCardinalityContractOwnsOneClaimEnvelope();
  return EXIT_SUCCESS;
}
