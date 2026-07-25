#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace fabric;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

ResourceContract takeContract(const char *test,
                              llvm::Expected<ResourceContract> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return std::move(*result);
}

void expectViolation(const char *test, const std::string &label,
                     llvm::Expected<ResourceContract> result,
                     ResourceContractViolation expected) {
  if (result)
    fail(test, label + ": expected a rejected declaration");

  std::optional<ResourceContractViolation> observed;
  llvm::handleAllErrors(result.takeError(),
                        [&](const ResourceContractError &error) {
                          observed = error.violation();
                        });
  if (!observed)
    fail(test, label + ": received a different error category");
  if (*observed != expected)
    fail(test, label + ": observed violation " +
                   getResourceContractViolationName(*observed).str() +
                   ", expected " +
                   getResourceContractViolationName(expected).str());
}

template <typename Key, typename Enum> constexpr Key key(Enum value) {
  return Key(static_cast<std::uint32_t>(value));
}

// One buffering resource whose two requesters claim disjoint capacity, so the
// verifier proves contention impossible and no grant policy exists.
namespace buffer {

enum class State : std::uint32_t { Queue };
enum class Dimension : std::uint32_t { StoredEntry, DequeuePort };
enum class Requester : std::uint32_t { Input, Output };
enum class Eligibility : std::uint32_t { InputTokenReady, OutputTokenReady };
enum class Event : std::uint32_t { Accept, Retire };
enum class Timing : std::uint32_t { OneCycleElastic };

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceState{{
      CapacityDimension{CapacityUnits(4), CapacityUnits(0)},
      CapacityDimension{CapacityUnits(1), CapacityUnits(0)},
  }}};
  declaration.requesterCount = 2;
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.timingContractCount = 1;
  declaration.usePatterns = {
      UsePattern{key<RequesterKey>(Requester::Input),
                 key<EligibilityKey>(Eligibility::InputTokenReady),
                 key<EventKey>(Event::Accept),
                 key<TimingContractKey>(Timing::OneCycleElastic),
                 {Claim{key<StateKey>(State::Queue),
                        key<CapacityDimensionKey>(Dimension::StoredEntry),
                        CapacityUnits(1), key<EventKey>(Event::Retire)}},
                 {}},
      UsePattern{key<RequesterKey>(Requester::Output),
                 key<EligibilityKey>(Eligibility::OutputTokenReady),
                 key<EventKey>(Event::Accept),
                 key<TimingContractKey>(Timing::OneCycleElastic),
                 {Claim{key<StateKey>(State::Queue),
                        key<CapacityDimensionKey>(Dimension::DequeuePort),
                        CapacityUnits(1), key<EventKey>(Event::Retire)}},
                 {}},
  };
  return declaration;
}

} // namespace buffer

// One output port contended by three transfer requesters, so an exact grant
// policy is part of the contract.
namespace crossbar {

enum class State : std::uint32_t { Egress };
enum class Dimension : std::uint32_t { TransferSlot };
enum class Requester : std::uint32_t { Ingress0, Ingress1, Ingress2 };
enum class Eligibility : std::uint32_t { RouteSelected };
enum class Event : std::uint32_t { Accept, Retire };
enum class Timing : std::uint32_t { SingleBeat };

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceState{{
      CapacityDimension{CapacityUnits(1), CapacityUnits(0)},
  }}};
  declaration.requesterCount = 3;
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContractCount = 1;
  for (std::uint32_t requester = 0; requester < 3; ++requester)
    declaration.usePatterns.push_back(
        UsePattern{RequesterKey(requester),
                   key<EligibilityKey>(Eligibility::RouteSelected),
                   key<EventKey>(Event::Accept),
                   key<TimingContractKey>(Timing::SingleBeat),
                   {Claim{key<StateKey>(State::Egress),
                          key<CapacityDimensionKey>(Dimension::TransferSlot),
                          CapacityUnits(1), key<EventKey>(Event::Retire)}},
                   {}});
  return declaration;
}

} // namespace crossbar

// One memory operation whose accepted use is realized by two ordered internal
// service beats inside the same external firing and retirement.
namespace memoryEngine {

enum class State : std::uint32_t { ServicePort, Bank };
enum class ServicePortDimension : std::uint32_t { Beat };
enum class BankDimension : std::uint32_t { Access };
enum class Requester : std::uint32_t { OperationRow };
enum class Eligibility : std::uint32_t { OperandTupleComplete };
enum class Event : std::uint32_t { Accept, Retire };
enum class Timing : std::uint32_t { TwoBeatVectorLoad };

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration declaration;
  declaration.states = {
      ResourceState{{CapacityDimension{CapacityUnits(1), CapacityUnits(0)}}},
      ResourceState{{CapacityDimension{CapacityUnits(2), CapacityUnits(0)}}},
  };
  declaration.requesterCount = 1;
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContractCount = 1;
  declaration.usePatterns = {
      UsePattern{key<RequesterKey>(Requester::OperationRow),
                 key<EligibilityKey>(Eligibility::OperandTupleComplete),
                 key<EventKey>(Event::Accept),
                 key<TimingContractKey>(Timing::TwoBeatVectorLoad),
                 {Claim{key<StateKey>(State::ServicePort),
                        key<CapacityDimensionKey>(ServicePortDimension::Beat),
                        CapacityUnits(1), key<EventKey>(Event::Retire)},
                  Claim{key<StateKey>(State::Bank),
                        key<CapacityDimensionKey>(BankDimension::Access),
                        CapacityUnits(2), key<EventKey>(Event::Retire)}},
                 {InternalTransaction{{ClaimKey(0)}},
                  InternalTransaction{{ClaimKey(0), ClaimKey(1)}}}}};
  return declaration;
}

} // namespace memoryEngine

void disjointResourceNeedsNoGrantPolicy() {
  const ResourceContract contract =
      takeContract(__func__, ResourceContract::create(buffer::declaration()));

  require(__func__, contract.stateCount() == 1, "declared state count differs");
  require(__func__,
          contract.state(key<StateKey>(buffer::State::Queue))
                  .capacityDimensions.size() == 2,
          "declared capacity dimension count differs");
  require(__func__,
          contract.state(key<StateKey>(buffer::State::Queue))
                  .capacityDimensions[0]
                  .initialOccupancy == CapacityUnits(0),
          "canonical initial occupancy differs");
  require(__func__, contract.usePatternCount() == 2,
          "declared use pattern count differs");
  require(__func__,
          contract.requesterCount() == 2 && contract.eligibilityCount() == 2 &&
              contract.eventCount() == 2 && contract.timingContractCount() == 1,
          "closed owner key domains differ");
  require(__func__, !contract.grantPolicy().has_value(),
          "disjoint requesters must not carry a grant policy");

  const UsePattern &enqueue = contract.usePattern(UsePatternKey(0));
  require(__func__,
          enqueue.requester == key<RequesterKey>(buffer::Requester::Input) &&
              enqueue.claims.size() == 1 &&
              enqueue.claims[0].amount == CapacityUnits(1) &&
              enqueue.claims[0].release == key<EventKey>(buffer::Event::Retire),
          "atomic use pattern content differs");
}

void sharedCapacityWithoutPolicyIsRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[1].claims[0].dimension =
      key<CapacityDimensionKey>(buffer::Dimension::StoredEntry);
  expectViolation(__func__, "two requesters on one capacity dimension",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::ContentionWithoutGrantPolicy);
}

void unknownAndUndeclaredKeysAreRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[0].claims[0].state = StateKey(1);
  expectViolation(__func__, "claim on an undeclared state",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownStateKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].claims[0].dimension = CapacityDimensionKey(2);
  expectViolation(__func__, "claim on an undeclared capacity dimension",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UndeclaredClaim);

  declaration = buffer::declaration();
  declaration.usePatterns[0].requester = RequesterKey(2);
  expectViolation(__func__, "use pattern with a foreign requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].eligibility = EligibilityKey(2);
  expectViolation(__func__, "use pattern with an undeclared eligibility",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEligibilityKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].claims[0].release = EventKey(2);
  expectViolation(__func__, "claim released by an undeclared event",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].timingAndProgress = TimingContractKey(1);
  expectViolation(__func__, "use pattern with an undeclared timing contract",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownTimingContractKey);
}

void capacityAndClaimOverflowAreRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.states[0].capacityDimensions[0].initialOccupancy =
      CapacityUnits(5);
  expectViolation(__func__, "initial occupancy above capacity",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::InitialOccupancyExceedsCapacity);

  declaration = buffer::declaration();
  declaration.usePatterns[0].claims[0].amount = CapacityUnits(5);
  expectViolation(__func__, "claim above capacity",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::ClaimExceedsCapacity);

  declaration = buffer::declaration();
  declaration.states[0].capacityDimensions[0].initialOccupancy =
      CapacityUnits(1);
  declaration.usePatterns[0].claims[0].amount = CapacityUnits(0xffffffffu);
  expectViolation(__func__, "claim whose occupancy sum is not representable",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::ClaimExceedsCapacity);
}

void splitClaimOnOneDimensionIsRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[0].claims.push_back(
      declaration.usePatterns[0].claims[0]);
  expectViolation(__func__, "one dimension claimed twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateClaim);

  declaration = buffer::declaration();
  Claim staged = declaration.usePatterns[0].claims[0];
  staged.release = key<EventKey>(buffer::Event::Accept);
  declaration.usePatterns[0].claims.push_back(staged);
  expectViolation(__func__, "one dimension released at two events",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::AmbiguousRelease);
}

void violationPrecedenceIsIndependentOfDeclarationOrder() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[0].acquire = EventKey(2);
  declaration.usePatterns[1].claims.push_back(
      declaration.usePatterns[1].claims[0]);
  expectViolation(__func__, "unknown key before duplicate claim",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);

  std::swap(declaration.usePatterns[0], declaration.usePatterns[1]);
  expectViolation(__func__, "duplicate claim before unknown key",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);
}

void fixedPriorityGrantsFirstEligibleRequester() {
  ResourceContractDeclaration declaration = crossbar::declaration();
  declaration.grantPolicy = GrantPolicy(
      FixedPriority{{key<RequesterKey>(crossbar::Requester::Ingress2),
                     key<RequesterKey>(crossbar::Requester::Ingress0),
                     key<RequesterKey>(crossbar::Requester::Ingress1)}});
  const ResourceContract contract =
      takeContract(__func__, ResourceContract::create(declaration));
  require(__func__, contract.grantPolicy().has_value(),
          "contended capacity must carry a grant policy");

  const GrantPolicy &policy = *contract.grantPolicy();
  const RequesterKey origin = resetGrantCursor(policy);
  require(__func__, origin == key<RequesterKey>(crossbar::Requester::Ingress2),
          "fixed priority scans from the front of its permutation");

  const bool lowPriorityPair[] = {true, true, false};
  const GrantDecision low = arbitrate(policy, origin, lowPriorityPair);
  require(__func__,
          low.granted == key<RequesterKey>(crossbar::Requester::Ingress0),
          "fixed priority must grant the first eligible requester in order");
  require(__func__, low.nextCursor == origin,
          "fixed priority must keep its scan origin");

  const bool highPriorityPair[] = {false, true, true};
  const GrantDecision high =
      arbitrate(policy, low.nextCursor, highPriorityPair);
  require(__func__,
          high.granted == key<RequesterKey>(crossbar::Requester::Ingress2),
          "fixed priority must prefer the earlier permutation entry");

  const bool noneEligible[] = {false, false, false};
  const GrantDecision idle = arbitrate(policy, origin, noneEligible);
  require(__func__, !idle.granted.has_value() && idle.nextCursor == origin,
          "an ineligible cycle must not grant or move the cursor");
}

void grantPolicyMustBeAnExactRequesterPermutation() {
  ResourceContractDeclaration declaration = crossbar::declaration();
  declaration.grantPolicy = GrantPolicy(
      FixedPriority{{key<RequesterKey>(crossbar::Requester::Ingress0),
                     key<RequesterKey>(crossbar::Requester::Ingress1)}});
  expectViolation(__func__, "policy omitting a declared requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::RequesterOmittedFromGrantPolicy);

  declaration.grantPolicy = GrantPolicy(
      FixedPriority{{key<RequesterKey>(crossbar::Requester::Ingress0),
                     key<RequesterKey>(crossbar::Requester::Ingress0),
                     key<RequesterKey>(crossbar::Requester::Ingress1)}});
  expectViolation(__func__, "policy repeating a requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateRequesterInGrantPolicy);

  declaration.grantPolicy = GrantPolicy(FixedPriority{
      {key<RequesterKey>(crossbar::Requester::Ingress0),
       key<RequesterKey>(crossbar::Requester::Ingress1), RequesterKey(3)}});
  expectViolation(__func__, "policy naming a foreign requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);
}

void roundRobinAdvancesOnlyOnASuccessfulGrant() {
  ResourceContractDeclaration declaration = crossbar::declaration();
  declaration.grantPolicy =
      GrantPolicy(RoundRobin{{key<RequesterKey>(crossbar::Requester::Ingress0),
                              key<RequesterKey>(crossbar::Requester::Ingress1),
                              key<RequesterKey>(crossbar::Requester::Ingress2)},
                             key<RequesterKey>(crossbar::Requester::Ingress1)});
  const ResourceContract contract =
      takeContract(__func__, ResourceContract::create(declaration));
  const GrantPolicy &policy = *contract.grantPolicy();

  const RequesterKey reset = resetGrantCursor(policy);
  require(__func__, reset == key<RequesterKey>(crossbar::Requester::Ingress1),
          "reset must establish the declared cursor");

  const bool noneEligible[] = {false, false, false};
  const GrantDecision idle = arbitrate(policy, reset, noneEligible);
  require(__func__, !idle.granted.has_value() && idle.nextCursor == reset,
          "a cycle without a grant must preserve the cursor");

  const bool wrappedEligible[] = {true, false, false};
  const GrantDecision wrapped = arbitrate(policy, reset, wrappedEligible);
  require(__func__,
          wrapped.granted == key<RequesterKey>(crossbar::Requester::Ingress0),
          "round robin must scan its exact cycle from the cursor");
  require(__func__,
          wrapped.nextCursor ==
              key<RequesterKey>(crossbar::Requester::Ingress1),
          "a successful grant must advance past the granted requester");

  const bool cursorEligible[] = {true, true, false};
  const GrantDecision atCursor =
      arbitrate(policy, wrapped.nextCursor, cursorEligible);
  require(__func__,
          atCursor.granted == key<RequesterKey>(crossbar::Requester::Ingress1),
          "the cursor entry is scanned first");
  require(__func__,
          atCursor.nextCursor ==
              key<RequesterKey>(crossbar::Requester::Ingress2),
          "the cursor advances to the successor of the granted requester");

  declaration.grantPolicy =
      GrantPolicy(RoundRobin{{key<RequesterKey>(crossbar::Requester::Ingress0),
                              key<RequesterKey>(crossbar::Requester::Ingress1),
                              key<RequesterKey>(crossbar::Requester::Ingress2)},
                             RequesterKey(3)});
  expectViolation(__func__, "reset cursor outside the requester domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);
}

void internalTransactionsRefineOneAcceptedUse() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(memoryEngine::declaration()));

  const UsePattern &operation = contract.usePattern(UsePatternKey(0));
  require(__func__,
          operation.requester ==
                  key<RequesterKey>(memoryEngine::Requester::OperationRow) &&
              operation.acquire == key<EventKey>(memoryEngine::Event::Accept) &&
              operation.claims.size() == 2,
          "the external firing and claim envelope differ");
  require(__func__, operation.internalTransactions.size() == 2,
          "internal transaction count differs");
  require(__func__,
          operation.internalTransactions[0].claims.size() == 1 &&
              operation.internalTransactions[0].claims[0] == ClaimKey(0),
          "the first internal transaction differs");
  require(__func__,
          operation.internalTransactions[1].claims.size() == 2 &&
              operation.internalTransactions[1].claims[0] == ClaimKey(0) &&
              operation.internalTransactions[1].claims[1] == ClaimKey(1),
          "the second internal transaction differs");

  ResourceContractDeclaration declaration = memoryEngine::declaration();
  declaration.usePatterns[0].internalTransactions[0].claims[0] = ClaimKey(2);
  expectViolation(__func__, "internal transaction outside the claim envelope",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownClaimKey);

  declaration = memoryEngine::declaration();
  declaration.usePatterns[0].internalTransactions[1].claims[1] = ClaimKey(0);
  expectViolation(__func__, "internal transaction repeating one claim",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateClaim);
}

} // namespace

int main() {
  disjointResourceNeedsNoGrantPolicy();
  sharedCapacityWithoutPolicyIsRejected();
  unknownAndUndeclaredKeysAreRejected();
  capacityAndClaimOverflowAreRejected();
  splitClaimOnOneDimensionIsRejected();
  violationPrecedenceIsIndependentOfDeclarationOrder();
  fixedPriorityGrantsFirstEligibleRequester();
  grantPolicyMustBeAnExactRequesterPermutation();
  roundRobinAdvancesOnlyOnASuccessfulGrant();
  internalTransactionsRefineOneAcceptedUse();
  return 0;
}
