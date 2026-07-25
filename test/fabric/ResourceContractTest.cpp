#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
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

// One malformed inventory must report the same typed violation however its
// unordered entries were declared.
template <typename Reorder>
void expectStableViolation(const char *test, const std::string &label,
                           ResourceContractDeclaration declaration,
                           ResourceContractViolation expected,
                           Reorder reorder) {
  expectViolation(test, label + " as declared",
                  ResourceContract::create(declaration), expected);
  reorder(declaration);
  expectViolation(test, label + " reordered",
                  ResourceContract::create(declaration), expected);
}

template <typename Key, typename Enum> constexpr Key key(Enum value) {
  return Key(static_cast<std::uint32_t>(value));
}

template <typename T, typename = void>
struct HasPerClaimRelease : std::false_type {};

template <typename T>
struct HasPerClaimRelease<
    T, std::void_t<decltype(std::declval<T>().release)>> : std::true_type {};

static_assert(!HasPerClaimRelease<ClaimDeclaration>::value,
              "release belongs to the atomic use pattern, not each claim");

// Every fixture below declares its events in time order, so an event's own
// ordinal is the relative time its timing contract establishes.
template <typename Enum> constexpr std::uint32_t rank(Enum value) {
  return static_cast<std::uint32_t>(value);
}

bool sameClaim(const Claim &lhs, const Claim &rhs) {
  return lhs.state == rhs.state && lhs.dimension == rhs.dimension &&
         lhs.amount == rhs.amount;
}

// One buffering resource whose two requesters claim disjoint capacity, so the
// verifier proves contention impossible and no requester order exists.
namespace buffer {

enum class State : std::uint32_t { Queue };
enum class Dimension : std::uint32_t { StoredEntry, DequeuePort };
enum class Requester : std::uint32_t { Input, Output };
enum class Eligibility : std::uint32_t { InputTokenReady, OutputTokenReady };
enum class Event : std::uint32_t { Accept, Retire };
enum class Timing : std::uint32_t { OneCycleElastic };

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceStateDeclaration{
      key<StateKey>(State::Queue),
      {CapacityDimensionDeclaration{
           key<CapacityDimensionKey>(Dimension::StoredEntry), CapacityUnits(4),
           CapacityUnits(0)},
       CapacityDimensionDeclaration{
           key<CapacityDimensionKey>(Dimension::DequeuePort), CapacityUnits(1),
           CapacityUnits(0)}}}};
  declaration.requesters = {key<RequesterKey>(Requester::Input),
                            key<RequesterKey>(Requester::Output)};
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.timingContracts = {TimingContractDeclaration{
      TimingContractKey(0), {rank(Event::Accept), rank(Event::Retire)}}};
  declaration.usePatterns = {
      UsePatternDeclaration{
          UsePatternKey(0),
          key<RequesterKey>(Requester::Input),
          key<EligibilityKey>(Eligibility::InputTokenReady),
          key<EventKey>(Event::Accept),
          key<EventKey>(Event::Retire),
          std::nullopt,
          key<TimingContractKey>(Timing::OneCycleElastic),
          {ClaimDeclaration{ClaimKey(0), key<StateKey>(State::Queue),
                            key<CapacityDimensionKey>(Dimension::StoredEntry),
                            CapacityUnits(1)}},
          {}},
      UsePatternDeclaration{
          UsePatternKey(1),
          key<RequesterKey>(Requester::Output),
          key<EligibilityKey>(Eligibility::OutputTokenReady),
          key<EventKey>(Event::Accept),
          key<EventKey>(Event::Retire),
          std::nullopt,
          key<TimingContractKey>(Timing::OneCycleElastic),
          {ClaimDeclaration{ClaimKey(0), key<StateKey>(State::Queue),
                            key<CapacityDimensionKey>(Dimension::DequeuePort),
                            CapacityUnits(1)}},
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
  declaration.states = {ResourceStateDeclaration{
      key<StateKey>(State::Egress),
      {CapacityDimensionDeclaration{
          key<CapacityDimensionKey>(Dimension::TransferSlot), CapacityUnits(1),
          CapacityUnits(0)}}}};
  declaration.requesters = {key<RequesterKey>(Requester::Ingress0),
                            key<RequesterKey>(Requester::Ingress1),
                            key<RequesterKey>(Requester::Ingress2)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {TimingContractDeclaration{
      TimingContractKey(0), {rank(Event::Accept), rank(Event::Retire)}}};
  for (std::uint32_t requester = 0; requester < 3; ++requester)
    declaration.usePatterns.push_back(UsePatternDeclaration{
        UsePatternKey(requester),
        RequesterKey(requester),
        key<EligibilityKey>(Eligibility::RouteSelected),
        key<EventKey>(Event::Accept),
        key<EventKey>(Event::Retire),
        std::nullopt,
        key<TimingContractKey>(Timing::SingleBeat),
        {ClaimDeclaration{ClaimKey(0), key<StateKey>(State::Egress),
                          key<CapacityDimensionKey>(Dimension::TransferSlot),
                          CapacityUnits(1)}},
        {}});
  return declaration;
}

ResourceContractDeclaration fixedPriorityDeclaration() {
  ResourceContractDeclaration declared = declaration();
  declared.grantPolicy = GrantPolicyDeclaration(
      FixedPriorityDeclaration{{key<RequesterKey>(Requester::Ingress2),
                                key<RequesterKey>(Requester::Ingress0),
                                key<RequesterKey>(Requester::Ingress1)}});
  return declared;
}

ResourceContractDeclaration roundRobinDeclaration() {
  ResourceContractDeclaration declared = declaration();
  declared.grantPolicy = GrantPolicyDeclaration(
      RoundRobinDeclaration{{key<RequesterKey>(Requester::Ingress0),
                             key<RequesterKey>(Requester::Ingress1),
                             key<RequesterKey>(Requester::Ingress2)},
                            key<RequesterKey>(Requester::Ingress1)});
  return declared;
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
      ResourceStateDeclaration{
          key<StateKey>(State::ServicePort),
          {CapacityDimensionDeclaration{
              key<CapacityDimensionKey>(ServicePortDimension::Beat),
              CapacityUnits(1), CapacityUnits(0)}}},
      ResourceStateDeclaration{
          key<StateKey>(State::Bank),
          {CapacityDimensionDeclaration{
              key<CapacityDimensionKey>(BankDimension::Access),
              CapacityUnits(2), CapacityUnits(0)}}},
  };
  declaration.requesters = {key<RequesterKey>(Requester::OperationRow)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {TimingContractDeclaration{
      TimingContractKey(0), {rank(Event::Accept), rank(Event::Retire)}}};
  declaration.usePatterns = {UsePatternDeclaration{
      UsePatternKey(0),
      key<RequesterKey>(Requester::OperationRow),
      key<EligibilityKey>(Eligibility::OperandTupleComplete),
      key<EventKey>(Event::Accept),
      key<EventKey>(Event::Retire),
      std::nullopt,
      key<TimingContractKey>(Timing::TwoBeatVectorLoad),
      {ClaimDeclaration{ClaimKey(0), key<StateKey>(State::ServicePort),
                        key<CapacityDimensionKey>(ServicePortDimension::Beat),
                        CapacityUnits(1)},
       ClaimDeclaration{ClaimKey(1), key<StateKey>(State::Bank),
                        key<CapacityDimensionKey>(BankDimension::Access),
                        CapacityUnits(2)}},
      {InternalTransactionDeclaration{{ClaimKey(0)}},
       InternalTransactionDeclaration{{ClaimKey(0), ClaimKey(1)}}}}};
  return declaration;
}

} // namespace memoryEngine

// One stateful store whose service reservation lives for a single cycle while
// its queue contents are durable. The stored dimension is never claimed: only a
// committed transition changes it, and no later use releases an earlier claim.
namespace operandStore {

enum class State : std::uint32_t { Service, Queue };
enum class ServiceDimension : std::uint32_t { Slot };
enum class QueueDimension : std::uint32_t { Stored };
enum class Transition : std::uint32_t { Append, Remove };
enum class Requester : std::uint32_t { Port };
enum class Eligibility : std::uint32_t { FreeEntry, StoredToken };
enum class Event : std::uint32_t { Commit, NextBoundary };
enum class Timing : std::uint32_t { ServiceWindow };

UsePatternDeclaration pattern(UsePatternKey patternKey, Eligibility eligibility,
                              Transition transition) {
  return UsePatternDeclaration{
      patternKey,
      key<RequesterKey>(Requester::Port),
      key<EligibilityKey>(eligibility),
      key<EventKey>(Event::Commit),
      key<EventKey>(Event::NextBoundary),
      CommitDeclaration{key<EventKey>(Event::Commit),
                        key<ResourceTransitionKey>(transition)},
      key<TimingContractKey>(Timing::ServiceWindow),
      {ClaimDeclaration{ClaimKey(0), key<StateKey>(State::Service),
                        key<CapacityDimensionKey>(ServiceDimension::Slot),
                        CapacityUnits(1)}},
      {}};
}

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration declaration;
  declaration.states = {
      ResourceStateDeclaration{
          key<StateKey>(State::Service),
          {CapacityDimensionDeclaration{
              key<CapacityDimensionKey>(ServiceDimension::Slot),
              CapacityUnits(1), CapacityUnits(0)}}},
      ResourceStateDeclaration{
          key<StateKey>(State::Queue),
          {CapacityDimensionDeclaration{
              key<CapacityDimensionKey>(QueueDimension::Stored),
              CapacityUnits(4), CapacityUnits(0)}}},
  };
  declaration.resourceTransitions = {
      key<ResourceTransitionKey>(Transition::Append),
      key<ResourceTransitionKey>(Transition::Remove)};
  declaration.timingContracts = {TimingContractDeclaration{
      key<TimingContractKey>(Timing::ServiceWindow),
      {rank(Event::Commit), rank(Event::NextBoundary)}}};
  declaration.requesters = {key<RequesterKey>(Requester::Port)};
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.usePatterns = {
      pattern(UsePatternKey(0), Eligibility::FreeEntry, Transition::Append),
      pattern(UsePatternKey(1), Eligibility::StoredToken, Transition::Remove),
  };
  return declaration;
}

} // namespace operandStore

void disjointResourceNeedsNoGrantPolicy() {
  const ResourceContract contract =
      takeContract(__func__, ResourceContract::create(buffer::declaration()));

  require(__func__, contract.stateCount() == 1, "declared state count differs");
  require(__func__, contract.usePatternCount() == 2,
          "declared use pattern count differs");
  require(__func__,
          contract.requesterCount() == 2 && contract.eligibilityCount() == 2 &&
              contract.eventCount() == 2 && contract.timingContractCount() == 1,
          "closed owner key domains differ");
  require(__func__, !contract.grantPolicy().has_value(),
          "disjoint requesters must not carry a requester order");

  const llvm::ArrayRef<CapacityDimension> dimensions =
      contract.capacityDimensions(key<StateKey>(buffer::State::Queue));
  require(__func__,
          dimensions.size() == 2 &&
              dimensions[0].capacity == CapacityUnits(4) &&
              dimensions[0].initialOccupancy == CapacityUnits(0) &&
              dimensions[1].capacity == CapacityUnits(1),
          "canonical capacity dimensions differ");

  const UsePattern enqueue = contract.usePattern(UsePatternKey(0));
  require(__func__,
          enqueue.requester == key<RequesterKey>(buffer::Requester::Input) &&
              enqueue.acquire == key<EventKey>(buffer::Event::Accept) &&
              enqueue.release == key<EventKey>(buffer::Event::Retire) &&
              enqueue.internalTransactionCount == 0,
          "atomic use pattern content differs");
  require(__func__,
          enqueue.claims.size() == 1 &&
              sameClaim(enqueue.claims[0],
                        Claim{key<StateKey>(buffer::State::Queue),
                              key<CapacityDimensionKey>(
                                  buffer::Dimension::StoredEntry),
                              CapacityUnits(1)}),
          "atomic claim envelope differs");
}

void declarationOrderDoesNotChangeTheContract() {
  ResourceContractDeclaration shuffled = buffer::declaration();
  std::swap(shuffled.usePatterns[0], shuffled.usePatterns[1]);
  std::swap(shuffled.states[0].capacityDimensions[0],
            shuffled.states[0].capacityDimensions[1]);
  std::swap(shuffled.requesters[0], shuffled.requesters[1]);

  const ResourceContract contract =
      takeContract(__func__, ResourceContract::create(shuffled));

  const llvm::ArrayRef<CapacityDimension> dimensions =
      contract.capacityDimensions(key<StateKey>(buffer::State::Queue));
  require(__func__,
          dimensions[0].capacity == CapacityUnits(4) &&
              dimensions[1].capacity == CapacityUnits(1),
          "capacity dimensions are not normalized by key");
  require(__func__,
          contract.usePattern(UsePatternKey(0)).requester ==
              key<RequesterKey>(buffer::Requester::Input),
          "use patterns are not normalized by key");
}

void sharedCapacityWithoutPolicyIsRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[1].claims[0].dimension =
      key<CapacityDimensionKey>(buffer::Dimension::StoredEntry);
  expectViolation(__func__, "two requesters on one capacity dimension",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::ContentionWithoutGrantPolicy);
}

void unobservableRequesterOrderIsRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.grantPolicy = GrantPolicyDeclaration(
      FixedPriorityDeclaration{{key<RequesterKey>(buffer::Requester::Input),
                                key<RequesterKey>(buffer::Requester::Output)}});
  expectViolation(__func__, "requester order without reachable contention",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::GrantPolicyWithoutContention);
}

void duplicateDeclaredKeysAreRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.states.push_back(declaration.states[0]);
  expectViolation(__func__, "one state key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateStateKey);

  declaration = buffer::declaration();
  declaration.states[0].capacityDimensions[1].key =
      key<CapacityDimensionKey>(buffer::Dimension::StoredEntry);
  expectViolation(__func__, "one capacity dimension key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateCapacityDimensionKey);

  declaration = buffer::declaration();
  declaration.requesters[1] = key<RequesterKey>(buffer::Requester::Input);
  expectViolation(__func__, "one requester key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateRequesterKey);

  declaration = buffer::declaration();
  declaration.usePatterns[1].key = UsePatternKey(0);
  expectViolation(__func__, "one use pattern key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateUsePatternKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].claims.push_back(
      declaration.usePatterns[0].claims[0]);
  expectViolation(__func__, "one claim key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateClaimKey);
}

void nonClosedDeclaredKeysAreRejected() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.states[0].key = StateKey(1);
  expectViolation(__func__, "state key outside the closed state domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownStateKey);

  declaration = buffer::declaration();
  declaration.states[0].capacityDimensions[1].key = CapacityDimensionKey(2);
  expectViolation(__func__, "capacity dimension key outside its domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownCapacityDimensionKey);

  declaration = buffer::declaration();
  declaration.requesters[1] = RequesterKey(2);
  expectViolation(__func__, "requester key outside the closed domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);

  declaration = buffer::declaration();
  declaration.usePatterns[1].key = UsePatternKey(2);
  expectViolation(__func__, "use pattern key outside the closed domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownUsePatternKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].claims[0].key = ClaimKey(1);
  expectViolation(__func__, "claim key outside the pattern envelope",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownClaimKey);
}

void unknownReferencedKeysAreRejected() {
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
  declaration.usePatterns[0].release = EventKey(2);
  expectViolation(__func__, "use pattern released by an undeclared event",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);

  declaration = buffer::declaration();
  declaration.usePatterns[0].timingAndProgress = TimingContractKey(1);
  expectViolation(__func__, "use pattern with an undeclared timing contract",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownTimingContractKey);
}

// A use may reserve capacity briefly and still change durable state: the claim
// envelope holds only the service slot, while the queue contents move under the
// committed transition.
void shortLivedClaimAndDurableCommitAreDistinct() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(operandStore::declaration()));
  require(__func__, contract.resourceTransitionCount() == 2,
          "the closed transition inventory differs");
  require(__func__,
          contract.eventOrder(
              key<TimingContractKey>(operandStore::Timing::ServiceWindow)) ==
              llvm::ArrayRef<std::uint32_t>({0, 1}),
          "the declared event order differs");

  const UsePattern append = contract.usePattern(UsePatternKey(0));
  require(__func__,
          append.claims.size() == 1 &&
              sameClaim(append.claims[0],
                        Claim{key<StateKey>(operandStore::State::Service),
                              key<CapacityDimensionKey>(
                                  operandStore::ServiceDimension::Slot),
                              CapacityUnits(1)}),
          "durable queue contents must not appear in the claim envelope");
  require(__func__,
          append.commit.has_value() &&
              append.commit->transition ==
                  key<ResourceTransitionKey>(
                      operandStore::Transition::Append) &&
              append.commit->event == append.acquire &&
              append.release ==
                  key<EventKey>(operandStore::Event::NextBoundary),
          "the committed transition or its event differs");

  const UsePattern remove = contract.usePattern(UsePatternKey(1));
  require(__func__,
          remove.commit.has_value() &&
              remove.commit->transition ==
                  key<ResourceTransitionKey>(operandStore::Transition::Remove),
          "the second pattern must commit its own transition");
}

void unknownCommitReferencesAreRejected() {
  ResourceContractDeclaration declaration = operandStore::declaration();
  declaration.usePatterns[0].commit->transition = ResourceTransitionKey(2);
  expectViolation(__func__, "commit naming an undeclared transition",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownResourceTransitionKey);

  declaration = operandStore::declaration();
  declaration.usePatterns[0].commit->event = EventKey(2);
  expectViolation(__func__, "commit naming an undeclared event",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);

  declaration = operandStore::declaration();
  declaration.resourceTransitions[1] =
      key<ResourceTransitionKey>(operandStore::Transition::Append);
  expectViolation(__func__, "one transition key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateResourceTransitionKey);
}

void theTimingContractMustOrderAcquireCommitAndRelease() {
  ResourceContractDeclaration declaration = operandStore::declaration();
  declaration.timingContracts[0].eventRank = {1, 0};
  expectViolation(__func__, "a commit ranked after its release",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::TimingContractDoesNotOrderUse);

  declaration = operandStore::declaration();
  declaration.timingContracts[0].eventRank = {0};
  expectViolation(__func__, "a timing contract that ranks only some events",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::TimingContractDoesNotOrderUse);

  declaration = operandStore::declaration();
  declaration.timingContracts.push_back(declaration.timingContracts[0]);
  expectViolation(__func__, "one timing contract key declared twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateTimingContractKey);
}

void oneEnvelopeHasOneReleaseAndOneClaimPerDimension() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(memoryEngine::declaration()));
  const UsePattern pattern = contract.usePattern(UsePatternKey(0));
  require(__func__, pattern.claims.size() == 2 &&
                        pattern.release ==
                            key<EventKey>(memoryEngine::Event::Retire),
          "the enclosing pattern must release its complete claim envelope");

  ResourceContractDeclaration declaration = memoryEngine::declaration();
  declaration.usePatterns[0].claims[1].state =
      key<StateKey>(memoryEngine::State::ServicePort);
  declaration.usePatterns[0].claims[1].dimension =
      key<CapacityDimensionKey>(memoryEngine::ServicePortDimension::Beat);
  expectViolation(__func__, "one capacity dimension claimed twice",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateCapacityClaim);
}

void capacityArithmeticFailuresAreDistinct() {
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
  declaration.states[0].capacityDimensions[0].capacity =
      CapacityUnits(0xffffffffu);
  declaration.states[0].capacityDimensions[0].initialOccupancy =
      CapacityUnits(1);
  declaration.usePatterns[0].claims[0].amount = CapacityUnits(0xffffffffu);
  expectViolation(__func__, "occupancy sum that is not representable",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::CapacityArithmeticOverflow);
}

void fixedPriorityGrantsFirstEligibleRequester() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(crossbar::fixedPriorityDeclaration()));
  const std::optional<GrantPolicyView> policy = contract.grantPolicy();
  require(__func__, policy.has_value(),
          "contended capacity must carry a requester order");
  require(__func__, std::holds_alternative<FixedPriorityView>(*policy),
          "a declared fixed priority must read back as fixed priority");

  const FixedPriorityView &order = std::get<FixedPriorityView>(*policy);
  require(__func__,
          order.requesterOrder().size() == 3 &&
              order.requesterOrder()[0] ==
                  key<RequesterKey>(crossbar::Requester::Ingress2),
          "the exact permutation differs");

  const bool lowerPair[] = {true, true, false};
  require(__func__,
          order.grant(lowerPair) ==
              key<RequesterKey>(crossbar::Requester::Ingress0),
          "fixed priority must grant the first eligible requester in order");

  const bool upperPair[] = {false, true, true};
  require(__func__,
          order.grant(upperPair) ==
              key<RequesterKey>(crossbar::Requester::Ingress2),
          "fixed priority must prefer the earlier permutation entry");

  const bool noneEligible[] = {false, false, false};
  require(__func__, !order.grant(noneEligible).has_value(),
          "an ineligible cycle must not grant");
}

void grantPolicyMustBeAnExactRequesterPermutation() {
  ResourceContractDeclaration declaration = crossbar::declaration();
  declaration.grantPolicy = GrantPolicyDeclaration(FixedPriorityDeclaration{
      {key<RequesterKey>(crossbar::Requester::Ingress0),
       key<RequesterKey>(crossbar::Requester::Ingress1)}});
  expectViolation(__func__, "policy omitting a declared requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::RequesterOmittedFromGrantPolicy);

  declaration.grantPolicy = GrantPolicyDeclaration(FixedPriorityDeclaration{
      {key<RequesterKey>(crossbar::Requester::Ingress0),
       key<RequesterKey>(crossbar::Requester::Ingress0),
       key<RequesterKey>(crossbar::Requester::Ingress1)}});
  expectViolation(__func__, "policy repeating a requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::DuplicateRequesterInGrantPolicy);

  declaration.grantPolicy = GrantPolicyDeclaration(FixedPriorityDeclaration{
      {key<RequesterKey>(crossbar::Requester::Ingress0),
       key<RequesterKey>(crossbar::Requester::Ingress1), RequesterKey(3)}});
  expectViolation(__func__, "policy naming a foreign requester",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);
}

void roundRobinAdvancesOnlyOnASuccessfulGrant() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(crossbar::roundRobinDeclaration()));
  const std::optional<GrantPolicyView> policy = contract.grantPolicy();
  require(__func__,
          policy.has_value() && std::holds_alternative<RoundRobinView>(*policy),
          "a declared round robin must read back as round robin");

  const RoundRobinView &cycle = std::get<RoundRobinView>(*policy);
  const RequesterKey reset = cycle.resetCursor();
  require(__func__, reset == key<RequesterKey>(crossbar::Requester::Ingress1),
          "reset must establish the declared cursor");

  const bool noneEligible[] = {false, false, false};
  const RoundRobinGrant idle = cycle.grant(reset, noneEligible);
  require(__func__, !idle.granted.has_value() && idle.nextCursor == reset,
          "a cycle without a grant must preserve the cursor");

  const bool wrappedEligible[] = {true, false, false};
  const RoundRobinGrant wrapped = cycle.grant(reset, wrappedEligible);
  require(__func__,
          wrapped.granted == key<RequesterKey>(crossbar::Requester::Ingress0),
          "round robin must scan its exact cycle from the cursor");
  require(__func__,
          wrapped.nextCursor ==
              key<RequesterKey>(crossbar::Requester::Ingress1),
          "a successful grant must advance past the granted requester");

  const bool cursorEligible[] = {true, true, false};
  const RoundRobinGrant atCursor =
      cycle.grant(wrapped.nextCursor, cursorEligible);
  require(__func__,
          atCursor.granted == key<RequesterKey>(crossbar::Requester::Ingress1),
          "the cursor entry is scanned first");
  require(__func__,
          atCursor.nextCursor ==
              key<RequesterKey>(crossbar::Requester::Ingress2),
          "the cursor advances to the successor of the granted requester");

  ResourceContractDeclaration declaration = crossbar::roundRobinDeclaration();
  std::get<RoundRobinDeclaration>(*declaration.grantPolicy).resetCursor =
      RequesterKey(3);
  expectViolation(__func__, "reset cursor outside the requester domain",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownRequesterKey);
}

void internalTransactionsRefineOneAcceptedUse() {
  const ResourceContract contract = takeContract(
      __func__, ResourceContract::create(memoryEngine::declaration()));

  const UsePattern operation = contract.usePattern(UsePatternKey(0));
  require(__func__,
          operation.requester ==
                  key<RequesterKey>(memoryEngine::Requester::OperationRow) &&
              operation.acquire == key<EventKey>(memoryEngine::Event::Accept) &&
              operation.release == key<EventKey>(memoryEngine::Event::Retire) &&
              operation.claims.size() == 2,
          "the external firing, retirement, and claim envelope differ");
  require(__func__, operation.internalTransactionCount == 2,
          "internal transaction count differs");

  const llvm::ArrayRef<ClaimKey> beat =
      contract.internalTransaction(UsePatternKey(0), 0);
  const llvm::ArrayRef<ClaimKey> beatWithBank =
      contract.internalTransaction(UsePatternKey(0), 1);
  require(__func__, beat.size() == 1 && beat[0] == ClaimKey(0),
          "the first internal transaction differs");
  require(__func__,
          beatWithBank.size() == 2 && beatWithBank[0] == ClaimKey(0) &&
              beatWithBank[1] == ClaimKey(1),
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
                  ResourceContractViolation::DuplicateClaimKey);

  // A transaction selects an unordered set of the envelope's claims, so a
  // malformed selection reports the same typed violation in any entry order.
  declaration = memoryEngine::declaration();
  declaration.usePatterns[0].internalTransactions[1].claims = {
      ClaimKey(5), ClaimKey(0), ClaimKey(0)};
  expectStableViolation(
      __func__, "internal transaction selection", declaration,
      ResourceContractViolation::DuplicateClaimKey,
      [](ResourceContractDeclaration &declared) {
        std::reverse(
            declared.usePatterns[0].internalTransactions[1].claims.begin(),
            declared.usePatterns[0].internalTransactions[1].claims.end());
      });
}

void violationPrecedenceIsIndependentOfDeclarationOrder() {
  ResourceContractDeclaration declaration = buffer::declaration();
  declaration.usePatterns[0].acquire = EventKey(2);
  declaration.usePatterns[1].claims.push_back(ClaimDeclaration{
      ClaimKey(1), key<StateKey>(buffer::State::Queue),
      key<CapacityDimensionKey>(buffer::Dimension::DequeuePort),
      CapacityUnits(1)});
  expectViolation(__func__, "unknown key before duplicate capacity claim",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);

  std::swap(declaration.usePatterns[0], declaration.usePatterns[1]);
  expectViolation(__func__, "duplicate capacity claim before unknown key",
                  ResourceContract::create(declaration),
                  ResourceContractViolation::UnknownEventKey);
}

// Every closed inventory below carries one repeated key and one key outside its
// domain. An inventory is unordered, so both orders must report the duplicate.
void inventoryViolationsAreInvariantUnderDeclarationOrder() {
  ResourceContractDeclaration declaration = buffer::declaration();
  const ResourceStateDeclaration declaredState = declaration.states[0];
  ResourceStateDeclaration foreignState = declaredState;
  foreignState.key = StateKey(5);
  declaration.states = {declaredState, declaredState, foreignState};
  expectStableViolation(__func__, "state inventory", declaration,
                        ResourceContractViolation::DuplicateStateKey,
                        [](ResourceContractDeclaration &declared) {
                          std::reverse(declared.states.begin(),
                                       declared.states.end());
                        });

  declaration = buffer::declaration();
  const CapacityDimensionDeclaration declaredDimension =
      declaration.states[0].capacityDimensions[0];
  CapacityDimensionDeclaration foreignDimension = declaredDimension;
  foreignDimension.key = CapacityDimensionKey(5);
  declaration.states[0].capacityDimensions = {
      declaredDimension, declaredDimension, foreignDimension};
  expectStableViolation(
      __func__, "capacity dimension inventory", declaration,
      ResourceContractViolation::DuplicateCapacityDimensionKey,
      [](ResourceContractDeclaration &declared) {
        std::reverse(declared.states[0].capacityDimensions.begin(),
                     declared.states[0].capacityDimensions.end());
      });

  declaration = buffer::declaration();
  declaration.requesters = {RequesterKey(0), RequesterKey(0), RequesterKey(5)};
  expectStableViolation(__func__, "requester inventory", declaration,
                        ResourceContractViolation::DuplicateRequesterKey,
                        [](ResourceContractDeclaration &declared) {
                          std::reverse(declared.requesters.begin(),
                                       declared.requesters.end());
                        });

  declaration = buffer::declaration();
  const UsePatternDeclaration declaredPattern = declaration.usePatterns[0];
  UsePatternDeclaration foreignPattern = declaredPattern;
  foreignPattern.key = UsePatternKey(5);
  declaration.usePatterns = {declaredPattern, declaredPattern, foreignPattern};
  expectStableViolation(__func__, "use pattern inventory", declaration,
                        ResourceContractViolation::DuplicateUsePatternKey,
                        [](ResourceContractDeclaration &declared) {
                          std::reverse(declared.usePatterns.begin(),
                                       declared.usePatterns.end());
                        });

  declaration = buffer::declaration();
  const ClaimDeclaration declaredClaim = declaration.usePatterns[0].claims[0];
  ClaimDeclaration foreignClaim = declaredClaim;
  foreignClaim.key = ClaimKey(5);
  declaration.usePatterns[0].claims = {declaredClaim, declaredClaim,
                                       foreignClaim};
  expectStableViolation(__func__, "claim inventory", declaration,
                        ResourceContractViolation::DuplicateClaimKey,
                        [](ResourceContractDeclaration &declared) {
                          std::reverse(declared.usePatterns[0].claims.begin(),
                                       declared.usePatterns[0].claims.end());
                        });
}

} // namespace

int main() {
  disjointResourceNeedsNoGrantPolicy();
  declarationOrderDoesNotChangeTheContract();
  sharedCapacityWithoutPolicyIsRejected();
  unobservableRequesterOrderIsRejected();
  duplicateDeclaredKeysAreRejected();
  nonClosedDeclaredKeysAreRejected();
  unknownReferencedKeysAreRejected();
  shortLivedClaimAndDurableCommitAreDistinct();
  unknownCommitReferencesAreRejected();
  theTimingContractMustOrderAcquireCommitAndRelease();
  oneEnvelopeHasOneReleaseAndOneClaimPerDimension();
  capacityArithmeticFailuresAreDistinct();
  fixedPriorityGrantsFirstEligibleRequester();
  grantPolicyMustBeAnExactRequesterPermutation();
  roundRobinAdvancesOnlyOnASuccessfulGrant();
  internalTransactionsRefineOneAcceptedUse();
  violationPrecedenceIsIndependentOfDeclarationOrder();
  inventoryViolationsAreInvariantUnderDeclarationOrder();
  return 0;
}
