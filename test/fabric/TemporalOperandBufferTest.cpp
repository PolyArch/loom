#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/IR/TemporalPeResourceContract.h"

#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
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

const loom::fabric::FabricPeOccurrenceRef kPe{7};

// Two resident contexts over one FU occurrence with two inputs and a second FU
// occurrence with one input: six logical queues, three FU ingress banks.
const std::uint32_t kFuInputCounts[] = {2, 1};

TemporalOperandBufferDeclaration declaration(OperandBufferMode mode,
                                             std::uint32_t entries) {
  TemporalOperandBufferDeclaration declared;
  declared.pe = kPe;
  declared.contextCount = 2;
  declared.fuInputCounts = kFuInputCounts;
  declared.mode = mode;
  declared.entriesPerAllocationUnit = entries;
  return declared;
}

TemporalOperandBufferContract
takeContract(const char *test,
             llvm::Expected<TemporalOperandBufferContract> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return std::move(*result);
}

TemporalPeResourceContract
takePeContract(const char *test,
               llvm::Expected<TemporalPeResourceContract> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return std::move(*result);
}

void expectViolation(const char *test, const std::string &label,
                     llvm::Expected<TemporalOperandBufferContract> result,
                     TemporalOperandBufferViolation expected) {
  if (result)
    fail(test, label + ": expected a rejected declaration");

  std::optional<TemporalOperandBufferViolation> observed;
  llvm::handleAllErrors(result.takeError(),
                        [&](const TemporalOperandBufferError &error) {
                          observed = error.violation();
                        });
  if (!observed)
    fail(test, label + ": received a different error category");
  if (*observed != expected)
    fail(test, label + ": observed violation " +
                   getTemporalOperandBufferViolationName(*observed).str() +
                   ", expected " +
                   getTemporalOperandBufferViolationName(expected).str());
}

const OperandBufferMode kModes[] = {OperandBufferMode::PerInstruction,
                                    OperandBufferMode::PerInputPort,
                                    OperandBufferMode::AllFuShare};

template <typename Key, typename Enum> constexpr Key key(Enum value) {
  return Key(static_cast<std::uint32_t>(value));
}

CapacityUnits capacityOf(const TemporalOperandBufferContract &contract,
                         StateKey state, std::uint32_t dimension) {
  return contract.resourceContract()
      .capacityDimensions(state)[dimension]
      .capacity;
}

// Every mode rejects an absent or zero `operand_buffer_size`: the typed API
// only ever receives the value the Fabric hardware parameter carries, and zero
// is exactly what an omitted parameter would mean.
void everyModeRequiresAPositiveEntryCapacity() {
  for (OperandBufferMode mode : kModes) {
    expectViolation(__func__, stringifyOperandBufferMode(mode).str(),
                    TemporalOperandBufferContract::create(declaration(mode, 0)),
                    TemporalOperandBufferViolation::NonPositiveEntryCapacity);

    TemporalOperandBufferDeclaration emptyContexts = declaration(mode, 2);
    emptyContexts.contextCount = 0;
    expectViolation(__func__, stringifyOperandBufferMode(mode).str(),
                    TemporalOperandBufferContract::create(emptyContexts),
                    TemporalOperandBufferViolation::EmptyContextDomain);
  }
}

// The canonical logical-queue domain is the complete key set in lexicographic
// order, and each mode projects it onto allocation units exactly as specified.
void modeProjectionDiffersExactly() {
  const TemporalOperandBufferContract dedicated =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::PerInstruction, 2)));

  const llvm::ArrayRef<LogicalOperandQueueKey> queues =
      dedicated.logicalQueues();
  require(__func__, queues.size() == 6, "logical queue domain size differs");
  const LogicalOperandQueueKey expected[] = {
      {{kPe, 0}, 0, 0}, {{kPe, 0}, 0, 1}, {{kPe, 0}, 1, 0},
      {{kPe, 1}, 0, 0}, {{kPe, 1}, 0, 1}, {{kPe, 1}, 1, 0},
  };
  for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
    require(__func__, queues[queue] == expected[queue],
            "canonical logical queue order differs at " +
                std::to_string(queue));
  for (std::uint32_t queue = 1; queue != queues.size(); ++queue)
    require(__func__, queues[queue - 1] < queues[queue],
            "canonical order is not the lexicographic key order");

  require(__func__, dedicated.allocationUnitCount() == 6,
          "per_instruction must give every logical queue its own unit");
  for (std::uint32_t queue = 0; queue != queues.size(); ++queue) {
    const std::uint32_t unit = dedicated.allocationUnitOf(queue);
    const auto *whole =
        std::get_if<DedicatedQueueUnit>(&dedicated.allocationUnit(unit));
    require(__func__, whole != nullptr,
            "per_instruction must preserve the whole logical key");
    require(__func__, whole->queue == queues[queue],
            "per_instruction unit key differs from its logical queue");
    require(__func__, dedicated.queuesOf(unit).size() == 1,
            "a dedicated queue must not share its unit");
  }
  require(__func__, !dedicated.resourceContract().grantPolicy().has_value(),
          "a dedicated queue proves at most one requester per capacity");

  const TemporalOperandBufferContract banked =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::PerInputPort, 2)));
  require(__func__, banked.allocationUnitCount() == 3,
          "per_input_port must give one unit per FU ingress");
  const FuInputUnit expectedBanks[] = {{0, 0}, {0, 1}, {1, 0}};
  for (std::uint32_t unit = 0; unit != 3; ++unit) {
    const auto *bank = std::get_if<FuInputUnit>(&banked.allocationUnit(unit));
    require(__func__, bank != nullptr,
            "per_input_port must project onto FU occurrence and FU input");
    require(__func__,
            bank->fuOccurrence == expectedBanks[unit].fuOccurrence &&
                bank->fuInput == expectedBanks[unit].fuInput,
            "per_input_port unit differs at " + std::to_string(unit));
    require(__func__, banked.queuesOf(unit).size() == 2,
            "every resident context must share its FU ingress bank");
    for (std::uint32_t queue : banked.queuesOf(unit))
      require(__func__,
              banked.logicalQueues()[queue].fuOccurrence ==
                      expectedBanks[unit].fuOccurrence &&
                  banked.logicalQueues()[queue].fuInput ==
                      expectedBanks[unit].fuInput,
              "per_input_port membership differs at " + std::to_string(unit));
  }

  const TemporalOperandBufferContract shared =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::AllFuShare, 2)));
  require(__func__, shared.allocationUnitCount() == 1,
          "all_fu_share must project onto the whole temporal PE");
  const auto *pe = std::get_if<WholeTemporalPeUnit>(&shared.allocationUnit(0));
  require(__func__, pe != nullptr && pe->pe == kPe,
          "all_fu_share unit must be the owning temporal PE");
  require(__func__, shared.queuesOf(0).size() == 6,
          "all_fu_share must pool every logical queue");
  for (std::uint32_t queue = 0; queue != 6; ++queue)
    require(__func__, shared.allocationUnitOf(queue) == 0,
            "all_fu_share must project every queue onto one unit");
}

// An enqueue reserves one service slot for one PE clock cycle and durably
// appends through its own commit transition. The pool and the queue are state,
// never claims, so nothing an enqueue holds can be released by a later dequeue.
void enqueueClaimsOneServiceAndCommitsADurableAppend() {
  const TemporalOperandBufferContract contract =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::AllFuShare, 4)));

  const StateKey pool = contract.entryPoolState(0);
  const StateKey enqueueService = contract.enqueueServiceState(0);
  const StateKey dequeueService = contract.dequeueServiceState(0);
  require(__func__, enqueueService != dequeueService,
          "enqueue and dequeue must not share one service state");
  require(__func__,
          capacityOf(contract, pool,
                     static_cast<std::uint32_t>(
                         OperandEntryPoolDimension::OccupiedEntry)) ==
              CapacityUnits(4),
          "entry pool capacity must equal operand_buffer_size");
  require(
      __func__,
      capacityOf(contract, enqueueService,
                 static_cast<std::uint32_t>(OperandServiceDimension::Slot)) ==
              CapacityUnits(1) &&
          capacityOf(contract, dequeueService,
                     static_cast<std::uint32_t>(
                         OperandServiceDimension::Slot)) == CapacityUnits(1),
      "each allocation unit serves one enqueue and one dequeue per cycle");

  const UsePattern enqueue =
      contract.resourceContract().usePattern(contract.enqueuePattern(0));
  require(__func__,
          enqueue.claims.size() == 1 &&
              enqueue.claims[0].state == enqueueService,
          "an enqueue must claim only its enqueue service slot");
  require(__func__,
          enqueue.acquire == key<EventKey>(OperandBufferEvent::EnqueueCommit) &&
              enqueue.release ==
                  key<EventKey>(OperandBufferEvent::NextPeClockBoundary),
          "an enqueue must hold its slot until the next PE clock boundary");
  require(__func__,
          enqueue.commit.has_value() &&
              enqueue.commit->event == enqueue.acquire &&
              enqueue.commit->transition == contract.appendTransition(0),
          "an enqueue must atomically commit AppendOperand");

  const StateKey queue = contract.queueState(0);
  for (std::uint32_t pattern = 0;
       pattern != contract.resourceContract().usePatternCount(); ++pattern)
    for (const Claim &claim :
         contract.resourceContract().usePattern(UsePatternKey(pattern)).claims)
      require(__func__, claim.state != pool && claim.state != queue,
              "durable pool and queue state must never be claimed");
}

// A dequeue mirrors it: one service slot and its own durable removal.
void dequeueCommitsADurableRemoval() {
  const TemporalOperandBufferContract contract =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::AllFuShare, 4)));

  const UsePattern dequeue =
      contract.resourceContract().usePattern(contract.dequeuePattern(3));
  require(__func__,
          dequeue.claims.size() == 1 &&
              dequeue.claims[0].state == contract.dequeueServiceState(0),
          "a dequeue must claim only its dequeue service slot");
  require(
      __func__,
      dequeue.eligibility ==
          key<EligibilityKey>(OperandBufferEligibility::CycleStartHeadPresent),
      "a dequeue must observe only a cycle-start head");
  require(__func__,
          dequeue.commit.has_value() &&
              dequeue.commit->event ==
                  key<EventKey>(OperandBufferEvent::DequeueCommit) &&
              dequeue.commit->transition == contract.removeTransition(3),
          "a dequeue must atomically commit RemoveOperand");
  require(__func__, contract.resourceContract().resourceTransitionCount() == 12,
          "every logical queue owns one append and one remove transition");
}

// A full allocation unit admits a pop together with a push, the pushed operand
// cannot satisfy that cycle's dequeue, and the next occupancy is exactly
// `O - D + E`.
void fullUnitAdmitsPopWithPushWithoutBypass() {
  const TemporalOperandBufferContract contract =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::AllFuShare, 2)));
  const CapacityUnits full = contract.entriesPerAllocationUnit();

  require(__func__, !contract.admits(full, {false, true}),
          "a full allocation unit must backpressure a lone enqueue");
  require(__func__, contract.admits(full, {true, true}),
          "a dequeue must give same-cycle capacity to an enqueue");
  require(__func__, contract.occupancyAfter(full, {true, true}) == full,
          "pop with push must leave occupancy unchanged");

  const CapacityUnits empty(0);
  require(__func__, !contract.admits(empty, {true, false}),
          "an empty queue has no cycle-start head to remove");
  require(__func__, !contract.admits(empty, {true, true}),
          "an operand appended this cycle cannot satisfy this cycle's dequeue");
  require(__func__,
          contract.admits(empty, {false, true}) &&
              contract.occupancyAfter(empty, {false, true}) == CapacityUnits(1),
          "an empty unit must admit an enqueue and hold one operand");

  const CapacityUnits one(1);
  require(__func__,
          contract.occupancyAfter(one, {true, false}) == empty &&
              contract.occupancyAfter(one, {false, true}) == full,
          "the next occupancy must be exactly O - D + E");
}

// `per_instruction` depth 1 and depth 2 are different hardware: they differ in
// the Fabric-owned entry capacity and in backpressure.
void dedicatedDepthOneAndTwoDiffer() {
  const TemporalOperandBufferContract depthOne =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::PerInstruction, 1)));
  const TemporalOperandBufferContract depthTwo =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::PerInstruction, 2)));
  require(__func__,
          depthOne.entriesPerAllocationUnit() !=
              depthTwo.entriesPerAllocationUnit(),
          "declared entry capacity must distinguish depth 1 from depth 2");
  require(__func__,
          !depthOne.admits(CapacityUnits(1), {false, true}) &&
              depthTwo.admits(CapacityUnits(1), {false, true}),
          "depth 1 and depth 2 must backpressure differently");
}

// Two logical queues sharing one FU ingress bank contend for its enqueue and
// dequeue services, which arbitrate through the canonical round-robin order
// with independent cursors.
void roundRobinContentionBetweenTwoLogicalQueues() {
  const std::uint32_t oneIngress[] = {1};
  TemporalOperandBufferDeclaration declared =
      declaration(OperandBufferMode::PerInputPort, 2);
  declared.fuInputCounts = oneIngress;

  const TemporalOperandBufferContract contract =
      takeContract(__func__, TemporalOperandBufferContract::create(declared));
  require(__func__,
          contract.logicalQueues().size() == 2 &&
              contract.allocationUnitCount() == 1,
          "two resident contexts must share one FU ingress bank");
  require(__func__, contract.queuesOf(0).size() == 2,
          "the shared unit must filter to both logical queues");

  const std::optional<GrantPolicyView> policy =
      contract.resourceContract().grantPolicy();
  require(__func__, policy.has_value(),
          "a contended service requires an exact grant policy");
  const auto *roundRobin = std::get_if<RoundRobinView>(&*policy);
  require(__func__, roundRobin != nullptr,
          "operand services arbitrate by round robin");
  require(__func__,
          roundRobin->requesterCycle().size() == 2 &&
              roundRobin->requesterCycle()[0] == contract.requester(0) &&
              roundRobin->requesterCycle()[1] == contract.requester(1) &&
              roundRobin->resetCursor() == contract.requester(0),
          "the requester cycle must be the canonical logical-queue order");

  // Both queues request the enqueue service; the dequeue service is requested
  // by the second queue alone. The two cursors advance independently and only
  // on a successful grant.
  const bool both[] = {true, true};
  const bool secondOnly[] = {false, true};

  RequesterKey enqueueCursor = roundRobin->resetCursor();
  RequesterKey dequeueCursor = roundRobin->resetCursor();

  RoundRobinGrant granted = roundRobin->grant(enqueueCursor, both);
  require(__func__, granted.granted == contract.requester(0),
          "the reset cursor must grant the first canonical requester");
  enqueueCursor = granted.nextCursor;

  granted = roundRobin->grant(dequeueCursor, secondOnly);
  require(__func__, granted.granted == contract.requester(1),
          "an independent dequeue cursor must scan its own eligible set");
  dequeueCursor = granted.nextCursor;

  granted = roundRobin->grant(enqueueCursor, both);
  require(__func__, granted.granted == contract.requester(1),
          "the enqueue cursor must advance past its granted requester");
  enqueueCursor = granted.nextCursor;

  const bool none[] = {false, false};
  granted = roundRobin->grant(enqueueCursor, none);
  require(__func__,
          !granted.granted.has_value() && granted.nextCursor == enqueueCursor,
          "a cursor advances only on a successful grant");

  granted = roundRobin->grant(enqueueCursor, both);
  require(__func__, granted.granted == contract.requester(0),
          "round-robin service order must be deterministic");
}

// One actor transition removes every head it needs under its single commit
// activation. Two required heads in one allocation unit exceed that unit's one
// dequeue service, so the binding is invalid rather than privately serialized.
void actorRequiredDequeueOvercapacityIsRejected() {
  const TemporalOperandBufferContract banked =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::PerInputPort, 2)));
  require(
      __func__, banked.admitsActorDequeueSet({0, 1}),
      "two heads in different FU ingress banks each have their own service");
  require(__func__, !banked.admitsActorDequeueSet({0, 3}),
          "two contexts of one ingress bank exceed its one dequeue service");

  const TemporalOperandBufferContract shared =
      takeContract(__func__, TemporalOperandBufferContract::create(declaration(
                                 OperandBufferMode::AllFuShare, 2)));
  require(__func__, shared.admitsActorDequeueSet({2}),
          "one head always fits the pooled dequeue service");
  require(__func__, !shared.admitsActorDequeueSet({0, 1}),
          "a shared pool serves one dequeue per PE clock cycle");
}

void registerFifoPortCountOwnsServiceConcurrency() {
  const auto declared = [](std::uint32_t ports) {
    return TemporalPeResourceDeclaration{
        kPe, 2, kFuInputCounts, OperandBufferMode::PerInputPort, 2,
        1,   4, ports};
  };
  const TemporalPeResourceContract single =
      takePeContract(__func__, TemporalPeResourceContract::create(declared(1)));
  const TemporalPeResourceContract dual =
      takePeContract(__func__, TemporalPeResourceContract::create(declared(2)));

  const UsePattern singleWrite =
      single.resourceContract().usePattern(single.registerFifoWritePattern(0));
  const UsePattern singleRead =
      single.resourceContract().usePattern(single.registerFifoReadPattern(0));
  require(__func__,
          singleWrite.claims.size() == 1 && singleRead.claims.size() == 1 &&
              singleWrite.claims.front().state == single.registerFifoState(0) &&
              singleRead.claims.front().state == single.registerFifoState(0) &&
              singleWrite.claims.front().dimension ==
                  singleRead.claims.front().dimension,
          "single-port register FIFO read and write must share one service");

  const UsePattern dualWrite =
      dual.resourceContract().usePattern(dual.registerFifoWritePattern(0));
  const UsePattern dualRead =
      dual.resourceContract().usePattern(dual.registerFifoReadPattern(0));
  require(__func__,
          dualWrite.claims.size() == 1 && dualRead.claims.size() == 1 &&
              dualWrite.claims.front().state == dual.registerFifoState(0) &&
              dualRead.claims.front().state == dual.registerFifoState(0) &&
              dualWrite.claims.front().dimension !=
                  dualRead.claims.front().dimension,
          "dual-port register FIFO read and write need independent services");
  require(__func__,
          singleWrite.commit.has_value() && singleRead.commit.has_value() &&
              dualWrite.commit.has_value() && dualRead.commit.has_value() &&
              singleWrite.acquire == singleWrite.commit->event &&
              singleRead.acquire == singleRead.commit->event &&
              singleWrite.release == singleRead.release &&
              dualWrite.acquire == dualWrite.commit->event &&
              dualRead.acquire == dualRead.commit->event &&
              dualWrite.release == dualRead.release,
          "register FIFO service and commit timing changed");
}

} // namespace

int main() {
  everyModeRequiresAPositiveEntryCapacity();
  modeProjectionDiffersExactly();
  enqueueClaimsOneServiceAndCommitsADurableAppend();
  dequeueCommitsADurableRemoval();
  fullUnitAdmitsPopWithPushWithoutBypass();
  dedicatedDepthOneAndTwoDiffer();
  roundRobinContentionBetweenTwoLogicalQueues();
  actorRequiredDequeueOvercapacityIsRejected();
  registerFifoPortCountOwnsServiceConcurrency();
  return 0;
}
