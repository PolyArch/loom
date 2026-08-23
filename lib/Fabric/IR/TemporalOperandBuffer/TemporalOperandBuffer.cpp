//===- TemporalOperandBuffer.cpp - Temporal-PE operand-buffer contract ----===//
//
// Derives the exact operand-buffer resource contract of one temporal PE from
// its two Fabric hardware parameters. The canonical logical-queue domain, the
// mode-derived allocation units, the entry pool and per-queue queue state, the
// two one-slot services, the two durable queue transitions, and the round-robin
// grant relation are all mechanical consequences of `operand_buffer_mode` and
// `operand_buffer_size`. Nothing here selects a workload, a default depth, or a
// private priority, and no capacity claim outlives its one PE clock cycle.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/TemporalOperandBuffer.h"

#include "Fabric/IR/FabricEnums.h"

#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <limits>
#include <utility>

using namespace fabric;

using loom::fabric::FabricOrdinal;
using loom::fabric::InstructionContextRef;

namespace {

template <typename Key, typename Enum> constexpr Key key(Enum value) {
  return Key(static_cast<std::uint32_t>(value));
}

llvm::Error rejected(TemporalOperandBufferViolation violation,
                     const llvm::Twine &message) {
  return llvm::make_error<TemporalOperandBufferError>(
      violation,
      (getTemporalOperandBufferViolationName(violation) + ": " + message)
          .str());
}

// The largest derived key domain is the state inventory, which holds one queue
// state per logical queue plus three states per allocation unit, and an
// allocation unit never outnumbers the queues.
constexpr std::uint64_t kMaxLogicalQueues =
    std::numeric_limits<std::uint32_t>::max() / 4;

// The derived key layout. Construction and every accessor read it here, so the
// contract's state, transition, and pattern ordinals are written down once.
struct KeyLayout {
  std::uint32_t unitCount = 0;
  std::uint32_t queueCount = 0;

  StateKey entryPool(std::uint32_t unit) const { return StateKey(unit); }
  StateKey queue(std::uint32_t queue) const {
    return StateKey(unitCount + queue);
  }
  StateKey enqueueService(std::uint32_t unit) const {
    return StateKey(unitCount + queueCount + unit);
  }
  StateKey dequeueService(std::uint32_t unit) const {
    return StateKey(2 * unitCount + queueCount + unit);
  }
  ResourceTransitionKey append(std::uint32_t queue) const {
    return ResourceTransitionKey(queue);
  }
  ResourceTransitionKey remove(std::uint32_t queue) const {
    return ResourceTransitionKey(queueCount + queue);
  }
  UsePatternKey enqueue(std::uint32_t queue) const {
    return UsePatternKey(queue);
  }
  UsePatternKey dequeue(std::uint32_t queue) const {
    return UsePatternKey(queueCount + queue);
  }
};

KeyLayout layoutOf(const TemporalOperandBufferContract &contract) {
  return KeyLayout{contract.allocationUnitCount(),
                   static_cast<std::uint32_t>(contract.logicalQueues().size())};
}

// One `OperandEntryPool` state. Its occupancy is durable operand-buffer state
// that only a committed transition changes, so no pattern claims it.
ResourceStateDeclaration declareEntryPool(StateKey state,
                                          CapacityUnits entries) {
  return ResourceStateDeclaration{
      state,
      {CapacityDimensionDeclaration{
          key<CapacityDimensionKey>(OperandEntryPoolDimension::OccupiedEntry),
          entries, CapacityUnits(0)}}};
}

// One `OperandQueue` state, empty after reset. A single logical queue can hold
// at most the entries its allocation unit pools, which is the derived bound
// below rather than a second hardware parameter.
ResourceStateDeclaration declareQueue(StateKey state, CapacityUnits entries) {
  return ResourceStateDeclaration{
      state,
      {CapacityDimensionDeclaration{
          key<CapacityDimensionKey>(OperandQueueDimension::QueuedOperand),
          entries, CapacityUnits(0)}}};
}

// One `OperandEnqueueService` or `OperandDequeueService` state: one slot per
// allocation unit per PE clock cycle, free after reset.
ResourceStateDeclaration declareService(StateKey state) {
  return ResourceStateDeclaration{
      state,
      {CapacityDimensionDeclaration{
          key<CapacityDimensionKey>(OperandServiceDimension::Slot),
          CapacityUnits(1), CapacityUnits(0)}}};
}

} // namespace

bool fabric::operator==(const LogicalOperandQueueKey &lhs,
                        const LogicalOperandQueueKey &rhs) {
  return lhs.context == rhs.context && lhs.fuOccurrence == rhs.fuOccurrence &&
         lhs.fuInput == rhs.fuInput;
}

bool fabric::operator!=(const LogicalOperandQueueKey &lhs,
                        const LogicalOperandQueueKey &rhs) {
  return !(lhs == rhs);
}

bool fabric::operator<(const LogicalOperandQueueKey &lhs,
                       const LogicalOperandQueueKey &rhs) {
  if (lhs.context.pe.id() != rhs.context.pe.id())
    return lhs.context.pe.id() < rhs.context.pe.id();
  if (lhs.context.ordinal != rhs.context.ordinal)
    return lhs.context.ordinal < rhs.context.ordinal;
  if (lhs.fuOccurrence != rhs.fuOccurrence)
    return lhs.fuOccurrence < rhs.fuOccurrence;
  return lhs.fuInput < rhs.fuInput;
}

llvm::StringRef fabric::getTemporalOperandBufferViolationName(
    TemporalOperandBufferViolation violation) {
  switch (violation) {
  case TemporalOperandBufferViolation::NonPositiveEntryCapacity:
    return "non_positive_entry_capacity";
  case TemporalOperandBufferViolation::EmptyContextDomain:
    return "empty_context_domain";
  case TemporalOperandBufferViolation::LogicalQueueDomainOverflow:
    return "logical_queue_domain_overflow";
  case TemporalOperandBufferViolation::AdmittedCommitSetViolatesInvariant:
    return "admitted_commit_set_violates_invariant";
  }
  llvm_unreachable("unhandled temporal operand buffer violation");
}

char TemporalOperandBufferError::ID = 0;

void TemporalOperandBufferError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code TemporalOperandBufferError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<TemporalOperandBufferContract>
TemporalOperandBufferContract::create(
    const TemporalOperandBufferDeclaration &declaration) {
  if (declaration.entriesPerAllocationUnit == 0)
    return rejected(TemporalOperandBufferViolation::NonPositiveEntryCapacity,
                    "every operand_buffer_mode requires a positive "
                    "operand_buffer_size");
  if (declaration.contextCount == 0)
    return rejected(TemporalOperandBufferViolation::EmptyContextDomain,
                    "num_instruction must admit at least one resident context");

  std::uint64_t ingressCount = 0;
  for (std::uint32_t inputs : declaration.fuInputCounts)
    ingressCount += inputs;
  const std::uint64_t queueCount =
      ingressCount * static_cast<std::uint64_t>(declaration.contextCount);
  if (queueCount > kMaxLogicalQueues)
    return rejected(
        TemporalOperandBufferViolation::LogicalQueueDomainOverflow,
        "the logical operand queue domain exceeds the owner key domain");

  // The complete canonical logical-queue domain, in the lexicographic order of
  // context, concrete FU occurrence, and FU input ordinal.
  std::vector<LogicalOperandQueueKey> queues;
  queues.reserve(queueCount);
  for (std::uint32_t context = 0; context != declaration.contextCount;
       ++context)
    for (std::size_t fu = 0; fu != declaration.fuInputCounts.size(); ++fu)
      for (std::uint32_t input = 0; input != declaration.fuInputCounts[fu];
           ++input)
        queues.push_back(LogicalOperandQueueKey{
            InstructionContextRef{declaration.pe, context},
            static_cast<FabricOrdinal>(fu), static_cast<FabricOrdinal>(input)});

  // The total mechanical projection onto allocation units.
  std::vector<OperandAllocationUnit> units;
  std::vector<std::uint32_t> unitOfQueue(queues.size(), 0);
  switch (declaration.mode) {
  case OperandBufferMode::PerInstruction:
    units.reserve(queues.size());
    for (std::size_t queue = 0; queue != queues.size(); ++queue) {
      unitOfQueue[queue] = static_cast<std::uint32_t>(units.size());
      units.push_back(DedicatedQueueUnit{queues[queue]});
    }
    break;
  case OperandBufferMode::PerInputPort: {
    std::vector<std::uint32_t> bankOfFu(declaration.fuInputCounts.size(), 0);
    for (std::size_t fu = 0; fu != declaration.fuInputCounts.size(); ++fu) {
      bankOfFu[fu] = static_cast<std::uint32_t>(units.size());
      for (std::uint32_t input = 0; input != declaration.fuInputCounts[fu];
           ++input)
        units.push_back(FuInputUnit{static_cast<FabricOrdinal>(fu),
                                    static_cast<FabricOrdinal>(input)});
    }
    for (std::size_t queue = 0; queue != queues.size(); ++queue)
      unitOfQueue[queue] = bankOfFu[queues[queue].fuOccurrence] +
                           static_cast<std::uint32_t>(queues[queue].fuInput);
    break;
  }
  case OperandBufferMode::AllFuShare:
    if (!queues.empty())
      units.push_back(WholeTemporalPeUnit{declaration.pe});
    break;
  }

  const std::uint32_t unitCount = static_cast<std::uint32_t>(units.size());
  const std::uint32_t queueTotal = static_cast<std::uint32_t>(queues.size());

  // The queues each allocation unit pools, in canonical order. This is the
  // cycle a contended service filters the canonical requester order to.
  std::vector<Span> unitSpans(unitCount);
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue)
    ++unitSpans[unitOfQueue[queue]].count;
  std::uint32_t offset = 0;
  for (Span &span : unitSpans) {
    span.first = offset;
    offset += span.count;
  }
  std::vector<std::uint32_t> unitQueues(queueTotal, 0);
  std::vector<std::uint32_t> filled(unitCount, 0);
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue) {
    const std::uint32_t unit = unitOfQueue[queue];
    unitQueues[unitSpans[unit].first + filled[unit]++] = queue;
  }

  const CapacityUnits entries(declaration.entriesPerAllocationUnit);
  const KeyLayout layout{unitCount, queueTotal};

  ResourceContractDeclaration contract;
  contract.states.reserve(3 * static_cast<std::size_t>(unitCount) + queueTotal);
  for (std::uint32_t unit = 0; unit != unitCount; ++unit)
    contract.states.push_back(
        declareEntryPool(layout.entryPool(unit), entries));
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue)
    contract.states.push_back(declareQueue(layout.queue(queue), entries));
  for (std::uint32_t unit = 0; unit != unitCount; ++unit)
    contract.states.push_back(declareService(layout.enqueueService(unit)));
  for (std::uint32_t unit = 0; unit != unitCount; ++unit)
    contract.states.push_back(declareService(layout.dequeueService(unit)));

  // One append and one remove transition per logical queue.
  contract.resourceTransitions.reserve(2 *
                                       static_cast<std::size_t>(queueTotal));
  for (std::uint32_t transition = 0; transition != 2 * queueTotal; ++transition)
    contract.resourceTransitions.push_back(ResourceTransitionKey(transition));

  contract.requesters.reserve(queueTotal);
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue)
    contract.requesters.push_back(RequesterKey(queue));
  contract.eligibilityCount = 2;
  contract.eventCount = 3;

  // Both commits happen within the cycle the claim is acquired, and the claim
  // returns at the next PE clock boundary.
  contract.timingContracts = {TimingContractDeclaration{
      key<TimingContractKey>(
          OperandBufferTiming::ServiceClaimUntilNextPeClockBoundary),
      {0, 0, 1}}};

  const EventKey boundary =
      key<EventKey>(OperandBufferEvent::NextPeClockBoundary);
  const TimingContractKey timing = key<TimingContractKey>(
      OperandBufferTiming::ServiceClaimUntilNextPeClockBoundary);

  const auto servicePattern =
      [&](UsePatternKey patternKey, std::uint32_t queue, StateKey service,
          OperandBufferEligibility eligibility, OperandBufferEvent commitEvent,
          ResourceTransitionKey transition) {
        return UsePatternDeclaration{
            patternKey,
            RequesterKey(queue),
            key<EligibilityKey>(eligibility),
            key<EventKey>(commitEvent),
            boundary,
            CommitDeclaration{key<EventKey>(commitEvent), transition},
            timing,
            {ClaimDeclaration{
                ClaimKey(0), service,
                key<CapacityDimensionKey>(OperandServiceDimension::Slot),
                CapacityUnits(1)}},
            {}};
      };

  contract.usePatterns.reserve(2 * static_cast<std::size_t>(queueTotal));
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue)
    contract.usePatterns.push_back(servicePattern(
        layout.enqueue(queue), queue, layout.enqueueService(unitOfQueue[queue]),
        OperandBufferEligibility::CycleStartFreeEntry,
        OperandBufferEvent::EnqueueCommit, layout.append(queue)));
  for (std::uint32_t queue = 0; queue != queueTotal; ++queue)
    contract.usePatterns.push_back(servicePattern(
        layout.dequeue(queue), queue, layout.dequeueService(unitOfQueue[queue]),
        OperandBufferEligibility::CycleStartHeadPresent,
        OperandBufferEvent::DequeueCommit, layout.remove(queue)));

  // Two logical queues can request one service exactly when they pool one
  // allocation unit. The declared cycle is the canonical requester order; a
  // service filters it to its own unit, and its running cursor is caller-owned,
  // so enqueue and dequeue advance independently.
  bool contended = false;
  for (const Span &span : unitSpans)
    contended = contended || span.count > 1;
  if (contended) {
    contract.grantPolicy = GrantPolicyDeclaration(
        RoundRobinDeclaration{contract.requesters, RequesterKey(0)});
  }

  llvm::Expected<ResourceContract> resourceContract =
      ResourceContract::create(contract);
  if (!resourceContract)
    return resourceContract.takeError();

  TemporalOperandBufferContract derived(std::move(*resourceContract));
  derived.queues_ = std::move(queues);
  derived.units_ = std::move(units);
  derived.unitOfQueue_ = std::move(unitOfQueue);
  derived.unitQueues_ = std::move(unitQueues);
  derived.unitSpans_ = std::move(unitSpans);
  derived.entryCapacity_ = entries;
  derived.mode_ = declaration.mode;
  derived.admissionPolicy_ =
      unitCount != queueTotal ? OperandAdmissionPolicy::PerActiveQueueCredit
                              : OperandAdmissionPolicy::Unreserved;

  // Every admitted concurrent commit set must leave the pool inside its
  // declared bounds. `O - D + E` is linear in `O` with `D` and `E` in `{0, 1}`,
  // so the only occupancies that can break `0 <= O - D + E <= capacity` are the
  // two extremes and their neighbours; checking those with all four selections
  // is a complete case analysis rather than a sample. The one comparison covers
  // both directions, because an underflowed unsigned result also exceeds
  // capacity.
  const std::uint32_t capacity = entries.value();
  const std::uint32_t probes[] = {0, 1, capacity - 1, capacity};
  for (std::uint32_t occupancy : probes)
    for (bool dequeue : {false, true})
      for (bool enqueue : {false, true}) {
        const OperandCommitSelection selection{dequeue, enqueue};
        if (!derived.admits(CapacityUnits(occupancy), selection))
          continue;
        const std::uint32_t next =
            derived.occupancyAfter(CapacityUnits(occupancy), selection).value();
        if (next > capacity)
          return rejected(
              TemporalOperandBufferViolation::
                  AdmittedCommitSetViolatesInvariant,
              "an admitted commit set would overfill an allocation unit");
      }

  return derived;
}

const OperandAllocationUnit &
TemporalOperandBufferContract::allocationUnit(std::uint32_t unit) const {
  assert(unit < units_.size() && "undeclared allocation unit");
  return units_[unit];
}

std::uint32_t
TemporalOperandBufferContract::allocationUnitOf(std::uint32_t queue) const {
  assert(queue < unitOfQueue_.size() && "undeclared logical queue");
  return unitOfQueue_[queue];
}

llvm::ArrayRef<std::uint32_t>
TemporalOperandBufferContract::queuesOf(std::uint32_t unit) const {
  assert(unit < unitSpans_.size() && "undeclared allocation unit");
  const Span span = unitSpans_[unit];
  return llvm::ArrayRef<std::uint32_t>(unitQueues_)
      .slice(span.first, span.count);
}

StateKey
TemporalOperandBufferContract::entryPoolState(std::uint32_t unit) const {
  assert(unit < units_.size() && "undeclared allocation unit");
  return layoutOf(*this).entryPool(unit);
}

StateKey TemporalOperandBufferContract::queueState(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return layoutOf(*this).queue(queue);
}

StateKey
TemporalOperandBufferContract::enqueueServiceState(std::uint32_t unit) const {
  assert(unit < units_.size() && "undeclared allocation unit");
  return layoutOf(*this).enqueueService(unit);
}

StateKey
TemporalOperandBufferContract::dequeueServiceState(std::uint32_t unit) const {
  assert(unit < units_.size() && "undeclared allocation unit");
  return layoutOf(*this).dequeueService(unit);
}

ResourceTransitionKey
TemporalOperandBufferContract::appendTransition(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return layoutOf(*this).append(queue);
}

ResourceTransitionKey
TemporalOperandBufferContract::removeTransition(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return layoutOf(*this).remove(queue);
}

RequesterKey
TemporalOperandBufferContract::requester(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return RequesterKey(queue);
}

UsePatternKey
TemporalOperandBufferContract::enqueuePattern(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return layoutOf(*this).enqueue(queue);
}

UsePatternKey
TemporalOperandBufferContract::dequeuePattern(std::uint32_t queue) const {
  assert(queue < queues_.size() && "undeclared logical queue");
  return layoutOf(*this).dequeue(queue);
}

bool TemporalOperandBufferContract::admits(
    CapacityUnits cycleStartOccupancy, OperandCommitSelection selection) const {
  if (cycleStartOccupancy > entryCapacity_)
    return false;
  const std::uint32_t occupancy = cycleStartOccupancy.value();
  const std::uint32_t dequeues = selection.dequeue ? 1 : 0;
  if (dequeues > occupancy)
    return false;
  if (selection.enqueue && occupancy >= entryCapacity_.value())
    return false;
  return true;
}

CapacityUnits TemporalOperandBufferContract::occupancyAfter(
    CapacityUnits cycleStartOccupancy, OperandCommitSelection selection) const {
  assert(admits(cycleStartOccupancy, selection) &&
         "the commit set is not admitted at this cycle-start occupancy");
  return CapacityUnits(cycleStartOccupancy.value() -
                       (selection.dequeue ? 1 : 0) +
                       (selection.enqueue ? 1 : 0));
}

bool TemporalOperandBufferContract::admitsActorDequeueSet(
    llvm::ArrayRef<std::uint32_t> queues) const {
  std::vector<bool> served(units_.size(), false);
  for (std::uint32_t queue : queues) {
    if (queue >= unitOfQueue_.size())
      return false;
    const std::uint32_t unit = unitOfQueue_[queue];
    if (served[unit])
      return false;
    served[unit] = true;
  }
  return true;
}

bool TemporalOperandBufferContract::admitsIngressEnqueueSet(
    llvm::ArrayRef<std::uint32_t> queues) const {
  std::vector<bool> served(units_.size(), false);
  for (std::uint32_t queue : queues) {
    if (queue >= unitOfQueue_.size())
      return false;
    const std::uint32_t unit = unitOfQueue_[queue];
    if (served[unit])
      return false;
    served[unit] = true;
  }
  return true;
}
