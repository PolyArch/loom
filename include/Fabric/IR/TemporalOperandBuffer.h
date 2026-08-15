#ifndef FABRIC_IR_TEMPORALOPERANDBUFFER_H
#define FABRIC_IR_TEMPORALOPERANDBUFFER_H

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace fabric {

/// The temporal-PE operand-buffer physical organization. The enumerators live
/// with the dialect attribute; only the closed type is needed here.
enum class OperandBufferMode : std::uint32_t;

/// One potential logical operand queue of one temporal PE. Fabric owns this key
/// alone: the resident context is the one canonical
/// `loom::fabric::InstructionContextRef`, and the other two components are the
/// concrete FU occurrence ordinal and one FU input ordinal of that occurrence.
/// The key is owner-local, so it is neither a standalone entity nor a
/// persistent reference, and it never becomes a second context identity.
struct LogicalOperandQueueKey {
  loom::fabric::InstructionContextRef context;
  loom::fabric::FabricOrdinal fuOccurrence = 0;
  loom::fabric::FabricOrdinal fuInput = 0;
};

bool operator==(const LogicalOperandQueueKey &lhs,
                const LogicalOperandQueueKey &rhs);
bool operator!=(const LogicalOperandQueueKey &lhs,
                const LogicalOperandQueueKey &rhs);

/// The canonical order of the logical-queue domain: `InstructionContextRef`
/// first, then concrete FU occurrence, then FU input ordinal.
bool operator<(const LogicalOperandQueueKey &lhs,
               const LogicalOperandQueueKey &rhs);

/// `per_instruction`: the whole logical key is preserved, so the queue has
/// dedicated storage.
struct DedicatedQueueUnit {
  LogicalOperandQueueKey queue;
};

/// `per_input_port`: the key is projected onto one FU ingress bank, so every
/// resident context sharing that bank shares one entry pool.
struct FuInputUnit {
  loom::fabric::FabricOrdinal fuOccurrence = 0;
  loom::fabric::FabricOrdinal fuInput = 0;
};

/// `all_fu_share`: the key is projected onto the temporal PE, so one entry pool
/// serves every logical queue.
struct WholeTemporalPeUnit {
  loom::fabric::FabricPeOccurrenceRef pe;
};

/// The mode-derived allocation unit of one logical operand queue. The three
/// constructors are the total mechanical projection of `operand_buffer_mode`;
/// they are not a Mapping record, a backend choice, or an extension point.
using OperandAllocationUnit =
    std::variant<DedicatedQueueUnit, FuInputUnit, WholeTemporalPeUnit>;

/// The one capacity dimension of `OperandEntryPool`: the entries of one
/// allocation unit's pool. Occupancy is durable operand-buffer state that only
/// a committed transition changes; no use pattern claims it.
enum class OperandEntryPoolDimension : std::uint32_t { OccupiedEntry };

/// The one capacity dimension of `OperandQueue`: the operands one logical queue
/// holds in arrival order. Every allocation unit keeps this state per logical
/// queue, so a shared pool never merges contexts, tags, or streams into a
/// global arrival-order head. Its contents are durable state, not a claim.
enum class OperandQueueDimension : std::uint32_t { QueuedOperand };

/// The one capacity dimension shared by `OperandEnqueueService` and
/// `OperandDequeueService`: one service slot per allocation unit per PE clock
/// cycle. This is the only capacity a use pattern claims.
enum class OperandServiceDimension : std::uint32_t { Slot };

/// The closed eligibility domain of the two patterns.
enum class OperandBufferEligibility : std::uint32_t {
  /// The logical queue holds a head token that was present at cycle start, so
  /// an operand appended in this cycle can never satisfy this dequeue.
  CycleStartHeadPresent,
  /// The allocation unit has a free entry at the start of this PE clock cycle,
  /// that is `O < capacity` independently of this cycle's dequeue.
  CycleStartFreeEntry,
};

/// The closed acquire, commit, and release event domain. `NextPeClockBoundary`
/// is mechanically the next rising edge of the exact Clock domain that contains
/// the PE; that membership stays owned by the Fabric root's hardware-domain
/// relation and is not restated here.
enum class OperandBufferEvent : std::uint32_t {
  EnqueueCommit,
  DequeueCommit,
  NextPeClockBoundary,
};

/// The one timing-and-progress contract: a service slot is claimed at the same
/// atomic event that commits the queue transition and returns at the next PE
/// clock boundary.
enum class OperandBufferTiming : std::uint32_t {
  ServiceClaimUntilNextPeClockBoundary,
};

/// What one temporal PE declares about its operand buffering.
/// `contextCount` is `num_instruction`, `fuInputCounts` is the FU input count
/// of every concrete FU occurrence in canonical occurrence order, and
/// `entriesPerAllocationUnit` is the required positive `operand_buffer_size`.
/// No field carries a default meaning: a zero entry count is rejected in every
/// mode, so no implicit dedicated depth and no builder or backend default can
/// exist.
struct TemporalOperandBufferDeclaration {
  loom::fabric::FabricPeOccurrenceRef pe;
  std::uint32_t contextCount = 0;
  llvm::ArrayRef<std::uint32_t> fuInputCounts;
  OperandBufferMode mode{};
  std::uint32_t entriesPerAllocationUnit = 0;
};

/// Typed rejection of one operand-buffer declaration.
enum class TemporalOperandBufferViolation : std::uint32_t {
  /// `operand_buffer_size` is absent or zero. Every mode requires a positive
  /// value; depths 1 and 2 are different hardware.
  NonPositiveEntryCapacity,
  /// `num_instruction` must admit at least one resident context.
  EmptyContextDomain,
  /// The derived logical-queue domain does not fit the owner-local key domain.
  LogicalQueueDomainOverflow,
  /// An admitted concurrent commit set would leave the queue or pool state
  /// outside its declared bounds.
  AdmittedCommitSetViolatesInvariant,
};

llvm::StringRef
getTemporalOperandBufferViolationName(TemporalOperandBufferViolation violation);

class TemporalOperandBufferError final
    : public llvm::ErrorInfo<TemporalOperandBufferError> {
public:
  static char ID;

  TemporalOperandBufferError(TemporalOperandBufferViolation violation,
                             std::string message)
      : violation_(violation), message_(std::move(message)) {}

  TemporalOperandBufferViolation violation() const { return violation_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  TemporalOperandBufferViolation violation_;
  std::string message_;
};

/// The commit set one allocation unit may select in one PE clock cycle. Version
/// 1.0 admits at most one dequeue and at most one enqueue, matching the two
/// one-slot services.
struct OperandCommitSelection {
  bool dequeue = false;
  bool enqueue = false;
};

/// The complete operand-buffer contract of one temporal PE: the canonical
/// logical-queue domain, the mode-derived allocation units, and the exact
/// `ResourceContract` those two induce.
///
/// Per allocation unit the derived contract declares one `OperandEntryPool` of
/// `operand_buffer_size` entries, one `OperandEnqueueService` slot, and one
/// `OperandDequeueService` slot; per logical queue it declares one
/// `OperandQueue` and the two transitions `AppendOperand` and `RemoveOperand`.
/// Its two atomic patterns are exactly:
///
///   * `Enqueue` claims one enqueue service slot at `EnqueueCommit`, commits
///     `AppendOperand` at that same atomic event, and releases the slot at
///     `NextPeClockBoundary`.
///   * `Dequeue` claims one dequeue service slot at `DequeueCommit`, commits
///     `RemoveOperand` at that same atomic event, and releases the slot at
///     `NextPeClockBoundary`.
///
/// Queue contents, head and tail positions, and pool occupancy are durable
/// state that only those transitions change. Nothing claims an entry, so no use
/// releases or inherits another use's claim.
///
/// Because enqueue and dequeue claim disjoint services, both may commit on one
/// allocation unit in one cycle. When two logical queues project to one unit
/// the contract carries the shared `RoundRobin` GrantPolicy over the canonical
/// logical-queue order, reset to the first canonical requester; a service
/// filters that cycle to `queuesOf` its unit through eligibility, and its
/// running cursor is caller-owned execution state, so enqueue and dequeue
/// arbitrate independently. Dedicated queues prove at most one requester per
/// capacity dimension and therefore carry no policy at all.
class TemporalOperandBufferContract {
public:
  /// Validates one declaration and either returns the contract or the first
  /// violation in the order the violation enumeration lists.
  static llvm::Expected<TemporalOperandBufferContract>
  create(const TemporalOperandBufferDeclaration &declaration);

  OperandBufferMode mode() const { return mode_; }
  CapacityUnits entriesPerAllocationUnit() const { return entryCapacity_; }

  /// The canonical logical-queue domain, in canonical order. A configured view
  /// makes only its selected queues eligible; every other key stays empty.
  llvm::ArrayRef<LogicalOperandQueueKey> logicalQueues() const {
    return queues_;
  }

  std::uint32_t allocationUnitCount() const {
    return static_cast<std::uint32_t>(units_.size());
  }
  const OperandAllocationUnit &allocationUnit(std::uint32_t unit) const;

  /// The allocation unit the mode projects one logical queue onto.
  std::uint32_t allocationUnitOf(std::uint32_t queue) const;

  /// The logical queues that project to one allocation unit, in canonical
  /// order. This is exactly the requester cycle a contended service filters to.
  llvm::ArrayRef<std::uint32_t> queuesOf(std::uint32_t unit) const;

  const ResourceContract &resourceContract() const { return contract_; }

  StateKey entryPoolState(std::uint32_t unit) const;
  StateKey queueState(std::uint32_t queue) const;
  StateKey enqueueServiceState(std::uint32_t unit) const;
  StateKey dequeueServiceState(std::uint32_t unit) const;

  ResourceTransitionKey appendTransition(std::uint32_t queue) const;
  ResourceTransitionKey removeTransition(std::uint32_t queue) const;

  RequesterKey requester(std::uint32_t queue) const;
  UsePatternKey enqueuePattern(std::uint32_t queue) const;
  UsePatternKey dequeuePattern(std::uint32_t queue) const;

  /// Whether `selection` is admitted on one allocation unit whose occupancy at
  /// the start of the PE clock cycle is `cycleStartOccupancy`. A dequeue
  /// observes only a token present at cycle start, and an enqueue observes only
  /// capacity free at cycle start. A full unit therefore rejects an enqueue
  /// even when a dequeue commits in the same cycle.
  bool admits(CapacityUnits cycleStartOccupancy,
              OperandCommitSelection selection) const;

  /// The occupancy `O - D + E` that an admitted `selection` establishes for the
  /// next PE clock cycle.
  CapacityUnits occupancyAfter(CapacityUnits cycleStartOccupancy,
                               OperandCommitSelection selection) const;

  /// Whether every head removal one Canonical Dataflow actor transition
  /// requires fits the one-dequeue service of its allocation unit. Two required
  /// heads that project to one unit make the binding invalid; no implementation
  /// may serialize the removals privately after consuming part of the inputs.
  bool admitsActorDequeueSet(llvm::ArrayRef<std::uint32_t> queues) const;

  /// Whether one ingress token may atomically enqueue every matching logical
  /// queue. Version 1 has one enqueue service per allocation unit, so a match
  /// group may contain at most one queue from each unit. Atomic fanout across
  /// distinct units remains admitted.
  bool admitsIngressEnqueueSet(llvm::ArrayRef<std::uint32_t> queues) const;

private:
  struct Span {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
  };

  explicit TemporalOperandBufferContract(ResourceContract contract)
      : contract_(std::move(contract)) {}

  ResourceContract contract_;
  std::vector<LogicalOperandQueueKey> queues_;
  std::vector<OperandAllocationUnit> units_;
  std::vector<std::uint32_t> unitOfQueue_;
  std::vector<std::uint32_t> unitQueues_;
  std::vector<Span> unitSpans_;
  CapacityUnits entryCapacity_{0};
  OperandBufferMode mode_{};
};

} // namespace fabric

#endif // FABRIC_IR_TEMPORALOPERANDBUFFER_H
