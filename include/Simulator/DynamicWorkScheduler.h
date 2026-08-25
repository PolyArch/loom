#ifndef LOOM_SIMULATOR_DYNAMICWORKSCHEDULER_H
#define LOOM_SIMULATOR_DYNAMICWORKSCHEDULER_H

#include "Simulator/DynamicWorkDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace loom::sim {

enum class DynamicWorkScheduleActionKind : std::uint8_t {
  AdmitRoot,
  AcquireLocal,
  Steal,
  PublishChild,
  RequestCancellation,
  CancelQueued,
  CancelActive,
  Complete,
};

/// One scheduler transition in canonical replay order. Worker ordinals are
/// transient assignments; `item` remains the sole logical work identity.
struct DynamicWorkScheduleAction final {
  DynamicWorkScheduleActionKind kind = DynamicWorkScheduleActionKind::AdmitRoot;
  WorkItemId item;
  std::optional<std::uint32_t> sourceWorker;
  std::optional<std::uint32_t> targetWorker;
};

class DynamicWorkAssignment final {
public:
  DynamicWorkAssignment(const DynamicWorkAssignment &) = delete;
  DynamicWorkAssignment &operator=(const DynamicWorkAssignment &) = delete;
  DynamicWorkAssignment(DynamicWorkAssignment &&) noexcept = default;
  DynamicWorkAssignment &operator=(DynamicWorkAssignment &&) = delete;

  WorkItemId id() const { return id_; }
  std::uint32_t workerOrdinal() const { return workerOrdinal_; }
  llvm::ArrayRef<std::uint8_t> payload() const { return payload_; }

private:
  friend class DynamicWorkScheduler;

  DynamicWorkAssignment(WorkItemId id, std::uint32_t workerOrdinal,
                        std::vector<std::uint8_t> payload,
                        const std::shared_ptr<const void> &ownerIdentity)
      : id_(std::move(id)), workerOrdinal_(workerOrdinal),
        payload_(std::move(payload)), ownerIdentity_(ownerIdentity) {}

  WorkItemId id_;
  std::uint32_t workerOrdinal_ = 0;
  std::vector<std::uint8_t> payload_;
  std::shared_ptr<const void> ownerIdentity_;
};

enum class DynamicWorkPublishKind : std::uint8_t {
  Published,
  WouldBlock,
  CancellationRequested,
};

struct DynamicWorkPublishResult final {
  DynamicWorkPublishKind kind = DynamicWorkPublishKind::WouldBlock;
  std::vector<WorkItemId> children;
};

enum class DynamicWorkCancellationKind : std::uint8_t {
  CancelledQueued,
  RequestedActive,
  AlreadyRequested,
};

/// A queued cancellation always carries the exact domain retirement effect;
/// active request outcomes carry none. Only the scheduler constructs results.
class DynamicWorkCancellationResult final {
public:
  DynamicWorkCancellationKind kind() const { return kind_; }
  std::optional<RetirementEffect> retirementEffect() const {
    return retirementEffect_;
  }

private:
  friend class DynamicWorkScheduler;

  DynamicWorkCancellationResult(
      DynamicWorkCancellationKind kind,
      std::optional<RetirementEffect> retirementEffect)
      : kind_(kind), retirementEffect_(retirementEffect) {}

  DynamicWorkCancellationKind kind_;
  std::optional<RetirementEffect> retirementEffect_;
};

class DynamicWorkSchedulerError final
    : public llvm::ErrorInfo<DynamicWorkSchedulerError> {
public:
  enum class Kind {
    InvalidConfiguration,
    InvalidWorker,
    WorkerBusy,
    InvalidAssignment,
    UnknownItem,
    CancellationPending,
    CancellationNotRequested,
  };

  static char ID;

  DynamicWorkSchedulerError(Kind kind, std::string message);

  Kind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// A bounded execution-local work-stealing scheduler. DynamicWorkDomain owns
/// active responsibility and completion; this class owns only queue placement,
/// transient worker assignment, cancellation delivery, and replay order.
///
/// All public transitions are serialized by one scheduler mutex. Publication
/// before unlock and acquisition after lock establish the required
/// release/acquire visibility for payloads and responsibility state. A worker
/// pops the back of its local deque; an idle worker steals the front of the
/// first nonempty victim in cyclic ordinal order.
///
/// A successful assignment is a linear obligation. Its owner must consume it
/// through complete or, after a cancellation request, cancel. Destroying the
/// handle does not retire the domain responsibility.
class DynamicWorkScheduler final {
public:
  static llvm::Expected<std::unique_ptr<DynamicWorkScheduler>>
  create(ThreadDispatchOccurrenceId dispatchOccurrence,
         std::uint32_t workerCount, std::size_t queueCapacityPerWorker,
         llvm::ArrayRef<std::uint8_t> rootPayload);

  DynamicWorkScheduler(const DynamicWorkScheduler &) = delete;
  DynamicWorkScheduler &operator=(const DynamicWorkScheduler &) = delete;
  DynamicWorkScheduler(DynamicWorkScheduler &&) = delete;
  DynamicWorkScheduler &operator=(DynamicWorkScheduler &&) = delete;

  llvm::Expected<std::optional<DynamicWorkAssignment>>
  acquire(std::uint32_t workerOrdinal);

  llvm::Expected<DynamicWorkPublishResult>
  publishChild(const DynamicWorkAssignment &parent,
               llvm::ArrayRef<std::uint8_t> payload);

  /// Atomically publishes one ordered group. Capacity or cancellation refusal
  /// publishes none of the group and consumes no child ordinal.
  llvm::Expected<DynamicWorkPublishResult> publishChildren(
      const DynamicWorkAssignment &parent,
      llvm::ArrayRef<std::vector<std::uint8_t>> payloads);

  llvm::Expected<DynamicWorkCancellationResult>
  requestCancellation(const WorkItemId &item);

  llvm::Expected<bool>
  cancellationRequested(const DynamicWorkAssignment &assignment) const;

  llvm::Expected<RetirementEffect> complete(DynamicWorkAssignment &&assignment);
  llvm::Expected<RetirementEffect> cancel(DynamicWorkAssignment &&assignment);

  /// Cancels every queued responsibility in worker/front order. Active
  /// assignments remain owned by their workers.
  llvm::Expected<std::optional<RetirementEffect>> cancelQueued();

  std::size_t activeCount() const;
  std::size_t queuedCount() const;
  bool completed() const;
  std::vector<DynamicWorkScheduleAction> replay() const;

private:
  enum class ItemState : std::uint8_t { Queued, Active };

  struct Item final {
    std::unique_ptr<WorkResponsibility> responsibility;
    ItemState state = ItemState::Queued;
    std::uint32_t workerOrdinal = 0;
    bool cancellationRequested = false;
    std::vector<std::uint8_t> payload;
  };

  DynamicWorkScheduler(ThreadDispatchOccurrenceId dispatchOccurrence,
                       std::uint32_t workerCount,
                       std::size_t queueCapacityPerWorker);

  llvm::Error validateWorker(std::uint32_t workerOrdinal) const;
  llvm::Error validateAssignment(const DynamicWorkAssignment &assignment) const;
  llvm::Expected<RetirementEffect>
  retireAssignment(DynamicWorkAssignment &&assignment,
                   DynamicWorkScheduleActionKind action);
  void record(DynamicWorkScheduleActionKind kind, const WorkItemId &item,
              std::optional<std::uint32_t> sourceWorker,
              std::optional<std::uint32_t> targetWorker);

  mutable std::mutex mutex_;
  DynamicWorkDomain domain_;
  std::size_t queueCapacityPerWorker_ = 0;
  std::shared_ptr<const void> ownerIdentity_;
  std::vector<std::deque<WorkItemId>> queues_;
  std::map<WorkItemId, Item> items_;
  std::vector<DynamicWorkScheduleAction> replay_;
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_DYNAMICWORKSCHEDULER_H
