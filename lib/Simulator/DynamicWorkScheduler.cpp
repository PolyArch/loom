#include "Simulator/DynamicWorkScheduler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

namespace loom::sim {
namespace {

llvm::Error reject(DynamicWorkSchedulerError::Kind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<DynamicWorkSchedulerError>(kind, message.str());
}

} // namespace

char DynamicWorkSchedulerError::ID = 0;

DynamicWorkSchedulerError::DynamicWorkSchedulerError(Kind kind,
                                                     std::string message)
    : kind_(kind), message_(std::move(message)) {}

void DynamicWorkSchedulerError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code DynamicWorkSchedulerError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

DynamicWorkScheduler::DynamicWorkScheduler(
    ThreadDispatchOccurrenceId dispatchOccurrence, std::uint32_t workerCount,
    std::size_t queueCapacityPerWorker)
    : domain_(dispatchOccurrence),
      queueCapacityPerWorker_(queueCapacityPerWorker),
      ownerIdentity_(std::make_shared<const std::uint8_t>(0)),
      queues_(workerCount) {}

llvm::Expected<std::unique_ptr<DynamicWorkScheduler>>
DynamicWorkScheduler::create(ThreadDispatchOccurrenceId dispatchOccurrence,
                             std::uint32_t workerCount,
                             std::size_t queueCapacityPerWorker,
                             llvm::ArrayRef<std::uint8_t> rootPayload) {
  if (workerCount == 0)
    return reject(DynamicWorkSchedulerError::Kind::InvalidConfiguration,
                  "dynamic-work scheduler requires at least one worker");
  if (queueCapacityPerWorker == 0)
    return reject(DynamicWorkSchedulerError::Kind::InvalidConfiguration,
                  "dynamic-work scheduler queue capacity must be positive");

  auto scheduler =
      std::unique_ptr<DynamicWorkScheduler>(new DynamicWorkScheduler(
          dispatchOccurrence, workerCount, queueCapacityPerWorker));
  auto root = scheduler->domain_.admitRoot();
  if (!root)
    return root.takeError();
  const WorkItemId rootId = root->id();
  Item rootItem;
  rootItem.responsibility =
      std::make_unique<WorkResponsibility>(std::move(*root));
  rootItem.payload.assign(rootPayload.begin(), rootPayload.end());
  scheduler->items_.emplace(rootId, std::move(rootItem));
  scheduler->queues_.front().push_back(rootId);
  scheduler->record(DynamicWorkScheduleActionKind::AdmitRoot, rootId,
                    std::nullopt, 0);
  return scheduler;
}

llvm::Error
DynamicWorkScheduler::validateWorker(std::uint32_t workerOrdinal) const {
  if (workerOrdinal >= queues_.size())
    return reject(DynamicWorkSchedulerError::Kind::InvalidWorker,
                  "dynamic-work scheduler names an unknown worker");
  return llvm::Error::success();
}

llvm::Error DynamicWorkScheduler::validateAssignment(
    const DynamicWorkAssignment &assignment) const {
  const auto item = items_.find(assignment.id_);
  if (assignment.ownerIdentity_ != ownerIdentity_ || item == items_.end() ||
      item->second.state != ItemState::Active ||
      item->second.workerOrdinal != assignment.workerOrdinal_)
    return reject(DynamicWorkSchedulerError::Kind::InvalidAssignment,
                  "dynamic-work assignment is not the live worker transfer");
  return llvm::Error::success();
}

void DynamicWorkScheduler::record(DynamicWorkScheduleActionKind kind,
                                  const WorkItemId &item,
                                  std::optional<std::uint32_t> sourceWorker,
                                  std::optional<std::uint32_t> targetWorker) {
  replay_.push_back(
      DynamicWorkScheduleAction{kind, item, sourceWorker, targetWorker});
}

llvm::Expected<std::optional<DynamicWorkAssignment>>
DynamicWorkScheduler::acquire(std::uint32_t workerOrdinal) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (llvm::Error error = validateWorker(workerOrdinal))
    return std::move(error);
  if (llvm::any_of(items_, [workerOrdinal](const auto &entry) {
        return entry.second.state == ItemState::Active &&
               entry.second.workerOrdinal == workerOrdinal;
      }))
    return reject(DynamicWorkSchedulerError::Kind::WorkerBusy,
                  "dynamic-work worker already owns an active assignment");

  std::uint32_t sourceWorker = workerOrdinal;
  DynamicWorkScheduleActionKind action =
      DynamicWorkScheduleActionKind::AcquireLocal;
  if (queues_[workerOrdinal].empty()) {
    bool found = false;
    for (std::size_t offset = 1; offset < queues_.size(); ++offset) {
      const std::uint32_t victim = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(workerOrdinal) +
           static_cast<std::uint64_t>(offset)) %
          static_cast<std::uint64_t>(queues_.size()));
      if (!queues_[victim].empty()) {
        sourceWorker = victim;
        action = DynamicWorkScheduleActionKind::Steal;
        found = true;
        break;
      }
    }
    if (!found)
      return std::optional<DynamicWorkAssignment>{};
  }

  WorkItemId id = action == DynamicWorkScheduleActionKind::AcquireLocal
                      ? queues_[sourceWorker].back()
                      : queues_[sourceWorker].front();
  if (action == DynamicWorkScheduleActionKind::AcquireLocal)
    queues_[sourceWorker].pop_back();
  else
    queues_[sourceWorker].pop_front();

  auto item = items_.find(id);
  if (item == items_.end() || item->second.state != ItemState::Queued)
    llvm::report_fatal_error(
        "DynamicWorkScheduler invariant failure: queue item is not queued");
  Item &state = item->second;
  state.state = ItemState::Active;
  state.workerOrdinal = workerOrdinal;
  std::vector<std::uint8_t> payload = std::move(state.payload);
  record(action, id, sourceWorker, workerOrdinal);
  return std::optional<DynamicWorkAssignment>(DynamicWorkAssignment(
      id, workerOrdinal, std::move(payload), ownerIdentity_));
}

llvm::Expected<DynamicWorkPublishResult>
DynamicWorkScheduler::publishChild(const DynamicWorkAssignment &parent,
                                   llvm::ArrayRef<std::uint8_t> payload) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (llvm::Error error = validateAssignment(parent))
    return std::move(error);
  Item &parentState = items_.find(parent.id_)->second;
  if (parentState.cancellationRequested)
    return DynamicWorkPublishResult{
        DynamicWorkPublishKind::CancellationRequested, std::nullopt};
  auto &queue = queues_[parent.workerOrdinal_];
  if (queue.size() >= queueCapacityPerWorker_)
    return DynamicWorkPublishResult{DynamicWorkPublishKind::WouldBlock,
                                    std::nullopt};

  auto child = domain_.spawnChild(*parentState.responsibility);
  if (!child)
    return child.takeError();
  WorkItemId childId = child->id();
  Item childState;
  childState.responsibility =
      std::make_unique<WorkResponsibility>(std::move(*child));
  childState.workerOrdinal = parent.workerOrdinal_;
  childState.payload.assign(payload.begin(), payload.end());
  if (!items_.emplace(childId, std::move(childState)).second)
    llvm::report_fatal_error(
        "DynamicWorkScheduler invariant failure: duplicate child identity");
  queue.push_back(childId);
  record(DynamicWorkScheduleActionKind::PublishChild, childId,
         parent.workerOrdinal_, parent.workerOrdinal_);
  return DynamicWorkPublishResult{DynamicWorkPublishKind::Published, childId};
}

llvm::Expected<DynamicWorkCancellationResult>
DynamicWorkScheduler::requestCancellation(const WorkItemId &item) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = items_.find(item);
  if (found == items_.end())
    return reject(DynamicWorkSchedulerError::Kind::UnknownItem,
                  "dynamic-work cancellation names an unknown item");
  Item &state = found->second;
  if (state.state == ItemState::Active) {
    if (state.cancellationRequested)
      return DynamicWorkCancellationResult{
          DynamicWorkCancellationKind::AlreadyRequested, std::nullopt};
    state.cancellationRequested = true;
    record(DynamicWorkScheduleActionKind::RequestCancellation, item,
           state.workerOrdinal, state.workerOrdinal);
    return DynamicWorkCancellationResult{
        DynamicWorkCancellationKind::RequestedActive, std::nullopt};
  }

  auto &queue = queues_[state.workerOrdinal];
  const auto queued = llvm::find(queue, item);
  if (queued == queue.end())
    llvm::report_fatal_error(
        "DynamicWorkScheduler invariant failure: queued item is absent");
  const std::uint32_t workerOrdinal = state.workerOrdinal;
  auto retired = domain_.retire(std::move(*state.responsibility));
  if (!retired)
    return retired.takeError();
  queue.erase(queued);
  items_.erase(found);
  record(DynamicWorkScheduleActionKind::CancelQueued, item, workerOrdinal,
         std::nullopt);
  return DynamicWorkCancellationResult{
      DynamicWorkCancellationKind::CancelledQueued, *retired};
}

llvm::Expected<bool> DynamicWorkScheduler::cancellationRequested(
    const DynamicWorkAssignment &assignment) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (llvm::Error error = validateAssignment(assignment))
    return std::move(error);
  return items_.find(assignment.id_)->second.cancellationRequested;
}

llvm::Expected<RetirementEffect>
DynamicWorkScheduler::retireAssignment(DynamicWorkAssignment &&assignment,
                                       DynamicWorkScheduleActionKind action) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (llvm::Error error = validateAssignment(assignment))
    return std::move(error);
  auto found = items_.find(assignment.id_);
  const bool cancellationRequested = found->second.cancellationRequested;
  if (action == DynamicWorkScheduleActionKind::Complete &&
      cancellationRequested)
    return reject(DynamicWorkSchedulerError::Kind::CancellationPending,
                  "dynamic-work completion has a pending cancellation");
  if (action == DynamicWorkScheduleActionKind::CancelActive &&
      !cancellationRequested)
    return reject(
        DynamicWorkSchedulerError::Kind::CancellationNotRequested,
        "dynamic-work cancellation was not requested for this assignment");
  const WorkItemId item = assignment.id_;
  const std::uint32_t workerOrdinal = assignment.workerOrdinal_;
  auto retired = domain_.retire(std::move(*found->second.responsibility));
  if (!retired)
    return retired.takeError();
  items_.erase(found);
  assignment.ownerIdentity_.reset();
  record(action, item, workerOrdinal, std::nullopt);
  return *retired;
}

llvm::Expected<RetirementEffect>
DynamicWorkScheduler::complete(DynamicWorkAssignment &&assignment) {
  return retireAssignment(std::move(assignment),
                          DynamicWorkScheduleActionKind::Complete);
}

llvm::Expected<RetirementEffect>
DynamicWorkScheduler::cancel(DynamicWorkAssignment &&assignment) {
  return retireAssignment(std::move(assignment),
                          DynamicWorkScheduleActionKind::CancelActive);
}

std::size_t DynamicWorkScheduler::activeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return domain_.activeCount();
}

std::size_t DynamicWorkScheduler::queuedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t count = 0;
  for (const auto &queue : queues_)
    count += queue.size();
  return count;
}

bool DynamicWorkScheduler::completed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return domain_.completed();
}

std::vector<DynamicWorkScheduleAction> DynamicWorkScheduler::replay() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return replay_;
}

} // namespace loom::sim
