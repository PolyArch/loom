#include "Simulator/DynamicWorkDomain.h"

#include "DynamicWorkOrdinal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace loom {
namespace sim {

struct WorkResponsibility::ControlState {
  explicit ControlState(ThreadDispatchOccurrenceId dispatchOccurrence)
      : dispatchOccurrence(dispatchOccurrence) {}

  ControlState(const ControlState &) = delete;
  ControlState &operator=(const ControlState &) = delete;
  ControlState(ControlState &&) = delete;
  ControlState &operator=(ControlState &&) = delete;

  ThreadDispatchOccurrenceId dispatchOccurrence;
  bool rootSourceClosed = false;
  std::map<WorkItemId, detail::ChildOrdinalCursor> childCursor;
  std::set<WorkItemId> active;
};

namespace {

std::string describe(const WorkItemId &item) {
  llvm::SmallVector<std::uint64_t, 4> path;
  for (std::optional<WorkItemId> node = item; node; node = node->parent())
    path.push_back(node->ordinal());

  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "work item (dispatch " << item.domainInstance().value()
         << ", ordinals ";
  for (std::size_t i = 0; i < path.size(); ++i) {
    if (i != 0)
      stream << '.';
    stream << path[path.size() - 1 - i];
  }
  stream << ")";
  return stream.str();
}

llvm::Error reject(DynamicWorkDomainError::Kind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<DynamicWorkDomainError>(kind, message.str());
}

} // namespace

WorkItemId WorkItemId::root(ThreadDispatchOccurrenceId domainInstance) {
  const std::uint64_t rootOrdinal = 0;
  return WorkItemId(domainInstance, rootOrdinal);
}

char DynamicWorkStableItemProjectionError::ID = 0;

void DynamicWorkStableItemProjectionError::log(
    llvm::raw_ostream &stream) const {
  stream << "dynamic_work_stable_item_projection_unavailable: " << message_;
}

std::error_code
DynamicWorkStableItemProjectionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<dataflow::DynamicWorkStableItemKey>
projectDynamicWorkStableItemKey(const WorkItemId &item) {
  if (!item.isRoot())
    return llvm::make_error<DynamicWorkStableItemProjectionError>(
        "child WorkItemId has no Dataflow-owned publication lineage");
  return dataflow::DynamicWorkStableItemKey{};
}

WorkItemId WorkItemId::child(const WorkItemId &parent, std::uint64_t ordinal) {
  llvm::SmallVector<std::uint64_t, 4> ordinals(parent.ordinals_.begin(),
                                               parent.ordinals_.end());
  ordinals.push_back(ordinal);
  return WorkItemId(parent.domainInstance_, ordinals);
}

std::optional<WorkItemId> WorkItemId::parent() const {
  if (isRoot())
    return std::nullopt;
  return WorkItemId(domainInstance_,
                    llvm::ArrayRef<std::uint64_t>(ordinals_).drop_back());
}

WorkResponsibility::WorkResponsibility(
    WorkItemId id, const std::shared_ptr<ControlState> &control)
    : id_(std::move(id)), control_(control) {}

char DynamicWorkDomainError::ID = 0;

DynamicWorkDomainError::DynamicWorkDomainError(Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void DynamicWorkDomainError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code DynamicWorkDomainError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

DynamicWorkDomain::DynamicWorkDomain(
    ThreadDispatchOccurrenceId dispatchOccurrence)
    : control_(std::make_shared<ControlState>(dispatchOccurrence)) {}

std::size_t DynamicWorkDomain::activeCount() const {
  return control_->active.size();
}

bool DynamicWorkDomain::completed() const {
  return control_->rootSourceClosed && control_->active.empty();
}

llvm::Error DynamicWorkDomain::validateCapability(
    const WorkResponsibility &responsibility) const {
  std::shared_ptr<const ControlState> owner = responsibility.control_.lock();
  if (!owner)
    return reject(DynamicWorkDomainError::Kind::InvalidResponsibility,
                  "responsibility capability is empty or no longer live");
  if (owner.get() != control_.get())
    return reject(DynamicWorkDomainError::Kind::ForeignDomain,
                  describe(responsibility.id_) +
                      " belongs to another domain coordinator");
  return llvm::Error::success();
}

llvm::Expected<WorkResponsibility> DynamicWorkDomain::admitRoot() {
  if (control_->rootSourceClosed)
    return reject(DynamicWorkDomainError::Kind::RootAlreadyAdmitted,
                  "dispatch occurrence " +
                      llvm::Twine(control_->dispatchOccurrence.value()) +
                      " already admitted its root");

  WorkItemId root = WorkItemId::root(control_->dispatchOccurrence);
  if (!control_->active.insert(root).second)
    llvm::report_fatal_error(
        "DynamicWorkDomain invariant failure: duplicate active root");
  control_->rootSourceClosed = true;
  return WorkResponsibility(std::move(root), control_);
}

llvm::Expected<WorkResponsibility>
DynamicWorkDomain::spawnChild(const WorkResponsibility &parent) {
  if (llvm::Error error = validateCapability(parent))
    return std::move(error);
  if (control_->active.count(parent.id_) == 0)
    llvm::report_fatal_error(
        "DynamicWorkDomain invariant failure: inactive parent capability");

  auto currentCursor = control_->childCursor.find(parent.id_);
  detail::ChildOrdinalCursor nextCursor =
      currentCursor == control_->childCursor.end()
          ? detail::ChildOrdinalCursor()
          : currentCursor->second;
  std::optional<std::uint64_t> ordinal = nextCursor.take();
  if (!ordinal)
    return reject(DynamicWorkDomainError::Kind::ChildOrdinalExhausted,
                  describe(parent.id_) + " has exhausted its child ordinals");

  WorkItemId child = WorkItemId::child(parent.id_, *ordinal);
  if (!control_->active.insert(child).second)
    llvm::report_fatal_error(
        "DynamicWorkDomain invariant failure: duplicate active child");
  control_->childCursor[parent.id_] = nextCursor;
  return WorkResponsibility(std::move(child), control_);
}

llvm::Expected<RetirementEffect>
DynamicWorkDomain::retire(WorkResponsibility &&responsibility) {
  if (llvm::Error error = validateCapability(responsibility))
    return std::move(error);

  auto active = control_->active.find(responsibility.id_);
  if (active == control_->active.end())
    llvm::report_fatal_error(
        "DynamicWorkDomain invariant failure: inactive live capability");
  control_->active.erase(active);
  control_->childCursor.erase(responsibility.id_);
  responsibility.control_.reset();

  return completed() ? RetirementEffect::DomainCompleted
                     : RetirementEffect::DomainStillActive;
}

} // namespace sim
} // namespace loom
