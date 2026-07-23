#include "Simulator/DynamicWorkDomain.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace loom {
namespace sim {
namespace {

std::string describe(const WorkItemId &item) {
  // Reconstruct the ordinal path from the public ancestry, root first, so an
  // error names the exact identity without exposing internal storage.
  llvm::SmallVector<std::uint64_t, 4> path;
  for (std::optional<WorkItemId> node = item; node; node = node->parent())
    path.push_back(node->ordinal());

  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "work item (instance " << item.instance().value() << ", ordinals ";
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

WorkItemId WorkItemId::root(DomainInstanceId instance) {
  // The root's own ordinal is zero and its parent is the distinguished Root,
  // which the empty prefix of this one-element path represents.
  const std::uint64_t rootOrdinal = 0;
  return WorkItemId(instance, rootOrdinal);
}

WorkItemId WorkItemId::child(const WorkItemId &parent, std::uint64_t ordinal) {
  llvm::SmallVector<std::uint64_t, 4> ordinals(parent.ordinals_.begin(),
                                               parent.ordinals_.end());
  ordinals.push_back(ordinal);
  return WorkItemId(parent.instance_, ordinals);
}

std::optional<WorkItemId> WorkItemId::parent() const {
  if (isRoot())
    return std::nullopt;
  return WorkItemId(instance_,
                    llvm::ArrayRef<std::uint64_t>(ordinals_).drop_back());
}

char DynamicWorkDomainError::ID = 0;

DynamicWorkDomainError::DynamicWorkDomainError(Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void DynamicWorkDomainError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code DynamicWorkDomainError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

std::uint64_t
DynamicWorkDomain::nextChildOrdinal(const WorkItemId &parent) const {
  auto cursor = childCursor_.find(parent);
  return cursor == childCursor_.end() ? 0 : cursor->second;
}

bool DynamicWorkDomain::everAcquired(const WorkItemId &item) const {
  if (item.instance() != instance_)
    return false;
  // Walk from the item to the root. Each step confirms the child ordinal was
  // one this parent actually handed out; a cursor only advances inside an
  // accepted spawn, so passing every step proves the whole ancestry was
  // acquired.
  WorkItemId node = item;
  while (!node.isRoot()) {
    WorkItemId parent = *node.parent();
    if (node.ordinal() >= nextChildOrdinal(parent))
      return false;
    node = parent;
  }
  // The remaining root-shaped identity was acquired exactly when the domain
  // admitted its root.
  return rootSourceClosed_;
}

llvm::Error DynamicWorkDomain::requireActive(const WorkItemId &item) const {
  if (item.instance() != instance_)
    return reject(DynamicWorkDomainError::Kind::ForeignDomain,
                  describe(item) + " belongs to another domain instance than " +
                      llvm::Twine(instance_.value()));
  if (active_.count(item) != 0)
    return llvm::Error::success();
  if (everAcquired(item))
    return reject(DynamicWorkDomainError::Kind::AlreadyRetired,
                  describe(item) + " was already retired");
  return reject(DynamicWorkDomainError::Kind::UnknownItem,
                describe(item) + " was never published by this domain");
}

llvm::Expected<WorkItemId> DynamicWorkDomain::admitRoot() {
  if (rootSourceClosed_)
    return reject(DynamicWorkDomainError::Kind::RootAlreadyAdmitted,
                  "domain instance " + llvm::Twine(instance_.value()) +
                      " already admitted its root");
  WorkItemId root = WorkItemId::root(instance_);
  // Acquire the root responsibility, then close the root source: one
  // transaction, so a later item can arise only through a registered spawn.
  active_.insert(root);
  rootSourceClosed_ = true;
  return root;
}

llvm::Expected<WorkItemId>
DynamicWorkDomain::spawnChild(const WorkItemId &parent) {
  if (llvm::Error error = requireActive(parent))
    return std::move(error);

  std::uint64_t ordinal = nextChildOrdinal(parent);
  // A saturated parent can never mint another ordinal; reject before mutating
  // any state so the cursor, active set, and completion are unchanged.
  if (ordinal == std::numeric_limits<std::uint64_t>::max())
    return reject(DynamicWorkDomainError::Kind::ChildOrdinalExhausted,
                  describe(parent) + " has exhausted its child ordinals");

  WorkItemId child = WorkItemId::child(parent, ordinal);
  // Acquire the child responsibility before publishing the identity. A
  // monotonic, non-wrapping per-parent ordinal makes a duplicate structurally
  // impossible, so a failed insertion is a non-returning invariant failure
  // raised before the ordinal is consumed.
  if (!active_.insert(child).second)
    llvm::report_fatal_error(
        "DynamicWorkDomain invariant failure: duplicate active child");
  // Consume the ordinal only after the responsibility is acquired, before the
  // child identity is published to the caller.
  childCursor_[parent] = ordinal + 1;
  return child;
}

llvm::Expected<RetirementEffect>
DynamicWorkDomain::retire(const WorkItemId &item) {
  if (llvm::Error error = requireActive(item))
    return std::move(error);

  active_.erase(item);
  // completed() is false before this erase because the item kept the set
  // non-empty, so a true reading here is the one completion transition.
  return completed() ? RetirementEffect::DomainCompleted
                     : RetirementEffect::DomainStillActive;
}

} // namespace sim
} // namespace loom
