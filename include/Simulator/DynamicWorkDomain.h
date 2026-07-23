#ifndef LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H
#define LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

namespace loom {
namespace sim {

/// Caller-supplied execution-local observation identity of one dynamic work
/// domain instance. It is never a persistent artifact id or an authorization
/// credential. Distinct DynamicWorkDomain coordinators may use the same value
/// while remaining isolated by their private control state.
class DomainInstanceId {
public:
  explicit constexpr DomainInstanceId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(DomainInstanceId lhs, DomainInstanceId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(DomainInstanceId lhs, DomainInstanceId rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(DomainInstanceId lhs, DomainInstanceId rhs) {
    return lhs.value_ < rhs.value_;
  }

private:
  std::uint64_t value_;
};

/// The exact runtime identity of one dynamic work item:
/// `(domain instance, root-or-parent item, child launch ordinal)`. The root is
/// `(instance, Root, 0)`; every other item recursively names its parent and the
/// zero-based program-order ordinal it was spawned at. The identity is a pure
/// value: naming an item never acquires responsibility, which the domain owns.
///
/// The ancestry is stored as the ordinal path from the root, so the root is the
/// one-element path `{0}`, its child at ordinal k is `{0, k}`, and the empty
/// path is the distinguished Root parent, which is never an item.
class WorkItemId {
public:
  /// The canonical root identity of one domain instance: `(instance, Root, 0)`.
  static WorkItemId root(DomainInstanceId instance);

  /// The identity of `parent`'s child at a zero-based ordinal. It names the
  /// child without acquiring any responsibility.
  static WorkItemId child(const WorkItemId &parent, std::uint64_t ordinal);

  DomainInstanceId instance() const { return instance_; }

  /// The child launch ordinal of this item within its parent. The root's is 0.
  std::uint64_t ordinal() const { return ordinals_.back(); }

  /// The parent identity, or nullopt when the parent is the distinguished Root.
  std::optional<WorkItemId> parent() const;

  /// True for the root identity, whose parent is Root.
  bool isRoot() const { return ordinals_.size() == 1; }

  friend bool operator==(const WorkItemId &lhs, const WorkItemId &rhs) {
    return lhs.instance_ == rhs.instance_ && lhs.ordinals_ == rhs.ordinals_;
  }
  friend bool operator!=(const WorkItemId &lhs, const WorkItemId &rhs) {
    return !(lhs == rhs);
  }
  /// Total order over the exact identity, used only to index domain state.
  friend bool operator<(const WorkItemId &lhs, const WorkItemId &rhs) {
    if (lhs.instance_ != rhs.instance_)
      return lhs.instance_ < rhs.instance_;
    return std::lexicographical_compare(
        lhs.ordinals_.begin(), lhs.ordinals_.end(), rhs.ordinals_.begin(),
        rhs.ordinals_.end());
  }

private:
  WorkItemId(DomainInstanceId instance, llvm::ArrayRef<std::uint64_t> ordinals)
      : instance_(instance), ordinals_(ordinals.begin(), ordinals.end()) {}

  DomainInstanceId instance_;
  llvm::SmallVector<std::uint64_t, 4> ordinals_;
};

class DynamicWorkDomain;

/// Move-only proof that one DynamicWorkDomain currently owns responsibility for
/// `id()`. Only root admission and child spawn can create a live capability.
/// Its WorkItemId remains a copyable observation and carries no authority by
/// itself.
class WorkResponsibility {
public:
  WorkResponsibility(const WorkResponsibility &) = delete;
  WorkResponsibility &operator=(const WorkResponsibility &) = delete;
  WorkResponsibility(WorkResponsibility &&) noexcept = default;

  // Assignment could silently discard an unretired responsibility. Transfer
  // is therefore construction-only.
  WorkResponsibility &operator=(WorkResponsibility &&) = delete;

  const WorkItemId &id() const { return id_; }

private:
  struct ControlState;
  friend class DynamicWorkDomain;

  WorkResponsibility(WorkItemId id,
                     const std::shared_ptr<ControlState> &control);

  WorkItemId id_;
  std::weak_ptr<const ControlState> control_;
};

/// The effect of a retirement on domain termination. `DomainCompleted` is the
/// single completion transition: exactly one retirement empties the active set
/// after the root source is closed.
enum class RetirementEffect {
  DomainStillActive,
  DomainCompleted,
};

class DynamicWorkDomainError final
    : public llvm::ErrorInfo<DynamicWorkDomainError> {
public:
  enum class Kind {
    ForeignDomain,
    RootAlreadyAdmitted,
    InvalidResponsibility,
    ChildOrdinalExhausted,
  };

  static char ID;

  DynamicWorkDomainError(Kind kind, std::string message);

  Kind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// Standalone execution-local responsibility kernel for one dynamic work domain
/// instance. It owns the termination decision of a single logical coordinator:
/// nothing else.
///
/// The active responsibility set is the sole termination authority. Admission
/// acquires the root responsibility and closes the root source in one
/// transaction; a spawn acquires a child responsibility and consumes the
/// parent's next program-order ordinal before the child identity is published;
/// a retirement releases exactly one responsibility. The domain completes at
/// the single moment the active set empties after the root source is closed.
/// Any count is a derived view of the set, never a second authority.
///
/// The kernel owns no payloads, queues, schedulers, channels, workers, atomics,
/// coherence, stealing, priority, cancellation, migration, or hardware state.
///
/// Every rejected action is atomic: it acquires no responsibility, consumes no
/// ordinal, changes no active membership, and produces no completion.
class DynamicWorkDomain {
public:
  explicit DynamicWorkDomain(DomainInstanceId instance);

  // Copying would duplicate termination authority. Moving would introduce
  // coordinator-transfer semantics that this standalone kernel does not own.
  DynamicWorkDomain(const DynamicWorkDomain &) = delete;
  DynamicWorkDomain &operator=(const DynamicWorkDomain &) = delete;
  DynamicWorkDomain(DynamicWorkDomain &&) = delete;
  DynamicWorkDomain &operator=(DynamicWorkDomain &&) = delete;

  /// Admits the root in one transaction: it acquires the root responsibility,
  /// closes the root source, then publishes its capability. Rejected once the
  /// root is already admitted.
  llvm::Expected<WorkResponsibility> admitRoot();

  /// Publishes one child of `parent` to this domain. It acquires the child
  /// responsibility and consumes `parent`'s next program-order ordinal before
  /// returning the child capability, while borrowing and preserving the parent
  /// responsibility. Rejected when `parent` is foreign or invalid.
  llvm::Expected<WorkResponsibility>
  spawnChild(const WorkResponsibility &parent);

  /// Consumes one live capability, retires its active item exactly once, and
  /// reports whether this retirement is the completion transition. A foreign
  /// or invalid capability is rejected without consuming it or changing state.
  llvm::Expected<RetirementEffect> retire(WorkResponsibility &&responsibility);

  /// The number of active responsibilities, derived from the active set.
  std::size_t activeCount() const;

  /// True once the root source is closed and the active set is empty. Stable
  /// after completion; it reads the same facts the completion transition does.
  bool completed() const;

private:
  /// Rejects an empty or foreign capability without consulting observation
  /// identity, and otherwise confirms this coordinator issued it.
  llvm::Error
  validateCapability(const WorkResponsibility &responsibility) const;

  /// The next zero-based child ordinal for `parent`, defaulting to zero.
  std::uint64_t nextChildOrdinal(const WorkItemId &parent) const;

  using ControlState = WorkResponsibility::ControlState;
  std::shared_ptr<ControlState> control_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H
