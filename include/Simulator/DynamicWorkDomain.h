#ifndef LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H
#define LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H

#include "Simulator/ThreadDispatchIdentity.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace loom {
namespace sim {

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

  WorkItemId id() const { return id_; }

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
  explicit DynamicWorkDomain(ThreadDispatchOccurrenceId dispatchOccurrence);

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

  /// Publishes a finite ordered child group as one responsibility transaction.
  /// Failure acquires no child and consumes no ordinal.
  llvm::Expected<std::vector<WorkResponsibility>>
  spawnChildren(const WorkResponsibility &parent, std::size_t count);

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

  using ControlState = WorkResponsibility::ControlState;
  std::shared_ptr<ControlState> control_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_DYNAMICWORKDOMAIN_H
