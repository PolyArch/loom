#ifndef LOOM_SIMULATOR_THREADDISPATCHIDENTITY_H
#define LOOM_SIMULATOR_THREADDISPATCHIDENTITY_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

namespace loom {
namespace sim {

/// Transient identity assigned to one concrete thread dispatch. It is unique
/// only within its owning execution session and never selects persistent
/// Mapping, binary, route, or configuration state.
class ThreadDispatchOccurrenceId {
public:
  explicit constexpr ThreadDispatchOccurrenceId(std::uint64_t value)
      : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(ThreadDispatchOccurrenceId lhs,
                                   ThreadDispatchOccurrenceId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(ThreadDispatchOccurrenceId lhs,
                                   ThreadDispatchOccurrenceId rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(ThreadDispatchOccurrenceId lhs,
                                  ThreadDispatchOccurrenceId rhs) {
    return lhs.value_ < rhs.value_;
  }

private:
  std::uint64_t value_;
};

/// The exact runtime identity of one dynamic work item:
/// `(domain instance, root-or-parent item, child launch ordinal)`. The domain
/// instance is the owning dispatch occurrence rather than a second counter.
/// The root is `(dispatch, Root, 0)`; every other item recursively names its
/// parent and zero-based program-order spawn ordinal.
class WorkItemId {
public:
  static WorkItemId root(ThreadDispatchOccurrenceId domainInstance);
  static WorkItemId child(const WorkItemId &parent, std::uint64_t ordinal);

  ThreadDispatchOccurrenceId domainInstance() const { return domainInstance_; }

  std::uint64_t ordinal() const { return ordinals_.back(); }
  std::optional<WorkItemId> parent() const;
  bool isRoot() const { return ordinals_.size() == 1; }

  friend bool operator==(const WorkItemId &lhs, const WorkItemId &rhs) {
    return lhs.domainInstance_ == rhs.domainInstance_ &&
           lhs.ordinals_ == rhs.ordinals_;
  }
  friend bool operator!=(const WorkItemId &lhs, const WorkItemId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const WorkItemId &lhs, const WorkItemId &rhs) {
    if (lhs.domainInstance_ != rhs.domainInstance_)
      return lhs.domainInstance_ < rhs.domainInstance_;
    return std::lexicographical_compare(
        lhs.ordinals_.begin(), lhs.ordinals_.end(), rhs.ordinals_.begin(),
        rhs.ordinals_.end());
  }

private:
  WorkItemId(ThreadDispatchOccurrenceId domainInstance,
             llvm::ArrayRef<std::uint64_t> ordinals)
      : domainInstance_(domainInstance),
        ordinals_(ordinals.begin(), ordinals.end()) {}

  ThreadDispatchOccurrenceId domainInstance_;
  llvm::SmallVector<std::uint64_t, 4> ordinals_;
};

/// Removes execution-local dispatch and item lineage from one DynamicWork
/// item. The selected thread owns one stable execution class for the complete
/// domain; WorkItemId remains the sole logical identity of an item.
llvm::Expected<dataflow::DynamicWorkStableItemKey>
projectDynamicWorkStableItemKey(const WorkItemId &item);

/// One point in a dense logical domain. Root launch identity distinguishes
/// otherwise equal coordinates belonging to different static launch sites.
struct DenseLogicalThreadPoint {
  DenseLogicalThreadPoint(dataflow::RootThreadLaunchRef rootThreadLaunch,
                          llvm::ArrayRef<std::uint64_t> coordinates)
      : rootThreadLaunch(std::move(rootThreadLaunch)),
        coordinates(coordinates.begin(), coordinates.end()) {}

  dataflow::RootThreadLaunchRef rootThreadLaunch;
  llvm::SmallVector<std::uint64_t, 4> coordinates;

  friend bool operator==(const DenseLogicalThreadPoint &lhs,
                         const DenseLogicalThreadPoint &rhs) {
    return lhs.rootThreadLaunch == rhs.rootThreadLaunch &&
           lhs.coordinates == rhs.coordinates;
  }
  friend bool operator!=(const DenseLogicalThreadPoint &lhs,
                         const DenseLogicalThreadPoint &rhs) {
    return !(lhs == rhs);
  }
};

/// One point in a DynamicWork logical domain. WorkItemId already carries the
/// dispatch occurrence used as its domain_instance component.
struct DynamicLogicalThreadPoint {
  dataflow::RootThreadLaunchRef rootThreadLaunch;
  WorkItemId workItem;

  friend bool operator==(const DynamicLogicalThreadPoint &lhs,
                         const DynamicLogicalThreadPoint &rhs) {
    return lhs.rootThreadLaunch == rhs.rootThreadLaunch &&
           lhs.workItem == rhs.workItem;
  }
  friend bool operator!=(const DynamicLogicalThreadPoint &lhs,
                         const DynamicLogicalThreadPoint &rhs) {
    return !(lhs == rhs);
  }
};

/// The closed logical point union. It is derived from an exact rooted launch;
/// it is not persistent identity and does not identify a physical AccCore.
using LogicalThreadPoint =
    std::variant<DenseLogicalThreadPoint, DynamicLogicalThreadPoint>;

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_THREADDISPATCHIDENTITY_H
