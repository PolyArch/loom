#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace loom::sim::detail {

/// Execution-local identity of one token waiting in an exact selected
/// traversal storage occurrence. The token value remains in the owning
/// transport slot; this entry only retains the continuation needed on dequeue.
struct CgraTransportStorageEntry final {
  std::uint64_t transferSlot = 0;
  std::uint64_t traversalNodeOrdinal = 0;
  std::uint64_t physicalTagOrdinal = std::numeric_limits<std::uint64_t>::max();
};

struct CgraTransportStorageCommit final {
  std::optional<CgraTransportStorageEntry> dequeued;
  bool enqueued = false;
};

/// Fixed-capacity durable FIFO state derived from one selected Fabric storage
/// traversal. Resource claims and timing stay in CgraPhysicalActionRuntime;
/// this owner applies only the exact queue transition at the commit event.
class CgraTransportStorageRuntime final {
public:
  static llvm::Expected<CgraTransportStorageRuntime>
  create(std::uint32_t capacity);

  std::uint32_t capacity() const {
    return static_cast<std::uint32_t>(entries_.size());
  }
  std::uint32_t occupancy() const { return occupancy_; }
  bool empty() const { return occupancy_ == 0; }
  bool full() const { return occupancy_ == entries_.size(); }
  const CgraTransportStorageEntry &front() const;
  bool admits(bool enqueue, bool dequeue) const;

  /// Atomically applies one cycle-start dequeue and/or enqueue. A dequeue sees
  /// only the cycle-start queue, while its released slot is available to the
  /// same commit's enqueue. Consequently an empty simultaneous operation is
  /// rejected and a full simultaneous replacement is admitted.
  llvm::Expected<CgraTransportStorageCommit>
  commit(std::optional<CgraTransportStorageEntry> enqueue, bool dequeue);

private:
  explicit CgraTransportStorageRuntime(std::uint32_t capacity)
      : entries_(capacity) {}

  std::vector<std::optional<CgraTransportStorageEntry>> entries_;
  std::uint32_t head_ = 0;
  std::uint32_t tail_ = 0;
  std::uint32_t occupancy_ = 0;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H
