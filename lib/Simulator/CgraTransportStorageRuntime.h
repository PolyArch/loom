#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H

#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
  /// Virtual-channel identity of this token: the dense rank of its canonical
  /// Physical Tag value among the distinct tag values of the execution plan.
  /// The plan's tag ordinal is an internal segment index that two equal tag
  /// values may not share, and hardware never observes it, so the tag value
  /// is the only identity a queue discipline may schedule on. Ranks follow
  /// ascending unsigned tag value, which is the same canonical order an
  /// arbiter can implement.
  std::uint32_t virtualChannelKey = 0;
};

struct CgraTransportStorageCommit final {
  std::optional<CgraTransportStorageEntry> dequeued;
  bool enqueued = false;
};

/// Fixed-capacity durable queue state derived from one selected Fabric storage
/// traversal. Resource claims and timing stay in CgraPhysicalActionRuntime;
/// this owner applies only the exact queue transition at the commit event and
/// enforces the declared dequeue scheduling discipline.
///
/// Entries are held in arrival order. A StrictFifo queue offers only the
/// oldest entry, so a token its consumer cannot take blocks every later
/// token. A PerTagVirtualChannel queue keeps the arrival order within one
/// Physical Tag value and offers the oldest entry of exactly one virtual
/// channel per cycle, so a token blocks only later tokens of its own tag.
///
/// The queue presents one head per cycle because the physical port carries a
/// single valid/ready pair: a discipline that inspected the downstream
/// readiness of every channel at once would be an ability the hardware does
/// not have. The offered channel is chosen by a round robin over canonical
/// ascending tag values. `advanceOffer` moves the cursor when the current
/// offer was not taken this cycle, and a grant moves it past the granted
/// channel, so with N occupied channels every channel is offered at least
/// once every N cycles and the choice is a function of queue state alone.
class CgraTransportStorageRuntime final {
public:
  static llvm::Expected<CgraTransportStorageRuntime>
  create(std::uint32_t capacity, bool fullReplacementAllowed = false,
         ::fabric::FifoQueueDiscipline discipline =
             ::fabric::FifoQueueDiscipline::StrictFifo);

  std::uint32_t capacity() const { return capacity_; }
  std::uint32_t occupancy() const {
    return static_cast<std::uint32_t>(entries_.size());
  }
  bool empty() const { return entries_.empty(); }
  bool full() const { return entries_.size() == capacity_; }
  ::fabric::FifoQueueDiscipline discipline() const { return discipline_; }
  const CgraTransportStorageEntry &front() const;
  bool admits(bool enqueue, bool dequeue) const;

  /// Appends the resident entries from the oldest toward the newest. The index
  /// of an appended entry is its exact queue position, so a closed-wait
  /// certificate can state how far an awaited token sits behind the head.
  void appendQueueOrder(std::vector<CgraTransportStorageEntry> &entries) const {
    entries.insert(entries.end(), entries_.begin(), entries_.end());
  }

  /// The single entry this queue presents for dequeue this cycle, or absent
  /// when the queue is empty. Only this entry may be passed to `commit`.
  std::optional<CgraTransportStorageEntry> offeredEntry() const;

  /// The number of distinct resident virtual channels. This bounds one probe
  /// epoch: after that many consecutive refused offers every channel has been
  /// presented once without a grant.
  std::uint32_t distinctResidentChannels() const {
    llvm::SmallVector<std::uint32_t, 4> seen;
    for (const CgraTransportStorageEntry &entry : entries_)
      if (!llvm::is_contained(seen, entry.virtualChannelKey))
        seen.push_back(entry.virtualChannelKey);
    return static_cast<std::uint32_t>(seen.size());
  }

  /// The channel the round robin resumes at, exposed for per-event
  /// simulator/RTL conformance comparison.
  std::uint32_t offerCursor() const { return offerCursor_; }

  /// Moves the round robin past the channel currently offered. The caller
  /// invokes this exactly once for a cycle in which the offer was presented
  /// and not taken, so a channel whose consumer is not ready yields the port
  /// instead of holding it.
  void advanceOffer();

  /// Atomically applies one cycle-start dequeue and/or enqueue. An ordinary
  /// queue admits enqueue from cycle-start capacity, so a dequeue from a full
  /// queue releases its slot for the next cycle. Independently serviced
  /// storage may opt into full simultaneous replacement at construction. A
  /// dequeue must name an entry this discipline currently offers.
  llvm::Expected<CgraTransportStorageCommit>
  commit(std::optional<CgraTransportStorageEntry> enqueue,
         std::optional<CgraTransportStorageEntry> dequeue);

private:
  explicit CgraTransportStorageRuntime(std::uint32_t capacity,
                                       bool fullReplacementAllowed,
                                       ::fabric::FifoQueueDiscipline discipline)
      : capacity_(capacity), fullReplacementAllowed_(fullReplacementAllowed),
        discipline_(discipline) {}

  llvm::SmallVector<CgraTransportStorageEntry, 4> entries_;
  std::uint32_t capacity_ = 0;
  bool fullReplacementAllowed_ = false;
  ::fabric::FifoQueueDiscipline discipline_ =
      ::fabric::FifoQueueDiscipline::StrictFifo;
  /// The virtual channel the round robin resumes at. Rotation is over the
  /// canonical ascending tag-value ranks, wrapping at the highest observed
  /// rank, so the cursor is meaningful even when the channel it names is
  /// currently empty.
  std::uint32_t offerCursor_ = 0;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTSTORAGERUNTIME_H
