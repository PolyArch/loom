#include "CgraTransportStorageRuntime.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <system_error>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

bool sameEntry(const CgraTransportStorageEntry &lhs,
               const CgraTransportStorageEntry &rhs) {
  return lhs.transferSlot == rhs.transferSlot &&
         lhs.traversalNodeOrdinal == rhs.traversalNodeOrdinal &&
         lhs.physicalTagOrdinal == rhs.physicalTagOrdinal &&
         lhs.virtualChannelKey == rhs.virtualChannelKey;
}

} // namespace

llvm::Expected<CgraTransportStorageRuntime>
CgraTransportStorageRuntime::create(std::uint32_t capacity,
                                    bool fullReplacementAllowed,
                                    ::fabric::FifoQueueDiscipline discipline) {
  if (capacity == 0)
    return invalid("CGRA traversal storage capacity must be positive");
  return CgraTransportStorageRuntime(capacity, fullReplacementAllowed,
                                     discipline);
}

const CgraTransportStorageEntry &CgraTransportStorageRuntime::front() const {
  assert(!empty() && "front of empty CGRA storage queue");
  return entries_.front();
}

std::optional<CgraTransportStorageEntry>
CgraTransportStorageRuntime::offeredEntry() const {
  if (entries_.empty())
    return std::nullopt;
  if (discipline_ == ::fabric::FifoQueueDiscipline::StrictFifo)
    return entries_.front();
  // Scan the canonical ascending channel order starting at the cursor and
  // wrapping once. Entries are in arrival order, so the first entry seen for
  // a channel is that channel's head and arrival order within a channel is
  // preserved.
  std::optional<CgraTransportStorageEntry> atOrAfterCursor;
  std::optional<CgraTransportStorageEntry> beforeCursor;
  for (const CgraTransportStorageEntry &entry : entries_) {
    std::optional<CgraTransportStorageEntry> &slot =
        entry.virtualChannelKey >= offerCursor_ ? atOrAfterCursor
                                                : beforeCursor;
    if (!slot || entry.virtualChannelKey < slot->virtualChannelKey)
      slot = entry;
  }
  return atOrAfterCursor ? atOrAfterCursor : beforeCursor;
}

void CgraTransportStorageRuntime::advanceOffer() {
  const auto offered = offeredEntry();
  if (!offered)
    return;
  offerCursor_ =
      offered->virtualChannelKey == std::numeric_limits<std::uint32_t>::max()
          ? 0
          : offered->virtualChannelKey + 1;
}

bool CgraTransportStorageRuntime::admits(bool enqueue, bool dequeue) const {
  if (!enqueue && !dequeue)
    return false;
  if (dequeue && empty())
    return false;
  if (!enqueue || !full())
    return true;
  return dequeue && fullReplacementAllowed_;
}

llvm::Expected<CgraTransportStorageCommit> CgraTransportStorageRuntime::commit(
    std::optional<CgraTransportStorageEntry> enqueue,
    std::optional<CgraTransportStorageEntry> dequeue) {
  if (!admits(enqueue.has_value(), dequeue.has_value()))
    return invalid("CGRA traversal storage commit violates queue capacity");

  CgraTransportStorageCommit result;
  if (dequeue) {
    const auto offered = offeredEntry();
    if (!offered || !sameEntry(*offered, *dequeue))
      return invalid("CGRA traversal storage dequeue is not the entry its "
                     "queue discipline offers");
    const auto found =
        llvm::find_if(entries_, [&](const CgraTransportStorageEntry &entry) {
          return sameEntry(entry, *dequeue);
        });
    if (found == entries_.end())
      return invalid("CGRA traversal storage dequeue names an absent entry");
    result.dequeued = *found;
    advanceOffer();
    entries_.erase(found);
  }
  if (enqueue) {
    if (entries_.size() == capacity_)
      return invalid("CGRA traversal storage tail is occupied");
    entries_.push_back(std::move(*enqueue));
    result.enqueued = true;
  }
  return result;
}

} // namespace loom::sim::detail
