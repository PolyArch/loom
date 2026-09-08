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
                                    ::fabric::FifoQueueDiscipline discipline,
                                    std::uint32_t reservedChannels) {
  if (capacity == 0)
    return invalid("CGRA traversal storage capacity must be positive");
  if (discipline != ::fabric::FifoQueueDiscipline::PerTagVirtualChannel &&
      reservedChannels != 0)
    return invalid("only a per-tag virtual channel storage reserves channels");
  if (reservedChannels > capacity)
    return invalid("CGRA storage reserves more channels than it has slots");
  return CgraTransportStorageRuntime(capacity, fullReplacementAllowed,
                                     discipline, reservedChannels);
}

bool CgraTransportStorageRuntime::channelResident(
    std::uint32_t channel) const {
  return llvm::any_of(entries_,
                      [&](const CgraTransportStorageEntry &entry) {
                        return entry.virtualChannelKey == channel;
                      }) ||
         llvm::any_of(reservations_,
                      [&](const auto &held) { return held.first == channel; });
}

std::uint32_t CgraTransportStorageRuntime::residentChannelCount() const {
  llvm::SmallVector<std::uint32_t, 4> seen;
  for (const CgraTransportStorageEntry &entry : entries_)
    if (!llvm::is_contained(seen, entry.virtualChannelKey))
      seen.push_back(entry.virtualChannelKey);
  for (const auto &held : reservations_)
    if (!llvm::is_contained(seen, held.first))
      seen.push_back(held.first);
  return static_cast<std::uint32_t>(seen.size());
}

std::uint32_t
CgraTransportStorageRuntime::claimableCapacity(std::uint32_t channel) const {
  const std::uint32_t held = occupancy() + reservationCount_;
  const std::uint32_t free = held < capacity_ ? capacity_ - held : 0;
  if (discipline_ != ::fabric::FifoQueueDiscipline::PerTagVirtualChannel ||
      reservedChannels_ == 0)
    return free;
  // Each guaranteed channel that is still absent keeps one slot back; the
  // arriving channel claims its own guarantee rather than a stranger's.
  const std::uint32_t claimed =
      residentChannelCount() + (channelResident(channel) ? 0 : 1);
  const std::uint32_t reservedForOthers =
      claimed < reservedChannels_ ? reservedChannels_ - claimed : 0;
  return free > reservedForOthers ? free - reservedForOthers : 0;
}

llvm::Error CgraTransportStorageRuntime::reserve(std::uint32_t channel) {
  if (claimableCapacity(channel) == 0)
    return invalid("CGRA storage reservation exceeds the channel's claimable "
                   "capacity");
  for (auto &held : reservations_)
    if (held.first == channel) {
      ++held.second;
      ++reservationCount_;
      return llvm::Error::success();
    }
  reservations_.push_back({channel, 1});
  ++reservationCount_;
  return llvm::Error::success();
}

llvm::Error CgraTransportStorageRuntime::unreserve(std::uint32_t channel) {
  for (auto it = reservations_.begin(); it != reservations_.end(); ++it) {
    if (it->first != channel)
      continue;
    if (--it->second == 0)
      reservations_.erase(it);
    --reservationCount_;
    return llvm::Error::success();
  }
  return invalid("CGRA storage releases a reservation it does not hold");
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

bool CgraTransportStorageRuntime::admits(
    std::optional<std::uint32_t> enqueueChannel, bool enqueueReserved,
    bool dequeue) const {
  if (!enqueueChannel && !dequeue)
    return false;
  if (dequeue && empty())
    return false;
  if (!enqueueChannel)
    return true;
  if (enqueueReserved)
    return !full();
  if (claimableCapacity(*enqueueChannel) != 0)
    return true;
  return dequeue && fullReplacementAllowed_ && reservationCount_ == 0;
}

llvm::Expected<CgraTransportStorageCommit> CgraTransportStorageRuntime::commit(
    std::optional<CgraTransportStorageEntry> enqueue, bool enqueueReserved,
    std::optional<CgraTransportStorageEntry> dequeue) {
  if (enqueueReserved && !enqueue)
    return invalid("CGRA traversal storage commit reserves no enqueue");
  if (!admits(enqueue ? std::optional<std::uint32_t>(enqueue->virtualChannelKey)
                      : std::nullopt,
              enqueueReserved, dequeue.has_value()))
    return invalid("CGRA traversal storage commit violates queue capacity");
  if (enqueueReserved)
    if (llvm::Error error = unreserve(enqueue->virtualChannelKey))
      return std::move(error);

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
