#include "CgraTransportStorageRuntime.h"

#include <cassert>
#include <system_error>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<CgraTransportStorageRuntime>
CgraTransportStorageRuntime::create(std::uint32_t capacity) {
  if (capacity == 0)
    return invalid("CGRA traversal storage capacity must be positive");
  return CgraTransportStorageRuntime(capacity);
}

const CgraTransportStorageEntry &CgraTransportStorageRuntime::front() const {
  assert(!empty() && entries_[head_] && "front of empty CGRA storage queue");
  return *entries_[head_];
}

bool CgraTransportStorageRuntime::admits(bool enqueue, bool dequeue) const {
  if (!enqueue && !dequeue)
    return false;
  const std::uint32_t dequeues = dequeue ? 1 : 0;
  return dequeues <= occupancy_ &&
         (!enqueue || occupancy_ - dequeues < entries_.size());
}

llvm::Expected<CgraTransportStorageCommit> CgraTransportStorageRuntime::commit(
    std::optional<CgraTransportStorageEntry> enqueue, bool dequeue) {
  if (!admits(enqueue.has_value(), dequeue))
    return invalid("CGRA traversal storage commit violates queue capacity");

  CgraTransportStorageCommit result;
  if (dequeue) {
    if (!entries_[head_])
      return invalid("CGRA traversal storage head is absent");
    result.dequeued = std::move(entries_[head_]);
    entries_[head_].reset();
    head_ = (head_ + 1) % entries_.size();
    --occupancy_;
  }
  if (enqueue) {
    if (entries_[tail_])
      return invalid("CGRA traversal storage tail is occupied");
    entries_[tail_] = std::move(*enqueue);
    tail_ = (tail_ + 1) % entries_.size();
    ++occupancy_;
    result.enqueued = true;
  }
  return result;
}

} // namespace loom::sim::detail
