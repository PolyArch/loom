#ifndef LOOM_RUNTIME_ORDEREDCHANNELSEQUENCE_H
#define LOOM_RUNTIME_ORDEREDCHANNELSEQUENCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace loom::runtime {

enum class OrderedChannelReceiveKind : std::uint8_t {
  Message,
  WouldBlock,
  Closed,
};

struct OrderedChannelReceive final {
  OrderedChannelReceiveKind kind = OrderedChannelReceiveKind::WouldBlock;
  std::uint64_t sequence = 0;
  std::vector<std::uint8_t> payload;
};

/// Invocation-local ordered message state for one logical channel instance.
/// Dataflow owns the endpoint correspondence; this class owns only mutable
/// SendSeq/RecvSeq cursors, reservations, acknowledgements, and bounded
/// storage. It supports multicast by retaining each message until every
/// consumer commits it.
class OrderedChannelSequence final {
public:
  static llvm::Expected<OrderedChannelSequence>
  create(std::uint64_t capacityBytes, std::uint32_t consumerCount);

  OrderedChannelSequence(OrderedChannelSequence &&) noexcept = default;
  OrderedChannelSequence &operator=(OrderedChannelSequence &&) noexcept =
      default;
  OrderedChannelSequence(const OrderedChannelSequence &) = delete;
  OrderedChannelSequence &operator=(const OrderedChannelSequence &) = delete;

  llvm::Expected<std::uint64_t>
  publish(llvm::ArrayRef<std::uint8_t> payload);

  llvm::Expected<OrderedChannelReceive>
  reserve(std::uint32_t consumerOrdinal);

  llvm::Error commit(std::uint32_t consumerOrdinal,
                    std::uint64_t sequence);
  llvm::Error cancel(std::uint32_t consumerOrdinal, std::uint64_t sequence);
  llvm::Error close();

  std::uint64_t capacityBytes() const { return capacityBytes_; }
  std::uint64_t occupiedBytes() const { return occupiedBytes_; }
  std::uint64_t nextSendSequence() const { return nextSendSequence_; }
  std::uint64_t nextReceiveSequence(std::uint32_t consumerOrdinal) const;
  std::uint32_t consumerCount() const {
    return static_cast<std::uint32_t>(nextReceiveSequences_.size());
  }
  bool closed() const { return closed_; }

private:
  struct Message final {
    std::uint64_t sequence = 0;
    std::vector<std::uint8_t> payload;
    std::vector<bool> reserved;
    std::vector<bool> committed;
  };

  OrderedChannelSequence(std::uint64_t capacityBytes,
                         std::uint32_t consumerCount)
      : capacityBytes_(capacityBytes),
        nextReceiveSequences_(consumerCount, 0),
        reservations_(consumerCount) {}

  llvm::Error validateConsumer(std::uint32_t consumerOrdinal) const;
  Message *findMessage(std::uint64_t sequence);
  const Message *findMessage(std::uint64_t sequence) const;
  void reclaimCommittedPrefix();

  std::uint64_t capacityBytes_ = 0;
  std::uint64_t occupiedBytes_ = 0;
  std::uint64_t nextSendSequence_ = 0;
  std::vector<std::uint64_t> nextReceiveSequences_;
  std::vector<std::optional<std::uint64_t>> reservations_;
  std::deque<Message> messages_;
  bool closed_ = false;
};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_ORDEREDCHANNELSEQUENCE_H
