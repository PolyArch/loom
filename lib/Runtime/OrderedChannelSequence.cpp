#include "Runtime/OrderedChannelSequence.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <system_error>

namespace loom::runtime {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "ordered_channel_invalid: " + message);
}

} // namespace

llvm::Expected<OrderedChannelSequence>
OrderedChannelSequence::create(std::uint64_t capacityBytes,
                               std::uint32_t consumerCount) {
  if (capacityBytes == 0)
    return invalid("channel capacity must be positive");
  if (consumerCount == 0)
    return invalid("channel must have at least one consumer");
  return OrderedChannelSequence(capacityBytes, consumerCount);
}

llvm::Error
OrderedChannelSequence::validateConsumer(std::uint32_t consumerOrdinal) const {
  if (consumerOrdinal >= nextReceiveSequences_.size())
    return invalid("consumer ordinal is outside the channel branch domain");
  return llvm::Error::success();
}

OrderedChannelSequence::Message *
OrderedChannelSequence::findMessage(std::uint64_t sequence) {
  for (Message &message : messages_)
    if (message.sequence == sequence)
      return &message;
  return nullptr;
}

const OrderedChannelSequence::Message *
OrderedChannelSequence::findMessage(std::uint64_t sequence) const {
  for (const Message &message : messages_)
    if (message.sequence == sequence)
      return &message;
  return nullptr;
}

llvm::Expected<std::uint64_t>
OrderedChannelSequence::publish(llvm::ArrayRef<std::uint8_t> payload) {
  if (closed_)
    return invalid("publish followed channel close");
  if (payload.size() > capacityBytes_ - occupiedBytes_)
    return llvm::createStringError(std::errc::no_space_on_device,
                                   "ordered channel capacity is exhausted");
  if (nextSendSequence_ == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "SendSeq overflow");
  Message message;
  message.sequence = nextSendSequence_++;
  message.payload.assign(payload.begin(), payload.end());
  message.reserved.assign(nextReceiveSequences_.size(), false);
  message.committed.assign(nextReceiveSequences_.size(), false);
  messages_.push_back(std::move(message));
  occupiedBytes_ += payload.size();
  return nextSendSequence_ - 1;
}

llvm::Expected<OrderedChannelReceive>
OrderedChannelSequence::reserve(std::uint32_t consumerOrdinal) {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return std::move(error);
  if (reservations_[consumerOrdinal])
    return invalid("consumer already has an uncommitted reservation");
  const std::uint64_t sequence = nextReceiveSequences_[consumerOrdinal];
  Message *message = findMessage(sequence);
  if (!message) {
    if (closed_ && sequence >= nextSendSequence_)
      return OrderedChannelReceive{OrderedChannelReceiveKind::Closed, sequence,
                                   {}};
    return OrderedChannelReceive{OrderedChannelReceiveKind::WouldBlock,
                                 sequence, {}};
  }
  if (message->committed[consumerOrdinal] ||
      message->reserved[consumerOrdinal])
    return invalid("consumer cursor state is inconsistent");
  message->reserved[consumerOrdinal] = true;
  reservations_[consumerOrdinal] = sequence;
  return OrderedChannelReceive{OrderedChannelReceiveKind::Message, sequence,
                               message->payload};
}

llvm::Error OrderedChannelSequence::commit(std::uint32_t consumerOrdinal,
                                           std::uint64_t sequence) {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return error;
  if (!reservations_[consumerOrdinal] ||
      *reservations_[consumerOrdinal] != sequence)
    return invalid("receive commit does not match its reservation");
  Message *message = findMessage(sequence);
  if (!message || !message->reserved[consumerOrdinal] ||
      message->committed[consumerOrdinal] ||
      nextReceiveSequences_[consumerOrdinal] != sequence)
    return invalid("receive commit does not match the ordered cursor");
  message->reserved[consumerOrdinal] = false;
  message->committed[consumerOrdinal] = true;
  reservations_[consumerOrdinal].reset();
  if (sequence == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "RecvSeq overflow");
  ++nextReceiveSequences_[consumerOrdinal];
  reclaimCommittedPrefix();
  return llvm::Error::success();
}

llvm::Error OrderedChannelSequence::cancel(std::uint32_t consumerOrdinal,
                                           std::uint64_t sequence) {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return error;
  if (!reservations_[consumerOrdinal] ||
      *reservations_[consumerOrdinal] != sequence)
    return invalid("receive cancellation does not match its reservation");
  Message *message = findMessage(sequence);
  if (!message || !message->reserved[consumerOrdinal] ||
      message->committed[consumerOrdinal])
    return invalid("receive cancellation names an invalid message");
  message->reserved[consumerOrdinal] = false;
  reservations_[consumerOrdinal].reset();
  return llvm::Error::success();
}

llvm::Error OrderedChannelSequence::close() {
  if (closed_)
    return invalid("channel was closed twice");
  closed_ = true;
  return llvm::Error::success();
}

std::uint64_t OrderedChannelSequence::nextReceiveSequence(
    std::uint32_t consumerOrdinal) const {
  return consumerOrdinal < nextReceiveSequences_.size()
             ? nextReceiveSequences_[consumerOrdinal]
             : std::numeric_limits<std::uint64_t>::max();
}

void OrderedChannelSequence::reclaimCommittedPrefix() {
  while (!messages_.empty()) {
    const Message &message = messages_.front();
    if (!llvm::all_of(message.committed, [](bool committed) {
          return committed;
        }))
      break;
    occupiedBytes_ -= message.payload.size();
    messages_.pop_front();
  }
}

} // namespace loom::runtime
