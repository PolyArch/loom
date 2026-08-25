#include "Runtime/OrderedChannelABI.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <limits>
#include <utility>

namespace loom::runtime {
namespace {

llvm::Error reject(OrderedChannelABIError::Kind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<OrderedChannelABIError>(kind, message.str());
}

} // namespace

char OrderedChannelABIError::ID = 0;

OrderedChannelABIError::OrderedChannelABIError(Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void OrderedChannelABIError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code OrderedChannelABIError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<OrderedChannelABI>
OrderedChannelABI::create(std::uint64_t capacityMessages,
                          std::uint32_t consumerCount) {
  if (capacityMessages == 0)
    return reject(OrderedChannelABIError::Kind::InvalidConfiguration,
                  "ordered channel capacity must be positive");
  if (consumerCount == 0)
    return reject(OrderedChannelABIError::Kind::InvalidConfiguration,
                  "ordered channel must have at least one consumer");
  return OrderedChannelABI(capacityMessages, consumerCount);
}

llvm::Error
OrderedChannelABI::validateConsumer(std::uint32_t consumerOrdinal) const {
  if (consumerOrdinal >= consumerCount())
    return reject(OrderedChannelABIError::Kind::InvalidConsumer,
                  "ordered channel receive names an unknown consumer");
  return llvm::Error::success();
}

OrderedChannelSend
OrderedChannelABI::send(llvm::ArrayRef<std::uint8_t> payload) {
  if (nextSendSequence_ == std::numeric_limits<std::uint64_t>::max())
    return {OrderedChannelSendKind::SequenceExhausted, nextSendSequence_};
  if (messages_.size() >= capacityMessages_)
    return {OrderedChannelSendKind::WouldBlock, nextSendSequence_};
  Message message;
  message.sequence = nextSendSequence_++;
  message.payload.assign(payload.begin(), payload.end());
  messages_.push_back(std::move(message));
  return {OrderedChannelSendKind::Accepted, nextSendSequence_ - 1};
}

OrderedChannelABI::Message *
OrderedChannelABI::findMessage(std::uint64_t sequence) {
  for (Message &message : messages_)
    if (message.sequence == sequence)
      return &message;
  return nullptr;
}

llvm::Expected<OrderedChannelReceiveTicket>
OrderedChannelABI::receive(std::uint32_t consumerOrdinal) {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return std::move(error);
  if (reservations_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::OutstandingReservation,
                  "ordered channel consumer already has a live reservation");
  const std::uint64_t sequence = nextReceiveSequences_[consumerOrdinal];
  Message *message = findMessage(sequence);
  if (!message) {
    OrderedChannelReceiveTicket ticket;
    ticket.consumerOrdinal = consumerOrdinal;
    ticket.sequence = sequence;
    return ticket;
  }
  if (nextReservationIdentity_ == std::numeric_limits<std::uint64_t>::max())
    return reject(OrderedChannelABIError::Kind::ReservationIdentityExhausted,
                  "ordered channel reservation identity domain is exhausted");
  const std::uint64_t reservationIdentity = nextReservationIdentity_++;
  reservations_[consumerOrdinal] = Reservation{sequence, reservationIdentity};
  OrderedChannelReceiveTicket ticket;
  ticket.kind = OrderedChannelReceiveKind::Message;
  ticket.consumerOrdinal = consumerOrdinal;
  ticket.sequence = sequence;
  ticket.payload = message->payload;
  ticket.ownerIdentity_ = ownerIdentity_;
  ticket.reservationIdentity_ = reservationIdentity;
  return ticket;
}

llvm::Error OrderedChannelABI::validateTicket(
    const OrderedChannelReceiveTicket &ticket) const {
  if (ticket.kind != OrderedChannelReceiveKind::Message ||
      ticket.consumerOrdinal >= consumerCount() ||
      ticket.ownerIdentity_ != ownerIdentity_ ||
      !reservations_[ticket.consumerOrdinal] ||
      reservations_[ticket.consumerOrdinal]->sequence != ticket.sequence ||
      reservations_[ticket.consumerOrdinal]->identity !=
          ticket.reservationIdentity_)
    return reject(OrderedChannelABIError::Kind::InvalidTicket,
                  "ordered channel ticket does not name a live reservation");
  return llvm::Error::success();
}

llvm::Error
OrderedChannelABI::acknowledge(const OrderedChannelReceiveTicket &ticket) {
  if (llvm::Error error = validateTicket(ticket))
    return error;
  assert(findMessage(ticket.sequence) &&
         nextReceiveSequences_[ticket.consumerOrdinal] == ticket.sequence &&
         "live reservation must name its consumer cursor");
  reservations_[ticket.consumerOrdinal].reset();
  ++nextReceiveSequences_[ticket.consumerOrdinal];
  reclaimAcknowledgedPrefix();
  return llvm::Error::success();
}

llvm::Error
OrderedChannelABI::cancel(const OrderedChannelReceiveTicket &ticket) {
  if (llvm::Error error = validateTicket(ticket))
    return error;
  reservations_[ticket.consumerOrdinal].reset();
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
OrderedChannelABI::nextReceiveSequence(std::uint32_t consumerOrdinal) const {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return std::move(error);
  return nextReceiveSequences_[consumerOrdinal];
}

void OrderedChannelABI::reclaimAcknowledgedPrefix() {
  while (!messages_.empty()) {
    const Message &message = messages_.front();
    if (!llvm::all_of(nextReceiveSequences_, [&](std::uint64_t sequence) {
          return sequence > message.sequence;
        }))
      break;
    messages_.pop_front();
  }
}

} // namespace loom::runtime
