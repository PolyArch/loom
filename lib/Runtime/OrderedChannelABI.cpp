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

llvm::Error OrderedChannelABI::validateActiveGeneration() const {
  if (generationState_ == GenerationState::Cancelled)
    return reject(OrderedChannelABIError::Kind::GenerationCancelled,
                  "ordered channel generation is cancelled");
  if (generationState_ == GenerationState::Complete)
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel generation is complete");
  return llvm::Error::success();
}

OrderedChannelSend
OrderedChannelABI::send(llvm::ArrayRef<std::uint8_t> payload) {
  if (generationState_ == GenerationState::Cancelled)
    return {OrderedChannelSendKind::GenerationCancelled, nextSendSequence_};
  if (generationState_ == GenerationState::Complete || producerFinished_)
    return {OrderedChannelSendKind::InvalidLifecycle, nextSendSequence_};
  if (expectedMessages_ && nextSendSequence_ >= *expectedMessages_)
    return {OrderedChannelSendKind::StaticRateExceeded, nextSendSequence_};
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

llvm::Error OrderedChannelABI::openGeneration(
    std::uint64_t producerMessageCount,
    llvm::ArrayRef<std::uint64_t> consumerMessageCounts) {
  if (llvm::Error error = validateActiveGeneration())
    return error;
  if (expectedMessages_ || nextSendSequence_ != 0 ||
      llvm::any_of(nextReceiveSequences_,
                   [](std::uint64_t sequence) { return sequence != 0; }) ||
      llvm::any_of(
          reservations_,
          [](const auto &reservation) { return reservation.has_value(); }) ||
      producerFinished_ || llvm::is_contained(finishedConsumers_, true) ||
      !messages_.empty())
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel generation is not pristine");
  if (consumerMessageCounts.size() != consumerCount())
    return reject(OrderedChannelABIError::Kind::InvalidConfiguration,
                  "ordered channel static rates omit a consumer");
  expectedMessages_ = producerMessageCount;
  for (const auto indexed : llvm::enumerate(consumerMessageCounts))
    expectedConsumerMessages_[indexed.index()] = indexed.value();
  return llvm::Error::success();
}

llvm::Error OrderedChannelABI::finishProducer() {
  if (llvm::Error error = validateActiveGeneration())
    return error;
  if (!expectedMessages_ || producerFinished_)
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel producer cannot finish now");
  if (nextSendSequence_ < *expectedMessages_)
    return reject(OrderedChannelABIError::Kind::GenerationDeficit,
                  "ordered channel producer has not published its static rate");
  producerFinished_ = true;
  generationState_ = GenerationState::Closing;
  return llvm::Error::success();
}

llvm::Error OrderedChannelABI::finishConsumer(std::uint32_t consumerOrdinal) {
  if (llvm::Error error = validateConsumer(consumerOrdinal))
    return error;
  if (llvm::Error error = validateActiveGeneration())
    return error;
  if (!expectedConsumerMessages_[consumerOrdinal] ||
      finishedConsumers_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel consumer cannot finish now");
  if (reservations_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::OutstandingReservation,
                  "ordered channel consumer has a live reservation");
  const std::uint64_t expected = *expectedConsumerMessages_[consumerOrdinal];
  const std::uint64_t observed = nextReceiveSequences_[consumerOrdinal];
  if (observed < expected)
    return reject(OrderedChannelABIError::Kind::GenerationDeficit,
                  "ordered channel consumer has not received its static rate");
  finishedConsumers_[consumerOrdinal] = true;
  generationState_ = GenerationState::Closing;
  return llvm::Error::success();
}

llvm::Error OrderedChannelABI::joinGeneration() {
  if (llvm::Error error = validateActiveGeneration())
    return error;
  if (!expectedMessages_ || !producerFinished_)
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel producer has not finished");
  if (llvm::any_of(reservations_, [](const auto &reservation) {
        return reservation.has_value();
      }))
    return reject(OrderedChannelABIError::Kind::OutstandingReservation,
                  "ordered channel generation has a live reservation");
  if (llvm::is_contained(finishedConsumers_, false) || !messages_.empty())
    return reject(OrderedChannelABIError::Kind::PendingConsumer,
                  "ordered channel generation has a pending consumer");
  generationState_ = GenerationState::Complete;
  return llvm::Error::success();
}

llvm::Error OrderedChannelABI::cancelGeneration() {
  if (generationState_ == GenerationState::Cancelled)
    return reject(OrderedChannelABIError::Kind::GenerationCancelled,
                  "ordered channel generation is already cancelled");
  if (generationState_ == GenerationState::Complete)
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel completed generation cannot be cancelled");
  for (std::optional<Reservation> &reservation : reservations_)
    reservation.reset();
  messages_.clear();
  generationState_ = GenerationState::Cancelled;
  return llvm::Error::success();
}

llvm::Error OrderedChannelABI::reset() {
  if (generationState_ != GenerationState::Complete &&
      generationState_ != GenerationState::Cancelled) {
    if (llvm::any_of(reservations_, [](const auto &reservation) {
          return reservation.has_value();
        }))
      return reject(OrderedChannelABIError::Kind::OutstandingReservation,
                    "ordered channel generation has a live reservation");
    if (producerFinished_ && llvm::is_contained(finishedConsumers_, false))
      return reject(OrderedChannelABIError::Kind::PendingConsumer,
                    "ordered channel generation has not joined");
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel active generation cannot be reset");
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max())
    return reject(OrderedChannelABIError::Kind::GenerationIdentityExhausted,
                  "ordered channel generation identity domain is exhausted");
  ++generation_;
  nextReservationIdentity_ = 1;
  nextSendSequence_ = 0;
  producerFinished_ = false;
  llvm::fill(nextReceiveSequences_, 0);
  for (std::optional<Reservation> &reservation : reservations_)
    reservation.reset();
  llvm::fill(finishedConsumers_, false);
  expectedMessages_.reset();
  for (std::optional<std::uint64_t> &expected : expectedConsumerMessages_)
    expected.reset();
  messages_.clear();
  generationState_ = GenerationState::Open;
  return llvm::Error::success();
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
  if (llvm::Error error = validateActiveGeneration())
    return std::move(error);
  if (finishedConsumers_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel consumer is already finished");
  if (reservations_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::OutstandingReservation,
                  "ordered channel consumer already has a live reservation");
  const std::uint64_t sequence = nextReceiveSequences_[consumerOrdinal];
  if (expectedConsumerMessages_[consumerOrdinal] &&
      sequence >= *expectedConsumerMessages_[consumerOrdinal])
    return reject(OrderedChannelABIError::Kind::StaticRateExceeded,
                  "ordered channel receive exceeds its static rate");
  Message *message = findMessage(sequence);
  if (!message) {
    OrderedChannelReceiveTicket ticket;
    ticket.consumerOrdinal = consumerOrdinal;
    ticket.sequence = sequence;
    ticket.generation = generation_;
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
  ticket.generation = generation_;
  ticket.payload = message->payload;
  ticket.ownerIdentity_ = ownerIdentity_;
  ticket.reservationIdentity_ = reservationIdentity;
  return ticket;
}

llvm::Error OrderedChannelABI::validateTicket(
    const OrderedChannelReceiveTicket &ticket) const {
  if (ticket.kind != OrderedChannelReceiveKind::Message ||
      ticket.consumerOrdinal >= consumerCount() ||
      ticket.ownerIdentity_ != ownerIdentity_)
    return reject(OrderedChannelABIError::Kind::InvalidTicket,
                  "ordered channel ticket names a foreign reservation");
  if (ticket.generation != generation_)
    return reject(OrderedChannelABIError::Kind::StaleGeneration,
                  "ordered channel ticket names a stale generation");
  if (generationState_ == GenerationState::Cancelled)
    return reject(OrderedChannelABIError::Kind::GenerationCancelled,
                  "ordered channel ticket names a cancelled generation");
  if (generationState_ == GenerationState::Complete)
    return reject(OrderedChannelABIError::Kind::InvalidLifecycle,
                  "ordered channel generation is complete");
  if (!reservations_[ticket.consumerOrdinal] ||
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
