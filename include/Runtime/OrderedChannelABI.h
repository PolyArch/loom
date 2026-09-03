#ifndef LOOM_RUNTIME_ORDEREDCHANNELABI_H
#define LOOM_RUNTIME_ORDEREDCHANNELABI_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace loom::runtime {

enum class OrderedChannelSendKind : std::uint8_t {
  Accepted,
  WouldBlock,
  SequenceExhausted,
  StaticRateExceeded,
  InvalidLifecycle,
  GenerationCancelled,
};

struct OrderedChannelSend final {
  OrderedChannelSendKind kind = OrderedChannelSendKind::WouldBlock;
  std::uint64_t sequence = 0;
};

enum class OrderedChannelReceiveKind : std::uint8_t {
  Message,
  WouldBlock,
  EndOfGeneration,
};

/// A receive reservation and its acknowledgement coordinates. Generation and
/// private transient identity bind copies to the exact live reservation and
/// owner.
struct OrderedChannelReceiveTicket final {
  OrderedChannelReceiveKind kind = OrderedChannelReceiveKind::WouldBlock;
  std::uint32_t consumerOrdinal = 0;
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::vector<std::uint8_t> payload;

private:
  friend class OrderedChannelABI;

  std::shared_ptr<const void> ownerIdentity_;
  std::uint64_t reservationIdentity_ = 0;
};

class OrderedChannelABIError final
    : public llvm::ErrorInfo<OrderedChannelABIError> {
public:
  enum class Kind {
    InvalidConfiguration,
    InvalidConsumer,
    WouldBlock,
    SequenceExhausted,
    OutstandingReservation,
    ReservationIdentityExhausted,
    InvalidTicket,
    InvalidLifecycle,
    GenerationDeficit,
    StaticRateExceeded,
    PendingConsumer,
    GenerationCancelled,
    StaleGeneration,
    GenerationIdentityExhausted,
  };

  static char ID;

  OrderedChannelABIError(Kind kind, std::string message);

  Kind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// Converts a rejected send result into the corresponding typed ABI error.
/// Accepted sends return success; callers do not reconstruct or stringify
/// send-state distinctions.
llvm::Error orderedChannelSendError(OrderedChannelSendKind kind);

/// Direct invocation-local ordered-channel ABI and sequence owner. It owns the
/// FIFO cursors, reservations, acknowledgements, multicast retention, and
/// bounded capacity without creating persistent channel or session identity.
/// The caller must submit sends in the canonical event commit order; this
/// owner does not choose ordering from concurrent arrival. Optional static
/// rates close one transient generation before reset opens the next.
class OrderedChannelABI final {
public:
  static llvm::Expected<OrderedChannelABI>
  create(std::uint64_t capacityMessages, std::uint32_t consumerCount);

  OrderedChannelABI(OrderedChannelABI &&) noexcept = default;
  OrderedChannelABI &operator=(OrderedChannelABI &&) noexcept = default;
  OrderedChannelABI(const OrderedChannelABI &) = delete;
  OrderedChannelABI &operator=(const OrderedChannelABI &) = delete;

  OrderedChannelSend send(llvm::ArrayRef<std::uint8_t> payload);

  /// Declares the static flat event counts for the current pristine
  /// generation. Creation and reset already leave one generation open; this
  /// call only supplies its optional terminal-count contract.
  llvm::Error
  openGeneration(std::uint64_t producerMessageCount,
                 llvm::ArrayRef<std::uint64_t> consumerMessageCounts);

  llvm::Error finishProducer();
  llvm::Error finishConsumer(std::uint32_t consumerOrdinal);
  llvm::Error joinGeneration();
  llvm::Error cancelGeneration();
  llvm::Error reset();

  std::uint64_t generation() const { return generation_; }
  bool hasStaticRateContract() const { return expectedMessages_.has_value(); }
  bool generationJoined() const;
  llvm::Expected<bool>
  consumerFinished(std::uint32_t consumerOrdinal) const;

  llvm::Expected<OrderedChannelReceiveTicket>
  receive(std::uint32_t consumerOrdinal);

  llvm::Error acknowledge(const OrderedChannelReceiveTicket &ticket);
  llvm::Error cancel(const OrderedChannelReceiveTicket &ticket);

  std::uint32_t consumerCount() const {
    return static_cast<std::uint32_t>(nextReceiveSequences_.size());
  }
  std::uint64_t nextSendSequence() const { return nextSendSequence_; }
  llvm::Expected<std::uint64_t>
  nextReceiveSequence(std::uint32_t consumerOrdinal) const;

private:
  enum class GenerationState : std::uint8_t {
    Open,
    Closing,
    Complete,
    Cancelled,
  };

  struct Message final {
    std::uint64_t sequence = 0;
    std::vector<std::uint8_t> payload;
  };

  struct Reservation final {
    std::uint64_t sequence = 0;
    std::uint64_t identity = 0;
  };

  OrderedChannelABI(std::uint64_t capacityMessages, std::uint32_t consumerCount)
      : capacityMessages_(capacityMessages),
        ownerIdentity_(std::make_shared<const std::uint8_t>(0)),
        nextReceiveSequences_(consumerCount, 0), reservations_(consumerCount),
        finishedConsumers_(consumerCount, false),
        expectedConsumerMessages_(consumerCount) {}

  llvm::Error validateConsumer(std::uint32_t consumerOrdinal) const;
  llvm::Error validateTicket(const OrderedChannelReceiveTicket &ticket) const;
  llvm::Error validateActiveGeneration() const;
  Message *findMessage(std::uint64_t sequence);
  void reclaimAcknowledgedPrefix();

  std::uint64_t capacityMessages_ = 0;
  std::shared_ptr<const void> ownerIdentity_;
  std::uint64_t nextReservationIdentity_ = 1;
  std::uint64_t generation_ = 0;
  std::uint64_t nextSendSequence_ = 0;
  bool producerFinished_ = false;
  std::vector<std::uint64_t> nextReceiveSequences_;
  std::vector<std::optional<Reservation>> reservations_;
  std::vector<bool> finishedConsumers_;
  std::optional<std::uint64_t> expectedMessages_;
  std::vector<std::optional<std::uint64_t>> expectedConsumerMessages_;
  GenerationState generationState_ = GenerationState::Open;
  std::deque<Message> messages_;
};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_ORDEREDCHANNELABI_H
