#include "Runtime/OrderedChannelABI.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "OrderedChannelABITest: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectABIError(llvm::Error error, OrderedChannelABIError::Kind expected,
                    const std::string &message) {
  std::optional<OrderedChannelABIError::Kind> actual;
  llvm::handleAllErrors(
      std::move(error),
      [&](const OrderedChannelABIError &failure) { actual = failure.kind(); });
  require(actual && *actual == expected, message);
}

void multicastCancellationAndBackpressure() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(1, 2));
  const std::vector<std::uint8_t> first{5};
  const auto sent = channel.send(first);
  require(sent.kind == OrderedChannelSendKind::Accepted && sent.sequence == 0,
          "direct send did not return its accepted SendSeq");
  const auto blocked = channel.send({6});
  require(blocked.kind == OrderedChannelSendKind::WouldBlock &&
              blocked.sequence == 1 && channel.nextSendSequence() == 1,
          "direct send hid bounded backpressure or advanced SendSeq");

  auto left = take(channel.receive(0));
  require(left.kind == OrderedChannelReceiveKind::Message &&
              left.consumerOrdinal == 0 && left.sequence == 0 &&
              left.payload == first,
          "direct receive did not return its endpoint reservation");
  auto right = take(channel.receive(1));
  require(right.kind == OrderedChannelReceiveKind::Message &&
              right.sequence == left.sequence && right.payload == first,
          "multicast receive changed sequence correspondence");
  require(!channel.acknowledge(left),
          "acknowledgement for the first branch failed");
  auto duplicateAck = channel.acknowledge(left);
  expectABIError(std::move(duplicateAck),
                 OrderedChannelABIError::Kind::InvalidTicket,
                 "duplicate acknowledgement was not typed");
  auto waitingLeft = take(channel.receive(0));
  require(
      waitingLeft.kind == OrderedChannelReceiveKind::WouldBlock,
      "a committed branch received ahead of the retained multicast message");
  require(!channel.cancel(right), "receive cancellation failed");
  auto retriedRight = take(channel.receive(1));
  require(retriedRight.kind == OrderedChannelReceiveKind::Message &&
              retriedRight.sequence == right.sequence &&
              retriedRight.payload == right.payload,
          "cancelled receive did not remain retryable");
  require(!channel.acknowledge(retriedRight),
          "acknowledgement for the second branch failed");

  const auto second = channel.send({6});
  require(second.kind == OrderedChannelSendKind::Accepted &&
              second.sequence == 1,
          "direct send did not reuse released capacity");
  auto secondLeft = take(channel.receive(0));
  auto secondRight = take(channel.receive(1));
  require(secondLeft.sequence == 1 && secondRight.sequence == 1,
          "independent branch rates skipped the second SendSeq");
  require(!channel.acknowledge(secondLeft),
          "second left acknowledgement failed");
  require(!channel.acknowledge(secondRight),
          "second right acknowledgement failed");
  const auto third = channel.send({7});
  require(third.kind == OrderedChannelSendKind::Accepted && third.sequence == 2,
          "fully acknowledged message did not release capacity");
  auto thirdLeft = take(channel.receive(0));
  auto thirdRight = take(channel.receive(1));
  require(!channel.acknowledge(thirdLeft), "third left acknowledgement failed");
  require(!channel.acknowledge(thirdRight),
          "third right acknowledgement failed");
  require(take(channel.nextReceiveSequence(0)) == channel.nextSendSequence() &&
              take(channel.nextReceiveSequence(1)) ==
                  channel.nextSendSequence(),
          "branch cursors did not expose the complete flat sequence");
}

void flatSequenceIgnoresCallerRateGroups() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(4, 2));
  for (std::uint8_t payload = 0; payload != 4; ++payload) {
    const auto sent = channel.send({payload});
    require(sent.kind == OrderedChannelSendKind::Accepted &&
                sent.sequence == payload,
            "producer rate group changed its flat SendSeq");
  }

  // One caller consumes all messages as a group; the other represents four
  // one-message groups. Both must observe the same flat sequence.
  for (std::uint64_t sequence = 0; sequence != 4; ++sequence) {
    auto grouped = take(channel.receive(0));
    require(grouped.sequence == sequence && grouped.payload.size() == 1 &&
                grouped.payload.front() == sequence,
            "multi-message consumer group lost flat correspondence");
    require(!channel.acknowledge(grouped),
            "multi-message consumer acknowledgement failed");
  }
  for (std::uint64_t sequence = 0; sequence != 4; ++sequence) {
    auto individual = take(channel.receive(1));
    require(individual.sequence == sequence && individual.payload.size() == 1 &&
                individual.payload.front() == sequence,
            "single-message consumer groups lost flat correspondence");
    require(!channel.acknowledge(individual),
            "single-message consumer acknowledgement failed");
  }
}

void invalidTicketsAndConsumersDoNotMutateState() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(1, 1));
  auto invalidConsumer = channel.receive(1);
  require(!invalidConsumer, "direct ABI accepted an unknown consumer");
  expectABIError(invalidConsumer.takeError(),
                 OrderedChannelABIError::Kind::InvalidConsumer,
                 "unknown consumer did not retain its typed outcome");

  require(channel.send({9}).sequence == 0, "ticket fixture publication failed");
  auto ticket = take(channel.receive(0));
  auto duplicateReceive = channel.receive(0);
  require(!duplicateReceive, "consumer acquired two simultaneous reservations");
  expectABIError(duplicateReceive.takeError(),
                 OrderedChannelABIError::Kind::OutstandingReservation,
                 "duplicate reservation did not retain its typed outcome");
  auto foreign = ticket;
  ++foreign.sequence;
  auto foreignError = channel.acknowledge(foreign);
  expectABIError(std::move(foreignError),
                 OrderedChannelABIError::Kind::InvalidTicket,
                 "foreign sequence ticket was not rejected atomically");
  require(!channel.cancel(ticket), "ticket cancellation failed");
  auto replacement = take(channel.receive(0));
  auto revokedError = channel.acknowledge(ticket);
  expectABIError(std::move(revokedError),
                 OrderedChannelABIError::Kind::InvalidTicket,
                 "cancelled ticket acknowledged a replacement reservation");

  OrderedChannelABI peer = take(OrderedChannelABI::create(1, 1));
  require(peer.send({8}).kind == OrderedChannelSendKind::Accepted,
          "peer ticket fixture publication failed");
  auto peerTicket = take(peer.receive(0));
  auto foreignOwnerError = peer.acknowledge(replacement);
  expectABIError(std::move(foreignOwnerError),
                 OrderedChannelABIError::Kind::InvalidTicket,
                 "ticket crossed ordered channel owners");
  require(!channel.acknowledge(replacement),
          "revoked ticket disturbed the replacement reservation");
  require(!peer.acknowledge(peerTicket),
          "foreign-owner ticket disturbed the peer reservation");
}

} // namespace

int main() {
  multicastCancellationAndBackpressure();
  flatSequenceIgnoresCallerRateGroups();
  invalidTicketsAndConsumersDoNotMutateState();
  return EXIT_SUCCESS;
}
