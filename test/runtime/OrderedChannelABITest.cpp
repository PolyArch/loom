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
  auto cancelledCopy = ticket;
  require(!channel.cancel(ticket), "ticket cancellation failed");
  auto replacement = take(channel.receive(0));
  auto revokedError = channel.acknowledge(cancelledCopy);
  expectABIError(
      std::move(revokedError), OrderedChannelABIError::Kind::InvalidTicket,
      "cancelled ticket copy acknowledged a replacement reservation");

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

void boundedGenerationCompletesAndRepeats() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(2, 2));
  require(channel.generation() == 0, "initial generation is not canonical");
  require(!channel.openGeneration(2, {2, 2}),
          "static generation rates were rejected");
  const auto firstSend = channel.send({3});
  const auto secondSend = channel.send({4});
  require(firstSend.kind == OrderedChannelSendKind::Accepted &&
              secondSend.kind == OrderedChannelSendKind::Accepted,
          "bounded generation did not publish its static producer rate");
  require(channel.send({5}).kind == OrderedChannelSendKind::StaticRateExceeded,
          "producer exceeded the declared generation rate");
  require(!channel.finishProducer(), "complete producer could not finish");

  auto firstLeft = take(channel.receive(0));
  auto outstanding = channel.finishConsumer(0);
  expectABIError(std::move(outstanding),
                 OrderedChannelABIError::Kind::OutstandingReservation,
                 "consumer finished with a live reservation");
  auto staleCandidate = firstLeft;
  require(!channel.acknowledge(firstLeft),
          "first generation acknowledgement failed");
  auto secondLeft = take(channel.receive(0));
  require(!channel.acknowledge(secondLeft),
          "left branch did not consume its static rate");
  auto excessReceive = channel.receive(0);
  require(!excessReceive, "consumer exceeded the declared generation rate");
  expectABIError(excessReceive.takeError(),
                 OrderedChannelABIError::Kind::StaticRateExceeded,
                 "excess receive did not retain its typed outcome");
  require(!channel.finishConsumer(0), "left branch could not finish");
  auto pending = channel.joinGeneration();
  expectABIError(std::move(pending),
                 OrderedChannelABIError::Kind::PendingConsumer,
                 "generation joined before every multicast branch");

  auto firstRight = take(channel.receive(1));
  require(!channel.acknowledge(firstRight),
          "first right acknowledgement failed");
  auto secondRight = take(channel.receive(1));
  require(!channel.acknowledge(secondRight),
          "right branch did not consume its static rate");
  require(!channel.finishConsumer(1), "right branch could not finish");
  require(!channel.joinGeneration(), "complete generation did not join");
  require(channel.send({6}).kind == OrderedChannelSendKind::InvalidLifecycle,
          "completed generation accepted another send");

  require(!channel.reset(), "completed generation did not reset");
  require(channel.generation() == 1 && channel.nextSendSequence() == 0 &&
              take(channel.nextReceiveSequence(0)) == 0 &&
              take(channel.nextReceiveSequence(1)) == 0,
          "reset did not open a fresh generation");
  auto stale = channel.acknowledge(staleCandidate);
  expectABIError(std::move(stale),
                 OrderedChannelABIError::Kind::StaleGeneration,
                 "old receive ticket crossed a generation reset");

  require(!channel.openGeneration(1, {1, 1}),
          "second generation rates were rejected");
  require(channel.send({9}).kind == OrderedChannelSendKind::Accepted,
          "second generation did not restart SendSeq");
  for (std::uint32_t consumer = 0; consumer != 2; ++consumer) {
    auto ticket = take(channel.receive(consumer));
    require(ticket.generation == 1 && ticket.sequence == 0,
            "receive ticket omitted its generation coordinates");
    require(!channel.acknowledge(ticket),
            "second generation acknowledgement failed");
    require(!channel.finishConsumer(consumer),
            "second generation consumer could not finish");
  }
  require(!channel.finishProducer(),
          "second generation producer could not finish");
  require(!channel.joinGeneration(), "second complete generation did not join");
}

void generationDeficitsAndLifecycleAreTyped() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(2, 1));
  require(!channel.openGeneration(2, {2}),
          "deficit fixture rates were rejected");
  auto producerDeficit = channel.finishProducer();
  expectABIError(std::move(producerDeficit),
                 OrderedChannelABIError::Kind::GenerationDeficit,
                 "producer deficit was not typed");
  require(channel.send({1}).kind == OrderedChannelSendKind::Accepted,
          "deficit fixture first send failed");
  producerDeficit = channel.finishProducer();
  expectABIError(std::move(producerDeficit),
                 OrderedChannelABIError::Kind::GenerationDeficit,
                 "partial producer rate was accepted");
  require(channel.send({2}).kind == OrderedChannelSendKind::Accepted,
          "deficit fixture second send failed");
  require(!channel.finishProducer(), "producer could not finish at its rate");

  auto first = take(channel.receive(0));
  require(!channel.acknowledge(first), "deficit fixture receive failed");
  auto consumerDeficit = channel.finishConsumer(0);
  expectABIError(std::move(consumerDeficit),
                 OrderedChannelABIError::Kind::GenerationDeficit,
                 "consumer deficit was not typed");
  auto pending = channel.joinGeneration();
  expectABIError(std::move(pending),
                 OrderedChannelABIError::Kind::PendingConsumer,
                 "unfinished consumer was not typed separately");
  auto second = take(channel.receive(0));
  require(!channel.acknowledge(second),
          "deficit fixture second receive failed");
  require(!channel.finishConsumer(0), "consumer could not finish at its rate");
  require(!channel.joinGeneration(), "balanced generation did not join");
  auto duplicateJoin = channel.joinGeneration();
  expectABIError(std::move(duplicateJoin),
                 OrderedChannelABIError::Kind::InvalidLifecycle,
                 "duplicate generation join was not rejected");

  OrderedChannelABI branchDeficit = take(OrderedChannelABI::create(1, 2));
  require(!branchDeficit.openGeneration(1, {1, 2}),
          "branch-specific static rates were rejected");
  require(branchDeficit.send({3}).kind == OrderedChannelSendKind::Accepted,
          "branch deficit fixture send failed");
  require(!branchDeficit.finishProducer(),
          "branch deficit producer could not finish");
  for (std::uint32_t consumer = 0; consumer != 2; ++consumer) {
    auto received = take(branchDeficit.receive(consumer));
    require(!branchDeficit.acknowledge(received),
            "branch deficit acknowledgement failed");
  }
  require(!branchDeficit.finishConsumer(0),
          "complete branch could not finish independently");
  auto waiting = take(branchDeficit.receive(1));
  require(waiting.kind == OrderedChannelReceiveKind::WouldBlock &&
              waiting.generation == 0 && waiting.sequence == 1,
          "deficit branch did not retain its next receive coordinate");
  auto branchDeficitError = branchDeficit.finishConsumer(1);
  expectABIError(std::move(branchDeficitError),
                 OrderedChannelABIError::Kind::GenerationDeficit,
                 "branch-specific deficit was not typed");
  auto pendingBranch = branchDeficit.joinGeneration();
  expectABIError(std::move(pendingBranch),
                 OrderedChannelABIError::Kind::PendingConsumer,
                 "deficit branch did not keep the generation pending");

  OrderedChannelABI unconfigured = take(OrderedChannelABI::create(1, 2));
  auto omittedRate = unconfigured.openGeneration(1, {1});
  expectABIError(std::move(omittedRate),
                 OrderedChannelABIError::Kind::InvalidConfiguration,
                 "incomplete branch rate table was accepted");
  auto activeReset = unconfigured.reset();
  expectABIError(std::move(activeReset),
                 OrderedChannelABIError::Kind::InvalidLifecycle,
                 "active generation reset was not rejected");
  auto unconfiguredFinish = unconfigured.finishProducer();
  expectABIError(std::move(unconfiguredFinish),
                 OrderedChannelABIError::Kind::InvalidLifecycle,
                 "generation without static rates finished");
}

void cancelledGenerationInvalidatesTicketsAndReopens() {
  OrderedChannelABI channel = take(OrderedChannelABI::create(1, 1));
  require(!channel.openGeneration(1, {1}),
          "cancelled generation rates were rejected");
  require(channel.send({7}).kind == OrderedChannelSendKind::Accepted,
          "cancelled generation fixture send failed");
  auto ticket = take(channel.receive(0));
  auto activeReset = channel.reset();
  expectABIError(std::move(activeReset),
                 OrderedChannelABIError::Kind::OutstandingReservation,
                 "reset ignored a live receive reservation");
  require(!channel.cancelGeneration(), "generation cancellation failed");
  require(channel.send({8}).kind == OrderedChannelSendKind::GenerationCancelled,
          "cancelled generation accepted another send");
  auto cancelledReceive = channel.receive(0);
  require(!cancelledReceive, "cancelled generation accepted another receive");
  expectABIError(cancelledReceive.takeError(),
                 OrderedChannelABIError::Kind::GenerationCancelled,
                 "cancelled receive did not retain its typed outcome");
  auto cancelledTicket = channel.acknowledge(ticket);
  expectABIError(std::move(cancelledTicket),
                 OrderedChannelABIError::Kind::GenerationCancelled,
                 "current cancelled ticket was not typed");
  auto duplicateCancel = channel.cancelGeneration();
  expectABIError(std::move(duplicateCancel),
                 OrderedChannelABIError::Kind::GenerationCancelled,
                 "duplicate generation cancellation was not typed");

  require(!channel.reset(), "cancelled generation did not reset");
  require(channel.generation() == 1,
          "cancelled generation did not advance its identity");
  auto staleTicket = channel.cancel(ticket);
  expectABIError(std::move(staleTicket),
                 OrderedChannelABIError::Kind::StaleGeneration,
                 "cancelled ticket was not stale after reset");
  require(!channel.openGeneration(0, {0}),
          "zero-rate replacement generation was rejected");
  require(!channel.finishProducer(), "zero-rate producer could not finish");
  require(!channel.finishConsumer(0), "zero-rate consumer could not finish");
  require(!channel.joinGeneration(),
          "zero-rate replacement generation did not join");
}

} // namespace

int main() {
  multicastCancellationAndBackpressure();
  flatSequenceIgnoresCallerRateGroups();
  invalidTicketsAndConsumersDoNotMutateState();
  boundedGenerationCompletesAndRepeats();
  generationDeficitsAndLifecycleAreTyped();
  cancelledGenerationInvalidatesTicketsAndReopens();
  return EXIT_SUCCESS;
}
