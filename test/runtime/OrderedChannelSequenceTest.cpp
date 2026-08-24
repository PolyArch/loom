#include "Runtime/OrderedChannelSequence.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace loom::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "OrderedChannelSequenceTest: " << message << '\n';
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

void multicastAndRateConversion() {
  OrderedChannelSequence channel = take(OrderedChannelSequence::create(2, 2));
  require(channel.canPublish(2), "empty channel rejected its exact capacity");
  const std::vector<std::uint8_t> first{1, 2};
  const std::vector<std::uint8_t> second{3, 4};
  require(take(channel.publish(first)) == 0, "first SendSeq is not zero");
  require(channel.canPublish(1), "one retained message lost one free slot");
  require(take(channel.publish(second)) == 1, "second SendSeq is not one");
  require(!channel.canPublish(1), "full channel admitted another message");

  auto consumer0 = take(channel.reserve(0));
  require(consumer0.kind == OrderedChannelReceiveKind::Message &&
              consumer0.sequence == 0 && consumer0.payload == first,
          "consumer zero did not receive SendSeq zero");
  require(!channel.commit(0, 0), "consumer zero commit failed");

  auto consumer1 = take(channel.reserve(1));
  require(consumer1.kind == OrderedChannelReceiveKind::Message &&
              consumer1.sequence == 0 && consumer1.payload == first,
          "consumer one did not receive the multicast first message");
  require(!channel.commit(1, 0), "consumer one commit failed");
  require(channel.retainedMessageCount() == 1 &&
              channel.retainedBytes() == second.size(),
          "uncommitted multicast message was reclaimed incorrectly");

  // Consumer zero advances at a different rate, but cannot skip SendSeq one.
  auto next0 = take(channel.reserve(0));
  require(next0.sequence == 1 && next0.payload == second,
          "rate-converted consumer skipped an ordered message");
  require(!channel.cancel(0, 1), "reservation cancellation failed");
  auto retry0 = take(channel.reserve(0));
  require(retry0.sequence == 1 && retry0.payload == second,
          "cancelled reservation did not remain retryable");
  require(!channel.commit(0, 1), "consumer zero second commit failed");

  auto next1 = take(channel.reserve(1));
  require(next1.sequence == 1 && next1.payload == second,
          "consumer one did not preserve sequence order");
  require(!channel.commit(1, 1), "consumer one second commit failed");
  require(channel.retainedMessageCount() == 0 && channel.retainedBytes() == 0,
          "committed prefix was not reclaimed");
  require(channel.canPublish(2), "reclaimed capacity was not reusable");
}

void CapacityAndClose() {
  OrderedChannelSequence channel = take(OrderedChannelSequence::create(1, 1));
  const std::vector<std::uint8_t> payload{7, 8};
  require(take(channel.publish(payload)) == 0,
          "capacity fixture publish failed");
  auto full = channel.publish({9});
  require(!full, "capacity exhaustion overwrote a live message");
  llvm::consumeError(full.takeError());
  auto receive = take(channel.reserve(0));
  require(receive.kind == OrderedChannelReceiveKind::Message,
          "capacity fixture discarded its live message");
  require(!channel.commit(0, 0), "capacity fixture commit failed");
  require(take(channel.publish({9})) == 1,
          "reclaimed capacity was not reusable");
  require(!channel.close(), "channel close failed");
  auto wrapped = take(channel.reserve(0));
  require(wrapped.kind == OrderedChannelReceiveKind::Message &&
              wrapped.sequence == 1 &&
              wrapped.payload == std::vector<std::uint8_t>{9},
          "capacity reuse changed the ordered message");
  require(!channel.commit(0, 1), "reused-capacity commit failed");
  auto closed = take(channel.reserve(0));
  require(closed.kind == OrderedChannelReceiveKind::Closed,
          "closed channel did not expose typed end state");
}

} // namespace

int main() {
  multicastAndRateConversion();
  CapacityAndClose();
  return EXIT_SUCCESS;
}
