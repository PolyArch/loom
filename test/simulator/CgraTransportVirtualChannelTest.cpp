#include "CGRATransportPlan.h"
#include "CgraTransportStorageRuntime.h"

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

namespace {

using loom::sim::detail::CgraPhysicalTagPlan;
using loom::sim::detail::CgraTransportStorageEntry;
using loom::sim::detail::CgraTransportStorageRuntime;
using loom::sim::detail::internPhysicalTagChannelRanks;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA transport virtual channel test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

CgraTransportStorageRuntime virtualChannel(std::uint32_t capacity) {
  return take(CgraTransportStorageRuntime::create(
      capacity, false, ::fabric::FifoQueueDiscipline::PerTagVirtualChannel));
}

CgraTransportStorageEntry entry(std::uint64_t slot, std::uint64_t tagOrdinal,
                                std::uint32_t channelKey) {
  return CgraTransportStorageEntry{slot, /*traversalNodeOrdinal=*/slot,
                                   tagOrdinal, channelKey};
}

std::uint64_t offeredSlot(const CgraTransportStorageRuntime &queue) {
  const auto offered = queue.offeredEntry();
  if (!offered)
    fail("the queue presents no entry");
  return offered->transferSlot;
}

void grantOffered(CgraTransportStorageRuntime &queue) {
  (void)take(queue.commit(std::nullopt, queue.offeredEntry()));
}

/// The first offer names the lowest resident channel, and a refused offer
/// presents the next channel on the next evaluation.
void blockedChannelIsBypassed() {
  auto queue = virtualChannel(4);
  (void)take(queue.commit(entry(1, 0, 2), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 1), std::nullopt));
  require(offeredSlot(queue) == 2 && queue.offerCursor() == 0,
          "the first offer did not name the lowest resident channel");
  queue.advanceOffer();
  require(offeredSlot(queue) == 1,
          "a refused offer did not yield the port to the next channel");
  require(queue.offerCursor() == 2, "the cursor did not move past channel 1");
  grantOffered(queue);
  require(offeredSlot(queue) == 2,
          "the blocked channel head did not survive the grant of its peer");
}

/// Equal tag values share one channel and keep arrival order no matter which
/// plan segment ordinal carried the value.
void equalTagValuesKeepChannelOrder() {
  auto queue = virtualChannel(4);
  (void)take(queue.commit(entry(1, 5, 2), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 1), std::nullopt));
  (void)take(queue.commit(entry(3, 9, 2), std::nullopt));
  require(queue.distinctResidentChannels() == 2,
          "equal tag values did not share one channel");
  // Channel 2 holds slots 1 and 3; channel 1 is offered first, then channel 2
  // must present slot 1 before slot 3.
  require(offeredSlot(queue) == 2, "lowest channel was not offered first");
  grantOffered(queue);
  require(offeredSlot(queue) == 1,
          "the older same-channel token did not head its channel");
  grantOffered(queue);
  require(offeredSlot(queue) == 3,
          "a same-channel token overtook its older peer");
}

/// With N resident channels and every consumer refusing, the port presents
/// each channel exactly once in N evaluations and then wraps, so no channel
/// starves.
void sustainedBacklogRotatesFairly() {
  auto queue = virtualChannel(3);
  (void)take(queue.commit(entry(1, 0, 3), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 1), std::nullopt));
  (void)take(queue.commit(entry(3, 2, 2), std::nullopt));
  require(queue.distinctResidentChannels() == 3, "channel count differs");
  require(offeredSlot(queue) == 2, "rotation did not start at channel 1");
  queue.advanceOffer();
  require(offeredSlot(queue) == 3, "rotation skipped channel 2");
  queue.advanceOffer();
  require(offeredSlot(queue) == 1, "rotation skipped channel 3");
  queue.advanceOffer();
  require(offeredSlot(queue) == 2 &&
              queue.offerCursor() == 4,
          "a full rotation did not wrap the scan to the lowest channel");
}

/// A channel that drains leaves the rotation and later re-enters at its
/// canonical position, and the emptied queue returns to canonical state.
void channelDrainAndReentry() {
  auto queue = virtualChannel(2);
  (void)take(queue.commit(entry(1, 0, 1), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 2), std::nullopt));
  grantOffered(queue); // channel 1
  require(offeredSlot(queue) == 2, "the surviving channel was not offered");
  grantOffered(queue); // channel 2, queue drains
  require(!queue.offeredEntry() && queue.empty(),
          "a drained queue still presents an entry");
  require(queue.offerCursor() == 3,
          "the cursor did not move past the granted channel");
  (void)take(queue.commit(entry(3, 2, 1), std::nullopt));
  require(offeredSlot(queue) == 3,
          "a refilled channel did not re-enter through cursor wraparound");
  require(queue.offerCursor() == 3,
          "an enqueue must not move the offer cursor");
}

/// The shared pool bounds every channel together; a full queue admits neither
/// a plain enqueue nor a same-cycle replacement, and a dequeue releases its
/// slot only for the following cycle.
void sharedPoolCapacityIsExact() {
  auto queue = virtualChannel(2);
  (void)take(queue.commit(entry(1, 0, 1), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 2), std::nullopt));
  require(queue.full(), "occupancy did not reach the shared capacity");
  llvm::Error fullEnqueue = queue.commit(entry(3, 2, 3), std::nullopt)
                                .takeError();
  require(static_cast<bool>(fullEnqueue),
          "a full virtual channel pool accepted an enqueue");
  llvm::consumeError(std::move(fullEnqueue));
  llvm::Error fullReplace =
      queue.commit(entry(3, 2, 3), queue.offeredEntry()).takeError();
  require(static_cast<bool>(fullReplace),
          "a full queue borrowed current-cycle dequeue capacity");
  llvm::consumeError(std::move(fullReplace));
  grantOffered(queue);
  require(!queue.full() && queue.occupancy() == 1,
          "a grant did not release one shared slot");
  (void)take(queue.commit(entry(3, 2, 3), std::nullopt));
  require(queue.full(), "the released slot was unavailable next cycle");
}

/// Below capacity, one dequeue and one enqueue of distinct channels complete
/// together, and the grant moves the cursor past the dequeued channel.
void simultaneousGrantUsesOfferedChannel() {
  auto queue = virtualChannel(4);
  (void)take(queue.commit(entry(1, 0, 2), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 1), std::nullopt));
  const auto granted = take(queue.commit(entry(3, 2, 3), queue.offeredEntry()));
  require(granted.dequeued && granted.dequeued->transferSlot == 2 &&
              granted.enqueued && queue.occupancy() == 2,
          "simultaneous dequeue/enqueue did not commit atomically");
  require(queue.offerCursor() == 2,
          "a grant did not move the cursor to the channel successor");
  require(offeredSlot(queue) == 1,
          "the cursor did not resume at the successor channel");
}

/// A dequeue that names an entry the discipline does not present is rejected.
void unofferedEntryCannotDequeue() {
  auto queue = virtualChannel(4);
  (void)take(queue.commit(entry(1, 0, 1), std::nullopt));
  (void)take(queue.commit(entry(2, 1, 2), std::nullopt));
  llvm::Error skip = queue.commit(std::nullopt, entry(2, 1, 2)).takeError();
  require(static_cast<bool>(skip),
          "a channel tail overtook the offered channel head");
  llvm::consumeError(std::move(skip));
  require(queue.occupancy() == 2, "a rejected dequeue removed an entry");
}

/// The plan tag-value interning shares one rank per equal value and orders
/// ranks by the canonical unsigned tag value order.
void planTagInterningFollowsCanonicalOrder() {
  const auto tag = [](std::uint64_t value) {
    return CgraPhysicalTagPlan{llvm::APInt(4, value)};
  };
  const std::vector<CgraPhysicalTagPlan> tags = {tag(5), tag(2), tag(5),
                                                 tag(9), tag(2)};
  const std::vector<std::uint32_t> ranks = internPhysicalTagChannelRanks(tags);
  require(ranks == std::vector<std::uint32_t>({1, 0, 1, 2, 0}),
          "equal tag values did not share one canonical rank");
  // A cold rebuild from the same plan reproduces the cache exactly.
  require(internPhysicalTagChannelRanks(tags) == ranks,
          "cold tag-rank rebuild disagreed with the cache");
  require(::fabric::comparePhysicalTagValues(tags[4].value, tags[0].value) ==
                  ::fabric::comparePhysicalTagValues(tags[1].value,
                                                     tags[0].value),
          "the dense rank order diverged from the tag value order");
}

} // namespace

int main() {
  blockedChannelIsBypassed();
  equalTagValuesKeepChannelOrder();
  sustainedBacklogRotatesFairly();
  channelDrainAndReentry();
  sharedPoolCapacityIsExact();
  simultaneousGrantUsesOfferedChannel();
  unofferedEntryCannotDequeue();
  planTagInterningFollowsCanonicalOrder();
  return EXIT_SUCCESS;
}
