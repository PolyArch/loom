#include "CgraTransportStorageRuntime.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace {

using loom::sim::detail::CgraTransportStorageEntry;
using loom::sim::detail::CgraTransportStorageRuntime;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA transport storage runtime test: " << message << '\n';
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

void bufferedQueueUsesCycleStartCapacity() {
  auto queue = take(CgraTransportStorageRuntime::create(2));
  require(queue.empty() && queue.capacity() == 2,
          "new queue does not expose its exact empty capacity");
  require(!queue.offeredEntry(), "an empty queue presents an entry");

  llvm::Error emptyDequeue =
      queue.commit(std::nullopt, CgraTransportStorageEntry{7, 70}).takeError();
  require(static_cast<bool>(emptyDequeue), "empty dequeue was accepted");
  llvm::consumeError(std::move(emptyDequeue));

  auto first = take(queue.commit(CgraTransportStorageEntry{7, 70},
                                 std::nullopt));
  require(!first.dequeued && first.enqueued && queue.occupancy() == 1,
          "enqueue did not append one durable entry");
  require(queue.offeredEntry() &&
              queue.offeredEntry()->transferSlot == 7,
          "a strict FIFO does not present its oldest entry");

  auto replace =
      take(queue.commit(CgraTransportStorageEntry{8, 80},
                        queue.offeredEntry()));
  require(replace.dequeued && replace.dequeued->transferSlot == 7 &&
              replace.dequeued->traversalNodeOrdinal == 70 &&
              replace.enqueued && queue.occupancy() == 1,
          "simultaneous dequeue/enqueue did not use cycle-start head state");

  auto second = take(queue.commit(std::nullopt, queue.offeredEntry()));
  require(second.dequeued && second.dequeued->transferSlot == 8 &&
              queue.empty(),
          "newly enqueued entry bypassed FIFO order");

  (void)take(queue.commit(CgraTransportStorageEntry{9, 90}, std::nullopt));
  (void)take(queue.commit(CgraTransportStorageEntry{10, 100}, std::nullopt));
  llvm::Error fullEnqueue =
      queue.commit(CgraTransportStorageEntry{11, 110}, std::nullopt)
          .takeError();
  require(static_cast<bool>(fullEnqueue), "full enqueue was accepted");
  llvm::consumeError(std::move(fullEnqueue));

  llvm::Error fullReplace =
      queue.commit(CgraTransportStorageEntry{11, 110}, queue.offeredEntry())
          .takeError();
  require(static_cast<bool>(fullReplace),
          "full queue borrowed current-cycle dequeue capacity");
  llvm::consumeError(std::move(fullReplace));

  auto released = take(queue.commit(std::nullopt, queue.offeredEntry()));
  require(released.dequeued && released.dequeued->transferSlot == 9 &&
              queue.occupancy() == 1,
          "full-queue dequeue did not release next-cycle capacity");
  auto nextCycle =
      take(queue.commit(CgraTransportStorageEntry{11, 110}, std::nullopt));
  require(nextCycle.enqueued && queue.occupancy() == 2,
          "released capacity was unavailable in the following cycle");
  auto retained = take(queue.commit(std::nullopt, queue.offeredEntry()));
  auto appended = take(queue.commit(std::nullopt, queue.offeredEntry()));
  require(retained.dequeued && retained.dequeued->transferSlot == 10 &&
              appended.dequeued && appended.dequeued->transferSlot == 11 &&
              queue.empty(),
          "full-queue replacement changed canonical FIFO order");
}

void independentStorageAllowsFullReplacement() {
  auto storage = take(CgraTransportStorageRuntime::create(
      1, /*fullReplacementAllowed=*/true));
  (void)take(storage.commit(CgraTransportStorageEntry{1, 10}, std::nullopt));
  auto replacement =
      take(storage.commit(CgraTransportStorageEntry{2, 20},
                          storage.offeredEntry()));
  require(replacement.dequeued &&
              replacement.dequeued->transferSlot == 1 &&
              replacement.enqueued && storage.full() &&
              storage.front().transferSlot == 2,
          "independently serviced storage lost full replacement");
}

} // namespace

int main() {
  bufferedQueueUsesCycleStartCapacity();
  independentStorageAllowsFullReplacement();
  return EXIT_SUCCESS;
}
