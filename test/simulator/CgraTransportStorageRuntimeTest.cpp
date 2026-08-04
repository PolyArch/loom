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

void bufferedQueuePreservesAtomicCycleSemantics() {
  auto queue = take(CgraTransportStorageRuntime::create(2));
  require(queue.empty() && queue.capacity() == 2,
          "new queue does not expose its exact empty capacity");

  llvm::Error emptyDequeue = queue.commit(std::nullopt, true).takeError();
  require(static_cast<bool>(emptyDequeue), "empty dequeue was accepted");
  llvm::consumeError(std::move(emptyDequeue));

  auto first = take(queue.commit(CgraTransportStorageEntry{7, 70}, false));
  require(!first.dequeued && first.enqueued && queue.occupancy() == 1,
          "enqueue did not append one durable entry");

  auto replace = take(queue.commit(CgraTransportStorageEntry{8, 80}, true));
  require(replace.dequeued && replace.dequeued->transferSlot == 7 &&
              replace.dequeued->traversalNodeOrdinal == 70 &&
              replace.enqueued && queue.occupancy() == 1,
          "simultaneous dequeue/enqueue did not use cycle-start head state");

  auto second = take(queue.commit(std::nullopt, true));
  require(second.dequeued && second.dequeued->transferSlot == 8 &&
              queue.empty(),
          "newly enqueued entry bypassed FIFO order");

  (void)take(queue.commit(CgraTransportStorageEntry{9, 90}, false));
  (void)take(queue.commit(CgraTransportStorageEntry{10, 100}, false));
  llvm::Error fullEnqueue =
      queue.commit(CgraTransportStorageEntry{11, 110}, false).takeError();
  require(static_cast<bool>(fullEnqueue), "full enqueue was accepted");
  llvm::consumeError(std::move(fullEnqueue));

  auto fullReplace =
      take(queue.commit(CgraTransportStorageEntry{11, 110}, true));
  require(fullReplace.dequeued && fullReplace.dequeued->transferSlot == 9 &&
              queue.occupancy() == 2,
          "same-cycle dequeue did not make one full-queue slot available");
  auto retained = take(queue.commit(std::nullopt, true));
  auto appended = take(queue.commit(std::nullopt, true));
  require(retained.dequeued && retained.dequeued->transferSlot == 10 &&
              appended.dequeued && appended.dequeued->transferSlot == 11 &&
              queue.empty(),
          "full-queue replacement changed canonical FIFO order");
}

} // namespace

int main() {
  bufferedQueuePreservesAtomicCycleSemantics();
  return EXIT_SUCCESS;
}
