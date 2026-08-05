#include "Simulator/CGRA/EventQueue.h"

#include "Evaluation/NumericValue.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA event queue test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::sim::SpatialEventCoordinate coordinate(std::uint64_t numerator,
                                             std::uint64_t denominator,
                                             std::uint64_t delta = 0) {
  return {take(loom::evaluation::ExactRatio::get(numerator, denominator)),
          delta};
}

loom::sim::CgraScheduledEvent event(loom::sim::SpatialEventCoordinate at,
                                    std::uint64_t action,
                                    std::uint64_t occurrence,
                                    std::uint32_t ownerEvent,
                                    std::uint64_t payload) {
  return {{std::move(at), action, occurrence, ownerEvent}, payload};
}

void exactCoordinatesAndStructuralKeysDetermineOrder() {
  loom::sim::CgraEventQueue queue;
  queue.schedule(event(coordinate(2, 3), 0, 0, 0, 5));
  queue.schedule(event(coordinate(1, 2), 2, 0, 0, 4));
  queue.schedule(event(coordinate(2, 4), 1, 3, 0, 3));
  queue.schedule(event(coordinate(1, 2), 1, 2, 1, 2));
  queue.schedule(event(coordinate(1, 2), 1, 2, 0, 1));
  queue.schedule(event(coordinate(1, 2, 1), 0, 0, 0, 6));

  const auto firstCoordinate = queue.nextCoordinate();
  if (!firstCoordinate ||
      firstCoordinate->referenceCycle !=
          take(loom::evaluation::ExactRatio::get(1, 2)) ||
      firstCoordinate->delta != 0 || queue.size() != 6)
    fail("next-coordinate projection consumed or reordered the queue");

  auto first = take(queue.popNextFrameView());
  if (!first || first->events.size() != 4 ||
      first->coordinate.referenceCycle !=
          take(loom::evaluation::ExactRatio::get(1, 2)) ||
      first->coordinate.delta != 0)
    fail("first exact-coordinate frame is malformed");
  for (std::uint64_t ordinal = 0; ordinal != first->events.size(); ++ordinal)
    if (first->events[ordinal].payload != ordinal + 1)
      fail("same-coordinate events ignored structural ordering");

  auto second = take(queue.popNextFrameView());
  if (!second || second->events.size() != 1 ||
      second->events.front().payload != 6)
    fail("delta order changed");
  auto third = take(queue.popNextFrameView());
  if (!third || third->events.size() != 1 || third->events.front().payload != 5)
    fail("exact rational order changed");
  if (take(queue.popNextFrameView()))
    fail("empty event queue returned a frame");
}

void duplicateCanonicalEventKeyIsRejected() {
  loom::sim::CgraEventQueue queue;
  queue.schedule(event(coordinate(7, 1), 4, 9, 2, 1));
  queue.schedule(event(coordinate(7, 1), 4, 9, 2, 2));
  auto frame = queue.popNextFrameView();
  if (frame)
    fail("duplicate canonical event key was accepted");
  llvm::consumeError(frame.takeError());
}

} // namespace

int main() {
  exactCoordinatesAndStructuralKeysDetermineOrder();
  duplicateCanonicalEventKeyIsRejected();
  return EXIT_SUCCESS;
}
