#include "Simulator/CGRA/EventQueue.h"

#include <algorithm>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace {

int compareEventKeys(const CgraEventOrderKey &lhs,
                     const CgraEventOrderKey &rhs) {
  if (const int coordinate =
          compareSpatialEventCoordinates(lhs.coordinate, rhs.coordinate))
    return coordinate;
  if (lhs.structuralActionOrdinal != rhs.structuralActionOrdinal)
    return lhs.structuralActionOrdinal < rhs.structuralActionOrdinal ? -1 : 1;
  if (lhs.occurrenceOrdinal != rhs.occurrenceOrdinal)
    return lhs.occurrenceOrdinal < rhs.occurrenceOrdinal ? -1 : 1;
  if (lhs.ownerEventOrdinal == rhs.ownerEventOrdinal)
    return 0;
  return lhs.ownerEventOrdinal < rhs.ownerEventOrdinal ? -1 : 1;
}

struct LaterEvent final {
  bool operator()(const CgraScheduledEvent &lhs,
                  const CgraScheduledEvent &rhs) const {
    return compareEventKeys(lhs.order, rhs.order) > 0;
  }
};

CgraScheduledEvent popMinimum(std::vector<CgraScheduledEvent> &heap) {
  std::pop_heap(heap.begin(), heap.end(), LaterEvent{});
  CgraScheduledEvent result = std::move(heap.back());
  heap.pop_back();
  return result;
}

} // namespace

void CgraEventQueue::schedule(CgraScheduledEvent event) {
  heap_.push_back(std::move(event));
  std::push_heap(heap_.begin(), heap_.end(), LaterEvent{});
}

llvm::Expected<std::optional<CgraEventFrame>> CgraEventQueue::popNextFrame() {
  if (heap_.empty())
    return std::optional<CgraEventFrame>{};

  CgraScheduledEvent first = popMinimum(heap_);
  CgraEventFrame frame{first.order.coordinate, {}};
  frame.events.push_back(std::move(first));
  while (!heap_.empty() &&
         compareSpatialEventCoordinates(heap_.front().order.coordinate,
                                        frame.coordinate) == 0)
    frame.events.push_back(popMinimum(heap_));

  for (std::size_t ordinal = 1; ordinal < frame.events.size(); ++ordinal)
    if (compareEventKeys(frame.events[ordinal - 1].order,
                         frame.events[ordinal].order) == 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CGRA event queue contains a duplicate canonical event key");
  return std::optional<CgraEventFrame>(std::move(frame));
}

} // namespace loom::sim
