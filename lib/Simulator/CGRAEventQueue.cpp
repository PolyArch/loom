#include "Simulator/CGRA/EventQueue.h"

#include <algorithm>
#include <limits>
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

llvm::Expected<SpatialEventCoordinate>
nextSpatialDelta(const SpatialEventCoordinate &coordinate) {
  if (coordinate.delta == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "CGRA delta cycle overflows u64");
  return SpatialEventCoordinate{coordinate.referenceCycle,
                                coordinate.delta + 1};
}

void CgraEventQueue::schedule(CgraScheduledEvent event) {
  heap_.push_back(std::move(event));
  std::push_heap(heap_.begin(), heap_.end(), LaterEvent{});
}

std::optional<SpatialEventCoordinate> CgraEventQueue::nextCoordinate() const {
  if (heap_.empty())
    return std::nullopt;
  return heap_.front().order.coordinate;
}

llvm::Expected<std::optional<CgraEventFrameView>>
CgraEventQueue::popNextFrameView() {
  frameEvents_.clear();
  if (heap_.empty())
    return std::optional<CgraEventFrameView>{};

  CgraScheduledEvent first = popMinimum(heap_);
  const SpatialEventCoordinate coordinate = first.order.coordinate;
  frameEvents_.push_back(std::move(first));
  while (!heap_.empty() && compareSpatialEventCoordinates(
                               heap_.front().order.coordinate, coordinate) == 0)
    frameEvents_.push_back(popMinimum(heap_));

  for (std::size_t ordinal = 1; ordinal < frameEvents_.size(); ++ordinal)
    if (compareEventKeys(frameEvents_[ordinal - 1].order,
                         frameEvents_[ordinal].order) == 0) {
      const CgraEventOrderKey &key = frameEvents_[ordinal].order;
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s queue contains duplicate key action=%llu "
          "occurrence=%llu owner_event=%u delta=%llu",
          owner_.c_str(),
          static_cast<unsigned long long>(key.structuralActionOrdinal),
          static_cast<unsigned long long>(key.occurrenceOrdinal),
          key.ownerEventOrdinal,
          static_cast<unsigned long long>(key.coordinate.delta));
    }
  return std::optional<CgraEventFrameView>(
      CgraEventFrameView{coordinate, frameEvents_});
}

} // namespace loom::sim
