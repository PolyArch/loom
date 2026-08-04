#ifndef LOOM_SIMULATOR_CGRA_EVENTQUEUE_H
#define LOOM_SIMULATOR_CGRA_EVENTQUEUE_H

#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim {

llvm::Expected<SpatialEventCoordinate>
nextSpatialDelta(const SpatialEventCoordinate &coordinate);

/// Complete execution-local order key. structuralActionOrdinal is assigned by
/// sorting the exact canonical typed action keys during preparation; it is a
/// removable dense cache and never replaces those persistent references.
struct CgraEventOrderKey final {
  SpatialEventCoordinate coordinate;
  std::uint64_t structuralActionOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t ownerEventOrdinal = 0;
};

struct CgraScheduledEvent final {
  CgraEventOrderKey order;
  std::uint64_t payload = 0;
};

struct CgraEventFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraScheduledEvent> events;
};

/// Min-heap event calendar with exact rational coordinates. Equal-coordinate
/// events are emitted in canonical structural order without comparing MLIR
/// pointers, strings, or reference byte vectors in the hot queue.
class CgraEventQueue final {
public:
  void schedule(CgraScheduledEvent event);

  std::optional<SpatialEventCoordinate> nextCoordinate() const;

  llvm::Expected<std::optional<CgraEventFrame>> popNextFrame();

  bool empty() const { return heap_.empty(); }
  std::size_t size() const { return heap_.size(); }

private:
  std::vector<CgraScheduledEvent> heap_;
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRA_EVENTQUEUE_H
