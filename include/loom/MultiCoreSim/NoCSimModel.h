#ifndef LOOM_MULTICORESIM_NOCSIMMODEL_H
#define LOOM_MULTICORESIM_NOCSIMMODEL_H

#include "loom/MultiCoreSim/TapestryTypes.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

// A flit is the basic unit of data that travels through the NoC.
struct Flit {
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;
  unsigned channelId = 0;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
  uint64_t injectionCycle = 0;
  unsigned flitIndex = 0;
  unsigned totalFlits = 0;
  bool isHead = false;
  bool isTail = false;
};

// Tracks per-link usage for utilization statistics.
struct LinkUsage {
  unsigned srcRow = 0;
  unsigned srcCol = 0;
  unsigned dstRow = 0;
  unsigned dstCol = 0;
  uint64_t flitsTransferred = 0;
  uint64_t activeCycles = 0;
};

// Simple cycle-accurate NoC model using XY-routing over a 2D mesh.
// Each link has a bandwidth of 1 flit per cycle, and a configurable
// per-hop latency.
class NoCSimModel {
public:
  NoCSimModel(unsigned meshRows, unsigned meshCols,
              unsigned perHopLatency = 1);
  ~NoCSimModel();

  NoCSimModel(const NoCSimModel &) = delete;
  NoCSimModel &operator=(const NoCSimModel &) = delete;

  // Configure routing from a NoCSchedule.
  std::string configure(const NoCSchedule &schedule);

  // Map a core ID to a (row, col) position in the mesh.
  // By default, cores are placed in row-major order.
  void setCorePosition(unsigned coreId, unsigned row, unsigned col);

  // Inject a flit into the NoC from a source core.
  void injectFlit(const Flit &flit);

  // Advance the NoC simulation by one cycle.
  void stepOneCycle();

  // Check if any flit has arrived at the given destination core.
  bool hasArrivedFlits(unsigned dstCoreId) const;

  // Drain all arrived flits for the given destination core.
  std::vector<Flit> drainArrivedFlits(unsigned dstCoreId);

  // Check if the NoC is idle (no in-flight flits).
  bool isIdle() const;

  // Get the current cycle count.
  uint64_t getCurrentCycle() const { return currentCycle_; }

  // Gather statistics.
  NoCStats getStats() const;

private:
  // Internal flit-in-flight representation.
  struct InFlightFlit {
    Flit flit;
    unsigned currentRow = 0;
    unsigned currentCol = 0;
    unsigned targetRow = 0;
    unsigned targetCol = 0;
    uint64_t arrivalCycle = 0;
    unsigned remainingHopLatency = 0;
  };

  std::pair<unsigned, unsigned> getCorePosition(unsigned coreId) const;
  void routeFlit(InFlightFlit &inflight);

  unsigned meshRows_ = 1;
  unsigned meshCols_ = 1;
  unsigned perHopLatency_ = 1;
  uint64_t currentCycle_ = 0;

  // Core ID to (row, col) mapping.
  std::vector<std::pair<unsigned, unsigned>> corePositions_;

  // In-flight flits currently traversing the mesh.
  std::deque<InFlightFlit> inFlightFlits_;

  // Flits that have arrived at their destination, keyed by dstCoreId.
  std::vector<std::deque<Flit>> arrivedFlits_;

  // Statistics.
  uint64_t totalFlitsInjected_ = 0;
  uint64_t totalFlitsDelivered_ = 0;
  uint64_t totalHops_ = 0;
  uint64_t totalLatency_ = 0;
  std::vector<LinkUsage> linkUsage_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_NOCSIMMODEL_H
