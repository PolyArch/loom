#ifndef LOOM_MULTICORESIM_NOCSIMMODEL_H
#define LOOM_MULTICORESIM_NOCSIMMODEL_H

#include "loom/MultiCoreSim/FlitModel.h"
#include "loom/MultiCoreSim/RouterModel.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// NoCSimConfig -- Configuration for the NoC simulation model
//===----------------------------------------------------------------------===//

struct NoCSimConfig {
  /// Mesh dimensions (rows x cols). Derived from core positions if zero.
  unsigned meshRows = 0;
  unsigned meshCols = 0;

  /// Flit width in bits.
  unsigned flitWidthBits = 32;

  /// Number of virtual channels per port.
  unsigned numVCs = 2;

  /// Buffer depth (in flits) per VC per input port.
  unsigned bufferDepth = 4;

  /// Router pipeline latency in stages.
  unsigned routerPipelineStages = 2;

  /// Link bandwidth in flits per cycle per direction.
  unsigned linkBandwidthFlitsPerCycle = 1;
};

//===----------------------------------------------------------------------===//
// NoCDetailedStats -- Detailed statistics from the NoC simulation
//===----------------------------------------------------------------------===//

struct NoCDetailedStats {
  uint64_t totalFlitsInjected = 0;
  uint64_t totalFlitsDelivered = 0;
  uint64_t totalFlitsInFlight = 0;
  double avgLatencyCycles = 0.0;
  double maxLatencyCycles = 0.0;

  /// Per-link statistics.
  struct LinkStats {
    unsigned srcRouter = 0;
    unsigned dstRouter = 0;
    int srcRow = 0;
    int srcCol = 0;
    int dstRow = 0;
    int dstCol = 0;
    uint64_t flitsTransferred = 0;
    uint64_t stallCycles = 0;
    double utilization = 0.0;
  };
  std::vector<LinkStats> linkStats;

  /// Per-VC statistics.
  struct VCStats {
    unsigned vcId = 0;
    uint64_t flitsTransferred = 0;
    double avgOccupancy = 0.0;
  };
  std::vector<VCStats> vcStats;
};

//===----------------------------------------------------------------------===//
// NoCSimModel -- Top-level mesh NoC simulation model
//===----------------------------------------------------------------------===//

/// Cycle-accurate mesh NoC simulation model.
///
/// Manages a grid of RouterModel instances connected in a 2D mesh topology.
/// Each router is connected to its NORTH, EAST, SOUTH, and WEST neighbors,
/// plus a LOCAL port for core injection/ejection.
///
/// The simulation runs in three phases each cycle:
///   1. Route Compute (RC) -- all routers compute output directions
///   2. Switch Traversal (ST) -- arbitrate and move flits between routers
///   3. Credit return -- downstream routers return credits to upstream
///
/// Cores are mapped to router LOCAL ports via a core-to-position mapping.
class NoCSimModel {
public:
  NoCSimModel();
  ~NoCSimModel();

  // Non-copyable, movable.
  NoCSimModel(const NoCSimModel &) = delete;
  NoCSimModel &operator=(const NoCSimModel &) = delete;
  NoCSimModel(NoCSimModel &&) noexcept;
  NoCSimModel &operator=(NoCSimModel &&) noexcept;

  /// Configure the mesh NoC from the given config and core positions.
  /// corePositions maps coreId -> (row, col) in the mesh.
  void configure(const NoCSimConfig &config,
                 const std::vector<std::pair<int, int>> &corePositions);

  /// Check if a flit can be injected at the given core's local port.
  bool canInject(unsigned srcCoreId, unsigned vcId) const;

  /// Inject a flit at the source core's local router port.
  void inject(const SimFlit &flit);

  /// Check if there are delivered flits waiting at the given core's local port.
  bool hasDelivery(unsigned dstCoreId) const;

  /// Receive (eject) a delivered flit from the given core's local port.
  SimFlit receive(unsigned dstCoreId);

  /// Advance the entire NoC by one cycle.
  void tick(uint64_t globalCycle);

  /// Get detailed statistics.
  NoCDetailedStats getDetailedStats() const;

  /// Get the number of rows in the mesh.
  unsigned getNumRows() const { return numRows_; }

  /// Get the number of columns in the mesh.
  unsigned getNumCols() const { return numCols_; }

  /// Check if the model has been configured.
  bool isConfigured() const { return configured_; }

private:
  /// Get the router at mesh position (row, col).
  RouterModel &routerAt(int row, int col);
  const RouterModel &routerAt(int row, int col) const;

  /// Get the router ID for a mesh position.
  unsigned routerIdAt(int row, int col) const;

  /// Get the neighbor router in the given direction, or nullptr if at the
  /// mesh boundary.
  RouterModel *neighborRouter(int row, int col, RouterDir dir);

  /// Initialize link statistics tracking structures.
  void initLinkStats();

  /// Record a flit traversal on a link for statistics.
  void recordLinkTraversal(int srcRow, int srcCol, int dstRow, int dstCol);

  bool configured_ = false;
  NoCSimConfig config_;
  unsigned numRows_ = 0;
  unsigned numCols_ = 0;

  /// Flattened 2D grid of routers. Index = row * numCols + col.
  std::vector<RouterModel> routers_;

  /// Mapping from core ID to mesh position (row, col).
  std::unordered_map<unsigned, std::pair<int, int>> coreToPosition_;

  /// Mapping from mesh position to core ID (for ejection).
  std::unordered_map<unsigned, unsigned> positionToCore_;

  /// Statistics tracking.
  uint64_t totalFlitsInjected_ = 0;
  uint64_t totalFlitsDelivered_ = 0;
  double totalLatencySum_ = 0.0;
  double maxLatency_ = 0.0;

  /// Per-link flit counters. Key is (srcRouterId * maxRouters + dstRouterId).
  struct LinkCounter {
    int srcRow = 0;
    int srcCol = 0;
    int dstRow = 0;
    int dstCol = 0;
    unsigned srcRouterId = 0;
    unsigned dstRouterId = 0;
    uint64_t flitsTransferred = 0;
    uint64_t stallCycles = 0;
  };
  std::vector<LinkCounter> linkCounters_;

  /// Per-VC statistics accumulators.
  std::vector<uint64_t> vcFlitCounts_;

  /// Total cycles elapsed since configuration.
  uint64_t totalCyclesElapsed_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_NOCSIMMODEL_H
