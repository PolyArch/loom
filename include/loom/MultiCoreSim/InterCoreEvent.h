#ifndef LOOM_MULTICORESIM_INTERCOREEVENT_H
#define LOOM_MULTICORESIM_INTERCOREEVENT_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Inter-Core Transfer Descriptor
//===----------------------------------------------------------------------===//

/// Describes a pending inter-core data transfer through the NoC.
struct PendingTransfer {
  /// Name of the contract edge that initiated this transfer.
  std::string contractEdgeName;

  /// Source and destination core names.
  std::string srcCoreName;
  std::string dstCoreName;

  /// Source and destination core indices.
  unsigned srcCoreIdx = 0;
  unsigned dstCoreIdx = 0;

  /// Payload data being transferred.
  std::vector<uint64_t> data;

  /// Global cycle at which this transfer was initiated.
  uint64_t startCycle = 0;

  /// Remaining cycles until the transfer completes delivery.
  unsigned remainingCycles = 0;

  /// Total latency assigned to this transfer (for statistics).
  unsigned totalLatencyCycles = 0;

  /// NoC port index on the destination core.
  unsigned dstNoCPortIdx = 0;
};

//===----------------------------------------------------------------------===//
// NoC Link Descriptor
//===----------------------------------------------------------------------===//

/// Describes a directed link in the NoC mesh for utilization tracking.
struct NoCLinkState {
  int srcRow = 0;
  int srcCol = 0;
  int dstRow = 0;
  int dstCol = 0;

  /// Number of flits traversing this link in the current cycle.
  uint64_t flitsThisCycle = 0;

  /// Maximum flits per cycle this link supports.
  unsigned bandwidthFlitsPerCycle = 1;

  /// Cumulative flits transferred across this link.
  uint64_t totalFlits = 0;

  /// Number of cycles with at least one flit on this link.
  uint64_t activeCycles = 0;
};

//===----------------------------------------------------------------------===//
// Inter-Core Transfer Route
//===----------------------------------------------------------------------===//

/// Describes the route and buffer allocation for one contract edge.
struct InterCoreRoute {
  std::string contractEdgeName;
  std::string producerCoreName;
  std::string consumerCoreName;
  unsigned producerCoreIdx = 0;
  unsigned consumerCoreIdx = 0;

  /// NoC port indices on producer and consumer cores.
  unsigned producerNoCPortIdx = 0;
  unsigned consumerNoCPortIdx = 0;

  /// Sequence of (row, col) hops through the mesh.
  std::vector<std::pair<int, int>> hops;

  /// Transfer latency in cycles (based on hop count and pipeline stages).
  unsigned transferLatencyCycles = 0;

  /// Bandwidth in flits per cycle on each link.
  unsigned bandwidthFlitsPerCycle = 1;

  /// Buffer size in elements at the destination.
  unsigned bufferSizeElements = 0;
};

//===----------------------------------------------------------------------===//
// Transfer Statistics
//===----------------------------------------------------------------------===//

/// Aggregate statistics for all inter-core transfers during a simulation run.
struct TransferStats {
  uint64_t totalTransfersInitiated = 0;
  uint64_t totalTransfersCompleted = 0;
  uint64_t totalFlitsTransferred = 0;
  uint64_t totalTransferCycles = 0;
  uint64_t contentionStallCycles = 0;
  double avgLinkUtilization = 0.0;
  double maxLinkUtilization = 0.0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_INTERCOREEVENT_H
