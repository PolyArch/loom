#ifndef LOOM_SYSTEMCOMPILER_NOCSCHEDULER_H
#define LOOM_SYSTEMCOMPILER_NOCSCHEDULER_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// NoC Route
//===----------------------------------------------------------------------===//

/// A single route for an inter-core contract edge through the NoC mesh.
struct NoCRoute {
  std::string contractEdgeName;
  std::string producerCore;
  std::string consumerCore;

  /// Sequence of (row, col) positions along the route, including src and dst.
  std::vector<std::pair<int, int>> hops;
  unsigned numHops = 0;
  unsigned bandwidthFlitsPerCycle = 0;

  /// Pipeline latency: hops * router_pipeline_stages.
  unsigned transferLatencyCycles = 0;

  /// Total payload in flits: data_volume / flit_width.
  uint64_t totalFlits = 0;

  /// Serialization time: totalFlits / link_bandwidth.
  unsigned transferDurationCycles = 0;

  /// Producer and consumer core instance indices.
  unsigned producerCoreIdx = 0;
  unsigned consumerCoreIdx = 0;
};

//===----------------------------------------------------------------------===//
// NoC Schedule
//===----------------------------------------------------------------------===//

/// Per-link utilization information.
struct LinkUtilization {
  std::pair<int, int> srcNode;
  std::pair<int, int> dstNode;
  double utilization = 0.0;
  std::vector<std::string> contracts;
};

/// Complete NoC schedule for all inter-core transfers.
struct NoCSchedule {
  std::vector<NoCRoute> routes;
  std::vector<LinkUtilization> linkUtilizations;

  double maxLinkUtilization = 0.0;
  double avgLinkUtilization = 0.0;
  unsigned totalTransferCycles = 0;
  bool hasContention = false;
};

//===----------------------------------------------------------------------===//
// NoC Scheduler
//===----------------------------------------------------------------------===//

/// Options for the NoC scheduler.
struct NoCSchedulerOptions {
  enum RoutingPolicy { XY_DOR, YX_DOR };
  RoutingPolicy routing = XY_DOR;
  bool enableTDM = true;
  unsigned tdmSlotCycles = 1;
  bool verbose = false;
};

/// Computes NoC routes and schedules for inter-core data transfers.
///
/// Given a core assignment and contract edges, the scheduler:
/// 1. Identifies cross-core contract edges.
/// 2. Computes dimension-ordered (XY or YX) routes on the mesh.
/// 3. Estimates per-route transfer latency and duration.
/// 4. Tracks per-link bandwidth utilization and detects contention.
class NoCScheduler {
public:
  /// Schedule all inter-core transfers.
  NoCSchedule schedule(const AssignmentResult &assignment,
                       const std::vector<ContractSpec> &contracts,
                       const SystemArchitecture &arch,
                       const NoCSchedulerOptions &opts);
};

//===----------------------------------------------------------------------===//
// Routing Utilities
//===----------------------------------------------------------------------===//

/// Compute XY dimension-ordered route from src to dst on a mesh.
/// Returns a sequence of (row, col) positions including src and dst.
std::vector<std::pair<int, int>> computeXYRoute(std::pair<int, int> src,
                                                std::pair<int, int> dst);

/// Compute YX dimension-ordered route from src to dst on a mesh.
std::vector<std::pair<int, int>> computeYXRoute(std::pair<int, int> src,
                                                std::pair<int, int> dst);

/// Get the (row, col) position of a core instance on the NoC mesh.
std::pair<int, int> corePosition(unsigned coreInstanceIdx,
                                 unsigned meshCols);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_NOCSCHEDULER_H
