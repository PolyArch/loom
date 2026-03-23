#ifndef LOOM_MULTICORESIM_ROUTERMODEL_H
#define LOOM_MULTICORESIM_ROUTERMODEL_H

#include "loom/MultiCoreSim/FlitModel.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Router Direction Enum
//===----------------------------------------------------------------------===//

/// Cardinal directions for mesh router ports.
enum class RouterDir : unsigned {
  NORTH = 0,
  EAST = 1,
  SOUTH = 2,
  WEST = 3,
  LOCAL = 4,
  NUM_DIRS = 5,
};

/// Return the opposite direction (NORTH <-> SOUTH, EAST <-> WEST).
/// LOCAL maps to LOCAL.
RouterDir reverseDirection(RouterDir dir);

/// String name for a direction (for debug printing).
const char *directionName(RouterDir dir);

//===----------------------------------------------------------------------===//
// RouterConfig
//===----------------------------------------------------------------------===//

/// Configuration parameters for a single router.
struct RouterConfig {
  /// Number of virtual channels per port.
  unsigned numVCs = 2;

  /// Buffer depth (in flits) per VC per input port.
  unsigned bufferDepth = 4;

  /// Router pipeline stages (route compute + switch traversal = 2).
  unsigned pipelineStages = 2;
};

//===----------------------------------------------------------------------===//
// RouterModel -- Per-router cycle-accurate model
//===----------------------------------------------------------------------===//

/// Models a single router in the mesh NoC.
///
/// Pipeline: 2-stage (Route Compute + Switch Traversal).
///   Stage 1 (RC): Determine output direction via XY routing.
///   Stage 2 (ST): Arbitrate among inputs and move winning flit to output.
///
/// Flow control: Credit-based. Each output port tracks downstream buffer
/// credits. A flit can only be sent if the downstream VC has available credits.
///
/// Arbitration: Round-robin among input ports requesting the same output.
class RouterModel {
public:
  static constexpr unsigned kNumDirs =
      static_cast<unsigned>(RouterDir::NUM_DIRS);

  RouterModel();
  RouterModel(unsigned routerId, int row, int col, const RouterConfig &cfg);

  // --- Accessors ---
  unsigned getRouterId() const { return routerId_; }
  int getRow() const { return row_; }
  int getCol() const { return col_; }
  const RouterConfig &getConfig() const { return config_; }

  // --- Input buffer interface ---

  /// Check if the input buffer for (dir, vc) can accept a flit.
  bool canAcceptFlit(RouterDir dir, unsigned vc) const;

  /// Push a flit into an input buffer (called by upstream router or local
  /// injection). Returns false if the buffer is full.
  bool acceptFlit(RouterDir dir, unsigned vc, const SimFlit &flit);

  // --- Credit interface ---

  /// Return a credit to an output port's VC (called when downstream consumes
  /// a flit). This restores capacity for sending on that output port/vc.
  void returnCredit(RouterDir outputDir, unsigned vc);

  /// Set the initial credit count for an output port/vc (called during
  /// initialization based on downstream buffer depth).
  void initCredits(RouterDir outputDir, unsigned vc, unsigned credits);

  /// Get current credits for an output port/vc.
  unsigned getCredits(RouterDir outputDir, unsigned vc) const;

  // --- Pipeline execution (called by NoCSimModel per cycle) ---

  /// Stage 1: Route Compute.
  /// For each input that has a HEAD/SINGLE flit, compute the output direction
  /// using XY routing and store the result.
  void stageRouteCompute();

  /// Stage 2: Switch Arbitration and Traversal.
  /// Arbitrate among inputs requesting the same output, then move the winning
  /// flits into transit storage. Returns flits that should be forwarded to
  /// neighbor routers or delivered locally.
  struct ForwardedFlit {
    SimFlit flit;
    RouterDir outputDir;
    unsigned vc;
  };
  std::vector<ForwardedFlit> stageSwitchTraversal();

  // --- Local ejection ---

  /// Check if there are flits in the local delivery queue.
  bool hasLocalDelivery() const;

  /// Dequeue a flit from the local delivery queue.
  SimFlit dequeueLocalDelivery();

  // --- Statistics ---

  struct RouterStats {
    uint64_t totalFlitsRouted = 0;
    uint64_t totalStallCycles = 0;
    uint64_t totalArbitrationConflicts = 0;
  };
  const RouterStats &getStats() const { return stats_; }

private:
  /// Compute XY output direction for a flit destined to (dstRow, dstCol).
  RouterDir computeXYRoute(int dstRow, int dstCol) const;

  unsigned routerId_ = 0;
  int row_ = 0;
  int col_ = 0;
  RouterConfig config_;

  // --- Input ports ---
  // Indexed as [dir][vc].
  struct InputPort {
    SimFlitBuffer buffer;
    /// Computed output direction for the current head flit (set during RC
    /// stage, consumed during ST stage). nullopt if no route computed yet.
    std::optional<RouterDir> computedRoute;
    /// Whether this input/vc is locked by a wormhole message in progress.
    bool wormholeLocked = false;
    RouterDir wormholeOutputDir = RouterDir::LOCAL;
  };
  std::vector<std::vector<InputPort>> inputPorts_; // [dir][vc]

  // --- Output port credits ---
  // Indexed as [dir][vc]. Tracks how many flits can be sent downstream.
  std::vector<std::vector<unsigned>> outputCredits_; // [dir][vc]

  // --- Arbitration state ---
  // Round-robin priority pointer per output direction.
  std::vector<unsigned> arbiterPriority_; // [dir]

  // --- Local delivery queue ---
  std::deque<SimFlit> localDeliveryQueue_;

  // --- Statistics ---
  RouterStats stats_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_ROUTERMODEL_H
