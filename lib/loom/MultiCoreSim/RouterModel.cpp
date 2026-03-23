#include "loom/MultiCoreSim/RouterModel.h"

#include <algorithm>
#include <cassert>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Direction helpers
//===----------------------------------------------------------------------===//

RouterDir reverseDirection(RouterDir dir) {
  switch (dir) {
  case RouterDir::NORTH:
    return RouterDir::SOUTH;
  case RouterDir::SOUTH:
    return RouterDir::NORTH;
  case RouterDir::EAST:
    return RouterDir::WEST;
  case RouterDir::WEST:
    return RouterDir::EAST;
  case RouterDir::LOCAL:
    return RouterDir::LOCAL;
  default:
    return RouterDir::LOCAL;
  }
}

const char *directionName(RouterDir dir) {
  switch (dir) {
  case RouterDir::NORTH:
    return "NORTH";
  case RouterDir::EAST:
    return "EAST";
  case RouterDir::SOUTH:
    return "SOUTH";
  case RouterDir::WEST:
    return "WEST";
  case RouterDir::LOCAL:
    return "LOCAL";
  default:
    return "UNKNOWN";
  }
}

//===----------------------------------------------------------------------===//
// RouterModel Construction
//===----------------------------------------------------------------------===//

RouterModel::RouterModel() : RouterModel(0, 0, 0, RouterConfig{}) {}

RouterModel::RouterModel(unsigned routerId, int row, int col,
                         const RouterConfig &cfg)
    : routerId_(routerId), row_(row), col_(col), config_(cfg) {
  unsigned numDirs = kNumDirs;
  unsigned numVCs = cfg.numVCs;

  // Initialize input ports: [dir][vc].
  inputPorts_.resize(numDirs);
  for (unsigned d = 0; d < numDirs; ++d) {
    inputPorts_[d].resize(numVCs);
    for (unsigned v = 0; v < numVCs; ++v) {
      inputPorts_[d][v].buffer.setCapacity(cfg.bufferDepth);
      inputPorts_[d][v].computedRoute = std::nullopt;
      inputPorts_[d][v].wormholeLocked = false;
    }
  }

  // Initialize output credits: [dir][vc].
  // Credits are set to 0 initially; they will be initialized by the
  // NoCSimModel based on downstream buffer depth.
  outputCredits_.resize(numDirs);
  for (unsigned d = 0; d < numDirs; ++d) {
    outputCredits_[d].resize(numVCs, 0);
  }

  // Initialize round-robin arbiter priority pointers.
  arbiterPriority_.resize(numDirs, 0);
}

//===----------------------------------------------------------------------===//
// Input buffer interface
//===----------------------------------------------------------------------===//

bool RouterModel::canAcceptFlit(RouterDir dir, unsigned vc) const {
  unsigned d = static_cast<unsigned>(dir);
  assert(d < kNumDirs && "invalid direction");
  assert(vc < config_.numVCs && "invalid VC");
  return inputPorts_[d][vc].buffer.canEnqueue();
}

bool RouterModel::acceptFlit(RouterDir dir, unsigned vc, const SimFlit &flit) {
  unsigned d = static_cast<unsigned>(dir);
  assert(d < kNumDirs && "invalid direction");
  assert(vc < config_.numVCs && "invalid VC");
  return inputPorts_[d][vc].buffer.enqueue(flit);
}

//===----------------------------------------------------------------------===//
// Credit interface
//===----------------------------------------------------------------------===//

void RouterModel::returnCredit(RouterDir outputDir, unsigned vc) {
  unsigned d = static_cast<unsigned>(outputDir);
  assert(d < kNumDirs && "invalid direction");
  assert(vc < config_.numVCs && "invalid VC");
  ++outputCredits_[d][vc];
}

void RouterModel::initCredits(RouterDir outputDir, unsigned vc,
                               unsigned credits) {
  unsigned d = static_cast<unsigned>(outputDir);
  assert(d < kNumDirs && "invalid direction");
  assert(vc < config_.numVCs && "invalid VC");
  outputCredits_[d][vc] = credits;
}

unsigned RouterModel::getCredits(RouterDir outputDir, unsigned vc) const {
  unsigned d = static_cast<unsigned>(outputDir);
  assert(d < kNumDirs && "invalid direction");
  assert(vc < config_.numVCs && "invalid VC");
  return outputCredits_[d][vc];
}

//===----------------------------------------------------------------------===//
// XY Routing
//===----------------------------------------------------------------------===//

RouterDir RouterModel::computeXYRoute(int dstRow, int dstCol) const {
  // Dimension-ordered XY routing: route in X (column) first, then Y (row).
  if (col_ < dstCol)
    return RouterDir::EAST;
  if (col_ > dstCol)
    return RouterDir::WEST;
  if (row_ < dstRow)
    return RouterDir::SOUTH;
  if (row_ > dstRow)
    return RouterDir::NORTH;
  // Already at destination.
  return RouterDir::LOCAL;
}

//===----------------------------------------------------------------------===//
// Pipeline Stage 1: Route Compute
//===----------------------------------------------------------------------===//

void RouterModel::stageRouteCompute() {
  for (unsigned d = 0; d < kNumDirs; ++d) {
    for (unsigned v = 0; v < config_.numVCs; ++v) {
      auto &port = inputPorts_[d][v];
      if (!port.buffer.hasFlits())
        continue;

      const SimFlit &flit = port.buffer.front();

      if (port.wormholeLocked) {
        // Body/tail flits follow the path established by HEAD.
        port.computedRoute = port.wormholeOutputDir;
        continue;
      }

      // Only HEAD and SINGLE flits trigger a new route computation.
      if (flit.type == SimFlitType::HEAD || flit.type == SimFlitType::SINGLE) {
        RouterDir outDir = computeXYRoute(flit.dstRow, flit.dstCol);
        port.computedRoute = outDir;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pipeline Stage 2: Switch Arbitration and Traversal
//===----------------------------------------------------------------------===//

std::vector<RouterModel::ForwardedFlit> RouterModel::stageSwitchTraversal() {
  std::vector<ForwardedFlit> forwarded;

  // For each output direction, collect candidates and arbitrate.
  for (unsigned outD = 0; outD < kNumDirs; ++outD) {
    auto outDir = static_cast<RouterDir>(outD);

    // Collect candidate (inputDir, vc) pairs requesting this output.
    struct Candidate {
      unsigned inputDir;
      unsigned vc;
    };
    std::vector<Candidate> candidates;

    for (unsigned inD = 0; inD < kNumDirs; ++inD) {
      for (unsigned v = 0; v < config_.numVCs; ++v) {
        auto &port = inputPorts_[inD][v];
        if (!port.computedRoute.has_value())
          continue;
        if (port.computedRoute.value() != outDir)
          continue;
        if (!port.buffer.hasFlits())
          continue;

        // Check credit availability for downstream (or local delivery).
        if (outDir != RouterDir::LOCAL) {
          if (outputCredits_[outD][v] == 0)
            continue;
        }

        candidates.push_back({inD, v});
      }
    }

    if (candidates.empty())
      continue;

    // Round-robin arbitration: pick the candidate at or after the priority
    // pointer.
    unsigned priority = arbiterPriority_[outD];
    unsigned numCandidates = static_cast<unsigned>(candidates.size());

    // Flatten candidate index for round-robin: find the one closest to
    // priority in a circular sense. Priority is over (dir * numVCs + vc).
    unsigned bestIdx = 0;
    unsigned bestKey = UINT32_MAX;
    unsigned numInputSlots = kNumDirs * config_.numVCs;
    for (unsigned ci = 0; ci < numCandidates; ++ci) {
      unsigned key = candidates[ci].inputDir * config_.numVCs + candidates[ci].vc;
      unsigned distance = (key >= priority) ? (key - priority)
                                            : (key + numInputSlots - priority);
      if (distance < bestKey) {
        bestKey = distance;
        bestIdx = ci;
      }
    }

    if (numCandidates > 1) {
      ++stats_.totalArbitrationConflicts;
    }

    auto &winner = candidates[bestIdx];
    auto &port = inputPorts_[winner.inputDir][winner.vc];

    // Dequeue the winning flit.
    SimFlit flit = port.buffer.dequeue();

    // Update wormhole locking state.
    if (flit.type == SimFlitType::HEAD) {
      port.wormholeLocked = true;
      port.wormholeOutputDir = outDir;
    } else if (flit.type == SimFlitType::TAIL ||
               flit.type == SimFlitType::SINGLE) {
      port.wormholeLocked = false;
      port.computedRoute = std::nullopt;
    }

    // Clear computed route if not wormhole-locked (consumed this flit).
    if (!port.wormholeLocked && flit.type != SimFlitType::HEAD) {
      port.computedRoute = std::nullopt;
    }

    // Consume credit for non-local outputs.
    if (outDir != RouterDir::LOCAL) {
      assert(outputCredits_[outD][winner.vc] > 0);
      --outputCredits_[outD][winner.vc];
    }

    // Advance round-robin priority.
    arbiterPriority_[outD] =
        (winner.inputDir * config_.numVCs + winner.vc + 1) % numInputSlots;

    ++stats_.totalFlitsRouted;

    if (outDir == RouterDir::LOCAL) {
      // Deliver locally.
      localDeliveryQueue_.push_back(std::move(flit));
    } else {
      // Forward to neighbor.
      forwarded.push_back({std::move(flit), outDir, winner.vc});
    }
  }

  // Count stalls: inputs that had flits and routes computed but did not win.
  for (unsigned d = 0; d < kNumDirs; ++d) {
    for (unsigned v = 0; v < config_.numVCs; ++v) {
      auto &port = inputPorts_[d][v];
      if (port.computedRoute.has_value() && port.buffer.hasFlits()) {
        // This input had a flit ready but did not get served this cycle.
        ++stats_.totalStallCycles;
      }
    }
  }

  return forwarded;
}

//===----------------------------------------------------------------------===//
// Local ejection
//===----------------------------------------------------------------------===//

bool RouterModel::hasLocalDelivery() const {
  return !localDeliveryQueue_.empty();
}

SimFlit RouterModel::dequeueLocalDelivery() {
  assert(!localDeliveryQueue_.empty() && "no local delivery available");
  SimFlit flit = std::move(localDeliveryQueue_.front());
  localDeliveryQueue_.pop_front();
  return flit;
}

} // namespace mcsim
} // namespace loom
