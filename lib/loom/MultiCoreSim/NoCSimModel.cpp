#include "loom/MultiCoreSim/NoCSimModel.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Construction / destruction / move
//===----------------------------------------------------------------------===//

NoCSimModel::NoCSimModel() = default;
NoCSimModel::~NoCSimModel() = default;
NoCSimModel::NoCSimModel(NoCSimModel &&) noexcept = default;
NoCSimModel &NoCSimModel::operator=(NoCSimModel &&) noexcept = default;

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

void NoCSimModel::configure(
    const NoCSimConfig &config,
    const std::vector<std::pair<int, int>> &corePositions) {
  config_ = config;

  // Determine mesh dimensions from core positions if not explicitly set.
  int maxRow = 0;
  int maxCol = 0;
  for (unsigned i = 0; i < corePositions.size(); ++i) {
    maxRow = std::max(maxRow, corePositions[i].first);
    maxCol = std::max(maxCol, corePositions[i].second);
  }

  numRows_ = (config.meshRows > 0) ? config.meshRows
                                     : static_cast<unsigned>(maxRow + 1);
  numCols_ = (config.meshCols > 0) ? config.meshCols
                                     : static_cast<unsigned>(maxCol + 1);

  // Ensure at least 1x1.
  if (numRows_ == 0)
    numRows_ = 1;
  if (numCols_ == 0)
    numCols_ = 1;

  // Build core <-> position mappings.
  coreToPosition_.clear();
  positionToCore_.clear();
  for (unsigned coreId = 0; coreId < corePositions.size(); ++coreId) {
    auto pos = corePositions[coreId];
    coreToPosition_[coreId] = pos;
    unsigned posKey = static_cast<unsigned>(pos.first) * numCols_ +
                      static_cast<unsigned>(pos.second);
    positionToCore_[posKey] = coreId;
  }

  // Build router configuration.
  RouterConfig rcfg;
  rcfg.numVCs = config.numVCs;
  rcfg.bufferDepth = config.bufferDepth;
  rcfg.pipelineStages = config.routerPipelineStages;

  // Instantiate routers.
  routers_.clear();
  routers_.reserve(numRows_ * numCols_);
  for (unsigned r = 0; r < numRows_; ++r) {
    for (unsigned c = 0; c < numCols_; ++c) {
      unsigned id = r * numCols_ + c;
      routers_.emplace_back(id, static_cast<int>(r), static_cast<int>(c), rcfg);
    }
  }

  // Initialize credits between connected routers.
  // Each output port's credits = downstream input buffer depth.
  for (unsigned r = 0; r < numRows_; ++r) {
    for (unsigned c = 0; c < numCols_; ++c) {
      auto &router = routerAt(static_cast<int>(r), static_cast<int>(c));
      for (unsigned d = 0; d < RouterModel::kNumDirs; ++d) {
        auto dir = static_cast<RouterDir>(d);
        RouterModel *neighbor =
            neighborRouter(static_cast<int>(r), static_cast<int>(c), dir);
        if (neighbor) {
          // This router's output in direction 'dir' connects to the neighbor's
          // input in the reverse direction. Credits = neighbor's buffer depth.
          for (unsigned v = 0; v < config.numVCs; ++v) {
            router.initCredits(dir, v, config.bufferDepth);
          }
        } else if (dir == RouterDir::LOCAL) {
          // Local port: set high credits (local ejection is unbounded).
          for (unsigned v = 0; v < config.numVCs; ++v) {
            router.initCredits(dir, v, 1024);
          }
        }
        // Boundary ports with no neighbor: leave credits at 0 (cannot send).
      }
    }
  }

  // Initialize statistics.
  totalFlitsInjected_ = 0;
  totalFlitsDelivered_ = 0;
  totalLatencySum_ = 0.0;
  maxLatency_ = 0.0;
  totalCyclesElapsed_ = 0;

  vcFlitCounts_.clear();
  vcFlitCounts_.resize(config.numVCs, 0);

  initLinkStats();

  configured_ = true;
}

//===----------------------------------------------------------------------===//
// Injection and ejection
//===----------------------------------------------------------------------===//

bool NoCSimModel::canInject(unsigned srcCoreId, unsigned vcId) const {
  assert(configured_ && "NoCSimModel not configured");
  auto it = coreToPosition_.find(srcCoreId);
  if (it == coreToPosition_.end())
    return false;

  auto [row, col] = it->second;
  const auto &router = routerAt(row, col);
  return router.canAcceptFlit(RouterDir::LOCAL, vcId);
}

void NoCSimModel::inject(const SimFlit &flit) {
  assert(configured_ && "NoCSimModel not configured");
  auto it = coreToPosition_.find(flit.srcCoreId);
  assert(it != coreToPosition_.end() && "unknown source core ID");

  auto [row, col] = it->second;
  auto &router = routerAt(row, col);
  bool ok = router.acceptFlit(RouterDir::LOCAL, flit.vcId, flit);
  assert(ok && "injection failed: buffer full (check canInject first)");
  (void)ok;

  ++totalFlitsInjected_;
  if (flit.vcId < vcFlitCounts_.size()) {
    ++vcFlitCounts_[flit.vcId];
  }
}

bool NoCSimModel::hasDelivery(unsigned dstCoreId) const {
  assert(configured_ && "NoCSimModel not configured");
  auto it = coreToPosition_.find(dstCoreId);
  if (it == coreToPosition_.end())
    return false;

  auto [row, col] = it->second;
  const auto &router = routerAt(row, col);
  return router.hasLocalDelivery();
}

SimFlit NoCSimModel::receive(unsigned dstCoreId) {
  assert(configured_ && "NoCSimModel not configured");
  auto it = coreToPosition_.find(dstCoreId);
  assert(it != coreToPosition_.end() && "unknown destination core ID");

  auto [row, col] = it->second;
  auto &router = routerAt(row, col);
  SimFlit flit = router.dequeueLocalDelivery();

  ++totalFlitsDelivered_;
  double latency =
      static_cast<double>(totalCyclesElapsed_) -
      static_cast<double>(flit.injectionCycle);
  if (latency < 0)
    latency = 0;
  totalLatencySum_ += latency;
  if (latency > maxLatency_)
    maxLatency_ = latency;

  return flit;
}

//===----------------------------------------------------------------------===//
// Tick -- advance one cycle
//===----------------------------------------------------------------------===//

void NoCSimModel::tick(uint64_t globalCycle) {
  assert(configured_ && "NoCSimModel not configured");

  // Phase 1: Route Compute for all routers.
  for (auto &router : routers_) {
    router.stageRouteCompute();
  }

  // Phase 2: Switch Traversal for all routers.
  // Collect forwarded flits from all routers first, then deliver them.
  // This avoids order-dependent behavior within a single cycle.
  struct PendingDelivery {
    SimFlit flit;
    int dstRow;
    int dstCol;
    RouterDir inputDir;
    unsigned vc;
  };
  std::vector<PendingDelivery> pendingDeliveries;

  for (auto &router : routers_) {
    auto forwarded = router.stageSwitchTraversal();
    for (auto &fw : forwarded) {
      int nRow = router.getRow();
      int nCol = router.getCol();
      switch (fw.outputDir) {
      case RouterDir::NORTH:
        nRow -= 1;
        break;
      case RouterDir::SOUTH:
        nRow += 1;
        break;
      case RouterDir::EAST:
        nCol += 1;
        break;
      case RouterDir::WEST:
        nCol -= 1;
        break;
      default:
        break;
      }

      recordLinkTraversal(router.getRow(), router.getCol(), nRow, nCol);

      pendingDeliveries.push_back(
          {std::move(fw.flit), nRow, nCol,
           reverseDirection(fw.outputDir), fw.vc});
    }
  }

  // Phase 3: Deliver forwarded flits to downstream input buffers and
  // return credits to upstream output ports.
  for (auto &pd : pendingDeliveries) {
    if (pd.dstRow < 0 || pd.dstRow >= static_cast<int>(numRows_) ||
        pd.dstCol < 0 || pd.dstCol >= static_cast<int>(numCols_)) {
      // Out of bounds -- should not happen with correct XY routing.
      continue;
    }

    auto &dstRouter = routerAt(pd.dstRow, pd.dstCol);
    bool accepted = dstRouter.acceptFlit(pd.inputDir, pd.vc, pd.flit);
    (void)accepted;
    // If not accepted, the flit is dropped (should not happen with correct
    // credit-based flow control).

    // Return credit to the upstream router that just sent this flit.
    // The upstream router's output in the forward direction consumed a credit;
    // now the downstream has accepted, so we return it to the source router.
    // The source router is in the reverse direction from the destination.
    RouterDir upstreamDir = reverseDirection(pd.inputDir);
    // The source is the neighbor of dstRouter in upstreamDir.
    // But it's simpler: the credit goes to the router that sent the flit.
    // That router is at (srcRow, srcCol) which we can infer.
    int srcRow = pd.dstRow;
    int srcCol = pd.dstCol;
    switch (pd.inputDir) {
    case RouterDir::NORTH:
      srcRow -= 1;
      break;
    case RouterDir::SOUTH:
      srcRow += 1;
      break;
    case RouterDir::EAST:
      srcCol += 1;
      break;
    case RouterDir::WEST:
      srcCol -= 1;
      break;
    default:
      break;
    }
    if (srcRow >= 0 && srcRow < static_cast<int>(numRows_) && srcCol >= 0 &&
        srcCol < static_cast<int>(numCols_)) {
      auto &srcRouter = routerAt(srcRow, srcCol);
      srcRouter.returnCredit(upstreamDir, pd.vc);
    }
  }

  ++totalCyclesElapsed_;
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

NoCDetailedStats NoCSimModel::getDetailedStats() const {
  NoCDetailedStats stats;
  stats.totalFlitsInjected = totalFlitsInjected_;
  stats.totalFlitsDelivered = totalFlitsDelivered_;

  // Count in-flight flits (injected but not yet delivered).
  if (totalFlitsInjected_ > totalFlitsDelivered_) {
    stats.totalFlitsInFlight = totalFlitsInjected_ - totalFlitsDelivered_;
  }

  // Average and max latency.
  if (totalFlitsDelivered_ > 0) {
    stats.avgLatencyCycles = totalLatencySum_ / totalFlitsDelivered_;
  }
  stats.maxLatencyCycles = maxLatency_;

  // Per-link statistics.
  for (const auto &lc : linkCounters_) {
    NoCDetailedStats::LinkStats ls;
    ls.srcRouter = lc.srcRouterId;
    ls.dstRouter = lc.dstRouterId;
    ls.srcRow = lc.srcRow;
    ls.srcCol = lc.srcCol;
    ls.dstRow = lc.dstRow;
    ls.dstCol = lc.dstCol;
    ls.flitsTransferred = lc.flitsTransferred;
    ls.stallCycles = lc.stallCycles;
    if (totalCyclesElapsed_ > 0) {
      ls.utilization = static_cast<double>(lc.flitsTransferred) /
                       static_cast<double>(totalCyclesElapsed_ *
                                           config_.linkBandwidthFlitsPerCycle);
    }
    stats.linkStats.push_back(ls);
  }

  // Per-VC statistics.
  for (unsigned v = 0; v < config_.numVCs; ++v) {
    NoCDetailedStats::VCStats vs;
    vs.vcId = v;
    vs.flitsTransferred = (v < vcFlitCounts_.size()) ? vcFlitCounts_[v] : 0;
    vs.avgOccupancy = 0.0; // TODO: track per-cycle occupancy for accuracy
    stats.vcStats.push_back(vs);
  }

  return stats;
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

RouterModel &NoCSimModel::routerAt(int row, int col) {
  unsigned idx = static_cast<unsigned>(row) * numCols_ +
                 static_cast<unsigned>(col);
  assert(idx < routers_.size() && "router index out of bounds");
  return routers_[idx];
}

const RouterModel &NoCSimModel::routerAt(int row, int col) const {
  unsigned idx = static_cast<unsigned>(row) * numCols_ +
                 static_cast<unsigned>(col);
  assert(idx < routers_.size() && "router index out of bounds");
  return routers_[idx];
}

unsigned NoCSimModel::routerIdAt(int row, int col) const {
  return static_cast<unsigned>(row) * numCols_ + static_cast<unsigned>(col);
}

RouterModel *NoCSimModel::neighborRouter(int row, int col, RouterDir dir) {
  int nRow = row;
  int nCol = col;
  switch (dir) {
  case RouterDir::NORTH:
    nRow -= 1;
    break;
  case RouterDir::SOUTH:
    nRow += 1;
    break;
  case RouterDir::EAST:
    nCol += 1;
    break;
  case RouterDir::WEST:
    nCol -= 1;
    break;
  case RouterDir::LOCAL:
    return nullptr; // LOCAL has no neighbor router
  default:
    return nullptr;
  }

  if (nRow < 0 || nRow >= static_cast<int>(numRows_) || nCol < 0 ||
      nCol >= static_cast<int>(numCols_)) {
    return nullptr;
  }

  return &routerAt(nRow, nCol);
}

void NoCSimModel::initLinkStats() {
  linkCounters_.clear();

  // Create a counter for each directed link in the mesh.
  for (unsigned r = 0; r < numRows_; ++r) {
    for (unsigned c = 0; c < numCols_; ++c) {
      int row = static_cast<int>(r);
      int col = static_cast<int>(c);

      // East link: (r,c) -> (r,c+1)
      if (c + 1 < numCols_) {
        LinkCounter lc;
        lc.srcRow = row;
        lc.srcCol = col;
        lc.dstRow = row;
        lc.dstCol = col + 1;
        lc.srcRouterId = routerIdAt(row, col);
        lc.dstRouterId = routerIdAt(row, col + 1);
        linkCounters_.push_back(lc);
      }
      // West link: (r,c) -> (r,c-1)
      if (c > 0) {
        LinkCounter lc;
        lc.srcRow = row;
        lc.srcCol = col;
        lc.dstRow = row;
        lc.dstCol = col - 1;
        lc.srcRouterId = routerIdAt(row, col);
        lc.dstRouterId = routerIdAt(row, col - 1);
        linkCounters_.push_back(lc);
      }
      // South link: (r,c) -> (r+1,c)
      if (r + 1 < numRows_) {
        LinkCounter lc;
        lc.srcRow = row;
        lc.srcCol = col;
        lc.dstRow = row + 1;
        lc.dstCol = col;
        lc.srcRouterId = routerIdAt(row, col);
        lc.dstRouterId = routerIdAt(row + 1, col);
        linkCounters_.push_back(lc);
      }
      // North link: (r,c) -> (r-1,c)
      if (r > 0) {
        LinkCounter lc;
        lc.srcRow = row;
        lc.srcCol = col;
        lc.dstRow = row - 1;
        lc.dstCol = col;
        lc.srcRouterId = routerIdAt(row, col);
        lc.dstRouterId = routerIdAt(row - 1, col);
        linkCounters_.push_back(lc);
      }
    }
  }
}

void NoCSimModel::recordLinkTraversal(int srcRow, int srcCol, int dstRow,
                                       int dstCol) {
  for (auto &lc : linkCounters_) {
    if (lc.srcRow == srcRow && lc.srcCol == srcCol && lc.dstRow == dstRow &&
        lc.dstCol == dstCol) {
      ++lc.flitsTransferred;
      return;
    }
  }
}

} // namespace mcsim
} // namespace loom
