#include "loom/MultiCoreSim/NoCSimModel.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace loom {
namespace mcsim {

NoCSimModel::NoCSimModel(unsigned meshRows, unsigned meshCols,
                         unsigned perHopLatency)
    : meshRows_(meshRows), meshCols_(meshCols), perHopLatency_(perHopLatency) {
  // Default placement: cores in row-major order.
  unsigned totalCores = meshRows_ * meshCols_;
  corePositions_.resize(totalCores);
  for (unsigned idx = 0; idx < totalCores; ++idx) {
    corePositions_[idx] = {idx / meshCols_, idx % meshCols_};
  }
  arrivedFlits_.resize(totalCores);
}

NoCSimModel::~NoCSimModel() = default;

std::string NoCSimModel::configure(const NoCSchedule &schedule) {
  meshRows_ = schedule.meshRows;
  meshCols_ = schedule.meshCols;
  unsigned totalCores = meshRows_ * meshCols_;
  corePositions_.resize(totalCores);
  for (unsigned idx = 0; idx < totalCores; ++idx) {
    corePositions_[idx] = {idx / meshCols_, idx % meshCols_};
  }
  arrivedFlits_.resize(totalCores);
  return {};
}

void NoCSimModel::setCorePosition(unsigned coreId, unsigned row, unsigned col) {
  if (coreId >= corePositions_.size())
    corePositions_.resize(coreId + 1, {0, 0});
  corePositions_[coreId] = {row, col};
  if (coreId >= arrivedFlits_.size())
    arrivedFlits_.resize(coreId + 1);
}

void NoCSimModel::injectFlit(const Flit &flit) {
  InFlightFlit inflight;
  inflight.flit = flit;

  auto [srcRow, srcCol] = getCorePosition(flit.srcCoreId);
  auto [dstRow, dstCol] = getCorePosition(flit.dstCoreId);

  inflight.currentRow = srcRow;
  inflight.currentCol = srcCol;
  inflight.targetRow = dstRow;
  inflight.targetCol = dstCol;
  inflight.remainingHopLatency = perHopLatency_;
  inflight.arrivalCycle = 0; // will be set on delivery

  inFlightFlits_.push_back(inflight);
  ++totalFlitsInjected_;
}

void NoCSimModel::stepOneCycle() {
  ++currentCycle_;

  // Process each in-flight flit.
  auto it = inFlightFlits_.begin();
  while (it != inFlightFlits_.end()) {
    InFlightFlit &inflight = *it;

    // Decrement hop latency counter.
    if (inflight.remainingHopLatency > 0) {
      --inflight.remainingHopLatency;
    }

    // If the latency for the current hop is done, advance to next hop.
    if (inflight.remainingHopLatency == 0) {
      routeFlit(inflight);

      // Check if arrived at destination.
      if (inflight.currentRow == inflight.targetRow &&
          inflight.currentCol == inflight.targetCol) {
        // Flit has arrived.
        inflight.arrivalCycle = currentCycle_;
        unsigned dstId = inflight.flit.dstCoreId;
        if (dstId >= arrivedFlits_.size())
          arrivedFlits_.resize(dstId + 1);
        arrivedFlits_[dstId].push_back(inflight.flit);

        // Compute latency (injection to arrival).
        uint64_t latency = currentCycle_ - inflight.flit.injectionCycle;
        totalLatency_ += latency;
        ++totalFlitsDelivered_;

        it = inFlightFlits_.erase(it);
        continue;
      }

      // Reset hop latency for next hop.
      inflight.remainingHopLatency = perHopLatency_;
      ++totalHops_;
    }

    ++it;
  }
}

bool NoCSimModel::hasArrivedFlits(unsigned dstCoreId) const {
  if (dstCoreId >= arrivedFlits_.size())
    return false;
  return !arrivedFlits_[dstCoreId].empty();
}

std::vector<Flit> NoCSimModel::drainArrivedFlits(unsigned dstCoreId) {
  if (dstCoreId >= arrivedFlits_.size())
    return {};
  std::vector<Flit> result(arrivedFlits_[dstCoreId].begin(),
                           arrivedFlits_[dstCoreId].end());
  arrivedFlits_[dstCoreId].clear();
  return result;
}

bool NoCSimModel::isIdle() const { return inFlightFlits_.empty(); }

NoCStats NoCSimModel::getStats() const {
  NoCStats stats;
  stats.totalFlitsInjected = totalFlitsInjected_;
  stats.totalFlitsDelivered = totalFlitsDelivered_;
  stats.totalHops = totalHops_;
  stats.averageLatency =
      totalFlitsDelivered_ > 0
          ? static_cast<double>(totalLatency_) / totalFlitsDelivered_
          : 0.0;

  // Compute link utilization as ratio of total hops to
  // (total links * total cycles).
  unsigned totalLinks = 2 * meshRows_ * meshCols_; // rough estimate
  if (totalLinks > 0 && currentCycle_ > 0) {
    stats.linkUtilization =
        static_cast<double>(totalHops_) / (totalLinks * currentCycle_);
  }
  return stats;
}

std::pair<unsigned, unsigned>
NoCSimModel::getCorePosition(unsigned coreId) const {
  if (coreId < corePositions_.size())
    return corePositions_[coreId];
  // Default: row-major placement.
  return {coreId / meshCols_, coreId % meshCols_};
}

void NoCSimModel::routeFlit(InFlightFlit &inflight) {
  // XY routing: first move along X (column), then along Y (row).
  if (inflight.currentCol < inflight.targetCol) {
    ++inflight.currentCol;
  } else if (inflight.currentCol > inflight.targetCol) {
    --inflight.currentCol;
  } else if (inflight.currentRow < inflight.targetRow) {
    ++inflight.currentRow;
  } else if (inflight.currentRow > inflight.targetRow) {
    --inflight.currentRow;
  }
}

} // namespace mcsim
} // namespace loom
