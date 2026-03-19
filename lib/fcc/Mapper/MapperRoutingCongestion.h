#ifndef FCC_MAPPER_MAPPERROUTINGCONGESTION_H
#define FCC_MAPPER_MAPPERROUTINGCONGESTION_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "llvm/ADT/ArrayRef.h"

#include <vector>

namespace fcc {

/// Per-port congestion tracking for PathFinder-style negotiated routing.
/// Historical congestion penalties bias future routing iterations away from
/// popular routing outputs, even when they were legally used.
struct CongestionState {
  // Dense vectors indexed by port ID for hot-path performance.
  std::vector<double> historicalCongestion;
  std::vector<unsigned> presentDemand;
  std::vector<unsigned> capacity;

  double historyIncrement = 1.0;
  double historyScale = 1.5;
  double presentFactor = 1.0;
  double saturationPenalty = 0.5;

  /// Initialize vectors sized to the ADG port count. Sets capacity to 1 for
  /// routing crossbar output ports, 0 for others.
  void init(const Graph &adg);

  /// Compute the congestion-based cost multiplier for a routing output port.
  /// Returns baseCost * (1 + hist) * (1 + present * factor).
  double resourceCost(IdIndex portId) const;

  /// Increment presentDemand for crossbar output ports along a routed path.
  void commitRoute(llvm::ArrayRef<IdIndex> path, const Graph &adg);

  /// Decrement presentDemand for crossbar output ports along a routed path.
  void uncommitRoute(llvm::ArrayRef<IdIndex> path, const Graph &adg);

  /// Check if any routing output has presentDemand > capacity.
  bool hasOveruse() const;

  /// Update historical congestion: hist += increment * max(0, demand - cap).
  void updateHistory();

  /// Apply multiplicative decay to historical congestion.
  void decayHistory(double factor);

  /// Zero all presentDemand for a new routing iteration.
  void resetPresentDemand();
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPERROUTINGCONGESTION_H
