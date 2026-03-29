#ifndef LOOM_MAPPER_MAPPERCONGESTIONESTIMATOR_H
#define LOOM_MAPPER_MAPPERCONGESTIONESTIMATOR_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "llvm/ADT/DenseMap.h"

namespace loom {

/// Estimate per-switch-output routing demand based on current placement.
/// For each DFG edge, identify a topology-aware routing corridor between the
/// mapped endpoints and distribute demand over routing crossbar outputs whose
/// parent nodes lie close to that corridor.
struct CongestionEstimator {
  /// Per switch output port demand estimate.
  llvm::DenseMap<IdIndex, double> switchOutputDemand;

  /// Compute switch output demand estimates from current placement.
  void estimate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const ADGFlattener &flattener);

  /// Get the demand/capacity ratio sum for switch outputs in the topology
  /// corridor of the given src/dst HW node positions.
  double demandCapacityRatio(IdIndex srcHwNode, IdIndex dstHwNode,
                             const Graph &adg,
                             const ADGFlattener &flattener) const;

  /// Compute total demand excess: sum of max(0, demand - 1.0) across all
  /// switch outputs. Each switch output has capacity 1.0.
  double totalDemandExcess() const;
};

} // namespace loom

#endif // LOOM_MAPPER_MAPPERCONGESTIONESTIMATOR_H
