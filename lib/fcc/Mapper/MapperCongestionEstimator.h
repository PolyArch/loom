#ifndef FCC_MAPPER_MAPPERCONGESTIONESTIMATOR_H
#define FCC_MAPPER_MAPPERCONGESTIONESTIMATOR_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "llvm/ADT/DenseMap.h"

namespace fcc {

/// Estimate per-switch-output routing demand based on current placement.
/// For each DFG edge, compute bounding box of src/dst PEs, identify switch
/// outputs in the bbox, and distribute demand proportionally.
struct CongestionEstimator {
  /// Per switch output port demand estimate.
  llvm::DenseMap<IdIndex, double> switchOutputDemand;

  /// Compute switch output demand estimates from current placement.
  void estimate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const ADGFlattener &flattener);

  /// Get the demand/capacity ratio sum for switch outputs in the bounding
  /// box of the given src/dst HW node positions.
  double demandCapacityRatio(IdIndex srcHwNode, IdIndex dstHwNode,
                             const Graph &adg,
                             const ADGFlattener &flattener) const;
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPERCONGESTIONESTIMATOR_H
