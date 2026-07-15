#ifndef FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H

#include "Fabric/Tech/Partitioner/Partitioner.h"

namespace fabric {

// ILP-backed partitioner.
//
// On small inputs (graph body <= `kILPMaxOps` ops) and when the build was
// configured with `LOOM_ENABLE_ILP`, the partitioner formulates a mixed
// integer program over rooted multi-op template assignments and dispatches
// it to the HiGHS MIP solver. The MIP encodes:
//
//   * a binary variable x[i, t] for every (op i, template t) such that the
//     template is admissible at op i (root-op-name match + VF2 isomorphism
//     succeeds for a candidate rooted at i); when t has bodyOpCount > 1 the
//     covered ops are precisely those returned by collectMultiOpCandidate.
//   * a binary variable y[j] for every op j, set to 1 iff op j is left at
//     the graph level (uncovered).
//   * a binary variable e[j, k] for every SSA def-use edge (k -> j) inside
//     the partitionable op set, set to 1 iff the producer and the consumer
//     end up in different blocks (or either is at the graph level).
//   * objective: alpha * sum_{(i,t)} x[i,t]
//              + gamma * sum_{(i,t)} (1 - K_t/M_t) * x[i,t]
//              + (alpha + 1) * sum_j y[j]
//              + beta * sum_{(j,k)} e[j, k]
//
// The gamma term is the linearization of the cost-model `density` formula:
// CostModel takes the arithmetic mean of K_t/M_t across bound blocks, while
// the MIP takes the per-block density `deficit` (1 - K_t/M_t) and sums it.
// Both push the optimizer toward larger templates: the cost-model rewards
// blocks closer to peak density, the MIP penalizes per-block deficit and is
// 0 only when the block fully utilizes the largest template available for
// that root. The linearization keeps the MIP in standard form and stays
// faithful to the cost-model's intent.
//
// When the input is too large (more than `kILPMaxOps` ops), when HiGHS
// fails or times out (configurable via the LOOM_ILP_TIMEOUT_S env var,
// default 30 seconds), or when the build was configured with
// `LOOM_ENABLE_ILP=OFF`, the partitioner emits a module-level warning and
// delegates to the greedy partitioner so the pass still produces a valid
// partition.
class ILPPartitioner : public IPartitioner {
public:
  // Maximum graph-body op count for which the MIP path is taken. Beyond this
  // threshold the partitioner falls back to greedy.
  static constexpr unsigned kILPMaxOps = 200;

  PartitionResult run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                      const ::loom::ResolvedFabricTechMapConfig &cfg) override;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H
