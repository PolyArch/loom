#ifndef FABRIC_TECH_PARTITIONER_MATERIALIZER_H
#define FABRIC_TECH_PARTITIONER_MATERIALIZER_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/Partitioner/Partitioner.h"
#include "mlir/IR/Builders.h"

namespace fabric {

// Rewrite the body of `graph` according to `partition`:
//
//   * For each Block whose `tpl != nullptr`, lift its ops into a fresh
//     dataflow.subgraph inserted at the location of the block's first op
//     in program order. External operands become subgraph inputs and
//     external uses become subgraph results; internal SSA edges are
//     preserved by an IRMapping during cloning.
//   * Blocks with `tpl == nullptr` are left in place at the graph level.
//
// The graph's terminator (dataflow.yield) is never touched. The builder's
// insertion point is used only as a scratchpad; the caller is responsible
// for any post-pass cleanup.
void applyPartition(::dataflow::GraphOp graph,
                    const PartitionResult &partition,
                    ::mlir::OpBuilder &builder);

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_MATERIALIZER_H
