#ifndef LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace lowering {

// The canonical Dataflow memory actors that graph-region lowering has no
// transformation for: `dataflow.fence`, `dataflow.atomic_rmw`, and
// `dataflow.cmpxchg`. This is the single authority both the serial classifier
// (`isSupportedGraphLoweringLeaf`) and the parallel completion check consult,
// so every serial and parallel nesting shape rejects these effectful actors
// during preflight instead of aborting inside lowering.
bool isUnloweredGraphMemoryActor(::mlir::Operation *op);

bool isSupportedGraphLoweringLeaf(::mlir::Operation *op);

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module);

// `indexBits` is the canonical index width the caller's pass boundary already
// resolved; region lowering never resolves it again.
::mlir::LogicalResult lowerGraphRegions(::dataflow::GraphOp graph,
                                        unsigned indexBits);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
