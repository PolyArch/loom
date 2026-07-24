#ifndef LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace lowering {

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
