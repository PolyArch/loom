#ifndef LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace lowering {

bool isSupportedGraphLoweringLeaf(::mlir::Operation *op);
bool hasGraphOwnedParallelProvenance(::mlir::Operation *op);

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module);

::mlir::LogicalResult lowerGraphRegions(::dataflow::GraphOp graph);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
