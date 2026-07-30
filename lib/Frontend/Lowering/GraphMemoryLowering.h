#ifndef LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace lowering {

// Construction-local projection used by the atomic Spatial-to-Dataflow
// publisher. Each canonical graph memory input names the source memory operand
// of the pre-finalization graph. It never enters MLIR or artifact identity.
struct GraphMemoryInputProjection {
  ::dataflow::GraphOp graph;
  ::llvm::SmallVector<unsigned, 4> sourceOrdinals;
};

::mlir::LogicalResult lowerGraphMemory(
    ::mlir::ModuleOp module,
    ::llvm::SmallVectorImpl<GraphMemoryInputProjection> *projections = nullptr);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H
