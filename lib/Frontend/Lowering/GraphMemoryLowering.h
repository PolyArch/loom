#ifndef LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace lowering {

enum class GraphMemoryInputSourceKind : unsigned char {
  ExistingMemory,
  PointerService,
};

// Construction-local projection used by the atomic Spatial-to-Dataflow
// publisher. An existing memory input retains its source memory ordinal. A
// pointer-addressed access names the source value ordinal from which the
// enclosing thread acquires an explicit object-scoped memory service. This
// projection never enters MLIR or artifact identity.
struct GraphMemoryInputSource {
  GraphMemoryInputSourceKind kind = GraphMemoryInputSourceKind::ExistingMemory;
  unsigned sourceOrdinal = 0;
};

struct GraphMemoryInputProjection {
  ::dataflow::GraphOp graph;
  ::llvm::SmallVector<GraphMemoryInputSource, 4> sources;
};

::mlir::LogicalResult lowerGraphMemory(
    ::mlir::ModuleOp module,
    ::llvm::SmallVectorImpl<GraphMemoryInputProjection> *projections = nullptr);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_MEMORY_LOWERING_H
