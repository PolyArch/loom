#ifndef LOOM_FRONTEND_LOWERING_GRAPH_REGION_ADMISSION_H
#define LOOM_FRONTEND_LOWERING_GRAPH_REGION_ADMISSION_H

namespace mlir {
class Operation;
}

namespace loom::lowering::detail {

bool isGraphRegionControlOperation(mlir::Operation *operation);

// Whether graph publication replaces this exact representation-only LLVM
// bitcast with a canonical Dataflow adapter.
bool isGraphRegionRepresentationBitcast(mlir::Operation *operation);

} // namespace loom::lowering::detail

#endif // LOOM_FRONTEND_LOWERING_GRAPH_REGION_ADMISSION_H
