#ifndef LOOM_FRONTEND_LOWERING_LOOPINDEPENDENCE_H
#define LOOM_FRONTEND_LOWERING_LOOPINDEPENDENCE_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mlir::scf {
class ForOp;
class ForallOp;
} // namespace mlir::scf

namespace loom::lowering {

enum class ParallelDependenceResult : std::uint8_t {
  ProvenIndependent,
  ProvenDependent,
  ProofNotEstablished,
};

// Iteration-independence admission shared by schedule materialization and
// ownership validation. Unknown geometry preserves serial semantics.
ParallelDependenceResult proveIndependentIterations(::mlir::scf::ForOp loop);
ParallelDependenceResult
proveIndependentIterations(::llvm::ArrayRef<::mlir::scf::ForOp> nest);

// Re-proves disjoint point intervals after a strip-mined loop becomes forall.
// The proof reads the IR and its byte geometry, never provider metadata.
ParallelDependenceResult proveIndependentIterations(::mlir::scf::ForallOp loop);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_LOOPINDEPENDENCE_H
