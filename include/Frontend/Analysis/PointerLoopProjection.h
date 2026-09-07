#ifndef LOOM_FRONTEND_ANALYSIS_POINTERLOOPPROJECTION_H
#define LOOM_FRONTEND_ANALYSIS_POINTERLOOPPROJECTION_H

#include "Frontend/Analysis/MemoryAddressProjection.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace loom::frontend::analysis {

/// One post-tested pointer recurrence whose equality predicate compares its
/// updated address against an invariant address in the same LLVM object.
/// The projection does not assume a trip count or remove the source guard.
/// Replacing the pointer lane by an address-width byte/element offset preserves
/// the exact predicate, feedback lanes, and failed-condition result.
struct PointerLoopTerminationProjection final {
  mlir::scf::WhileOp loop;
  unsigned pointerLane;
  mlir::LLVM::ICmpOp comparison;
  mlir::LLVM::GEPOp update;
  ResolvedLinearMemoryAddress begin;
  ResolvedLinearMemoryAddress end;
};

/// Admission and normalization use the same proof; arbitrary pointer
/// comparisons do not become graph compute operations.
std::optional<PointerLoopTerminationProjection>
projectPointerLoopTermination(mlir::scf::WhileOp loop);

} // namespace loom::frontend::analysis

#endif
