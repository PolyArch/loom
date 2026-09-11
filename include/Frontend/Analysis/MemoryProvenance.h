#ifndef LOOM_FRONTEND_ANALYSIS_MEMORYPROVENANCE_H
#define LOOM_FRONTEND_ANALYSIS_MEMORYPROVENANCE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace loom::frontend::analysis {

/// Removes exact pointer and view derivations without crossing a region or
/// callable input boundary. This is the root owned by an analysis local to the
/// current SSA scope.
mlir::Value projectMemoryDerivationRoot(mlir::Value value);

/// Removes exact pointer/view derivations and transparent Spatial input
/// boundaries, then returns the SSA value that owns the underlying memory
/// object or unresolved pointer provenance.
mlir::Value projectMemoryRoot(mlir::Value value);

/// Proves that two memory roots cannot denote the same object. Distinct SSA
/// values are not proof: accepted evidence is limited to distinct allocation
/// operations, distinct static symbols, allocation-versus-enclosing-input
/// provenance, explicit noalias attributes on distinct function arguments, or
/// a closed private callable whose complete set of exact direct call sites
/// supplies a proven-distinct pair. Callable evidence is rederived through
/// Spatial and thread-launch boundaries; an indirect or escaping use rejects
/// the proof.
bool haveProvenDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs);

/// Resolves the one static LLVM global that a memory root addresses at every
/// exact call site, through the same closed direct-call domain the distinct
/// roots proof walks. A root that is not the address of one global at every
/// site, or that reaches an indirect or escaping use, returns no value.
std::optional<mlir::LLVM::GlobalOp> resolveUniqueStaticGlobalRoot(mlir::Value value);

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_MEMORYPROVENANCE_H
