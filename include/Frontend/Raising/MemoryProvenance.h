#ifndef LOOM_FRONTEND_RAISING_MEMORYPROVENANCE_H
#define LOOM_FRONTEND_RAISING_MEMORYPROVENANCE_H

#include "mlir/IR/Value.h"

namespace loom::raising {

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

} // namespace loom::raising

#endif // LOOM_FRONTEND_RAISING_MEMORYPROVENANCE_H
