#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDPOINTERMEMREFPROMOTION_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDPOINTERMEMREFPROMOTION_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace loom::frontend {

/// Rewrites, inside every `loom.spatial_region` of one materialized Structured
/// Program, each top-level `scf.for` whose memory traffic is entirely
/// `llvm.getelementptr` element addressing over the region's pointer inputs
/// into the memref form the structured schedule analyses own: one
/// `loom.pointer_view` per input, `memref.distinct_objects` across them,
/// `memref.assume_alignment` at the input's proven alignment, and
/// `memref.load`/`memref.store` at the element index.
///
/// A loop is promoted only when every pointer input it accesses resolves to
/// one static LLVM array global at every exact launch site, every access
/// through it reads or writes that array's element type, and the inputs are
/// pairwise proven distinct. Any other loop is left untouched. Promoted
/// globals are raised to the vector transport alignment because the same
/// program owns their storage.
llvm::Error promoteStaticPointerLoopsToMemRef(mlir::ModuleOp module);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDPOINTERMEMREFPROMOTION_H
