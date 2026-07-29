#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace loom::frontend::detail {

/// Validate the LLVM module-owned DataLayout and materialize its exact
/// endianness as the one DLTI projection consumed by Structured/Dataflow
/// lowering. Existing DLTI endianness must agree; no default is inferred.
llvm::Error materializeDataLayoutEndiannessProjection(mlir::ModuleOp module);

bool provesThreadDomainExtentFits(mlir::OpFoldResult lower,
                                  mlir::OpFoldResult upper,
                                  mlir::OpFoldResult step,
                                  unsigned targetWidth);

/// Whether this exact scope still needs a candidate-owned fixed index-width
/// decision. Constant-only addresses do not create such a decision unless
/// they form a proven pointer induction that must become integer loop state.
bool requiresCanonicalAddressIndexDecision(mlir::ModuleOp module,
                                           mlir::Operation *selectedOperation);

/// Enforces an explicit canonical index contract for LLVM GEPs. A selected
/// width is materialized in the Structured Program and wider operands are
/// narrowed only when their complete signed value domain is proven to fit.
/// Without a selected width, an existing explicit module declaration is
/// required. The caller owns failure atomicity through a private clone.
llvm::Expected<mlir::Operation *>
materializeAddressIndexContract(mlir::ModuleOp module,
                                mlir::Operation *selectedOperation,
                                std::optional<unsigned> canonicalIndexWidth);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
