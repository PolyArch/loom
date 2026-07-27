#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace loom::frontend::detail {

/// Enforces an explicit canonical index contract for LLVM GEPs. A selected
/// width is materialized in the Structured Program and wider operands are
/// narrowed only when their complete signed value domain is proven to fit.
/// Without a selected width, an existing explicit module declaration is
/// required. The caller owns failure atomicity through a private clone.
llvm::Error
materializeAddressIndexContract(mlir::ModuleOp module,
                                mlir::Operation *selectedOperation,
                                std::optional<unsigned> canonicalIndexWidth);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
