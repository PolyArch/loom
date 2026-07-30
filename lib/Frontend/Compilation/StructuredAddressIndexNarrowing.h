#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace loom::frontend::detail {

using BlockReplacementObserver =
    llvm::function_ref<llvm::Error(mlir::Block *, mlir::Block *)>;

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

/// Explains why structured control in this exact scope cannot be normalized to
/// invariant memory capabilities plus integer state by the production address
/// normalizer. Absence means only that this normalization boundary is
/// supported; later materialization and graph verification remain
/// authoritative.
std::optional<std::string>
explainAddressStateNormalizationRejection(mlir::Operation *selectedOperation);

/// Enforces an explicit canonical index contract for LLVM GEPs. A selected
/// width is materialized in the Structured Program and wider operands are
/// narrowed only when their complete signed value domain is proven to fit.
/// Without a selected width, an existing explicit module declaration is
/// required. The caller owns failure atomicity through a private clone.
llvm::Expected<mlir::Operation *>
materializeAddressIndexContract(mlir::ModuleOp module,
                                mlir::Operation *selectedOperation,
                                std::optional<unsigned> canonicalIndexWidth,
                                BlockReplacementObserver observeReplacement);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDADDRESSINDEXNARROWING_H
