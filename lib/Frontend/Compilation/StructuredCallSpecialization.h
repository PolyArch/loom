#ifndef LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H
#define LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H

#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <optional>

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace loom::frontend::detail {

llvm::Expected<bool>
hasUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                          mlir::Operation *selection);

/// Returns nullopt when exact specialization legitimately removes the selected
/// nested scope. The caller owns classification of that candidate-local
/// outcome; malformed input and broken specialization invariants remain errors.
llvm::Expected<std::optional<mlir::Operation *>>
materializeUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                                  mlir::Operation *selection);

/// Returns true exactly when `callSite` is a direct call inside `selection`
/// whose defined, non-variadic, single-block callee contains no general call.
/// This is a candidate-domain predicate, not a profitability decision.
bool isExactDirectCallSiteInlineable(mlir::ModuleOp module,
                                     mlir::Operation *selection,
                                     mlir::Operation *callSite);

/// The exact call coordinate and its mechanically resolved local definition.
/// The pair is ephemeral and exists only while constructing one ownership
/// candidate domain.
struct ExactDirectCallSiteInliningCandidate final {
  mlir::Operation *callSite;
  mlir::Operation *callee;
};

/// Returns the sole exact direct leaf call admitted by the atomic inlining
/// domain together with its callee, or null when the selected scope has no
/// such call coordinate.
std::optional<ExactDirectCallSiteInliningCandidate>
findExactDirectCallSiteInliningCandidate(mlir::ModuleOp module,
                                         mlir::Operation *selection);

/// Returns whether inlining the admitted exact call exposes an address-index
/// decision owned by its callee body. The caller combines this with the
/// selected scope's own address requirements before enumerating candidates.
bool exactDirectCallSiteInliningRequiresCanonicalAddressIndexDecision(
    mlir::ModuleOp module, mlir::Operation *selection,
    mlir::Operation *callSite);

/// Inlines one exact direct call in the private candidate clone. The callee
/// definition remains present and the selected operation remains the owner of
/// the resulting dependency closure.
struct DirectCallInliningMaterialization final {
  mlir::Operation *selection;
  mlir::IRMapping clonedBlocks;
  /// Present only when the selected scope was the direct call itself. The
  /// exact-once block is private materialization state and must not survive
  /// ownership publication.
  mlir::Block *exactClosureBlock = nullptr;
};

/// A successful Expected containing nullopt is an ordinary refusal by the
/// pinned MLIR inliner. Errors are malformed input or implementation invariant
/// failures and must remain invocation failures.
llvm::Expected<std::optional<DirectCallInliningMaterialization>>
materializeExactDirectCallSiteInlining(mlir::ModuleOp module,
                                       mlir::Operation *selection,
                                       mlir::Operation *callSite);

} // namespace loom::frontend::detail

#endif // LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H
