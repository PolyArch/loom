#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALMATERIALIZER_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALMATERIALIZER_H

#include "Frontend/Compilation/StructuredScop.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace loom::frontend::detail {

/// Returns no refusal when the exact frozen relation has a complete
/// ordinary-MLIR realization. The derived ISL AST is invocation-local and
/// removable.
llvm::Expected<std::optional<StructuredScopRefusalKind>>
classifyPinnedIslScheduleMaterialization(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop);

/// Materializes the exact frozen schedule before root and erases root only
/// after every source statement has been cloned with exact coordinate and SSA
/// correspondence. A returned refusal denotes a representation outside the
/// current ordinary-MLIR realization.
llvm::Expected<std::optional<StructuredScopRefusalKind>>
materializePinnedIslSchedule(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop,
    const StructuredProgramCandidateView &parentView,
    const mlir::IRMapping &cloneMapping,
    llvm::SmallVectorImpl<mlir::Operation *> &materializedOperations);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALMATERIALIZER_H
