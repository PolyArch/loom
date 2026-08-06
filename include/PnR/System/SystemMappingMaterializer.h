#ifndef LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
#define LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H

#include "PnR/System/SystemCandidateState.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

/// Materializes the selected execution bindings and reduced System service
/// routes. The returned root remains a non-published draft until target,
/// refinement, and ResourceUse closure are added.
llvm::Expected<mlir::OwningOpRef<mlir::Operation *>>
materializeSystemCandidateDraft(const SystemCandidateState &candidate,
                                mlir::MLIRContext &context);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
