#ifndef LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
#define LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H

#include "PnR/System/SystemCandidateState.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

/// Materializes only the tracked ExecutionBinding records. The returned root
/// is a non-published draft until service and ResourceUse closure are added.
llvm::Expected<mlir::OwningOpRef<mlir::Operation *>>
materializeSystemExecutionBindings(const SystemCandidateState &candidate,
                                   mlir::MLIRContext &context);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
