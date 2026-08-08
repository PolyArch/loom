#ifndef LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
#define LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H

#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/System/SystemCandidateState.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

/// Materializes the complete selected execution, service, route, target, and
/// ResourceUse closure. The returned root remains unpublished.
llvm::Expected<mlir::OwningOpRef<mlir::Operation *>>
materializeSystemCandidateDraft(const SystemCandidateState &candidate,
                                mlir::MLIRContext &context);

llvm::Expected<::loom::mapping::FinalizedSystemMapping>
finalizeSystemMappingCandidate(
    const SystemCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::SystemMappingConstraintSetView &constraints,
    const ArtifactStore &store, mlir::MLIRContext &context);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMMAPPINGMATERIALIZER_H
