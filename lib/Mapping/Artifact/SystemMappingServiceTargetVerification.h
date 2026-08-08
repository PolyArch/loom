#ifndef LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGSERVICETARGETVERIFICATION_H
#define LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGSERVICETARGETVERIFICATION_H

#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "SystemMappingExecutionProjection.h"

namespace loom::mapping::detail {

llvm::Error verifySystemServiceTargetClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store,
    const SystemServiceObligationProjection &obligation,
    const SystemExecutionContextProjection &contexts,
    llvm::ArrayRef<SystemServicePlanView> plans,
    llvm::ArrayRef<SystemServicePlanSelectionView> selections);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGSERVICETARGETVERIFICATION_H
