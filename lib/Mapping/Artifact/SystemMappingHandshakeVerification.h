#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHANDSHAKEVERIFICATION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHANDSHAKEVERIFICATION_H

#include "Mapping/Artifact/SystemMappingArtifact.h"

namespace loom::mapping::detail {

llvm::Error verifySystemMappingHandshakeClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    const ArtifactStore &store);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGHANDSHAKEVERIFICATION_H
