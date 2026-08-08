#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCAPACITYVERIFICATION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCAPACITYVERIFICATION_H

#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::mapping::detail {

llvm::Error verifySystemMappingCapacity(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    llvm::ArrayRef<SystemResourceUseView> resourceUses,
    llvm::ArrayRef<std::string> resourceUseActivationKeys,
    const ArtifactStore &store);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCAPACITYVERIFICATION_H
