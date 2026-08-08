#ifndef LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSURE_H
#define LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSURE_H

#include "Mapping/Artifact/SystemMappingArtifact.h"

namespace loom::mapping::detail {

struct ImportedSystemClosure final {
  std::vector<SystemServiceRealizationView> services;
  std::vector<SystemResourceUseView> resourceUses;
};

llvm::Expected<ImportedSystemClosure> importSystemMappingClosure(
    ::mapping::SystemOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution, const ArtifactStore &store);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_SYSTEMMAPPINGCLOSURE_H
