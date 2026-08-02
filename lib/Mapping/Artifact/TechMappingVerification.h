#ifndef LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGVERIFICATION_H
#define LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGVERIFICATION_H

#include "Mapping/Artifact/MappingArtifact.h"

namespace loom::mapping::detail {

llvm::Error verifyTechComputeRealizationClosure(
    const TechComputeRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGVERIFICATION_H
