#ifndef LOOM_DSE_SPATIALMAPPINGFEEDBACKCANDIDATEGENERATOR_H
#define LOOM_DSE_SPATIALMAPPINGFEEDBACKCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "PnR/PnrConfig.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    spatialMappingFeedbackCandidateGeneratorKind(8);

const CandidateGeneratorDescriptor &
spatialMappingFeedbackCandidateGeneratorDescriptor();
llvm::Error registerSpatialMappingFeedbackCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialMappingFeedbackCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactRootReference &constraints,
    llvm::ArrayRef<ArtifactRootReference> evaluationEvidence,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialMappingFeedbackCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALMAPPINGFEEDBACKCANDIDATEGENERATOR_H
