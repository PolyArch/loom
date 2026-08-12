#ifndef LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H
#define LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "Mapping/Tech/TechMappingConfig.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rootCompleteTechMappingCandidateGeneratorKind(6);
inline constexpr CandidateGeneratorKind
    applicationGraphTechMappingCandidateGeneratorKind(21);

const CandidateGeneratorDescriptor &
rootCompleteTechMappingCandidateGeneratorDescriptor();
llvm::Error registerRootCompleteTechMappingCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteTechMappingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> dataflowCandidates,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteTechMappingCandidateGeneratorBinding(
    const mapping::ResolvedTechMappingConfigView &config);

const CandidateGeneratorDescriptor &
applicationGraphTechMappingCandidateGeneratorDescriptor();
llvm::Error registerApplicationGraphTechMappingCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindApplicationGraphTechMappingCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &systemConstraints,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveApplicationGraphTechMappingCandidateGeneratorBinding(
    const mapping::ResolvedTechMappingConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H
