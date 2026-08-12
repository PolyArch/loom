#ifndef LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H
#define LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "Mapping/Tech/TechMappingConfig.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rootCompleteTechMappingCandidateGeneratorKind(6);
inline constexpr CandidateGeneratorKind
    canonicalGraphTechMappingCandidateGeneratorKind(21);

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
canonicalGraphTechMappingCandidateGeneratorDescriptor();
llvm::Error registerCanonicalGraphTechMappingCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindCanonicalGraphTechMappingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> dataflowCandidates,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveCanonicalGraphTechMappingCandidateGeneratorBinding(
    const mapping::ResolvedTechMappingConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_ROOTCOMPLETETECHMAPPINGCANDIDATEGENERATOR_H
