#ifndef LOOM_DSE_ROOTCOMPLETESPATIALPNRCANDIDATEGENERATOR_H
#define LOOM_DSE_ROOTCOMPLETESPATIALPNRCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "PnR/PnrConfig.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rootCompleteSpatialPnrCandidateGeneratorKind(7);

const CandidateGeneratorDescriptor &
rootCompleteSpatialPnrCandidateGeneratorDescriptor();
llvm::Error registerRootCompleteSpatialPnrCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSpatialPnrCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> techMappingCandidates,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_ROOTCOMPLETESPATIALPNRCANDIDATEGENERATOR_H
