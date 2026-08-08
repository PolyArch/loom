#ifndef LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H
#define LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H

#include "DSE/MappingCandidateGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/System/SystemPnrGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rootCompleteSystemPnrCandidateGeneratorKind(9);

const CandidateGeneratorDescriptor &
rootCompleteSystemPnrCandidateGeneratorDescriptor();
llvm::Error registerRootCompleteSystemPnrCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSystemPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingCandidates,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSystemPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

std::vector<CandidateGeneratorWorkUnitSummary>
rootCompleteSystemPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SystemPnrGenerationAccounting &accounting);

} // namespace loom::dse

#endif // LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H
