#ifndef LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H
#define LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H

#include "DSE/MappingCandidateGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/System/SystemPnrGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rootCompleteSystemPnrCandidateGeneratorKind(9);
inline constexpr CandidateGeneratorKind
    applicationSystemPnrCandidateGeneratorKind(22);

const CandidateGeneratorDescriptor &
rootCompleteSystemPnrCandidateGeneratorDescriptor();
llvm::Error registerRootCompleteSystemPnrCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRootCompleteSystemPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingCandidates,
    const ArtifactRootReference &fabric,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRootCompleteSystemPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

const CandidateGeneratorDescriptor &
applicationSystemPnrCandidateGeneratorDescriptor();
llvm::Error registerApplicationSystemPnrCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindApplicationSystemPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingCandidates,
    const ArtifactRootReference &fabric,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles,
    const ArtifactRootReference &systemConstraints);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveApplicationSystemPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

std::vector<CandidateGeneratorWorkUnitSummary>
rootCompleteSystemPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SystemPnrGenerationAccounting &accounting);

} // namespace loom::dse

#endif // LOOM_DSE_ROOTCOMPLETESYSTEMPNRCANDIDATEGENERATOR_H
