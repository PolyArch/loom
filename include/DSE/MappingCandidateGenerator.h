#ifndef LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H
#define LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "PnR/SpatialPnrGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind spatialPnrCandidateGeneratorKind(0);

const CandidateGeneratorDescriptor &spatialPnrCandidateGeneratorDescriptor();
llvm::Error registerSpatialPnrCandidateGenerator();

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialPnrCandidateGeneratorBinding(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &techMapping,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &constraints,
    const ::loom::pnr::ResolvedPnrConfigView &config);

/// Strictly imports one exact D/T/F/C/K binding and invokes the Spatial PnR
/// owner. Import or coupling failures are returned through the owner's Invalid
/// outcome; no partial candidate set is exposed.
::loom::pnr::SpatialPnrGenerationOutcome invokeSpatialPnrCandidateGenerator(
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H
