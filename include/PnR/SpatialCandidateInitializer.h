#ifndef LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H
#define LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H

#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

/// Builds the canonical assignment used by initializer attempt zero before
/// the explicit global routing Action. The implementation walks factorized
/// domains directly and leaves every RouteTree visibly unrouted.
llvm::Expected<SpatialCandidateStateHandle>
createCanonicalSpatialCandidate(FrozenSpatialPnrProblemHandle problem);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H
