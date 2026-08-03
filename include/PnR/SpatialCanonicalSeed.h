#ifndef LOOM_PNR_SPATIALCANONICALSEED_H
#define LOOM_PNR_SPATIALCANONICALSEED_H

#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace loom::pnr {

struct SpatialCanonicalSeed final {
  SpatialCandidateStateHandle candidate;
  SpatialPathFinderClosureResult routing;
};

/// Builds initializer attempt zero and applies its explicit global PathFinder
/// routing Action. The returned Candidate remains ephemeral and may still
/// carry policy-admitted non-routing violations; this function never
/// materializes or publishes a SpatialMapping.
llvm::Expected<SpatialCanonicalSeed> createCanonicalPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem,
    llvm::ArrayRef<RouteCost> evaluationPriorities = {});

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANONICALSEED_H
