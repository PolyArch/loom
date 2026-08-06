#ifndef LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H
#define LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H

#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

struct SpatialCandidateInitializerAttempt final {
  SpatialCandidateStateHandle candidate;
};

/// Builds one exact fixed initializer-attempt slot. Attempt zero uses
/// canonical choice order; every other slot uses only its domain-separated
/// InitializerDiversification stream. The returned candidate leaves every
/// RouteTree visibly unrouted for the explicit global routing Action.
llvm::Expected<SpatialCandidateInitializerAttempt>
createSpatialCandidateInitializerAttempt(FrozenSpatialPnrProblemHandle problem,
                                         std::uint32_t attemptOrdinal,
                                         std::uint64_t &assignmentAttempts);

/// Builds the canonical assignment used by initializer attempt zero before
/// the explicit global routing Action. The implementation walks factorized
/// domains directly and leaves every RouteTree visibly unrouted.
llvm::Expected<SpatialCandidateStateHandle>
createCanonicalSpatialCandidate(FrozenSpatialPnrProblemHandle problem);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H
