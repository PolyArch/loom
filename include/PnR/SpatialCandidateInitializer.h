#ifndef LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H
#define LOOM_PNR_SPATIALCANDIDATEINITIALIZER_H

#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

/// Immutable projection of the canonical root-placement preference used by
/// one initializer attempt. It contains no CandidateState, solver journal,
/// scratch storage, random stream, or remaining-work state.
struct SpatialCandidateInitializerPreference final {
  std::uint64_t residualExternalSinkCount = 0;
  std::uint64_t selectedRegisterFifoTransferCount = 0;
  std::uint64_t topologyUnreachableSelectionCount = 0;
  std::uint64_t topologyHopSum = 0;
  std::uint64_t topologyRefinementUnreachableSelectionCount = 0;
  std::uint64_t topologyRefinementHopSum = 0;
  std::uint64_t maximumComputeOccurrenceSelections = 0;
  std::uint64_t maximumEndpointSelections = 0;
  std::uint64_t staticSchedulePressure = 0;
};

struct SpatialCandidateInitializerAttempt final {
  SpatialCandidateStateHandle candidate;
  SpatialCandidateInitializerPreference preference;
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
