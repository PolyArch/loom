#ifndef LOOM_PNR_SPATIALCANONICALSEED_H
#define LOOM_PNR_SPATIALCANONICALSEED_H

#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

struct SpatialPathFinderSeedWorkSummary final {
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
};

struct SpatialPathFinderSeed final {
  SpatialCandidateStateHandle candidate;
  std::uint32_t attemptOrdinal = 0;
  SpatialCandidateInitializerPreference initializerPreference;
};

/// Builds one exact initializer slot and applies its explicit global
/// PathFinder routing Action. A failed slot is returned as an error and is not
/// replaced by another attempt ordinal.
llvm::Expected<SpatialPathFinderSeed> createPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem, std::uint32_t attemptOrdinal,
    SpatialPathFinderSeedWorkSummary &workSummary,
    llvm::ArrayRef<RouteCost> evaluationPriorities = {});

/// Builds initializer attempt zero and applies its explicit global PathFinder
/// routing Action. The returned Candidate remains ephemeral and may still
/// carry policy-admitted temporary violations; this function never
/// materializes or publishes a SpatialMapping.
llvm::Expected<SpatialPathFinderSeed> createCanonicalPathFinderSpatialSeed(
    FrozenSpatialPnrProblemHandle problem,
    SpatialPathFinderSeedWorkSummary &workSummary,
    llvm::ArrayRef<RouteCost> evaluationPriorities = {});

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANONICALSEED_H
