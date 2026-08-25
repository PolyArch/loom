#ifndef LOOM_PNR_SPATIALCANONICALSEED_H
#define LOOM_PNR_SPATIALCANONICALSEED_H

#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

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

/// Invocation-local ownership handoff for one already constructed restart.
/// Before consumption exactly one of seed and failure is populated. The formal
/// Spatial owner consumes this object once; it is never a cache or part of
/// Mapping identity.
struct SpatialPathFinderSeedHandoff final {
  std::uint32_t attemptOrdinal = 0;
  std::optional<FrozenSpatialPnrCacheKey> problemCacheKey;
  SpatialPathFinderSeedWorkSummary workSummary;
  std::optional<SpatialPathFinderSeed> seed;
  std::optional<llvm::Error> failure;
  bool consumed = false;

  ~SpatialPathFinderSeedHandoff() {
    if (failure) {
      llvm::consumeError(std::move(*failure));
      failure.reset();
    }
  }

  SpatialPathFinderSeedHandoff(const SpatialPathFinderSeedHandoff &) = delete;
  SpatialPathFinderSeedHandoff &
  operator=(const SpatialPathFinderSeedHandoff &) = delete;
  SpatialPathFinderSeedHandoff() = default;
  SpatialPathFinderSeedHandoff(SpatialPathFinderSeedHandoff &&) = delete;
  SpatialPathFinderSeedHandoff &operator=(SpatialPathFinderSeedHandoff &&) =
      delete;
};

using SpatialPathFinderSeedHandoffHandle =
    std::shared_ptr<SpatialPathFinderSeedHandoff>;

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
