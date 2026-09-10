#ifndef LOOM_LIB_PNR_SPATIALCANDIDATEINITIALIZATIONPREFERENCE_H
#define LOOM_LIB_PNR_SPATIALCANDIDATEINITIALIZATIONPREFERENCE_H

#include "InitializerRelationSolver.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

llvm::Error spatialInitializerError(const llvm::Twine &message);

struct PreferredRootAssignment final {
  std::vector<PnrIndex> choices;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t changedComputeRoots = 0;
  std::uint64_t selectedTemporalComputeRoots = 0;
  std::uint64_t maximumContextSelections = 0;
  std::uint64_t maximumComputeOccurrenceSelections = 0;
  std::uint64_t distinctComputeOccurrences = 0;
  std::uint64_t changedMemoryRoots = 0;
  std::uint64_t selectedTemporalMemoryRoots = 0;
  std::uint64_t maximumMemorySelections = 0;
  std::uint64_t distinctMemoryOccurrences = 0;
  std::uint64_t topologyScoredRoots = 0;
  std::uint64_t topologyBoundaryAnchorIncidences = 0;
  std::uint64_t topologyHopSum = 0;
  std::uint64_t topologyUnreachableSelections = 0;
  std::uint64_t topologyRefinedComputeRoots = 0;
  std::uint64_t topologyRefinementHopSum = 0;
  std::uint64_t topologyRefinementUnreachableSelections = 0;
  std::uint64_t changedPortAttachments = 0;
  std::uint64_t changedGraphBoundaryAttachments = 0;
  std::uint64_t pairingScoredPortAttachments = 0;
  std::uint64_t preferredSharedOperandIngressPressure = 0;
  std::uint64_t structurallyAdjustedRootPreferences = 0;
  std::uint64_t localTransferRefinedComputeRoots = 0;
  std::uint64_t maximumEndpointSelections = 0;
  bool applied = false;
  llvm::StringRef status = "unchanged";
};

/// Refine the complete feasible root assignment with soft placement and
/// attachment preferences, preserving the relation solver's hard domains.
llvm::Expected<PreferredRootAssignment> preferScheduleAwareRootPlacements(
    const FrozenSpatialPnrProblem &problem, std::uint32_t attemptOrdinal,
    InitializerRelationSolver &solver, InitializerRelationSolveResult baseline,
    std::uint64_t assignmentLimit);

} // namespace loom::pnr::detail

#endif
