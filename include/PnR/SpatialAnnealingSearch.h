#ifndef LOOM_PNR_SPATIALANNEALINGSEARCH_H
#define LOOM_PNR_SPATIALANNEALINGSEARCH_H

#include "Common/ExecutionControl.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::pnr {

struct SpatialPathFinderSeed;

struct SpatialAnnealingStatistics final {
  bool interrupted = false;
  bool exactClosureReached = false;
  bool completionGoalReached = false;
  std::uint64_t initialTemperature = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t calibrationProbeCount = 0;
  std::uint64_t calibrationTransitionFailureCount = 0;
  std::uint64_t temperatureLevelCount = 0;
  std::uint64_t minimumTemperatureLevelCount = 0;
  std::uint64_t annealingProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t annealingProbeCount = 0;
  std::uint64_t acceptedActionCount = 0;
  std::uint64_t acceptedWorseningActionCount = 0;
  std::uint64_t rejectedActionCount = 0;
  std::uint64_t semanticNoopActionCount = 0;
  std::uint64_t cachedInactiveActionCount = 0;
  std::uint64_t annealingTransitionFailureCount = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
  bool bestFeasibleIncumbentRestored = false;

  friend bool operator==(const SpatialAnnealingStatistics &lhs,
                         const SpatialAnnealingStatistics &rhs) {
    return lhs.interrupted == rhs.interrupted &&
           lhs.exactClosureReached == rhs.exactClosureReached &&
           lhs.completionGoalReached == rhs.completionGoalReached &&
           lhs.initialTemperature == rhs.initialTemperature &&
           lhs.calibrationProposalSlots == rhs.calibrationProposalSlots &&
           lhs.calibrationProbeCount == rhs.calibrationProbeCount &&
           lhs.calibrationTransitionFailureCount ==
               rhs.calibrationTransitionFailureCount &&
           lhs.temperatureLevelCount == rhs.temperatureLevelCount &&
           lhs.minimumTemperatureLevelCount ==
               rhs.minimumTemperatureLevelCount &&
           lhs.annealingProposalSlots == rhs.annealingProposalSlots &&
           lhs.annealingBaseProposalSlots == rhs.annealingBaseProposalSlots &&
           lhs.annealingMovableProposalSlots ==
               rhs.annealingMovableProposalSlots &&
           lhs.annealingProbeCount == rhs.annealingProbeCount &&
           lhs.acceptedActionCount == rhs.acceptedActionCount &&
           lhs.acceptedWorseningActionCount ==
               rhs.acceptedWorseningActionCount &&
           lhs.rejectedActionCount == rhs.rejectedActionCount &&
           lhs.semanticNoopActionCount == rhs.semanticNoopActionCount &&
           lhs.cachedInactiveActionCount == rhs.cachedInactiveActionCount &&
           lhs.annealingTransitionFailureCount ==
               rhs.annealingTransitionFailureCount &&
           lhs.endpointExpansions == rhs.endpointExpansions &&
           lhs.negotiationIterations == rhs.negotiationIterations &&
           lhs.bestFeasibleIncumbentRestored ==
               rhs.bestFeasibleIncumbentRestored;
  }
};

/// Reusable worker-local orchestration for one fixed Spatial restart. The
/// candidate remains the sole mutable owner; this scratch retains only Action
/// domains, transition machinery, and calibration samples.
class SpatialAnnealingSearchScratch final {
public:
  llvm::Expected<SpatialAnnealingStatistics>
  run(SpatialCandidateStateHandle &candidate, std::uint64_t seedAttemptOrdinal,
      ExecutionControlView executionControl = {});

  llvm::Expected<SpatialAnnealingStatistics>
  run(SpatialPathFinderSeed &seed, ExecutionControlView executionControl = {});

  std::size_t retainedStorageBytes() const;

private:
  llvm::Expected<bool> consumeTransitionFailure(llvm::Error failure);

  SpatialActionDomainScratch actionDomain_;
  SpatialActionExecutorScratch actionExecutor_;
  std::vector<dse::ObjectiveWideValue> positiveCalibrationDeltas_;
  std::vector<SpatialActionKey> inactiveActionKeys_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALANNEALINGSEARCH_H
