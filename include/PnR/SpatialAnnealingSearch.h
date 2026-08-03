#ifndef LOOM_PNR_SPATIALANNEALINGSEARCH_H
#define LOOM_PNR_SPATIALANNEALINGSEARCH_H

#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::pnr {

struct SpatialAnnealingStatistics final {
  std::uint64_t initialTemperature = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t calibrationProbeCount = 0;
  std::uint64_t calibrationTransitionFailureCount = 0;
  std::uint64_t temperatureLevelCount = 0;
  std::uint64_t minimumTemperatureLevelCount = 0;
  std::uint64_t annealingProposalSlots = 0;
  std::uint64_t annealingProbeCount = 0;
  std::uint64_t acceptedActionCount = 0;
  std::uint64_t rejectedActionCount = 0;
  std::uint64_t annealingTransitionFailureCount = 0;

  friend bool operator==(const SpatialAnnealingStatistics &lhs,
                         const SpatialAnnealingStatistics &rhs) {
    return lhs.initialTemperature == rhs.initialTemperature &&
           lhs.calibrationProposalSlots == rhs.calibrationProposalSlots &&
           lhs.calibrationProbeCount == rhs.calibrationProbeCount &&
           lhs.calibrationTransitionFailureCount ==
               rhs.calibrationTransitionFailureCount &&
           lhs.temperatureLevelCount == rhs.temperatureLevelCount &&
           lhs.minimumTemperatureLevelCount ==
               rhs.minimumTemperatureLevelCount &&
           lhs.annealingProposalSlots == rhs.annealingProposalSlots &&
           lhs.annealingProbeCount == rhs.annealingProbeCount &&
           lhs.acceptedActionCount == rhs.acceptedActionCount &&
           lhs.rejectedActionCount == rhs.rejectedActionCount &&
           lhs.annealingTransitionFailureCount ==
               rhs.annealingTransitionFailureCount;
  }
};

/// Reusable worker-local orchestration for one fixed Spatial restart. The
/// candidate remains the sole mutable owner; this scratch retains only Action
/// domains, transition machinery, and calibration samples.
class SpatialAnnealingSearchScratch final {
public:
  llvm::Expected<SpatialAnnealingStatistics>
  run(SpatialCandidateState &candidate, std::uint64_t seedAttemptOrdinal);

  std::size_t retainedStorageBytes() const;

private:
  llvm::Expected<bool> consumeTransitionFailure(llvm::Error failure);

  SpatialActionDomainScratch actionDomain_;
  SpatialActionExecutorScratch actionExecutor_;
  std::vector<dse::ObjectiveWideValue> positiveCalibrationDeltas_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALANNEALINGSEARCH_H
