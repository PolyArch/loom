#ifndef LOOM_PNR_DETERMINISTICSEARCHPROTOCOL_H
#define LOOM_PNR_DETERMINISTICSEARCHPROTOCOL_H

#include "DSE/Objective.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>

namespace loom::pnr {

enum class PnrRandomStreamPurpose : std::uint32_t {
  InitializerDiversification = 0,
  Calibration = 1,
  ActionProposal = 2,
  Acceptance = 3,
  ExactRepair = 4,
};

/// Replay-stable Sha256SeededXoshiro256StarStar_1_0 stream.
class DeterministicPnrRandomStream final {
public:
  static DeterministicPnrRandomStream create(std::uint64_t masterSeed,
                                             std::uint64_t seedIndex,
                                             PnrRandomStreamPurpose purpose);

  std::uint64_t nextU64();
  llvm::Expected<std::uint64_t> nextBounded(std::uint64_t upperBound);

private:
  explicit DeterministicPnrRandomStream(std::array<std::uint64_t, 4> state)
      : state_(state) {}

  std::array<std::uint64_t, 4> state_;
};

/// Checked-in ExpNegativeQ64Table_1_0 constants for positive ratio indexes.
llvm::ArrayRef<std::uint64_t> expNegativeQ64Thresholds();

/// Requires a positive ratio index. Values beyond the checked-in table are 0.
std::uint64_t expNegativeQ64Threshold(std::uint64_t ratioIndex);

/// Applies ExpNegativeQ64Table_1_0 to new-minus-old selected search energy.
llvm::Expected<bool>
acceptAnnealingDelta(dse::ObjectiveSignedDifference delta,
                     std::uint64_t temperature,
                     DeterministicPnrRandomStream &acceptanceStream);

/// Calibrates the smallest integer temperature whose table probability reaches
/// the configured target for the stable quantile of positive energy deltas.
llvm::Expected<std::uint64_t> calibrateAnnealingTemperature(
    const ResolvedPnrAnnealingPolicy &policy,
    llvm::ArrayRef<dse::ObjectiveWideValue> positiveDeltas);

llvm::Expected<std::uint64_t>
annealingProposalsPerLevel(const ResolvedPnrAnnealingPolicy &policy,
                           std::uint64_t movableDecisionCount);

/// Finite cooling schedule with exactly one complete minimum-temperature level.
class AnnealingTemperatureSchedule final {
public:
  static llvm::Expected<AnnealingTemperatureSchedule>
  create(const ResolvedPnrAnnealingPolicy &policy,
         std::uint64_t initialTemperature);

  std::uint64_t temperature() const { return temperature_; }
  bool isFinalLevel() const { return temperature_ == minimumTemperature_; }

  /// Advances after the current level completes. Returns false after the
  /// minimum-temperature level rather than executing it again.
  bool advanceAfterCompletedLevel();

private:
  AnnealingTemperatureSchedule(std::uint64_t minimumTemperature,
                               ResolvedExactRatio coolingRatio,
                               std::uint64_t levelLimit,
                               std::uint64_t temperature)
      : minimumTemperature_(minimumTemperature), coolingRatio_(coolingRatio),
        levelLimit_(levelLimit), temperature_(temperature) {}

  std::uint64_t minimumTemperature_;
  ResolvedExactRatio coolingRatio_;
  std::uint64_t levelLimit_;
  std::uint64_t completedLevelCount_ = 0;
  std::uint64_t temperature_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_DETERMINISTICSEARCHPROTOCOL_H
