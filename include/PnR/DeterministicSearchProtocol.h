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

} // namespace loom::pnr

#endif // LOOM_PNR_DETERMINISTICSEARCHPROTOCOL_H
