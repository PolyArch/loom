#ifndef LOOM_LIB_PNR_SPATIALCANDIDATELOCALTRANSFERPREFERENCE_H
#define LOOM_LIB_PNR_SPATIALCANDIDATELOCALTRANSFERPREFERENCE_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct SpatialCandidateLocalTransferScores final {
  std::vector<std::uint64_t> matchedNets;
  std::vector<std::uint64_t> unmatchedNets;
  std::uint64_t activeNets = 0;
};

/// Scores compute placements by the RegFIFO dispositions they enable with
/// already selected neighboring roots. This is a search preference only;
/// binding relations and the local-transfer allocator remain the legality
/// owners.
class SpatialCandidateLocalTransferPreference final {
public:
  static llvm::Expected<SpatialCandidateLocalTransferPreference>
  create(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<SpatialCandidateLocalTransferScores>
  scoreChoices(PnrIndex realization, llvm::ArrayRef<PnrIndex> choicePlacements,
               llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals) const;

private:
  explicit SpatialCandidateLocalTransferPreference(
      const FrozenSpatialPnrProblem &problem)
      : problem_(&problem) {}

  const FrozenSpatialPnrProblem *problem_;
  std::vector<std::vector<PnrIndex>> logicalNetsByRealization_;
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALCANDIDATELOCALTRANSFERPREFERENCE_H
