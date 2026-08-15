#ifndef LOOM_LIB_PNR_SPATIALCANDIDATETOPOLOGYPREFERENCE_H
#define LOOM_LIB_PNR_SPATIALCANDIDATETOPOLOGYPREFERENCE_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct SpatialCandidateTopologyScores final {
  std::vector<std::uint64_t> distances;
  std::vector<std::uint8_t> unreachable;
  std::uint64_t activeIncidences = 0;
  std::uint64_t activeBoundaryAnchorIncidences = 0;
};

/// Scores candidate root placements against already selected neighboring roots
/// and fixed Module-boundary attachment domains. Compute and memory roots share
/// the binding relation model's dense root prefix.
class SpatialCandidateTopologyPreference final {
public:
  static llvm::Expected<SpatialCandidateTopologyPreference>
  create(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<SpatialCandidateTopologyScores>
  scoreChoices(PnrIndex root, llvm::ArrayRef<PnrIndex> choicePlacements,
               llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals);

  llvm::Expected<llvm::ArrayRef<std::uint32_t>>
  hopDistancesFrom(llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions,
                   std::uint32_t payloadWidthBits, bool forward);

private:
  struct Incidence final {
    PnrIndex neighbor = getInvalidPnrIndex();
    PnrIndex graphBoundary = getInvalidPnrIndex();
    PnrIndex candidateDemand = 0;
    PnrIndex neighborDemand = 0;
    std::uint32_t payloadWidthBits = 0;
    std::uint64_t distanceWeight = 1;
    bool candidateIsSource = false;
  };

  explicit SpatialCandidateTopologyPreference(
      const FrozenSpatialPnrProblem &problem)
      : problem_(&problem) {}

  llvm::Expected<PnrIndex>
  selectedRootPlacement(PnrIndex root,
                        llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals) const;

  llvm::Expected<llvm::ArrayRef<FrozenSpatialAttachmentOption>>
  attachmentOptionsForPlacement(PnrIndex demandOrdinal,
                                PnrIndex placement) const;

  llvm::Error
  fillHopDistances(llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions,
                   std::uint32_t payloadWidthBits, bool forward);

  const FrozenSpatialPnrProblem *problem_;
  std::vector<std::vector<Incidence>> incidences_;
  std::vector<std::uint32_t> hopDistances_;
  std::vector<PnrIndex> hopWorklist_;
  std::vector<FrozenSpatialAttachmentOption> fixedOptionScratch_;
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALCANDIDATETOPOLOGYPREFERENCE_H
