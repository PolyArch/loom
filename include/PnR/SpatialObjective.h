#ifndef LOOM_PNR_SPATIALOBJECTIVE_H
#define LOOM_PNR_SPATIALOBJECTIVE_H

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>

namespace loom::pnr {

class SpatialCandidateState;

/// Preflighted, removable adapter from the exact selected objective catalog to
/// Mapping-owned Spatial Candidate values. It stores only selected source
/// masks; MappingObjective and ObjectiveProgram remain the semantic owners.
class SpatialObjectiveProgram final {
public:
  static llvm::Expected<SpatialObjectiveProgram>
  get(const ResolvedObjectiveCatalogs &catalogs,
      const ResolvedPnrObjectiveSelection &selection);

  llvm::Expected<dse::ObjectiveVector>
  evaluate(const SpatialCandidateState &candidate) const;
  llvm::Expected<dse::ObjectiveWideValue>
  selectedEnergy(const dse::ObjectiveVector &vector) const;
  llvm::Expected<dse::ObjectiveSignedDifference>
  selectedEnergyDifference(const dse::ObjectiveVector &left,
                           const dse::ObjectiveVector &right) const;
  llvm::Expected<int>
  compareSelectedRank(const dse::ObjectiveVector &left,
                      llvm::ArrayRef<std::uint8_t> leftCandidateKey,
                      const dse::ObjectiveVector &right,
                      llvm::ArrayRef<std::uint8_t> rightCandidateKey) const;

private:
  SpatialObjectiveProgram(dse::ObjectiveProgram program,
                          std::uint64_t selectedViolations,
                          std::uint64_t selectedMeasures,
                          std::uint32_t selectedTotalOrdering,
                          std::uint32_t selectedSearchEnergy)
      : program_(std::move(program)), selectedViolations_(selectedViolations),
        selectedMeasures_(selectedMeasures),
        selectedTotalOrdering_(selectedTotalOrdering),
        selectedSearchEnergy_(selectedSearchEnergy) {}

  dse::ObjectiveProgram program_;
  std::uint64_t selectedViolations_ = 0;
  std::uint64_t selectedMeasures_ = 0;
  std::uint32_t selectedTotalOrdering_ = 0;
  std::uint32_t selectedSearchEnergy_ = 0;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALOBJECTIVE_H
