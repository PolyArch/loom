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
  get(const ResolvedObjectiveCatalogs &catalogs);

  llvm::Expected<dse::ObjectiveVector>
  evaluate(const SpatialCandidateState &candidate) const;

private:
  SpatialObjectiveProgram(dse::ObjectiveProgram program,
                          std::uint64_t selectedViolations,
                          std::uint64_t selectedMeasures)
      : program_(std::move(program)), selectedViolations_(selectedViolations),
        selectedMeasures_(selectedMeasures) {}

  dse::ObjectiveProgram program_;
  std::uint64_t selectedViolations_ = 0;
  std::uint64_t selectedMeasures_ = 0;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALOBJECTIVE_H
