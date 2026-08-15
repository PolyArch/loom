#ifndef LOOM_LIB_PNR_CPSAT_SPATIALLOCALDISPOSITIONMODEL_H
#define LOOM_LIB_PNR_CPSAT_SPATIALLOCALDISPOSITIONMODEL_H

#include "SpatialBindingRelationModel.h"

#include "PnR/SpatialCandidateState.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

/// Exact-repair variables for the closed external-or-RegFIFO disposition of
/// each affected logical net. Local values are domain-local option ordinals;
/// the value after the last local option denotes external routing.
class SpatialLocalDispositionModel final {
public:
  static llvm::Expected<SpatialLocalDispositionModel>
  build(operations_research::sat::CpModelBuilder &model,
        const SpatialCandidateState &candidate,
        const SpatialBindingRelationModel &bindings,
        llvm::ArrayRef<operations_research::sat::IntVar> bindingVariables,
        llvm::ArrayRef<int> decisionVariables,
        llvm::ArrayRef<PnrIndex> affectedLogicalNets);

  llvm::ArrayRef<operations_research::sat::IntVar> variables() const {
    return variables_;
  }
  llvm::ArrayRef<PnrIndex> logicalNets() const { return logicalNets_; }
  llvm::ArrayRef<std::int64_t> currentValues() const { return currentValues_; }
  llvm::ArrayRef<std::int64_t> legalValues(PnrIndex local) const {
    return legalValues_[local];
  }
  std::int64_t externalValue(PnrIndex local) const {
    return externalValues_[local];
  }

  std::optional<PnrIndex> localForLogicalNet(PnrIndex logicalNet) const;
  std::optional<operations_research::sat::BoolVar>
  localSelected(PnrIndex logicalNet) const;
  llvm::Expected<std::optional<PnrIndex>>
  selectedOption(PnrIndex local, std::int64_t value) const;

private:
  const FrozenSpatialPnrProblem *problem_ = nullptr;
  std::vector<operations_research::sat::IntVar> variables_;
  std::vector<operations_research::sat::BoolVar> localSelected_;
  std::vector<PnrIndex> logicalNets_;
  std::vector<int> localByLogicalNet_;
  std::vector<std::vector<std::int64_t>> legalValues_;
  std::vector<std::int64_t> externalValues_;
  std::vector<std::int64_t> currentValues_;
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_CPSAT_SPATIALLOCALDISPOSITIONMODEL_H
