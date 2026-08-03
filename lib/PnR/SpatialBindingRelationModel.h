#ifndef LOOM_PNR_SPATIALBINDINGRELATIONMODEL_H
#define LOOM_PNR_SPATIALBINDINGRELATIONMODEL_H

#include "InitializerRelationSolver.h"

#include "PnR/FrozenConstraintIndex.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

struct SpatialComputeBindingChoice final {
  PnrIndex placement = 0;
  PnrIndex instructionContext = 0;
};

struct SpatialMemoryBindingChoice final {
  PnrIndex placement = 0;
};

class SpatialBindingRelationModel final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialBindingRelationModel>>
  create(const FrozenSpatialRealizationIndex &realizations,
         const FrozenConstraintIndex &constraints);

  const InitializerRelationModel &relations() const { return relations_; }
  PnrIndex computeDecisionCount() const {
    return static_cast<PnrIndex>(computeChoiceOffsets_.size() - 1);
  }
  PnrIndex decisionCount() const { return relations_.decisionCount(); }
  llvm::ArrayRef<SpatialComputeBindingChoice>
  computeChoices(PnrIndex realization) const;
  llvm::ArrayRef<SpatialMemoryBindingChoice>
  memoryChoices(PnrIndex realization) const;
  std::optional<PnrIndex>
  computeChoiceOrdinal(PnrIndex realization, PnrIndex placement,
                       PnrIndex instructionContext) const;
  std::optional<PnrIndex> memoryChoiceOrdinal(PnrIndex realization,
                                              PnrIndex placement) const;
  llvm::ArrayRef<PnrIndex> decisionRelations(PnrIndex decision) const;
  bool relationSatisfied(PnrIndex relation,
                         llvm::ArrayRef<PnrIndex> choices) const {
    return relations_.relationSatisfied(relation, choices);
  }
  llvm::Error verifyChoices(llvm::ArrayRef<PnrIndex> choices) const {
    return relations_.verifyChoices(choices);
  }
  std::optional<::mapping::SpatialConstraintProjection>
  deferredProjection() const {
    return deferredProjection_;
  }

private:
  SpatialBindingRelationModel(
      InitializerRelationModel relations,
      std::vector<PnrIndex> computeChoiceOffsets,
      std::vector<SpatialComputeBindingChoice> computeChoices,
      std::vector<PnrIndex> computeContextChoiceOrdinals,
      std::vector<PnrIndex> memoryChoiceOffsets,
      std::vector<SpatialMemoryBindingChoice> memoryChoices,
      std::vector<PnrIndex> memoryPlacementChoiceOrdinals,
      std::optional<::mapping::SpatialConstraintProjection> deferredProjection)
      : relations_(std::move(relations)),
        computeChoiceOffsets_(std::move(computeChoiceOffsets)),
        computeChoices_(std::move(computeChoices)),
        computeContextChoiceOrdinals_(std::move(computeContextChoiceOrdinals)),
        memoryChoiceOffsets_(std::move(memoryChoiceOffsets)),
        memoryChoices_(std::move(memoryChoices)),
        memoryPlacementChoiceOrdinals_(
            std::move(memoryPlacementChoiceOrdinals)),
        deferredProjection_(deferredProjection) {}

  InitializerRelationModel relations_;
  std::vector<PnrIndex> computeChoiceOffsets_;
  std::vector<SpatialComputeBindingChoice> computeChoices_;
  std::vector<PnrIndex> computeContextChoiceOrdinals_;
  std::vector<PnrIndex> memoryChoiceOffsets_;
  std::vector<SpatialMemoryBindingChoice> memoryChoices_;
  std::vector<PnrIndex> memoryPlacementChoiceOrdinals_;
  std::optional<::mapping::SpatialConstraintProjection> deferredProjection_;
};

} // namespace loom::pnr::detail

#endif // LOOM_PNR_SPATIALBINDINGRELATIONMODEL_H
