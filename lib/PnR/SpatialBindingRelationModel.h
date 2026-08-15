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

enum class SpatialBindingRelationRole : std::uint8_t {
  Structural,
  Constraint,
  Progress,
};

class SpatialBindingRelationModel final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialBindingRelationModel>>
  create(const ArtifactIdentity &dataflowIdentity,
         const FrozenSpatialRealizationIndex &realizations,
         const FrozenConstraintIndex &constraints,
         const FrozenSpatialTransferIndex &transfers,
         const FrozenSpatialPortIndex &ports,
         const FrozenSpatialRoutingGraph &routing);

  const InitializerRelationModel &relations() const { return relations_; }
  PnrIndex computeDecisionCount() const {
    return static_cast<PnrIndex>(computeChoiceOffsets_.size() - 1);
  }
  PnrIndex memoryDecisionCount() const {
    return static_cast<PnrIndex>(memoryChoiceOffsets_.size() - 1);
  }
  PnrIndex portDecisionOffset() const {
    return computeDecisionCount() + memoryDecisionCount();
  }
  PnrIndex graphBoundaryDecisionOffset() const {
    return portDecisionOffset() + portDecisionCount();
  }
  PnrIndex portDecisionCount() const {
    return static_cast<PnrIndex>(portAttachmentChoiceOffsets_.size() - 1);
  }
  PnrIndex graphBoundaryDecisionCount() const {
    return static_cast<PnrIndex>(graphBoundaryAttachmentChoiceOffsets_.size() -
                                 1);
  }
  PnrIndex realizationDecisionCount() const {
    return computeDecisionCount() + memoryDecisionCount();
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
  llvm::ArrayRef<PnrIndex> portAttachmentChoices(PnrIndex demand) const;
  llvm::ArrayRef<PnrIndex>
  graphBoundaryAttachmentChoices(PnrIndex boundary) const;
  std::optional<PnrIndex> portAttachmentChoiceOrdinal(PnrIndex demand,
                                                      PnrIndex option) const;
  std::optional<PnrIndex>
  graphBoundaryAttachmentChoiceOrdinal(PnrIndex boundary,
                                       PnrIndex option) const;
  llvm::ArrayRef<PnrIndex> decisionRelations(PnrIndex decision) const;
  bool relationIsConstraint(PnrIndex relation) const {
    return relationRoles_[relation] == SpatialBindingRelationRole::Constraint;
  }
  bool relationIsStructural(PnrIndex relation) const {
    return relationRoles_[relation] == SpatialBindingRelationRole::Structural;
  }
  bool relationRequiresRouteRepairEncoding(PnrIndex relation) const {
    return !relationIsStructural(relation);
  }
  bool relationSatisfied(PnrIndex relation,
                         llvm::ArrayRef<PnrIndex> choices) const {
    return relations_.relationSatisfied(relation, choices);
  }
  llvm::Error verifyChoices(llvm::ArrayRef<PnrIndex> choices) const {
    return relations_.verifyChoices(choices);
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
      std::vector<PnrIndex> portAttachmentChoiceOffsets,
      std::vector<PnrIndex> graphBoundaryAttachmentChoiceOffsets,
      std::vector<PnrIndex> attachmentChoices,
      std::vector<PnrIndex> attachmentOptionChoiceOrdinals,
      std::vector<SpatialBindingRelationRole> relationRoles)
      : relations_(std::move(relations)),
        computeChoiceOffsets_(std::move(computeChoiceOffsets)),
        computeChoices_(std::move(computeChoices)),
        computeContextChoiceOrdinals_(std::move(computeContextChoiceOrdinals)),
        memoryChoiceOffsets_(std::move(memoryChoiceOffsets)),
        memoryChoices_(std::move(memoryChoices)),
        memoryPlacementChoiceOrdinals_(
            std::move(memoryPlacementChoiceOrdinals)),
        portAttachmentChoiceOffsets_(std::move(portAttachmentChoiceOffsets)),
        graphBoundaryAttachmentChoiceOffsets_(
            std::move(graphBoundaryAttachmentChoiceOffsets)),
        attachmentChoices_(std::move(attachmentChoices)),
        attachmentOptionChoiceOrdinals_(
            std::move(attachmentOptionChoiceOrdinals)),
        relationRoles_(std::move(relationRoles)) {}

  InitializerRelationModel relations_;
  std::vector<PnrIndex> computeChoiceOffsets_;
  std::vector<SpatialComputeBindingChoice> computeChoices_;
  std::vector<PnrIndex> computeContextChoiceOrdinals_;
  std::vector<PnrIndex> memoryChoiceOffsets_;
  std::vector<SpatialMemoryBindingChoice> memoryChoices_;
  std::vector<PnrIndex> memoryPlacementChoiceOrdinals_;
  std::vector<PnrIndex> portAttachmentChoiceOffsets_;
  std::vector<PnrIndex> graphBoundaryAttachmentChoiceOffsets_;
  std::vector<PnrIndex> attachmentChoices_;
  std::vector<PnrIndex> attachmentOptionChoiceOrdinals_;
  std::vector<SpatialBindingRelationRole> relationRoles_;
};

} // namespace loom::pnr::detail

#endif // LOOM_PNR_SPATIALBINDINGRELATIONMODEL_H
