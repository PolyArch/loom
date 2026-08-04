#ifndef LOOM_DSE_PLAN_H
#define LOOM_DSE_PLAN_H

#include "DSE/CandidateGenerator.h"
#include "DSE/PlanValue.h"
#include "DSE/PromotionAcquisition.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {

class ResolvedDseConfigView;

struct GeneratePlanNodeDefinition final {
  CandidateGeneratorDescriptorRef descriptor;
  std::vector<PlanInputBinding> inputBindings;
  std::vector<std::uint8_t> canonicalConfigBytes;
  ComponentViewDigest configDigest;
};

enum class PromotePurpose : std::uint32_t {
  CandidateSelection = 0,
  ModelRelease = 1,
};

struct PromotePlanNodeDefinition final {
  PromotionAcquisitionDescriptorRef acquisition;
  std::vector<PlanInputBinding> inputBindings;
  std::vector<std::uint8_t> canonicalConfigBytes;
  ComponentViewDigest configDigest;
  QualityGatePolicyRef qualityGate;
  CandidateSelectionPolicy selection;
  PromotePurpose purpose = PromotePurpose::CandidateSelection;
};

using DsePlanNodeDefinition =
    std::variant<GeneratePlanNodeDefinition, PromotePlanNodeDefinition>;

class ResolvedGeneratePlanNode final {
public:
  CandidateGeneratorDescriptorRef descriptorRef() const { return descriptor_; }
  llvm::ArrayRef<PlanInputBinding> inputBindings() const {
    return inputBindings_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }

private:
  ResolvedGeneratePlanNode(CandidateGeneratorDescriptorRef descriptor,
                           std::vector<PlanInputBinding> inputBindings,
                           std::vector<std::uint8_t> canonicalConfigBytes,
                           ComponentViewDigest configDigest)
      : descriptor_(descriptor), inputBindings_(std::move(inputBindings)),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest) {}

  CandidateGeneratorDescriptorRef descriptor_;
  std::vector<PlanInputBinding> inputBindings_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;

  friend class ResolvedDsePlan;
};

class ResolvedPromotePlanNode final {
public:
  PromotionAcquisitionDescriptorRef acquisitionRef() const {
    return acquisitionBinding_.descriptorRef();
  }
  const ResolvedPromotionAcquisitionBinding &acquisitionBinding() const {
    return acquisitionBinding_;
  }
  llvm::ArrayRef<PlanInputBinding> inputBindings() const {
    return inputBindings_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return acquisitionBinding_.canonicalConfigBytes();
  }
  const ComponentViewDigest &configDigest() const {
    return acquisitionBinding_.configDigest();
  }
  QualityGatePolicyRef qualityGateRef() const { return qualityGate_; }
  const CandidateSelectionPolicy &selection() const { return selection_; }
  llvm::ArrayRef<EvidenceObligationTemplateRef> objectiveObligations() const {
    return objectiveObligations_;
  }
  PromotePurpose purpose() const { return purpose_; }

private:
  ResolvedPromotePlanNode(
      ResolvedPromotionAcquisitionBinding acquisitionBinding,
      std::vector<PlanInputBinding> inputBindings,
      QualityGatePolicyRef qualityGate, CandidateSelectionPolicy selection,
      std::vector<EvidenceObligationTemplateRef> objectiveObligations,
      PromotePurpose purpose)
      : acquisitionBinding_(std::move(acquisitionBinding)),
        inputBindings_(std::move(inputBindings)),
        qualityGate_(std::move(qualityGate)), selection_(std::move(selection)),
        objectiveObligations_(std::move(objectiveObligations)),
        purpose_(purpose) {}

  ResolvedPromotionAcquisitionBinding acquisitionBinding_;
  std::vector<PlanInputBinding> inputBindings_;
  QualityGatePolicyRef qualityGate_;
  CandidateSelectionPolicy selection_;
  std::vector<EvidenceObligationTemplateRef> objectiveObligations_;
  PromotePurpose purpose_;

  friend class ResolvedDsePlan;
};

using ResolvedDsePlanNode =
    std::variant<ResolvedGeneratePlanNode, ResolvedPromotePlanNode>;

/// Immutable typed use-def resolution for one ordered Generate/Promote block.
class ResolvedDsePlan final {
public:
  static llvm::Expected<ResolvedDsePlan>
  get(llvm::ArrayRef<DsePlanNodeDefinition> nodes,
      llvm::ArrayRef<EvidenceObligationTemplate> evidenceObligationTemplates,
      const ResolvedObjectiveCatalogs &objectiveCatalogs,
      llvm::ArrayRef<QualityGatePolicy> qualityGates);

  llvm::ArrayRef<ResolvedDsePlanNode> nodes() const { return nodes_; }
  const PlanValueDescriptor *resolve(PlanOutputRef output) const;
  const QualityGatePolicy *resolve(QualityGatePolicyRef gate) const;
  llvm::ArrayRef<QualityGatePolicy> qualityGatePolicies() const {
    return qualityGates_;
  }
  const ObjectiveProgram *objectiveProgram() const {
    return objectiveProgram_ ? &*objectiveProgram_ : nullptr;
  }

private:
  ResolvedDsePlan(std::vector<ResolvedDsePlanNode> nodes,
                  std::vector<std::uint64_t> outputOffsets,
                  std::vector<PlanValueDescriptor> outputs,
                  std::vector<QualityGatePolicy> qualityGates,
                  std::optional<ObjectiveProgram> objectiveProgram)
      : nodes_(std::move(nodes)), outputOffsets_(std::move(outputOffsets)),
        outputs_(std::move(outputs)), qualityGates_(std::move(qualityGates)),
        objectiveProgram_(std::move(objectiveProgram)) {}

  std::vector<ResolvedDsePlanNode> nodes_;
  std::vector<std::uint64_t> outputOffsets_;
  std::vector<PlanValueDescriptor> outputs_;
  std::vector<QualityGatePolicy> qualityGates_;
  std::optional<ObjectiveProgram> objectiveProgram_;
};

class CompletedDsePlanExecution final {
public:
  CompletedDsePlanExecution(
      std::vector<std::uint64_t> outputOffsets,
      std::vector<std::vector<ArtifactRootReference>> outputs)
      : outputOffsets_(std::move(outputOffsets)), outputs_(std::move(outputs)) {
  }

  llvm::ArrayRef<ArtifactRootReference> resolve(PlanOutputRef output) const;

private:
  std::vector<std::uint64_t> outputOffsets_;
  std::vector<std::vector<ArtifactRootReference>> outputs_;
};

using DsePlanIncompleteReason =
    std::variant<CandidateGeneratorIncompleteReason,
                 PromotionAcquisitionIncompleteReason,
                 IncompleteSelectionReason>;

struct IncompleteDsePlanExecution final {
  std::uint64_t nodeOrdinal = 0;
  DsePlanIncompleteReason reason;
  CompletedDsePlanExecution completedPrefix;
  std::vector<std::vector<ArtifactRootReference>> retainedOutputs;
};

using DsePlanExecutionOutcome =
    std::variant<CompletedDsePlanExecution, IncompleteDsePlanExecution>;

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDseConfigView &view, const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_PLAN_H
