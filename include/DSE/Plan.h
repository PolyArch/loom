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

struct GeneratePlanNodeDefinition final {
  CandidateGeneratorDescriptorRef descriptor;
  std::vector<PlanInputBinding> inputBindings;
  std::vector<std::uint8_t> canonicalConfigBytes;
  ComponentViewDigest configDigest;
};

struct PromotePlanNodeDefinition final {
  PromotionAcquisitionDescriptorRef acquisition;
  std::vector<PlanInputBinding> inputBindings;
  std::vector<std::uint8_t> canonicalConfigBytes;
  ComponentViewDigest configDigest;
  QualityGatePolicy qualityGate;
  CandidateSelectionPolicy selection;
  ResolvedObjectiveCatalogs objectiveCatalogs;
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
    return acquisition_;
  }
  llvm::ArrayRef<PlanInputBinding> inputBindings() const {
    return inputBindings_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }
  const QualityGatePolicy &qualityGate() const { return qualityGate_; }
  const CandidateSelectionPolicy &selection() const { return selection_; }
  const ObjectiveProgram *objectiveProgram() const {
    return objectiveProgram_ ? &*objectiveProgram_ : nullptr;
  }

private:
  ResolvedPromotePlanNode(PromotionAcquisitionDescriptorRef acquisition,
                          std::vector<PlanInputBinding> inputBindings,
                          std::vector<std::uint8_t> canonicalConfigBytes,
                          ComponentViewDigest configDigest,
                          QualityGatePolicy qualityGate,
                          CandidateSelectionPolicy selection,
                          std::optional<ObjectiveProgram> objectiveProgram)
      : acquisition_(acquisition), inputBindings_(std::move(inputBindings)),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest), qualityGate_(std::move(qualityGate)),
        selection_(std::move(selection)),
        objectiveProgram_(std::move(objectiveProgram)) {}

  PromotionAcquisitionDescriptorRef acquisition_;
  std::vector<PlanInputBinding> inputBindings_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;
  QualityGatePolicy qualityGate_;
  CandidateSelectionPolicy selection_;
  std::optional<ObjectiveProgram> objectiveProgram_;

  friend class ResolvedDsePlan;
};

using ResolvedDsePlanNode =
    std::variant<ResolvedGeneratePlanNode, ResolvedPromotePlanNode>;

/// Immutable typed use-def resolution for one ordered Generate/Promote block.
class ResolvedDsePlan final {
public:
  static llvm::Expected<ResolvedDsePlan>
  get(std::vector<DsePlanNodeDefinition> nodes);

  llvm::ArrayRef<ResolvedDsePlanNode> nodes() const { return nodes_; }
  const PlanValueDescriptor *resolve(PlanOutputRef output) const;

private:
  ResolvedDsePlan(std::vector<ResolvedDsePlanNode> nodes,
                  std::vector<std::uint64_t> outputOffsets,
                  std::vector<PlanValueDescriptor> outputs)
      : nodes_(std::move(nodes)), outputOffsets_(std::move(outputOffsets)),
        outputs_(std::move(outputs)) {}

  std::vector<ResolvedDsePlanNode> nodes_;
  std::vector<std::uint64_t> outputOffsets_;
  std::vector<PlanValueDescriptor> outputs_;
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
executeDsePlan(const ResolvedDsePlan &plan, const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_PLAN_H
