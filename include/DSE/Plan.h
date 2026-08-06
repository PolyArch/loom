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
class DsePlanExecutionBuilder;

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

/// Exact invocation-local record for one executed Generate plan node. The
/// enclosing plan outcome owns whether this record completed or is the node
/// at which execution became incomplete.
struct GenerateInvocationRecord final {
  std::uint64_t planNodeOrdinal;
  std::vector<CandidateGeneratorInputBinding> inputBindings;
  ResolvedCandidateGeneratorBinding generatorBinding;
  std::vector<CandidateGeneratorOutputBinding> outputBindings;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
};

/// Nonsemantic execution accounting paired to one GenerateInvocationRecord by
/// exact PlanNodeRef. It remains separate from the lineage record because it
/// cannot affect candidate identity or derivation.
struct GenerateInvocationWorkSummary final {
  std::uint64_t planNodeOrdinal;
  std::vector<CandidateGeneratorWorkUnitSummary> units;
};

class CompletedDsePlanExecution final {
public:
  CompletedDsePlanExecution(CompletedDsePlanExecution &&) = default;
  CompletedDsePlanExecution &operator=(CompletedDsePlanExecution &&) = default;
  CompletedDsePlanExecution(const CompletedDsePlanExecution &) = delete;
  CompletedDsePlanExecution &
  operator=(const CompletedDsePlanExecution &) = delete;

  bool hasOutput(PlanOutputRef output) const;
  llvm::ArrayRef<ArtifactRootReference> resolve(PlanOutputRef output) const;
  const ComponentViewDigest &resolvedDseConfigViewDigest() const {
    return resolvedDseConfigViewDigest_;
  }
  llvm::ArrayRef<GenerateInvocationRecord> generateInvocations() const {
    return generateInvocations_;
  }
  llvm::ArrayRef<GenerateInvocationWorkSummary> generateWorkSummaries() const {
    return generateWorkSummaries_;
  }

private:
  struct GenerateNodeOutputs final {
    std::size_t invocationOrdinal;
  };

  struct PromoteNodeOutputs final {
    std::vector<std::vector<ArtifactRootReference>> outputBindings;
  };

  using NodeOutputs = std::variant<GenerateNodeOutputs, PromoteNodeOutputs>;

  explicit CompletedDsePlanExecution(ComponentViewDigest digest)
      : resolvedDseConfigViewDigest_(digest) {}
  llvm::Error appendGenerate(GenerateInvocationRecord invocation,
                             GenerateInvocationWorkSummary workSummary);
  void
  appendPromote(std::vector<std::vector<ArtifactRootReference>> outputBindings);

  std::vector<NodeOutputs> nodeOutputs_;
  std::vector<GenerateInvocationRecord> generateInvocations_;
  std::vector<GenerateInvocationWorkSummary> generateWorkSummaries_;
  ComponentViewDigest resolvedDseConfigViewDigest_;

  friend class DsePlanExecutionBuilder;
};

using DsePlanIncompleteReason =
    std::variant<CandidateGeneratorIncompleteReason,
                 PromotionAcquisitionIncompleteReason,
                 IncompleteSelectionReason>;

llvm::StringRef toString(const DsePlanIncompleteReason &reason);

class IncompleteDsePlanExecution final {
public:
  IncompleteDsePlanExecution(IncompleteDsePlanExecution &&) = default;
  IncompleteDsePlanExecution &
  operator=(IncompleteDsePlanExecution &&) = default;
  IncompleteDsePlanExecution(const IncompleteDsePlanExecution &) = delete;
  IncompleteDsePlanExecution &
  operator=(const IncompleteDsePlanExecution &) = delete;

  std::uint64_t nodeOrdinal() const { return nodeOrdinal_; }
  const DsePlanIncompleteReason &reason() const { return reason_; }
  const CompletedDsePlanExecution &completedPrefix() const {
    return completedPrefix_;
  }
  std::size_t retainedOutputCount() const;
  llvm::ArrayRef<ArtifactRootReference>
  retainedOutput(std::size_t outputSlotOrdinal) const;
  const GenerateInvocationRecord *incompleteGenerateInvocation() const;
  const GenerateInvocationWorkSummary *incompleteGenerateWorkSummary() const;

private:
  struct PromoteRetainedOutputs final {
    std::vector<std::vector<ArtifactRootReference>> outputBindings;
  };

  struct IncompleteGenerateNode final {
    GenerateInvocationRecord invocation;
    GenerateInvocationWorkSummary workSummary;
  };

  using IncompleteNode =
      std::variant<IncompleteGenerateNode, PromoteRetainedOutputs>;

  IncompleteDsePlanExecution(std::uint64_t nodeOrdinal,
                             DsePlanIncompleteReason reason,
                             CompletedDsePlanExecution completedPrefix,
                             IncompleteNode incompleteNode)
      : nodeOrdinal_(nodeOrdinal), reason_(std::move(reason)),
        completedPrefix_(std::move(completedPrefix)),
        incompleteNode_(std::move(incompleteNode)) {}

  std::uint64_t nodeOrdinal_ = 0;
  DsePlanIncompleteReason reason_;
  CompletedDsePlanExecution completedPrefix_;
  IncompleteNode incompleteNode_;

  friend class DsePlanExecutionBuilder;
};

using DsePlanExecutionOutcome =
    std::variant<CompletedDsePlanExecution, IncompleteDsePlanExecution>;

/// Invocation-local transfer projection for InvocationManifest.
/// The component-view digest scopes every PlanNodeRef to one exact resolved
/// plan. This record is not an Artifact or a candidate identity.
class DsePlanGenerateInvocationRecords final {
public:
  const ComponentViewDigest &resolvedDseConfigViewDigest() const {
    return resolvedDseConfigViewDigest_;
  }
  llvm::ArrayRef<GenerateInvocationRecord> completed() const {
    return completed_;
  }
  llvm::ArrayRef<GenerateInvocationWorkSummary> completedWorkSummaries() const {
    return completedWorkSummaries_;
  }
  const std::optional<GenerateInvocationRecord> &incomplete() const {
    return incomplete_;
  }
  const std::optional<GenerateInvocationWorkSummary> &
  incompleteWorkSummary() const {
    return incompleteWorkSummary_;
  }

private:
  DsePlanGenerateInvocationRecords(
      ComponentViewDigest resolvedDseConfigViewDigest,
      std::vector<GenerateInvocationRecord> completed,
      std::vector<GenerateInvocationWorkSummary> completedWorkSummaries,
      std::optional<GenerateInvocationRecord> incomplete,
      std::optional<GenerateInvocationWorkSummary> incompleteWorkSummary)
      : resolvedDseConfigViewDigest_(resolvedDseConfigViewDigest),
        completed_(std::move(completed)),
        completedWorkSummaries_(std::move(completedWorkSummaries)),
        incomplete_(std::move(incomplete)),
        incompleteWorkSummary_(std::move(incompleteWorkSummary)) {}

  ComponentViewDigest resolvedDseConfigViewDigest_;
  std::vector<GenerateInvocationRecord> completed_;
  std::vector<GenerateInvocationWorkSummary> completedWorkSummaries_;
  std::optional<GenerateInvocationRecord> incomplete_;
  std::optional<GenerateInvocationWorkSummary> incompleteWorkSummary_;

  friend class DsePlanExecutionBuilder;
};

DsePlanGenerateInvocationRecords
takeDsePlanGenerateInvocationRecords(DsePlanExecutionOutcome outcome);

/// Derived diagnostics produced while strictly consuming immutable Generate
/// invocation records. The records and referenced Artifacts remain the only
/// semantic authority.
struct DsePlanGenerateInvocationSummary final {
  std::uint64_t planExecutions = 0;
  std::uint64_t completedInvocations = 0;
  std::uint64_t incompleteInvocations = 0;
  std::uint64_t inputBindings = 0;
  std::uint64_t inputArtifacts = 0;
  std::uint64_t outputBindings = 0;
  std::uint64_t outputArtifacts = 0;
  std::uint64_t lineageEdges = 0;
  std::uint64_t workUnitSummaries = 0;
  std::uint64_t plannedWorkSlots = 0;
  std::uint64_t consumedWorkSlots = 0;
};

llvm::Expected<DsePlanGenerateInvocationSummary>
validateAndSummarizeDsePlanGenerateInvocations(
    llvm::ArrayRef<DsePlanGenerateInvocationRecords> records,
    const ArtifactStore &store);

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDseConfigView &view, const ArtifactStore &store,
               const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_PLAN_H
