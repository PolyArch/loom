#ifndef LOOM_DSE_PLAN_H
#define LOOM_DSE_PLAN_H

#include "DSE/CandidateGenerator.h"
#include "DSE/PlanValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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

  friend class ResolvedGeneratePlan;
};

/// Immutable typed use-def resolution for an ordered Generate-node block.
class ResolvedGeneratePlan final {
public:
  static llvm::Expected<ResolvedGeneratePlan>
  get(std::vector<GeneratePlanNodeDefinition> nodes);

  llvm::ArrayRef<ResolvedGeneratePlanNode> nodes() const { return nodes_; }
  const PlanValueDescriptor *resolve(PlanOutputRef output) const;

private:
  ResolvedGeneratePlan(std::vector<ResolvedGeneratePlanNode> nodes,
                       std::vector<std::uint64_t> outputOffsets,
                       std::vector<PlanValueDescriptor> outputs)
      : nodes_(std::move(nodes)), outputOffsets_(std::move(outputOffsets)),
        outputs_(std::move(outputs)) {}

  std::vector<ResolvedGeneratePlanNode> nodes_;
  std::vector<std::uint64_t> outputOffsets_;
  std::vector<PlanValueDescriptor> outputs_;
};

class CompletedGeneratePlanExecution final {
public:
  CompletedGeneratePlanExecution(
      std::vector<std::uint64_t> outputOffsets,
      std::vector<std::vector<ArtifactRootReference>> outputs)
      : outputOffsets_(std::move(outputOffsets)), outputs_(std::move(outputs)) {
  }

  llvm::ArrayRef<ArtifactRootReference> resolve(PlanOutputRef output) const;

private:
  std::vector<std::uint64_t> outputOffsets_;
  std::vector<std::vector<ArtifactRootReference>> outputs_;
};

struct IncompleteGeneratePlanExecution final {
  std::uint64_t nodeOrdinal = 0;
  CandidateGeneratorIncompleteReason reason;
  CompletedGeneratePlanExecution completedPrefix;
  std::vector<CandidateGeneratorOutputBinding> retainedOutputBindings;
};

using GeneratePlanExecutionOutcome =
    std::variant<CompletedGeneratePlanExecution,
                 IncompleteGeneratePlanExecution>;

llvm::Expected<GeneratePlanExecutionOutcome>
executeGeneratePlan(const ResolvedGeneratePlan &plan,
                    const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_PLAN_H
