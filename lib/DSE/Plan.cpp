#include "DSE/Plan.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_plan_invalid: " + message);
}

bool matchesSchema(const ArtifactRootReference &reference,
                   const ArtifactSchemaDescriptor &schema) {
  return reference.schemaIdentity == schema.identity &&
         reference.schemaVersion == schema.version;
}

bool validRole(PlanValueRole role) {
  return static_cast<std::uint32_t>(role) <=
         static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet);
}

bool validCardinality(PlanValueCardinality cardinality) {
  return static_cast<std::uint32_t>(cardinality) <=
         static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet);
}

llvm::Expected<ExactPlanArtifacts>
canonicalizeExactArtifacts(ExactPlanArtifacts value,
                           const PlanValueDescriptor &expected) {
  for (const ArtifactRootReference &artifact : value.artifacts)
    if (!matchesSchema(artifact, expected.schema))
      return invalid("static input artifact schema does not match its slot");
  llvm::sort(value.artifacts, artifactRootReferenceLess);
  value.artifacts.erase(
      std::unique(value.artifacts.begin(), value.artifacts.end()),
      value.artifacts.end());
  if (!planCardinalityContains(expected.cardinality, value.artifacts.size()))
    return invalid("static input violates its slot cardinality");
  return value;
}

llvm::Expected<PlanValueDescriptor>
inputDescriptor(const CandidateGeneratorInputSlotDescriptor &slot) {
  if (!slot.schema || !validRole(slot.role) ||
      !validCardinality(slot.cardinality))
    return invalid("candidate generator input slot is not plan-typed");
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality};
}

llvm::Expected<PlanValueDescriptor>
inputDescriptor(const PromotionAcquisitionInputSlotDescriptor &slot) {
  if (!slot.schema || !validRole(slot.role) ||
      !validCardinality(slot.cardinality))
    return invalid("promotion acquisition input slot is not plan-typed");
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality};
}

llvm::Expected<PlanValueDescriptor>
outputDescriptor(const CandidateGeneratorOutputSlotDescriptor &slot) {
  if (!slot.schema || !validRole(slot.role) ||
      !validCardinality(slot.cardinality))
    return invalid("candidate generator output slot is not plan-typed");
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality};
}

bool compatible(const PlanValueDescriptor &producer,
                const PlanValueDescriptor &consumer) {
  return producer.role == consumer.role && producer.schema == consumer.schema &&
         planCardinalityCanFlow(producer.cardinality, consumer.cardinality);
}

llvm::Error resolveInputBindings(std::vector<PlanInputBinding> &bindings,
                                 llvm::ArrayRef<PlanValueDescriptor> expected,
                                 std::size_t nodeIndex,
                                 llvm::ArrayRef<ResolvedDsePlanNode> nodes,
                                 llvm::ArrayRef<std::uint64_t> outputOffsets,
                                 llvm::ArrayRef<PlanValueDescriptor> outputs) {
  if (bindings.size() != expected.size())
    return invalid("plan node does not bind every input slot");
  for (std::size_t inputIndex = 0; inputIndex < bindings.size(); ++inputIndex) {
    PlanInputBinding &binding = bindings[inputIndex];
    if (auto *exact = std::get_if<ExactPlanArtifacts>(&binding)) {
      auto canonical =
          canonicalizeExactArtifacts(std::move(*exact), expected[inputIndex]);
      if (!canonical)
        return canonical.takeError();
      binding = std::move(*canonical);
      continue;
    }
    const PlanOutputRef output = std::get<PlanOutputRef>(binding);
    if (output.producerNodeOrdinal >= nodeIndex ||
        output.producerNodeOrdinal >= nodes.size())
      return invalid("produced input must reference an earlier node");
    const std::uint64_t begin = outputOffsets[output.producerNodeOrdinal];
    const std::uint64_t end = outputOffsets[output.producerNodeOrdinal + 1];
    if (output.outputSlotOrdinal >= end - begin)
      return invalid("produced input references an unknown output slot");
    if (!compatible(outputs[begin + output.outputSlotOrdinal],
                    expected[inputIndex]))
      return invalid("produced input role, artifact schema, or cardinality "
                     "does not match its slot");
  }
  return llvm::Error::success();
}

llvm::Error validateSelection(const CandidateSelectionPolicy &selection,
                              const ObjectiveProgram *program) {
  if (std::holds_alternative<AllPassingSelection>(selection))
    return llvm::Error::success();
  if (!program)
    return invalid("objective program is unavailable for ranked selection");
  if (const auto *topK = std::get_if<TopKSelection>(&selection)) {
    if (topK->k == 0)
      return invalid("TopK requires positive k");
    if (topK->totalOrdering >= program->totalOrderingCount())
      return invalid("TopK total ordering reference is out of range");
  } else {
    const auto &dimensions =
        std::get<ParetoSelection>(selection).objectiveDimensions;
    if (dimensions.empty() || !llvm::is_sorted(dimensions) ||
        std::adjacent_find(dimensions.begin(), dimensions.end()) !=
            dimensions.end())
      return invalid("Pareto dimensions are not a canonical nonempty set");
    if (dimensions.back() >= program->dimensionCount())
      return invalid("Pareto dimension reference is out of range");
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<ArtifactRootReference>>
publishEvidence(llvm::ArrayRef<PromotionEvidence> records,
                const ArtifactStore &store) {
  std::vector<ArtifactRootReference> references;
  references.reserve(records.size());
  for (const PromotionEvidence &record : records) {
    auto reference =
        evaluation::publishEvaluationEvidence(record.evidence, store);
    if (!reference)
      return reference.takeError();
    references.push_back(*reference);
  }
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
  return references;
}

llvm::Expected<std::vector<ArtifactRootReference>> resolveRuntimeInput(
    const PlanInputBinding &input, llvm::ArrayRef<std::uint64_t> outputOffsets,
    llvm::ArrayRef<std::vector<ArtifactRootReference>> outputs) {
  if (const auto *exact = std::get_if<ExactPlanArtifacts>(&input))
    return exact->artifacts;
  const PlanOutputRef output = std::get<PlanOutputRef>(input);
  if (output.producerNodeOrdinal >= outputOffsets.size() - 1)
    return invalid("resolved use-def references an unavailable output");
  const std::uint64_t begin = outputOffsets[output.producerNodeOrdinal];
  const std::uint64_t end = outputOffsets[output.producerNodeOrdinal + 1];
  if (output.outputSlotOrdinal >= end - begin)
    return invalid("resolved use-def references an unavailable slot");
  return outputs[begin + output.outputSlotOrdinal];
}

} // namespace

llvm::Expected<ResolvedDsePlan>
ResolvedDsePlan::get(llvm::ArrayRef<DsePlanNodeDefinition> definitions,
                     const ResolvedObjectiveCatalogs &objectiveCatalogs,
                     llvm::ArrayRef<QualityGatePolicy> qualityGates) {
  std::vector<ResolvedDsePlanNode> nodes;
  std::vector<std::uint64_t> outputOffsets;
  std::vector<PlanValueDescriptor> outputs;
  nodes.reserve(definitions.size());
  outputOffsets.reserve(definitions.size() + 1);
  outputOffsets.push_back(0);

  const bool needsObjectives =
      llvm::any_of(definitions, [](const DsePlanNodeDefinition &definition) {
        const auto *promote =
            std::get_if<PromotePlanNodeDefinition>(&definition);
        return promote &&
               !std::holds_alternative<AllPassingSelection>(promote->selection);
      });
  std::optional<ObjectiveProgram> objectiveProgram;
  if (needsObjectives) {
    auto compiled = ObjectiveProgram::get(objectiveCatalogs);
    if (!compiled)
      return compiled.takeError();
    objectiveProgram = std::move(*compiled);
  }

  for (std::size_t nodeIndex = 0; nodeIndex < definitions.size(); ++nodeIndex) {
    if (const auto *definition =
            std::get_if<GeneratePlanNodeDefinition>(&definitions[nodeIndex])) {
      const CandidateGeneratorDescriptor *descriptor =
          definition->descriptor.descriptor();
      if (!descriptor)
        return invalid("Generate node references an unregistered descriptor");
      if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
              definition->canonicalConfigBytes, definition->configDigest))
        return std::move(error);
      std::vector<PlanValueDescriptor> expected;
      expected.reserve(descriptor->inputSlots.size());
      for (const CandidateGeneratorInputSlotDescriptor &slot :
           descriptor->inputSlots) {
        auto value = inputDescriptor(slot);
        if (!value)
          return value.takeError();
        expected.push_back(*value);
      }
      std::vector<PlanInputBinding> inputBindings = definition->inputBindings;
      if (llvm::Error error =
              resolveInputBindings(inputBindings, expected, nodeIndex, nodes,
                                   outputOffsets, outputs))
        return std::move(error);
      if (descriptor->outputSlots.size() >
          std::numeric_limits<std::uint64_t>::max() - outputs.size())
        return invalid("plan output count overflows uint64");
      for (const CandidateGeneratorOutputSlotDescriptor &slot :
           descriptor->outputSlots) {
        auto value = outputDescriptor(slot);
        if (!value)
          return value.takeError();
        outputs.push_back(*value);
      }
      outputOffsets.push_back(outputs.size());
      nodes.emplace_back(ResolvedGeneratePlanNode(
          definition->descriptor, std::move(inputBindings),
          definition->canonicalConfigBytes, definition->configDigest));
      continue;
    }

    const auto &definition =
        std::get<PromotePlanNodeDefinition>(definitions[nodeIndex]);
    const PromotionAcquisitionDescriptor *descriptor =
        definition.acquisition.descriptor();
    if (!descriptor)
      return invalid("Promote node references an unregistered acquisition");
    if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
            definition.canonicalConfigBytes, definition.configDigest))
      return std::move(error);
    std::vector<PlanValueDescriptor> expected;
    expected.reserve(descriptor->inputSlots.size());
    for (const PromotionAcquisitionInputSlotDescriptor &slot :
         descriptor->inputSlots) {
      auto value = inputDescriptor(slot);
      if (!value)
        return value.takeError();
      expected.push_back(*value);
    }
    std::vector<PlanInputBinding> inputBindings = definition.inputBindings;
    if (llvm::Error error = resolveInputBindings(
            inputBindings, expected, nodeIndex, nodes, outputOffsets, outputs))
      return std::move(error);
    if (definition.qualityGate.ordinal() >= qualityGates.size())
      return invalid("Promote quality gate reference is out of range");
    if (static_cast<std::uint32_t>(definition.purpose) >
        static_cast<std::uint32_t>(PromotePurpose::ModelRelease))
      return invalid("Promote purpose is unknown");
    if (llvm::Error error =
            validateSelection(definition.selection,
                              objectiveProgram ? &*objectiveProgram : nullptr))
      return std::move(error);
    const PromotionAcquisitionInputSlotDescriptor *candidateSlot =
        descriptor->findInputSlot(descriptor->candidateInputSlot);
    if (!candidateSlot || !candidateSlot->schema)
      return invalid("Promote candidate input slot is unavailable");
    if (outputs.size() > std::numeric_limits<std::uint64_t>::max() - 2)
      return invalid("plan output count overflows uint64");
    outputs.push_back({PlanValueRole::CandidateSet, *candidateSlot->schema,
                       PlanValueCardinality::FiniteSet});
    outputs.push_back({PlanValueRole::EvidenceSet,
                       evaluation::EvaluationEvidence::artifactSchema,
                       PlanValueCardinality::FiniteSet});
    outputOffsets.push_back(outputs.size());
    nodes.emplace_back(ResolvedPromotePlanNode(
        definition.acquisition, std::move(inputBindings),
        definition.canonicalConfigBytes, definition.configDigest,
        definition.qualityGate, definition.selection, definition.purpose));
  }
  return ResolvedDsePlan(std::move(nodes), std::move(outputOffsets),
                         std::move(outputs), qualityGates.vec(),
                         std::move(objectiveProgram));
}

const PlanValueDescriptor *
ResolvedDsePlan::resolve(PlanOutputRef output) const {
  if (output.producerNodeOrdinal >= nodes_.size())
    return nullptr;
  const std::uint64_t begin = outputOffsets_[output.producerNodeOrdinal];
  const std::uint64_t end = outputOffsets_[output.producerNodeOrdinal + 1];
  if (output.outputSlotOrdinal >= end - begin)
    return nullptr;
  return &outputs_[begin + output.outputSlotOrdinal];
}

const QualityGatePolicy *
ResolvedDsePlan::resolve(QualityGatePolicyRef gate) const {
  if (gate.ordinal() >= qualityGates_.size())
    return nullptr;
  return &qualityGates_[gate.ordinal()];
}

llvm::ArrayRef<ArtifactRootReference>
CompletedDsePlanExecution::resolve(PlanOutputRef output) const {
  if (outputOffsets_.empty() ||
      output.producerNodeOrdinal >= outputOffsets_.size() - 1)
    return {};
  const std::uint64_t begin = outputOffsets_[output.producerNodeOrdinal];
  const std::uint64_t end = outputOffsets_[output.producerNodeOrdinal + 1];
  if (output.outputSlotOrdinal >= end - begin)
    return {};
  return outputs_[begin + output.outputSlotOrdinal];
}

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDsePlan &plan, const ArtifactStore &store) {
  std::vector<std::uint64_t> outputOffsets;
  std::vector<std::vector<ArtifactRootReference>> outputs;
  outputOffsets.reserve(plan.nodes().size() + 1);
  outputOffsets.push_back(0);

  for (std::size_t nodeIndex = 0; nodeIndex < plan.nodes().size();
       ++nodeIndex) {
    const ResolvedDsePlanNode &node = plan.nodes()[nodeIndex];
    if (const auto *generate = std::get_if<ResolvedGeneratePlanNode>(&node)) {
      std::vector<CandidateGeneratorInputBinding> inputs;
      inputs.reserve(generate->inputBindings().size());
      for (std::size_t index = 0; index < generate->inputBindings().size();
           ++index) {
        auto artifacts = resolveRuntimeInput(generate->inputBindings()[index],
                                             outputOffsets, outputs);
        if (!artifacts)
          return artifacts.takeError();
        inputs.push_back(
            {CandidateGeneratorInputSlotRef(static_cast<std::uint32_t>(index)),
             std::move(*artifacts)});
      }
      auto binding = ResolvedCandidateGeneratorBinding::get(
          generate->descriptorRef(), generate->canonicalConfigBytes(),
          generate->configDigest());
      if (!binding)
        return binding.takeError();
      auto outcome = invokeCandidateGenerator(inputs, *binding, store);
      if (!outcome)
        return outcome.takeError();
      if (auto *incomplete =
              std::get_if<IncompleteCandidateGeneratorInvocation>(&*outcome)) {
        std::vector<std::vector<ArtifactRootReference>> retained;
        retained.reserve(incomplete->retainedOutputBindings.size());
        for (CandidateGeneratorOutputBinding &output :
             incomplete->retainedOutputBindings)
          retained.push_back(std::move(output.artifacts));
        return DsePlanExecutionOutcome{IncompleteDsePlanExecution{
            static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
            CompletedDsePlanExecution(std::move(outputOffsets),
                                      std::move(outputs)),
            std::move(retained)}};
      }
      auto &completed =
          std::get<CompletedCandidateGeneratorInvocation>(*outcome);
      for (CandidateGeneratorOutputBinding &output : completed.outputBindings)
        outputs.push_back(std::move(output.artifacts));
      outputOffsets.push_back(outputs.size());
      continue;
    }

    const auto &promote = std::get<ResolvedPromotePlanNode>(node);
    const PromotionAcquisitionDescriptor *descriptor =
        promote.acquisitionRef().descriptor();
    if (!descriptor)
      return invalid("resolved Promote node lost its descriptor");
    std::vector<PromotionAcquisitionInputBinding> inputs;
    inputs.reserve(promote.inputBindings().size());
    for (std::size_t index = 0; index < promote.inputBindings().size();
         ++index) {
      auto artifacts = resolveRuntimeInput(promote.inputBindings()[index],
                                           outputOffsets, outputs);
      if (!artifacts)
        return artifacts.takeError();
      inputs.push_back(
          {PromotionAcquisitionInputSlotRef(static_cast<std::uint32_t>(index)),
           std::move(*artifacts)});
    }
    auto binding = ResolvedPromotionAcquisitionBinding::get(
        promote.acquisitionRef(), std::move(inputs),
        promote.canonicalConfigBytes(), promote.configDigest());
    if (!binding)
      return binding.takeError();
    auto acquisition = invokePromotionAcquisition(*binding, store);
    if (!acquisition)
      return acquisition.takeError();
    if (auto *incomplete =
            std::get_if<IncompletePromotionAcquisition>(&*acquisition)) {
      auto retained = publishEvidence(incomplete->retainedEvidence, store);
      if (!retained)
        return retained.takeError();
      return DsePlanExecutionOutcome{IncompleteDsePlanExecution{
          static_cast<std::uint64_t>(nodeIndex),
          incomplete->reason,
          CompletedDsePlanExecution(std::move(outputOffsets),
                                    std::move(outputs)),
          {{}, std::move(*retained)}}};
    }

    auto &completed = std::get<CompletedPromotionAcquisition>(*acquisition);
    const PromotionAcquisitionInputBinding *candidateInput =
        binding->findInputBinding(descriptor->candidateInputSlot);
    const PromotionAcquisitionInputSlotDescriptor *candidateSlot =
        descriptor->findInputSlot(descriptor->candidateInputSlot);
    if (!candidateInput || !candidateSlot || !candidateSlot->schema)
      return invalid("Promote candidate input is unavailable");
    auto candidateSet =
        CandidateSet::get(*candidateSlot->schema, candidateInput->artifacts);
    if (!candidateSet)
      return candidateSet.takeError();
    auto promotion = promoteCandidates(
        *candidateSet, descriptor->candidateRole, completed.evidence,
        *plan.resolve(promote.qualityGateRef()), promote.selection(),
        plan.objectiveProgram(), store);
    if (!promotion)
      return promotion.takeError();
    if (auto *incomplete = std::get_if<IncompleteSelection>(&*promotion))
      return DsePlanExecutionOutcome{IncompleteDsePlanExecution{
          static_cast<std::uint64_t>(nodeIndex),
          incomplete->reason,
          CompletedDsePlanExecution(std::move(outputOffsets),
                                    std::move(outputs)),
          {{}, std::move(incomplete->retainedEvidence)}}};
    if (auto *none = std::get_if<CompletedNoFeasibleCandidate>(&*promotion)) {
      outputs.push_back({});
      outputs.push_back(std::move(none->satisfiedEvidence));
    } else {
      auto &selected = std::get<CompletedSelection>(*promotion);
      outputs.push_back(std::move(selected.selected));
      outputs.push_back(std::move(selected.satisfiedEvidence));
    }
    outputOffsets.push_back(outputs.size());
  }
  return DsePlanExecutionOutcome{
      CompletedDsePlanExecution(std::move(outputOffsets), std::move(outputs))};
}

} // namespace loom::dse
