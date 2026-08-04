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

} // namespace

llvm::Expected<ResolvedGeneratePlan>
ResolvedGeneratePlan::get(std::vector<GeneratePlanNodeDefinition> definitions) {
  std::vector<ResolvedGeneratePlanNode> nodes;
  std::vector<std::uint64_t> outputOffsets;
  std::vector<PlanValueDescriptor> outputs;
  nodes.reserve(definitions.size());
  outputOffsets.reserve(definitions.size() + 1);
  outputOffsets.push_back(0);

  for (std::size_t nodeIndex = 0; nodeIndex < definitions.size(); ++nodeIndex) {
    GeneratePlanNodeDefinition &definition = definitions[nodeIndex];
    const CandidateGeneratorDescriptor *descriptor =
        definition.descriptor.descriptor();
    if (!descriptor)
      return invalid("Generate node references an unregistered descriptor");
    if (definition.inputBindings.size() != descriptor->inputSlots.size())
      return invalid("Generate node does not bind every input slot");
    if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
            definition.canonicalConfigBytes, definition.configDigest))
      return std::move(error);

    for (std::size_t inputIndex = 0;
         inputIndex < definition.inputBindings.size(); ++inputIndex) {
      auto expected = inputDescriptor(descriptor->inputSlots[inputIndex]);
      if (!expected)
        return expected.takeError();
      PlanInputBinding &binding = definition.inputBindings[inputIndex];
      if (auto *exact = std::get_if<ExactPlanArtifacts>(&binding)) {
        auto canonical =
            canonicalizeExactArtifacts(std::move(*exact), *expected);
        if (!canonical)
          return canonical.takeError();
        binding = std::move(*canonical);
        continue;
      }

      const PlanOutputRef output = std::get<PlanOutputRef>(binding);
      if (output.producerNodeOrdinal >= nodeIndex)
        return invalid("produced input must reference an earlier node");
      if (output.producerNodeOrdinal >= nodes.size())
        return invalid("produced input references an unknown node");
      const std::uint64_t begin = outputOffsets[output.producerNodeOrdinal];
      const std::uint64_t end = outputOffsets[output.producerNodeOrdinal + 1];
      if (output.outputSlotOrdinal >= end - begin)
        return invalid("produced input references an unknown output slot");
      const PlanValueDescriptor &produced =
          outputs[begin + output.outputSlotOrdinal];
      if (!compatible(produced, *expected))
        return invalid("produced input role, artifact schema, or cardinality "
                       "does not match its slot");
    }

    if (descriptor->outputSlots.size() >
        std::numeric_limits<std::uint64_t>::max() - outputs.size())
      return invalid("plan output count overflows uint64");
    for (const CandidateGeneratorOutputSlotDescriptor &slot :
         descriptor->outputSlots) {
      auto output = outputDescriptor(slot);
      if (!output)
        return output.takeError();
      outputs.push_back(std::move(*output));
    }
    outputOffsets.push_back(outputs.size());
    nodes.push_back(ResolvedGeneratePlanNode(
        definition.descriptor, std::move(definition.inputBindings),
        std::move(definition.canonicalConfigBytes), definition.configDigest));
  }

  return ResolvedGeneratePlan(std::move(nodes), std::move(outputOffsets),
                              std::move(outputs));
}

const PlanValueDescriptor *
ResolvedGeneratePlan::resolve(PlanOutputRef output) const {
  if (output.producerNodeOrdinal >= nodes_.size())
    return nullptr;
  const std::uint64_t begin = outputOffsets_[output.producerNodeOrdinal];
  const std::uint64_t end = outputOffsets_[output.producerNodeOrdinal + 1];
  if (output.outputSlotOrdinal >= end - begin)
    return nullptr;
  return &outputs_[begin + output.outputSlotOrdinal];
}

} // namespace loom::dse
