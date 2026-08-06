#include "DSE/Plan.h"

#include "Common/ArtifactLocalReference.h"
#include "DSE/ResolvedConfigView.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <type_traits>
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

void collectGateObligations(const QualityGatePolicy &gate,
                            std::vector<std::uint32_t> &result) {
  for (const QualityGateClause &clause : gate.clauses()) {
    for (const QualityGateAtom &atom : clause.atoms) {
      if (const auto *metric = std::get_if<MetricGate>(&atom))
        result.push_back(metric->evidenceObligationTemplate);
      else
        result.push_back(
            std::get<FindingGate>(atom).evidenceObligationTemplate);
    }
  }
}

void collectDimensionObligation(const ResolvedObjectiveCatalogs &catalogs,
                                std::uint32_t dimension,
                                std::vector<std::uint32_t> &result) {
  if (dimension >= catalogs.dimensions.size())
    return;
  if (const auto *metric = std::get_if<ResolvedEvaluationMetricObjectiveSource>(
          &catalogs.dimensions[dimension].source))
    result.push_back(metric->evidenceObligationTemplate);
}

void collectSelectionObligations(const CandidateSelectionPolicy &selection,
                                 const ResolvedObjectiveCatalogs &catalogs,
                                 std::vector<std::uint32_t> &result) {
  if (const auto *topK = std::get_if<TopKSelection>(&selection)) {
    if (topK->totalOrdering >= catalogs.totalOrderings.size())
      return;
    for (std::uint32_t level :
         catalogs.totalOrderings[topK->totalOrdering].weightedLevels) {
      if (level >= catalogs.weightedLevels.size())
        continue;
      for (const ResolvedWeightedObjectiveTerm &term :
           catalogs.weightedLevels[level].terms)
        collectDimensionObligation(catalogs, term.dimension, result);
    }
    return;
  }
  if (const auto *pareto = std::get_if<ParetoSelection>(&selection))
    for (std::uint32_t dimension : pareto->objectiveDimensions)
      collectDimensionObligation(catalogs, dimension, result);
}

bool acceptsSchema(const evaluation::CaseSubjectRoleDescriptor &role,
                   const ArtifactSchemaDescriptor &schema) {
  return llvm::any_of(role.acceptedSchemas,
                      [&](const ArtifactSchemaDescriptor *accepted) {
                        return accepted && *accepted == schema;
                      });
}

llvm::Error validateObligationRole(
    const evaluation::EvaluationCaseSignatureDescriptor &signature,
    evaluation::CaseSubjectRoleRef roleRef,
    const PromotionAcquisitionInputSlotDescriptor &slot) {
  const evaluation::CaseSubjectRoleDescriptor *role =
      signature.findSubjectRole(roleRef);
  if (!role)
    return invalid("Evidence obligation references a foreign case role");
  if (!slot.schema || !acceptsSchema(*role, *slot.schema))
    return invalid("Evidence obligation role rejects its acquisition slot "
                   "schema");
  const PlanValueCardinality required =
      role->cardinality == evaluation::SubjectRoleCardinality::ExactlyOne
          ? PlanValueCardinality::ExactlyOne
          : PlanValueCardinality::NonEmptySet;
  if (planCardinalityBounds(slot.cardinality).minimum <
      planCardinalityBounds(required).minimum)
    return invalid("Evidence obligation role and acquisition slot cardinality "
                   "do not match");
  return llvm::Error::success();
}

llvm::Error validateAcquisitionObligations(
    const ResolvedPromotionAcquisitionBinding &binding,
    const PromotionAcquisitionDescriptor &descriptor,
    llvm::ArrayRef<EvidenceObligationTemplate> templates,
    const QualityGatePolicy &gate, const CandidateSelectionPolicy &selection,
    const ResolvedObjectiveCatalogs &catalogs) {
  std::vector<std::uint32_t> required;
  collectGateObligations(gate, required);
  collectSelectionObligations(selection, catalogs, required);
  llvm::sort(required);
  required.erase(std::unique(required.begin(), required.end()), required.end());

  const auto selected = binding.evidenceObligations();
  for (std::uint32_t ordinal : required) {
    if (!llvm::binary_search(selected, EvidenceObligationTemplateRef(ordinal),
                             [](EvidenceObligationTemplateRef lhs,
                                EvidenceObligationTemplateRef rhs) {
                               return lhs.ordinal() < rhs.ordinal();
                             }))
      return invalid("acquisition policy omits a required Evidence obligation");
  }

  const PromotionAcquisitionInputSlotDescriptor *candidateSlot =
      descriptor.findInputSlot(descriptor.candidateInputSlot);
  if (!candidateSlot || !candidateSlot->schema)
    return invalid("Promote candidate input slot is unavailable");

  for (EvidenceObligationTemplateRef ref : selected) {
    if (ref.ordinal() >= templates.size())
      return invalid("acquisition references a foreign Evidence obligation");
    const EvidenceObligationTemplate &obligation = templates[ref.ordinal()];
    if (obligation.candidateRole() != descriptor.candidateRole)
      return invalid("acquisition and Evidence obligation candidate roles "
                     "do not match");
    const evaluation::EvaluationModelDescriptor *model =
        obligation.modelBinding().descriptorRef().descriptor();
    const evaluation::EvaluationCaseSignatureDescriptor *signature =
        model ? model->caseSignature.descriptor() : nullptr;
    if (!signature)
      return invalid("Evidence obligation has no registered case signature");
    const evaluation::CaseSubjectRoleDescriptor *candidateRole =
        signature->findSubjectRole(descriptor.candidateRole);
    if (!candidateRole ||
        !acceptsSchema(*candidateRole, *candidateSlot->schema))
      return invalid("Evidence obligation candidate role rejects the candidate "
                     "slot schema");

    for (const InputSubjectBinding &subject :
         obligation.inputSubjectBindings()) {
      if (subject.inputSlot.ordinal() >= descriptor.inputSlots.size())
        return invalid("Evidence obligation references an unavailable "
                       "acquisition slot");
      if (llvm::Error error = validateObligationRole(
              *signature, subject.role,
              descriptor.inputSlots[subject.inputSlot.ordinal()]))
        return error;
    }
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

void appendReferences(std::vector<ArtifactRootReference> &destination,
                      llvm::ArrayRef<ArtifactRootReference> additional) {
  destination.insert(destination.end(), additional.begin(), additional.end());
}

void canonicalizeReferences(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

struct CompletedStagedTopK final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> evidence;
};

struct IncompleteStagedTopK final {
  DsePlanIncompleteReason reason;
  std::vector<ArtifactRootReference> evidence;
};

using StagedTopKOutcome =
    std::variant<CompletedStagedTopK, IncompleteStagedTopK>;

llvm::Expected<StagedTopKOutcome> executeStagedTopK(
    const ResolvedPromotePlanNode &promote,
    const PromotionAcquisitionDescriptor &descriptor,
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputs,
    const CandidateSet &candidateSet,
    llvm::ArrayRef<EvidenceObligationTemplate> evidenceObligations,
    const QualityGatePolicy &qualityGate, const ObjectiveProgram &objectives,
    const TopKSelection &selection, const ArtifactStore &store,
    const BlobStore &blobs) {
  std::vector<std::uint32_t> objectiveOrdinals;
  objectiveOrdinals.reserve(promote.objectiveObligations().size());
  for (EvidenceObligationTemplateRef ref : promote.objectiveObligations())
    objectiveOrdinals.push_back(ref.ordinal());

  auto objectiveAcquisition = invokePromotionAcquisition(
      inputs, promote.acquisitionBinding(), evidenceObligations,
      {candidateSet.candidates(), promote.objectiveObligations()}, store,
      blobs);
  if (!objectiveAcquisition)
    return objectiveAcquisition.takeError();
  if (auto *incomplete =
          std::get_if<IncompletePromotionAcquisition>(&*objectiveAcquisition)) {
    auto retained = publishEvidence(incomplete->retainedEvidence, store);
    if (!retained)
      return retained.takeError();
    return StagedTopKOutcome{IncompleteStagedTopK{
        DsePlanIncompleteReason{incomplete->reason}, std::move(*retained)}};
  }
  std::vector<PromotionEvidence> objectiveEvidence = std::move(
      std::get<CompletedPromotionAcquisition>(*objectiveAcquisition).evidence);
  auto ranking = rankCandidatesByObjective(
      candidateSet, descriptor.candidateRole, objectiveEvidence,
      objectiveOrdinals, selection.totalOrdering, objectives, store);
  if (!ranking)
    return ranking.takeError();
  if (auto *incomplete = std::get_if<IncompleteSelection>(&*ranking))
    return StagedTopKOutcome{
        IncompleteStagedTopK{DsePlanIncompleteReason{incomplete->reason},
                             std::move(incomplete->retainedEvidence)}};

  auto &ranked = std::get<CompletedCandidateObjectiveRanking>(*ranking);
  std::vector<ArtifactRootReference> retainedEvidence =
      std::move(ranked.retainedEvidence);
  const auto obligationLess = [](EvidenceObligationTemplateRef lhs,
                                 EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  };
  std::vector<EvidenceObligationTemplateRef> deferredObligations;
  std::set_difference(
      promote.acquisitionBinding().evidenceObligations().begin(),
      promote.acquisitionBinding().evidenceObligations().end(),
      promote.objectiveObligations().begin(),
      promote.objectiveObligations().end(),
      std::back_inserter(deferredObligations), obligationLess);

  std::vector<ArtifactRootReference> selected;
  selected.reserve(static_cast<std::size_t>(
      std::min<std::uint64_t>(selection.k, ranked.rankedCandidates.size())));
  const std::size_t objectiveCount = promote.objectiveObligations().size();
  if (objectiveCount != 0 &&
      objectiveEvidence.size() !=
          candidateSet.candidates().size() * objectiveCount)
    return invalid("objective acquisition lost its positional task shape");

  std::size_t cursor = 0;
  while (cursor < ranked.rankedCandidates.size() &&
         selected.size() < selection.k) {
    const std::uint64_t missing = selection.k - selected.size();
    const std::size_t batchSize =
        static_cast<std::size_t>(std::min<std::uint64_t>(
            missing, ranked.rankedCandidates.size() - cursor));
    llvm::ArrayRef<ArtifactRootReference> rankedBatch(
        ranked.rankedCandidates.data() + cursor, batchSize);
    std::vector<ArtifactRootReference> candidateDomain(rankedBatch.begin(),
                                                       rankedBatch.end());
    llvm::sort(candidateDomain, artifactRootReferenceLess);
    PromotionAcquisitionOutcome deferred = CompletedPromotionAcquisition{{}};
    if (!deferredObligations.empty()) {
      auto acquired = invokePromotionAcquisition(
          inputs, promote.acquisitionBinding(), evidenceObligations,
          {candidateDomain, deferredObligations}, store, blobs);
      if (!acquired)
        return acquired.takeError();
      deferred = std::move(*acquired);
    }
    if (auto *incomplete =
            std::get_if<IncompletePromotionAcquisition>(&deferred)) {
      auto references = publishEvidence(incomplete->retainedEvidence, store);
      if (!references)
        return references.takeError();
      appendReferences(retainedEvidence, *references);
      canonicalizeReferences(retainedEvidence);
      return StagedTopKOutcome{
          IncompleteStagedTopK{DsePlanIncompleteReason{incomplete->reason},
                               std::move(retainedEvidence)}};
    }

    auto &completedDeferred = std::get<CompletedPromotionAcquisition>(deferred);
    std::map<ArtifactRootReference, std::vector<const PromotionEvidence *>,
             decltype(&artifactRootReferenceLess)>
        deferredByCandidate(&artifactRootReferenceLess);
    for (const PromotionEvidence &record : completedDeferred.evidence) {
      const llvm::ArrayRef<ArtifactRootReference> subjects =
          record.request.subjectBindings().subjects(descriptor.candidateRole);
      if (subjects.size() != 1)
        return invalid("deferred Evidence lost its candidate association");
      deferredByCandidate[subjects.front()].push_back(&record);
    }

    for (const ArtifactRootReference &candidate : rankedBatch) {
      const auto canonicalCandidate = llvm::lower_bound(
          candidateSet.candidates(), candidate, artifactRootReferenceLess);
      if (canonicalCandidate == candidateSet.candidates().end() ||
          *canonicalCandidate != candidate)
        return invalid("ranked candidate is outside the canonical input set");
      const std::size_t candidateOrdinal = static_cast<std::size_t>(
          canonicalCandidate - candidateSet.candidates().begin());
      std::vector<PromotionEvidence> candidateEvidence;
      candidateEvidence.reserve(
          promote.acquisitionBinding().evidenceObligations().size());
      const std::size_t objectiveBegin = candidateOrdinal * objectiveCount;
      for (std::size_t index = 0; index < objectiveCount; ++index)
        candidateEvidence.push_back(objectiveEvidence[objectiveBegin + index]);
      const auto deferredRecords = deferredByCandidate.find(candidate);
      if (deferredRecords != deferredByCandidate.end())
        for (const PromotionEvidence *record : deferredRecords->second)
          candidateEvidence.push_back(*record);

      const std::array<ArtifactRootReference, 1> singletonDomain = {candidate};
      auto singleton =
          CandidateSet::get(candidateSet.schema(), singletonDomain);
      if (!singleton)
        return singleton.takeError();
      auto gate = promoteCandidates(*singleton, descriptor.candidateRole,
                                    candidateEvidence, qualityGate,
                                    AllPassingSelection{}, nullptr, store);
      if (!gate)
        return gate.takeError();
      if (auto *incomplete = std::get_if<IncompleteSelection>(&*gate)) {
        appendReferences(retainedEvidence, incomplete->retainedEvidence);
        canonicalizeReferences(retainedEvidence);
        return StagedTopKOutcome{
            IncompleteStagedTopK{DsePlanIncompleteReason{incomplete->reason},
                                 std::move(retainedEvidence)}};
      }
      if (auto *passed = std::get_if<CompletedSelection>(&*gate)) {
        selected.push_back(candidate);
        appendReferences(retainedEvidence, passed->satisfiedEvidence);
      } else {
        appendReferences(
            retainedEvidence,
            std::get<CompletedNoFeasibleCandidate>(*gate).satisfiedEvidence);
      }
    }
    cursor += batchSize;
  }

  llvm::sort(selected, artifactRootReferenceLess);
  canonicalizeReferences(retainedEvidence);
  return StagedTopKOutcome{
      CompletedStagedTopK{std::move(selected), std::move(retainedEvidence)}};
}

llvm::Expected<std::vector<ArtifactRootReference>>
resolveRuntimeInput(const PlanInputBinding &input,
                    const CompletedDsePlanExecution &completed) {
  if (const auto *exact = std::get_if<ExactPlanArtifacts>(&input))
    return exact->artifacts;
  const PlanOutputRef output = std::get<PlanOutputRef>(input);
  if (!completed.hasOutput(output))
    return invalid("resolved use-def references an unavailable output");
  return completed.resolve(output).vec();
}

} // namespace

class DsePlanExecutionBuilder final {
public:
  static CompletedDsePlanExecution createCompleted(ComponentViewDigest digest) {
    return CompletedDsePlanExecution(digest);
  }

  static llvm::Error appendGenerate(CompletedDsePlanExecution &completed,
                                    GenerateInvocationRecord invocation,
                                    GenerateInvocationWorkSummary workSummary) {
    return completed.appendGenerate(std::move(invocation),
                                    std::move(workSummary));
  }

  static void appendPromote(
      CompletedDsePlanExecution &completed,
      std::vector<std::vector<ArtifactRootReference>> outputBindings) {
    completed.appendPromote(std::move(outputBindings));
  }

  static IncompleteDsePlanExecution
  incompleteGenerate(std::uint64_t nodeOrdinal, DsePlanIncompleteReason reason,
                     CompletedDsePlanExecution completedPrefix,
                     GenerateInvocationRecord invocation,
                     GenerateInvocationWorkSummary workSummary) {
    return IncompleteDsePlanExecution(
        nodeOrdinal, std::move(reason), std::move(completedPrefix),
        IncompleteDsePlanExecution::IncompleteNode(
            IncompleteDsePlanExecution::IncompleteGenerateNode{
                std::move(invocation), std::move(workSummary)}));
  }

  static IncompleteDsePlanExecution incompletePromote(
      std::uint64_t nodeOrdinal, DsePlanIncompleteReason reason,
      CompletedDsePlanExecution completedPrefix,
      std::vector<std::vector<ArtifactRootReference>> outputBindings) {
    return IncompleteDsePlanExecution(
        nodeOrdinal, std::move(reason), std::move(completedPrefix),
        IncompleteDsePlanExecution::IncompleteNode(
            IncompleteDsePlanExecution::PromoteRetainedOutputs{
                std::move(outputBindings)}));
  }

  static DsePlanGenerateInvocationRecords
  takeGenerateInvocationRecords(DsePlanExecutionOutcome outcome) {
    if (auto *completed = std::get_if<CompletedDsePlanExecution>(&outcome))
      return DsePlanGenerateInvocationRecords{
          completed->resolvedDseConfigViewDigest_,
          std::move(completed->generateInvocations_),
          std::move(completed->generateWorkSummaries_), std::nullopt,
          std::nullopt};
    auto &incomplete = std::get<IncompleteDsePlanExecution>(outcome);
    std::optional<GenerateInvocationRecord> stopped;
    std::optional<GenerateInvocationWorkSummary> stoppedWork;
    if (auto *node =
            std::get_if<IncompleteDsePlanExecution::IncompleteGenerateNode>(
                &incomplete.incompleteNode_)) {
      stopped.emplace(std::move(node->invocation));
      stoppedWork.emplace(std::move(node->workSummary));
    }
    return DsePlanGenerateInvocationRecords{
        incomplete.completedPrefix_.resolvedDseConfigViewDigest_,
        std::move(incomplete.completedPrefix_.generateInvocations_),
        std::move(incomplete.completedPrefix_.generateWorkSummaries_),
        std::move(stopped), std::move(stoppedWork)};
  }
};

llvm::Expected<ResolvedDsePlan> ResolvedDsePlan::get(
    llvm::ArrayRef<DsePlanNodeDefinition> definitions,
    llvm::ArrayRef<EvidenceObligationTemplate> evidenceObligationTemplates,
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
    auto acquisitionBinding = ResolvedPromotionAcquisitionBinding::get(
        definition.acquisition, definition.canonicalConfigBytes,
        definition.configDigest);
    if (!acquisitionBinding)
      return acquisitionBinding.takeError();
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
    if (llvm::Error error = validateAcquisitionObligations(
            *acquisitionBinding, *descriptor, evidenceObligationTemplates,
            qualityGates[definition.qualityGate.ordinal()],
            definition.selection, objectiveCatalogs))
      return std::move(error);
    std::vector<std::uint32_t> objectiveOrdinals;
    collectSelectionObligations(definition.selection, objectiveCatalogs,
                                objectiveOrdinals);
    llvm::sort(objectiveOrdinals);
    objectiveOrdinals.erase(
        std::unique(objectiveOrdinals.begin(), objectiveOrdinals.end()),
        objectiveOrdinals.end());
    std::vector<EvidenceObligationTemplateRef> objectiveObligations;
    objectiveObligations.reserve(objectiveOrdinals.size());
    for (std::uint32_t ordinal : objectiveOrdinals)
      objectiveObligations.emplace_back(ordinal);
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
        std::move(*acquisitionBinding), std::move(inputBindings),
        definition.qualityGate, definition.selection,
        std::move(objectiveObligations), definition.purpose));
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
  if (!hasOutput(output))
    return {};
  const NodeOutputs &node = nodeOutputs_[output.producerNodeOrdinal];
  if (const auto *generate = std::get_if<GenerateNodeOutputs>(&node))
    return generateInvocations_[generate->invocationOrdinal]
        .outputBindings[output.outputSlotOrdinal]
        .artifacts;
  return std::get<PromoteNodeOutputs>(node)
      .outputBindings[output.outputSlotOrdinal];
}

bool CompletedDsePlanExecution::hasOutput(PlanOutputRef output) const {
  if (output.producerNodeOrdinal >= nodeOutputs_.size())
    return false;
  const NodeOutputs &node = nodeOutputs_[output.producerNodeOrdinal];
  if (const auto *generate = std::get_if<GenerateNodeOutputs>(&node)) {
    if (generate->invocationOrdinal >= generateInvocations_.size())
      return false;
    return output.outputSlotOrdinal <
           generateInvocations_[generate->invocationOrdinal]
               .outputBindings.size();
  }
  return output.outputSlotOrdinal <
         std::get<PromoteNodeOutputs>(node).outputBindings.size();
}

llvm::Error CompletedDsePlanExecution::appendGenerate(
    GenerateInvocationRecord invocation,
    GenerateInvocationWorkSummary workSummary) {
  if (invocation.planNodeOrdinal != nodeOutputs_.size())
    return invalid("Generate invocation ordinal does not follow plan order");
  if (workSummary.planNodeOrdinal != invocation.planNodeOrdinal)
    return invalid("Generate work summary names a different plan node");
  if (llvm::Error error = validateCandidateGeneratorWorkSummary(
          invocation.generatorBinding.descriptorRef(), workSummary.units))
    return error;
  const std::size_t invocationOrdinal = generateInvocations_.size();
  generateInvocations_.push_back(std::move(invocation));
  generateWorkSummaries_.push_back(std::move(workSummary));
  nodeOutputs_.push_back(GenerateNodeOutputs{invocationOrdinal});
  return llvm::Error::success();
}

void CompletedDsePlanExecution::appendPromote(
    std::vector<std::vector<ArtifactRootReference>> outputBindings) {
  nodeOutputs_.push_back(PromoteNodeOutputs{std::move(outputBindings)});
}

std::size_t IncompleteDsePlanExecution::retainedOutputCount() const {
  if (const auto *generate =
          std::get_if<IncompleteGenerateNode>(&incompleteNode_))
    return generate->invocation.outputBindings.size();
  return std::get<PromoteRetainedOutputs>(incompleteNode_)
      .outputBindings.size();
}

llvm::ArrayRef<ArtifactRootReference>
IncompleteDsePlanExecution::retainedOutput(
    std::size_t outputSlotOrdinal) const {
  if (const auto *generate =
          std::get_if<IncompleteGenerateNode>(&incompleteNode_)) {
    if (outputSlotOrdinal >= generate->invocation.outputBindings.size())
      return {};
    return generate->invocation.outputBindings[outputSlotOrdinal].artifacts;
  }
  const auto &outputs =
      std::get<PromoteRetainedOutputs>(incompleteNode_).outputBindings;
  if (outputSlotOrdinal >= outputs.size())
    return {};
  return outputs[outputSlotOrdinal];
}

const GenerateInvocationRecord *
IncompleteDsePlanExecution::incompleteGenerateInvocation() const {
  const auto *generate = std::get_if<IncompleteGenerateNode>(&incompleteNode_);
  return generate ? &generate->invocation : nullptr;
}

const GenerateInvocationWorkSummary *
IncompleteDsePlanExecution::incompleteGenerateWorkSummary() const {
  const auto *generate = std::get_if<IncompleteGenerateNode>(&incompleteNode_);
  return generate ? &generate->workSummary : nullptr;
}

DsePlanGenerateInvocationRecords
takeDsePlanGenerateInvocationRecords(DsePlanExecutionOutcome outcome) {
  return DsePlanExecutionBuilder::takeGenerateInvocationRecords(
      std::move(outcome));
}

llvm::Expected<DsePlanGenerateInvocationSummary>
validateAndSummarizeDsePlanGenerateInvocations(
    llvm::ArrayRef<DsePlanGenerateInvocationRecords> records,
    const ArtifactStore &store) {
  DsePlanGenerateInvocationSummary summary;
  auto add = [](std::uint64_t &total, std::size_t amount,
                llvm::StringRef field) -> llvm::Error {
    if (amount > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("Generate invocation " + field + " count overflows u64");
    total += static_cast<std::uint64_t>(amount);
    return llvm::Error::success();
  };
  auto addU64 = [](std::uint64_t &total, std::uint64_t amount,
                   llvm::StringRef field) -> llvm::Error {
    if (amount > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("Generate invocation " + field + " count overflows u64");
    total += amount;
    return llvm::Error::success();
  };
  auto consume = [&](const GenerateInvocationRecord &invocation,
                     const GenerateInvocationWorkSummary &workSummary,
                     bool completed) -> llvm::Error {
    if (workSummary.planNodeOrdinal != invocation.planNodeOrdinal)
      return invalid("Generate work summary names a different plan node");
    if (llvm::Error error = validateCandidateGeneratorWorkSummary(
            invocation.generatorBinding.descriptorRef(), workSummary.units))
      return error;
    if (llvm::Error error = validateCanonicalCandidateGeneratorInvocation(
            invocation.inputBindings, invocation.generatorBinding,
            invocation.outputBindings, invocation.lineageEdges, completed,
            store))
      return error;
    if (llvm::Error error =
            add(summary.inputBindings, invocation.inputBindings.size(),
                "input binding"))
      return error;
    for (const CandidateGeneratorInputBinding &input : invocation.inputBindings)
      if (llvm::Error error = add(summary.inputArtifacts,
                                  input.artifacts.size(), "input artifact"))
        return error;
    if (llvm::Error error =
            add(summary.outputBindings, invocation.outputBindings.size(),
                "output binding"))
      return error;
    for (const CandidateGeneratorOutputBinding &output :
         invocation.outputBindings)
      if (llvm::Error error = add(summary.outputArtifacts,
                                  output.artifacts.size(), "output artifact"))
        return error;
    if (llvm::Error error = add(summary.lineageEdges,
                                invocation.lineageEdges.size(), "lineage edge"))
      return error;
    if (llvm::Error error = add(summary.workUnitSummaries,
                                workSummary.units.size(), "work-unit summary"))
      return error;
    for (const CandidateGeneratorWorkUnitSummary &unit : workSummary.units) {
      if (llvm::Error error = addU64(summary.plannedWorkSlots, unit.planned,
                                     "planned work slot"))
        return error;
      if (llvm::Error error = addU64(summary.consumedWorkSlots, unit.consumed,
                                     "consumed work slot"))
        return error;
    }
    return llvm::Error::success();
  };

  for (const DsePlanGenerateInvocationRecords &planRecords : records) {
    if (llvm::Error error = add(summary.planExecutions, 1, "plan execution"))
      return std::move(error);
    if (planRecords.completed().size() !=
        planRecords.completedWorkSummaries().size())
      return invalid("completed Generate records and work summaries differ "
                     "in width");
    std::optional<std::uint64_t> previousNode;
    for (std::size_t ordinal = 0; ordinal != planRecords.completed().size();
         ++ordinal) {
      const GenerateInvocationRecord &invocation =
          planRecords.completed()[ordinal];
      const GenerateInvocationWorkSummary &workSummary =
          planRecords.completedWorkSummaries()[ordinal];
      if (previousNode && invocation.planNodeOrdinal <= *previousNode)
        return invalid("completed Generate invocation ordinals are not "
                       "strictly increasing");
      if (llvm::Error error = consume(invocation, workSummary, true))
        return std::move(error);
      if (llvm::Error error =
              add(summary.completedInvocations, 1, "completed invocation"))
        return std::move(error);
      previousNode = invocation.planNodeOrdinal;
    }
    if (planRecords.incomplete().has_value() !=
        planRecords.incompleteWorkSummary().has_value())
      return invalid("incomplete Generate record and work summary presence "
                     "differs");
    if (planRecords.incomplete()) {
      const GenerateInvocationRecord &invocation = *planRecords.incomplete();
      const GenerateInvocationWorkSummary &workSummary =
          *planRecords.incompleteWorkSummary();
      if (previousNode && invocation.planNodeOrdinal <= *previousNode)
        return invalid("incomplete Generate invocation does not follow the "
                       "completed prefix");
      if (llvm::Error error = consume(invocation, workSummary, false))
        return std::move(error);
      if (llvm::Error error =
              add(summary.incompleteInvocations, 1, "incomplete invocation"))
        return std::move(error);
    }
  }
  return summary;
}

llvm::StringRef toString(const DsePlanIncompleteReason &reason) {
  return std::visit(
      [](const auto &value) -> llvm::StringRef {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>) {
          switch (value) {
          case CandidateGeneratorIncompleteReason::ProofNotEstablished:
            return "candidate_proof_not_established";
          case CandidateGeneratorIncompleteReason::SemanticLimitReached:
            return "candidate_semantic_limit_reached";
          case CandidateGeneratorIncompleteReason::ProviderUnavailable:
            return "candidate_provider_unavailable";
          case CandidateGeneratorIncompleteReason::Unsupported:
            return "candidate_generation_unsupported";
          case CandidateGeneratorIncompleteReason::ExecutionFailed:
            return "candidate_execution_failed";
          case CandidateGeneratorIncompleteReason::CancelledOrTimeout:
            return "candidate_cancelled_or_timeout";
          }
        } else if constexpr (std::is_same_v<
                                 T, PromotionAcquisitionIncompleteReason>) {
          switch (value) {
          case PromotionAcquisitionIncompleteReason::ProviderUnavailable:
            return "evidence_provider_unavailable";
          case PromotionAcquisitionIncompleteReason::SemanticWorkLimit:
            return "evidence_semantic_work_limit";
          case PromotionAcquisitionIncompleteReason::ObjectiveUnavailable:
            return "evidence_objective_unavailable";
          case PromotionAcquisitionIncompleteReason::Unsupported:
            return "evidence_unsupported";
          }
        } else {
          return dse::toString(value);
        }
        llvm_unreachable("unknown DSE plan incomplete reason");
      },
      reason);
}

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDseConfigView &view, const ArtifactStore &store,
               const BlobStore &blobs) {
  const ResolvedDsePlan &plan = view.plan();
  CompletedDsePlanExecution completed =
      DsePlanExecutionBuilder::createCompleted(view.digest());

  for (std::size_t nodeIndex = 0; nodeIndex < plan.nodes().size();
       ++nodeIndex) {
    const ResolvedDsePlanNode &node = plan.nodes()[nodeIndex];
    if (const auto *generate = std::get_if<ResolvedGeneratePlanNode>(&node)) {
      std::vector<CandidateGeneratorInputBinding> inputs;
      inputs.reserve(generate->inputBindings().size());
      for (std::size_t index = 0; index < generate->inputBindings().size();
           ++index) {
        auto artifacts =
            resolveRuntimeInput(generate->inputBindings()[index], completed);
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
      auto result = invokeCandidateGenerator(inputs, *binding, store, blobs);
      if (!result)
        return result.takeError();
      if (auto *incomplete =
              std::get_if<IncompleteCandidateGeneratorResult>(
                  &result->outcome)) {
        GenerateInvocationRecord invocationRecord{
            static_cast<std::uint64_t>(nodeIndex), std::move(inputs),
            std::move(*binding), std::move(incomplete->retainedOutputBindings),
            std::move(incomplete->lineageEdges)};
        GenerateInvocationWorkSummary workSummary{
            static_cast<std::uint64_t>(nodeIndex),
            std::move(result->workSummary)};
        return DsePlanExecutionOutcome{
            DsePlanExecutionBuilder::incompleteGenerate(
                static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
                std::move(completed), std::move(invocationRecord),
                std::move(workSummary))};
      }
      auto &generated =
          std::get<CompletedCandidateGeneratorResult>(result->outcome);
      GenerateInvocationRecord invocationRecord{
          static_cast<std::uint64_t>(nodeIndex), std::move(inputs),
          std::move(*binding), std::move(generated.outputBindings),
          std::move(generated.lineageEdges)};
      GenerateInvocationWorkSummary workSummary{
          static_cast<std::uint64_t>(nodeIndex),
          std::move(result->workSummary)};
      if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
              completed, std::move(invocationRecord), std::move(workSummary)))
        return std::move(error);
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
      auto artifacts =
          resolveRuntimeInput(promote.inputBindings()[index], completed);
      if (!artifacts)
        return artifacts.takeError();
      inputs.push_back(
          {PromotionAcquisitionInputSlotRef(static_cast<std::uint32_t>(index)),
           std::move(*artifacts)});
    }
    const PromotionAcquisitionInputBinding *candidateInput =
        descriptor->candidateInputSlot.ordinal() < inputs.size()
            ? &inputs[descriptor->candidateInputSlot.ordinal()]
            : nullptr;
    const PromotionAcquisitionInputSlotDescriptor *candidateSlot =
        descriptor->findInputSlot(descriptor->candidateInputSlot);
    if (!candidateInput || !candidateSlot || !candidateSlot->schema)
      return invalid("Promote candidate input is unavailable");
    auto candidateSet =
        CandidateSet::get(*candidateSlot->schema, candidateInput->artifacts);
    if (!candidateSet)
      return candidateSet.takeError();
    const QualityGatePolicy *qualityGate =
        plan.resolve(promote.qualityGateRef());
    if (!qualityGate)
      return invalid("resolved Promote node lost its quality gate");

    if (const auto *topK = std::get_if<TopKSelection>(&promote.selection());
        topK && !promote.objectiveObligations().empty()) {
      if (!plan.objectiveProgram())
        return invalid("resolved TopK node lost its ObjectiveProgram");
      auto staged =
          executeStagedTopK(promote, *descriptor, inputs, *candidateSet,
                            view.evidenceObligationTemplates(), *qualityGate,
                            *plan.objectiveProgram(), *topK, store, blobs);
      if (!staged)
        return staged.takeError();
      if (auto *incomplete = std::get_if<IncompleteStagedTopK>(&*staged))
        return DsePlanExecutionOutcome{
            DsePlanExecutionBuilder::incompletePromote(
                static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
                std::move(completed), {{}, std::move(incomplete->evidence)})};
      auto &stagedCompleted = std::get<CompletedStagedTopK>(*staged);
      DsePlanExecutionBuilder::appendPromote(
          completed, {std::move(stagedCompleted.selected),
                      std::move(stagedCompleted.evidence)});
      continue;
    }

    auto acquisition = invokePromotionAcquisition(
        inputs, promote.acquisitionBinding(),
        view.evidenceObligationTemplates(),
        {candidateSet->candidates(),
         promote.acquisitionBinding().evidenceObligations()},
        store, blobs);
    if (!acquisition)
      return acquisition.takeError();
    if (auto *incomplete =
            std::get_if<IncompletePromotionAcquisition>(&*acquisition)) {
      auto retained = publishEvidence(incomplete->retainedEvidence, store);
      if (!retained)
        return retained.takeError();
      return DsePlanExecutionOutcome{DsePlanExecutionBuilder::incompletePromote(
          static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
          std::move(completed), {{}, std::move(*retained)})};
    }

    auto &acquired = std::get<CompletedPromotionAcquisition>(*acquisition);
    auto promotion = promoteCandidates(
        *candidateSet, descriptor->candidateRole, acquired.evidence,
        *qualityGate, promote.selection(), plan.objectiveProgram(), store);
    if (!promotion)
      return promotion.takeError();
    if (auto *incomplete = std::get_if<IncompleteSelection>(&*promotion))
      return DsePlanExecutionOutcome{DsePlanExecutionBuilder::incompletePromote(
          static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
          std::move(completed), {{}, std::move(incomplete->retainedEvidence)})};
    if (auto *none = std::get_if<CompletedNoFeasibleCandidate>(&*promotion)) {
      DsePlanExecutionBuilder::appendPromote(
          completed, {{}, std::move(none->satisfiedEvidence)});
    } else {
      auto &selected = std::get<CompletedSelection>(*promotion);
      DsePlanExecutionBuilder::appendPromote(
          completed, {std::move(selected.selected),
                      std::move(selected.satisfiedEvidence)});
    }
  }
  return DsePlanExecutionOutcome{std::move(completed)};
}

} // namespace loom::dse
