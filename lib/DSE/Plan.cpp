#include "DSE/Plan.h"

#include "Common/ArtifactLocalReference.h"
#include "DSE/PlanExecutor.h"
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
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality,
                             slot.modelParameterContract
                                 ? std::optional(*slot.modelParameterContract)
                                 : std::nullopt,
                             slot.calibrationPartitionRole};
}

llvm::Expected<PlanValueDescriptor>
inputDescriptor(const PromotionAcquisitionInputSlotDescriptor &slot) {
  if (!slot.schema || !validRole(slot.role) ||
      !validCardinality(slot.cardinality))
    return invalid("promotion acquisition input slot is not plan-typed");
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality,
                             slot.modelParameterContract
                                 ? std::optional(*slot.modelParameterContract)
                                 : std::nullopt,
                             slot.calibrationPartitionRole};
}

llvm::Expected<PlanValueDescriptor>
outputDescriptor(const CandidateGeneratorOutputSlotDescriptor &slot) {
  if (!slot.schema || !validRole(slot.role) ||
      !validCardinality(slot.cardinality))
    return invalid("candidate generator output slot is not plan-typed");
  return PlanValueDescriptor{slot.role, *slot.schema, slot.cardinality,
                             slot.modelParameterContract
                                 ? std::optional(*slot.modelParameterContract)
                                 : std::nullopt,
                             slot.calibrationPartitionRole};
}

bool compatible(const PlanValueDescriptor &producer,
                const PlanValueDescriptor &consumer) {
  return producer.role == consumer.role && producer.schema == consumer.schema &&
         planCardinalityCanFlow(producer.cardinality, consumer.cardinality) &&
         producer.modelParameterContract == consumer.modelParameterContract &&
         producer.calibrationPartitionRole == consumer.calibrationPartitionRole;
}

llvm::Expected<const PlanValueDescriptor *>
resolvePriorOutput(PlanOutputRef output, std::size_t nodeIndex,
                   llvm::ArrayRef<ResolvedDsePlanNode> nodes,
                   llvm::ArrayRef<std::uint64_t> outputOffsets,
                   llvm::ArrayRef<PlanValueDescriptor> outputs) {
  if (output.producerNodeOrdinal >= nodeIndex ||
      output.producerNodeOrdinal >= nodes.size())
    return invalid("produced input must reference an earlier node");
  const std::uint64_t begin = outputOffsets[output.producerNodeOrdinal];
  const std::uint64_t end = outputOffsets[output.producerNodeOrdinal + 1];
  if (output.outputSlotOrdinal >= end - begin)
    return invalid("produced input references an unknown output slot");
  return &outputs[begin + output.outputSlotOrdinal];
}

llvm::Expected<std::optional<CalibrationPartitionRole>>
evidencePartitionRole(const ResolvedPromotionAcquisitionBinding &binding,
                      llvm::ArrayRef<EvidenceObligationTemplate> templates) {
  std::optional<CalibrationPartitionRole> selected;
  bool sawUnpartitioned = false;
  for (EvidenceObligationTemplateRef reference :
       binding.evidenceObligations()) {
    if (reference.ordinal() >= templates.size())
      return invalid("acquisition references a foreign Evidence obligation");
    const std::optional<CalibrationPartitionRole> role =
        templates[reference.ordinal()].calibrationPartitionRole();
    if (!role) {
      sawUnpartitioned = true;
      continue;
    }
    if (selected && *selected != *role)
      return invalid("one Promote output cannot mix calibration partitions");
    selected = role;
  }
  if (selected && sawUnpartitioned)
    return invalid(
        "one Promote output cannot mix partitioned and unpartitioned Evidence");
  return selected;
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
    if (const auto *output = std::get_if<PlanOutputRef>(&binding)) {
      auto descriptor =
          resolvePriorOutput(*output, nodeIndex, nodes, outputOffsets, outputs);
      if (!descriptor)
        return descriptor.takeError();
      if (!compatible(**descriptor, expected[inputIndex]))
        return invalid("produced input role, artifact schema, or cardinality "
                       "does not match its slot");
      continue;
    }

    auto &join = std::get<BoundedPlanOutputJoin>(binding);
    if ((join.outputs.empty() && join.exactArtifacts.empty()) ||
        join.maximumArtifacts == 0 ||
        join.producerArtifactLimit() < join.maximumArtifacts)
      return invalid("bounded output join requires an exact or prior input "
                     "and a positive artifact bound with no smaller producer "
                     "bound");
    if (!llvm::is_sorted(join.outputs) ||
        std::adjacent_find(join.outputs.begin(), join.outputs.end()) !=
            join.outputs.end())
      return invalid("bounded output join sources are not canonical and "
                     "unique");
    for (const ArtifactRootReference &artifact : join.exactArtifacts)
      if (!matchesSchema(artifact, expected[inputIndex].schema))
        return invalid(
            "bounded output join exact artifact has the wrong schema");
    llvm::sort(join.exactArtifacts, artifactRootReferenceLess);
    join.exactArtifacts.erase(
        std::unique(join.exactArtifacts.begin(), join.exactArtifacts.end()),
        join.exactArtifacts.end());
    if (join.exactArtifacts.size() > join.maximumArtifacts)
      return invalid("bounded output join exact artifacts exceed its bound");
    std::optional<PlanValueDescriptor> joined;
    bool guaranteesArtifact = !join.exactArtifacts.empty();
    for (PlanOutputRef output : join.outputs) {
      auto descriptor =
          resolvePriorOutput(output, nodeIndex, nodes, outputOffsets, outputs);
      if (!descriptor)
        return descriptor.takeError();
      guaranteesArtifact |=
          planCardinalityBounds((*descriptor)->cardinality).minimum != 0;
      if (!joined) {
        joined = **descriptor;
        continue;
      }
      if ((*descriptor)->role != joined->role ||
          (*descriptor)->schema != joined->schema ||
          (*descriptor)->modelParameterContract !=
              joined->modelParameterContract ||
          (*descriptor)->calibrationPartitionRole !=
              joined->calibrationPartitionRole)
        return invalid("bounded output join mixes incompatible plan values");
    }
    if (joined) {
      // Deduplication can collapse every nonempty producer to one root, but
      // a positive retention bound cannot turn a nonempty union into empty.
      joined->cardinality =
          join.maximumArtifacts == 1
              ? (guaranteesArtifact ? PlanValueCardinality::ExactlyOne
                                    : PlanValueCardinality::ZeroOrOne)
              : (guaranteesArtifact ? PlanValueCardinality::NonEmptySet
                                    : PlanValueCardinality::FiniteSet);
    }
    if (joined && !compatible(*joined, expected[inputIndex]))
      return invalid("bounded output join role, artifact schema, or "
                     "cardinality does not match its slot");
    if (!joined && !planCardinalityContains(expected[inputIndex].cardinality,
                                            join.exactArtifacts.size()))
      return invalid("exact-only bounded output join violates its slot "
                     "cardinality");
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
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
};

struct IncompleteStagedTopK final {
  DsePlanIncompleteReason reason;
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
  bool executionStopped = true;
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
    const BlobStore &blobs, PromotionEvidenceExecutor *executor) {
  std::vector<std::uint32_t> objectiveOrdinals;
  objectiveOrdinals.reserve(promote.objectiveObligations().size());
  for (EvidenceObligationTemplateRef ref : promote.objectiveObligations())
    objectiveOrdinals.push_back(ref.ordinal());

  auto objectiveAcquisition = invokePromotionAcquisition(
      inputs, promote.acquisitionBinding(), evidenceObligations,
      {candidateSet.candidates(), promote.objectiveObligations()}, store, blobs,
      executor);
  if (!objectiveAcquisition)
    return objectiveAcquisition.takeError();
  if (auto *incomplete =
          std::get_if<IncompletePromotionAcquisition>(&*objectiveAcquisition)) {
    auto retained = publishEvidence(incomplete->retainedEvidence, store);
    if (!retained)
      return retained.takeError();
    return StagedTopKOutcome{
        IncompleteStagedTopK{DsePlanIncompleteReason{incomplete->reason},
                             {},
                             {},
                             std::move(*retained),
                             true}};
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
                             {},
                             {},
                             std::move(incomplete->retainedEvidence),
                             true}};

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
  std::optional<DsePlanIncompleteReason> retainedIncompleteness;

  std::size_t cursor = 0;
  while (cursor < ranked.rankedCandidates.size() &&
         selected.size() < selection.k) {
    const std::size_t batchSize = 1;
    llvm::ArrayRef<ArtifactRootReference> rankedBatch(
        ranked.rankedCandidates.data() + cursor, batchSize);
    std::vector<ArtifactRootReference> candidateDomain(rankedBatch.begin(),
                                                       rankedBatch.end());
    llvm::sort(candidateDomain, artifactRootReferenceLess);
    PromotionAcquisitionOutcome deferred = CompletedPromotionAcquisition{{}};
    if (!deferredObligations.empty()) {
      auto acquired = invokePromotionAcquisition(
          inputs, promote.acquisitionBinding(), evidenceObligations,
          {candidateDomain, deferredObligations}, store, blobs, executor);
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
      if (incomplete->reason ==
              PromotionAcquisitionIncompleteReason::Unsupported &&
          incomplete->candidate &&
          *incomplete->candidate == rankedBatch.front()) {
        if (!retainedIncompleteness)
          retainedIncompleteness = DsePlanIncompleteReason{incomplete->reason};
        ++cursor;
        continue;
      }
      const bool executionStopped = selected.empty();
      std::vector<ArtifactRootReference> preferenceOrder = selected;
      llvm::sort(selected, artifactRootReferenceLess);
      return StagedTopKOutcome{
          IncompleteStagedTopK{DsePlanIncompleteReason{incomplete->reason},
                               std::move(selected), std::move(preferenceOrder),
                               std::move(retainedEvidence), executionStopped}};
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
        if (incomplete->reason ==
            IncompleteSelectionReason::UnsupportedEvidence) {
          if (!retainedIncompleteness)
            retainedIncompleteness =
                DsePlanIncompleteReason{incomplete->reason};
          continue;
        }
        const bool executionStopped = selected.empty();
        std::vector<ArtifactRootReference> preferenceOrder = selected;
        llvm::sort(selected, artifactRootReferenceLess);
        return StagedTopKOutcome{IncompleteStagedTopK{
            DsePlanIncompleteReason{incomplete->reason}, std::move(selected),
            std::move(preferenceOrder), std::move(retainedEvidence),
            executionStopped}};
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

  std::vector<ArtifactRootReference> preferenceOrder = selected;
  llvm::sort(selected, artifactRootReferenceLess);
  canonicalizeReferences(retainedEvidence);
  if (retainedIncompleteness) {
    const bool executionStopped = selected.empty();
    return StagedTopKOutcome{
        IncompleteStagedTopK{std::move(*retainedIncompleteness),
                             std::move(selected), std::move(preferenceOrder),
                             std::move(retainedEvidence), executionStopped}};
  }
  return StagedTopKOutcome{CompletedStagedTopK{std::move(selected),
                                               std::move(preferenceOrder),
                                               std::move(retainedEvidence)}};
}

llvm::Expected<std::vector<ArtifactRootReference>>
resolveRuntimeInput(const PlanInputBinding &input,
                    const CompletedDsePlanExecution &completed) {
  if (const auto *exact = std::get_if<ExactPlanArtifacts>(&input))
    return exact->artifacts;
  if (const auto *output = std::get_if<PlanOutputRef>(&input)) {
    if (!completed.hasOutput(*output))
      return invalid("resolved use-def references an unavailable output");
    return completed.resolve(*output).vec();
  }
  const auto &join = std::get<BoundedPlanOutputJoin>(input);
  std::vector<ArtifactRootReference> artifacts = join.exactArtifacts;
  for (PlanOutputRef output : join.outputs) {
    if (!completed.hasOutput(output))
      return invalid("bounded output join references an unavailable output");
    llvm::ArrayRef<ArtifactRootReference> source = completed.resolve(output);
    artifacts.insert(artifacts.end(), source.begin(), source.end());
  }
  llvm::sort(artifacts, artifactRootReferenceLess);
  artifacts.erase(std::unique(artifacts.begin(), artifacts.end()),
                  artifacts.end());
  if (artifacts.size() > join.maximumArtifacts)
    artifacts.erase(artifacts.begin() +
                        static_cast<std::size_t>(join.maximumArtifacts),
                    artifacts.end());
  return artifacts;
}

llvm::ArrayRef<PlanInputBinding>
planNodeInputs(const ResolvedDsePlanNode &node) {
  if (const auto *generate = std::get_if<ResolvedGeneratePlanNode>(&node))
    return generate->inputBindings();
  return std::get<ResolvedPromotePlanNode>(node).inputBindings();
}

std::vector<CandidateGeneratorOutputDemand>
deriveOutputDemands(const ResolvedDsePlan &plan,
                    std::size_t producerNodeOrdinal,
                    const CandidateGeneratorDescriptor &descriptor) {
  std::vector<CandidateGeneratorOutputDemand> demands;
  demands.reserve(descriptor.outputSlots.size());
  for (const CandidateGeneratorOutputSlotDescriptor &slot :
       descriptor.outputSlots) {
    const PlanOutputRef produced{
        static_cast<std::uint64_t>(producerNodeOrdinal), slot.slot.ordinal()};
    bool referenced = false;
    bool unbounded = false;
    std::uint64_t maximumArtifacts = 0;
    for (std::size_t consumer = producerNodeOrdinal + 1;
         consumer != plan.nodes().size() && !unbounded; ++consumer) {
      for (const PlanInputBinding &input :
           planNodeInputs(plan.nodes()[consumer])) {
        if (const auto *output = std::get_if<PlanOutputRef>(&input)) {
          if (*output == produced) {
            referenced = true;
            unbounded = true;
            break;
          }
          continue;
        }
        const auto *join = std::get_if<BoundedPlanOutputJoin>(&input);
        if (!join || !llvm::is_contained(join->outputs, produced))
          continue;
        referenced = true;
        maximumArtifacts =
            std::max(maximumArtifacts, join->producerArtifactLimit());
      }
    }
    demands.push_back(CandidateGeneratorOutputDemand{
        slot.slot, referenced && !unbounded
                       ? std::optional<std::uint64_t>(maximumArtifacts)
                       : std::nullopt});
  }
  return demands;
}

} // namespace

class DsePlanExecutionBuilder final {
public:
  static CompletedDsePlanExecution createCompleted(ComponentViewDigest digest) {
    return CompletedDsePlanExecution(digest);
  }

  static llvm::Error appendGenerate(
      CompletedDsePlanExecution &completed, GenerateInvocationRecord invocation,
      GenerateInvocationWorkSummary workSummary,
      std::optional<std::vector<std::uint8_t>> feedback, bool dispatched) {
    return completed.appendGenerate(std::move(invocation),
                                    std::move(workSummary), std::move(feedback),
                                    dispatched);
  }

  static void
  appendPromote(CompletedDsePlanExecution &completed,
                std::vector<std::vector<ArtifactRootReference>> outputBindings,
                std::vector<ArtifactRootReference> preferenceOrder = {}) {
    completed.appendPromote(std::move(outputBindings),
                            std::move(preferenceOrder));
  }

  static IncompleteDsePlanExecution
  incompleteGenerate(std::uint64_t nodeOrdinal, DsePlanIncompleteReason reason,
                     CompletedDsePlanExecution availableExecution,
                     bool executionStopped) {
    return IncompleteDsePlanExecution(nodeOrdinal, std::move(reason),
                                      std::move(availableExecution), true,
                                      executionStopped);
  }

  static IncompleteDsePlanExecution
  incompleteRetained(std::uint64_t nodeOrdinal, DsePlanIncompleteReason reason,
                     CompletedDsePlanExecution availableExecution,
                     bool generateNode) {
    return IncompleteDsePlanExecution(nodeOrdinal, std::move(reason),
                                      std::move(availableExecution),
                                      generateNode, false);
  }

  static IncompleteDsePlanExecution incompletePromote(
      std::uint64_t nodeOrdinal, DsePlanIncompleteReason reason,
      CompletedDsePlanExecution availableExecution,
      std::vector<std::vector<ArtifactRootReference>> outputBindings,
      std::vector<ArtifactRootReference> preferenceOrder = {},
      bool executionStopped = true) {
    availableExecution.appendPromote(std::move(outputBindings),
                                     std::move(preferenceOrder));
    return IncompleteDsePlanExecution(nodeOrdinal, std::move(reason),
                                      std::move(availableExecution), false,
                                      executionStopped);
  }

  static DsePlanGenerateInvocationRecords
  takeGenerateInvocationRecords(DsePlanExecutionOutcome outcome) {
    CompletedDsePlanExecution *execution =
        std::get_if<CompletedDsePlanExecution>(&outcome);
    if (!execution)
      execution =
          &std::get<IncompleteDsePlanExecution>(outcome).availableExecution_;
    std::vector<GenerateInvocationRecord> completed;
    std::vector<GenerateInvocationWorkSummary> completedWork;
    std::vector<GenerateInvocationRecord> incomplete;
    std::vector<GenerateInvocationWorkSummary> incompleteWork;
    for (std::size_t ordinal = 0;
         ordinal < execution->generateInvocations_.size(); ++ordinal) {
      GenerateInvocationRecord invocation =
          std::move(execution->generateInvocations_[ordinal]);
      GenerateInvocationWorkSummary work =
          std::move(execution->generateWorkSummaries_[ordinal]);
      if (invocation.incompleteReason) {
        incomplete.push_back(std::move(invocation));
        incompleteWork.push_back(std::move(work));
      } else {
        completed.push_back(std::move(invocation));
        completedWork.push_back(std::move(work));
      }
    }
    return DsePlanGenerateInvocationRecords{
        execution->resolvedDseConfigViewDigest_, std::move(completed),
        std::move(completedWork), std::move(incomplete),
        std::move(incompleteWork)};
  }

  static DsePlanGenerateInvocationRecords
  projectGenerateInvocationRecords(const DsePlanExecutionOutcome &outcome) {
    const CompletedDsePlanExecution *execution =
        std::get_if<CompletedDsePlanExecution>(&outcome);
    const IncompleteDsePlanExecution *incompleteExecution = nullptr;
    if (!execution) {
      incompleteExecution = &std::get<IncompleteDsePlanExecution>(outcome);
      execution = &incompleteExecution->availableExecution_;
    }
    std::vector<GenerateInvocationRecord> completed;
    std::vector<GenerateInvocationWorkSummary> completedWork;
    std::vector<GenerateInvocationRecord> incomplete;
    std::vector<GenerateInvocationWorkSummary> incompleteWork;
    for (std::size_t ordinal = 0;
         ordinal < execution->generateInvocations_.size(); ++ordinal) {
      const GenerateInvocationRecord &invocation =
          execution->generateInvocations_[ordinal];
      const GenerateInvocationWorkSummary &work =
          execution->generateWorkSummaries_[ordinal];
      if (incompleteExecution && incompleteExecution->executionStopped() &&
          invocation.planNodeOrdinal > incompleteExecution->nodeOrdinal())
        continue;
      if (invocation.incompleteReason) {
        incomplete.push_back(invocation);
        incompleteWork.push_back(work);
      } else {
        completed.push_back(invocation);
        completedWork.push_back(work);
      }
    }
    return DsePlanGenerateInvocationRecords{
        execution->resolvedDseConfigViewDigest_, std::move(completed),
        std::move(completedWork), std::move(incomplete),
        std::move(incompleteWork)};
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
    auto evidencePartition =
        evidencePartitionRole(*acquisitionBinding, evidenceObligationTemplates);
    if (!evidencePartition)
      return evidencePartition.takeError();
    const bool heldOut =
        *evidencePartition == CalibrationPartitionRole::HeldOut;
    const bool modelRelease =
        definition.purpose == PromotePurpose::ModelRelease;
    if (heldOut != modelRelease)
      return invalid("held-out Evidence is permitted only in a terminal "
                     "ModelRelease Promote node");
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
    outputs.push_back(
        {PlanValueRole::CandidateSet, *candidateSlot->schema,
         PlanValueCardinality::FiniteSet,
         candidateSlot->modelParameterContract
             ? std::optional(*candidateSlot->modelParameterContract)
             : std::nullopt});
    outputs.push_back({PlanValueRole::EvidenceSet,
                       evaluation::EvaluationEvidence::artifactSchema,
                       PlanValueCardinality::FiniteSet, std::nullopt,
                       *evidencePartition});
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

llvm::ArrayRef<ArtifactRootReference>
CompletedDsePlanExecution::resolvePreferenceOrder(PlanOutputRef output) const {
  if (!hasOutput(output) || output.outputSlotOrdinal != 0)
    return {};
  const NodeOutputs &node = nodeOutputs_[output.producerNodeOrdinal];
  const auto *promote = std::get_if<PromoteNodeOutputs>(&node);
  return promote
             ? llvm::ArrayRef<ArtifactRootReference>(promote->preferenceOrder)
             : llvm::ArrayRef<ArtifactRootReference>();
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

bool CompletedDsePlanExecution::generateInvocationWasDispatched(
    std::size_t ordinal) const {
  return ordinal < generateDispatched_.size() && generateDispatched_[ordinal];
}

llvm::Error CompletedDsePlanExecution::appendGenerate(
    GenerateInvocationRecord invocation,
    GenerateInvocationWorkSummary workSummary,
    std::optional<std::vector<std::uint8_t>> feedback, bool dispatched) {
  if (invocation.planNodeOrdinal != nodeOutputs_.size())
    return invalid("Generate invocation ordinal does not follow plan order");
  if (invocation.incompleteReason && invocation.infeasibilityProof)
    return invalid("Generate invocation has both incomplete and "
                   "ProvenInfeasible outcomes");
  if (workSummary.planNodeOrdinal != invocation.planNodeOrdinal)
    return invalid("Generate work summary names a different plan node");
  if (llvm::Error error = validateCandidateGeneratorWorkSummary(
          invocation.generatorBinding.descriptorRef(), workSummary.units))
    return error;
  const std::size_t invocationOrdinal = generateInvocations_.size();
  const std::uint64_t planNodeOrdinal = invocation.planNodeOrdinal;
  generateInvocations_.push_back(std::move(invocation));
  generateWorkSummaries_.push_back(std::move(workSummary));
  generateDispatched_.push_back(dispatched);
  if (feedback)
    generateFeedback_.push_back({planNodeOrdinal, std::move(*feedback)});
  nodeOutputs_.push_back(GenerateNodeOutputs{invocationOrdinal});
  return llvm::Error::success();
}

void CompletedDsePlanExecution::appendPromote(
    std::vector<std::vector<ArtifactRootReference>> outputBindings,
    std::vector<ArtifactRootReference> preferenceOrder) {
  nodeOutputs_.push_back(PromoteNodeOutputs{std::move(outputBindings),
                                            std::move(preferenceOrder)});
}

std::size_t IncompleteDsePlanExecution::retainedOutputCount() const {
  std::size_t count = 0;
  while (availableExecution_.hasOutput(
      PlanOutputRef{nodeOrdinal_, static_cast<std::uint32_t>(count)}))
    ++count;
  return count;
}

llvm::ArrayRef<ArtifactRootReference>
IncompleteDsePlanExecution::retainedOutput(
    std::size_t outputSlotOrdinal) const {
  if (outputSlotOrdinal > std::numeric_limits<std::uint32_t>::max())
    return {};
  return availableExecution_.resolve(PlanOutputRef{
      nodeOrdinal_, static_cast<std::uint32_t>(outputSlotOrdinal)});
}

const GenerateInvocationRecord *
IncompleteDsePlanExecution::incompleteGenerateInvocation() const {
  if (!generateNode_)
    return nullptr;
  const auto found = llvm::find_if(
      availableExecution_.generateInvocations(), [&](const auto &invocation) {
        return invocation.planNodeOrdinal == nodeOrdinal_;
      });
  return found == availableExecution_.generateInvocations().end() ? nullptr
                                                                  : &*found;
}

const GenerateInvocationWorkSummary *
IncompleteDsePlanExecution::incompleteGenerateWorkSummary() const {
  if (!generateNode_)
    return nullptr;
  const auto found = llvm::find_if(
      availableExecution_.generateWorkSummaries(), [&](const auto &summary) {
        return summary.planNodeOrdinal == nodeOrdinal_;
      });
  return found == availableExecution_.generateWorkSummaries().end() ? nullptr
                                                                    : &*found;
}

DsePlanGenerateInvocationRecords
takeDsePlanGenerateInvocationRecords(DsePlanExecutionOutcome outcome) {
  return DsePlanExecutionBuilder::takeGenerateInvocationRecords(
      std::move(outcome));
}

DsePlanGenerateInvocationRecords projectDsePlanGenerateInvocationRecords(
    const DsePlanExecutionOutcome &outcome) {
  return DsePlanExecutionBuilder::projectGenerateInvocationRecords(outcome);
}

llvm::Expected<DsePlanGenerateInvocationSummary>
validateAndSummarizeDsePlanGenerateInvocations(
    llvm::ArrayRef<DsePlanGenerateInvocationRecords> records,
    const ArtifactStore &store, const BlobStore &blobs) {
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
    if (completed == invocation.incompleteReason.has_value())
      return invalid("Generate invocation completeness classification "
                     "disagrees with its typed outcome");
    if (invocation.incompleteReason && invocation.infeasibilityProof)
      return invalid("Generate invocation has both incomplete and "
                     "ProvenInfeasible outcomes");
    if (workSummary.planNodeOrdinal != invocation.planNodeOrdinal)
      return invalid("Generate work summary names a different plan node");
    if (llvm::Error error = validateCandidateGeneratorWorkSummary(
            invocation.generatorBinding.descriptorRef(), workSummary.units))
      return error;
    if (llvm::Error error = validateCanonicalCandidateGeneratorInvocation(
            invocation.inputBindings, invocation.generatorBinding,
            invocation.outputBindings, invocation.lineageEdges, completed,
            invocation.infeasibilityProof,
            store, blobs))
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
    if (planRecords.incomplete().size() !=
        planRecords.incompleteWorkSummaries().size())
      return invalid("incomplete Generate records and work summaries differ "
                     "in width");
    struct OrderedInvocation final {
      const GenerateInvocationRecord *invocation;
      const GenerateInvocationWorkSummary *workSummary;
      bool completed;
    };
    std::vector<OrderedInvocation> ordered;
    ordered.reserve(planRecords.completed().size() +
                    planRecords.incomplete().size());
    for (std::size_t ordinal = 0; ordinal != planRecords.completed().size();
         ++ordinal)
      ordered.push_back({&planRecords.completed()[ordinal],
                         &planRecords.completedWorkSummaries()[ordinal], true});
    for (std::size_t ordinal = 0; ordinal != planRecords.incomplete().size();
         ++ordinal)
      ordered.push_back({&planRecords.incomplete()[ordinal],
                         &planRecords.incompleteWorkSummaries()[ordinal],
                         false});
    llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
      return lhs.invocation->planNodeOrdinal < rhs.invocation->planNodeOrdinal;
    });
    std::optional<std::uint64_t> previousNode;
    for (const OrderedInvocation &entry : ordered) {
      const GenerateInvocationRecord &invocation = *entry.invocation;
      if (previousNode && invocation.planNodeOrdinal <= *previousNode)
        return invalid(
            "Generate invocation ordinals are not strictly increasing");
      if (llvm::Error error =
              consume(invocation, *entry.workSummary, entry.completed))
        return std::move(error);
      if (invocation.infeasibilityProof) {
        if (llvm::Error error = add(summary.provenInfeasibleInvocations, 1,
                                    "ProvenInfeasible invocation"))
          return std::move(error);
      }
      std::uint64_t &count = entry.completed ? summary.completedInvocations
                                             : summary.incompleteInvocations;
      if (llvm::Error error =
              add(count, 1, entry.completed ? "completed invocation"
                                            : "incomplete invocation"))
        return std::move(error);
      previousNode = invocation.planNodeOrdinal;
    }
  }
  return summary;
}

llvm::StringRef toString(const DsePlanIncompleteReason &reason) {
  return std::visit(
      [](const auto &value) -> llvm::StringRef {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>) {
          return candidateGeneratorIncompleteReasonSpelling(value);
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
          case PromotionAcquisitionIncompleteReason::CancelledOrTimeout:
            return "evidence_cancelled_or_timeout";
          }
        } else {
          return dse::toString(value);
        }
        llvm_unreachable("unknown DSE plan incomplete reason");
      },
      reason);
}

llvm::Expected<DsePlanExecutionOutcome> detail::executeDsePlanWithWorkExecutor(
    const ResolvedDseConfigView &view, const ArtifactStore &store,
    const BlobStore &blobs, detail::DsePlanWorkExecutor *executor,
    ExecutionControlView executionControl) {
  const ResolvedDsePlan &plan = view.plan();
  CompletedDsePlanExecution completed =
      DsePlanExecutionBuilder::createCompleted(view.digest());
  std::optional<std::pair<std::uint64_t, DsePlanIncompleteReason>>
      retainedIncompleteness;
  const auto canContinue = [](CandidateGeneratorIncompleteReason reason) {
    return reason == CandidateGeneratorIncompleteReason::ProofNotEstablished ||
           reason == CandidateGeneratorIncompleteReason::SemanticLimitReached;
  };

  for (std::size_t nodeIndex = 0; nodeIndex < plan.nodes().size();
       ++nodeIndex) {
    if (executionControl.stopRequested())
      return DsePlanExecutionOutcome{
          DsePlanExecutionBuilder::incompleteGenerate(
              static_cast<std::uint64_t>(nodeIndex),
              CandidateGeneratorIncompleteReason::CancelledOrTimeout,
              std::move(completed), true)};
    const ResolvedDsePlanNode &node = plan.nodes()[nodeIndex];
    if (executor && std::holds_alternative<ResolvedGeneratePlanNode>(node)) {
      std::vector<detail::DseGenerateExecutionTask> tasks;
      for (std::size_t candidateIndex = nodeIndex;
           candidateIndex < plan.nodes().size(); ++candidateIndex) {
        const auto *candidate = std::get_if<ResolvedGeneratePlanNode>(
            &plan.nodes()[candidateIndex]);
        if (!candidate)
          break;
        const bool dependsOnCurrentFrontier = llvm::any_of(
            candidate->inputBindings(), [&](const PlanInputBinding &binding) {
              if (const auto *output = std::get_if<PlanOutputRef>(&binding))
                return output->producerNodeOrdinal >= nodeIndex;
              const auto *join = std::get_if<BoundedPlanOutputJoin>(&binding);
              return join &&
                     llvm::any_of(join->outputs, [&](PlanOutputRef output) {
                       return output.producerNodeOrdinal >= nodeIndex;
                     });
            });
        if (dependsOnCurrentFrontier)
          break;

        std::vector<CandidateGeneratorInputBinding> inputs;
        inputs.reserve(candidate->inputBindings().size());
        for (std::size_t inputIndex = 0;
             inputIndex < candidate->inputBindings().size(); ++inputIndex) {
          auto artifacts = resolveRuntimeInput(
              candidate->inputBindings()[inputIndex], completed);
          if (!artifacts)
            return artifacts.takeError();
          inputs.push_back({CandidateGeneratorInputSlotRef(
                                static_cast<std::uint32_t>(inputIndex)),
                            std::move(*artifacts)});
        }
        auto binding = ResolvedCandidateGeneratorBinding::get(
            candidate->descriptorRef(), candidate->canonicalConfigBytes(),
            candidate->configDigest());
        if (!binding)
          return binding.takeError();
        const CandidateGeneratorDescriptor *descriptor =
            candidate->descriptorRef().descriptor();
        if (!descriptor)
          return invalid("Generate frontier lost its descriptor");
        tasks.push_back({static_cast<std::uint64_t>(candidateIndex),
                         std::move(inputs),
                         deriveOutputDemands(plan, candidateIndex, *descriptor),
                         std::move(*binding)});
      }
      if (tasks.empty())
        return invalid("Generate frontier did not contain its first node");
      auto results = executor->executeGenerateBatch(tasks, store, blobs);
      if (!results)
        return results.takeError();
      if (results->size() != tasks.size())
        return invalid("Generate frontier result width changed");
      std::optional<
          std::pair<std::uint64_t, CandidateGeneratorIncompleteReason>>
          blockingIncompleteness;
      for (std::size_t taskIndex = 0; taskIndex != tasks.size(); ++taskIndex) {
        detail::DseGenerateExecutionTask &task = tasks[taskIndex];
        CandidateGeneratorProviderResult &result = (*results)[taskIndex];
        if (auto *incomplete = std::get_if<IncompleteCandidateGeneratorResult>(
                &result.outcome)) {
          GenerateInvocationRecord invocationRecord{
              task.planNodeOrdinal,
              std::move(task.inputs),
              std::move(task.binding),
              std::move(incomplete->retainedOutputBindings),
              std::move(incomplete->lineageEdges),
              incomplete->reason};
          GenerateInvocationWorkSummary workSummary{
              task.planNodeOrdinal, std::move(result.workSummary)};
          if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
                  completed, std::move(invocationRecord),
                  std::move(workSummary), std::move(result.ownerFeedback),
                  result.dispatched))
            return std::move(error);
          if (canContinue(incomplete->reason)) {
            if (!retainedIncompleteness)
              retainedIncompleteness =
                  std::pair{task.planNodeOrdinal,
                            DsePlanIncompleteReason{incomplete->reason}};
          } else if (!blockingIncompleteness) {
            blockingIncompleteness =
                std::pair{task.planNodeOrdinal, incomplete->reason};
          }
          continue;
        }
        if (auto *proven =
                std::get_if<ProvenInfeasibleCandidateGeneratorResult>(
                    &result.outcome)) {
          GenerateInvocationRecord invocationRecord{
              task.planNodeOrdinal,
              std::move(task.inputs),
              std::move(task.binding),
              std::move(proven->outputBindings),
              {},
              std::nullopt,
              std::move(proven->proof)};
          GenerateInvocationWorkSummary workSummary{
              task.planNodeOrdinal, std::move(result.workSummary)};
          if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
                  completed, std::move(invocationRecord),
                  std::move(workSummary), std::move(result.ownerFeedback),
                  result.dispatched))
            return std::move(error);
          continue;
        }
        auto &generated =
            std::get<CompletedCandidateGeneratorResult>(result.outcome);
        GenerateInvocationRecord invocationRecord{
            task.planNodeOrdinal,
            std::move(task.inputs),
            std::move(task.binding),
            std::move(generated.outputBindings),
            std::move(generated.lineageEdges),
            std::nullopt};
        GenerateInvocationWorkSummary workSummary{
            task.planNodeOrdinal, std::move(result.workSummary)};
        if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
                completed, std::move(invocationRecord), std::move(workSummary),
                std::move(result.ownerFeedback), result.dispatched))
          return std::move(error);
      }
      if (blockingIncompleteness)
        return DsePlanExecutionOutcome{
            DsePlanExecutionBuilder::incompleteGenerate(
                blockingIncompleteness->first, blockingIncompleteness->second,
                std::move(completed), true)};
      nodeIndex += tasks.size() - 1;
      continue;
    }
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
      const CandidateGeneratorDescriptor *descriptor =
          generate->descriptorRef().descriptor();
      if (!descriptor)
        return invalid("Generate node lost its descriptor");
      std::vector<CandidateGeneratorOutputDemand> outputDemands =
          deriveOutputDemands(plan, nodeIndex, *descriptor);
      auto result =
          executor
              ? executor->executeGenerate(static_cast<std::uint64_t>(nodeIndex),
                                          inputs, outputDemands, *binding,
                                          store, blobs)
              : invokeCandidateGenerator(inputs, *binding, store, blobs,
                                         CandidateGeneratorInvocationView(
                                             executionControl, outputDemands));
      if (!result)
        return result.takeError();
      if (auto *incomplete = std::get_if<IncompleteCandidateGeneratorResult>(
              &result->outcome)) {
        GenerateInvocationRecord invocationRecord{
            static_cast<std::uint64_t>(nodeIndex),
            std::move(inputs),
            std::move(*binding),
            std::move(incomplete->retainedOutputBindings),
            std::move(incomplete->lineageEdges),
            incomplete->reason};
        GenerateInvocationWorkSummary workSummary{
            static_cast<std::uint64_t>(nodeIndex),
            std::move(result->workSummary)};
        if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
                completed, std::move(invocationRecord), std::move(workSummary),
                std::move(result->ownerFeedback), result->dispatched))
          return std::move(error);
        if (canContinue(incomplete->reason)) {
          if (!retainedIncompleteness)
            retainedIncompleteness =
                std::pair{static_cast<std::uint64_t>(nodeIndex),
                          DsePlanIncompleteReason{incomplete->reason}};
          continue;
        }
        return DsePlanExecutionOutcome{
            DsePlanExecutionBuilder::incompleteGenerate(
                static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
                std::move(completed), true)};
      }
      if (auto *proven =
              std::get_if<ProvenInfeasibleCandidateGeneratorResult>(
                  &result->outcome)) {
        GenerateInvocationRecord invocationRecord{
            static_cast<std::uint64_t>(nodeIndex),
            std::move(inputs),
            std::move(*binding),
            std::move(proven->outputBindings),
            {},
            std::nullopt,
            std::move(proven->proof)};
        GenerateInvocationWorkSummary workSummary{
            static_cast<std::uint64_t>(nodeIndex),
            std::move(result->workSummary)};
        if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
                completed, std::move(invocationRecord),
                std::move(workSummary), std::move(result->ownerFeedback),
                result->dispatched))
          return std::move(error);
        continue;
      }
      auto &generated =
          std::get<CompletedCandidateGeneratorResult>(result->outcome);
      GenerateInvocationRecord invocationRecord{
          static_cast<std::uint64_t>(nodeIndex),
          std::move(inputs),
          std::move(*binding),
          std::move(generated.outputBindings),
          std::move(generated.lineageEdges),
          std::nullopt};
      GenerateInvocationWorkSummary workSummary{
          static_cast<std::uint64_t>(nodeIndex),
          std::move(result->workSummary)};
      if (llvm::Error error = DsePlanExecutionBuilder::appendGenerate(
              completed, std::move(invocationRecord), std::move(workSummary),
              std::move(result->ownerFeedback), result->dispatched))
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
    if (executor) {
      if (llvm::Error error = executor->beginPromotion(
              static_cast<std::uint64_t>(nodeIndex), candidateSet->candidates(),
              promote.acquisitionBinding().evidenceObligations()))
        return std::move(error);
      if (executor->shouldStopBeforeDispatch())
        return DsePlanExecutionOutcome{
            DsePlanExecutionBuilder::incompletePromote(
                static_cast<std::uint64_t>(nodeIndex),
                IncompleteSelectionReason::MissingEvidence,
                std::move(completed), {{}, {}})};
    }

    if (const auto *topK = std::get_if<TopKSelection>(&promote.selection());
        topK && !promote.objectiveObligations().empty()) {
      if (!plan.objectiveProgram())
        return invalid("resolved TopK node lost its ObjectiveProgram");
      auto staged = executeStagedTopK(
          promote, *descriptor, inputs, *candidateSet,
          view.evidenceObligationTemplates(), *qualityGate,
          *plan.objectiveProgram(), *topK, store, blobs, executor);
      if (!staged)
        return staged.takeError();
      if (auto *incomplete = std::get_if<IncompleteStagedTopK>(&*staged)) {
        if (incomplete->executionStopped)
          return DsePlanExecutionOutcome{
              DsePlanExecutionBuilder::incompletePromote(
                  static_cast<std::uint64_t>(nodeIndex), incomplete->reason,
                  std::move(completed),
                  {std::move(incomplete->selected),
                   std::move(incomplete->evidence)},
                  std::move(incomplete->preferenceOrder), true)};
        DsePlanExecutionBuilder::appendPromote(
            completed,
            {std::move(incomplete->selected), std::move(incomplete->evidence)},
            std::move(incomplete->preferenceOrder));
        if (!retainedIncompleteness)
          retainedIncompleteness = std::pair{
              static_cast<std::uint64_t>(nodeIndex), incomplete->reason};
        continue;
      }
      auto &stagedCompleted = std::get<CompletedStagedTopK>(*staged);
      DsePlanExecutionBuilder::appendPromote(
          completed,
          {std::move(stagedCompleted.selected),
           std::move(stagedCompleted.evidence)},
          std::move(stagedCompleted.preferenceOrder));
      continue;
    }

    auto acquisition = invokePromotionAcquisition(
        inputs, promote.acquisitionBinding(),
        view.evidenceObligationTemplates(),
        {candidateSet->candidates(),
         promote.acquisitionBinding().evidenceObligations()},
        store, blobs, executor);
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
  if (retainedIncompleteness) {
    const bool generateNode =
        std::holds_alternative<CandidateGeneratorIncompleteReason>(
            retainedIncompleteness->second);
    return DsePlanExecutionOutcome{DsePlanExecutionBuilder::incompleteRetained(
        retainedIncompleteness->first, retainedIncompleteness->second,
        std::move(completed), generateNode)};
  }
  return DsePlanExecutionOutcome{std::move(completed)};
}

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDseConfigView &view, const ArtifactStore &store,
               const BlobStore &blobs) {
  return detail::executeDsePlanWithWorkExecutor(view, store, blobs, nullptr,
                                                {});
}

llvm::Expected<DsePlanExecutionOutcome>
executeDsePlan(const ResolvedDseConfigView &view, const ArtifactStore &store,
               const BlobStore &blobs,
               const ExecutionControlView &executionControl) {
  return detail::executeDsePlanWithWorkExecutor(view, store, blobs, nullptr,
                                                executionControl);
}

} // namespace loom::dse
