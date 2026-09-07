#include "PreMappingPlanExecution.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Config/ResolvedConfig.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/StructuredEvaluationAcquisition.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocation.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse::pre_mapping {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pre_mapping_exploration_invalid: " + message);
}

bool isCancellationReason(const DsePlanIncompleteReason &reason) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>)
          return value ==
                 CandidateGeneratorIncompleteReason::CancelledOrTimeout;
        else if constexpr (std::is_same_v<T,
                                          PromotionAcquisitionIncompleteReason>)
          return value ==
                 PromotionAcquisitionIncompleteReason::CancelledOrTimeout;
        else
          return value == IncompleteSelectionReason::CancelledOrTimeoutEvidence;
      },
      reason);
}

PreMappingCandidatePlanningDisposition
planningDispositionForIncomplete(const DsePlanIncompleteReason &reason) {
  return std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>) {
          switch (value) {
          case CandidateGeneratorIncompleteReason::SemanticLimitReached:
            return PreMappingCandidatePlanningDisposition::
                DataflowPromotionBudget;
          case CandidateGeneratorIncompleteReason::CancelledOrTimeout:
            return PreMappingCandidatePlanningDisposition::CancelledOrTimeout;
          case CandidateGeneratorIncompleteReason::Unsupported:
          case CandidateGeneratorIncompleteReason::ProviderUnavailable:
            return PreMappingCandidatePlanningDisposition::Unsupported;
          case CandidateGeneratorIncompleteReason::ProofNotEstablished:
          case CandidateGeneratorIncompleteReason::ExecutionFailed:
            return PreMappingCandidatePlanningDisposition::Unknown;
          }
        } else if constexpr (std::is_same_v<
                                 T, PromotionAcquisitionIncompleteReason>) {
          switch (value) {
          case PromotionAcquisitionIncompleteReason::SemanticWorkLimit:
            return PreMappingCandidatePlanningDisposition::
                DataflowPromotionBudget;
          case PromotionAcquisitionIncompleteReason::ProviderUnavailable:
          case PromotionAcquisitionIncompleteReason::Unsupported:
            return PreMappingCandidatePlanningDisposition::Unsupported;
          case PromotionAcquisitionIncompleteReason::CancelledOrTimeout:
            return PreMappingCandidatePlanningDisposition::CancelledOrTimeout;
          case PromotionAcquisitionIncompleteReason::ObjectiveUnavailable:
            return PreMappingCandidatePlanningDisposition::Unknown;
          }
        } else {
          return value == IncompleteSelectionReason::CancelledOrTimeoutEvidence
                     ? PreMappingCandidatePlanningDisposition::
                           CancelledOrTimeout
                     : PreMappingCandidatePlanningDisposition::Unknown;
        }
        llvm_unreachable("unknown pre-mapping incompleteness reason");
      },
      reason);
}

void emitOwnershipRejections(
    llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions) {
  for (const auto &disposition : dispositions) {
    const auto *rejection =
        std::get_if<StructuredOwnershipCandidateRejectionRecord>(
            &disposition.result);
    if (!rejection)
      continue;
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
          fields["failure_scope"] = "structured_ownership_candidate";
          fields["closure_status"] = "proven_infeasible";
          fields["rejection_kind"] =
              rejection->kind ==
                      frontend::SpatialOwnershipCandidateRejectionKind::
                          NonFinalizable
                  ? "non_finalizable"
                  : "exact_fabric_inadmissible";
          fields["diagnostic"] = rejection->message;
          if (rejection->memoryContract)
            fields["memory_contract"] = dataflow::memoryContractClassSpelling(
                *rejection->memoryContract);
          fields["scope_ordinal"] =
              disposition.coordinate.scope.selection.ordinal;
          if (disposition.coordinate.decision) {
            const auto &decision = *disposition.coordinate.decision;
            if (decision.addressProjection) {
              if (const auto *rootRelative =
                      std::get_if<frontend::RootRelativeAddressProjection>(
                          &*decision.addressProjection)) {
                fields["address_projection"] = "root_relative";
                fields["canonical_index_width"] =
                    rootRelative->canonicalIndexWidth;
              } else {
                fields["address_projection"] = "pointer_addressed";
              }
            }
            if (decision.forallOwnershipShape)
              fields["forall_ownership_shape"] =
                  *decision.forallOwnershipShape ==
                          frontend::ForallOwnershipShape::LogicalThreadDomain
                      ? "logical_thread_domain"
                      : "graph_parallel";
          }
        });
  }
}

void mergeReferences(std::vector<ArtifactRootReference> &destination,
                     llvm::ArrayRef<ArtifactRootReference> source) {
  destination.insert(destination.end(), source.begin(), source.end());
  llvm::sort(destination, artifactRootReferenceLess);
  destination.erase(std::unique(destination.begin(), destination.end()),
                    destination.end());
}

namespace {

enum class CompilerObligationKind : std::uint8_t { Analytic, Functional };

struct TaggedObligation final {
  CompilerObligationKind kind;
  EvidenceObligationTemplate obligation;
};

struct CompilerObligations final {
  std::vector<EvidenceObligationTemplate> templates;
  EvidenceObligationTemplateRef analytic{0};
  EvidenceObligationTemplateRef functional{0};
};

bool obligationLess(const TaggedObligation &lhs, const TaggedObligation &rhs) {
  return std::lexicographical_compare(lhs.obligation.canonicalBytes().begin(),
                                      lhs.obligation.canonicalBytes().end(),
                                      rhs.obligation.canonicalBytes().begin(),
                                      rhs.obligation.canonicalBytes().end());
}

llvm::Expected<CompilerObligations>
canonicalizeObligations(EvidenceObligationTemplate analytic,
                        EvidenceObligationTemplate functional) {
  std::vector<TaggedObligation> tagged;
  tagged.push_back({CompilerObligationKind::Analytic, std::move(analytic)});
  tagged.push_back({CompilerObligationKind::Functional, std::move(functional)});
  llvm::sort(tagged, obligationLess);
  if (!obligationLess(tagged[0], tagged[1]))
    return invalid("compiler Evidence obligations are not distinct");

  CompilerObligations result;
  result.templates.reserve(tagged.size());
  for (std::uint32_t ordinal = 0; ordinal != tagged.size(); ++ordinal) {
    if (tagged[ordinal].kind == CompilerObligationKind::Analytic)
      result.analytic = EvidenceObligationTemplateRef(ordinal);
    else
      result.functional = EvidenceObligationTemplateRef(ordinal);
    result.templates.push_back(std::move(tagged[ordinal].obligation));
  }
  return result;
}

bool authorizationLess(const ModelAuthorization &lhs,
                       const ModelAuthorization &rhs) {
  const auto left = std::make_tuple(lhs.descriptor.schemaVersion().major,
                                    lhs.descriptor.schemaVersion().minor,
                                    lhs.descriptor.modelKind().ordinal());
  const auto right = std::make_tuple(rhs.descriptor.schemaVersion().major,
                                     rhs.descriptor.schemaVersion().minor,
                                     rhs.descriptor.modelKind().ordinal());
  return left < right;
}

std::vector<ModelAuthorization>
modelAuthorizations(const CompilerObligations &obligations) {
  std::vector<ModelAuthorization> result;
  result.reserve(obligations.templates.size());
  for (const EvidenceObligationTemplate &obligation : obligations.templates)
    result.push_back({obligation.modelBinding().descriptorRef()});
  llvm::sort(result, authorizationLess);
  result.erase(std::unique(result.begin(), result.end(),
                           [](const ModelAuthorization &lhs,
                              const ModelAuthorization &rhs) {
                             return lhs.descriptor == rhs.descriptor;
                           }),
               result.end());
  return result;
}

llvm::Expected<std::vector<ArtifactRootReference>>
publishEvidence(llvm::ArrayRef<PromotionEvidence> evidence,
                const ArtifactStore &store) {
  std::vector<ArtifactRootReference> result;
  result.reserve(evidence.size());
  for (const PromotionEvidence &record : evidence) {
    auto reference =
        evaluation::publishEvaluationEvidence(record.evidence, store);
    if (!reference)
      return reference.takeError();
    result.push_back(std::move(*reference));
  }
  llvm::sort(result, artifactRootReferenceLess);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

struct BaselineMetric final {
  evaluation::MetricValue value;
  ArtifactRootReference evidence;
};

using BaselineMetricOutcome =
    std::variant<BaselineMetric, IncompletePreMappingExploration>;

llvm::Expected<BaselineMetricOutcome>
acquireBaselineMetric(const CompilerObligations &obligations,
                      evaluation::MetricRequestOrdinal metricRequest,
                      const ArtifactRootReference &source,
                      const ArtifactRootReference &fabric,
                      const ArtifactRootReference &workload,
                      const ArtifactRootReference &runtimeInput,
                      const ArtifactStore &store, const BlobStore &blobs) {
  auto acquisitionConfig =
      projectResolvedEvidenceObligationSetConfigView({obligations.analytic});
  if (!acquisitionConfig)
    return acquisitionConfig.takeError();
  auto binding = resolveStructuredEvaluationPromotionAcquisitionBinding(
      *acquisitionConfig);
  if (!binding)
    return binding.takeError();
  auto inputs = bindStructuredEvaluationPromotionInputs({source}, fabric,
                                                        workload, runtimeInput);
  if (!inputs)
    return inputs.takeError();

  const std::array<ArtifactRootReference, 1> candidates = {source};
  const std::array<EvidenceObligationTemplateRef, 1> selectedObligations = {
      obligations.analytic};
  auto acquired = invokePromotionAcquisition(
      *inputs, *binding, obligations.templates,
      {candidates, selectedObligations}, store, blobs);
  if (!acquired)
    return acquired.takeError();
  if (auto *incomplete =
          std::get_if<IncompletePromotionAcquisition>(&*acquired)) {
    auto retained = publishEvidence(incomplete->retainedEvidence, store);
    if (!retained)
      return retained.takeError();
    return BaselineMetricOutcome{IncompletePreMappingExploration{
        std::nullopt,
        DsePlanIncompleteReason{incomplete->reason},
        std::move(*retained),
        {}}};
  }

  auto &completed = std::get<CompletedPromotionAcquisition>(*acquired);
  if (completed.evidence.size() != 1)
    return invalid("baseline acquisition did not produce one Evidence record");
  const PromotionEvidence &record = completed.evidence.front();
  const evaluation::MetricRequest *request =
      record.request.resolve(metricRequest);
  const auto *outcome =
      std::get_if<evaluation::CompletedEvidence>(&record.evidence.outcome());
  if (!request || !outcome ||
      metricRequest.ordinal() >= outcome->metricResults.size())
    return invalid("baseline metric Evidence is not complete and positional");
  const auto *point = std::get_if<evaluation::PointObservation>(
      &outcome->metricResults[metricRequest.ordinal()].observation);
  if (!point)
    return BaselineMetricOutcome{IncompletePreMappingExploration{
        std::nullopt,
        DsePlanIncompleteReason{
            IncompleteSelectionReason::NonComparableEvidence},
        {},
        {}}};
  auto evidenceReference =
      evaluation::publishEvaluationEvidence(record.evidence, store);
  if (!evidenceReference)
    return evidenceReference.takeError();
  return BaselineMetricOutcome{
      BaselineMetric{point->value, std::move(*evidenceReference)}};
}

llvm::Expected<evaluation::FindingRequestOrdinal>
functionalMismatchOrdinal(const EvidenceObligationTemplate &obligation) {
  std::optional<evaluation::FindingRequestOrdinal> result;
  for (std::uint64_t ordinal = 0;
       ordinal != obligation.findingRequests().size(); ++ordinal) {
    if (obligation.findingRequests()[ordinal].query.kind !=
        evaluation::standard_findings::FunctionalMismatch)
      continue;
    if (result)
      return invalid(
          "functional obligation contains duplicate mismatch queries");
    result.emplace(ordinal);
  }
  if (!result)
    return invalid("functional obligation omits functional_mismatch");
  return *result;
}

llvm::Expected<ResolvedObjectiveCatalogs> compilerObjectives(
    const CompilerObligations &obligations,
    evaluation::MetricRequestOrdinal metricRequest,
    ResolvedObjectiveDirection direction,
    llvm::Expected<std::int64_t> (*quantum)(evaluation::MetricKind)) {
  if (direction != ResolvedObjectiveDirection::Minimize &&
      direction != ResolvedObjectiveDirection::Maximize)
    return invalid("compiler objective direction is invalid");
  if (metricRequest.ordinal() >=
      obligations.templates[obligations.analytic.ordinal()]
          .metricRequests()
          .size())
    return invalid("compiler objective metric request is out of range");
  const evaluation::MetricKind metric =
      obligations.templates[obligations.analytic.ordinal()]
          .metricRequests()[metricRequest.ordinal()]
          .query.metric;
  auto exponent = quantum(metric);
  if (!exponent)
    return exponent.takeError();

  ResolvedObjectiveCatalogs catalogs;
  catalogs.dimensions.push_back(ResolvedObjectiveDimension{
      ResolvedEvaluationMetricObjectiveSource{obligations.analytic.ordinal(),
                                              metricRequest.ordinal()},
      direction == ResolvedObjectiveDirection::Minimize
          ? ResolvedObjectiveDirection::Minimize
          : ResolvedObjectiveDirection::Maximize,
      resolvedObjectiveDecimal(0, 0), resolvedObjectiveDecimal(1, *exponent), 0,
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())});
  catalogs.weightedLevels.push_back({{{0, 1}}});
  catalogs.totalOrderings.push_back({{0}});
  return catalogs;
}

llvm::Expected<QualityGatePolicy>
ownershipQualityGate(const CompilerObligations &obligations,
                     const StructuredOwnershipExplorationOptions &options,
                     const std::optional<BaselineMetric> &baseline) {
  std::vector<QualityGateClause> clauses;
  if (options.selectionMode ==
      StructuredOwnershipSelectionMode::BenefitQualified) {
    if (!baseline)
      return invalid("benefit-qualified selection has no baseline metric");
    clauses.push_back({{MetricGate{
        obligations.analytic.ordinal(), options.selection.metricRequest,
        options.selection.direction == ResolvedObjectiveDirection::Minimize
            ? MetricGateComparator::LT
            : MetricGateComparator::GT,
        baseline->value}}});
  }
  auto mismatch = functionalMismatchOrdinal(
      obligations.templates[obligations.functional.ordinal()]);
  if (!mismatch)
    return mismatch.takeError();
  clauses.push_back({{FindingGate{obligations.functional.ordinal(), *mismatch,
                                  RequiredFindingState::Absent}}});
  return QualityGatePolicy::get(std::move(clauses));
}

llvm::Expected<QualityGatePolicy> ownershipAnalyticQualityGate(
    const CompilerObligations &obligations,
    const StructuredOwnershipExplorationOptions &options,
    const std::optional<BaselineMetric> &baseline) {
  std::vector<QualityGateClause> clauses;
  if (options.selectionMode ==
      StructuredOwnershipSelectionMode::BenefitQualified) {
    if (!baseline)
      return invalid("benefit-qualified selection has no baseline metric");
    clauses.push_back({{MetricGate{
        obligations.analytic.ordinal(), options.selection.metricRequest,
        options.selection.direction == ResolvedObjectiveDirection::Minimize
            ? MetricGateComparator::LT
            : MetricGateComparator::GT,
        baseline->value}}});
  }
  return QualityGatePolicy::get(std::move(clauses));
}

llvm::Expected<QualityGatePolicy>
dataflowQualityGate(const CompilerObligations &obligations) {
  auto mismatch = functionalMismatchOrdinal(
      obligations.templates[obligations.functional.ordinal()]);
  if (!mismatch)
    return mismatch.takeError();
  return QualityGatePolicy::get(
      {{{FindingGate{obligations.functional.ordinal(), *mismatch,
                     RequiredFindingState::Absent}}}});
}

llvm::Expected<std::vector<ArtifactRootReference>>
selectedPreferenceOrder(const CompletedDsePlanExecution &execution,
                        PlanOutputRef selectedOutput) {
  const llvm::ArrayRef<ArtifactRootReference> canonical =
      execution.resolve(selectedOutput);
  const llvm::ArrayRef<ArtifactRootReference> preferred =
      execution.resolvePreferenceOrder(selectedOutput);
  if (canonical.empty())
    return std::vector<ArtifactRootReference>{};
  if (preferred.size() != canonical.size())
    return invalid("objective preference order changed the selected set size");

  std::vector<ArtifactRootReference> checked(preferred.begin(),
                                             preferred.end());
  std::vector<ArtifactRootReference> canonicalized = checked;
  llvm::sort(canonicalized, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalized.begin(), canonicalized.end()) !=
          canonicalized.end() ||
      !std::equal(canonicalized.begin(), canonicalized.end(), canonical.begin(),
                  canonical.end()))
    return invalid("objective preference order changed the selected set");
  return checked;
}

std::vector<ArtifactRootReference>
retainedEvidence(const IncompleteDsePlanExecution &incomplete,
                 llvm::ArrayRef<ArtifactRootReference> baselineEvidence) {
  std::vector<ArtifactRootReference> result(baselineEvidence.begin(),
                                            baselineEvidence.end());
  for (std::size_t ordinal = 0; ordinal < incomplete.retainedOutputCount();
       ++ordinal)
    for (const ArtifactRootReference &reference :
         incomplete.retainedOutput(ordinal))
      if (reference.schemaIdentity ==
              evaluation::EvaluationEvidence::artifactSchema.identity &&
          reference.schemaVersion ==
              evaluation::EvaluationEvidence::artifactSchema.version)
        result.push_back(reference);
  llvm::sort(result, artifactRootReferenceLess);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

llvm::StringRef spelling(StructuredScheduleGenerationIntent intent) {
  switch (intent) {
  case StructuredScheduleGenerationIntent::Balanced:
    return "balanced";
  case StructuredScheduleGenerationIntent::RequireLogicalThreadDomain:
    return "require_logical_thread_domain";
  case StructuredScheduleGenerationIntent::ForbidLogicalThreadDomain:
    return "forbid_logical_thread_domain";
  }
  llvm_unreachable("unknown Structured Schedule generation intent");
}

llvm::Expected<std::uint64_t>
consumedCompilerMaterializations(const CompletedDsePlanExecution &execution) {
  if (execution.generateInvocations().size() !=
      execution.generateWorkSummaries().size())
    return invalid("Generate invocation and work accounting widths differ");
  std::uint64_t total = 0;
  for (auto indexed : llvm::enumerate(execution.generateInvocations())) {
    const GenerateInvocationRecord &invocation = indexed.value();
    const GenerateInvocationWorkSummary &summary =
        execution.generateWorkSummaries()[indexed.index()];
    if (summary.planNodeOrdinal != invocation.planNodeOrdinal)
      return invalid("Generate invocation and work accounting order differs");
    const CandidateGeneratorKind kind =
        invocation.generatorBinding.descriptorRef().kind();
    std::uint32_t decisionUnit = 0;
    if (kind == structuredOwnershipCandidateGeneratorKind ||
        kind == structuredScheduleCandidateGeneratorKind ||
        kind == structuredMemoryCommunicationCandidateGeneratorKind)
      decisionUnit = 1;
    else if (kind != structuredExecutionShapeCandidateGeneratorKind &&
             kind != structuredSpecialMathAccuracyCandidateGeneratorKind)
      continue;
    const std::uint32_t endUnit =
        kind == structuredSpecialMathAccuracyCandidateGeneratorKind
            ? static_cast<std::uint32_t>(summary.units.size())
            : decisionUnit + 1;
    for (std::uint32_t unit = decisionUnit; unit != endUnit; ++unit) {
      if (unit >= summary.units.size() ||
          summary.units[unit].unit.ordinal() != unit)
        return invalid("compiler materialization work unit is not canonical");
      if (summary.units[unit].consumed >
          std::numeric_limits<std::uint64_t>::max() - total)
        return invalid("compiler materialization accounting overflows");
      total += summary.units[unit].consumed;
    }
  }
  return total;
}

} // namespace

llvm::Expected<OwnershipSelectionOutcome> exploreOwnershipCandidates(
    const frontend::StructuredCompilation &generationParent,
    const ArtifactRootReference &generationParentReference,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const ArtifactRootReference &sourceReference,
    const sim::CanonicalSimulationWorkload &workload,
    const ArtifactRootReference &workloadReference,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const ArtifactRootReference &runtimeInputReference,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    StructuredOwnershipGenerationIntent ownershipIntent,
    StructuredScheduleGenerationIntent scheduleIntent,
    std::uint64_t ownershipMaterializationAttemptLimit,
    std::uint64_t specialMathMaterializationAttemptLimit,
    std::uint64_t expansionLimit, bool generationParentFunctionallyVerified,
    bool requireFunctionalReplay, ExecutionControlView executionControl,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation) {
  if (ownershipMaterializationAttemptLimit == 0 ||
      specialMathMaterializationAttemptLimit == 0 || expansionLimit == 0 ||
      options.selection.k == 0 || options.selection.k > expansionLimit)
    return invalid("ownership beam and expansion bounds are inconsistent");
  // Keep the producer frontier bounded by the admitted expansion, while the
  // Promote node's TopK remains the smaller survivor width in semantic mode.
  // This is what makes analytic ranking a real pre-Mapping funnel instead of
  // replaying every generated candidate.
  const std::uint64_t layerWidth = expansionLimit;
  auto invocation = std::make_unique<StructuredOwnershipInvocation>(
      generationParent.structuredProgram, sourceProgram, workload, runtimeInput,
      fabric, config, options.lowering, options.candidateWorkerCount,
      options.functionalReplayLimits, generationParent.sourceProvenance,
      sharedEvaluation, executionControl, generationParentFunctionallyVerified);

  StructuredOwnershipInvocationScope invocationScope(*invocation);
  if (llvm::Error error = invocation->prepareInputs(
          generationParentReference, sourceReference, workloadReference,
          runtimeInputReference, artifactStore))
    return std::move(error);

  auto analytic = prepareStructuredFabricAnalyticEvidenceObligationTemplate(
      generationParentReference, fabric.reference(), workloadReference,
      runtimeInputReference, config, artifactStore, blobStore);
  if (!analytic)
    return analytic.takeError();
  auto functional =
      prepareStructuredProgramFunctionalEvidenceObligationTemplate(
          generationParentReference, workloadReference, runtimeInputReference,
          config, artifactStore, blobStore);
  if (!functional)
    return functional.takeError();
  auto obligations =
      canonicalizeObligations(std::move(*analytic), std::move(*functional));
  if (!obligations)
    return obligations.takeError();

  std::optional<BaselineMetric> baseline;
  std::vector<ArtifactRootReference> baselineEvidence;
  if (options.selectionMode ==
      StructuredOwnershipSelectionMode::BenefitQualified) {
    auto acquired = acquireBaselineMetric(
        *obligations, options.selection.metricRequest,
        generationParentReference, fabric.reference(), workloadReference,
        runtimeInputReference, artifactStore, blobStore);
    if (!acquired)
      return acquired.takeError();
    if (auto *incomplete =
            std::get_if<IncompletePreMappingExploration>(&*acquired))
      return OwnershipSelectionOutcome{std::move(*incomplete)};
    baseline.emplace(std::get<BaselineMetric>(std::move(*acquired)));
    baselineEvidence.push_back(baseline->evidence);
  }

  auto objectives = compilerObjectives(
      *obligations, options.selection.metricRequest,
      options.selection.direction,
      evaluation::models::structuredFabricAnalyticMetricQuantumBase10Exponent);
  if (!objectives)
    return objectives.takeError();
  auto gate = ownershipQualityGate(*obligations, options, baseline);
  if (!gate)
    return gate.takeError();
  auto analyticGate =
      ownershipAnalyticQualityGate(*obligations, options, baseline);
  if (!analyticGate)
    return analyticGate.takeError();
  auto generatorConfig = projectResolvedStructuredOwnershipGeneratorConfigView(
      config, options.protocolCallableRoots, ownershipIntent,
      ownershipMaterializationAttemptLimit);
  if (!generatorConfig)
    return generatorConfig.takeError();
  auto finalAcquisitionConfig = projectResolvedEvidenceObligationSetConfigView(
      {obligations->analytic, obligations->functional});
  if (!finalAcquisitionConfig)
    return finalAcquisitionConfig.takeError();
  auto analyticAcquisitionConfig =
      projectResolvedEvidenceObligationSetConfigView({obligations->analytic});
  if (!analyticAcquisitionConfig)
    return analyticAcquisitionConfig.takeError();
  auto scheduleConfig = projectResolvedStructuredScheduleGeneratorConfigView(
      config, scheduleIntent, expansionLimit);
  if (!scheduleConfig)
    return scheduleConfig.takeError();
  auto executionShapeConfig =
      projectResolvedStructuredExecutionShapeGeneratorConfigView();
  if (!executionShapeConfig)
    return executionShapeConfig.takeError();
  auto specialMathAccuracyConfig =
      projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
          specialMathMaterializationAttemptLimit);
  if (!specialMathAccuracyConfig)
    return specialMathAccuracyConfig.takeError();
  auto memoryCommunicationConfig =
      projectResolvedStructuredMemoryCommunicationGeneratorConfigView(config);
  if (!memoryCommunicationConfig)
    return memoryCommunicationConfig.takeError();

  ResolvedConfig planConfig = config;
  planConfig.dse.modelAuthorizations = modelAuthorizations(*obligations);
  planConfig.dse.evidenceObligationTemplates = obligations->templates;
  planConfig.dse.objectiveCatalogs = std::move(*objectives);
  planConfig.dse.qualityGatePolicies = {*analyticGate, *gate};
  planConfig.dse.planNodes = {
      GeneratePlanNodeDefinition{
          structuredOwnershipCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{generationParentReference}},
           ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{workloadReference}},
           ExactPlanArtifacts{{runtimeInputReference}}},
          generatorConfig->canonicalViewBytes().vec(),
          generatorConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredExecutionShapeCandidateGeneratorDescriptor().reference(),
          {BoundedPlanOutputJoin{
              {PlanOutputRef{0, 1}}, layerWidth, expansionLimit}},
          executionShapeConfig->canonicalViewBytes().vec(),
          executionShapeConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredScheduleCandidateGeneratorDescriptor().reference(),
          {BoundedPlanOutputJoin{
               {PlanOutputRef{1, 0}}, layerWidth, expansionLimit},
           ExactPlanArtifacts{{fabric.reference()}}},
          scheduleConfig->canonicalViewBytes().vec(),
          scheduleConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredMemoryCommunicationCandidateGeneratorDescriptor()
              .reference(),
          {BoundedPlanOutputJoin{
              {PlanOutputRef{2, 0}}, layerWidth, expansionLimit}},
          memoryCommunicationConfig->canonicalViewBytes().vec(),
          memoryCommunicationConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredSpecialMathAccuracyCandidateGeneratorDescriptor()
              .reference(),
          {BoundedPlanOutputJoin{
               {PlanOutputRef{3, 0}}, layerWidth, expansionLimit},
           ExactPlanArtifacts{{fabric.reference()}}},
          specialMathAccuracyConfig->canonicalViewBytes().vec(),
          specialMathAccuracyConfig->digest()},
      PromotePlanNodeDefinition{
          structuredEvaluationPromotionAcquisitionDescriptor().reference(),
          {BoundedPlanOutputJoin{
               {PlanOutputRef{4, 0}}, layerWidth, expansionLimit},
           ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{workloadReference}},
           ExactPlanArtifacts{{runtimeInputReference}}},
          analyticAcquisitionConfig->canonicalViewBytes().vec(),
          analyticAcquisitionConfig->digest(),
          QualityGatePolicyRef(0),
          TopKSelection{0, options.selection.k},
          PromotePurpose::CandidateSelection},
      PromotePlanNodeDefinition{
          structuredEvaluationPromotionAcquisitionDescriptor().reference(),
          {BoundedPlanOutputJoin{
               {PlanOutputRef{5, 0}}, options.selection.k, options.selection.k},
           ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{workloadReference}},
           ExactPlanArtifacts{{runtimeInputReference}}},
          finalAcquisitionConfig->canonicalViewBytes().vec(),
          finalAcquisitionConfig->digest(),
          QualityGatePolicyRef(1),
          TopKSelection{0, options.selection.k},
          PromotePurpose::CandidateSelection}};
  const std::uint64_t selectionNode = requireFunctionalReplay ? 6 : 5;
  if (!requireFunctionalReplay) {
    planConfig.dse.qualityGatePolicies = {*analyticGate};
    planConfig.dse.planNodes.pop_back();
  }
  auto view = projectResolvedDseConfigView(planConfig);
  if (!view)
    return view.takeError();
  auto executed =
      executeDsePlan(*view, artifactStore, blobStore, executionControl);
  if (!executed)
    return executed.takeError();
  const CompletedDsePlanExecution *selectionExecution =
      std::get_if<CompletedDsePlanExecution>(&*executed);
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
  if (auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&*executed)) {
    selectionExecution = &incomplete->availableExecution();
    if (incomplete->executionStopped() ||
        !selectionExecution->hasOutput({selectionNode, 0}) ||
        selectionExecution->resolve({selectionNode, 0}).empty()) {
      const std::uint64_t nodeOrdinal = incomplete->nodeOrdinal();
      const DsePlanIncompleteReason reason = incomplete->reason();
      auto programMaterializations =
          consumedCompilerMaterializations(*selectionExecution);
      if (!programMaterializations)
        return programMaterializations.takeError();
      const std::uint64_t evaluationCandidateCount =
          selectionExecution->hasOutput({4, 0})
              ? std::min<std::uint64_t>(
                    selectionExecution->resolve({4, 0}).size(), layerWidth)
              : 0;
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::DerivedContext,
          [&](llvm::json::Object &fields) {
            const auto count = [&](std::uint64_t node, std::uint32_t slot) {
              const PlanOutputRef output{node, slot};
              return selectionExecution->hasOutput(output)
                         ? selectionExecution->resolve(output).size()
                         : 0;
            };
            fields["context_kind"] = "structured_schedule_intent";
            fields["generation_intent"] = spelling(scheduleIntent);
            fields["incomplete_node_ordinal"] = nodeOrdinal;
            fields["incomplete_reason"] = dse::toString(reason);
            fields["ownership_count"] = count(0, 1);
            fields["execution_shape_count"] = count(1, 0);
            fields["schedule_count"] = count(2, 0);
            fields["memory_communication_count"] = count(3, 0);
            fields["special_math_count"] = count(4, 0);
            fields["analytic_selected_count"] = count(5, 0);
            fields["selected_count"] = count(selectionNode, 0);
            fields["logical_thread_domain_count"] = 0;
          });
      auto evidence = retainedEvidence(*incomplete, baselineEvidence);
      const StructuredOwnershipEvaluationTiming evaluationTiming =
          invocation->evaluationTiming();
      std::vector<DsePlanGenerateInvocationRecords> generateInvocations;
      generateInvocations.push_back(
          takeDsePlanGenerateInvocationRecords(std::move(*executed)));
      IncompletePreMappingExploration result{
          nodeOrdinal,
          reason,
          std::move(evidence),
          std::move(generateInvocations),
          *programMaterializations,
          (baseline ? 1 : 0) + evaluationCandidateCount,
          evaluationTiming.functionalReplayCalls,
          evaluationTiming};
      result.resolvedDseConfigViewDigest = view->digest();
      return OwnershipSelectionOutcome{std::move(result)};
    }
    retainedIncompleteness.emplace(RetainedDsePlanIncompleteness{
        selectionExecution->resolvedDseConfigViewDigest(),
        incomplete->nodeOrdinal(), incomplete->reason()});
  }

  if (!selectionExecution)
    return invalid("DSE plan outcome has no available execution");
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        const auto count = [&](std::uint64_t node, std::uint32_t slot) {
          const PlanOutputRef output{node, slot};
          return selectionExecution->hasOutput(output)
                     ? selectionExecution->resolve(output).size()
                     : 0;
        };
        fields["context_kind"] = "structured_compiler_frontier";
        fields["ownership_count"] = count(0, 1);
        fields["execution_shape_count"] = count(1, 0);
        fields["schedule_count"] = count(2, 0);
        fields["memory_communication_count"] = count(3, 0);
        fields["special_math_count"] = count(4, 0);
        fields["analytic_selected_count"] = count(5, 0);
        fields["selected_count"] = count(selectionNode, 0);
        fields["evidence_count"] = count(selectionNode, 1);
      });
  std::vector<ArtifactRootReference> selected(
      selectionExecution->resolve({selectionNode, 0}).begin(),
      selectionExecution->resolve({selectionNode, 0}).end());
  auto preferenceOrder =
      selectedPreferenceOrder(*selectionExecution, {selectionNode, 0});
  if (!preferenceOrder)
    return preferenceOrder.takeError();
  if (requireFunctionalReplay)
    for (const ArtifactRootReference &reference : *preferenceOrder)
      if (llvm::Error error =
              invocation->ensureSelectedCandidateFunctionalReplay(
                  reference, artifactStore))
        return std::move(error);
  std::uint64_t logicalThreadDomainCount = 0;
  for (const ArtifactRootReference &reference : *preferenceOrder) {
    auto hasLogicalThreadDomain =
        invocation->selectedCandidateHasLogicalThreadDomain(reference);
    if (!hasLogicalThreadDomain)
      return hasLogicalThreadDomain.takeError();
    logicalThreadDomainCount += *hasLogicalThreadDomain ? 1 : 0;
  }
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "structured_schedule_intent";
        fields["generation_intent"] = spelling(scheduleIntent);
        fields["selected_count"] = preferenceOrder->size();
        fields["logical_thread_domain_count"] = logicalThreadDomainCount;
      });
  std::vector<ArtifactRootReference> evidence = std::move(baselineEvidence);
  mergeReferences(evidence, selectionExecution->resolve({selectionNode, 1}));
  std::vector<StructuredOwnershipCandidateDisposition> dispositions(
      invocation->dispositions().begin(), invocation->dispositions().end());
  std::vector<StructuredOwnershipFinalizationRejection> finalizationRejections(
      invocation->finalizationRejections().begin(),
      invocation->finalizationRejections().end());
  auto programMaterializations =
      consumedCompilerMaterializations(*selectionExecution);
  if (!programMaterializations)
    return programMaterializations.takeError();
  const std::uint64_t evaluationCandidateCount = std::min<std::uint64_t>(
      selectionExecution->resolve({4, 0}).size(), layerWidth);
  const StructuredOwnershipEvaluationTiming evaluationTiming =
      invocation->evaluationTiming();
  return OwnershipSelectionOutcome{CompletedOwnershipSelection{
      std::move(invocation), std::move(selected), std::move(*preferenceOrder),
      std::move(evidence), std::move(dispositions),
      std::move(finalizationRejections),
      takeDsePlanGenerateInvocationRecords(std::move(*executed)),
      std::move(retainedIncompleteness), *programMaterializations,
      (baseline ? 1 : 0) + evaluationCandidateCount,
      evaluationTiming.functionalReplayCalls, evaluationTiming}};
}

llvm::Expected<DataflowSelectionOutcome> exploreDataflowCandidates(
    const ArtifactRootReference &d0,
    const ArtifactRootReference &structuredParent,
    const fabric::FinalizedFabricRoot &fabric,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const StructuredOwnershipTopKSelection &selection,
    StructuredOwnershipSelectionMode selectionMode,
    bool allowRewriteExploration, ExecutionControlView executionControl,
    const ArtifactStore &store, const BlobStore &blobs) {
  bool requiresRewriteExploration = allowRewriteExploration;
  if (!allowRewriteExploration &&
      selectionMode == StructuredOwnershipSelectionMode::SemanticConformance) {
    auto initial = dataflow::importCanonicalDataflow(d0, store);
    if (!initial)
      return initial.takeError();
    frontend::FabricCapabilityIndex capabilities(fabric.view());
    auto missing = capabilities.firstInadmissibleActor(*initial);
    if (!missing)
      return missing.takeError();
    requiresRewriteExploration = missing->has_value();
  }
  auto analytic =
      prepareCanonicalDataflowFabricAnalyticEvidenceObligationTemplate(
          d0, fabric.reference(), config, store, blobs);
  if (!analytic)
    return analytic.takeError();
  auto functional =
      prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
          d0, structuredParent, workload, runtimeInput, config, store, blobs);
  if (!functional)
    return functional.takeError();
  auto obligations =
      canonicalizeObligations(std::move(*analytic), std::move(*functional));
  if (!obligations)
    return obligations.takeError();
  auto objectives = compilerObjectives(
      *obligations, selection.metricRequest, selection.direction,
      evaluation::models::
          canonicalDataflowFabricAnalyticMetricQuantumBase10Exponent);
  if (!objectives)
    return objectives.takeError();
  auto gate = dataflowQualityGate(*obligations);
  if (!gate)
    return gate.takeError();
  auto generatorConfig =
      projectResolvedDataflowRewriteGeneratorConfigView(config);
  if (!generatorConfig)
    return generatorConfig.takeError();
  auto acquisitionConfig = projectResolvedEvidenceObligationSetConfigView(
      {obligations->analytic, obligations->functional});
  if (!acquisitionConfig)
    return acquisitionConfig.takeError();

  ResolvedConfig planConfig = config;
  planConfig.dse.modelAuthorizations = modelAuthorizations(*obligations);
  planConfig.dse.evidenceObligationTemplates = obligations->templates;
  planConfig.dse.objectiveCatalogs = std::move(*objectives);
  planConfig.dse.qualityGatePolicies = {*gate};
  std::uint64_t selectionNode = 0;
  if (requiresRewriteExploration) {
    planConfig.dse.planNodes = {
        GeneratePlanNodeDefinition{
            dataflowRewriteCandidateGeneratorDescriptor().reference(),
            {ExactPlanArtifacts{{d0}},
             ExactPlanArtifacts{{fabric.reference()}}},
            generatorConfig->canonicalViewBytes().vec(),
            generatorConfig->digest()},
        PromotePlanNodeDefinition{
            dataflowEvaluationPromotionAcquisitionDescriptor().reference(),
            {BoundedPlanOutputJoin{{PlanOutputRef{0, 0}}, selection.k},
             ExactPlanArtifacts{{structuredParent}},
             ExactPlanArtifacts{{fabric.reference()}},
             ExactPlanArtifacts{{workload}},
             ExactPlanArtifacts{{runtimeInput}}},
            acquisitionConfig->canonicalViewBytes().vec(),
            acquisitionConfig->digest(),
            QualityGatePolicyRef(0),
            TopKSelection{0, selection.k},
            PromotePurpose::CandidateSelection}};
    selectionNode = 1;
  } else {
    planConfig.dse.planNodes = {PromotePlanNodeDefinition{
        dataflowEvaluationPromotionAcquisitionDescriptor().reference(),
        {ExactPlanArtifacts{{d0}}, ExactPlanArtifacts{{structuredParent}},
         ExactPlanArtifacts{{fabric.reference()}},
         ExactPlanArtifacts{{workload}}, ExactPlanArtifacts{{runtimeInput}}},
        acquisitionConfig->canonicalViewBytes().vec(),
        acquisitionConfig->digest(),
        QualityGatePolicyRef(0),
        TopKSelection{0, selection.k},
        PromotePurpose::CandidateSelection}};
  }
  auto view = projectResolvedDseConfigView(planConfig);
  if (!view)
    return view.takeError();
  auto executed = executeDsePlan(*view, store, blobs, executionControl);
  if (!executed)
    return executed.takeError();
  const CompletedDsePlanExecution *selectionExecution =
      std::get_if<CompletedDsePlanExecution>(&*executed);
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
  if (auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&*executed)) {
    selectionExecution = &incomplete->availableExecution();
    if (incomplete->executionStopped() ||
        selectionExecution->resolve({selectionNode, 0}).empty()) {
      const std::uint64_t nodeOrdinal = incomplete->nodeOrdinal();
      const DsePlanIncompleteReason reason = incomplete->reason();
      auto evidence = retainedEvidence(*incomplete, {});
      std::vector<DsePlanGenerateInvocationRecords> generateInvocations;
      generateInvocations.push_back(
          takeDsePlanGenerateInvocationRecords(std::move(*executed)));
      IncompletePreMappingExploration result{nodeOrdinal, reason,
                                             std::move(evidence),
                                             std::move(generateInvocations)};
      result.resolvedDseConfigViewDigest = view->digest();
      return DataflowSelectionOutcome{std::move(result)};
    }
    retainedIncompleteness.emplace(RetainedDsePlanIncompleteness{
        selectionExecution->resolvedDseConfigViewDigest(),
        incomplete->nodeOrdinal(), incomplete->reason()});
  }
  if (!selectionExecution)
    return invalid("DSE plan outcome has no available execution");
  std::vector<ArtifactRootReference> selected =
      selectionExecution->resolve({selectionNode, 0}).vec();
  auto preferenceOrder =
      selectedPreferenceOrder(*selectionExecution, {selectionNode, 0});
  if (!preferenceOrder)
    return preferenceOrder.takeError();
  std::vector<ArtifactRootReference> evidence =
      selectionExecution->resolve({selectionNode, 1}).vec();
  return DataflowSelectionOutcome{CompletedDataflowSelection{
      std::move(selected), std::move(*preferenceOrder), std::move(evidence),
      takeDsePlanGenerateInvocationRecords(std::move(*executed)),
      std::move(retainedIncompleteness)}};
}

} // namespace loom::dse::pre_mapping
