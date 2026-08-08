#include "DSE/PreMappingExploration.h"

#include "Common/ArtifactStore.h"
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
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pre_mapping_exploration_invalid: " + message);
}

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

void mergeReferences(std::vector<ArtifactRootReference> &destination,
                     llvm::ArrayRef<ArtifactRootReference> source) {
  destination.insert(destination.end(), source.begin(), source.end());
  llvm::sort(destination, artifactRootReferenceLess);
  destination.erase(std::unique(destination.begin(), destination.end()),
                    destination.end());
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

struct CompletedDataflowSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> evidence;
  DsePlanGenerateInvocationRecords generateInvocations;
};

using DataflowSelectionOutcome =
    std::variant<CompletedDataflowSelection, IncompletePreMappingExploration>;

llvm::Expected<DataflowSelectionOutcome> exploreDataflowCandidates(
    const ArtifactRootReference &d0,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const StructuredOwnershipTopKSelection &selection,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto analytic =
      prepareCanonicalDataflowFabricAnalyticEvidenceObligationTemplate(
          d0, fabric, config, store);
  if (!analytic)
    return analytic.takeError();
  auto functional =
      prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
          d0, structuredParent, workload, runtimeInput, config, store);
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
  planConfig.dse.planNodes = {
      GeneratePlanNodeDefinition{
          dataflowRewriteCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{d0}}, ExactPlanArtifacts{{fabric}}},
          generatorConfig->canonicalViewBytes().vec(),
          generatorConfig->digest()},
      PromotePlanNodeDefinition{
          dataflowEvaluationPromotionAcquisitionDescriptor().reference(),
          {PlanOutputRef{0, 0}, ExactPlanArtifacts{{structuredParent}},
           ExactPlanArtifacts{{fabric}}, ExactPlanArtifacts{{workload}},
           ExactPlanArtifacts{{runtimeInput}}},
          acquisitionConfig->canonicalViewBytes().vec(),
          acquisitionConfig->digest(),
          QualityGatePolicyRef(0),
          TopKSelection{0, selection.k},
          PromotePurpose::CandidateSelection}};
  auto view = projectResolvedDseConfigView(planConfig);
  if (!view)
    return view.takeError();
  auto executed = executeDsePlan(*view, store, blobs);
  if (!executed)
    return executed.takeError();
  if (auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&*executed)) {
    const std::uint64_t nodeOrdinal = incomplete->nodeOrdinal();
    const DsePlanIncompleteReason reason = incomplete->reason();
    auto evidence = retainedEvidence(*incomplete, {});
    std::vector<DsePlanGenerateInvocationRecords> generateInvocations;
    generateInvocations.push_back(
        takeDsePlanGenerateInvocationRecords(std::move(*executed)));
    return DataflowSelectionOutcome{IncompletePreMappingExploration{
        nodeOrdinal, reason, std::move(evidence),
        std::move(generateInvocations)}};
  }
  auto &completed = std::get<CompletedDsePlanExecution>(*executed);
  std::vector<ArtifactRootReference> selected = completed.resolve({1, 0}).vec();
  std::vector<ArtifactRootReference> evidence = completed.resolve({1, 1}).vec();
  return DataflowSelectionOutcome{CompletedDataflowSelection{
      std::move(selected), std::move(evidence),
      takeDsePlanGenerateInvocationRecords(std::move(*executed))}};
}

} // namespace

llvm::Expected<PreMappingExplorationOutcome>
exploreStructuredCompilationToPreMapping(
    frontend::StructuredCompilation compilation,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const PreMappingExplorationOptions &options,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (compilation.fabric != fabric.reference())
    return invalid("Structured compilation and Fabric references differ");
  if (options.ownership.selection.k == 0)
    return invalid("ownership TopK requires positive k");
  if (llvm::Error error = registerStructuredOwnershipCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerStructuredExecutionShapeCandidateGenerator())
    return std::move(error);
  if (llvm::Error error =
          registerStructuredSpecialMathAccuracyCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerStructuredScheduleCandidateGenerator())
    return std::move(error);
  if (llvm::Error error =
          registerStructuredMemoryCommunicationCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  if (llvm::Error error = registerDataflowRewriteCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerDataflowEvaluationPromotionAcquisition())
    return std::move(error);

  auto sourceReference = frontend::publishStructuredProgram(
      compilation.structuredProgram, artifactStore);
  if (!sourceReference)
    return sourceReference.takeError();
  auto workloadReference =
      sim::publishSimulationWorkload(workload, artifactStore);
  if (!workloadReference)
    return workloadReference.takeError();
  auto runtimeInputReference =
      sim::publishSimulationRuntimeInput(runtimeInput, artifactStore);
  if (!runtimeInputReference)
    return runtimeInputReference.takeError();

  StructuredOwnershipInvocation invocation(
      compilation.structuredProgram, workload, runtimeInput, fabric, config,
      options.ownership.lowering, options.ownership.candidateWorkerCount,
      options.ownership.functionalReplayLimits, compilation.sourceProvenance);
  StructuredOwnershipInvocationScope invocationScope(invocation);
  if (llvm::Error error =
          invocation.prepareSource(*sourceReference, *workloadReference,
                                   *runtimeInputReference, artifactStore))
    return std::move(error);

  auto analytic = prepareStructuredFabricAnalyticEvidenceObligationTemplate(
      *sourceReference, fabric.reference(), *workloadReference,
      *runtimeInputReference, config, artifactStore);
  if (!analytic)
    return analytic.takeError();
  auto functional =
      prepareStructuredProgramFunctionalEvidenceObligationTemplate(
          *sourceReference, *workloadReference, *runtimeInputReference, config,
          artifactStore);
  if (!functional)
    return functional.takeError();
  auto obligations =
      canonicalizeObligations(std::move(*analytic), std::move(*functional));
  if (!obligations)
    return obligations.takeError();

  std::optional<BaselineMetric> baseline;
  std::vector<ArtifactRootReference> baselineEvidence;
  if (options.ownership.selectionMode ==
      StructuredOwnershipSelectionMode::BenefitQualified) {
    auto acquired = acquireBaselineMetric(
        *obligations, options.ownership.selection.metricRequest,
        *sourceReference, fabric.reference(), *workloadReference,
        *runtimeInputReference, artifactStore, blobStore);
    if (!acquired)
      return acquired.takeError();
    if (auto *incomplete =
            std::get_if<IncompletePreMappingExploration>(&*acquired))
      return PreMappingExplorationOutcome{std::move(*incomplete)};
    baseline.emplace(std::get<BaselineMetric>(std::move(*acquired)));
    baselineEvidence.push_back(baseline->evidence);
  }

  auto objectives = compilerObjectives(
      *obligations, options.ownership.selection.metricRequest,
      options.ownership.selection.direction,
      evaluation::models::structuredFabricAnalyticMetricQuantumBase10Exponent);
  if (!objectives)
    return objectives.takeError();
  auto gate = ownershipQualityGate(*obligations, options.ownership, baseline);
  if (!gate)
    return gate.takeError();
  auto generatorConfig = projectResolvedStructuredOwnershipGeneratorConfigView(
      config, options.ownership.protocolCallableRoots);
  if (!generatorConfig)
    return generatorConfig.takeError();
  auto acquisitionConfig = projectResolvedEvidenceObligationSetConfigView(
      {obligations->analytic, obligations->functional});
  if (!acquisitionConfig)
    return acquisitionConfig.takeError();
  auto scheduleConfig =
      projectResolvedStructuredScheduleGeneratorConfigView(config);
  if (!scheduleConfig)
    return scheduleConfig.takeError();
  auto executionShapeConfig =
      projectResolvedStructuredExecutionShapeGeneratorConfigView();
  if (!executionShapeConfig)
    return executionShapeConfig.takeError();
  auto specialMathAccuracyConfig =
      projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView();
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
  planConfig.dse.qualityGatePolicies = {*gate};
  planConfig.dse.planNodes = {
      GeneratePlanNodeDefinition{
          structuredOwnershipCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{*sourceReference}},
           ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{*workloadReference}},
           ExactPlanArtifacts{{*runtimeInputReference}}},
          generatorConfig->canonicalViewBytes().vec(),
          generatorConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredExecutionShapeCandidateGeneratorDescriptor().reference(),
          {PlanOutputRef{0, 1}},
          executionShapeConfig->canonicalViewBytes().vec(),
          executionShapeConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredSpecialMathAccuracyCandidateGeneratorDescriptor()
              .reference(),
          {PlanOutputRef{1, 0}, ExactPlanArtifacts{{fabric.reference()}}},
          specialMathAccuracyConfig->canonicalViewBytes().vec(),
          specialMathAccuracyConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredScheduleCandidateGeneratorDescriptor().reference(),
          {PlanOutputRef{2, 0}, ExactPlanArtifacts{{fabric.reference()}}},
          scheduleConfig->canonicalViewBytes().vec(),
          scheduleConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredMemoryCommunicationCandidateGeneratorDescriptor()
              .reference(),
          {PlanOutputRef{3, 0}, ExactPlanArtifacts{{fabric.reference()}}},
          memoryCommunicationConfig->canonicalViewBytes().vec(),
          memoryCommunicationConfig->digest()},
      PromotePlanNodeDefinition{
          structuredEvaluationPromotionAcquisitionDescriptor().reference(),
          {PlanOutputRef{4, 0}, ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{*workloadReference}},
           ExactPlanArtifacts{{*runtimeInputReference}}},
          acquisitionConfig->canonicalViewBytes().vec(),
          acquisitionConfig->digest(),
          QualityGatePolicyRef(0),
          TopKSelection{0, options.ownership.selection.k},
          PromotePurpose::CandidateSelection}};
  auto view = projectResolvedDseConfigView(planConfig);
  if (!view)
    return view.takeError();
  auto executed = executeDsePlan(*view, artifactStore, blobStore);
  if (!executed)
    return executed.takeError();
  if (auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&*executed)) {
    const std::uint64_t nodeOrdinal = incomplete->nodeOrdinal();
    const DsePlanIncompleteReason reason = incomplete->reason();
    auto evidence = retainedEvidence(*incomplete, baselineEvidence);
    std::vector<DsePlanGenerateInvocationRecords> generateInvocations;
    generateInvocations.push_back(
        takeDsePlanGenerateInvocationRecords(std::move(*executed)));
    return PreMappingExplorationOutcome{IncompletePreMappingExploration{
        nodeOrdinal, reason, std::move(evidence),
        std::move(generateInvocations)}};
  }

  auto &completed = std::get<CompletedDsePlanExecution>(*executed);
  std::vector<ArtifactRootReference> selectedReferences(
      completed.resolve({5, 0}).begin(), completed.resolve({5, 0}).end());
  std::vector<ArtifactRootReference> satisfiedEvidence = baselineEvidence;
  mergeReferences(satisfiedEvidence, completed.resolve({5, 1}));
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
  planGenerateInvocations.push_back(
      takeDsePlanGenerateInvocationRecords(std::move(*executed)));
  if (selectedReferences.empty()) {
    if (options.ownership.selectionMode ==
        StructuredOwnershipSelectionMode::SemanticConformance)
      return PreMappingExplorationOutcome{
          CompletedPreMappingNoFeasibleCandidate{
              std::move(satisfiedEvidence),
              std::move(planGenerateInvocations)}};
    selectedReferences.push_back(*sourceReference);
  }

  std::vector<SelectedPreMappingCompilation> selected;
  selected.reserve(selectedReferences.size() * options.ownership.selection.k);
  for (const ArtifactRootReference &reference : selectedReferences) {
    if (reference == *sourceReference) {
      auto candidate =
          invocation.materializeSelectedCandidate(reference, artifactStore);
      if (!candidate)
        return candidate.takeError();
      selected.push_back(SelectedPreMappingCompilation{
          frontend::PreMappingCompilation{
              compilation.fabric, compilation.staticGlobalMemory,
              std::move(candidate->candidate.structuredProgram),
              std::move(candidate->candidate.sourceProvenance),
              std::move(candidate->candidate.canonicalDataflow)},
          std::move(candidate->derivations),
          std::move(candidate->executionShapeDerivations),
          std::move(candidate->specialMathAccuracyDerivations),
          std::move(candidate->scheduleDerivations),
          std::move(candidate->memoryCommunicationDerivations),
          std::move(candidate->dataflowRewriteDerivations),
          std::move(candidate->functionalReplay)});
      continue;
    }

    auto d0 = invocation.prepareDataflowGeneration(reference, artifactStore);
    if (!d0)
      return d0.takeError();
    auto dataflowSelection = exploreDataflowCandidates(
        *d0, reference, fabric.reference(), *workloadReference,
        *runtimeInputReference, config, options.ownership.selection,
        artifactStore, blobStore);
    if (!dataflowSelection)
      return dataflowSelection.takeError();
    if (auto *incomplete =
            std::get_if<IncompletePreMappingExploration>(&*dataflowSelection)) {
      mergeReferences(incomplete->retainedEvidence, satisfiedEvidence);
      incomplete->planGenerateInvocations.insert(
          incomplete->planGenerateInvocations.begin(),
          std::make_move_iterator(planGenerateInvocations.begin()),
          std::make_move_iterator(planGenerateInvocations.end()));
      return PreMappingExplorationOutcome{std::move(*incomplete)};
    }
    auto &dataflowCompleted =
        std::get<CompletedDataflowSelection>(*dataflowSelection);
    mergeReferences(satisfiedEvidence, dataflowCompleted.evidence);
    planGenerateInvocations.push_back(
        std::move(dataflowCompleted.generateInvocations));
    for (const ArtifactRootReference &dataflowReference :
         dataflowCompleted.selected) {
      auto candidate = invocation.materializeSelectedDataflowCandidate(
          reference, dataflowReference, artifactStore);
      if (!candidate)
        return candidate.takeError();
      selected.push_back(SelectedPreMappingCompilation{
          frontend::PreMappingCompilation{
              compilation.fabric, compilation.staticGlobalMemory,
              std::move(candidate->candidate.structuredProgram),
              std::move(candidate->candidate.sourceProvenance),
              std::move(candidate->candidate.canonicalDataflow)},
          std::move(candidate->derivations),
          std::move(candidate->executionShapeDerivations),
          std::move(candidate->specialMathAccuracyDerivations),
          std::move(candidate->scheduleDerivations),
          std::move(candidate->memoryCommunicationDerivations),
          std::move(candidate->dataflowRewriteDerivations),
          std::move(candidate->functionalReplay)});
    }
  }
  if (selected.empty())
    return PreMappingExplorationOutcome{CompletedPreMappingNoFeasibleCandidate{
        std::move(satisfiedEvidence), std::move(planGenerateInvocations)}};
  return PreMappingExplorationOutcome{CompletedPreMappingSelection{
      std::move(selected), std::move(satisfiedEvidence),
      std::vector<StructuredOwnershipCandidateDisposition>(
          invocation.dispositions().begin(), invocation.dispositions().end()),
      std::move(planGenerateInvocations)}};
}

} // namespace loom::dse
