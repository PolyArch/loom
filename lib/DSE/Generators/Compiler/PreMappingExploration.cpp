#include "DSE/PreMappingExploration.h"

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
#include "Frontend/Analysis/StructuredProtocolDependencies.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
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
          fields["scope_ordinal"] =
              disposition.coordinate.scope.selection.ordinal;
        });
  }
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

struct CompletedOwnershipSelection final {
  std::unique_ptr<StructuredOwnershipInvocation> invocation;
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  DsePlanGenerateInvocationRecords generateInvocations;
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
};

using OwnershipSelectionOutcome =
    std::variant<CompletedOwnershipSelection, IncompletePreMappingExploration>;

struct ProtocolTraversalPlan final {
  std::vector<std::vector<std::size_t>> paths;
  std::vector<std::vector<std::uint64_t>> dependencyWeights;
};

llvm::Expected<ProtocolTraversalPlan> buildProtocolTraversalPlan(
    llvm::ArrayRef<frontend::StructuredEntityRef> roots,
    llvm::ArrayRef<frontend::analysis::StructuredProtocolDependency>
        dependencies) {
  ProtocolTraversalPlan plan;
  plan.dependencyWeights.assign(roots.size(),
                                std::vector<std::uint64_t>(roots.size(), 0));
  const auto rootOrdinal = [&](const frontend::StructuredEntityRef &root)
      -> std::optional<std::size_t> {
    for (auto item : llvm::enumerate(roots))
      if (item.value() == root)
        return item.index();
    return std::nullopt;
  };
  for (const auto &dependency : dependencies) {
    std::optional<std::size_t> producer = rootOrdinal(dependency.producer);
    std::optional<std::size_t> consumer = rootOrdinal(dependency.consumer);
    if (!producer || !consumer || *producer == *consumer ||
        dependency.sharedMemoryObjectCount == 0)
      return invalid("protocol dependency is outside its exact root set");
    std::uint64_t &weight = plan.dependencyWeights[*producer][*consumer];
    if (std::numeric_limits<std::uint64_t>::max() - weight <
        dependency.sharedMemoryObjectCount)
      return invalid("protocol dependency weight overflows");
    weight += dependency.sharedMemoryObjectCount;
  }
  if (roots.empty()) {
    plan.paths.push_back({});
    return plan;
  }

  const auto addPath = [&](std::vector<std::size_t> path) {
    if (!llvm::is_contained(plan.paths, path))
      plan.paths.push_back(std::move(path));
  };
  std::vector<std::size_t> canonical;
  canonical.reserve(roots.size());
  for (std::size_t ordinal = 0; ordinal != roots.size(); ++ordinal)
    canonical.push_back(ordinal);
  addPath(std::move(canonical));

  for (std::size_t seed = 0; seed != roots.size(); ++seed) {
    std::vector<bool> selected(roots.size(), false);
    std::vector<std::size_t> path;
    selected[seed] = true;
    path.push_back(seed);
    while (path.size() != roots.size()) {
      std::optional<std::size_t> best;
      std::uint64_t bestWeight = 0;
      for (std::size_t candidate = 0; candidate != roots.size(); ++candidate) {
        if (selected[candidate])
          continue;
        std::uint64_t weight = 0;
        for (std::size_t member : path) {
          const std::uint64_t forward =
              plan.dependencyWeights[member][candidate];
          const std::uint64_t reverse =
              plan.dependencyWeights[candidate][member];
          if (std::numeric_limits<std::uint64_t>::max() - weight < forward ||
              std::numeric_limits<std::uint64_t>::max() - weight - forward <
                  reverse)
            return invalid("protocol cut weight overflows");
          weight += forward + reverse;
        }
        if (weight > bestWeight) {
          best = candidate;
          bestWeight = weight;
        }
      }
      if (!best || bestWeight == 0)
        break;
      selected[*best] = true;
      path.push_back(*best);
    }
    addPath(std::move(path));
  }
  return plan;
}

std::pair<std::uint64_t, std::uint64_t> protocolDependencyPartition(
    llvm::ArrayRef<std::size_t> selected,
    llvm::ArrayRef<std::vector<std::uint64_t>> weights) {
  std::vector<bool> owned(weights.size(), false);
  for (std::size_t ordinal : selected)
    owned[ordinal] = true;
  std::uint64_t internal = 0;
  std::uint64_t cut = 0;
  for (std::size_t producer = 0; producer != weights.size(); ++producer)
    for (std::size_t consumer = 0; consumer != weights.size(); ++consumer) {
      if (owned[producer] && owned[consumer])
        internal += weights[producer][consumer];
      else if (owned[producer] != owned[consumer])
        cut += weights[producer][consumer];
    }
  return {internal, cut};
}

llvm::Expected<std::vector<std::string>>
protocolCallableSymbols(const frontend::StructuredProgramCandidate &source,
                        llvm::ArrayRef<frontend::StructuredEntityRef> roots) {
  auto view = source.view();
  if (!view)
    return view.takeError();
  std::vector<std::pair<frontend::StructuredEntityRef, std::string>> rooted;
  rooted.reserve(roots.size());
  for (const frontend::StructuredEntityRef &root : roots) {
    auto entity = view->resolve(root);
    if (!entity)
      return entity.takeError();
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity->operation);
    if (!function || function.isExternal())
      return invalid("operator protocol root is not a defined LLVM callable");
    const std::string symbol = function.getSymName().str();
    if (llvm::any_of(rooted,
                     [&](const auto &entry) { return entry.second == symbol; }))
      return invalid("operator protocol roots contain a duplicate callable");
    rooted.emplace_back(root, symbol);
  }
  llvm::sort(rooted, [](const auto &lhs, const auto &rhs) {
    if (lhs.first.parent.bytes() != rhs.first.parent.bytes())
      return lhs.first.parent.bytes() < rhs.first.parent.bytes();
    if (lhs.first.kind != rhs.first.kind)
      return static_cast<std::uint32_t>(lhs.first.kind) <
             static_cast<std::uint32_t>(rhs.first.kind);
    return lhs.first.ordinal < rhs.first.ordinal;
  });
  std::vector<std::string> symbols;
  symbols.reserve(rooted.size());
  for (auto &entry : rooted)
    symbols.push_back(std::move(entry.second));
  return symbols;
}

llvm::Expected<std::vector<frontend::StructuredEntityRef>>
resolveProtocolCallableRoots(const frontend::StructuredProgramCandidate &parent,
                             llvm::ArrayRef<std::string> symbols) {
  llvm::SmallVector<llvm::StringRef> names;
  names.reserve(symbols.size());
  for (const std::string &symbol : symbols)
    names.push_back(symbol);
  return frontend::resolveDefinedLlvmCallables(parent, names);
}

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
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation) {
  auto invocation = std::make_unique<StructuredOwnershipInvocation>(
      generationParent.structuredProgram, sourceProgram, workload, runtimeInput,
      fabric, config, options.lowering, options.candidateWorkerCount,
      options.functionalReplayLimits, generationParent.sourceProvenance,
      sharedEvaluation);

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
  auto generatorConfig = projectResolvedStructuredOwnershipGeneratorConfigView(
      config, options.protocolCallableRoots);
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
          {ExactPlanArtifacts{{generationParentReference}},
           ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{workloadReference}},
           ExactPlanArtifacts{{runtimeInputReference}}},
          generatorConfig->canonicalViewBytes().vec(),
          generatorConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredExecutionShapeCandidateGeneratorDescriptor().reference(),
          {PlanOutputRef{0, 1}},
          executionShapeConfig->canonicalViewBytes().vec(),
          executionShapeConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredScheduleCandidateGeneratorDescriptor().reference(),
          {PlanOutputRef{1, 0}, ExactPlanArtifacts{{fabric.reference()}}},
          scheduleConfig->canonicalViewBytes().vec(),
          scheduleConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredMemoryCommunicationCandidateGeneratorDescriptor()
              .reference(),
          {PlanOutputRef{2, 0}},
          memoryCommunicationConfig->canonicalViewBytes().vec(),
          memoryCommunicationConfig->digest()},
      GeneratePlanNodeDefinition{
          structuredSpecialMathAccuracyCandidateGeneratorDescriptor()
              .reference(),
          {PlanOutputRef{3, 0}, ExactPlanArtifacts{{fabric.reference()}}},
          specialMathAccuracyConfig->canonicalViewBytes().vec(),
          specialMathAccuracyConfig->digest()},
      PromotePlanNodeDefinition{
          structuredEvaluationPromotionAcquisitionDescriptor().reference(),
          {PlanOutputRef{4, 0}, ExactPlanArtifacts{{fabric.reference()}},
           ExactPlanArtifacts{{workloadReference}},
           ExactPlanArtifacts{{runtimeInputReference}}},
          acquisitionConfig->canonicalViewBytes().vec(),
          acquisitionConfig->digest(),
          QualityGatePolicyRef(0),
          TopKSelection{0, options.selection.k},
          PromotePurpose::CandidateSelection}};
  auto view = projectResolvedDseConfigView(planConfig);
  if (!view)
    return view.takeError();
  auto executed = executeDsePlan(*view, artifactStore, blobStore);
  if (!executed)
    return executed.takeError();
  const CompletedDsePlanExecution *selectionExecution =
      std::get_if<CompletedDsePlanExecution>(&*executed);
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
  if (auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&*executed)) {
    selectionExecution = &incomplete->availableExecution();
    if (incomplete->executionStopped() ||
        selectionExecution->resolve({5, 0}).empty()) {
      const std::uint64_t nodeOrdinal = incomplete->nodeOrdinal();
      const DsePlanIncompleteReason reason = incomplete->reason();
      auto evidence = retainedEvidence(*incomplete, baselineEvidence);
      std::vector<DsePlanGenerateInvocationRecords> generateInvocations;
      generateInvocations.push_back(
          takeDsePlanGenerateInvocationRecords(std::move(*executed)));
      return OwnershipSelectionOutcome{IncompletePreMappingExploration{
          nodeOrdinal, reason, std::move(evidence),
          std::move(generateInvocations)}};
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
        fields["selected_count"] = count(5, 0);
        fields["evidence_count"] = count(5, 1);
      });
  std::vector<ArtifactRootReference> selected(
      selectionExecution->resolve({5, 0}).begin(),
      selectionExecution->resolve({5, 0}).end());
  auto preferenceOrder = selectedPreferenceOrder(*selectionExecution, {5, 0});
  if (!preferenceOrder)
    return preferenceOrder.takeError();
  std::vector<ArtifactRootReference> evidence = std::move(baselineEvidence);
  mergeReferences(evidence, selectionExecution->resolve({5, 1}));
  std::vector<StructuredOwnershipCandidateDisposition> dispositions(
      invocation->dispositions().begin(), invocation->dispositions().end());
  return OwnershipSelectionOutcome{CompletedOwnershipSelection{
      std::move(invocation), std::move(selected), std::move(*preferenceOrder),
      std::move(evidence), std::move(dispositions),
      takeDsePlanGenerateInvocationRecords(std::move(*executed)),
      std::move(retainedIncompleteness)}};
}

struct CompletedDataflowSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> preferenceOrder;
  std::vector<ArtifactRootReference> evidence;
  DsePlanGenerateInvocationRecords generateInvocations;
  std::optional<RetainedDsePlanIncompleteness> retainedIncompleteness;
};

using DataflowSelectionOutcome =
    std::variant<CompletedDataflowSelection, IncompletePreMappingExploration>;

llvm::Expected<DataflowSelectionOutcome>
exploreDataflowCandidates(const ArtifactRootReference &d0,
                          const ArtifactRootReference &structuredParent,
                          const fabric::FinalizedFabricRoot &fabric,
                          const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ResolvedConfig &config,
                          const StructuredOwnershipTopKSelection &selection,
                          StructuredOwnershipSelectionMode selectionMode,
                          const ArtifactStore &store, const BlobStore &blobs) {
  bool requiresRewriteExploration = true;
  if (selectionMode == StructuredOwnershipSelectionMode::SemanticConformance) {
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
            {PlanOutputRef{0, 0}, ExactPlanArtifacts{{structuredParent}},
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
  auto executed = executeDsePlan(*view, store, blobs);
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
      return DataflowSelectionOutcome{IncompletePreMappingExploration{
          nodeOrdinal, reason, std::move(evidence),
          std::move(generateInvocations)}};
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

  auto sourceObservations = sim::executeNativeStructuredProgram(
      compilation.structuredProgram, workload, runtimeInput);
  if (!sourceObservations)
    return sourceObservations.takeError();
  evaluation::models::StructuredEvaluationInvocationCache evaluationCache;
  StructuredOwnershipSharedEvaluation sharedEvaluation(*sourceObservations,
                                                       evaluationCache);

  auto callableSymbols = protocolCallableSymbols(
      compilation.structuredProgram, options.ownership.protocolCallableRoots);
  if (!callableSymbols)
    return callableSymbols.takeError();
  auto sourceProtocolRoots = resolveProtocolCallableRoots(
      compilation.structuredProgram, *callableSymbols);
  if (!sourceProtocolRoots)
    return sourceProtocolRoots.takeError();
  auto protocolDependencies =
      frontend::analysis::projectStructuredProtocolDependencies(
          compilation.structuredProgram, *sourceProtocolRoots);
  if (!protocolDependencies)
    return protocolDependencies.takeError();
  auto projectedRootActivity =
      evaluation::models::projectStructuredScopeActivity(
          compilation.structuredProgram, *sourceObservations,
          *sourceProtocolRoots);
  if (!projectedRootActivity)
    return projectedRootActivity.takeError();
  std::vector<PreMappingProtocolRootActivity> protocolRootActivity;
  protocolRootActivity.reserve(projectedRootActivity->size());
  for (const auto &activity : *projectedRootActivity)
    protocolRootActivity.push_back({activity.scope, activity.dynamicActivations,
                                    activity.dynamicLeafExecutions});
  auto traversalPlan =
      buildProtocolTraversalPlan(*sourceProtocolRoots, *protocolDependencies);
  if (!traversalPlan)
    return traversalPlan.takeError();

  std::vector<std::unique_ptr<frontend::StructuredCompilation>> parentStorage;
  parentStorage.push_back(std::make_unique<frontend::StructuredCompilation>(
      std::move(compilation)));
  const frontend::StructuredCompilation &sourceCompilation =
      *parentStorage.front();

  struct RetainedOwnershipSelection final {
    std::unique_ptr<StructuredOwnershipInvocation> invocation;
    std::vector<ArtifactRootReference> selected;
    std::vector<ArtifactRootReference> preferenceOrder;
  };
  struct RetainedOwnershipAlternative final {
    std::size_t generationIndex = 0;
    std::vector<ArtifactRootReference> references;
    std::vector<StructuredOwnershipDerivation> ownershipPrefix;
    std::vector<StructuredExecutionShapeDerivation> executionShapePrefix;
    std::vector<StructuredSpecialMathAccuracyDerivation> specialMathPrefix;
    std::vector<StructuredScheduleDerivation> schedulePrefix;
    std::vector<StructuredMemoryCommunicationDerivation> memoryPrefix;
    std::vector<std::size_t> ownedProtocolOrdinals;
    std::uint64_t internalDependencyWeight = 0;
    std::uint64_t cutDependencyWeight = 0;
    std::optional<std::uint64_t> estimatedRuntimePicoseconds;
    bool derivationsIncluded = false;
  };
  std::vector<RetainedOwnershipSelection> generations;
  std::vector<RetainedOwnershipAlternative> semanticAlternatives;
  std::vector<PreMappingCandidatePlanningRecord> candidateInventory;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
  std::vector<RetainedDsePlanIncompleteness> retainedPlanIncompleteness;
  std::optional<std::size_t> finalGeneration;
  std::vector<ArtifactRootReference> finalReferences;
  std::vector<StructuredOwnershipDerivation> finalOwnershipPrefix;
  std::vector<StructuredExecutionShapeDerivation> finalExecutionShapePrefix;
  std::vector<StructuredSpecialMathAccuracyDerivation> finalSpecialMathPrefix;
  std::vector<StructuredScheduleDerivation> finalSchedulePrefix;
  std::vector<StructuredMemoryCommunicationDerivation> finalMemoryPrefix;
  bool finalDerivationsIncluded = false;
  std::vector<std::vector<std::size_t>> paths;
  if (options.ownership.selectionMode ==
      StructuredOwnershipSelectionMode::SemanticConformance)
    paths = traversalPlan->paths;
  else
    paths.push_back({});

  for (const std::vector<std::size_t> &path : paths) {
    auto parentProgram = frontend::importStructuredProgram(
        sourceCompilation.structuredProgram.identity(),
        sourceCompilation.structuredProgram.canonicalBytes());
    if (!parentProgram)
      return parentProgram.takeError();
    parentStorage.push_back(std::make_unique<frontend::StructuredCompilation>(
        frontend::StructuredCompilation{
            sourceCompilation.fabric, sourceCompilation.staticGlobalMemory,
            std::move(*parentProgram), sourceCompilation.sourceProvenance}));
    frontend::StructuredCompilation *parent = parentStorage.back().get();

    std::vector<StructuredOwnershipDerivation> ownershipPrefix;
    std::vector<StructuredExecutionShapeDerivation> executionShapePrefix;
    std::vector<StructuredSpecialMathAccuracyDerivation> specialMathPrefix;
    std::vector<StructuredScheduleDerivation> schedulePrefix;
    std::vector<StructuredMemoryCommunicationDerivation> memoryPrefix;
    std::vector<std::size_t> ownedOrdinals;
    std::optional<std::size_t> priorSelectedGeneration;
    std::optional<ArtifactRootReference> priorSelectedReference;
    bool priorDerivationsIncluded = false;

    const std::size_t generationCount =
        options.ownership.selectionMode ==
                StructuredOwnershipSelectionMode::SemanticConformance
            ? std::max<std::size_t>(1, path.size())
            : 1;
    for (std::size_t transformationOrdinal = 0;
         transformationOrdinal != generationCount; ++transformationOrdinal) {
      auto parentReference = frontend::publishStructuredProgram(
          parent->structuredProgram, artifactStore);
      if (!parentReference)
        return parentReference.takeError();
      llvm::ArrayRef<std::string> generationSymbols = *callableSymbols;
      if (options.ownership.selectionMode ==
          StructuredOwnershipSelectionMode::SemanticConformance) {
        if (path.empty())
          generationSymbols = {};
        else
          generationSymbols = llvm::ArrayRef<std::string>(*callableSymbols)
                                  .slice(path[transformationOrdinal], 1);
      }
      auto protocolRoots = resolveProtocolCallableRoots(
          parent->structuredProgram, generationSymbols);
      if (!protocolRoots)
        return protocolRoots.takeError();
      StructuredOwnershipExplorationOptions generationOptions =
          options.ownership;
      generationOptions.protocolCallableRoots = std::move(*protocolRoots);
      if (options.ownership.selectionMode ==
          StructuredOwnershipSelectionMode::SemanticConformance)
        generationOptions.selection.k = 1;
      auto explored = exploreOwnershipCandidates(
          *parent, *parentReference, sourceCompilation.structuredProgram,
          *sourceReference, workload, *workloadReference, runtimeInput,
          *runtimeInputReference, fabric, config, generationOptions,
          artifactStore, blobStore, &sharedEvaluation);
      if (!explored)
        return explored.takeError();
      if (auto *incomplete =
              std::get_if<IncompletePreMappingExploration>(&*explored)) {
        mergeReferences(incomplete->retainedEvidence, satisfiedEvidence);
        incomplete->planGenerateInvocations.insert(
            incomplete->planGenerateInvocations.begin(),
            std::make_move_iterator(planGenerateInvocations.begin()),
            std::make_move_iterator(planGenerateInvocations.end()));
        return PreMappingExplorationOutcome{std::move(*incomplete)};
      }

      auto completed =
          std::get<CompletedOwnershipSelection>(std::move(*explored));
      mergeReferences(satisfiedEvidence, completed.evidence);
      dispositions.insert(
          dispositions.end(),
          std::make_move_iterator(completed.dispositions.begin()),
          std::make_move_iterator(completed.dispositions.end()));
      planGenerateInvocations.push_back(
          std::move(completed.generateInvocations));
      if (completed.retainedIncompleteness)
        retainedPlanIncompleteness.push_back(
            std::move(*completed.retainedIncompleteness));
      generations.push_back(RetainedOwnershipSelection{
          std::move(completed.invocation), std::move(completed.selected),
          std::move(completed.preferenceOrder)});
      const std::size_t generationIndex = generations.size() - 1;
      RetainedOwnershipSelection &generation = generations.back();

      if (generation.selected.empty()) {
        if (options.ownership.selectionMode !=
            StructuredOwnershipSelectionMode::SemanticConformance) {
          finalGeneration = priorSelectedGeneration.value_or(generationIndex);
          finalReferences =
              priorSelectedReference
                  ? std::vector<ArtifactRootReference>{*priorSelectedReference}
                  : std::vector<ArtifactRootReference>{*sourceReference};
          finalOwnershipPrefix = ownershipPrefix;
          finalExecutionShapePrefix = executionShapePrefix;
          finalSpecialMathPrefix = specialMathPrefix;
          finalSchedulePrefix = schedulePrefix;
          finalMemoryPrefix = memoryPrefix;
          finalDerivationsIncluded = priorDerivationsIncluded;
        }
        break;
      }

      priorSelectedGeneration = generationIndex;
      priorSelectedReference = generation.preferenceOrder.front();
      priorDerivationsIncluded = false;
      if (options.ownership.selectionMode ==
          StructuredOwnershipSelectionMode::SemanticConformance) {
        if (!path.empty())
          ownedOrdinals.push_back(path[transformationOrdinal]);
        auto estimate =
            evaluation::models::lookupStructuredFabricAnalyticRuntimeEstimate(
                generation.preferenceOrder.front(), fabric.reference(),
                *workloadReference, *runtimeInputReference, config,
                evaluationCache);
        if (!estimate)
          return estimate.takeError();
        auto [internalWeight, cutWeight] = protocolDependencyPartition(
            ownedOrdinals, traversalPlan->dependencyWeights);
        semanticAlternatives.push_back(
            RetainedOwnershipAlternative{generationIndex,
                                         {generation.preferenceOrder.front()},
                                         ownershipPrefix,
                                         executionShapePrefix,
                                         specialMathPrefix,
                                         schedulePrefix,
                                         memoryPrefix,
                                         ownedOrdinals,
                                         internalWeight,
                                         cutWeight,
                                         *estimate,
                                         false});
      }
      if (transformationOrdinal + 1 == generationCount) {
        if (options.ownership.selectionMode !=
            StructuredOwnershipSelectionMode::SemanticConformance) {
          finalGeneration = generationIndex;
          finalReferences = generation.preferenceOrder;
          finalOwnershipPrefix = ownershipPrefix;
          finalExecutionShapePrefix = executionShapePrefix;
          finalSpecialMathPrefix = specialMathPrefix;
          finalSchedulePrefix = schedulePrefix;
          finalMemoryPrefix = memoryPrefix;
        }
        break;
      }

      StructuredOwnershipInvocationScope generationScope(
          *generation.invocation);
      auto selected = generation.invocation->materializeSelectedCandidate(
          generation.preferenceOrder.front(), artifactStore);
      if (!selected)
        return selected.takeError();
      ownershipPrefix.insert(
          ownershipPrefix.end(),
          std::make_move_iterator(selected->derivations.begin()),
          std::make_move_iterator(selected->derivations.end()));
      executionShapePrefix.insert(
          executionShapePrefix.end(),
          std::make_move_iterator(selected->executionShapeDerivations.begin()),
          std::make_move_iterator(selected->executionShapeDerivations.end()));
      specialMathPrefix.insert(
          specialMathPrefix.end(),
          std::make_move_iterator(
              selected->specialMathAccuracyDerivations.begin()),
          std::make_move_iterator(
              selected->specialMathAccuracyDerivations.end()));
      schedulePrefix.insert(
          schedulePrefix.end(),
          std::make_move_iterator(selected->scheduleDerivations.begin()),
          std::make_move_iterator(selected->scheduleDerivations.end()));
      memoryPrefix.insert(memoryPrefix.end(),
                          std::make_move_iterator(
                              selected->memoryCommunicationDerivations.begin()),
                          std::make_move_iterator(
                              selected->memoryCommunicationDerivations.end()));
      parentStorage.push_back(std::make_unique<frontend::StructuredCompilation>(
          frontend::StructuredCompilation{
              sourceCompilation.fabric, sourceCompilation.staticGlobalMemory,
              std::move(selected->candidate.structuredProgram),
              std::move(selected->candidate.sourceProvenance)}));
      parent = parentStorage.back().get();
      priorDerivationsIncluded = true;
    }
  }

  std::vector<RetainedOwnershipAlternative> alternatives;
  if (options.ownership.selectionMode ==
      StructuredOwnershipSelectionMode::SemanticConformance) {
    if (semanticAlternatives.empty()) {
      emitOwnershipRejections(dispositions);
      return PreMappingExplorationOutcome{
          CompletedPreMappingNoFeasibleCandidate{
              std::move(satisfiedEvidence),
              std::move(planGenerateInvocations)}};
    }
    const auto scoreLess = [](const RetainedOwnershipAlternative &lhs,
                              const RetainedOwnershipAlternative &rhs) {
      if (lhs.estimatedRuntimePicoseconds.has_value() !=
          rhs.estimatedRuntimePicoseconds.has_value())
        return lhs.estimatedRuntimePicoseconds.has_value();
      if (lhs.estimatedRuntimePicoseconds != rhs.estimatedRuntimePicoseconds)
        return lhs.estimatedRuntimePicoseconds <
               rhs.estimatedRuntimePicoseconds;
      if (lhs.cutDependencyWeight != rhs.cutDependencyWeight)
        return lhs.cutDependencyWeight < rhs.cutDependencyWeight;
      if (lhs.internalDependencyWeight != rhs.internalDependencyWeight)
        return lhs.internalDependencyWeight > rhs.internalDependencyWeight;
      if (lhs.ownedProtocolOrdinals != rhs.ownedProtocolOrdinals)
        return lhs.ownedProtocolOrdinals < rhs.ownedProtocolOrdinals;
      return artifactRootReferenceLess(lhs.references.front(),
                                       rhs.references.front());
    };
    llvm::sort(semanticAlternatives, scoreLess);
    std::vector<RetainedOwnershipAlternative> unique;
    for (RetainedOwnershipAlternative &alternative : semanticAlternatives) {
      auto duplicate = llvm::find_if(unique, [&](const auto &retained) {
        return retained.references.front() == alternative.references.front();
      });
      if (duplicate == unique.end())
        unique.push_back(std::move(alternative));
    }

    std::vector<std::size_t> diverseOrder;
    const auto retainIndex = [&](std::size_t index) {
      if (!llvm::is_contained(diverseOrder, index))
        diverseOrder.push_back(index);
    };
    retainIndex(0);
    const auto maximal = std::max_element(
        unique.begin(), unique.end(), [](const auto &lhs, const auto &rhs) {
          return lhs.ownedProtocolOrdinals.size() <
                 rhs.ownedProtocolOrdinals.size();
        });
    if (maximal != unique.end())
      retainIndex(std::distance(unique.begin(), maximal));
    const auto minimal = std::min_element(
        unique.begin(), unique.end(), [](const auto &lhs, const auto &rhs) {
          return lhs.ownedProtocolOrdinals.size() <
                 rhs.ownedProtocolOrdinals.size();
        });
    if (minimal != unique.end())
      retainIndex(std::distance(unique.begin(), minimal));
    for (std::size_t index = 0; index != unique.size(); ++index)
      retainIndex(index);

    const std::size_t retainedCount =
        static_cast<std::size_t>(std::min<std::uint64_t>(
            options.ownership.selection.k, diverseOrder.size()));
    candidateInventory.reserve(unique.size());
    for (std::size_t uniqueIndex = 0; uniqueIndex != unique.size();
         ++uniqueIndex) {
      std::optional<std::uint64_t> preferenceRank;
      for (std::size_t rank = 0; rank != retainedCount; ++rank)
        if (diverseOrder[rank] == uniqueIndex) {
          preferenceRank = rank;
          break;
        }
      std::vector<frontend::StructuredEntityRef> ownedRoots;
      ownedRoots.reserve(unique[uniqueIndex].ownedProtocolOrdinals.size());
      for (std::size_t ordinal : unique[uniqueIndex].ownedProtocolOrdinals) {
        if (ordinal >= sourceProtocolRoots->size())
          return invalid("candidate inventory owns a foreign protocol root");
        ownedRoots.push_back((*sourceProtocolRoots)[ordinal]);
      }
      candidateInventory.push_back(PreMappingCandidatePlanningRecord{
          unique[uniqueIndex].references.front(), std::move(ownedRoots),
          unique[uniqueIndex].estimatedRuntimePicoseconds,
          preferenceRank
              ? PreMappingCandidatePlanningDisposition::Retained
              : PreMappingCandidatePlanningDisposition::BoundedFrontierBudget,
          preferenceRank});
    }

    alternatives.reserve(retainedCount);
    for (std::size_t rank = 0; rank != retainedCount; ++rank) {
      const std::size_t index = diverseOrder[rank];
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "retain_dependency_aware_candidate";
            fields["preference_rank"] = rank;
            fields["owned_region_count"] =
                unique[index].ownedProtocolOrdinals.size();
            fields["internal_dependency_weight"] =
                unique[index].internalDependencyWeight;
            fields["cut_dependency_weight"] = unique[index].cutDependencyWeight;
            fields["analytic_runtime_supported"] =
                unique[index].estimatedRuntimePicoseconds.has_value();
            if (unique[index].estimatedRuntimePicoseconds)
              fields["estimated_runtime_ps"] =
                  *unique[index].estimatedRuntimePicoseconds;
          });
      alternatives.push_back(std::move(unique[index]));
    }
  } else {
    if (!finalGeneration || finalReferences.empty())
      return invalid(
          "ownership exploration did not choose a terminal candidate");
    alternatives.push_back(
        RetainedOwnershipAlternative{*finalGeneration,
                                     std::move(finalReferences),
                                     std::move(finalOwnershipPrefix),
                                     std::move(finalExecutionShapePrefix),
                                     std::move(finalSpecialMathPrefix),
                                     std::move(finalSchedulePrefix),
                                     std::move(finalMemoryPrefix),
                                     {},
                                     0,
                                     0,
                                     std::nullopt,
                                     finalDerivationsIncluded});
    for (std::size_t rank = 0; rank != alternatives.front().references.size();
         ++rank)
      candidateInventory.push_back(PreMappingCandidatePlanningRecord{
          alternatives.front().references[rank],
          {},
          std::nullopt,
          PreMappingCandidatePlanningDisposition::Retained,
          rank});
  }

  auto assemble = [&](SelectedStructuredOwnershipCandidate candidate,
                      const RetainedOwnershipAlternative &alternative) {
    std::vector<StructuredOwnershipDerivation> ownership =
        alternative.ownershipPrefix;
    std::vector<StructuredExecutionShapeDerivation> executionShape =
        alternative.executionShapePrefix;
    std::vector<StructuredSpecialMathAccuracyDerivation> specialMath =
        alternative.specialMathPrefix;
    std::vector<StructuredScheduleDerivation> schedule =
        alternative.schedulePrefix;
    std::vector<StructuredMemoryCommunicationDerivation> memory =
        alternative.memoryPrefix;
    if (!alternative.derivationsIncluded) {
      ownership.insert(ownership.end(),
                       std::make_move_iterator(candidate.derivations.begin()),
                       std::make_move_iterator(candidate.derivations.end()));
      executionShape.insert(
          executionShape.end(),
          std::make_move_iterator(candidate.executionShapeDerivations.begin()),
          std::make_move_iterator(candidate.executionShapeDerivations.end()));
      specialMath.insert(specialMath.end(),
                         std::make_move_iterator(
                             candidate.specialMathAccuracyDerivations.begin()),
                         std::make_move_iterator(
                             candidate.specialMathAccuracyDerivations.end()));
      schedule.insert(
          schedule.end(),
          std::make_move_iterator(candidate.scheduleDerivations.begin()),
          std::make_move_iterator(candidate.scheduleDerivations.end()));
      memory.insert(memory.end(),
                    std::make_move_iterator(
                        candidate.memoryCommunicationDerivations.begin()),
                    std::make_move_iterator(
                        candidate.memoryCommunicationDerivations.end()));
    }
    return SelectedPreMappingCompilation{
        0,
        frontend::PreMappingCompilation{
            sourceCompilation.fabric, sourceCompilation.staticGlobalMemory,
            std::move(candidate.candidate.structuredProgram),
            std::move(candidate.candidate.sourceProvenance),
            std::move(candidate.candidate.canonicalDataflow)},
        std::move(ownership),
        std::move(executionShape),
        std::move(specialMath),
        std::move(schedule),
        std::move(memory),
        std::move(candidate.dataflowRewriteDerivations),
        std::move(candidate.functionalReplay)};
  };

  std::vector<SelectedPreMappingCompilation> selected;
  selected.reserve(static_cast<std::size_t>(options.ownership.selection.k));
  StructuredOwnershipTopKSelection dataflowSelectionPolicy =
      options.ownership.selection;
  if (options.ownership.selectionMode ==
      StructuredOwnershipSelectionMode::SemanticConformance)
    dataflowSelectionPolicy.k = 1;
  for (const RetainedOwnershipAlternative &alternative : alternatives) {
    RetainedOwnershipSelection &generation =
        generations[alternative.generationIndex];
    StructuredOwnershipInvocationScope generationScope(*generation.invocation);
    for (const ArtifactRootReference &reference : alternative.references) {
      if (reference == *sourceReference) {
        auto candidate = generation.invocation->materializeSelectedCandidate(
            reference, artifactStore);
        if (!candidate)
          return candidate.takeError();
        selected.push_back(assemble(std::move(*candidate), alternative));
        continue;
      }

      auto d0 = generation.invocation->prepareDataflowGeneration(reference,
                                                                 artifactStore);
      if (!d0)
        return d0.takeError();
      auto dataflowSelection = exploreDataflowCandidates(
          *d0, reference, fabric, *workloadReference, *runtimeInputReference,
          config, dataflowSelectionPolicy, options.ownership.selectionMode,
          artifactStore, blobStore);
      if (!dataflowSelection)
        return dataflowSelection.takeError();
      if (auto *incomplete = std::get_if<IncompletePreMappingExploration>(
              &*dataflowSelection)) {
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
      if (dataflowCompleted.retainedIncompleteness)
        retainedPlanIncompleteness.push_back(
            std::move(*dataflowCompleted.retainedIncompleteness));
      for (const ArtifactRootReference &dataflowReference :
           dataflowCompleted.preferenceOrder) {
        auto candidate =
            generation.invocation->materializeSelectedDataflowCandidate(
                reference, dataflowReference, artifactStore);
        if (!candidate)
          return candidate.takeError();
        selected.push_back(assemble(std::move(*candidate), alternative));
      }
    }
  }
  std::vector<SelectedPreMappingCompilation> bounded;
  bounded.reserve(static_cast<std::size_t>(
      std::min<std::uint64_t>(options.ownership.selection.k, selected.size())));
  std::vector<ArtifactIdentity> retainedDataflows;
  retainedDataflows.reserve(bounded.capacity());
  for (SelectedPreMappingCompilation &candidate : selected) {
    const ArtifactIdentity identity =
        candidate.compilation.canonicalDataflow.identity();
    if (llvm::is_contained(retainedDataflows, identity))
      continue;
    candidate.preferenceRank = bounded.size();
    retainedDataflows.push_back(identity);
    bounded.push_back(std::move(candidate));
    if (bounded.size() == options.ownership.selection.k)
      break;
  }
  selected = std::move(bounded);
  if (selected.empty() && !retainedPlanIncompleteness.empty()) {
    const RetainedDsePlanIncompleteness &first =
        retainedPlanIncompleteness.front();
    return PreMappingExplorationOutcome{IncompletePreMappingExploration{
        first.nodeOrdinal, first.reason, std::move(satisfiedEvidence),
        std::move(planGenerateInvocations)}};
  }
  if (selected.empty())
    return PreMappingExplorationOutcome{CompletedPreMappingNoFeasibleCandidate{
        std::move(satisfiedEvidence), std::move(planGenerateInvocations)}};
  return PreMappingExplorationOutcome{CompletedPreMappingSelection{
      std::move(selected), std::move(satisfiedEvidence),
      std::move(dispositions), std::move(protocolRootActivity),
      std::move(*protocolDependencies), std::move(candidateInventory),
      std::move(planGenerateInvocations),
      std::move(retainedPlanIncompleteness)}};
}

} // namespace loom::dse
