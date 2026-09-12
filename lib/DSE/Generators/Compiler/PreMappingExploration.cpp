#include "DSE/PreMappingExploration.h"

#include "PreMappingPlanExecution.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
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
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Frontend/Analysis/StructuredProtocolDependencies.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

using pre_mapping::CompletedDataflowSelection;
using pre_mapping::CompletedOwnershipSelection;
using pre_mapping::DataflowSelectionOutcome;
using pre_mapping::emitOwnershipRejections;
using pre_mapping::exploreDataflowCandidates;
using pre_mapping::exploreOwnershipCandidates;
using pre_mapping::invalid;
using pre_mapping::isCancellationReason;
using pre_mapping::mergeReferences;
using pre_mapping::OwnershipSelectionOutcome;
using pre_mapping::planningDispositionForIncomplete;

class WorkTimer final {
public:
  explicit WorkTimer(PreMappingWorkCounter &counter)
      : counter_(counter), start_(std::chrono::steady_clock::now()) {}

  ~WorkTimer() {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start_)
                             .count();
    if (elapsed <= 0)
      return;
    const auto delta = static_cast<std::uint64_t>(elapsed);
    if (counter_.elapsedNanoseconds >
        std::numeric_limits<std::uint64_t>::max() - delta)
      counter_.elapsedNanoseconds = std::numeric_limits<std::uint64_t>::max();
    else
      counter_.elapsedNanoseconds += delta;
  }

private:
  PreMappingWorkCounter &counter_;
  std::chrono::steady_clock::time_point start_;
};

void addElapsedNanoseconds(PreMappingWorkCounter &counter,
                           std::uint64_t elapsed) {
  if (counter.elapsedNanoseconds >
      std::numeric_limits<std::uint64_t>::max() - elapsed)
    counter.elapsedNanoseconds = std::numeric_limits<std::uint64_t>::max();
  else
    counter.elapsedNanoseconds += elapsed;
}

void accountEvaluationTiming(
    PreMappingWorkAccounting &accounting,
    const StructuredOwnershipEvaluationTiming &timing) {
  addElapsedNanoseconds(accounting.analyticEvaluations,
                        timing.analyticElapsedNanoseconds);
  addElapsedNanoseconds(accounting.functionalReplays,
                        timing.functionalReplayElapsedNanoseconds);
}

void accumulateEvaluationTiming(StructuredOwnershipEvaluationTiming &total,
                                const StructuredOwnershipEvaluationTiming &part) {
  auto add = [](std::uint64_t &destination, std::uint64_t value) {
    if (destination > std::numeric_limits<std::uint64_t>::max() - value)
      destination = std::numeric_limits<std::uint64_t>::max();
    else
      destination += value;
  };
  add(total.analyticCalls, part.analyticCalls);
  add(total.analyticElapsedNanoseconds, part.analyticElapsedNanoseconds);
  add(total.functionalReplayCalls, part.functionalReplayCalls);
  add(total.functionalReplayElapsedNanoseconds,
      part.functionalReplayElapsedNanoseconds);
}

constexpr llvm::StringLiteral candidateIdentityDescriptor{
    "loom.dse.pre_mapping_candidate_identity.2"};

void appendIdentityU64(std::vector<std::uint8_t> &bytes,
                       std::uint64_t value) {
  for (unsigned index = 0; index != 8; ++index)
    bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

void appendIdentityBytes(std::vector<std::uint8_t> &bytes,
                         llvm::ArrayRef<std::uint8_t> value) {
  appendIdentityU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendIdentityRoot(std::vector<std::uint8_t> &bytes,
                        const ArtifactRootReference &reference) {
  appendIdentityBytes(bytes, encodeArtifactRootReference(reference));
}

void appendIdentityOptionalRoot(
    std::vector<std::uint8_t> &bytes,
    const std::optional<ArtifactRootReference> &reference) {
  bytes.push_back(reference ? 1 : 0);
  if (reference)
    appendIdentityRoot(bytes, *reference);
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

llvm::Expected<std::string> structuredWorkloadEntrySymbol(
    const frontend::StructuredProgramCandidate &source,
    const sim::CanonicalSimulationWorkload &workload) {
  const sim::StructuredProgramSimulationWorkload *structured =
      workload.structuredProgram();
  if (!structured)
    return invalid("pre-Mapping workload is not Structured");
  auto view = source.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(structured->entryRef);
  if (!entity)
    return entity.takeError();
  auto function =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity->operation);
  if (!function || function.isExternal())
    return invalid("Structured workload entry is not a defined LLVM callable");
  return function.getSymName().str();
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

} // namespace

llvm::StringRef toString(PreMappingCandidatePlanningDisposition value) {
  switch (value) {
  case PreMappingCandidatePlanningDisposition::Retained:
    return "retained";
  case PreMappingCandidatePlanningDisposition::HeuristicPruned:
    return "heuristic_pruned";
  case PreMappingCandidatePlanningDisposition::CoordinateBudget:
    return "coordinate_budget";
  case PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget:
    return "program_materialization_budget";
  case PreMappingCandidatePlanningDisposition::AnalyticEvaluationBudget:
    return "analytic_evaluation_budget";
  case PreMappingCandidatePlanningDisposition::FunctionalReplayBudget:
    return "functional_replay_budget";
  case PreMappingCandidatePlanningDisposition::DataflowPromotionBudget:
    return "dataflow_promotion_budget";
  case PreMappingCandidatePlanningDisposition::MappingPairBudget:
    return "mapping_pair_budget";
  case PreMappingCandidatePlanningDisposition::ExactGateRejected:
    return "exact_gate_rejected";
  case PreMappingCandidatePlanningDisposition::Unsupported:
    return "unsupported";
  case PreMappingCandidatePlanningDisposition::Unknown:
    return "unknown";
  case PreMappingCandidatePlanningDisposition::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown pre-Mapping candidate disposition");
}

llvm::Expected<ComponentViewDigest> computePreMappingCandidateIdentity(
    const PreMappingCandidatePlanningRecord &record,
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ComponentViewDigest &frontierPolicyDigest) {
  // Invocation roots and policy are provenance context, not candidate
  // identity. Keep the parameters at this API boundary so callers can still
  // validate the exact invocation tuple around the semantic join.
  (void)sourceProgram;
  (void)fabric;
  (void)workload;
  (void)runtimeInput;
  (void)frontierPolicyDigest;
  std::vector<std::uint8_t> bytes;
  appendIdentityOptionalRoot(bytes, record.structuredProgram);
  appendIdentityOptionalRoot(bytes, record.canonicalDataflow);

  std::vector<std::vector<std::uint8_t>> roots;
  roots.reserve(record.ownedProtocolRoots.size());
  for (const frontend::StructuredEntityRef &root : record.ownedProtocolRoots)
    roots.push_back(frontend::encodeStructuredEntityRef(root));
  llvm::sort(roots);
  appendIdentityU64(bytes, roots.size());
  for (const std::vector<std::uint8_t> &root : roots)
    appendIdentityBytes(bytes, root);

  bytes.push_back(record.projection ? 1 : 0);
  if (record.projection)
    appendIdentityBytes(bytes, record.projection->identity.bytes());
  // Logical-domain facts and verified spectrum classifications are evaluation
  // evidence. They deliberately do not alter the stable candidate identity.
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(candidateIdentityDescriptor.data()),
       candidateIdentityDescriptor.size()},
      bytes);
}

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
  if (llvm::Error error = validatePreMappingFrontierPolicy(options.frontier))
    return std::move(error);
  const StructuredOwnershipSelectionMode requestedPlannerMode =
      options.ownership.selectionMode;
  // BenefitQualified is retained as a compatibility spelling at the API
  // boundary, but it must not select the legacy unbounded planner. All
  // callers therefore share the same bounded semantic frontier and its
  // evidence contract.
  const StructuredOwnershipSelectionMode plannerMode =
      StructuredOwnershipSelectionMode::SemanticConformance;
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
  auto entrySymbol =
      structuredWorkloadEntrySymbol(compilation.structuredProgram, workload);
  if (!entrySymbol)
    return entrySymbol.takeError();
  auto systemView = fabric::requireSystemRoot(fabric.view());
  if (!systemView)
    return systemView.takeError();
  auto workloadReference =
      sim::publishSimulationWorkload(workload, artifactStore);
  if (!workloadReference)
    return workloadReference.takeError();
  auto runtimeInputReference =
      sim::publishSimulationRuntimeInput(runtimeInput, artifactStore);
  if (!runtimeInputReference)
    return runtimeInputReference.takeError();

  auto frontierPolicyDigest = options.frontier.digest();
  if (!frontierPolicyDigest)
    return frontierPolicyDigest.takeError();
  PreMappingWorkAccounting frontierAccounting =
      makePreMappingWorkAccounting(options.frontier.budget);
  std::optional<std::uint64_t> sourceHostOnlyRuntimePicoseconds;
  std::optional<std::uint64_t> sourceHostOnlyLeafExecutions;
  const auto cancelledBeforePlanning = [&]() -> PreMappingExplorationOutcome {
    IncompletePreMappingExploration result;
    result.reason = DsePlanIncompleteReason{
        CandidateGeneratorIncompleteReason::CancelledOrTimeout};
    result.checkpoint = PreMappingCheckpoint{
        PreMappingCheckpointBoundary::CoordinatePlanning,
        result.reason,
        *sourceReference,
        fabric.reference(),
        *workloadReference,
        *runtimeInputReference,
        *frontierPolicyDigest,
        frontierAccounting,
        {},
        0,
        false,
        {},
        {},
    };
    result.sourceHostOnlyRuntimePicoseconds = sourceHostOnlyRuntimePicoseconds;
    return result;
  };
  // The source-only baseline is a required pair-level decision input. It is
  // deliberately collected once before the DSE deadline can cancel candidate
  // exploration; the deadline still governs every subsequent frontier and
  // provider dispatch.
  ++frontierAccounting.sourceObservations.planned;
  ++frontierAccounting.sourceObservations.reserved;
  evaluation::models::StructuredEvaluationInvocationCache evaluationCache;
  auto sourceObservations =
      [&]() -> llvm::Expected<sim::NativeStructuredProgramObservations> {
    WorkTimer timer(frontierAccounting.sourceObservations);
    auto observations = sim::executeNativeStructuredProgram(
        compilation.structuredProgram, workload, runtimeInput);
    if (!observations)
      return observations.takeError();
    evaluation::models::StructuredEvaluationInvocationCacheScope cacheScope(
        evaluationCache);
    const evaluation::models::StructuredFabricAnalyticInvocation invocation{
        *workloadReference, *runtimeInputReference,        workload,
        runtimeInput,       compilation.structuredProgram, *observations};
    if (llvm::Error error =
            evaluation::models::primeStructuredFabricAnalyticResult(
                *sourceReference,
                {compilation.structuredProgram,
                 nullptr,
                 {},
                 {},
                 &*observations},
                invocation, fabric, config, artifactStore))
      return std::move(error);
    auto estimate = evaluation::models::lookupStructuredFabricAnalyticEstimate(
        *sourceReference, fabric.reference(), *workloadReference,
        *runtimeInputReference, config, evaluationCache);
    if (!estimate)
      return estimate.takeError();
    if (*estimate) {
      sourceHostOnlyRuntimePicoseconds = (**estimate).runtimePicoseconds;
      sourceHostOnlyLeafExecutions = (**estimate).hostDynamicLeafExecutions;
    }
    return std::move(*observations);
  }();
  if (!sourceObservations)
    return sourceObservations.takeError();
  ++frontierAccounting.sourceObservations.consumed;
  if (options.executionControl.stopRequested())
    return cancelledBeforePlanning();
  StructuredOwnershipSharedEvaluation sharedEvaluation(*sourceObservations,
                                                       evaluationCache);

  std::vector<frontend::StructuredEntityRef> protocolRootSelection(
      options.ownership.protocolCallableRoots.begin(),
      options.ownership.protocolCallableRoots.end());
  if (protocolRootSelection.empty()) {
    llvm::SmallVector<llvm::StringRef> workloadEntry{*entrySymbol};
    auto resolved = frontend::resolveDefinedLlvmCallables(
        compilation.structuredProgram, workloadEntry);
    if (!resolved)
      return resolved.takeError();
    protocolRootSelection = std::move(*resolved);
  }
  auto callableSymbols = protocolCallableSymbols(
      compilation.structuredProgram, protocolRootSelection);
  if (!callableSymbols)
    return callableSymbols.takeError();
  auto sourceProtocolRoots = resolveProtocolCallableRoots(
      compilation.structuredProgram, *callableSymbols);
  if (!sourceProtocolRoots)
    return sourceProtocolRoots.takeError();
  auto protocolDependencyProjection =
      frontend::analysis::projectStructuredProtocolDependencyProjection(
          compilation.structuredProgram, *sourceProtocolRoots);
  if (!protocolDependencyProjection)
    return protocolDependencyProjection.takeError();
  std::vector<frontend::analysis::StructuredProtocolDependency>
      protocolDependencies = protocolDependencyProjection->presentDependencies();
  auto projectedRootActivity =
      evaluation::models::projectStructuredScopeActivity(
          compilation.structuredProgram, *sourceObservations,
          *sourceProtocolRoots);
  if (!projectedRootActivity)
    return projectedRootActivity.takeError();
  std::vector<PreMappingProtocolRootActivity> protocolRootActivity;
  std::vector<PreMappingRootActivity> frontierRootActivity;
  protocolRootActivity.reserve(projectedRootActivity->size());
  frontierRootActivity.reserve(projectedRootActivity->size());
  for (const auto &activity : *projectedRootActivity) {
    protocolRootActivity.push_back({activity.scope, activity.dynamicActivations,
                                    activity.dynamicLeafExecutions});
    frontierRootActivity.push_back({activity.scope,
                                    activity.dynamicActivations,
                                    activity.dynamicLeafExecutions});
  }
  llvm::Expected<PreMappingCoordinatePlan> coordinatePlan = [&]() {
    WorkTimer timer(frontierAccounting.coordinates);
    return buildPreMappingCoordinatePlan(
        *sourceProtocolRoots, *protocolDependencyProjection,
        frontierRootActivity, options.frontier, frontierAccounting);
  }();
  if (!coordinatePlan)
    return coordinatePlan.takeError();
  std::vector<PreMappingCandidatePlanningRecord> candidateInventory;
  const auto incompleteAt =
      [&](PreMappingCheckpointBoundary boundary,
          DsePlanIncompleteReason reason,
          llvm::ArrayRef<ArtifactRootReference> retained,
          std::vector<ArtifactRootReference> evidence,
          std::vector<DsePlanGenerateInvocationRecords> invocations)
      -> PreMappingExplorationOutcome {
    IncompletePreMappingExploration result;
    result.reason = std::move(reason);
    result.completeness.domainComplete =
        !coordinatePlan->truncated && sourceProtocolRoots->size() <= 4;
    result.completeness.budgetComplete = false;
    result.completeness.providerComplete = false;
    result.completeness.evidenceComplete = false;
    result.completeness.selectionComplete = false;
    result.retainedEvidence = std::move(evidence);
    result.planGenerateInvocations = std::move(invocations);
    const bool cancellation = isCancellationReason(result.reason);
    for (PreMappingWorkCounter *counter : {
             &frontierAccounting.sourceObservations,
             &frontierAccounting.coordinates,
             &frontierAccounting.programMaterializations,
             &frontierAccounting.analyticEvaluations,
             &frontierAccounting.functionalReplays,
             &frontierAccounting.dataflowPromotions,
             &frontierAccounting.mappingPairs}) {
      if (counter->planned < counter->consumed)
        counter->planned = counter->consumed;
      counter->reserved = counter->planned;
      const std::uint64_t available = counter->reserved - counter->consumed;
      const std::uint64_t settledRejected =
          std::min(available, counter->rejected);
      const std::uint64_t remainingAfterRejected =
          available - settledRejected;
      const std::uint64_t settledCancelled =
          std::min(remainingAfterRejected, counter->cancelled);
      const std::uint64_t previouslySettled =
          settledRejected + settledCancelled;
      const std::uint64_t unsettled = available - previouslySettled;
      counter->rejected = settledRejected;
      counter->cancelled = settledCancelled;
      if (cancellation)
        counter->cancelled += unsettled;
      else
        counter->rejected += unsettled;
    }
    result.checkpoint = PreMappingCheckpoint{
        boundary, result.reason, *sourceReference, fabric.reference(),
        *workloadReference, *runtimeInputReference, *frontierPolicyDigest,
        frontierAccounting, result.completeness,
        coordinatePlan->eligibleCoordinateCount, coordinatePlan->truncated,
        std::vector<ArtifactRootReference>(retained.begin(), retained.end()),
        candidateInventory};
    result.sourceHostOnlyRuntimePicoseconds = sourceHostOnlyRuntimePicoseconds;
    return result;
  };
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
    std::optional<PreMappingCandidateProjection> projection;
    std::optional<PreMappingTemporalWitness> temporalWitness;
    std::vector<PreMappingSpectrumSeedKind> seedKinds;
    std::optional<PreMappingScheduleIntent> scheduleIntent;
    std::optional<std::size_t> planningRecordIndex;
    bool derivationsIncluded = false;
    std::optional<std::uint64_t> hostDynamicLeafExecutions = std::nullopt;
    std::vector<evaluation::models::AnalyticLaunchEstimate> launchEstimates;
  };
  std::vector<RetainedOwnershipSelection> generations;
  std::vector<RetainedOwnershipAlternative> semanticAlternatives;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::vector<StructuredOwnershipFinalizationRejection> finalizationRejections;
  std::vector<DsePlanGenerateInvocationRecords> planGenerateInvocations;
  std::vector<RetainedDsePlanIncompleteness> retainedPlanIncompleteness;
  StructuredOwnershipEvaluationTiming evaluationTiming;
  bool ownershipProviderIncomplete = false;
  std::optional<DsePlanIncompleteReason> ownershipIncompleteReason;
  bool dataflowProviderIncomplete = false;
  std::optional<DsePlanIncompleteReason> dataflowIncompleteReason;
  const auto deriveNoFeasibleCompleteness =
      [&]() -> llvm::Expected<PreMappingSearchCompleteness> {
    PreMappingSearchCompleteness completeness;
    // A coordinate the exact gate rejected, or whose completed generations
    // produced no candidate, was fully adjudicated; incomplete generations
    // are excluded separately through the provider flags below. An ownership
    // domain that planned coordinates yet kept no inventory at all was
    // likewise adjudicated before any candidate existed.
    const bool ownershipProvenInfeasible =
        candidateInventory.empty() && !dispositions.empty();
    const bool allExactGateRejected =
        ownershipProvenInfeasible ||
        (!candidateInventory.empty() &&
         llvm::all_of(candidateInventory,
                      [](const PreMappingCandidatePlanningRecord &record) {
                        return record.disposition ==
                                   PreMappingCandidatePlanningDisposition::
                                       ExactGateRejected ||
                               record.disposition ==
                                   PreMappingCandidatePlanningDisposition::
                                       Unsupported;
                      }));
    const bool dependenciesComplete = llvm::none_of(
        protocolDependencyProjection->relations, [](const auto &relation) {
          return relation.knowledge ==
                 frontend::analysis::
                     StructuredProtocolDependencyKnowledge::Unknown;
        });
    completeness.domainComplete =
        allExactGateRejected && !coordinatePlan->truncated &&
        sourceProtocolRoots->size() <= 4 && dependenciesComplete;
    if (completeness.domainComplete) {
      auto recall = evaluatePreMappingShadowRecall(
          sourceProtocolRoots->size(), *coordinatePlan);
      if (!recall)
        return recall.takeError();
      completeness.domainComplete = recall->missingSubsets.empty();
    }
    completeness.budgetComplete = allExactGateRejected;
    completeness.providerComplete =
        allExactGateRejected && retainedPlanIncompleteness.empty() &&
        !ownershipProviderIncomplete && !dataflowProviderIncomplete;
    completeness.evidenceComplete = allExactGateRejected;
    completeness.selectionComplete = allExactGateRejected;
    return completeness;
  };
  const auto retainedSemanticCandidates = [&]() {
    std::vector<ArtifactRootReference> retained;
    for (const RetainedOwnershipAlternative &alternative :
         semanticAlternatives)
      retained.insert(retained.end(), alternative.references.begin(),
                      alternative.references.end());
    llvm::sort(retained, artifactRootReferenceLess);
    retained.erase(std::unique(retained.begin(), retained.end()),
                   retained.end());
    return retained;
  };
  const auto cancelledAt =
      [&](PreMappingCheckpointBoundary boundary,
          llvm::ArrayRef<ArtifactRootReference> retained)
      -> PreMappingExplorationOutcome {
    return incompleteAt(
        boundary,
        DsePlanIncompleteReason{
            CandidateGeneratorIncompleteReason::CancelledOrTimeout},
        retained, std::move(satisfiedEvidence),
        std::move(planGenerateInvocations));
  };

  const auto ownedRootsFor = [&](llvm::ArrayRef<std::size_t> ordinals) {
    std::vector<frontend::StructuredEntityRef> roots;
    roots.reserve(ordinals.size());
    for (std::size_t ordinal : ordinals)
      roots.push_back((*sourceProtocolRoots)[ordinal]);
    return roots;
  };
  const auto ownedRootsForCandidate =
      [&](const frontend::StructuredProgramCandidate &candidate,
          llvm::ArrayRef<std::size_t> ordinals)
      -> llvm::Expected<std::vector<frontend::StructuredEntityRef>> {
    std::vector<std::string> symbols;
    symbols.reserve(ordinals.size());
    for (std::size_t ordinal : ordinals) {
      if (ordinal >= callableSymbols->size())
        return invalid("pre-Mapping protocol root ordinal is out of range");
      symbols.push_back((*callableSymbols)[ordinal]);
    }
    return resolveProtocolCallableRoots(candidate, symbols);
  };
  const auto addBudgetRecord =
      [&](const PreMappingCoordinate &coordinate,
        PreMappingCandidatePlanningDisposition disposition) {
        PreMappingCandidatePlanningRecord record{
            std::nullopt, std::nullopt,
            ownedRootsFor(coordinate.ownedProtocolOrdinals),
            coordinate.seedKinds, coordinate.projection, std::nullopt,
            disposition, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt};
        record.scheduleIntent = coordinate.scheduleIntent;
        candidateInventory.push_back(std::move(record));
      };

  if (coordinatePlan->truncated)
    candidateInventory.push_back(PreMappingCandidatePlanningRecord{
        std::nullopt, std::nullopt, {}, {}, std::nullopt, std::nullopt,
        PreMappingCandidatePlanningDisposition::CoordinateBudget,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt});

  if (plannerMode ==
      StructuredOwnershipSelectionMode::SemanticConformance) {
    struct SemanticBeamState final {
      frontend::StructuredCompilation *parent = nullptr;
      bool parentFunctionallyVerified = true;
      ArtifactRootReference rankReference;
      std::optional<std::uint64_t> estimatedRuntimePicoseconds;
      bool usesLogicalThreadDomain = false;
      bool hasPreOwnershipParallelDomain = false;
      std::vector<StructuredOwnershipDerivation> ownershipPrefix;
      std::vector<StructuredExecutionShapeDerivation> executionShapePrefix;
      std::vector<StructuredSpecialMathAccuracyDerivation> specialMathPrefix;
      std::vector<StructuredScheduleDerivation> schedulePrefix;
      std::vector<StructuredMemoryCommunicationDerivation> memoryPrefix;
    };

    const auto remaining = [](const PreMappingWorkCounter &counter) {
      return counter.planned >= counter.limit ? 0
                                               : counter.limit - counter.planned;
    };
    const auto settleReservation = [](PreMappingWorkCounter &counter,
                                      std::uint64_t grant,
                                      std::uint64_t consumedBefore,
                                      llvm::StringRef boundary) -> llvm::Error {
      if (counter.consumed < consumedBefore)
        return invalid(boundary + " consumption regressed");
      const std::uint64_t consumed = counter.consumed - consumedBefore;
      if (consumed > grant)
        return invalid(boundary + " exceeded its admitted work grant");
      const std::uint64_t unused = grant - consumed;
      if (unused > std::numeric_limits<std::uint64_t>::max() -
                       counter.rejected)
        return invalid(boundary + " rejected-work ledger overflows");
      // The grant was reserved before dispatch. Work that the provider did
      // not consume is a rejected unit, not an erased reservation; retaining
      // it makes the parent ledger auditable and prevents cheap gates from
      // disappearing from the accounting.
      counter.rejected += unused;
      return llvm::Error::success();
    };
    for (const auto indexedCoordinate :
         llvm::enumerate(coordinatePlan->coordinates)) {
      const PreMappingCoordinate &coordinate = indexedCoordinate.value();
      if (options.executionControl.stopRequested())
        return cancelledAt(PreMappingCheckpointBoundary::OwnershipGeneration,
                           retainedSemanticCandidates());
      if (coordinate.projection.exactGate ==
          PreMappingExactGateDisposition::Rejected) {
        addBudgetRecord(coordinate,
                        PreMappingCandidatePlanningDisposition::ExactGateRejected);
        continue;
      }

      const std::uint64_t remainingCoordinates =
          coordinatePlan->coordinates.size() - indexedCoordinate.index();
      const std::uint64_t depthCount = std::max<std::uint64_t>(
          1, coordinate.ownedProtocolOrdinals.size());
      const auto fairGrant = [&](const PreMappingWorkCounter &counter,
                                 std::uint64_t minimum) {
        const std::uint64_t available = remaining(counter);
        if (available == 0)
          return std::uint64_t{0};
        return std::min(
            available,
            std::max(minimum, available / remainingCoordinates));
      };
      std::uint64_t coordinateProgramAllowance =
          fairGrant(frontierAccounting.programMaterializations,
                    1 + depthCount * 6);
      std::uint64_t coordinateAnalyticAllowance =
          fairGrant(frontierAccounting.analyticEvaluations, depthCount * 3);
      std::uint64_t coordinateFunctionalAllowance =
          fairGrant(frontierAccounting.functionalReplays, depthCount);

      // Importing a fresh parent is the first candidate-specific materialization
      // boundary. Reserve it before cloning the MLIR program so the coordinate
      // planner cannot consume an unbounded number of parents before any
      // provider-level grant is checked.
      if (coordinateProgramAllowance == 0 ||
          frontierAccounting.programMaterializations.planned >=
              frontierAccounting.programMaterializations.limit) {
        addBudgetRecord(
            coordinate,
            PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget);
        continue;
      }
      ++frontierAccounting.programMaterializations.planned;
      ++frontierAccounting.programMaterializations.reserved;
      --coordinateProgramAllowance;

      llvm::Expected<frontend::StructuredProgramCandidate> parentProgram =
          [&]() {
            WorkTimer timer(frontierAccounting.programMaterializations);
            return frontend::importStructuredProgram(
                sourceCompilation.structuredProgram.identity(),
                sourceCompilation.structuredProgram.canonicalBytes());
          }();
      if (!parentProgram)
        return parentProgram.takeError();
      ++frontierAccounting.programMaterializations.consumed;
      parentStorage.push_back(std::make_unique<frontend::StructuredCompilation>(
          frontend::StructuredCompilation{
              sourceCompilation.fabric, sourceCompilation.staticGlobalMemory,
              std::move(*parentProgram), sourceCompilation.sourceProvenance,
              sourceCompilation.candidateHints}));
      frontend::StructuredCompilation *coordinateParent =
          parentStorage.back().get();
      std::vector<SemanticBeamState> beam;
      if (beam.empty())
        beam.push_back(SemanticBeamState{coordinateParent,
                                         true,
                                         *sourceReference,
                                         std::nullopt,
                                         false,
                                         false,
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}});
      bool coordinateCompleted = false;
      const std::size_t coordinateRejectionsBegin =
          finalizationRejections.size();
      for (std::size_t depth = 0; depth != depthCount && !beam.empty();
           ++depth) {
        if (options.executionControl.stopRequested())
          return cancelledAt(PreMappingCheckpointBoundary::OwnershipGeneration,
                             retainedSemanticCandidates());
        std::vector<SemanticBeamState> nextBeam;
        std::vector<RetainedOwnershipAlternative> terminal;
        for (SemanticBeamState &state : beam) {
          const std::uint64_t requestedBeam = options.frontier.beamWidth(depth);
          const std::uint64_t desiredExpansion =
              requestedBeam > std::numeric_limits<std::uint64_t>::max() / 2
                  ? std::numeric_limits<std::uint64_t>::max()
                  : requestedBeam * 2;
          const bool hasChildMaterialization = depth + 1 != depthCount;
          const bool requireFunctionalReplay = !hasChildMaterialization;
          const std::uint64_t materializationUnitsPerExpansion =
              hasChildMaterialization ? 6 : 5;
          // Candidate generation and survivor promotion are separate
          // boundaries.  The analytic frontier may inspect a bounded
          // expansion, while functional replay is reserved only for the
          // smaller beam that can reach the next transformation layer.
          const std::uint64_t expansionLimit = std::min(
              {desiredExpansion,
               remaining(frontierAccounting.programMaterializations) /
                   materializationUnitsPerExpansion,
               remaining(frontierAccounting.analyticEvaluations) / 2,
               coordinateProgramAllowance /
                   materializationUnitsPerExpansion,
               coordinateAnalyticAllowance / 2});
          std::uint64_t beamWidth =
              std::min(requestedBeam, expansionLimit);
          if (requireFunctionalReplay)
            beamWidth = std::min(
                {beamWidth,
                 remaining(frontierAccounting.functionalReplays),
                 coordinateFunctionalAllowance});
          if (expansionLimit == 0 || beamWidth == 0) {
            PreMappingCandidatePlanningDisposition disposition =
                PreMappingCandidatePlanningDisposition::FunctionalReplayBudget;
            if (remaining(frontierAccounting.programMaterializations) < 5)
              disposition = PreMappingCandidatePlanningDisposition::
                  ProgramMaterializationBudget;
            else if (remaining(frontierAccounting.analyticEvaluations) == 0)
              disposition = PreMappingCandidatePlanningDisposition::
                  AnalyticEvaluationBudget;
            addBudgetRecord(coordinate, disposition);
            break;
          }
          // Ownership decision screening is cheaper than analytic evaluation
          // and functional replay, but exact rejection must not consume a
          // downstream publication slot. Give it a distinct, fairly bounded
          // grant. Special-math closure has the same distinction, while the
          // remaining three transform layers retain the published expansion
          // width. A non-terminal survivor also consumes one child clone.
          const std::uint64_t childMaterializationGrant =
              hasChildMaterialization ? expansionLimit : 0;
          if (expansionLimit >
              (std::numeric_limits<std::uint64_t>::max() -
               childMaterializationGrant) /
                  3)
            return invalid("compiler frontier materialization grant overflows");
          const std::uint64_t fixedProgramGrant =
              expansionLimit * 3 + childMaterializationGrant;
          if (coordinateProgramAllowance < fixedProgramGrant ||
              coordinateProgramAllowance - fixedProgramGrant <
                  expansionLimit * 2)
            return invalid("compiler frontier lost its bounded screening "
                           "grants");
          const std::uint64_t diversityWidth =
              std::max<std::uint64_t>(
                  1, options.frontier.diversityCandidateCount);
          const std::uint64_t screeningAttemptTarget =
              expansionLimit > std::numeric_limits<std::uint64_t>::max() /
                                   diversityWidth
                  ? std::numeric_limits<std::uint64_t>::max()
                  : expansionLimit * diversityWidth;
          const std::uint64_t screeningAllowance =
              coordinateProgramAllowance - fixedProgramGrant;
          const std::uint64_t ownershipAttemptLimit = std::min(
              screeningAttemptTarget, screeningAllowance / 2);
          const std::uint64_t specialMathAttemptLimit = std::min(
              screeningAttemptTarget,
              screeningAllowance - ownershipAttemptLimit);
          if (ownershipAttemptLimit < expansionLimit ||
              specialMathAttemptLimit < expansionLimit)
            return invalid("compiler frontier screening grant is smaller "
                           "than its publication width");
          const std::uint64_t programGrant = fixedProgramGrant +
                                             ownershipAttemptLimit +
                                             specialMathAttemptLimit;
          if (expansionLimit >
              std::numeric_limits<std::uint64_t>::max() / 2)
            return invalid("compiler frontier analytic grant overflows");
          const std::uint64_t analyticGrant = expansionLimit * 2;
          coordinateProgramAllowance -= programGrant;
          coordinateAnalyticAllowance -= analyticGrant;
          const std::uint64_t functionalGrant =
              requireFunctionalReplay ? beamWidth : 0;
          coordinateFunctionalAllowance -= functionalGrant;
          const std::uint64_t programConsumedBefore =
              frontierAccounting.programMaterializations.consumed;
          const std::uint64_t analyticConsumedBefore =
              frontierAccounting.analyticEvaluations.consumed;
          const std::uint64_t functionalConsumedBefore =
              frontierAccounting.functionalReplays.consumed;
          frontierAccounting.programMaterializations.planned += programGrant;
          frontierAccounting.programMaterializations.reserved += programGrant;
          // Reserve one provider acquisition and one cached planner lookup for
          // every exposed survivor before either boundary begins.
          frontierAccounting.analyticEvaluations.planned += analyticGrant;
          frontierAccounting.analyticEvaluations.reserved += analyticGrant;
          frontierAccounting.functionalReplays.planned += functionalGrant;
          frontierAccounting.functionalReplays.reserved += functionalGrant;

          auto parentReference = frontend::publishStructuredProgram(
              state.parent->structuredProgram, artifactStore);
          if (!parentReference)
            return parentReference.takeError();
          llvm::ArrayRef<std::string> generationSymbols;
          if (!coordinate.ownedProtocolOrdinals.empty()) {
            const std::size_t rootDepth =
                depth % coordinate.ownedProtocolOrdinals.size();
            generationSymbols = llvm::ArrayRef<std::string>(*callableSymbols)
                                    .slice(
                                        coordinate.ownedProtocolOrdinals[rootDepth],
                                        1);
          }
          auto protocolRoots = resolveProtocolCallableRoots(
              state.parent->structuredProgram, generationSymbols);
          if (!protocolRoots)
            return protocolRoots.takeError();
          StructuredOwnershipExplorationOptions generationOptions =
              options.ownership;
          generationOptions.selectionMode = plannerMode;
          generationOptions.protocolCallableRoots = std::move(*protocolRoots);
          generationOptions.selection.k = beamWidth;
          const StructuredScheduleGenerationIntent scheduleIntent =
              coordinate.scheduleIntent ==
                          PreMappingScheduleIntent::TemporalReuse &&
                      state.hasPreOwnershipParallelDomain
                  ? StructuredScheduleGenerationIntent::
                        ForbidLogicalThreadDomain
                  : coordinate.scheduleIntent ==
                        PreMappingScheduleIntent::TemporalReuse
                  ? StructuredScheduleGenerationIntent::
                        RequireLogicalThreadDomain
                  : StructuredScheduleGenerationIntent::Balanced;
          const StructuredOwnershipGenerationIntent ownershipIntent =
              state.hasPreOwnershipParallelDomain
                  ? StructuredOwnershipGenerationIntent::
                        RequireLogicalThreadDomain
                  : StructuredOwnershipGenerationIntent::Balanced;
          llvm::Expected<OwnershipSelectionOutcome> explored =
              exploreOwnershipCandidates(
                  *state.parent, *parentReference,
                  sourceCompilation.structuredProgram, *sourceReference,
                  workload, *workloadReference, runtimeInput,
                  *runtimeInputReference, fabric, config, generationOptions,
                  ownershipIntent, scheduleIntent, ownershipAttemptLimit,
                  specialMathAttemptLimit, expansionLimit,
                  state.parentFunctionallyVerified,
                  requireFunctionalReplay,
                  options.executionControl, artifactStore, blobStore,
                  &sharedEvaluation);
          if (!explored)
            return explored.takeError();
          if (auto *incomplete =
                  std::get_if<IncompletePreMappingExploration>(&*explored)) {
            accountEvaluationTiming(frontierAccounting,
                                    incomplete->evaluationTiming);
            accumulateEvaluationTiming(evaluationTiming,
                                       incomplete->evaluationTiming);
            mergeReferences(incomplete->retainedEvidence, satisfiedEvidence);
            planGenerateInvocations.insert(
                planGenerateInvocations.end(),
                std::make_move_iterator(
                    incomplete->planGenerateInvocations.begin()),
                std::make_move_iterator(
                    incomplete->planGenerateInvocations.end()));
            ownershipProviderIncomplete = true;
            if (!ownershipIncompleteReason)
              ownershipIncompleteReason = incomplete->reason;
            if (incomplete->resolvedDseConfigViewDigest &&
                incomplete->planNodeOrdinal)
              retainedPlanIncompleteness.push_back(
                  RetainedDsePlanIncompleteness{
                      *incomplete->resolvedDseConfigViewDigest,
                      *incomplete->planNodeOrdinal, incomplete->reason});
            frontierAccounting.programMaterializations.consumed +=
                incomplete->programMaterializations;
            frontierAccounting.analyticEvaluations.consumed +=
                incomplete->analyticEvaluations;
            frontierAccounting.functionalReplays.consumed +=
                incomplete->functionalReplays;
            if (llvm::Error error = settleReservation(
                    frontierAccounting.programMaterializations, programGrant,
                    programConsumedBefore, "compiler materialization"))
              return std::move(error);
            if (llvm::Error error = settleReservation(
                    frontierAccounting.analyticEvaluations, analyticGrant,
                    analyticConsumedBefore, "compiler analytic evaluation"))
              return std::move(error);
            if (llvm::Error error = settleReservation(
                    frontierAccounting.functionalReplays, functionalGrant,
                    functionalConsumedBefore, "compiler functional replay"))
              return std::move(error);
            addBudgetRecord(
                coordinate, planningDispositionForIncomplete(
                                incomplete->reason));
            continue;
          }

          auto completed =
              std::get<CompletedOwnershipSelection>(std::move(*explored));
          accountEvaluationTiming(frontierAccounting,
                                  completed.evaluationTiming);
          accumulateEvaluationTiming(evaluationTiming,
                                     completed.evaluationTiming);
          if (ownershipIntent == StructuredOwnershipGenerationIntent::
                                     RequireLogicalThreadDomain)
            emitOwnershipRejections(completed.dispositions);
          if (completed.programMaterializations > programGrant ||
              completed.analyticEvaluations > analyticGrant ||
              completed.functionalReplays > functionalGrant)
            return invalid("compiler provider exceeded its admitted work grant: " +
                           llvm::Twine(completed.programMaterializations) + "/" +
                           llvm::Twine(programGrant) + ", analytic=" +
                           llvm::Twine(completed.analyticEvaluations) + "/" +
                           llvm::Twine(analyticGrant) + ", functional=" +
                           llvm::Twine(completed.functionalReplays) + "/" +
                           llvm::Twine(functionalGrant));
          frontierAccounting.programMaterializations.consumed +=
              completed.programMaterializations;
          frontierAccounting.analyticEvaluations.consumed +=
              completed.analyticEvaluations;
          frontierAccounting.functionalReplays.consumed +=
              completed.functionalReplays;
          mergeReferences(satisfiedEvidence, completed.evidence);
          dispositions.insert(
              dispositions.end(),
              std::make_move_iterator(completed.dispositions.begin()),
              std::make_move_iterator(completed.dispositions.end()));
          finalizationRejections.insert(
              finalizationRejections.end(),
              std::make_move_iterator(
                  completed.finalizationRejections.begin()),
              std::make_move_iterator(completed.finalizationRejections.end()));
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

          for (const ArtifactRootReference &reference :
               generation.preferenceOrder) {
            auto logicalThreadDomain = generation.invocation
                                           ->selectedCandidateHasLogicalThreadDomain(
                                               reference);
            if (!logicalThreadDomain)
              return logicalThreadDomain.takeError();
            const bool usesLogicalThreadDomain = *logicalThreadDomain;
            const bool cumulativeLogicalThreadDomain =
                state.usesLogicalThreadDomain || usesLogicalThreadDomain ||
                state.hasPreOwnershipParallelDomain;
            if (coordinate.scheduleIntent ==
                    PreMappingScheduleIntent::TemporalReuse &&
                depth + 1 == depthCount && !cumulativeLogicalThreadDomain)
              continue;
            if (frontierAccounting.analyticEvaluations.consumed >=
                frontierAccounting.analyticEvaluations.limit)
              return invalid("compiler frontier analytic budget was exceeded");
            ++frontierAccounting.analyticEvaluations.consumed;
            auto estimate =
                evaluation::models::lookupStructuredFabricAnalyticEstimate(
                    reference, fabric.reference(), *workloadReference,
                    *runtimeInputReference, config, evaluationCache);
            if (!estimate)
              return estimate.takeError();
            const auto runtime =
                *estimate ? (**estimate).runtimePicoseconds : std::nullopt;
            RetainedOwnershipAlternative alternative{
                generationIndex,
                {reference},
                state.ownershipPrefix,
                state.executionShapePrefix,
                state.specialMathPrefix,
                state.schedulePrefix,
                state.memoryPrefix,
                coordinate.ownedProtocolOrdinals,
                coordinate.projection.internalDependencyCount,
                coordinate.projection.cutDependencyCount,
                runtime,
                coordinate.projection,
                coordinate.temporalWitness,
                coordinate.seedKinds,
                std::nullopt,
                std::nullopt,
                false,
                *estimate ? std::optional<std::uint64_t>(
                                (**estimate).hostDynamicLeafExecutions)
                          : std::nullopt};
            if (*estimate)
              alternative.launchEstimates = (**estimate).launches;
            if (depth + 1 == depthCount) {
              terminal.push_back(std::move(alternative));
              continue;
            }

            StructuredOwnershipInvocationScope generationScope(
                *generation.invocation);
            if (frontierAccounting.programMaterializations.consumed >=
                frontierAccounting.programMaterializations.limit)
              return invalid("compiler frontier materialization budget was exceeded");
            ++frontierAccounting.programMaterializations.consumed;
            llvm::Expected<SelectedStructuredOwnershipCandidate> selected =
                [&]() {
                  WorkTimer timer(
                      frontierAccounting.programMaterializations);
                  return generation.invocation
                      ->materializeAnalyticContinuationCandidate(
                          reference, artifactStore);
                }();
            if (!selected)
              return selected.takeError();
            SemanticBeamState child{nullptr,
                                    false,
                                    reference,
                                    runtime,
                                    cumulativeLogicalThreadDomain,
                                    state.hasPreOwnershipParallelDomain,
                                    state.ownershipPrefix,
                                    state.executionShapePrefix,
                                    state.specialMathPrefix,
                                    state.schedulePrefix,
                                    state.memoryPrefix};
            child.ownershipPrefix.insert(
                child.ownershipPrefix.end(), selected->derivations.begin(),
                selected->derivations.end());
            child.executionShapePrefix.insert(
                child.executionShapePrefix.end(),
                selected->executionShapeDerivations.begin(),
                selected->executionShapeDerivations.end());
            child.specialMathPrefix.insert(
                child.specialMathPrefix.end(),
                selected->specialMathAccuracyDerivations.begin(),
                selected->specialMathAccuracyDerivations.end());
            child.schedulePrefix.insert(
                child.schedulePrefix.end(),
                selected->scheduleDerivations.begin(),
                selected->scheduleDerivations.end());
            child.memoryPrefix.insert(
                child.memoryPrefix.end(),
                selected->memoryCommunicationDerivations.begin(),
                selected->memoryCommunicationDerivations.end());
            parentStorage.push_back(
                std::make_unique<frontend::StructuredCompilation>(
                    frontend::StructuredCompilation{
                        sourceCompilation.fabric,
                        sourceCompilation.staticGlobalMemory,
                        std::move(selected->candidate.structuredProgram),
                        std::move(selected->candidate.sourceProvenance),
                        sourceCompilation.candidateHints}));
            child.parent = parentStorage.back().get();
            nextBeam.push_back(std::move(child));
          }
          if (llvm::Error error = settleReservation(
                  frontierAccounting.programMaterializations, programGrant,
                  programConsumedBefore, "compiler materialization"))
            return std::move(error);
          if (llvm::Error error = settleReservation(
                  frontierAccounting.analyticEvaluations, analyticGrant,
                  analyticConsumedBefore, "compiler analytic evaluation"))
            return std::move(error);
          if (llvm::Error error = settleReservation(
                  frontierAccounting.functionalReplays, functionalGrant,
                  functionalConsumedBefore, "compiler functional replay"))
            return std::move(error);
        }
        if (depth + 1 == depthCount) {
          for (RetainedOwnershipAlternative &alternative : terminal) {
            PreMappingCandidateProjection projection =
                *alternative.projection;
            if (alternative.estimatedRuntimePicoseconds &&
                projection.estimatedCutTrafficBytes &&
                projection.unknownCutPairCount == 0 &&
                projection.cutUnknownObjectCount == 0) {
              projection.estimateSupport = PreMappingEstimateSupport::Supported;
              projection.estimateConfidence =
                  PreMappingEstimateConfidence::Low;
            } else {
              projection.estimateSupport =
                  PreMappingEstimateSupport::Unsupported;
              projection.estimateConfidence =
                  PreMappingEstimateConfidence::None;
            }
            alternative.projection = projection;
            alternative.scheduleIntent = coordinate.scheduleIntent;
            candidateInventory.push_back(PreMappingCandidatePlanningRecord{
                alternative.references.front(), std::nullopt,
                ownedRootsFor(alternative.ownedProtocolOrdinals),
                alternative.seedKinds, projection,
                alternative.estimatedRuntimePicoseconds,
                PreMappingCandidatePlanningDisposition::HeuristicPruned,
                std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt});
            candidateInventory.back().scheduleIntent =
                coordinate.scheduleIntent;
            candidateInventory.back().hostDynamicLeafExecutions =
                alternative.hostDynamicLeafExecutions;
            candidateInventory.back().launchEstimates =
                alternative.launchEstimates;
            alternative.planningRecordIndex = candidateInventory.size() - 1;
            semanticAlternatives.push_back(std::move(alternative));
          }
          coordinateCompleted = !terminal.empty();
          break;
        }
        llvm::sort(nextBeam, [](const SemanticBeamState &lhs,
                               const SemanticBeamState &rhs) {
          return artifactRootReferenceLess(lhs.rankReference,
                                           rhs.rankReference);
        });
        nextBeam.erase(
            std::unique(nextBeam.begin(), nextBeam.end(),
                        [](const SemanticBeamState &lhs,
                           const SemanticBeamState &rhs) {
                          return lhs.rankReference == rhs.rankReference;
                        }),
            nextBeam.end());
        const std::uint64_t beamLimit = options.frontier.beamWidth(depth);
        std::vector<PreMappingFrontierCandidate> rankedInputs;
        rankedInputs.reserve(nextBeam.size());
        for (const SemanticBeamState &state : nextBeam)
          rankedInputs.push_back({state.rankReference, coordinate.projection,
                                  state.estimatedRuntimePicoseconds,
                                  coordinate.scheduleIntent,
                                  coordinate.seedKinds, std::nullopt});
        auto ranked = selectPreMappingFrontier(
            rankedInputs, beamLimit, options.frontier.diversityCandidateCount);
        if (!ranked)
          return ranked.takeError();
        beam.clear();
        beam.reserve(ranked->preferenceOrder.size());
        for (const ArtifactRootReference &reference :
             ranked->preferenceOrder) {
          auto found = llvm::find_if(nextBeam, [&](const auto &state) {
            return state.rankReference == reference;
          });
          if (found == nextBeam.end())
            return invalid("central frontier selection returned a foreign "
                           "beam candidate");
          beam.push_back(std::move(*found));
        }
      }
      if (!coordinateCompleted) {
        // The exact Fabric is the immutable exact gate applied at candidate
        // finalization: a coordinate whose generations retained such a
        // refusal was rejected by that gate, not left unsupported.
        const bool exactFabricRejected = llvm::any_of(
            llvm::drop_begin(finalizationRejections, coordinateRejectionsBegin),
            [](const StructuredOwnershipFinalizationRejection &rejection) {
              return rejection.rejection.kind ==
                     frontend::SpatialOwnershipCandidateRejectionKind::
                         ExactFabricInadmissible;
            });
        addBudgetRecord(
            coordinate,
            exactFabricRejected
                ? PreMappingCandidatePlanningDisposition::ExactGateRejected
                : PreMappingCandidatePlanningDisposition::Unsupported);
      }
    }
  }
  std::vector<RetainedOwnershipAlternative> alternatives;
  if (semanticAlternatives.empty()) {
    emitOwnershipRejections(dispositions);
    if (!retainedPlanIncompleteness.empty()) {
      const RetainedDsePlanIncompleteness &first =
          retainedPlanIncompleteness.front();
      return incompleteAt(PreMappingCheckpointBoundary::OwnershipGeneration,
                          first.reason, retainedSemanticCandidates(),
                          std::move(satisfiedEvidence),
                          std::move(planGenerateInvocations));
    }
    if (ownershipIncompleteReason)
      return incompleteAt(PreMappingCheckpointBoundary::OwnershipGeneration,
                          *ownershipIncompleteReason,
                          retainedSemanticCandidates(),
                          std::move(satisfiedEvidence),
                          std::move(planGenerateInvocations));
    auto completeness = deriveNoFeasibleCompleteness();
    if (!completeness)
      return completeness.takeError();
    return PreMappingExplorationOutcome{CompletedPreMappingNoFeasibleCandidate{
        std::move(satisfiedEvidence), std::move(planGenerateInvocations),
        *sourceReference, fabric.reference(), *workloadReference,
        *runtimeInputReference, std::move(candidateInventory),
        std::move(*completeness), *frontierPolicyDigest,
        sourceHostOnlyRuntimePicoseconds, std::move(finalizationRejections)}};
  }
    std::vector<PreMappingFrontierCandidate> rankedInputs;
    rankedInputs.reserve(semanticAlternatives.size());
    for (const RetainedOwnershipAlternative &alternative :
         semanticAlternatives) {
      if (alternative.references.size() != 1 || !alternative.projection)
        return invalid("semantic candidate has no exact frontier projection");
      rankedInputs.push_back({alternative.references.front(),
                              *alternative.projection,
                              alternative.estimatedRuntimePicoseconds,
                              alternative.scheduleIntent.value_or(
                                  PreMappingScheduleIntent::Unconstrained),
                              alternative.seedKinds,
                              std::nullopt});
    }
    // Only this selection hands out Mapping slots, so the host-only baseline
    // is applied here. The per-coordinate beam above keeps its regressions:
    // a parent that regresses on its own may still expand into an improving
    // schedule or memory-communication child.
    auto ranked = selectPreMappingFrontier(
        rankedInputs, options.ownership.selection.k,
        options.frontier.diversityCandidateCount,
        options.frontier.spectrumEndpoint, sourceHostOnlyRuntimePicoseconds);
    if (!ranked)
      return ranked.takeError();
    if (ranked->preferenceOrder.size() !=
        ranked->preferenceProjectionIdentities.size())
      return invalid("central frontier selection lost coordinate identity");
    const std::size_t retainedCount = ranked->preferenceOrder.size();
    std::vector<std::pair<ArtifactRootReference, ComponentViewDigest>>
        retainedRepresentatives;
    for (RetainedOwnershipAlternative &alternative : semanticAlternatives) {
      std::optional<std::uint64_t> preferenceRank;
      for (std::size_t rank = 0; rank != retainedCount; ++rank)
        if (ranked->preferenceOrder[rank] ==
                alternative.references.front() &&
            ranked->preferenceProjectionIdentities[rank] ==
                alternative.projection->identity &&
            llvm::find_if(
                retainedRepresentatives, [&](const auto &representative) {
                  return representative.first == alternative.references.front() &&
                         representative.second == alternative.projection->identity;
                }) == retainedRepresentatives.end()) {
          retainedRepresentatives.push_back(
              {alternative.references.front(), alternative.projection->identity});
          preferenceRank = rank;
          break;
        }
      if (!alternative.planningRecordIndex ||
          *alternative.planningRecordIndex >= candidateInventory.size())
        return invalid("semantic candidate lost its planning record");
      PreMappingCandidatePlanningRecord &record =
          candidateInventory[*alternative.planningRecordIndex];
      record.disposition =
          preferenceRank ? PreMappingCandidatePlanningDisposition::Retained
                         : PreMappingCandidatePlanningDisposition::HeuristicPruned;
      record.preferenceRank = preferenceRank;
    }

    alternatives.reserve(retainedCount);
    for (std::size_t rank = 0; rank != retainedCount; ++rank) {
      auto found = llvm::find_if(
          semanticAlternatives, [&](const auto &alternative) {
            return alternative.references.size() == 1 &&
                   alternative.references.front() ==
                       ranked->preferenceOrder[rank] &&
                   alternative.projection &&
                   alternative.projection->identity ==
                       ranked->preferenceProjectionIdentities[rank];
          });
      if (found == semanticAlternatives.end())
        return invalid("central frontier selection returned a foreign "
                       "semantic candidate");
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "retain_dependency_aware_candidate";
            fields["preference_rank"] = rank;
            fields["owned_region_count"] =
                found->ownedProtocolOrdinals.size();
            fields["internal_dependency_weight"] =
                found->internalDependencyWeight;
            fields["cut_dependency_weight"] = found->cutDependencyWeight;
            fields["analytic_runtime_supported"] =
                found->estimatedRuntimePicoseconds.has_value();
            if (found->estimatedRuntimePicoseconds)
              fields["estimated_runtime_ps"] =
                  *found->estimatedRuntimePicoseconds;
          });
      alternatives.push_back(std::move(*found));
    }

  auto assemble = [&](SelectedStructuredOwnershipCandidate candidate,
                      const RetainedOwnershipAlternative &alternative,
                      std::optional<std::size_t> planningRecordOrdinal) {
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
    // The schedule lineage of a selected compilation is the replayable join
    // between a Structured schedule proposal and the Mapping work that the
    // application owner records against this exact program.
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "selected_candidate_lineage";
          if (planningRecordOrdinal)
            fields["planning_record_ordinal"] = *planningRecordOrdinal;
          fields["structured_program"] = formatArtifactIdentityHex(
              candidate.candidate.structuredProgram.identity());
          fields["canonical_dataflow"] = formatArtifactIdentityHex(
              candidate.candidate.canonicalDataflow.identity());
          llvm::json::Array decisions;
          for (const StructuredScheduleDerivation &derivation : schedule) {
            llvm::json::Object entry;
            entry["kind"] = frontend::structuredScheduleDecisionKindSpelling(
                derivation.decision.kind);
            entry["factor"] = derivation.decision.factor;
            entry["loop_ordinal"] = derivation.decision.loop.ordinal;
            entry["parent"] =
                formatArtifactIdentityHex(derivation.parent.artifact);
            entry["child"] =
                formatArtifactIdentityHex(derivation.child.artifact);
            decisions.push_back(std::move(entry));
          }
          fields["schedule_decisions"] = std::move(decisions);
        });
    return SelectedPreMappingCompilation{
        0,
        planningRecordOrdinal,
        frontend::PreMappingCompilation{
            sourceCompilation.fabric, sourceCompilation.staticGlobalMemory,
            std::move(candidate.candidate.structuredProgram),
            std::move(candidate.candidate.sourceProvenance),
            sourceCompilation.candidateHints,
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
  std::vector<std::optional<std::size_t>> selectedPlanningRecords;
  selected.reserve(static_cast<std::size_t>(options.ownership.selection.k));
  StructuredOwnershipTopKSelection dataflowSelectionPolicy =
      options.ownership.selection;
  std::map<ArtifactRootReference, PreMappingMaterializedProjection,
           decltype(&artifactRootReferenceLess)>
      materializedProjections(&artifactRootReferenceLess);
  const auto planningRecordFor =
      [&](const ArtifactRootReference &structuredProgram,
          const ArtifactRootReference &canonicalDataflow,
          std::optional<std::size_t> preferred,
          std::optional<PreMappingScheduleIntent> scheduleIntent)
      -> llvm::Expected<std::size_t> {
    std::optional<std::size_t> found = preferred;
    if (!found) {
      for (auto indexed : llvm::enumerate(candidateInventory)) {
        if (indexed.value().structuredProgram == structuredProgram &&
            (!scheduleIntent ||
             indexed.value().scheduleIntent == scheduleIntent) &&
            (!indexed.value().canonicalDataflow ||
             indexed.value().canonicalDataflow == canonicalDataflow)) {
          found = indexed.index();
          break;
        }
      }
    }
    if (!found || *found >= candidateInventory.size())
      return invalid("selected candidate has no exact planning record");
    if (candidateInventory[*found].structuredProgram != structuredProgram)
      return invalid("selected candidate and planning record disagree");
    if (candidateInventory[*found].canonicalDataflow &&
        candidateInventory[*found].canonicalDataflow != canonicalDataflow) {
      PreMappingCandidatePlanningRecord split = candidateInventory[*found];
      split.canonicalDataflow = canonicalDataflow;
      split.preferenceRank = std::nullopt;
      split.materializedProjection = std::nullopt;
      split.disposition =
          PreMappingCandidatePlanningDisposition::HeuristicPruned;
      split.scheduleIntent = scheduleIntent;
      candidateInventory.push_back(std::move(split));
      found = candidateInventory.size() - 1;
    }
    auto &record = candidateInventory[*found];
    record.canonicalDataflow = canonicalDataflow;
    auto projection = materializedProjections.find(canonicalDataflow);
    if (projection == materializedProjections.end()) {
      llvm::Expected<dataflow::CanonicalDataflowArtifact> imported = [&]() {
        WorkTimer timer(frontierAccounting.programMaterializations);
        return dataflow::importCanonicalDataflow(canonicalDataflow,
                                                 artifactStore);
      }();
      if (!imported)
        return imported.takeError();
      const auto &view = imported->view();
      llvm::Expected<PreMappingMaterializedProjection> derived = [&]() {
        WorkTimer timer(frontierAccounting.programMaterializations);
        return projectPreMappingMaterializedCandidate(view, *systemView,
                                                      *entrySymbol);
      }();
      if (!derived)
        return derived.takeError();
      projection = materializedProjections
                       .emplace(canonicalDataflow, std::move(*derived))
                       .first;
    }
    record.materializedProjection = projection->second;
    // Retain the materialized logical-domain fact for diagnostics, but never
    // infer an endpoint from it. Only a verified SystemMapping schedule may
    // populate record.verifiedSpectrum.
    record.temporalWitness = projection->second.temporalWitness;
    return *found;
  };
  const auto reserveMappingAndProgramMaterialization =
      [&](std::optional<std::size_t> planningRecord) -> bool {
        if (frontierAccounting.mappingPairs.planned >=
            frontierAccounting.mappingPairs.limit) {
          if (planningRecord)
            candidateInventory[*planningRecord].disposition =
                PreMappingCandidatePlanningDisposition::MappingPairBudget;
          return false;
        }
        if (frontierAccounting.programMaterializations.planned >=
            frontierAccounting.programMaterializations.limit) {
          if (planningRecord)
            candidateInventory[*planningRecord].disposition =
                PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget;
          return false;
        }
        ++frontierAccounting.mappingPairs.planned;
        ++frontierAccounting.mappingPairs.reserved;
        ++frontierAccounting.programMaterializations.planned;
        ++frontierAccounting.programMaterializations.reserved;
        return true;
      };
  for (const RetainedOwnershipAlternative &alternative : alternatives) {
    if (options.executionControl.stopRequested())
      return cancelledAt(PreMappingCheckpointBoundary::DataflowPromotion,
                         retainedSemanticCandidates());
    RetainedOwnershipSelection &generation =
        generations[alternative.generationIndex];
    StructuredOwnershipInvocationScope generationScope(*generation.invocation);
    for (const ArtifactRootReference &reference : alternative.references) {
      if (reference == *sourceReference) {
        if (!reserveMappingAndProgramMaterialization(
                alternative.planningRecordIndex))
          continue;
        llvm::Expected<SelectedStructuredOwnershipCandidate> candidate =
            [&]() {
              WorkTimer timer(frontierAccounting.programMaterializations);
              return generation.invocation->materializeSelectedCandidate(
                  reference, artifactStore);
            }();
        if (!candidate)
          return candidate.takeError();
        ++frontierAccounting.programMaterializations.consumed;
        auto ownedRoots = ownedRootsForCandidate(
            candidate->candidate.structuredProgram,
            alternative.ownedProtocolOrdinals);
        if (!ownedRoots)
          return ownedRoots.takeError();
        auto dataflowReference = dataflow::publishCanonicalDataflow(
            candidate->candidate.canonicalDataflow, artifactStore);
        if (!dataflowReference)
          return dataflowReference.takeError();
        auto planningRecord = planningRecordFor(
            reference, *dataflowReference, alternative.planningRecordIndex,
            alternative.scheduleIntent);
        if (!planningRecord)
          return planningRecord.takeError();
        candidateInventory[*planningRecord].ownedProtocolRoots =
            std::move(*ownedRoots);
        selected.push_back(assemble(std::move(*candidate), alternative,
                                    *planningRecord));
        selectedPlanningRecords.push_back(*planningRecord);
        continue;
      }

      if (frontierAccounting.mappingPairs.planned >=
          frontierAccounting.mappingPairs.limit) {
        if (alternative.planningRecordIndex)
          candidateInventory[*alternative.planningRecordIndex].disposition =
              PreMappingCandidatePlanningDisposition::MappingPairBudget;
        continue;
      }
      if (frontierAccounting.dataflowPromotions.planned ==
          frontierAccounting.dataflowPromotions.limit) {
        if (alternative.planningRecordIndex)
          candidateInventory[*alternative.planningRecordIndex].disposition =
              PreMappingCandidatePlanningDisposition::DataflowPromotionBudget;
        continue;
      }
      ++frontierAccounting.dataflowPromotions.planned;
      ++frontierAccounting.dataflowPromotions.reserved;
      auto d0 = generation.invocation->prepareDataflowGeneration(reference,
                                                                 artifactStore);
      if (!d0)
        return d0.takeError();
      llvm::Expected<DataflowSelectionOutcome> dataflowSelection = [&]() {
        WorkTimer timer(frontierAccounting.dataflowPromotions);
        return exploreDataflowCandidates(
            *d0, reference, fabric, *workloadReference, *runtimeInputReference,
            config, dataflowSelectionPolicy, plannerMode,
            options.frontier.stoppingPolicy ==
                JointDesignStoppingPolicy::BoundedQuality,
            options.executionControl, artifactStore, blobStore);
      }();
      if (!dataflowSelection)
        return dataflowSelection.takeError();
      ++frontierAccounting.dataflowPromotions.consumed;
      if (auto *incomplete = std::get_if<IncompletePreMappingExploration>(
              &*dataflowSelection)) {
        const DsePlanIncompleteReason reason = incomplete->reason;
        mergeReferences(incomplete->retainedEvidence, satisfiedEvidence);
        planGenerateInvocations.insert(
            planGenerateInvocations.end(),
            std::make_move_iterator(
                incomplete->planGenerateInvocations.begin()),
            std::make_move_iterator(incomplete->planGenerateInvocations.end()));
        if (alternative.planningRecordIndex) {
          auto &record = candidateInventory[*alternative.planningRecordIndex];
          record.disposition = planningDispositionForIncomplete(reason);
          record.incompleteReason = reason;
        }
        if (isCancellationReason(reason))
          return incompleteAt(
              PreMappingCheckpointBoundary::DataflowPromotion, reason,
              retainedSemanticCandidates(), std::move(satisfiedEvidence),
              std::move(planGenerateInvocations));
        dataflowProviderIncomplete = true;
        if (!dataflowIncompleteReason)
          dataflowIncompleteReason = reason;
        continue;
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
        if (!reserveMappingAndProgramMaterialization(
                alternative.planningRecordIndex))
          continue;
        auto planningRecord = planningRecordFor(
            reference, dataflowReference, alternative.planningRecordIndex,
            alternative.scheduleIntent);
        if (!planningRecord)
          return planningRecord.takeError();
        llvm::Expected<SelectedStructuredOwnershipCandidate> candidate =
            [&]() {
              WorkTimer timer(frontierAccounting.programMaterializations);
              return generation.invocation
                  ->materializeSelectedDataflowCandidate(
                      reference, dataflowReference, artifactStore);
            }();
        if (!candidate)
          return candidate.takeError();
        ++frontierAccounting.programMaterializations.consumed;
        auto ownedRoots = ownedRootsForCandidate(
            candidate->candidate.structuredProgram,
            alternative.ownedProtocolOrdinals);
        if (!ownedRoots)
          return ownedRoots.takeError();
        candidateInventory[*planningRecord].ownedProtocolRoots =
            std::move(*ownedRoots);
        selected.push_back(assemble(std::move(*candidate), alternative,
                                    *planningRecord));
        selectedPlanningRecords.push_back(*planningRecord);
      }
    }
  }
  std::vector<SelectedPreMappingCompilation> bounded;
  bounded.reserve(static_cast<std::size_t>(
      std::min<std::uint64_t>(options.ownership.selection.k, selected.size())));
  std::vector<ArtifactIdentity> seenDataflows;
  seenDataflows.reserve(selected.size());
  // Preserve the retained Structured frontier before taking more rewrites
  // from any one parent. Parent order and each parent's Dataflow preference
  // order remain intact within these rounds.
  std::map<ArtifactIdentity::Storage, std::size_t> nextRewriteRank;
  std::vector<std::pair<std::size_t, std::size_t>> rewriteRankedCandidates;
  for (auto indexed : llvm::enumerate(selected)) {
    const ArtifactIdentity identity =
        indexed.value().compilation.canonicalDataflow.identity();
    if (llvm::is_contained(seenDataflows, identity))
      continue;
    seenDataflows.push_back(identity);
    rewriteRankedCandidates.emplace_back(
        nextRewriteRank[indexed.value()
                            .compilation.structuredProgram.identity().bytes()]++,
        indexed.index());
  }
  llvm::sort(rewriteRankedCandidates);
  for (const auto &[rewriteRank, selectedOrdinal] : rewriteRankedCandidates) {
    (void)rewriteRank;
    SelectedPreMappingCompilation &candidate = selected[selectedOrdinal];
    if (frontierAccounting.mappingPairs.consumed >=
        frontierAccounting.mappingPairs.limit)
      return invalid("compiler frontier retained more Mapping pairs than its "
                     "admitted bound");
    ++frontierAccounting.mappingPairs.consumed;
    candidate.preferenceRank = bounded.size();
    if (selectedPlanningRecords[selectedOrdinal]) {
      auto &record =
          candidateInventory[*selectedPlanningRecords[selectedOrdinal]];
      record.disposition = PreMappingCandidatePlanningDisposition::Retained;
      record.preferenceRank = candidate.preferenceRank;
    }
    bounded.push_back(std::move(candidate));
    if (bounded.size() == options.ownership.selection.k)
      break;
  }
  selected = std::move(bounded);
  const auto hasBudgetDisposition = llvm::any_of(
      candidateInventory, [](const PreMappingCandidatePlanningRecord &record) {
        switch (record.disposition) {
        case PreMappingCandidatePlanningDisposition::CoordinateBudget:
        case PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget:
        case PreMappingCandidatePlanningDisposition::AnalyticEvaluationBudget:
        case PreMappingCandidatePlanningDisposition::FunctionalReplayBudget:
        case PreMappingCandidatePlanningDisposition::DataflowPromotionBudget:
        case PreMappingCandidatePlanningDisposition::MappingPairBudget:
          return true;
        case PreMappingCandidatePlanningDisposition::Retained:
        case PreMappingCandidatePlanningDisposition::HeuristicPruned:
        case PreMappingCandidatePlanningDisposition::ExactGateRejected:
        case PreMappingCandidatePlanningDisposition::Unsupported:
        case PreMappingCandidatePlanningDisposition::Unknown:
        case PreMappingCandidatePlanningDisposition::CancelledOrTimeout:
          return false;
        }
        return false;
      });
  if (selected.empty() && !retainedPlanIncompleteness.empty()) {
    const RetainedDsePlanIncompleteness &first =
        retainedPlanIncompleteness.front();
    return incompleteAt(PreMappingCheckpointBoundary::EvidencePromotion,
                        first.reason, retainedSemanticCandidates(),
                        std::move(satisfiedEvidence),
                        std::move(planGenerateInvocations));
  }
  if (selected.empty() && dataflowIncompleteReason)
    return incompleteAt(PreMappingCheckpointBoundary::DataflowPromotion,
                        *dataflowIncompleteReason,
                        retainedSemanticCandidates(),
                        std::move(satisfiedEvidence),
                        std::move(planGenerateInvocations));
  if (selected.empty() && hasBudgetDisposition)
    return incompleteAt(
        PreMappingCheckpointBoundary::MappingAdmission,
        DsePlanIncompleteReason{
            CandidateGeneratorIncompleteReason::SemanticLimitReached},
        retainedSemanticCandidates(), std::move(satisfiedEvidence),
        std::move(planGenerateInvocations));
  if (selected.empty()) {
    auto noFeasibleCompleteness = deriveNoFeasibleCompleteness();
    if (!noFeasibleCompleteness)
      return noFeasibleCompleteness.takeError();
    return PreMappingExplorationOutcome{CompletedPreMappingNoFeasibleCandidate{
        std::move(satisfiedEvidence), std::move(planGenerateInvocations),
        *sourceReference, fabric.reference(), *workloadReference,
        *runtimeInputReference, std::move(candidateInventory),
        std::move(*noFeasibleCompleteness), *frontierPolicyDigest,
        sourceHostOnlyRuntimePicoseconds, std::move(finalizationRejections)}};
  }
  const StructuredOwnershipSharedEvaluationStatistics sharedStatistics =
      sharedEvaluation.statistics();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "structured_profile_cache";
        fields["cache_hits"] = sharedStatistics.profileCacheHits;
        fields["cache_misses"] = sharedStatistics.profileCacheMisses;
        fields["single_flight_waits"] =
            sharedStatistics.profileSingleFlightWaits;
        fields["analytic_requests"] = evaluationTiming.analyticCalls;
        fields["analytic_elapsed_nanoseconds"] =
            evaluationTiming.analyticElapsedNanoseconds;
        fields["functional_replay_requests"] =
            evaluationTiming.functionalReplayCalls;
        fields["functional_replay_elapsed_nanoseconds"] =
            evaluationTiming.functionalReplayElapsedNanoseconds;
      });
  std::optional<PreMappingShadowRecall> shadowRecall;
  if (sourceProtocolRoots->size() <= 4) {
    auto recall = evaluatePreMappingShadowRecall(
        sourceProtocolRoots->size(), *coordinatePlan);
    if (!recall)
      return recall.takeError();
    shadowRecall = std::move(*recall);
  }
  PreMappingSearchCompleteness completeness;
  completeness.budgetComplete = !hasBudgetDisposition;
  completeness.providerComplete = retainedPlanIncompleteness.empty() &&
                                  !ownershipProviderIncomplete &&
                                  !dataflowProviderIncomplete;
  completeness.evidenceComplete = llvm::all_of(
      selected, [](const SelectedPreMappingCompilation &candidate) {
        return candidate.functionalReplay &&
               candidate.functionalReplay->status ==
                   sim::SourceBackedDfgValidationStatus::Equivalent;
      });
  completeness.selectionComplete = !selected.empty();
  const bool planningFrontierComplete = llvm::all_of(
      candidateInventory, [](const PreMappingCandidatePlanningRecord &record) {
        switch (record.disposition) {
        case PreMappingCandidatePlanningDisposition::Retained:
        case PreMappingCandidatePlanningDisposition::ExactGateRejected:
          return true;
        case PreMappingCandidatePlanningDisposition::HeuristicPruned:
        case PreMappingCandidatePlanningDisposition::CoordinateBudget:
        case PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget:
        case PreMappingCandidatePlanningDisposition::AnalyticEvaluationBudget:
        case PreMappingCandidatePlanningDisposition::FunctionalReplayBudget:
        case PreMappingCandidatePlanningDisposition::DataflowPromotionBudget:
        case PreMappingCandidatePlanningDisposition::MappingPairBudget:
        case PreMappingCandidatePlanningDisposition::Unsupported:
        case PreMappingCandidatePlanningDisposition::Unknown:
        case PreMappingCandidatePlanningDisposition::CancelledOrTimeout:
          return false;
        }
        return false;
      });
  completeness.domainComplete =
      !coordinatePlan->truncated && sourceProtocolRoots->size() <= 4 &&
      shadowRecall && shadowRecall->missingSubsets.empty() &&
      planningFrontierComplete &&
      llvm::none_of(
          protocolDependencyProjection->relations,
          [](const auto &relation) {
            return relation.knowledge ==
                   frontend::analysis::
                       StructuredProtocolDependencyKnowledge::Unknown;
          });
  for (auto &counter : {&frontierAccounting.sourceObservations,
                        &frontierAccounting.coordinates,
                        &frontierAccounting.programMaterializations,
                        &frontierAccounting.analyticEvaluations,
                        &frontierAccounting.functionalReplays,
                        &frontierAccounting.dataflowPromotions,
                        &frontierAccounting.mappingPairs})
    {
      if (counter->reserved != counter->planned)
        return invalid("compiler frontier reservation ledger diverged from "
                       "planned work");
      if (counter->consumed > counter->reserved)
        return invalid("compiler frontier consumed work exceeds reservation");
      if (counter->consumed >
          std::numeric_limits<std::uint64_t>::max() - counter->rejected)
        return invalid("compiler frontier settled-work ledger overflows");
      const std::uint64_t consumedAndRejected =
          counter->consumed + counter->rejected;
      if (consumedAndRejected >
          std::numeric_limits<std::uint64_t>::max() - counter->cancelled)
        return invalid("compiler frontier settled-work ledger overflows");
      const std::uint64_t settled =
          consumedAndRejected + counter->cancelled;
      if (settled > counter->reserved)
        return invalid("compiler frontier settled work exceeds reservation");
      counter->rejected += counter->reserved - settled;
    }
  if (llvm::Error error =
          validatePreMappingWorkAccounting(frontierAccounting))
    return std::move(error);
  for (PreMappingCandidatePlanningRecord &record : candidateInventory) {
    if (!record.structuredProgram || !record.canonicalDataflow)
      continue;
    auto identity = computePreMappingCandidateIdentity(
        record, *sourceReference, fabric.reference(), *workloadReference,
        *runtimeInputReference, *frontierPolicyDigest);
    if (!identity)
      return identity.takeError();
    if (record.candidateIdentity && *record.candidateIdentity != *identity)
      return invalid("pre-Mapping candidate identity changed after planning");
    record.candidateIdentity = *identity;
  }
  std::uint64_t temporalHintRecordCount = 0;
  std::uint64_t spatialHintRecordCount = 0;
  std::uint64_t intermediateHintRecordCount = 0;
  std::uint64_t verifiedTemporalRecordCount = 0;
  std::uint64_t verifiedSpatialRecordCount = 0;
  std::uint64_t verifiedIntermediateRecordCount = 0;
  for (const PreMappingCandidatePlanningRecord &record : candidateInventory) {
    switch (record.scheduleIntent.value_or(
        PreMappingScheduleIntent::Unconstrained)) {
    case PreMappingScheduleIntent::TemporalReuse:
      ++temporalHintRecordCount;
      break;
    case PreMappingScheduleIntent::SpatialParallel:
      ++spatialHintRecordCount;
      break;
    case PreMappingScheduleIntent::Unconstrained:
      ++intermediateHintRecordCount;
      break;
    }
    verifiedTemporalRecordCount +=
        record.verifiedSpectrum == PreMappingSpectrumClass::MaxTemporal;
    verifiedSpatialRecordCount +=
        record.verifiedSpectrum == PreMappingSpectrumClass::MaxSpatial;
    verifiedIntermediateRecordCount +=
        record.verifiedSpectrum == PreMappingSpectrumClass::Intermediate;
  }
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "pre_mapping_spectrum_summary";
        fields["temporal_schedule_hint_count"] = temporalHintRecordCount;
        fields["spatial_schedule_hint_count"] = spatialHintRecordCount;
        fields["intermediate_schedule_hint_count"] =
            intermediateHintRecordCount;
        fields["verified_max_temporal_count"] = verifiedTemporalRecordCount;
        fields["verified_max_spatial_count"] = verifiedSpatialRecordCount;
        fields["verified_intermediate_count"] =
            verifiedIntermediateRecordCount;
      });
  return PreMappingExplorationOutcome{
      CompletedPreMappingSelection{std::move(selected),
                                   std::move(satisfiedEvidence),
                                   std::move(dispositions),
                                   std::move(protocolRootActivity),
                                   std::move(protocolDependencies),
                                   std::move(*protocolDependencyProjection),
                                   std::move(candidateInventory),
                                   options.frontier,
                                   coordinatePlan->eligibleCoordinateCount,
                                   coordinatePlan->truncated,
                                   std::move(frontierAccounting),
                                   sharedStatistics,
                                   evaluationTiming,
                                   evaluationCache.statistics(),
                                   std::move(planGenerateInvocations),
                                   std::move(retainedPlanIncompleteness),
                                   *sourceReference,
                                   fabric.reference(),
                                   *workloadReference,
                                   *runtimeInputReference,
                                   *frontierPolicyDigest,
                                   requestedPlannerMode,
                                   plannerMode,
                                   completeness,
                                   std::move(shadowRecall),
                                   sourceHostOnlyRuntimePicoseconds,
                                   sourceHostOnlyLeafExecutions}};
}

} // namespace loom::dse
