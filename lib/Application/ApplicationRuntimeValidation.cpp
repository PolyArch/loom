#include "ApplicationRuntimeValidationInternal.h"

#include "ExecutionGlue.h"

#include "Common/ArtifactText.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Common/MappingDebugLog.h"
#include "DSE/CandidateGenerator.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application::detail {
namespace {

using MonotonicClock = std::chrono::steady_clock;

constexpr std::uint64_t kApplicationReplayExecutionLimit = 1000000;

struct ApplicationSpatialRuntimePoint final {
  dataflow::RootThreadLaunchRef root;
  dataflow::RootedGraphLaunchRef graph;
  std::vector<std::uint64_t> denseCoordinates;
  mapping::SpatialExecutionContextKey context;

  friend bool operator==(const ApplicationSpatialRuntimePoint &lhs,
                         const ApplicationSpatialRuntimePoint &rhs) {
    return lhs.root == rhs.root && lhs.graph == rhs.graph &&
           lhs.denseCoordinates == rhs.denseCoordinates &&
           lhs.context == rhs.context;
  }
};

struct ResolvedApplicationReplay final {
  const sim::SourceBackedDfgReplayCaseReference *reference = nullptr;
  ApplicationSpatialRuntimePoint point;
  ArtifactRootReference module;
  ArtifactRootReference spatialMapping;
  std::optional<ArtifactRootReference> spatialConstraints;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_build_invalid: " + message);
}

llvm::Expected<ArtifactRootReference>
requireUniqueSystemMapping(const dse::JointDesignExecution &execution) {
  std::vector<ArtifactRootReference> mappings;
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    mappings.insert(mappings.end(), pair.systemMappings.begin(),
                    pair.systemMappings.end());
  llvm::sort(mappings, artifactRootReferenceLess);
  mappings.erase(std::unique(mappings.begin(), mappings.end()), mappings.end());
  if (!execution.summary.selectedMapping)
    return invalid("Deployment requires one explicitly selected SystemMapping");
  if (!llvm::is_contained(mappings, *execution.summary.selectedMapping))
    return invalid("selected SystemMapping is outside the verified result");
  return *execution.summary.selectedMapping;
}

llvm::Expected<const PreparedApplicationSoftware *>
findPreparedSoftwareImpl(const PreparedApplicationBuild &prepared,
                         const ArtifactIdentity &dataflowIdentity) {
  const PreparedApplicationSoftware *selected = nullptr;
  for (const PreparedApplicationSoftware &software : prepared.software) {
    if (software.compilation.canonicalDataflow.artifact != dataflowIdentity)
      continue;
    if (selected)
      return invalid("prepared build repeats one Canonical Dataflow owner");
    selected = &software;
  }
  if (!selected)
    return invalid("SystemMapping names a foreign prepared software owner");
  return selected;
}

ApplicationMappingRuntimeDisposition
runtimeDisposition(evaluation::EvidenceOutcomeKind outcome) {
  using Evidence = evaluation::EvidenceOutcomeKind;
  switch (outcome) {
  case Evidence::Completed:
    return ApplicationMappingRuntimeDisposition::Completed;
  case Evidence::Unsupported:
    return ApplicationMappingRuntimeDisposition::Unsupported;
  case Evidence::ExecutionFailed:
    return ApplicationMappingRuntimeDisposition::ExecutionFailed;
  case Evidence::CancelledOrTimeout:
    return ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
  }
  llvm_unreachable("unknown Evaluation Evidence outcome");
}

void emitRuntimeEvidenceFailure(
    llvm::StringRef model, const evaluation::EvaluationEvidence &evidence) {
  std::optional<evaluation::OutcomeReason> reason;
  std::visit(
      [&](const auto &outcome) {
        using Outcome = std::decay_t<decltype(outcome)>;
        if constexpr (!std::is_same_v<Outcome, evaluation::CompletedEvidence>)
          reason = outcome.reason;
      },
      evidence.outcome());
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "application_runtime_validation";
        fields["model"] = model;
        fields["outcome"] = evaluation::toString(evidence.outcomeKind());
        fields["reason"] = reason ? evaluation::toString(*reason) : "none";
      });
}

llvm::Expected<std::optional<MonotonicClock::time_point>>
applicationReplayDeadline(const dse::PlanExecutionPolicy &policy) {
  if (!policy.dispatchNotAfterUnixNanoseconds())
    return std::nullopt;
  if (*policy.dispatchNotAfterUnixNanoseconds() >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("Mapping deadline exceeds the clock representation");
  const MonotonicClock::time_point monotonicNow = MonotonicClock::now();
  const auto deadline = std::chrono::system_clock::time_point{
      std::chrono::nanoseconds{static_cast<std::int64_t>(
          *policy.dispatchNotAfterUnixNanoseconds())}};
  const auto remaining = deadline - std::chrono::system_clock::now();
  if (remaining <= std::chrono::system_clock::duration::zero())
    return monotonicNow;
  const MonotonicClock::duration monotonicRemaining =
      std::chrono::duration_cast<MonotonicClock::duration>(remaining);
  return monotonicRemaining > MonotonicClock::time_point::max() - monotonicNow
             ? MonotonicClock::time_point::max()
             : monotonicNow + monotonicRemaining;
}

llvm::Expected<ArtifactRootReference>
requireExecutionOutput(const evaluation::EvaluationEvidence &evidence) {
  std::vector<ArtifactRootReference> executions;
  for (const evaluation::ModelOutputBinding &binding :
       evidence.outputBindings())
    for (const ArtifactRootReference &reference : binding.artifacts)
      if (reference.schemaIdentity == sim::simulationExecutionSchema.identity &&
          reference.schemaVersion == sim::simulationExecutionSchema.version)
        executions.push_back(reference);
  llvm::sort(executions, artifactRootReferenceLess);
  executions.erase(std::unique(executions.begin(), executions.end()),
                   executions.end());
  if (executions.size() != 1)
    return invalid("completed simulation did not publish one execution");
  return executions.front();
}

llvm::Expected<std::uint64_t>
requireCompletedCycleMetric(const evaluation::EvaluationEvidence &evidence) {
  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed || completed->metricResults.size() != 1)
    return invalid("completed simulation did not publish one cycle metric");
  const auto *point = std::get_if<evaluation::PointObservation>(
      &completed->metricResults.front().observation);
  if (!point)
    return invalid("completed simulation cycle metric is not a point");
  const auto *integer = std::get_if<evaluation::IntegerValue>(&point->value);
  if (!integer || integer->value() < 0)
    return invalid("completed simulation cycle metric is not nonnegative");
  return static_cast<std::uint64_t>(integer->value());
}

llvm::Error accumulateCycle(std::optional<std::uint64_t> &total,
                            std::uint64_t value, llvm::StringRef subject) {
  const std::uint64_t current = total.value_or(0);
  if (value > std::numeric_limits<std::uint64_t>::max() - current)
    return invalid(subject + " cycle count overflows uint64");
  total = current + value;
  return llvm::Error::success();
}

llvm::Expected<ApplicationRuntimeValidation> validateApplicationReplay(
    const ResolvedApplicationReplay &resolved,
    const PreparedApplicationMappingAlternative &alternative,
    const ArtifactRootReference &systemMapping,
    std::optional<MonotonicClock::time_point> deadline,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationRuntimeValidation validation;
  validation.disposition = ApplicationMappingRuntimeDisposition::Completed;
  const sim::SourceBackedDfgReplayCaseReference &replay = *resolved.reference;
  if (deadline && MonotonicClock::now() >= *deadline) {
    validation.disposition =
        ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
    return validation;
  }
  auto preparedDfg = evaluation::models::prepareDfgSimulationEvaluation(
      alternative.dataflow, replay.workload, replay.runtimeInput,
      alternative.plan.resolvedConfig, artifacts, blobs);
  if (!preparedDfg)
    return preparedDfg.takeError();
  auto dfgEvidence = evaluation::models::evaluateDfgSimulation(
      *preparedDfg, {kApplicationReplayExecutionLimit, deadline}, artifacts,
      blobs);
  if (!dfgEvidence)
    return dfgEvidence.takeError();
  auto dfgEvidenceReference =
      evaluation::publishEvaluationEvidence(*dfgEvidence, artifacts);
  if (!dfgEvidenceReference)
    return dfgEvidenceReference.takeError();
  validation.evidence.push_back(*dfgEvidenceReference);
  if (dfgEvidence->outcomeKind() !=
      evaluation::EvidenceOutcomeKind::Completed) {
    emitRuntimeEvidenceFailure("dfg_simulation", *dfgEvidence);
    validation.disposition = runtimeDisposition(dfgEvidence->outcomeKind());
    return validation;
  }
  auto dfgExecution = requireExecutionOutput(*dfgEvidence);
  if (!dfgExecution)
    return dfgExecution.takeError();
  auto dfgCycles = requireCompletedCycleMetric(*dfgEvidence);
  if (!dfgCycles)
    return dfgCycles.takeError();
  validation.dfgCycles = *dfgCycles;

  auto preparedCgra = evaluation::models::prepareCgraSimulationEvaluation(
      alternative.dataflow, resolved.module, resolved.spatialMapping,
      replay.workload, replay.runtimeInput, alternative.plan.resolvedConfig,
      artifacts, blobs);
  if (!preparedCgra)
    return preparedCgra.takeError();
  auto cgraEvaluation =
      evaluation::models::evaluateCgraSimulationWithDiagnostics(
          *preparedCgra,
          {evaluation::models::cgraReplayEventFrameGrant(
               *dfgCycles, preparedCgra->execution),
           deadline},
          artifacts, blobs);
  if (!cgraEvaluation)
    return cgraEvaluation.takeError();
  evaluation::EvaluationEvidence &cgraEvidence = cgraEvaluation->evidence;
  auto cgraEvidenceReference =
      evaluation::publishEvaluationEvidence(cgraEvidence, artifacts);
  if (!cgraEvidenceReference)
    return cgraEvidenceReference.takeError();
  validation.evidence.push_back(*cgraEvidenceReference);
  if (cgraEvaluation->closedWait) {
    std::optional<evaluation::models::VerifiedCgraClosedWaitEvidence>
        verifiedClosedWait;
    auto importedClosedWait =
        evaluation::models::importVerifiedCgraClosedWaitEvidence(
            *cgraEvidenceReference, artifacts, blobs);
    if (importedClosedWait)
      verifiedClosedWait.emplace(std::move(*importedClosedWait));
    else
      llvm::consumeError(importedClosedWait.takeError());
    auto operandFeedback = dse::deriveSpatialOperandQueueRuntimeFeedback(
        systemMapping, *cgraEvaluation->closedWait, artifacts);
    if (!operandFeedback)
      return operandFeedback.takeError();
    dse::emitSpatialOperandQueueRuntimeFeedback(*operandFeedback);
    validation.spatialOperandQueueFeedback = std::move(*operandFeedback);
    auto feedback = dse::deriveSpatialFifoRuntimeFeedback(
        systemMapping, resolved.spatialMapping, *cgraEvaluation->closedWait,
        artifacts);
    if (!feedback)
      return feedback.takeError();
    dse::emitSpatialFifoRuntimeFeedback(*feedback);
    validation.spatialFifoFeedback = std::move(*feedback);
    dse::SpatialTransportRuntimeFeedback transportFeedback;
    if (resolved.spatialConstraints && verifiedClosedWait) {
      auto derived = dse::deriveSpatialTransportRuntimeFeedback(
          resolved.spatialMapping, *resolved.spatialConstraints,
          *verifiedClosedWait, artifacts, systemMapping);
      if (!derived)
        return derived.takeError();
      transportFeedback = std::move(*derived);
    } else {
      transportFeedback.parentMapping = systemMapping;
      transportFeedback.parentSpatialMapping = resolved.spatialMapping;
      transportFeedback.runtimeEvidence = *cgraEvidenceReference;
      transportFeedback.evaluationRequest =
          evaluation::evaluationRequestReference(preparedCgra->request);
      transportFeedback.owners = cgraEvaluation->closedWait->ownerReferences;
      transportFeedback.certificateEdgeCount =
          cgraEvaluation->closedWait->waitCertificate.size();
      if (verifiedClosedWait) {
        transportFeedback.runtimeExecution = verifiedClosedWait->execution();
        transportFeedback.certificateDigest =
            verifiedClosedWait->certificateDigest();
        transportFeedback.reason = dse::SpatialTransportRuntimeFeedbackReason::
            UnboundConstraintLineage;
      } else {
        transportFeedback.reason =
            dse::SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence;
      }
    }
    dse::emitSpatialTransportRuntimeFeedback(transportFeedback);
    validation.spatialTransportFeedback = std::move(transportFeedback);
    if (verifiedClosedWait) {
      validation.disposition =
          ApplicationMappingRuntimeDisposition::ExecutionFailed;
      return validation;
    }
  }
  if (cgraEvidence.outcomeKind() !=
      evaluation::EvidenceOutcomeKind::Completed) {
    emitRuntimeEvidenceFailure("cgra_simulation", cgraEvidence);
    validation.disposition = runtimeDisposition(cgraEvidence.outcomeKind());
    return validation;
  }
  auto cgraTerminal =
      evaluation::models::classifyCompletedCgraSimulationEvidence(
          cgraEvidence, preparedCgra->resolution, artifacts, blobs);
  if (!cgraTerminal)
    return cgraTerminal.takeError();
  if (*cgraTerminal ==
      evaluation::models::CgraSimulationEvidenceTerminal::ClosedWait) {
    validation.disposition =
        ApplicationMappingRuntimeDisposition::ExecutionFailed;
    return validation;
  }
  auto cgraExecution = requireExecutionOutput(cgraEvidence);
  if (!cgraExecution)
    return cgraExecution.takeError();
  auto cgraCycles = requireCompletedCycleMetric(cgraEvidence);
  if (!cgraCycles)
    return cgraCycles.takeError();
  validation.cgraCycles = *cgraCycles;
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields;
        fields["measurement_kind"] = "direct_and_derived";
        fields["direct"] = llvm::json::Object{{"dfg_cycles", *dfgCycles},
                                              {"cgra_cycles", *cgraCycles}};
        fields["derived"] = llvm::json::Object{
            {"cycle_delta",
             *cgraCycles >= *dfgCycles ? *cgraCycles - *dfgCycles : 0},
            {"cgra_to_dfg_ratio",
             llvm::json::Object{{"numerator", *cgraCycles},
                                {"denominator", *dfgCycles}}},
            {"cgra_is_slower", *cgraCycles > *dfgCycles}};
        fields["operation"] = "simulation_cycle_comparison";
        fields["dataflow"] =
            formatArtifactRootReferenceJson(alternative.dataflow);
        fields["spatial_mapping"] =
            formatArtifactRootReferenceJson(resolved.spatialMapping);
        fields["dfg_request"] = formatArtifactRootReferenceJson(
            evaluation::evaluationRequestReference(preparedDfg->request));
        fields["cgra_request"] = formatArtifactRootReferenceJson(
            evaluation::evaluationRequestReference(preparedCgra->request));
        fields["dfg_cycles"] = *dfgCycles;
        fields["cgra_cycles"] = *cgraCycles;
        fields["cycle_delta"] =
            *cgraCycles >= *dfgCycles ? *cgraCycles - *dfgCycles : 0;
        fields["cgra_to_dfg_ratio"] = llvm::json::Object{
            {"numerator", *cgraCycles}, {"denominator", *dfgCycles}};
        fields["cgra_is_slower"] = *cgraCycles > *dfgCycles;
        return llvm::json::Value(std::move(fields));
      });

  auto comparison = evaluation::models::prepareSimulationComparisonEvaluation(
      *dfgExecution, preparedDfg->resolution, *cgraExecution,
      preparedCgra->resolution, alternative.plan.resolvedConfig, artifacts,
      blobs);
  if (!comparison)
    return comparison.takeError();
  auto comparisonEvidence = evaluation::models::evaluateSimulationComparison(
      *comparison, artifacts, blobs);
  if (!comparisonEvidence)
    return comparisonEvidence.takeError();
  auto comparisonEvidenceReference =
      evaluation::publishEvaluationEvidence(*comparisonEvidence, artifacts);
  if (!comparisonEvidenceReference)
    return comparisonEvidenceReference.takeError();
  validation.evidence.push_back(*comparisonEvidenceReference);
  if (comparisonEvidence->outcomeKind() !=
      evaluation::EvidenceOutcomeKind::Completed) {
    emitRuntimeEvidenceFailure("simulation_comparison", *comparisonEvidence);
    validation.disposition =
        runtimeDisposition(comparisonEvidence->outcomeKind());
    return validation;
  }
  const auto *completed = std::get_if<evaluation::CompletedEvidence>(
      &comparisonEvidence->outcome());
  if (!completed || completed->findingResults.size() != 1)
    return invalid("simulation comparison has no unique result");
  const evaluation::FindingResultValue &comparisonResult =
      completed->findingResults.front().result;
  if (std::holds_alternative<evaluation::AbsentFinding>(comparisonResult)) {
    validation.oracleEvidence.push_back(*comparisonEvidenceReference);
    return validation;
  }
  validation.disposition =
      std::holds_alternative<evaluation::NotApplicableFinding>(comparisonResult)
          ? ApplicationMappingRuntimeDisposition::ProofNotEstablished
          : ApplicationMappingRuntimeDisposition::ExecutionFailed;
  return validation;
}

} // namespace

llvm::Expected<const PreparedApplicationSoftware *>
findPreparedSoftware(const PreparedApplicationBuild &prepared,
                     const ArtifactIdentity &dataflowIdentity) {
  return findPreparedSoftwareImpl(prepared, dataflowIdentity);
}

llvm::Expected<ImportedApplicationMapping>
importApplicationMapping(const dse::JointDesignExecution &execution,
                         const ArtifactStore &artifacts) {
  auto reference = requireUniqueSystemMapping(execution);
  if (!reference)
    return reference.takeError();
  auto mapping = mapping::importSystemMapping(*reference, artifacts);
  if (!mapping)
    return mapping.takeError();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping->view().dataflowIdentity()};
  const std::array<ArtifactRootReference, 1> dataflowReferences{
      dataflowReference};
  auto dataflow =
      evaluation::importCachedArtifact<dataflow::CanonicalDataflowArtifact>(
          artifacts, nullptr, dataflowReferences, [&] {
            return dataflow::importCanonicalDataflow(dataflowReference,
                                                     artifacts);
          });
  if (!dataflow)
    return dataflow.takeError();
  const ArtifactRootReference systemReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, mapping->view().fabricIdentity()};
  auto system = fabric::importEntireFabricRoot(systemReference, artifacts);
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();
  return ImportedApplicationMapping{std::move(*mapping), std::move(*dataflow),
                                    std::move(*system)};
}

llvm::Expected<ApplicationRuntimeValidation> validateApplicationMappingRuntime(
    const PreparedApplicationBuild &prepared,
    const PreparedApplicationMappingAlternative &alternative,
    const dse::JointDesignExecution &execution,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  evaluation::ArtifactImportCacheScope importCache(artifacts, &blobs);
  fabric::FabricArtifactImportSession fabricImports;
  auto imported = importApplicationMapping(execution, artifacts);
  if (!imported)
    return imported.takeError();
  const std::uint64_t resourceCoreCost = static_cast<std::uint64_t>(
      imported->system.view().accCoreOccurrences().size());
  if (imported->mapping.view().dataflowIdentity() !=
      alternative.dataflow.artifact)
    return invalid("runtime validation selected a foreign software owner");
  auto software = findPreparedSoftware(
      prepared, imported->mapping.view().dataflowIdentity());
  if (!software)
    return software.takeError();
  if ((*software)->replayCases.empty())
    return ApplicationRuntimeValidation{
        ApplicationMappingRuntimeDisposition::ProofNotEstablished,
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        resourceCoreCost,
        std::nullopt};

  auto contexts = mapping::projectSystemExecutionContexts(
      imported->dataflow->view(), imported->mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();
  auto invocationPlan = deriveApplicationSpatialInvocationPlan(
      imported->dataflow->view(), prepared.sourceInvocation.entrySymbol,
      (*software)->compilation.structuredProgram, prepared.preMappingWorkload,
      prepared.preMappingRuntimeInput, artifacts, (*software)->invocationCaptureByteLimit);
  if (!invocationPlan)
    return invocationPlan.takeError();

  std::vector<ApplicationSpatialRuntimePoint> requiredPoints;
  for (const ApplicationSpatialInvocationPlan::Launch &launch :
       invocationPlan->launches)
    for (const ApplicationSpatialInvocationPlan::Launch::Point &point :
         launch.points) {
      auto selected = mapping::selectSystemSpatialExecutionContext(
          *contexts, launch.graph, point.denseCoordinates);
      if (!selected)
        return selected.takeError();
      ApplicationSpatialRuntimePoint required{
          launch.root, launch.graph, point.denseCoordinates, selected->context};
      if (!llvm::is_contained(requiredPoints, required))
        requiredPoints.push_back(std::move(required));
    }
  if (requiredPoints.empty())
    return invalid(
        "selected SystemMapping has no ABI-reachable Spatial invocation");

  struct SpatialMappingLineage final {
    ArtifactRootReference module;
    std::optional<ArtifactRootReference> constraints;
  };
  // One immutable execution determines the lineage of each selected Mapping.
  // Dynamic replay activations only select from that finite SystemMapping
  // domain; they do not require another strict import of the same lineage.
  std::map<ArtifactRootReference, SpatialMappingLineage,
           decltype(&artifactRootReferenceLess)>
      spatialMappingLineages(&artifactRootReferenceLess);
  const auto replayResolutionStarted = MonotonicClock::now();
  std::vector<ResolvedApplicationReplay> resolvedReplays;
  resolvedReplays.reserve((*software)->replayCases.size());
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       (*software)->replayCases) {
    auto inputs =
        sim::importSpatialSimulationWorkload(replay.workload, artifacts);
    if (!inputs)
      return inputs.takeError();
    if (inputs->dataflow->identity() != alternative.dataflow.artifact)
      return invalid("source-backed replay names a foreign final Dataflow");
    const sim::SpatialSimulationWorkload *workload = inputs->workload.spatial();
    if (!workload)
      return invalid("source-backed replay is not a Spatial workload");
    auto selected = mapping::selectSystemSpatialExecutionContext(
        *contexts, workload->launchRef, workload->denseCoordinates);
    if (!selected)
      return selected.takeError();
    auto lineage = spatialMappingLineages.find(selected->spatialMapping);
    if (lineage == spatialMappingLineages.end()) {
      auto spatialMapping =
          mapping::importSpatialMapping(selected->spatialMapping, artifacts);
      if (!spatialMapping)
        return spatialMapping.takeError();
      const ArtifactRootReference module{
          fabric::fabricArtifactSchema.identity.str(),
          fabric::fabricArtifactSchema.version,
          spatialMapping->view().fabricIdentity()};
      auto spatialConstraints = dse::projectJointSpatialMappingConstraintSet(
          execution, selected->spatialMapping, artifacts);
      if (!spatialConstraints)
        return spatialConstraints.takeError();
      lineage = spatialMappingLineages
                    .emplace(selected->spatialMapping,
                             SpatialMappingLineage{
                                 module, std::move(*spatialConstraints)})
                    .first;
    }
    resolvedReplays.push_back(
        {&replay,
         {workload->launchRef.rootThreadLaunch, workload->launchRef,
          workload->denseCoordinates, selected->context},
         lineage->second.module,
         selected->spatialMapping,
         lineage->second.constraints});
  }
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "application_runtime_spatial_lineage";
        fields["replay_case_count"] = resolvedReplays.size();
        fields["spatial_mapping_count"] = spatialMappingLineages.size();
        fields["construction_time_ns"] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                MonotonicClock::now() - replayResolutionStarted)
                .count();
      });

  evaluation::emitArtifactImportCacheStatistics(
      evaluation::ArtifactImportCacheVerificationDomain::SourceInvocation,
      importCache.statistics());

  for (const ResolvedApplicationReplay &replay : resolvedReplays)
    if (!llvm::is_contained(requiredPoints, replay.point))
      return invalid("source-backed replay is outside the ABI invocation plan");

  for (const ApplicationSpatialRuntimePoint &required : requiredPoints) {
    const bool covered = llvm::any_of(
        resolvedReplays, [&](const ResolvedApplicationReplay &replay) {
          return replay.point == required;
        });
    if (covered)
      continue;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
          fields["failure_scope"] = "application_runtime_validation";
          fields["operation"] = "source_backed_context_coverage";
          fields["outcome"] = "proof_not_established";
          fields["required_point_count"] = requiredPoints.size();
          fields["covered_replay_count"] = resolvedReplays.size();
          fields["missing_root_entity"] = required.root.entity.value();
          fields["missing_graph_entity"] =
              required.graph.staticGraphLaunch.entity.value();
          llvm::json::Array coordinates;
          for (std::uint64_t coordinate : required.denseCoordinates)
            coordinates.push_back(coordinate);
          fields["missing_dense_coordinates"] = std::move(coordinates);
          fields["missing_spatial_mapping"] =
              formatArtifactIdentityHex(required.context.spatialMapping);
        });
    return ApplicationRuntimeValidation{
        ApplicationMappingRuntimeDisposition::ProofNotEstablished,
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        resourceCoreCost,
        std::nullopt};
  }
  auto deadline = applicationReplayDeadline(executionPolicy);
  if (!deadline)
    return deadline.takeError();

  const ExecutionResourceBudget executionBudget{
      executionPolicy.inProcessClaim().cpuCores(),
      executionPolicy.inProcessClaim().memoryBytes()};
  const std::size_t workerCount =
      imported->dataflow->module().getContext()->isMultithreadingEnabled()
          ? std::min<std::size_t>(
                dse::defaultCandidateWorkerCount(executionBudget),
                resolvedReplays.size())
          : 1;
  std::size_t actualWorkerCount = 1;
  const ExecutionResourceTracker resources;
  using ReplayResult = llvm::Expected<ApplicationRuntimeValidation>;
  std::vector<std::unique_ptr<ReplayResult>> results(resolvedReplays.size());
  std::atomic_size_t next{0};
  std::atomic_size_t firstFailure{resolvedReplays.size()};
  const auto execute = [&](std::size_t ordinal) {
    results[ordinal] = std::make_unique<ReplayResult>(validateApplicationReplay(
        resolvedReplays[ordinal], alternative, imported->mapping.reference(),
        *deadline, artifacts, blobs));
    ReplayResult &result = *results[ordinal];
    if (!result || result->disposition !=
                       ApplicationMappingRuntimeDisposition::Completed) {
      std::size_t previous = firstFailure.load(std::memory_order_relaxed);
      while (ordinal < previous &&
             !firstFailure.compare_exchange_weak(previous, ordinal,
                                                 std::memory_order_relaxed)) {
      }
    }
  };
  const auto run = [&] {
    while (true) {
      const std::size_t ordinal = next.fetch_add(1, std::memory_order_relaxed);
      if (ordinal >= resolvedReplays.size() ||
          ordinal > firstFailure.load(std::memory_order_relaxed))
        return;
      execute(ordinal);
    }
  };

  // The first real replay establishes the immutable preparations before
  // workers share them. It consumes the same deadline and contributes its
  // ordinary Evidence; every replay still starts fresh mutable sessions.
  execute(0);
  next.store(1, std::memory_order_relaxed);
  if (firstFailure.load(std::memory_order_relaxed) == resolvedReplays.size()) {
    if (workerCount <= 1) {
      run();
    } else {
      const auto artifactAttachment = importCache.attachment();
      const auto fabricAttachment = fabricImports.attachment();
      llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
          static_cast<unsigned>(workerCount)));
      actualWorkerCount = pool.getMaxConcurrency();
      for (std::size_t worker = 0; worker != workerCount; ++worker)
        pool.async([&] {
          evaluation::ArtifactImportCacheScope workerImports(
              artifactAttachment);
          fabric::FabricArtifactImportSession workerFabric(fabricAttachment);
          run();
        });
      pool.wait();
    }
  }
  const ExecutionResourceStatistics usage = resources.observe();
  const bool deadlineObserved =
      *deadline && MonotonicClock::now() >= **deadline;
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields{
            {"operation", "application_runtime_replay"},
            {"replay_case_count", resolvedReplays.size()},
            {"worker_count", actualWorkerCount},
            {"site_cpu_core_budget",
             executionPolicy.inProcessClaim().cpuCores()},
            {"executed_case_count", llvm::count_if(results,
                                                   [](const auto &result) {
                                                     return static_cast<bool>(
                                                         result);
                                                   })},
            {"active_wall_time_ns", usage.activeWallTimeNanoseconds},
            {"deadline_observed", deadlineObserved},
            {"allocated_memory_bytes", usage.allocatedMemoryBytes}};
        if (usage.processCpuTimeDeltaNanoseconds)
          fields["process_cpu_time_ns"] = *usage.processCpuTimeDeltaNanoseconds;
        if (usage.currentResidentMemoryBytes)
          fields["current_resident_memory_bytes"] =
              *usage.currentResidentMemoryBytes;
        if (usage.peakResidentMemoryBytes)
          fields["peak_resident_memory_bytes"] = *usage.peakResidentMemoryBytes;
        if (firstFailure.load(std::memory_order_relaxed) <
            resolvedReplays.size())
          fields["first_failed_case_ordinal"] =
              firstFailure.load(std::memory_order_relaxed);
        return llvm::json::Value(std::move(fields));
      });

  ApplicationRuntimeValidation validation;
  validation.disposition = ApplicationMappingRuntimeDisposition::Completed;
  validation.resourceCoreCost = resourceCoreCost;
  llvm::Error failure = llvm::Error::success();
  for (std::size_t ordinal = 0; ordinal != results.size(); ++ordinal) {
    if (!results[ordinal])
      continue;
    ReplayResult &result = *results[ordinal];
    if (ordinal > firstFailure.load(std::memory_order_relaxed)) {
      if (!result)
        llvm::consumeError(result.takeError());
      continue;
    }
    if (!result) {
      failure = llvm::joinErrors(std::move(failure), result.takeError());
      continue;
    }
    validation.evidence.insert(validation.evidence.end(),
                               result->evidence.begin(),
                               result->evidence.end());
    validation.oracleEvidence.insert(validation.oracleEvidence.end(),
                                     result->oracleEvidence.begin(),
                                     result->oracleEvidence.end());
    if (result->dfgCycles)
      if (llvm::Error error =
              accumulateCycle(validation.dfgCycles, *result->dfgCycles, "DFG"))
        failure = llvm::joinErrors(std::move(failure), std::move(error));
    if (result->cgraCycles)
      if (llvm::Error error = accumulateCycle(validation.cgraCycles,
                                              *result->cgraCycles, "CGRA"))
        failure = llvm::joinErrors(std::move(failure), std::move(error));
    if (ordinal == firstFailure.load(std::memory_order_relaxed)) {
      validation.disposition = result->disposition;
      validation.spatialFifoFeedback = std::move(result->spatialFifoFeedback);
      validation.spatialOperandQueueFeedback =
          std::move(result->spatialOperandQueueFeedback);
      validation.spatialTransportFeedback =
          std::move(result->spatialTransportFeedback);
      validation.cgraMemoryContractRefusal = result->cgraMemoryContractRefusal;
    }
  }
  if (failure)
    return std::move(failure);
  if (validation.disposition ==
          ApplicationMappingRuntimeDisposition::Completed &&
      deadlineObserved)
    validation.disposition =
        ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
  llvm::sort(validation.evidence, artifactRootReferenceLess);
  validation.evidence.erase(
      std::unique(validation.evidence.begin(), validation.evidence.end()),
      validation.evidence.end());
  llvm::sort(validation.oracleEvidence, artifactRootReferenceLess);
  validation.oracleEvidence.erase(std::unique(validation.oracleEvidence.begin(),
                                              validation.oracleEvidence.end()),
                                  validation.oracleEvidence.end());
  return validation;
}

} // namespace loom::application::detail
