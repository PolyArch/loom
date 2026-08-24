#include "ApplicationRuntimeValidationInternal.h"

#include "ExecutionGlue.h"

#include "Common/MappingDebugLog.h"
#include "Common/ArtifactText.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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
  const auto deadline = std::chrono::system_clock::time_point{
      std::chrono::nanoseconds{static_cast<std::int64_t>(
          *policy.dispatchNotAfterUnixNanoseconds())}};
  const auto remaining = deadline - std::chrono::system_clock::now();
  if (remaining <= std::chrono::system_clock::duration::zero())
    return MonotonicClock::now();
  return MonotonicClock::now() +
         std::chrono::duration_cast<MonotonicClock::duration>(remaining);
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
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
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
                                    std::move(*dataflowView),
                                    std::move(*system)};
}

llvm::Expected<ApplicationRuntimeValidation> validateApplicationMappingRuntime(
    const PreparedApplicationBuild &prepared,
    const PreparedApplicationMappingAlternative &alternative,
    const dse::JointDesignExecution &execution,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto imported = importApplicationMapping(execution, artifacts);
  if (!imported)
    return imported.takeError();
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
        std::nullopt};

  auto contexts = mapping::projectSystemExecutionContexts(
      imported->dataflowView, imported->mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();
  auto invocationPlan = deriveApplicationSpatialInvocationPlan(
      imported->dataflowView, prepared.sourceInvocation.entrySymbol);
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

  std::vector<ResolvedApplicationReplay> resolvedReplays;
  resolvedReplays.reserve((*software)->replayCases.size());
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       (*software)->replayCases) {
    auto inputs = sim::importSpatialSimulationInputs(
        replay.workload, replay.runtimeInput, artifacts);
    if (!inputs)
      return inputs.takeError();
    if (inputs->dataflow.identity() != alternative.dataflow.artifact)
      return invalid("source-backed replay names a foreign final Dataflow");
    const sim::SpatialSimulationWorkload *workload = inputs->workload.spatial();
    if (!workload)
      return invalid("source-backed replay is not a Spatial workload");
    auto selected = mapping::selectSystemSpatialExecutionContext(
        *contexts, workload->launchRef, workload->denseCoordinates);
    if (!selected)
      return selected.takeError();
    auto spatialMapping =
        mapping::importSpatialMapping(selected->spatialMapping, artifacts);
    if (!spatialMapping)
      return spatialMapping.takeError();
    const ArtifactRootReference module{
        fabric::fabricArtifactSchema.identity.str(),
        fabric::fabricArtifactSchema.version,
        spatialMapping->view().fabricIdentity()};
    resolvedReplays.push_back(
        {&replay,
         {workload->launchRef.rootThreadLaunch, workload->launchRef,
          workload->denseCoordinates, selected->context},
         module,
         selected->spatialMapping});
  }

  for (const ResolvedApplicationReplay &replay : resolvedReplays)
    if (!llvm::is_contained(requiredPoints, replay.point))
      return invalid(
          "source-backed replay is outside the ABI invocation plan");

  for (const ApplicationSpatialRuntimePoint &required : requiredPoints) {
    const bool covered = llvm::any_of(
        resolvedReplays, [&](const ResolvedApplicationReplay &replay) {
          return replay.point == required;
        });
    if (covered)
      continue;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::MappingFailure,
        [&](llvm::json::Object &fields) {
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
          fields["missing_spatial_mapping"] = formatArtifactIdentityHex(
              required.context.spatialMapping);
        });
    return ApplicationRuntimeValidation{
        ApplicationMappingRuntimeDisposition::ProofNotEstablished,
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
  }
  auto deadline = applicationReplayDeadline(executionPolicy);
  if (!deadline)
    return deadline.takeError();

  ApplicationRuntimeValidation validation;
  validation.disposition = ApplicationMappingRuntimeDisposition::Completed;
  for (const ResolvedApplicationReplay &resolved : resolvedReplays) {
    const sim::SourceBackedDfgReplayCaseReference &replay =
        *resolved.reference;
    if (*deadline && MonotonicClock::now() >= **deadline) {
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
        *preparedDfg, {kApplicationReplayExecutionLimit, *deadline}, artifacts,
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
    if (llvm::Error error =
            accumulateCycle(validation.dfgCycles, *dfgCycles, "DFG"))
      return std::move(error);

    auto preparedCgra = evaluation::models::prepareCgraSimulationEvaluation(
        alternative.dataflow, resolved.module, resolved.spatialMapping,
        replay.workload, replay.runtimeInput, alternative.plan.resolvedConfig,
        artifacts, blobs);
    if (!preparedCgra)
      return preparedCgra.takeError();
    auto cgraEvaluation =
        evaluation::models::evaluateCgraSimulationWithDiagnostics(
            *preparedCgra, {kApplicationReplayExecutionLimit, *deadline},
            artifacts, blobs);
    if (!cgraEvaluation)
      return cgraEvaluation.takeError();
    evaluation::EvaluationEvidence &cgraEvidence = cgraEvaluation->evidence;
    if (cgraEvaluation->closedWait) {
      auto operandFeedback = dse::deriveSpatialOperandQueueRuntimeFeedback(
          imported->mapping.reference(), *cgraEvaluation->closedWait,
          artifacts);
      if (!operandFeedback)
        return operandFeedback.takeError();
      dse::emitSpatialOperandQueueRuntimeFeedback(*operandFeedback);
      const auto operandPriority =
          [](dse::SpatialOperandQueueRuntimeFeedbackDisposition value) {
            switch (value) {
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact:
              return 2;
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::
                ProofNotEstablished:
              return 1;
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::
                Unsupported:
              return 0;
            }
            llvm_unreachable(
                "unknown Spatial operand-queue feedback disposition");
          };
      if (!validation.spatialOperandQueueFeedback ||
          operandPriority(operandFeedback->disposition) >
              operandPriority(
                  validation.spatialOperandQueueFeedback->disposition))
        validation.spatialOperandQueueFeedback = std::move(*operandFeedback);
      auto feedback = dse::deriveSpatialFifoRuntimeFeedback(
          imported->mapping.reference(), resolved.spatialMapping,
          *cgraEvaluation->closedWait, artifacts);
      if (!feedback)
        return feedback.takeError();
      dse::emitSpatialFifoRuntimeFeedback(*feedback);
      const auto priority = [](dse::SpatialFifoRuntimeFeedbackDisposition
                                   value) {
        switch (value) {
        case dse::SpatialFifoRuntimeFeedbackDisposition::Exact:
          return 2;
        case dse::SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished:
          return 1;
        case dse::SpatialFifoRuntimeFeedbackDisposition::Unsupported:
          return 0;
        }
        llvm_unreachable("unknown Spatial FIFO feedback disposition");
      };
      if (!validation.spatialFifoFeedback ||
          priority(feedback->disposition) >
              priority(validation.spatialFifoFeedback->disposition))
        validation.spatialFifoFeedback = std::move(*feedback);
      auto transportFeedback = dse::deriveSpatialTransportRuntimeFeedback(
          imported->mapping.reference(), *cgraEvaluation->closedWait,
          artifacts);
      if (!transportFeedback)
        return transportFeedback.takeError();
      dse::emitSpatialTransportRuntimeFeedback(*transportFeedback);
      const auto transportPriority =
          [](dse::SpatialTransportRuntimeFeedbackDisposition value) {
            switch (value) {
            case dse::SpatialTransportRuntimeFeedbackDisposition::Exact:
              return 2;
            case dse::SpatialTransportRuntimeFeedbackDisposition::
                ProofNotEstablished:
              return 1;
            case dse::SpatialTransportRuntimeFeedbackDisposition::Unsupported:
              return 0;
            }
            llvm_unreachable("unknown Spatial transport feedback disposition");
          };
      if (!validation.spatialTransportFeedback ||
          transportPriority(transportFeedback->disposition) >
              transportPriority(
                  validation.spatialTransportFeedback->disposition))
        validation.spatialTransportFeedback = std::move(*transportFeedback);
    }
    auto cgraEvidenceReference =
        evaluation::publishEvaluationEvidence(cgraEvidence, artifacts);
    if (!cgraEvidenceReference)
      return cgraEvidenceReference.takeError();
    validation.evidence.push_back(*cgraEvidenceReference);
    if (cgraEvidence.outcomeKind() !=
        evaluation::EvidenceOutcomeKind::Completed) {
      emitRuntimeEvidenceFailure("cgra_simulation", cgraEvidence);
      validation.disposition = runtimeDisposition(cgraEvidence.outcomeKind());
      return validation;
    }
    auto cgraExecution = requireExecutionOutput(cgraEvidence);
    if (!cgraExecution)
      return cgraExecution.takeError();
    auto cgraCycles = requireCompletedCycleMetric(cgraEvidence);
    if (!cgraCycles)
      return cgraCycles.takeError();
    if (llvm::Error error =
            accumulateCycle(validation.cgraCycles, *cgraCycles, "CGRA"))
      return std::move(error);
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
        InvocationDiagnosticEvent::Statistics, [&] {
          llvm::json::Object fields;
          fields["measurement_kind"] = "direct_and_derived";
          fields["direct"] = llvm::json::Object{
              {"dfg_cycles", *dfgCycles}, {"cgra_cycles", *cgraCycles}};
          fields["derived"] = llvm::json::Object{
              {"cycle_delta", *cgraCycles >= *dfgCycles
                                   ? *cgraCycles - *dfgCycles
                                   : 0},
              {"cgra_to_dfg_ratio",
               llvm::json::Object{{"numerator", *cgraCycles},
                                  {"denominator", *dfgCycles}}},
              {"cgra_is_slower", *cgraCycles > *dfgCycles}};
          fields["operation"] = "simulation_cycle_comparison";
          fields["dataflow"] = formatArtifactRootReferenceJson(
              alternative.dataflow);
          fields["spatial_mapping"] = formatArtifactRootReferenceJson(
              resolved.spatialMapping);
          fields["dfg_request"] = formatArtifactRootReferenceJson(
              evaluation::evaluationRequestReference(preparedDfg->request));
          fields["cgra_request"] = formatArtifactRootReferenceJson(
              evaluation::evaluationRequestReference(preparedCgra->request));
          fields["dfg_cycles"] = *dfgCycles;
          fields["cgra_cycles"] = *cgraCycles;
          fields["cycle_delta"] = *cgraCycles >= *dfgCycles
                                       ? *cgraCycles - *dfgCycles
                                       : 0;
          fields["cgra_to_dfg_ratio"] =
              llvm::json::Object{{"numerator", *cgraCycles},
                                 {"denominator", *dfgCycles}};
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
    if (std::holds_alternative<evaluation::AbsentFinding>(comparisonResult))
      continue;
    validation.disposition =
        std::holds_alternative<evaluation::NotApplicableFinding>(
            comparisonResult)
            ? ApplicationMappingRuntimeDisposition::ProofNotEstablished
            : ApplicationMappingRuntimeDisposition::ExecutionFailed;
    return validation;
  }
  llvm::sort(validation.evidence, artifactRootReferenceLess);
  validation.evidence.erase(
      std::unique(validation.evidence.begin(), validation.evidence.end()),
      validation.evidence.end());
  return validation;
}

} // namespace loom::application::detail
