#include "Application/ActivationDecision.h"

#include "ApplicationRuntimeValidationInternal.h"
#include "ApplicationSystemRuntimeEvidence.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/Deployment.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application::detail {
namespace {
llvm::Error reject(ApplicationActivationDecisionErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ApplicationActivationDecisionError>(reason,
                                                              message.str());
}
} // namespace

llvm::Expected<ApplicationRuntimeEvidenceJoin>
resolveApplicationRuntimeEvidenceJoin(
    llvm::ArrayRef<ArtifactRootReference> runtimeEvidence,
    llvm::ArrayRef<ArtifactRootReference> oracleEvidence,
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference> replayCases,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ApplicationSystemRuntimeEvidenceContext *systemContext) {
  evaluation::ArtifactImportCacheScope importCache(artifacts, &blobs);
  fabric::FabricArtifactImportSession fabricImports;
  if (runtimeEvidence.empty() || oracleEvidence.empty() || replayCases.empty())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "application runtime requires replay inputs, runtime and "
                  "oracle Evidence");
  const std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      runtimeRoots(runtimeEvidence.begin(), runtimeEvidence.end(),
                   artifactRootReferenceLess);
  const std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      oracleRoots(oracleEvidence.begin(), oracleEvidence.end(),
                  artifactRootReferenceLess);
  if (runtimeRoots.size() != runtimeEvidence.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence repeats an Evidence root");
  if (oracleRoots.size() != oracleEvidence.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "oracle Evidence repeats an Evidence root");
  for (const ArtifactRootReference &oracle : oracleEvidence)
    if (runtimeRoots.find(oracle) == runtimeRoots.end())
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "oracle Evidence is outside the runtime Evidence set");

  struct InputPair final {
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
  };
  const auto pairLess = [](const InputPair &lhs, const InputPair &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  };
  struct EvidenceFacts final {
    ArtifactRootReference evidence;
    evaluation::EvaluationEvidenceDependencyProjection projection;
    std::vector<ArtifactRootReference> requestReferences;
  };
  struct ExecutionRecord final {
    EvidenceFacts facts;
    ArtifactRootReference execution;
  };
  struct CgraRecord final {
    ExecutionRecord record;
    ArtifactRootReference spatialMapping;
  };
  struct ReplayCase final {
    std::optional<ExecutionRecord> dfg;
    std::optional<CgraRecord> cgra;
    std::optional<EvidenceFacts> comparison;
  };
  std::map<InputPair, ReplayCase, decltype(pairLess)> cases(pairLess);
  for (const sim::SourceBackedDfgReplayCaseReference &replay : replayCases)
    if (!cases
             .emplace(InputPair{replay.workload, replay.runtimeInput},
                      ReplayCase{})
             .second)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "source-backed replay cases repeat an input pair");

  enum class ExecutionKind : std::uint8_t { Dfg, Cgra };
  struct ExecutionBinding final {
    ReplayCase *replay;
    ExecutionKind kind;
  };
  std::map<ArtifactRootReference, ExecutionBinding,
           decltype(&artifactRootReferenceLess)>
      executionByRoot(artifactRootReferenceLess);
  std::vector<EvidenceFacts> comparisons;
  std::vector<ArtifactRootReference> allExecutionOutputs;
  ApplicationRuntimeEvidenceJoin result;
  std::vector<ArtifactRootReference> systemEvidence;
  comparisons.reserve(oracleEvidence.size());

  // First index only exact dependency projections. Full imports are grouped
  // below by replay input, so DFG, CGRA and comparison validation share one
  // strictly imported input instead of cycling through the entire portfolio.
  // Strict Evidence import validates every output; indexing does not acquire
  // payloads that would be evicted before their owning case is visited.
  for (const ArtifactRootReference &evidence : runtimeEvidence) {
    auto projection = evaluation::importEvaluationEvidenceDependencyProjection(
        evidence, artifacts);
    if (!projection)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence failed dependency projection: " +
                        llvm::toString(projection.takeError()));
    if (projection->outcomeKind != evaluation::EvidenceOutcomeKind::Completed)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence is not completed");
    auto requestReferences =
        evaluation::importEvaluationRequestArtifactReferences(
            projection->request, artifacts);
    if (!requestReferences)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence Request cannot be imported: " +
                        llvm::toString(requestReferences.takeError()));
    result.requestDependencies.insert(result.requestDependencies.end(),
                                      requestReferences->begin(),
                                      requestReferences->end());
    std::vector<ArtifactRootReference> outputExecutions;
    for (const evaluation::ModelOutputBinding &binding :
         projection->outputBindings)
      for (const ArtifactRootReference &root : binding.artifacts) {
        if (root.schemaIdentity == sim::simulationExecutionSchema.identity &&
            root.schemaVersion == sim::simulationExecutionSchema.version) {
          outputExecutions.push_back(root);
          allExecutionOutputs.push_back(root);
        }
      }
    const bool isSystem =
        llvm::any_of(*requestReferences, [](const auto &root) {
          return root.schemaIdentity == deployment::deploymentSchema.identity;
        });
    if (isSystem) {
      if (!systemContext || oracleRoots.count(evidence))
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "System runtime Evidence has no source invocation owner");
      systemEvidence.push_back(evidence);
      result.executionOutputs.insert(result.executionOutputs.end(),
                                     outputExecutions.begin(),
                                     outputExecutions.end());
      continue;
    }
    EvidenceFacts row{evidence, std::move(*projection),
                      std::move(*requestReferences)};
    std::optional<ArtifactRootReference> workload;
    std::optional<ArtifactRootReference> runtimeInput;
    for (const ArtifactRootReference &reference : row.requestReferences) {
      if (reference.schemaIdentity == sim::simulationWorkloadSchema.identity &&
          reference.schemaVersion == sim::simulationWorkloadSchema.version) {
        if (workload)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence Request repeats its SimulationWorkload");
        workload = reference;
      }
      if (reference.schemaIdentity ==
              sim::simulationRuntimeInputSchema.identity &&
          reference.schemaVersion ==
              sim::simulationRuntimeInputSchema.version) {
        if (runtimeInput)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence Request repeats its SimulationRuntimeInput");
        runtimeInput = reference;
      }
    }
    const bool hasDataflow =
        llvm::is_contained(row.requestReferences, dataflow);
    std::vector<ArtifactRootReference> selectedMappings;
    for (const ArtifactRootReference &mapping : spatialMappings)
      if (llvm::is_contained(row.requestReferences, mapping))
        selectedMappings.push_back(mapping);
    if (!hasDataflow && selectedMappings.empty()) {
      if (oracleRoots.find(evidence) == oracleRoots.end())
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "non-execution Evidence is not declared as oracle Evidence");
      comparisons.push_back(std::move(row));
      continue;
    }
    if (!workload || !runtimeInput)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "runtime Evidence Request has no exact workload and runtime input");
    if (selectedMappings.size() > 1)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "runtime Evidence Request repeats a selected SpatialMapping");
    if (oracleRoots.count(evidence) || outputExecutions.size() != 1)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime execution Evidence must have one execution output "
                    "and cannot serve as comparison Evidence");
    auto found = cases.find(InputPair{*workload, *runtimeInput});
    if (found == cases.end())
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence names a foreign replay input pair");
    ReplayCase &replay = found->second;
    const ExecutionKind kind =
        selectedMappings.empty() ? ExecutionKind::Dfg : ExecutionKind::Cgra;
    if (!executionByRoot
             .emplace(outputExecutions.front(), ExecutionBinding{&replay, kind})
             .second)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence repeats a SimulationExecution output");
    ExecutionRecord record{std::move(row), outputExecutions.front()};
    if (kind == ExecutionKind::Dfg) {
      if (replay.dfg)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "replay input has more than one DFG execution");
      replay.dfg = std::move(record);
    } else {
      if (replay.cgra)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "replay input has more than one CGRA execution");
      replay.cgra = CgraRecord{std::move(record), selectedMappings.front()};
    }
  }

  if (!systemEvidence.empty()) {
    auto ticks = resolveApplicationSystemRuntimeEvidenceJoin(
        systemEvidence, *systemContext, artifacts, blobs);
    if (!ticks)
      return ticks.takeError();
    result.systemComputationTicks = *ticks;
  }

  for (EvidenceFacts &row : comparisons) {
    std::vector<ExecutionBinding> compared;
    for (const ArtifactRootReference &reference : row.requestReferences)
      if (reference.schemaIdentity == sim::simulationExecutionSchema.identity &&
          reference.schemaVersion == sim::simulationExecutionSchema.version) {
        auto found = executionByRoot.find(reference);
        if (found == executionByRoot.end())
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "oracle Evidence names a foreign SimulationExecution");
        compared.push_back(found->second);
      }
    if (compared.size() != 2 || compared[0].kind == compared[1].kind)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence does not compare one DFG and one CGRA execution");
    if (compared[0].replay != compared[1].replay)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence compares executions from different replay inputs");
    ReplayCase &replay = *compared[0].replay;
    if (replay.comparison)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "replay input has more than one oracle comparison");
    replay.comparison = std::move(row);
  }

  auto importExecution =
      [&](const ExecutionRecord &record,
          const evaluation::CaseArtifactResolution &resolution,
          std::uint64_t &cycleTotal)
      -> llvm::Expected<evaluation::EvaluationEvidence> {
    auto strict = evaluation::importEvaluationEvidence(
        record.facts.evidence, resolution, artifacts, blobs);
    if (!strict)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence failed strict import: " +
                        llvm::toString(strict.takeError()));
    if (strict->requestRef() != record.facts.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "strict runtime Evidence differs from its dependency projection");
    const auto *completed =
        std::get_if<evaluation::CompletedEvidence>(&strict->outcome());
    const auto *point = completed && completed->metricResults.size() == 1
                            ? std::get_if<evaluation::PointObservation>(
                                  &completed->metricResults.front().observation)
                            : nullptr;
    const auto *cycles =
        point ? std::get_if<evaluation::IntegerValue>(&point->value) : nullptr;
    if (!cycles || cycles->value() < 0)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence has no nonnegative cycle metric");
    const std::uint64_t cycleValue =
        static_cast<std::uint64_t>(cycles->value());
    if (cycleValue > std::numeric_limits<std::uint64_t>::max() - cycleTotal)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence cycle total overflowed");
    cycleTotal += cycleValue;
    std::optional<ArtifactRootReference> execution;
    for (const evaluation::ModelOutputBinding &binding :
         strict->outputBindings())
      for (const ArtifactRootReference &output : binding.artifacts)
        if (output.schemaIdentity == sim::simulationExecutionSchema.identity &&
            output.schemaVersion == sim::simulationExecutionSchema.version) {
          if (execution)
            return reject(
                ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                "runtime Evidence repeats its SimulationExecution output");
          execution = output;
        }
    if (!execution || *execution != record.execution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence execution differs from its projection");
    auto executionRequest =
        sim::simulationExecutionRequestReference(*execution, artifacts);
    if (!executionRequest)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot import SimulationExecution Request: " +
                        llvm::toString(executionRequest.takeError()));
    if (*executionRequest != record.facts.projection.request)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "SimulationExecution and Evidence name different Requests");
    result.executionOutputs.push_back(*execution);
    return std::move(*strict);
  };

  for (const auto &[inputs, replay] : cases) {
    if (!replay.dfg || !replay.cgra || !replay.comparison)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "replay input lacks exact DFG, CGRA and oracle coverage");
    auto dfgResolution = evaluation::models::resolveDfgSimulationCase(
        dataflow, inputs.workload, inputs.runtimeInput, artifacts);
    if (!dfgResolution)
      return reject(
          ApplicationActivationDecisionErrorReason::DependencyMismatch,
          "source-backed replay case failed strict import: " +
              llvm::toString(dfgResolution.takeError()));
    auto owners = evaluation::models::resolveCgraSimulationCaseOwners(
        replay.cgra->spatialMapping, artifacts);
    if (!owners)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot resolve selected CGRA runtime case: " +
                        llvm::toString(owners.takeError()));
    if (owners->dataflow != dataflow)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "CGRA runtime case names a foreign canonical Dataflow");
    auto cgraResolution =
        evaluation::models::resolveCgraSimulationCaseResolution(
            *owners, inputs.workload, inputs.runtimeInput);
    if (!cgraResolution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot resolve selected CGRA runtime case: " +
                        llvm::toString(cgraResolution.takeError()));
    auto dfg = importExecution(*replay.dfg, *dfgResolution, result.dfgCycles);
    if (!dfg)
      return dfg.takeError();
    auto cgra = importExecution(replay.cgra->record, *cgraResolution,
                                result.cgraCycles);
    if (!cgra)
      return cgra.takeError();
    auto terminal = evaluation::models::classifyCompletedCgraSimulationEvidence(
        *cgra, *cgraResolution, artifacts, blobs);
    if (!terminal)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot classify CGRA runtime terminal: " +
                        llvm::toString(terminal.takeError()));
    result.allCgraExecutionsRetired &=
        *terminal ==
        evaluation::models::CgraSimulationEvidenceTerminal::Retired;
    auto comparisonResolution =
        evaluation::models::resolveSimulationComparisonCase(
            replay.dfg->execution, *dfgResolution,
            replay.cgra->record.execution, *cgraResolution, artifacts, blobs);
    if (!comparisonResolution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot resolve SimulationComparison Evidence: " +
                        llvm::toString(comparisonResolution.takeError()));
    const EvidenceFacts &comparison = *replay.comparison;
    auto strict = evaluation::importEvaluationEvidence(
        comparison.evidence, *comparisonResolution, artifacts, blobs);
    if (!strict)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence failed strict SimulationComparison import: " +
              llvm::toString(strict.takeError()));
    if (strict->requestRef() != comparison.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence failed strict SimulationComparison import");
    const auto *completed =
        std::get_if<evaluation::CompletedEvidence>(&strict->outcome());
    if (!completed || completed->findingResults.size() != 1 ||
        !std::holds_alternative<evaluation::AbsentFinding>(
            completed->findingResults.front().result))
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence did not establish an absent comparison finding");
  }

  llvm::sort(allExecutionOutputs, artifactRootReferenceLess);
  allExecutionOutputs.erase(
      std::unique(allExecutionOutputs.begin(), allExecutionOutputs.end()),
      allExecutionOutputs.end());
  llvm::sort(result.executionOutputs, artifactRootReferenceLess);
  if (result.executionOutputs != allExecutionOutputs)
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence has an unclassified or repeated "
                  "SimulationExecution output");
  llvm::sort(result.requestDependencies, artifactRootReferenceLess);
  result.requestDependencies.erase(
      std::unique(result.requestDependencies.begin(),
                  result.requestDependencies.end()),
      result.requestDependencies.end());
  return result;
}

} // namespace loom::application::detail
