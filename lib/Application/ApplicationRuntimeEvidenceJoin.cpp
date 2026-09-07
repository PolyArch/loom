#include "Application/ActivationDecision.h"

#include "ApplicationRuntimeValidationInternal.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
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
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  evaluation::ArtifactImportCacheScope importCache(artifacts, &blobs);
  fabric::FabricArtifactImportSession fabricImports;
  if (runtimeEvidence.empty() || oracleEvidence.empty())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "application runtime requires runtime and oracle Evidence");
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
  for (const ArtifactRootReference &oracle : oracleEvidence) {
    if (runtimeRoots.find(oracle) == runtimeRoots.end())
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "oracle Evidence is outside the runtime Evidence set");
  }

  struct InputPair final {
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
    bool operator==(const InputPair &other) const {
      return workload == other.workload && runtimeInput == other.runtimeInput;
    }
  };
  const auto pairLess = [](const InputPair &lhs, const InputPair &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  };
  std::map<InputPair, evaluation::CaseArtifactResolution, decltype(pairLess)>
      replayResolutions(pairLess);
  for (const sim::SourceBackedDfgReplayCaseReference &replay : replayCases) {
    auto resolution = evaluation::models::resolveDfgSimulationCase(
        dataflow, replay.workload, replay.runtimeInput, artifacts);
    if (!resolution)
      return reject(
          ApplicationActivationDecisionErrorReason::DependencyMismatch,
          "source-backed replay case failed strict import: " +
              llvm::toString(resolution.takeError()));
    if (!replayResolutions
             .emplace(InputPair{replay.workload, replay.runtimeInput},
                      std::move(*resolution))
             .second)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "source-backed replay cases repeat an input pair");
  }

  enum class ExecutionKind : std::uint8_t { Dfg, Cgra };
  struct ExecutionRecord final {
    ArtifactRootReference evidence;
    ArtifactRootReference execution;
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
    evaluation::CaseArtifactResolution resolution;
    ExecutionKind kind;
  };
  struct EvidenceFacts final {
    ArtifactRootReference evidence;
    evaluation::EvaluationEvidenceDependencyProjection projection;
    std::vector<ArtifactRootReference> requestReferences;
  };
  std::vector<EvidenceFacts> evidenceFacts;
  std::vector<ExecutionRecord> executions;
  std::vector<ArtifactRootReference> allExecutionOutputs;
  ApplicationRuntimeEvidenceJoin result;
  evidenceFacts.reserve(runtimeEvidence.size());
  executions.reserve(runtimeEvidence.size());
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
    for (const evaluation::ModelOutputBinding &binding :
         projection->outputBindings)
      for (const ArtifactRootReference &root : binding.artifacts) {
        auto stored = artifacts.get(root);
        if (!stored)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence output is unavailable: " +
                  llvm::toString(stored.takeError()));
        if (root.schemaIdentity == sim::simulationExecutionSchema.identity &&
            root.schemaVersion == sim::simulationExecutionSchema.version)
          allExecutionOutputs.push_back(root);
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
      evidenceFacts.push_back(std::move(row));
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
    std::optional<evaluation::CaseArtifactResolution> resolution;
    ExecutionKind kind = ExecutionKind::Dfg;
    if (!selectedMappings.empty()) {
      auto resolved = evaluation::models::resolveCgraSimulationCase(
          selectedMappings.front(), *workload, *runtimeInput, artifacts);
      if (!resolved)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "cannot resolve selected CGRA runtime case: " +
                llvm::toString(resolved.takeError()));
      if (resolved->canonicalDataflow != dataflow)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "CGRA runtime case names a foreign canonical Dataflow");
      resolution.emplace(std::move(resolved->resolution));
      kind = ExecutionKind::Cgra;
    } else {
      const auto resolved =
          replayResolutions.find(InputPair{*workload, *runtimeInput});
      if (resolved == replayResolutions.end())
        return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                      "DFG runtime Evidence names a foreign replay input pair");
      resolution.emplace(resolved->second);
    }
    auto strict = evaluation::importEvaluationEvidence(evidence, *resolution,
                                                       artifacts, blobs);
    if (!strict)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence failed strict import: " +
                        llvm::toString(strict.takeError()));
    if (strict->requestRef() != row.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "strict runtime Evidence differs from its dependency projection");
    if (kind == ExecutionKind::Cgra) {
      auto terminal =
          evaluation::models::classifyCompletedCgraSimulationEvidence(
              *strict, *resolution, artifacts, blobs);
      if (!terminal)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "cannot classify CGRA runtime terminal: " +
                llvm::toString(terminal.takeError()));
      result.allCgraExecutionsRetired &=
          *terminal ==
          evaluation::models::CgraSimulationEvidenceTerminal::Retired;
    }
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
    std::uint64_t &cycleTotal =
        kind == ExecutionKind::Dfg ? result.dfgCycles : result.cgraCycles;
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
    if (!execution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence has no SimulationExecution output");
    auto executionRequest =
        sim::simulationExecutionRequestReference(*execution, artifacts);
    if (!executionRequest)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot import SimulationExecution Request: " +
                        llvm::toString(executionRequest.takeError()));
    if (*executionRequest != row.projection.request)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "SimulationExecution and Evidence name different Requests");
    result.executionOutputs.push_back(*execution);
    executions.push_back({evidence, *execution, *workload, *runtimeInput,
                          std::move(*resolution), kind});
    evidenceFacts.push_back(std::move(row));
  }
  if (executions.empty() ||
      !llvm::any_of(executions,
                    [](const ExecutionRecord &record) {
                      return record.kind == ExecutionKind::Dfg;
                    }) ||
      !llvm::any_of(executions, [](const ExecutionRecord &record) {
        return record.kind == ExecutionKind::Cgra;
      }))
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence does not bind both DFG and CGRA execution");

  auto canonicalizePairs = [&](std::vector<InputPair> &pairs) {
    llvm::sort(pairs, pairLess);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  };
  std::vector<InputPair> expectedPairs;
  for (const auto &replay : replayResolutions)
    expectedPairs.push_back(replay.first);
  std::vector<InputPair> dfgPairs;
  std::vector<InputPair> cgraPairs;
  for (const ExecutionRecord &record : executions)
    (record.kind == ExecutionKind::Dfg ? dfgPairs : cgraPairs)
        .push_back({record.workload, record.runtimeInput});
  const std::size_t dfgExecutionCount = dfgPairs.size();
  const std::size_t cgraExecutionCount = cgraPairs.size();
  canonicalizePairs(dfgPairs);
  canonicalizePairs(cgraPairs);
  if (expectedPairs.empty() ||
      dfgPairs != expectedPairs || cgraPairs != expectedPairs ||
      dfgExecutionCount != expectedPairs.size() ||
      cgraExecutionCount != expectedPairs.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence does not join through exact source-backed "
                  "replay inputs");

  llvm::sort(allExecutionOutputs, artifactRootReferenceLess);
  allExecutionOutputs.erase(
      std::unique(allExecutionOutputs.begin(), allExecutionOutputs.end()),
      allExecutionOutputs.end());
  llvm::sort(result.executionOutputs, artifactRootReferenceLess);
  if (result.executionOutputs != allExecutionOutputs)
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence has an unclassified or repeated "
                  "SimulationExecution output");

  std::map<ArtifactRootReference, const ExecutionRecord *,
           decltype(&artifactRootReferenceLess)>
      executionByRoot(artifactRootReferenceLess);
  for (const ExecutionRecord &record : executions)
    executionByRoot.emplace(record.execution, &record);

  std::vector<InputPair> comparisonPairs;
  std::vector<ArtifactRootReference> comparisonEvidence;
  for (const EvidenceFacts &row : evidenceFacts) {
    if (oracleRoots.find(row.evidence) == oracleRoots.end())
      continue;
    std::vector<const ExecutionRecord *> compared;
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
    if (compared.size() != 2 || compared[0]->kind == compared[1]->kind)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence does not compare one DFG and one CGRA execution");
    const ExecutionRecord *dfg =
        compared[0]->kind == ExecutionKind::Dfg ? compared[0] : compared[1];
    const ExecutionRecord *cgra =
        compared[0]->kind == ExecutionKind::Cgra ? compared[0] : compared[1];
    const InputPair dfgPair{dfg->workload, dfg->runtimeInput};
    const InputPair cgraPair{cgra->workload, cgra->runtimeInput};
    if (!(dfgPair == cgraPair))
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence compares executions from different replay inputs");
    auto resolution = evaluation::models::resolveSimulationComparisonCase(
        dfg->execution, dfg->resolution, cgra->execution, cgra->resolution,
        artifacts, blobs);
    if (!resolution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot resolve SimulationComparison Evidence: " +
                        llvm::toString(resolution.takeError()));
    auto strict = evaluation::importEvaluationEvidence(
        row.evidence, *resolution, artifacts, blobs);
    if (!strict)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "oracle Evidence failed strict SimulationComparison import: " +
                        llvm::toString(strict.takeError()));
    if (strict->requestRef() != row.projection.request ||
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
    comparisonPairs.push_back(dfgPair);
    comparisonEvidence.push_back(row.evidence);
  }
  const std::size_t comparisonCount = comparisonPairs.size();
  canonicalizePairs(comparisonPairs);
  llvm::sort(comparisonEvidence, artifactRootReferenceLess);
  const std::vector<ArtifactRootReference> expectedOracleEvidence(
      oracleRoots.begin(), oracleRoots.end());
  if (comparisonEvidence != expectedOracleEvidence ||
      comparisonPairs != expectedPairs ||
      comparisonCount != expectedPairs.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "oracle Evidence does not provide exact one-to-one "
                  "SimulationComparison coverage for the source-backed "
                  "replay inputs");
  llvm::sort(result.requestDependencies, artifactRootReferenceLess);
  result.requestDependencies.erase(
      std::unique(result.requestDependencies.begin(),
                  result.requestDependencies.end()),
      result.requestDependencies.end());
  return result;
}

} // namespace loom::application::detail
