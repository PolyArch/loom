#include "DSE/StructuredOwnership.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_ownership_dse_invalid: " +
                                     message);
}

struct OwnershipWorkItem final {
  frontend::SpatialOwnershipScope scope;
  frontend::SpatialOwnershipDecisionPoint decision;
};

struct MaterializedOwnershipWorkItem final {
  ArtifactRootReference candidate;
};

using OwnershipAttemptResult =
    std::variant<MaterializedOwnershipWorkItem,
                 StructuredOwnershipCandidateRejectionRecord>;

void mergeEvidenceReferences(std::vector<ArtifactRootReference> &destination,
                             llvm::ArrayRef<ArtifactRootReference> additional) {
  destination.insert(destination.end(), additional.begin(), additional.end());
  llvm::sort(destination, artifactRootReferenceLess);
  destination.erase(std::unique(destination.begin(), destination.end()),
                    destination.end());
}

llvm::Expected<OwnershipAttemptResult> materializeOwnershipWorkItem(
    const frontend::StructuredProgramCandidate &parent,
    const sim::CanonicalSimulationWorkload &workload,
    const ArtifactRootReference &workloadReference,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const ArtifactRootReference &runtimeInputReference,
    const sim::NativeStructuredProgramObservations &sourceObservations,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore, const OwnershipWorkItem &workItem) {
  auto candidate = frontend::materializeSpatialOwnershipDecision(
      parent, workItem.scope, workItem.decision, fabric, options.lowering);
  if (!candidate) {
    std::optional<StructuredOwnershipCandidateRejectionRecord> rejection;
    llvm::Error unhandled = llvm::handleErrors(
        candidate.takeError(),
        [&](const frontend::SpatialOwnershipCandidateRejection &error) {
          rejection.emplace(StructuredOwnershipCandidateRejectionRecord{
              error.kind(), error.message()});
        });
    if (unhandled)
      return std::move(unhandled);
    if (!rejection)
      return invalid("candidate failed without a classified error");
    return OwnershipAttemptResult{std::move(*rejection)};
  }

  auto reference = frontend::publishStructuredProgram(
      candidate->structuredProgram, artifactStore);
  if (!reference)
    return reference.takeError();
  const evaluation::models::StructuredFabricAnalyticInvocation invocation{
      workloadReference, runtimeInputReference, parent, sourceObservations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              *reference,
              {candidate->structuredProgram, &candidate->canonicalDataflow,
               workItem.scope.selection},
              invocation, fabric, config, artifactStore))
    return std::move(error);
  return OwnershipAttemptResult{
      MaterializedOwnershipWorkItem{std::move(*reference)}};
}

} // namespace

llvm::Expected<StructuredOwnershipExplorationOutcome>
generateAndPromoteStructuredOwnership(
    const frontend::StructuredProgramCandidate &parent,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore) {
  if (options.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (llvm::Error error =
          evaluation::models::registerStructuredFabricAnalyticModel())
    return std::move(error);
  if (llvm::Error error =
          evaluation::models::registerStructuredProgramFunctionalModel())
    return std::move(error);

  std::vector<ArtifactRootReference> candidateReferences;
  auto parentReference =
      frontend::publishStructuredProgram(parent, artifactStore);
  if (!parentReference)
    return parentReference.takeError();
  auto workloadReference =
      sim::publishSimulationWorkload(workload, artifactStore);
  if (!workloadReference)
    return workloadReference.takeError();
  auto runtimeInputReference =
      sim::publishSimulationRuntimeInput(runtimeInput, artifactStore);
  if (!runtimeInputReference)
    return runtimeInputReference.takeError();
  auto sourceObservations =
      sim::executeNativeStructuredProgram(parent, workload, runtimeInput);
  if (!sourceObservations)
    return sourceObservations.takeError();
  const evaluation::models::StructuredFabricAnalyticInvocation invocation{
      *workloadReference, *runtimeInputReference, parent, *sourceObservations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              *parentReference,
              evaluation::models::StructuredFabricAnalyticCandidateProjection{
                  parent, nullptr, std::nullopt},
              invocation, fabric, config, artifactStore))
    return std::move(error);
  candidateReferences.push_back(*parentReference);

  auto domain = frontend::enumerateSpatialOwnershipScopeDomain(parent);
  if (!domain)
    return domain.takeError();
  std::vector<OwnershipWorkItem> workItems;
  struct PlannedDisposition final {
    StructuredOwnershipCandidateCoordinate coordinate;
    std::variant<std::size_t, StructuredOwnershipCandidateRejectionRecord>
        source;
  };
  std::vector<PlannedDisposition> plannedDispositions;
  for (const frontend::SpatialOwnershipScopeDomainEntry &entry : *domain) {
    if (const auto *rejected =
            std::get_if<frontend::RejectedSpatialOwnershipScope>(&entry)) {
      plannedDispositions.push_back(
          {StructuredOwnershipCandidateCoordinate{rejected->scope,
                                                  std::nullopt},
           StructuredOwnershipCandidateRejectionRecord{
               frontend::SpatialOwnershipCandidateRejectionKind::NonFinalizable,
               rejected->message}});
      continue;
    }
    const auto &scope = std::get<frontend::SpatialOwnershipScope>(entry);
    auto decisions = frontend::enumerateSpatialOwnershipDecisionDomain(
        parent, scope.selection);
    if (!decisions)
      return decisions.takeError();
    for (const frontend::SpatialOwnershipDecisionPoint &decision : *decisions) {
      const std::size_t workIndex = workItems.size();
      workItems.push_back({scope, decision});
      plannedDispositions.push_back(
          {StructuredOwnershipCandidateCoordinate{scope, decision}, workIndex});
    }
  }
  struct WorkResult final {
    std::optional<OwnershipAttemptResult> attempt;
    std::optional<llvm::Error> error;
  };
  std::vector<WorkResult> results(workItems.size());
  auto execute = [&](const frontend::StructuredProgramCandidate &workerParent,
                     std::size_t index) {
    auto result = materializeOwnershipWorkItem(
        workerParent, workload, *workloadReference, runtimeInput,
        *runtimeInputReference, *sourceObservations, fabric, config, options,
        artifactStore, workItems[index]);
    if (!result) {
      results[index].error.emplace(result.takeError());
      return;
    }
    results[index].attempt = std::move(*result);
  };

  const std::size_t workerCount =
      std::min<std::size_t>(options.candidateWorkerCount, workItems.size());
  if (workerCount <= 1) {
    for (std::size_t index = 0; index < workItems.size(); ++index)
      execute(parent, index);
  } else {
    // Each worker owns one independently imported parent and therefore one
    // thread-confined MLIRContext. Explicit DSE parallelism must not depend on
    // enabling an MLIRContext's implicit all-host thread pool or concurrently
    // mutate IR interned in another worker's context.
    std::vector<frontend::StructuredProgramCandidate> workerParents;
    workerParents.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
      auto imported = frontend::importStructuredProgram(
          parent.identity(), parent.canonicalBytes());
      if (!imported)
        return imported.takeError();
      workerParents.push_back(std::move(*imported));
    }

    llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
        static_cast<unsigned>(workerCount)));
    std::atomic_size_t nextWorkItem{0};
    for (std::size_t worker = 0; worker < workerCount; ++worker)
      pool.async([&, worker] {
        while (true) {
          const std::size_t index =
              nextWorkItem.fetch_add(1, std::memory_order_relaxed);
          if (index >= workItems.size())
            break;
          execute(workerParents[worker], index);
        }
      });
    pool.wait();
  }

  llvm::Error failures = llvm::Error::success();
  for (WorkResult &result : results)
    if (result.error) {
      failures =
          llvm::joinErrors(std::move(failures), std::move(*result.error));
    }
  if (failures)
    return std::move(failures);

  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  dispositions.reserve(plannedDispositions.size());
  for (const PlannedDisposition &planned : plannedDispositions) {
    if (const auto *rejection =
            std::get_if<StructuredOwnershipCandidateRejectionRecord>(
                &planned.source)) {
      dispositions.push_back({planned.coordinate, *rejection});
      continue;
    }
    const std::size_t workIndex = std::get<std::size_t>(planned.source);
    if (workIndex >= results.size() || !results[workIndex].attempt)
      return invalid("candidate work completed without a disposition");
    const OwnershipAttemptResult &attempt = *results[workIndex].attempt;
    if (const auto *materialized =
            std::get_if<MaterializedOwnershipWorkItem>(&attempt)) {
      candidateReferences.push_back(materialized->candidate);
      dispositions.push_back({planned.coordinate, materialized->candidate});
    } else {
      dispositions.push_back(
          {planned.coordinate,
           std::get<StructuredOwnershipCandidateRejectionRecord>(attempt)});
    }
  }

  auto candidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, candidateReferences);
  if (!candidateSet)
    return candidateSet.takeError();

  std::vector<PromotionEvidence> costEvidence;
  costEvidence.reserve(candidateSet->candidates().size());
  std::optional<evaluation::CaseSubjectRoleRef> costCandidateRole;
  for (const ArtifactRootReference &candidate : candidateSet->candidates()) {
    auto prepared = evaluation::models::prepareStructuredFabricEvaluation(
        candidate, fabric.reference(), *workloadReference,
        *runtimeInputReference, config, artifactStore);
    if (!prepared)
      return prepared.takeError();
    if (costCandidateRole && *costCandidateRole != prepared->candidateRole)
      return invalid(
          "cost model changed its candidate role across obligations");
    costCandidateRole = prepared->candidateRole;
    auto result = evaluation::evaluateRequest(
        prepared->request, prepared->resolution, artifactStore);
    if (!result)
      return result.takeError();
    costEvidence.push_back({std::move(prepared->request), std::move(*result)});
  }
  if (!costCandidateRole)
    return invalid(
        "nonempty candidate set produced no cost Evidence obligations");

  PointMetricTopKSelection benefitFilter = options.selection;
  benefitFilter.k =
      static_cast<std::uint64_t>(candidateSet->candidates().size());
  auto benefitPromoted = promoteMetricTopKAgainstBaseline(
      *candidateSet, *costCandidateRole, *parentReference, costEvidence,
      benefitFilter, artifactStore);
  if (!benefitPromoted)
    return benefitPromoted.takeError();
  if (const auto *incomplete =
          std::get_if<IncompleteSelection>(&*benefitPromoted))
    return StructuredOwnershipExplorationOutcome{*incomplete};
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(*benefitPromoted))
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};
  const auto &benefitSelection = std::get<CompletedSelection>(*benefitPromoted);

  std::vector<ArtifactRootReference> semanticCandidates =
      benefitSelection.selected;
  semanticCandidates.push_back(*parentReference);
  llvm::sort(semanticCandidates, artifactRootReferenceLess);
  semanticCandidates.erase(
      std::unique(semanticCandidates.begin(), semanticCandidates.end()),
      semanticCandidates.end());
  auto semanticCandidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, semanticCandidates);
  if (!semanticCandidateSet)
    return semanticCandidateSet.takeError();

  for (const ArtifactRootReference &candidate :
       semanticCandidateSet->candidates()) {
    if (candidate == *parentReference)
      continue;
    auto structured =
        frontend::importStructuredProgram(candidate, artifactStore);
    if (!structured)
      return structured.takeError();
    auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
        *structured, options.lowering);
    if (!dataflow)
      return dataflow.takeError();
    frontend::MaterializedOwnershipCandidate materialized{
        std::move(*structured), std::move(*dataflow)};
    bool hasDerivation = false;
    for (const StructuredOwnershipCandidateDisposition &disposition :
         dispositions) {
      const auto *derivedCandidate =
          std::get_if<ArtifactRootReference>(&disposition.result);
      if (!derivedCandidate || *derivedCandidate != candidate ||
          !disposition.coordinate.decision)
        continue;
      hasDerivation = true;
      if (llvm::Error error =
              evaluation::models::primeStructuredProgramFunctionalReplay(
                  candidate,
                  {*workloadReference, *runtimeInputReference, parent,
                   disposition.coordinate.scope,
                   *disposition.coordinate.decision, materialized, workload,
                   runtimeInput, *sourceObservations},
                  artifactStore))
        return std::move(error);
    }
    if (!hasDerivation)
      return invalid("beneficial ownership candidate has no derivation");
  }

  std::vector<PromotionEvidence> functionalEvidence;
  functionalEvidence.reserve(semanticCandidateSet->candidates().size());
  std::optional<evaluation::CaseSubjectRoleRef> functionalCandidateRole;
  std::optional<evaluation::FindingRequestOrdinal> functionalMismatchRequest;
  for (const ArtifactRootReference &candidate :
       semanticCandidateSet->candidates()) {
    auto prepared =
        evaluation::models::prepareStructuredProgramFunctionalEvaluation(
            candidate, *workloadReference, *runtimeInputReference, config,
            artifactStore);
    if (!prepared)
      return prepared.takeError();
    if (functionalCandidateRole &&
        *functionalCandidateRole != prepared->candidateRole)
      return invalid(
          "functional model changed its candidate role across obligations");
    functionalCandidateRole = prepared->candidateRole;
    if (functionalMismatchRequest &&
        *functionalMismatchRequest != prepared->functionalMismatchRequest)
      return invalid("functional model changed its finding request across "
                     "obligations");
    functionalMismatchRequest = prepared->functionalMismatchRequest;
    auto result = evaluation::evaluateRequest(
        prepared->request, prepared->resolution, artifactStore);
    if (!result)
      return result.takeError();
    functionalEvidence.push_back(
        {std::move(prepared->request), std::move(*result)});
  }
  if (!functionalCandidateRole || !functionalMismatchRequest)
    return invalid(
        "benefit-filtered candidate set produced no functional Evidence");

  auto semanticallyPromoted = promoteFindingAbsenceAllPassing(
      *semanticCandidateSet, *functionalCandidateRole, functionalEvidence,
      *functionalMismatchRequest, artifactStore);
  if (!semanticallyPromoted)
    return semanticallyPromoted.takeError();
  if (const auto *incomplete =
          std::get_if<IncompleteSelection>(&*semanticallyPromoted)) {
    IncompleteSelection combined = *incomplete;
    mergeEvidenceReferences(combined.retainedEvidence,
                            benefitSelection.satisfiedEvidence);
    return StructuredOwnershipExplorationOutcome{std::move(combined)};
  }
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(
          *semanticallyPromoted))
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};
  const auto &semanticSelection =
      std::get<CompletedSelection>(*semanticallyPromoted);

  auto passingCandidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, semanticSelection.selected);
  if (!passingCandidateSet)
    return passingCandidateSet.takeError();
  std::vector<PromotionEvidence> passingCostEvidence;
  passingCostEvidence.reserve(passingCandidateSet->candidates().size());
  for (PromotionEvidence &record : costEvidence) {
    llvm::ArrayRef<ArtifactRootReference> subjects =
        record.request.subjectBindings().subjects(*costCandidateRole);
    if (subjects.size() != 1)
      return invalid("cost Evidence candidate binding is not singular");
    if (std::binary_search(passingCandidateSet->candidates().begin(),
                           passingCandidateSet->candidates().end(),
                           subjects.front(), artifactRootReferenceLess))
      passingCostEvidence.push_back(std::move(record));
  }
  if (passingCostEvidence.size() != passingCandidateSet->candidates().size())
    return invalid("semantic survivor has no cost Evidence");

  auto promoted = promoteMetricTopKAgainstBaseline(
      *passingCandidateSet, *costCandidateRole, *parentReference,
      passingCostEvidence, options.selection, artifactStore);
  if (!promoted)
    return promoted.takeError();
  if (const auto *incomplete = std::get_if<IncompleteSelection>(&*promoted)) {
    IncompleteSelection combined = *incomplete;
    mergeEvidenceReferences(combined.retainedEvidence,
                            benefitSelection.satisfiedEvidence);
    mergeEvidenceReferences(combined.retainedEvidence,
                            semanticSelection.satisfiedEvidence);
    return StructuredOwnershipExplorationOutcome{std::move(combined)};
  }
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(*promoted))
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};

  const auto &selection = std::get<CompletedSelection>(*promoted);
  std::vector<ArtifactRootReference> satisfiedEvidence =
      benefitSelection.satisfiedEvidence;
  mergeEvidenceReferences(satisfiedEvidence,
                          semanticSelection.satisfiedEvidence);
  mergeEvidenceReferences(satisfiedEvidence, selection.satisfiedEvidence);
  std::vector<SelectedStructuredOwnershipCandidate> selected;
  selected.reserve(selection.selected.size());
  for (const ArtifactRootReference &reference : selection.selected) {
    auto structured =
        frontend::importStructuredProgram(reference, artifactStore);
    if (!structured)
      return structured.takeError();
    auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
        *structured, options.lowering);
    if (!dataflow)
      return dataflow.takeError();
    std::vector<StructuredOwnershipDerivation> derivations;
    for (const StructuredOwnershipCandidateDisposition &disposition :
         dispositions) {
      const auto *candidate =
          std::get_if<ArtifactRootReference>(&disposition.result);
      if (!candidate || *candidate != reference ||
          !disposition.coordinate.decision)
        continue;
      derivations.push_back(StructuredOwnershipDerivation{
          disposition.coordinate.scope, *disposition.coordinate.decision});
    }
    selected.push_back({frontend::MaterializedOwnershipCandidate{
                            std::move(*structured), std::move(*dataflow)},
                        std::move(derivations)});
  }
  return StructuredOwnershipExplorationOutcome{
      CompletedStructuredOwnershipSelection{std::move(selected),
                                            std::move(satisfiedEvidence),
                                            std::move(dispositions)}};
}

} // namespace loom::dse
