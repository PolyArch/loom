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
#include <map>
#include <optional>
#include <queue>
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
  ArtifactRootReference reference;
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
    const ArtifactStore &artifactStore, const OwnershipWorkItem &workItem,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance) {
  auto candidate = frontend::materializeSpatialOwnershipDecision(
      parent, workItem.scope, workItem.decision, fabric, options.lowering,
      sourceProvenance);
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
      workloadReference, runtimeInputReference, workload, runtimeInput, parent,
      sourceObservations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              *reference,
              {candidate->structuredProgram, &candidate->canonicalDataflow,
               candidate->spatialGraphs, candidate->blockActivityLineage},
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
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance) {
  if (options.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (config.dse.structuredOwnership.scopeExpansionLimit == 0)
    return invalid("ownership scope expansion limit must be positive");
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
      *workloadReference,
      *runtimeInputReference,
      workload,
      runtimeInput,
      parent,
      *sourceObservations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              *parentReference,
              evaluation::models::StructuredFabricAnalyticCandidateProjection{
                  parent, nullptr, {}, {}, &*sourceObservations},
              invocation, fabric, config, artifactStore))
    return std::move(error);
  candidateReferences.push_back(*parentReference);

  auto domain = options.protocolCallableRoots.empty()
                    ? frontend::enumerateSpatialOwnershipScopeDomain(parent)
                    : frontend::enumerateSpatialOwnershipScopeDomain(
                          parent, options.protocolCallableRoots);
  if (!domain)
    return domain.takeError();
  std::vector<frontend::StructuredEntityRef> scopeReferences;
  scopeReferences.reserve(domain->size());
  for (const frontend::SpatialOwnershipScopeDomainEntry &entry : *domain) {
    const auto &scope =
        std::holds_alternative<frontend::SpatialOwnershipScope>(entry)
            ? std::get<frontend::SpatialOwnershipScope>(entry)
            : std::get<frontend::RejectedSpatialOwnershipScope>(entry).scope;
    scopeReferences.push_back(scope.selection);
  }
  auto scopeActivity = evaluation::models::projectStructuredScopeActivity(
      parent, *sourceObservations, scopeReferences);
  if (!scopeActivity)
    return scopeActivity.takeError();
  if (scopeActivity->size() != domain->size())
    return invalid("scope activity projection is not total");

  std::vector<bool> activeScopes(domain->size(), false);
  std::vector<std::vector<std::size_t>> childScopes(domain->size());
  std::vector<std::size_t> rootScopes;
  for (std::size_t ordinal = 0; ordinal < domain->size(); ++ordinal) {
    const auto &activity = (*scopeActivity)[ordinal];
    if (activity.scope != scopeReferences[ordinal])
      return invalid("scope activity projection changed canonical order");
    if (activity.dynamicActivations == 0)
      continue;
    activeScopes[ordinal] = true;
  }
  for (std::size_t ordinal = 0; ordinal < domain->size(); ++ordinal) {
    if (!activeScopes[ordinal])
      continue;
    std::optional<std::uint64_t> parent = domain->parentScopeOrdinal(ordinal);
    if (!parent) {
      rootScopes.push_back(ordinal);
      continue;
    }
    if (*parent >= domain->size() || *parent == ordinal)
      return invalid("ownership scope hierarchy is malformed");
    if (!activeScopes[*parent])
      return invalid("active ownership scope has an inactive parent");
    childScopes[*parent].push_back(ordinal);
  }

  auto lessPromising = [&](std::size_t lhs, std::size_t rhs) {
    const auto &left = (*scopeActivity)[lhs];
    const auto &right = (*scopeActivity)[rhs];
    if (left.dynamicLeafExecutions != right.dynamicLeafExecutions)
      return left.dynamicLeafExecutions < right.dynamicLeafExecutions;
    if (left.dynamicActivations != right.dynamicActivations)
      return left.dynamicActivations < right.dynamicActivations;
    return lhs > rhs;
  };
  std::priority_queue<std::size_t, std::vector<std::size_t>,
                      decltype(lessPromising)>
      frontier(lessPromising);
  for (std::size_t root : rootScopes)
    frontier.push(root);

  std::vector<std::size_t> plannedScopeOrdinals;
  plannedScopeOrdinals.reserve(std::min<std::size_t>(
      config.dse.structuredOwnership.scopeExpansionLimit, domain->size()));
  while (!frontier.empty() &&
         plannedScopeOrdinals.size() <
             config.dse.structuredOwnership.scopeExpansionLimit) {
    const std::size_t ordinal = frontier.top();
    frontier.pop();
    plannedScopeOrdinals.push_back(ordinal);
    for (std::size_t child : childScopes[ordinal])
      frontier.push(child);
  }

  std::vector<OwnershipWorkItem> workItems;
  struct PlannedDisposition final {
    StructuredOwnershipCandidateCoordinate coordinate;
    std::variant<std::size_t, StructuredOwnershipCandidateRejectionRecord>
        source;
  };
  std::vector<PlannedDisposition> plannedDispositions;
  for (std::size_t domainOrdinal : plannedScopeOrdinals) {
    const frontend::SpatialOwnershipScopeDomainEntry &entry =
        (*domain)[domainOrdinal];
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
        artifactStore, workItems[index], sourceProvenance);
    if (!result) {
      results[index].error.emplace(result.takeError());
      return;
    }
    results[index].attempt = std::move(*result);
  };

  const std::size_t workerCount =
      std::min<std::size_t>(options.candidateWorkerCount, workItems.size());
  std::vector<frontend::StructuredProgramCandidate> workerParents;
  if (workerCount <= 1) {
    for (std::size_t index = 0; index < workItems.size(); ++index)
      execute(parent, index);
  } else {
    // Each worker owns one independently imported parent and therefore one
    // thread-confined MLIRContext. Explicit DSE parallelism must not depend on
    // enabling an MLIRContext's implicit all-host thread pool or concurrently
    // mutate IR interned in another worker's context.
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
    OwnershipAttemptResult &attempt = *results[workIndex].attempt;
    if (auto *materialized =
            std::get_if<MaterializedOwnershipWorkItem>(&attempt)) {
      candidateReferences.push_back(materialized->reference);
      dispositions.push_back({planned.coordinate, materialized->reference});
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

  auto analyticInvocation =
      evaluation::models::prepareStructuredFabricAnalyticInvocation(
          candidateSet->candidates(), fabric.reference(), *workloadReference,
          *runtimeInputReference, artifactStore);
  if (!analyticInvocation)
    return analyticInvocation.takeError();

  std::vector<PromotionEvidence> costEvidence;
  costEvidence.reserve(candidateSet->candidates().size());
  std::optional<evaluation::CaseSubjectRoleRef> costCandidateRole;
  for (const ArtifactRootReference &candidate : candidateSet->candidates()) {
    auto prepared = evaluation::models::prepareStructuredFabricEvaluation(
        candidate, *analyticInvocation, config, artifactStore);
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

  std::vector<ArtifactRootReference> costEvidenceReferences;
  std::vector<ArtifactRootReference> semanticCandidates;
  std::vector<PromotionEvidence> functionalEvidence;
  functionalEvidence.reserve(candidateSet->candidates().size());
  std::optional<evaluation::CaseSubjectRoleRef> functionalCandidateRole;
  std::optional<evaluation::FindingRequestOrdinal> functionalMismatchRequest;
  std::map<ArtifactRootReference, frontend::MaterializedOwnershipCandidate,
           decltype(&artifactRootReferenceLess)>
      functionalCandidates(&artifactRootReferenceLess);

  auto acquireFunctionalEvidence =
      [&](const ArtifactRootReference &candidate) -> llvm::Error {
    if (llvm::is_contained(semanticCandidates, candidate))
      return llvm::Error::success();
    if (candidate != *parentReference) {
      auto cached = functionalCandidates.find(candidate);
      if (cached == functionalCandidates.end()) {
        const StructuredOwnershipCandidateDisposition *derivation = nullptr;
        for (const StructuredOwnershipCandidateDisposition &disposition :
             dispositions) {
          const auto *derivedCandidate =
              std::get_if<ArtifactRootReference>(&disposition.result);
          if (derivedCandidate && *derivedCandidate == candidate &&
              disposition.coordinate.decision) {
            derivation = &disposition;
            break;
          }
        }
        if (!derivation)
          return invalid("cost-ranked ownership candidate has no derivation");
        auto rematerialized = frontend::materializeSpatialOwnershipDecision(
            parent, derivation->coordinate.scope,
            *derivation->coordinate.decision, fabric, options.lowering,
            sourceProvenance);
        if (!rematerialized)
          return rematerialized.takeError();
        if (rematerialized->structuredProgram.identity() != candidate.artifact)
          return invalid("rematerialized ownership candidate changed identity");
        auto stored = artifactStore.get(candidate);
        if (!stored)
          return stored.takeError();
        if (!stored->bytes().equals(
                rematerialized->structuredProgram.canonicalBytes().bytes()))
          return invalid("rematerialized ownership candidate changed bytes");
        cached = functionalCandidates
                     .try_emplace(candidate, std::move(*rematerialized))
                     .first;
      }
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
                     *disposition.coordinate.decision, cached->second, workload,
                     runtimeInput, *sourceObservations,
                     options.functionalReplayLimits},
                    artifactStore))
          return error;
      }
      if (!hasDerivation)
        return invalid("cost-ranked ownership candidate has no derivation");
    }

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
    semanticCandidates.push_back(candidate);
    llvm::sort(semanticCandidates, artifactRootReferenceLess);
    return llvm::Error::success();
  };

  if (llvm::Error error = acquireFunctionalEvidence(*parentReference))
    return std::move(error);

  const std::uint64_t acceleratorCandidateCount =
      static_cast<std::uint64_t>(candidateSet->candidates().size() - 1);
  const bool semanticConformance =
      options.selectionMode ==
      StructuredOwnershipSelectionMode::SemanticConformance;
  if (semanticConformance && acceleratorCandidateCount == 0)
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};

  std::optional<CandidateSet> acceleratorCandidateSet;
  std::vector<PromotionEvidence> acceleratorCostEvidence;
  if (semanticConformance) {
    std::vector<ArtifactRootReference> acceleratorCandidates;
    acceleratorCandidates.reserve(acceleratorCandidateCount);
    for (const ArtifactRootReference &candidate : candidateSet->candidates())
      if (candidate != *parentReference)
        acceleratorCandidates.push_back(candidate);
    auto set = CandidateSet::get(frontend::structuredProgramArtifactSchema,
                                 acceleratorCandidates);
    if (!set)
      return set.takeError();
    acceleratorCandidateSet.emplace(std::move(*set));

    acceleratorCostEvidence.reserve(acceleratorCandidateCount);
    for (PromotionEvidence &record : costEvidence) {
      llvm::ArrayRef<ArtifactRootReference> subjects =
          record.request.subjectBindings().subjects(*costCandidateRole);
      if (subjects.size() != 1)
        return invalid("cost Evidence candidate binding is not singular");
      if (subjects.front() != *parentReference)
        acceleratorCostEvidence.push_back(std::move(record));
    }
    if (acceleratorCostEvidence.size() != acceleratorCandidateCount)
      return invalid("accelerator candidate has no cost Evidence");
  }

  const CandidateSet &rankingCandidateSet =
      semanticConformance ? *acceleratorCandidateSet : *candidateSet;
  llvm::ArrayRef<PromotionEvidence> rankingCostEvidence =
      semanticConformance ? llvm::ArrayRef(acceleratorCostEvidence)
                          : llvm::ArrayRef(costEvidence);
  auto promoteCostCandidates = [&](const CandidateSet &set,
                                   llvm::ArrayRef<PromotionEvidence> evidence,
                                   const PointMetricTopKSelection &selection)
      -> llvm::Expected<PromotionOutcome> {
    if (semanticConformance)
      return promoteMetricTopK(set, *costCandidateRole, evidence, selection,
                               artifactStore);
    return promoteMetricTopKAgainstBaseline(set, *costCandidateRole,
                                            *parentReference, evidence,
                                            selection, artifactStore);
  };

  PointMetricTopKSelection rankingFilter = options.selection;
  if (acceleratorCandidateCount != 0)
    rankingFilter.k = std::min(rankingFilter.k, acceleratorCandidateCount);

  std::optional<CompletedSelection> semanticSelection;
  while (true) {
    auto ranked = promoteCostCandidates(rankingCandidateSet,
                                        rankingCostEvidence, rankingFilter);
    if (!ranked)
      return ranked.takeError();
    if (const auto *incomplete = std::get_if<IncompleteSelection>(&*ranked))
      return StructuredOwnershipExplorationOutcome{*incomplete};
    if (std::holds_alternative<CompletedNoFeasibleCandidate>(*ranked))
      return StructuredOwnershipExplorationOutcome{
          CompletedNoFeasibleCandidate{}};
    const auto &rankedSelection = std::get<CompletedSelection>(*ranked);
    mergeEvidenceReferences(costEvidenceReferences,
                            rankedSelection.satisfiedEvidence);

    for (const ArtifactRootReference &candidate : rankedSelection.selected)
      if (llvm::Error error = acquireFunctionalEvidence(candidate))
        return std::move(error);

    auto semanticCandidateSet = CandidateSet::get(
        frontend::structuredProgramArtifactSchema, semanticCandidates);
    if (!semanticCandidateSet)
      return semanticCandidateSet.takeError();
    if (!functionalCandidateRole || !functionalMismatchRequest)
      return invalid(
          "cost-ranked candidate set produced no functional Evidence");

    QualityGateClause functionalClause;
    functionalClause.atoms.push_back(FindingGate{0, *functionalMismatchRequest,
                                                 RequiredFindingState::Absent});
    auto functionalGate = QualityGatePolicy::get({std::move(functionalClause)});
    if (!functionalGate)
      return functionalGate.takeError();
    auto semanticallyPromoted = promoteCandidates(
        *semanticCandidateSet, *functionalCandidateRole, functionalEvidence,
        *functionalGate, {}, AllPassingSelection{}, nullptr, artifactStore);
    if (!semanticallyPromoted)
      return semanticallyPromoted.takeError();
    if (const auto *incomplete =
            std::get_if<IncompleteSelection>(&*semanticallyPromoted)) {
      IncompleteSelection combined = *incomplete;
      mergeEvidenceReferences(combined.retainedEvidence,
                              costEvidenceReferences);
      return StructuredOwnershipExplorationOutcome{std::move(combined)};
    }
    if (std::holds_alternative<CompletedNoFeasibleCandidate>(
            *semanticallyPromoted))
      return StructuredOwnershipExplorationOutcome{
          CompletedNoFeasibleCandidate{}};
    semanticSelection = std::get<CompletedSelection>(*semanticallyPromoted);

    const std::uint64_t passingAcceleratorCount = static_cast<std::uint64_t>(
        llvm::count_if(semanticSelection->selected,
                       [&](const ArtifactRootReference &candidate) {
                         return candidate != *parentReference;
                       }));
    const bool exhaustedRankedCandidates =
        rankedSelection.selected.size() < rankingFilter.k ||
        (!semanticConformance &&
         llvm::is_contained(rankedSelection.selected, *parentReference)) ||
        rankingFilter.k >= acceleratorCandidateCount;
    if (passingAcceleratorCount >= options.selection.k ||
        exhaustedRankedCandidates)
      break;

    const std::uint64_t missing = options.selection.k - passingAcceleratorCount;
    rankingFilter.k = missing >= acceleratorCandidateCount - rankingFilter.k
                          ? acceleratorCandidateCount
                          : rankingFilter.k + missing;
  }
  if (!semanticSelection)
    return invalid("ownership promotion produced no semantic selection");

  std::vector<ArtifactRootReference> passingCandidates =
      semanticSelection->selected;
  if (semanticConformance)
    passingCandidates.erase(std::remove(passingCandidates.begin(),
                                        passingCandidates.end(),
                                        *parentReference),
                            passingCandidates.end());
  if (passingCandidates.empty())
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};
  auto passingCandidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, passingCandidates);
  if (!passingCandidateSet)
    return passingCandidateSet.takeError();
  std::vector<PromotionEvidence> passingCostEvidence;
  passingCostEvidence.reserve(passingCandidateSet->candidates().size());
  std::vector<PromotionEvidence> &costEvidencePool =
      semanticConformance ? acceleratorCostEvidence : costEvidence;
  for (PromotionEvidence &record : costEvidencePool) {
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

  auto promoted = promoteCostCandidates(*passingCandidateSet,
                                        passingCostEvidence, options.selection);
  if (!promoted)
    return promoted.takeError();
  if (const auto *incomplete = std::get_if<IncompleteSelection>(&*promoted)) {
    IncompleteSelection combined = *incomplete;
    mergeEvidenceReferences(combined.retainedEvidence, costEvidenceReferences);
    mergeEvidenceReferences(combined.retainedEvidence,
                            semanticSelection->satisfiedEvidence);
    return StructuredOwnershipExplorationOutcome{std::move(combined)};
  }
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(*promoted))
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};

  const auto &selection = std::get<CompletedSelection>(*promoted);
  std::vector<ArtifactRootReference> satisfiedEvidence = costEvidenceReferences;
  mergeEvidenceReferences(satisfiedEvidence,
                          semanticSelection->satisfiedEvidence);
  mergeEvidenceReferences(satisfiedEvidence, selection.satisfiedEvidence);
  std::vector<SelectedStructuredOwnershipCandidate> selected;
  selected.reserve(selection.selected.size());
  for (const ArtifactRootReference &reference : selection.selected) {
    std::optional<frontend::MaterializedOwnershipCandidate> materialized;
    if (reference == *parentReference) {
      auto structured =
          frontend::importStructuredProgram(reference, artifactStore);
      if (!structured)
        return structured.takeError();
      auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
          *structured, options.lowering);
      if (!dataflow)
        return dataflow.takeError();
      materialized.emplace(frontend::MaterializedOwnershipCandidate{
          std::move(*structured), std::move(*dataflow), {}, {}, {}});
    } else {
      auto cached = functionalCandidates.find(reference);
      if (cached == functionalCandidates.end())
        return invalid("selected ownership candidate has no functional "
                       "invocation-local projection");
      // The retained canonical module borrows a candidate-worker context.
      // Crossing the invocation boundary therefore requires one owned strict
      // import for each selected result, not one import per explored candidate.
      auto dataflow = dataflow::importCanonicalDataflow(
          cached->second.canonicalDataflow.identity(),
          cached->second.canonicalDataflow.canonicalBytes());
      if (!dataflow)
        return dataflow.takeError();
      materialized.emplace(frontend::MaterializedOwnershipCandidate{
          std::move(cached->second.structuredProgram), std::move(*dataflow),
          std::move(cached->second.spatialGraphs),
          std::move(cached->second.blockActivityLineage),
          std::move(cached->second.sourceProvenance)});
    }
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
    std::optional<sim::SourceBackedDfgValidationResult> functionalReplay;
    if (reference != *parentReference) {
      auto replay =
          evaluation::models::getPrimedStructuredProgramFunctionalReplay(
              reference, *workloadReference, *runtimeInputReference);
      if (!replay)
        return replay.takeError();
      if (replay->status != sim::SourceBackedDfgValidationStatus::Equivalent)
        return invalid("selected accelerator candidate lacks equivalent "
                       "functional replay");
      functionalReplay.emplace(std::move(*replay));
    }
    selected.push_back({std::move(*materialized), std::move(derivations),
                        std::move(functionalReplay)});
  }
  return StructuredOwnershipExplorationOutcome{
      CompletedStructuredOwnershipSelection{std::move(selected),
                                            std::move(satisfiedEvidence),
                                            std::move(dispositions)}};
}

} // namespace loom::dse
