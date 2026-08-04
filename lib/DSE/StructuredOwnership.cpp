#include "DSE/StructuredOwnership.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
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
  frontend::MaterializedStructuredOwnershipCandidate candidate;
};

using OwnershipAttemptResult =
    std::variant<MaterializedOwnershipWorkItem,
                 StructuredOwnershipCandidateRejectionRecord>;

struct OwnershipGenerationState final {
  CompletedStructuredOwnershipGeneration completed;
  ArtifactRootReference parentReference;
  ArtifactRootReference workloadReference;
  ArtifactRootReference runtimeInputReference;
  std::vector<detail::StructuredOwnershipCandidateState> candidates;
};

llvm::Expected<OwnershipAttemptResult> materializeOwnershipWorkItem(
    const frontend::StructuredProgramCandidate &parent,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore, const OwnershipWorkItem &workItem,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance) {
  auto candidate = frontend::materializeStructuredSpatialOwnershipDecision(
      parent, workItem.scope, workItem.decision, sourceProvenance);
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
  return OwnershipAttemptResult{MaterializedOwnershipWorkItem{
      std::move(*reference), std::move(*candidate)}};
}

} // namespace

static llvm::Expected<OwnershipGenerationState>
generateStructuredOwnershipCandidatesImpl(
    const frontend::StructuredProgramCandidate &parent,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance,
    const ResolvedConfig *analyticConfig,
    evaluation::models::StructuredEvaluationInvocationCache *evaluationCache,
    const detail::StructuredOwnershipPreparedSource *preparedSource) {
  if (options.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (options.scopeExpansionLimit == 0)
    return invalid("ownership scope expansion limit must be positive");

  std::vector<ArtifactRootReference> candidateReferences;
  std::optional<ArtifactRootReference> ownedParentReference;
  std::optional<ArtifactRootReference> ownedWorkloadReference;
  std::optional<ArtifactRootReference> ownedRuntimeInputReference;
  std::optional<sim::NativeStructuredProgramObservations>
      ownedSourceObservations;
  const ArtifactRootReference *parentReference = nullptr;
  const ArtifactRootReference *workloadReference = nullptr;
  const ArtifactRootReference *runtimeInputReference = nullptr;
  const sim::NativeStructuredProgramObservations *sourceObservations = nullptr;
  if (preparedSource) {
    parentReference = &preparedSource->sourceReference;
    workloadReference = &preparedSource->workloadReference;
    runtimeInputReference = &preparedSource->runtimeInputReference;
    sourceObservations = &preparedSource->observations;
  } else {
    auto publishedParent =
        frontend::publishStructuredProgram(parent, artifactStore);
    if (!publishedParent)
      return publishedParent.takeError();
    auto publishedWorkload =
        sim::publishSimulationWorkload(workload, artifactStore);
    if (!publishedWorkload)
      return publishedWorkload.takeError();
    auto publishedRuntimeInput =
        sim::publishSimulationRuntimeInput(runtimeInput, artifactStore);
    if (!publishedRuntimeInput)
      return publishedRuntimeInput.takeError();
    auto observations =
        sim::executeNativeStructuredProgram(parent, workload, runtimeInput);
    if (!observations)
      return observations.takeError();
    ownedParentReference.emplace(std::move(*publishedParent));
    ownedWorkloadReference.emplace(std::move(*publishedWorkload));
    ownedRuntimeInputReference.emplace(std::move(*publishedRuntimeInput));
    ownedSourceObservations.emplace(std::move(*observations));
    parentReference = &*ownedParentReference;
    workloadReference = &*ownedWorkloadReference;
    runtimeInputReference = &*ownedRuntimeInputReference;
    sourceObservations = &*ownedSourceObservations;
  }
  if (analyticConfig && !preparedSource) {
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
                invocation, fabric, *analyticConfig, artifactStore))
      return std::move(error);
  }
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
  plannedScopeOrdinals.reserve(
      std::min<std::size_t>(options.scopeExpansionLimit, domain->size()));
  while (!frontier.empty() &&
         plannedScopeOrdinals.size() < options.scopeExpansionLimit) {
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
    auto result =
        materializeOwnershipWorkItem(workerParent, options, artifactStore,
                                     workItems[index], sourceProvenance);
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
        std::optional<
            evaluation::models::StructuredEvaluationInvocationCacheScope>
            workerEvaluationCacheScope;
        if (evaluationCache)
          workerEvaluationCacheScope.emplace(*evaluationCache);
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
  std::vector<detail::StructuredOwnershipCandidateState> materializedCandidates;
  materializedCandidates.reserve(workItems.size());
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
      materializedCandidates.push_back(
          {materialized->reference, std::move(materialized->candidate)});
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

  return OwnershipGenerationState{
      CompletedStructuredOwnershipGeneration{std::move(*candidateSet),
                                             std::move(dispositions)},
      *parentReference, *workloadReference, *runtimeInputReference,
      std::move(materializedCandidates)};
}

llvm::Expected<CompletedStructuredOwnershipGeneration>
generateStructuredOwnershipCandidates(
    const frontend::StructuredProgramCandidate &parent,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance) {
  StructuredOwnershipGenerationOptions effectiveOptions = options;
  const ResolvedConfig *analyticConfig = nullptr;
  evaluation::models::StructuredEvaluationInvocationCache *evaluationCache =
      nullptr;
  std::optional<detail::StructuredOwnershipPreparedSource> preparedSource;
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  if (invocation) {
    if (llvm::Error error =
            detail::StructuredOwnershipInvocationAccess::prepareGeneration(
                *invocation, parent, workload, runtimeInput, fabric,
                artifactStore, effectiveOptions))
      return std::move(error);
    analyticConfig =
        &detail::StructuredOwnershipInvocationAccess::config(*invocation);
    evaluationCache =
        &detail::StructuredOwnershipInvocationAccess::evaluationCache(
            *invocation);
    sourceProvenance =
        detail::StructuredOwnershipInvocationAccess::sourceProvenance(
            *invocation);
    auto source = detail::StructuredOwnershipInvocationAccess::preparedSource(
        *invocation);
    if (!source)
      return source.takeError();
    preparedSource.emplace(*source);
  }
  auto generated = generateStructuredOwnershipCandidatesImpl(
      parent, workload, runtimeInput, fabric, effectiveOptions, artifactStore,
      sourceProvenance, analyticConfig, evaluationCache,
      preparedSource ? &*preparedSource : nullptr);
  if (!generated)
    return generated.takeError();
  if (invocation)
    if (llvm::Error error =
            detail::StructuredOwnershipInvocationAccess::recordGeneration(
                *invocation, generated->parentReference,
                generated->workloadReference, generated->runtimeInputReference,
                generated->completed.dispositions,
                std::move(generated->candidates), artifactStore))
      return std::move(error);
  return std::move(generated->completed);
}

} // namespace loom::dse
