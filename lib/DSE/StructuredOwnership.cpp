#include "DSE/StructuredOwnership.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
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
#include <map>
#include <numeric>
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
  bool sourceIndexNarrowingRejected = false;
};

enum class AddressProjectionClass : std::uint8_t {
  None,
  RootRelativeI32,
  RootRelativeI64,
  PointerAddressed,
};

AddressProjectionClass
addressClass(const frontend::SpatialOwnershipDecisionPoint &decision) {
  if (!decision.addressProjection)
    return AddressProjectionClass::None;
  if (std::holds_alternative<frontend::PointerAddressedAddressProjection>(
          *decision.addressProjection))
    return AddressProjectionClass::PointerAddressed;
  const unsigned width = std::get<frontend::RootRelativeAddressProjection>(
                             *decision.addressProjection)
                             .canonicalIndexWidth;
  return width == 32 ? AddressProjectionClass::RootRelativeI32
                     : AddressProjectionClass::RootRelativeI64;
}

// Both admission boundaries use this protocol. Eligible work indices retain
// canonical scope/decision order, regardless of worker completion order.
std::vector<std::size_t>
selectOwnershipWorkItems(llvm::ArrayRef<OwnershipWorkItem> workItems,
                         llvm::ArrayRef<std::size_t> eligible,
                         std::size_t limit,
                         StructuredOwnershipGenerationIntent intent) {
  if (limit >= eligible.size())
    return {eligible.begin(), eligible.end()};
  std::vector<std::size_t> selectedIndices;
  selectedIndices.reserve(limit);
  std::vector<bool> selected(workItems.size(), false);
  const auto select = [&](std::size_t index) {
    selected[index] = true;
    selectedIndices.push_back(index);
  };
  const auto selectFirst = [&](auto predicate) {
    if (selectedIndices.size() == limit)
      return;
    for (std::size_t index : eligible)
      if (!selected[index] && predicate(workItems[index])) {
        select(index);
        return;
      }
  };
  const bool requireLogical =
      intent == StructuredOwnershipGenerationIntent::RequireLogicalThreadDomain;
  const auto isLogical = [](const OwnershipWorkItem &item) {
    return item.decision.forallOwnershipShape ==
           frontend::ForallOwnershipShape::LogicalThreadDomain;
  };
  if (requireLogical)
    selectFirst([&](const OwnershipWorkItem &item) {
      return isLogical(item) && addressClass(item.decision) ==
                                    AddressProjectionClass::PointerAddressed;
    });
  for (AddressProjectionClass category :
       {AddressProjectionClass::None, AddressProjectionClass::RootRelativeI32,
        AddressProjectionClass::RootRelativeI64,
        AddressProjectionClass::PointerAddressed})
    selectFirst([&](const OwnershipWorkItem &item) {
      return addressClass(item.decision) == category &&
             (!requireLogical || isLogical(item));
    });

  // A scope receives one remaining decision per round, so its address and
  // transformation variants cannot consume every slot ahead of another scope.
  std::vector<std::vector<std::size_t>> scopeDecisions;
  std::optional<std::uint64_t> previousScope;
  for (std::size_t index : eligible) {
    if (selected[index])
      continue;
    const std::uint64_t scope = workItems[index].scope.selection.ordinal;
    if (!previousScope || *previousScope != scope) {
      scopeDecisions.emplace_back();
      previousScope = scope;
    }
    scopeDecisions.back().push_back(index);
  }
  std::queue<std::pair<std::size_t, std::size_t>> nextDecisions;
  for (auto [scope, decisions] : llvm::enumerate(scopeDecisions)) {
    std::stable_partition(
        decisions.begin(), decisions.end(), [&](std::size_t index) {
          return !workItems[index].sourceIndexNarrowingRejected;
        });
    nextDecisions.emplace(scope, 0);
  }
  while (!nextDecisions.empty() && selectedIndices.size() != limit) {
    auto [scope, ordinal] = nextDecisions.front();
    nextDecisions.pop();
    select(scopeDecisions[scope][ordinal]);
    if (++ordinal < scopeDecisions[scope].size())
      nextDecisions.emplace(scope, ordinal);
  }
  return selectedIndices;
}

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
              error.kind(), error.message(), error.memoryContract()});
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

llvm::StringRef toString(StructuredOwnershipSelectionMode value) {
  switch (value) {
  case StructuredOwnershipSelectionMode::BenefitQualified:
    return "benefit_qualified";
  case StructuredOwnershipSelectionMode::SemanticConformance:
    return "semantic_conformance";
  }
  llvm_unreachable("unknown structured ownership selection mode");
}

static llvm::Expected<OwnershipGenerationState>
generateStructuredOwnershipCandidatesImpl(
    const frontend::StructuredProgramCandidate &parent,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance,
    evaluation::models::StructuredEvaluationInvocationCache *evaluationCache,
    const detail::StructuredOwnershipPreparedSource *preparedSource) {
  if (options.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (options.scopeExpansionLimit == 0)
    return invalid("ownership scope expansion limit must be positive");
  if (options.maximumMaterializationAttempts &&
      *options.maximumMaterializationAttempts == 0)
    return invalid("ownership materialization limit must be positive");
  if (options.maximumPublishedCandidates &&
      *options.maximumPublishedCandidates == 0)
    return invalid("ownership publication limit must be positive");

  std::vector<ArtifactRootReference> candidateReferences;
  std::optional<ArtifactRootReference> ownedParentReference;
  std::optional<ArtifactRootReference> ownedWorkloadReference;
  std::optional<ArtifactRootReference> ownedRuntimeInputReference;
  std::optional<sim::NativeStructuredProgramObservations>
      ownedParentObservations;
  const ArtifactRootReference *parentReference = nullptr;
  const ArtifactRootReference *workloadReference = nullptr;
  const ArtifactRootReference *runtimeInputReference = nullptr;
  const sim::NativeStructuredProgramObservations *parentObservations = nullptr;
  if (preparedSource) {
    parentReference = &preparedSource->generationParentReference;
    workloadReference = &preparedSource->workloadReference;
    runtimeInputReference = &preparedSource->runtimeInputReference;
    parentObservations = &preparedSource->generationParentObservations;
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
    auto sourceObservations = sim::executeNativeStructuredProgram(
        sourceProgram, workload, runtimeInput);
    if (!sourceObservations)
      return sourceObservations.takeError();
    if (parent.identity() == sourceProgram.identity()) {
      ownedParentObservations.emplace(std::move(*sourceObservations));
    } else {
      auto observations = sim::executeProfiledSelectedStructuredProgram(
          parent, sourceProgram, workload, runtimeInput);
      if (!observations)
        return observations.takeError();
      ownedParentObservations.emplace(std::move(*observations));
    }
    ownedParentReference.emplace(std::move(*publishedParent));
    ownedWorkloadReference.emplace(std::move(*publishedWorkload));
    ownedRuntimeInputReference.emplace(std::move(*publishedRuntimeInput));
    parentReference = &*ownedParentReference;
    workloadReference = &*ownedWorkloadReference;
    runtimeInputReference = &*ownedRuntimeInputReference;
    parentObservations = &*ownedParentObservations;
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
      parent, *parentObservations, scopeReferences);
  if (!scopeActivity)
    return scopeActivity.takeError();
  if (scopeActivity->size() != domain->size())
    return invalid("scope activity projection is not total");

  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        llvm::json::Array scopes;
        for (auto [ordinal, activity] : llvm::enumerate(*scopeActivity)) {
          llvm::json::Object scope;
          scope["domain_ordinal"] = ordinal;
          scope["scope_ordinal"] = activity.scope.ordinal;
          scope["dynamic_activations"] = activity.dynamicActivations;
          scope["dynamic_leaf_executions"] = activity.dynamicLeafExecutions;
          if (auto parent = domain->parentScopeOrdinal(ordinal))
            scope["parent_domain_ordinal"] = *parent;
          if (const auto *rejected =
                  std::get_if<frontend::RejectedSpatialOwnershipScope>(
                      &(*domain)[ordinal]))
            scope["rejection"] = rejected->message;
          scopes.push_back(std::move(scope));
        }
        fields["context_kind"] = "structured_scope_activity";
        fields["structured_program"] =
            formatArtifactIdentityHex(parent.identity());
        fields["scope_count"] = scopeActivity->size();
        fields["scopes"] = std::move(scopes);
      });

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
               rejected->message, std::nullopt}});
      continue;
    }
    const auto &scope = std::get<frontend::SpatialOwnershipScope>(entry);
    auto decisions = frontend::enumerateSpatialOwnershipDecisionDomain(
        parent, scope.selection);
    if (!decisions)
      return decisions.takeError();
    for (const frontend::SpatialOwnershipDecisionPoint &decision : *decisions) {
      const bool logicalThreadDomain =
          decision.forallOwnershipShape ==
          frontend::ForallOwnershipShape::LogicalThreadDomain;
      if ((options.generationIntent ==
               StructuredOwnershipGenerationIntent::
                   RequireLogicalThreadDomain &&
           !logicalThreadDomain) ||
          (options.generationIntent ==
               StructuredOwnershipGenerationIntent::
                   ForbidLogicalThreadDomain &&
           logicalThreadDomain))
        continue;
      const std::size_t workIndex = workItems.size();
      workItems.push_back({scope, decision});
      plannedDispositions.push_back(
          {StructuredOwnershipCandidateCoordinate{scope, decision}, workIndex});
    }
  }
  const std::uint64_t plannedDecisionAttemptCount = workItems.size();
  // A failed source proof is only an admission hint. The private clone can
  // change the proof through inlining or specialization before materialization.
  const bool admissionMayTruncate =
      (options.maximumMaterializationAttempts &&
       workItems.size() > *options.maximumMaterializationAttempts) ||
      (options.maximumPublishedCandidates &&
       workItems.size() > *options.maximumPublishedCandidates);
  if (admissionMayTruncate) {
    auto sourceView = parent.view();
    if (!sourceView)
      return sourceView.takeError();
    std::map<std::pair<std::uint64_t, unsigned>, bool> sourceProofs;
    for (OwnershipWorkItem &item : workItems) {
      auto width = item.decision.rootRelativeIndexWidth();
      if (!width)
        continue;
      const auto key = std::make_pair(item.scope.selection.ordinal, *width);
      auto known = sourceProofs.find(key);
      if (known == sourceProofs.end()) {
        auto rejection =
            frontend::explainSpatialOwnershipSourceIndexNarrowingRejection(
                *sourceView, item.scope, *width);
        if (!rejection)
          return rejection.takeError();
        known = sourceProofs.emplace(key, rejection->has_value()).first;
      }
      item.sourceIndexNarrowingRejected = known->second;
    }
  }
  bool candidateDomainTruncated = false;
  if (options.maximumMaterializationAttempts &&
      workItems.size() > *options.maximumMaterializationAttempts) {
    const std::size_t retained =
        static_cast<std::size_t>(*options.maximumMaterializationAttempts);
    if (retained == 0)
      return invalid("ownership materialization limit cannot be zero");

    std::vector<std::size_t> eligible(workItems.size());
    std::iota(eligible.begin(), eligible.end(), 0);
    auto selectedIndices = selectOwnershipWorkItems(
        workItems, eligible, retained, options.generationIntent);
    llvm::sort(selectedIndices);

    std::vector<std::size_t> remap(workItems.size(),
                                   std::numeric_limits<std::size_t>::max());
    std::vector<OwnershipWorkItem> retainedWorkItems;
    retainedWorkItems.reserve(selectedIndices.size());
    for (std::size_t newIndex = 0; newIndex != selectedIndices.size();
         ++newIndex) {
      const std::size_t oldIndex = selectedIndices[newIndex];
      remap[oldIndex] = newIndex;
      retainedWorkItems.push_back(std::move(workItems[oldIndex]));
    }
    workItems = std::move(retainedWorkItems);
    std::vector<PlannedDisposition> retainedDispositions;
    retainedDispositions.reserve(plannedDispositions.size());
    for (PlannedDisposition &disposition : plannedDispositions) {
      auto *index = std::get_if<std::size_t>(&disposition.source);
      if (!index) {
        retainedDispositions.push_back(std::move(disposition));
        continue;
      }
      if (*index >= remap.size() ||
          remap[*index] == std::numeric_limits<std::size_t>::max())
        continue;
      *index = remap[*index];
      retainedDispositions.push_back(std::move(disposition));
    }
    plannedDispositions = std::move(retainedDispositions);
    candidateDomainTruncated = true;
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

  std::vector<std::size_t> successfulRepresentatives;
  successfulRepresentatives.reserve(results.size());
  for (std::size_t index = 0; index != results.size(); ++index) {
    const auto *materialized =
        results[index].attempt
            ? std::get_if<MaterializedOwnershipWorkItem>(
                  &*results[index].attempt)
            : nullptr;
    if (!materialized)
      continue;
    const bool duplicate = llvm::any_of(
        successfulRepresentatives, [&](std::size_t representative) {
          const auto *existing = std::get_if<MaterializedOwnershipWorkItem>(
              &*results[representative].attempt);
          return existing && existing->reference == materialized->reference;
        });
    if (!duplicate)
      successfulRepresentatives.push_back(index);
  }

  std::vector<bool> publishedWorkItem(workItems.size(), false);
  const std::size_t publicationLimit =
      options.maximumPublishedCandidates
          ? static_cast<std::size_t>(
                std::min<std::uint64_t>(*options.maximumPublishedCandidates,
                                        successfulRepresentatives.size()))
          : successfulRepresentatives.size();
  auto publishedRepresentatives =
      selectOwnershipWorkItems(workItems, successfulRepresentatives,
                               publicationLimit, options.generationIntent);
  for (std::size_t representative : publishedRepresentatives) {
    const auto *selected = std::get_if<MaterializedOwnershipWorkItem>(
        &*results[representative].attempt);
    for (std::size_t index = 0; index != results.size(); ++index) {
      const auto *candidate =
          results[index].attempt
              ? std::get_if<MaterializedOwnershipWorkItem>(
                    &*results[index].attempt)
              : nullptr;
      if (candidate && candidate->reference == selected->reference)
        publishedWorkItem[index] = true;
    }
  }
  if (publishedRepresentatives.size() < successfulRepresentatives.size())
    candidateDomainTruncated = true;

  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "structured_ownership_work_funnel";
        fields["structured_program"] =
            formatArtifactIdentityHex(parent.identity());
        fields["planned_decision_attempts"] = plannedDecisionAttemptCount;
        fields["consumed_decision_attempts"] = workItems.size();
        fields["successful_candidates"] = successfulRepresentatives.size();
        fields["published_candidates"] = publishedRepresentatives.size();
        fields["candidate_domain_truncated"] = candidateDomainTruncated;
        llvm::json::Array scopes;
        for (std::size_t ordinal : plannedScopeOrdinals)
          scopes.push_back(ordinal);
        fields["planned_scope_ordinals"] = std::move(scopes);
        llvm::json::Array attempts;
        for (std::size_t index = 0; index != results.size(); ++index) {
          llvm::json::Object attempt;
          attempt["scope_ordinal"] = workItems[index].scope.selection.ordinal;
          attempt["address_projection_ordinal"] =
              static_cast<unsigned>(addressClass(workItems[index].decision));
          attempt["source_index_narrowing_hint_failed"] =
              workItems[index].sourceIndexNarrowingRejected;
          const OwnershipAttemptResult &result = *results[index].attempt;
          if (const auto *materialized =
                  std::get_if<MaterializedOwnershipWorkItem>(&result)) {
            attempt["structured_program"] =
                formatArtifactIdentityHex(materialized->reference.artifact);
            attempt["published"] = static_cast<bool>(publishedWorkItem[index]);
          } else {
            const auto &rejection =
                std::get<StructuredOwnershipCandidateRejectionRecord>(result);
            attempt["rejection_kind_ordinal"] =
                static_cast<unsigned>(rejection.kind);
            attempt["rejection"] = rejection.message;
          }
          attempts.push_back(std::move(attempt));
        }
        fields["attempts"] = std::move(attempts);
      });

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
      if (publishedWorkItem[workIndex])
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
      CompletedStructuredOwnershipGeneration{
          std::move(*candidateSet), std::move(dispositions),
          plannedScopeOrdinals.size(), plannedDecisionAttemptCount,
          workItems.size(), candidateDomainTruncated},
      *parentReference, *workloadReference, *runtimeInputReference,
      std::move(materializedCandidates)};
}

llvm::Expected<CompletedStructuredOwnershipGeneration>
generateStructuredOwnershipCandidates(
    const frontend::StructuredProgramCandidate &parent,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric,
    const StructuredOwnershipGenerationOptions &options,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance) {
  StructuredOwnershipGenerationOptions effectiveOptions = options;
  evaluation::models::StructuredEvaluationInvocationCache *evaluationCache =
      nullptr;
  std::optional<detail::StructuredOwnershipPreparedSource> preparedSource;
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  if (invocation) {
    if (llvm::Error error =
            detail::StructuredOwnershipInvocationAccess::prepareGeneration(
                *invocation, parent, sourceProgram, workload, runtimeInput,
                fabric, artifactStore, effectiveOptions))
      return std::move(error);
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
      parent, sourceProgram, workload, runtimeInput, fabric, effectiveOptions,
      artifactStore, sourceProvenance, evaluationCache,
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
