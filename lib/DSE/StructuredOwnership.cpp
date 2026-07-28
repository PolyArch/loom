#include "DSE/StructuredOwnership.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

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

llvm::Expected<bool> isExpectedCandidateRejection(llvm::Error error) {
  bool rejected = false;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(error),
      [&](const frontend::SpatialOwnershipCandidateRejection &) {
        rejected = true;
      });
  if (unhandled)
    return std::move(unhandled);
  return rejected;
}

struct OwnershipWorkItem final {
  frontend::SpatialOwnershipScope scope;
  frontend::SpatialOwnershipDecisionPoint decision;
};

struct MaterializedOwnershipWorkItem final {
  ArtifactRootReference candidate;
  StructuredOwnershipDerivation derivation;
};

llvm::Expected<std::optional<MaterializedOwnershipWorkItem>>
materializeOwnershipWorkItem(
    const frontend::StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore, const OwnershipWorkItem &workItem) {
  auto candidate = frontend::materializeSpatialOwnershipDecision(
      parent, workItem.scope, workItem.decision, fabric, options.lowering);
  if (!candidate) {
    auto rejected = isExpectedCandidateRejection(candidate.takeError());
    if (!rejected)
      return rejected.takeError();
    if (*rejected)
      return std::optional<MaterializedOwnershipWorkItem>{};
    return invalid("candidate failed without a classified error");
  }

  auto reference = frontend::publishStructuredProgram(
      candidate->structuredProgram, artifactStore);
  if (!reference)
    return reference.takeError();
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              *reference, candidate->structuredProgram,
              candidate->canonicalDataflow, fabric, config, artifactStore))
    return std::move(error);
  return std::optional<MaterializedOwnershipWorkItem>(
      MaterializedOwnershipWorkItem{
          std::move(*reference),
          StructuredOwnershipDerivation{workItem.scope, workItem.decision}});
}

} // namespace

llvm::Expected<StructuredOwnershipExplorationOutcome>
generateAndPromoteStructuredOwnership(
    const frontend::StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore) {
  if (options.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (llvm::Error error =
          evaluation::models::registerStructuredFabricAnalyticModel())
    return std::move(error);

  std::vector<ArtifactRootReference> candidateReferences;
  auto parentReference =
      frontend::publishStructuredProgram(parent, artifactStore);
  if (!parentReference)
    return parentReference.takeError();
  candidateReferences.push_back(*parentReference);

  auto scopes = frontend::enumerateSpatialOwnershipScopes(parent);
  if (!scopes)
    return scopes.takeError();
  std::vector<OwnershipWorkItem> workItems;
  for (const frontend::SpatialOwnershipScope &scope : *scopes) {
    auto decisions = frontend::enumerateSpatialOwnershipDecisionDomain(
        parent, scope.selection);
    if (!decisions)
      return decisions.takeError();
    for (const frontend::SpatialOwnershipDecisionPoint &decision : *decisions)
      workItems.push_back({scope, decision});
  }
  struct WorkResult final {
    std::optional<MaterializedOwnershipWorkItem> materialized;
    std::optional<llvm::Error> error;
  };
  std::vector<WorkResult> results(workItems.size());
  auto execute = [&](const frontend::StructuredProgramCandidate &workerParent,
                     std::size_t index) {
    auto result = materializeOwnershipWorkItem(
        workerParent, fabric, config, options, artifactStore, workItems[index]);
    if (!result) {
      results[index].error.emplace(result.takeError());
      return;
    }
    results[index].materialized = std::move(*result);
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
  for (WorkResult &result : results) {
    if (result.error) {
      failures =
          llvm::joinErrors(std::move(failures), std::move(*result.error));
      continue;
    }
    if (result.materialized)
      candidateReferences.push_back(result.materialized->candidate);
  }
  if (failures)
    return std::move(failures);

  auto candidateSet = CandidateSet::get(
      frontend::structuredProgramArtifactSchema, candidateReferences);
  if (!candidateSet)
    return candidateSet.takeError();

  std::vector<PromotionEvidence> evidence;
  evidence.reserve(candidateSet->candidates().size());
  std::optional<evaluation::CaseSubjectRoleRef> candidateRole;
  for (const ArtifactRootReference &candidate : candidateSet->candidates()) {
    auto prepared = evaluation::models::prepareStructuredFabricEvaluation(
        candidate, fabric.reference(), config, artifactStore);
    if (!prepared)
      return prepared.takeError();
    if (candidateRole && *candidateRole != prepared->candidateRole)
      return invalid("model changed its candidate role across obligations");
    candidateRole = prepared->candidateRole;
    auto result = evaluation::evaluateRequest(
        prepared->request, prepared->resolution, artifactStore);
    if (!result)
      return result.takeError();
    evidence.push_back({std::move(prepared->request), std::move(*result)});
  }
  if (!candidateRole)
    return invalid("nonempty candidate set produced no Evidence obligations");

  auto promoted = promoteMetricTopK(*candidateSet, *candidateRole, evidence,
                                    options.selection, artifactStore);
  if (!promoted)
    return promoted.takeError();
  if (const auto *incomplete = std::get_if<IncompleteSelection>(&*promoted))
    return StructuredOwnershipExplorationOutcome{*incomplete};
  if (std::holds_alternative<CompletedNoFeasibleCandidate>(*promoted))
    return StructuredOwnershipExplorationOutcome{
        CompletedNoFeasibleCandidate{}};

  const auto &selection = std::get<CompletedSelection>(*promoted);
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
    for (const WorkResult &result : results)
      if (result.materialized && result.materialized->candidate == reference)
        derivations.push_back(result.materialized->derivation);
    selected.push_back({frontend::MaterializedOwnershipCandidate{
                            std::move(*structured), std::move(*dataflow)},
                        std::move(derivations)});
  }
  return StructuredOwnershipExplorationOutcome{
      CompletedStructuredOwnershipSelection{std::move(selected),
                                            selection.satisfiedEvidence}};
}

} // namespace loom::dse
