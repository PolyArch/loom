#include "DSE/StructuredOwnership.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

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

} // namespace

llvm::Expected<StructuredOwnershipExplorationOutcome>
generateAndPromoteStructuredOwnership(
    const frontend::StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const StructuredOwnershipExplorationOptions &options,
    const ArtifactStore &artifactStore) {
  std::vector<ArtifactRootReference> candidateReferences;
  auto parentReference =
      frontend::publishStructuredProgram(parent, artifactStore);
  if (!parentReference)
    return parentReference.takeError();
  candidateReferences.push_back(*parentReference);

  auto scopes = frontend::enumerateSpatialOwnershipScopes(parent);
  if (!scopes)
    return scopes.takeError();
  for (const frontend::SpatialOwnershipScope &scope : *scopes) {
    auto decisions = frontend::enumerateSpatialOwnershipDecisionDomain(
        parent, scope.selection);
    if (!decisions)
      return decisions.takeError();
    for (const frontend::SpatialOwnershipDecisionPoint &decision : *decisions) {
      auto candidate = frontend::materializeSpatialOwnershipDecision(
          parent, scope, decision, fabric, options.lowering);
      if (!candidate) {
        auto rejected = isExpectedCandidateRejection(candidate.takeError());
        if (!rejected)
          return rejected.takeError();
        if (*rejected)
          continue;
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
      candidateReferences.push_back(std::move(*reference));
    }
  }

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
  std::vector<frontend::MaterializedOwnershipCandidate> selected;
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
    selected.push_back({std::move(*structured), std::move(*dataflow)});
  }
  return StructuredOwnershipExplorationOutcome{
      CompletedStructuredOwnershipSelection{std::move(selected),
                                            selection.satisfiedEvidence}};
}

} // namespace loom::dse
