#include "Mapping/Tech/TechMappingGenerator.h"

#include "TechMappingCandidate.h"

#include "llvm/ADT/STLExtras.h"

#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {
namespace {

struct ValidatedTechMappingInvocation final {
  std::vector<::dataflow::CanonicalActorView> selectedActors;
};

using TechMappingInvocationValidation =
    std::variant<ValidatedTechMappingInvocation, InvalidTechMappingGeneration>;

InvalidTechMappingGeneration invalid(InvalidTechMappingGenerationReason reason,
                                     llvm::StringRef diagnostic) {
  return InvalidTechMappingGeneration{reason, {}, diagnostic.str()};
}

TechMappingInvocationValidation
validateInvocation(const TechMappingGenerationInputs &inputs) {
  if (inputs.covers.empty())
    return invalid(InvalidTechMappingGenerationReason::EmptyGraphCover,
                   "graph cover scope is empty");

  std::set<std::uint64_t> coverIds;
  std::uint64_t previous = 0;
  bool first = true;
  for (const ::dataflow::GraphRef &graph : inputs.covers) {
    if (graph.artifact != inputs.dataflow.identity())
      return invalid(InvalidTechMappingGenerationReason::ForeignGraphReference,
                     "graph cover scope contains a foreign reference");
    auto resolved = inputs.dataflow.resolve(graph);
    if (!resolved) {
      llvm::consumeError(resolved.takeError());
      return invalid(
          InvalidTechMappingGenerationReason::UnresolvedGraphReference,
          "graph cover scope contains an unresolved graph reference");
    }
    const std::uint64_t id = graph.entity.value();
    if (!first && id <= previous)
      return invalid(InvalidTechMappingGenerationReason::NonCanonicalGraphCover,
                     "graph cover scope is not a canonical set");
    first = false;
    previous = id;
    coverIds.insert(id);
  }

  ValidatedTechMappingInvocation validated;
  for (const ::dataflow::CanonicalActorView &actor : inputs.dataflow.actors())
    if (coverIds.count(actor.graph.entity.value()))
      validated.selectedActors.push_back(actor);
  llvm::sort(validated.selectedActors, [](const auto &lhs, const auto &rhs) {
    return lhs.ref.entity.value() < rhs.ref.entity.value();
  });
  if (validated.selectedActors.empty())
    return invalid(InvalidTechMappingGenerationReason::GraphCoverHasNoActors,
                   "graph cover scope contains no actor");
  return validated;
}

InternalTechMappingGeneration
internal(InternalTechMappingGenerationReason reason,
         const TechMappingGenerationAccounting &accounting, llvm::Error error) {
  return InternalTechMappingGeneration{reason, accounting,
                                       llvm::toString(std::move(error))};
}

} // namespace

TechMappingGenerationOutcome
generateTechMappings(const TechMappingGenerationInputs &inputs) {
  TechMappingGenerationAccounting accounting;
  TechMappingInvocationValidation validation = validateInvocation(inputs);
  if (auto *invalid = std::get_if<InvalidTechMappingGeneration>(&validation))
    return std::move(*invalid);
  auto domain = detail::deriveTechMatchDomain(
      inputs,
      std::get<ValidatedTechMappingInvocation>(validation).selectedActors,
      accounting);
  if (!domain)
    return internal(
        InternalTechMappingGenerationReason::MatchRowDerivationFailed,
        accounting, domain.takeError());

  auto search =
      detail::searchTechMatchCovers(*domain, inputs.config, accounting);
  std::vector<ArtifactRootReference> candidates;
  for (const auto &cover : search.covers) {
    if (accounting.publicationSlots >=
        inputs.config.candidatePublicationLimit()) {
      search.exhausted = false;
      break;
    }
    ++accounting.publicationSlots;
    auto candidate = detail::materializeTechMappingCandidate(inputs, cover);
    if (!candidate)
      return internal(
          InternalTechMappingGenerationReason::CandidateFinalizationFailed,
          accounting, candidate.takeError());
    if (!llvm::is_contained(candidates, *candidate))
      candidates.push_back(std::move(*candidate));
  }

  const bool exhausted = domain->exhausted && search.exhausted;
  if (!candidates.empty())
    return TechMappingGenerationOutcome(GeneratedTechMappings{
        std::move(candidates),
        exhausted ? TechMappingGenerationTermination::SearchExhausted
                  : TechMappingGenerationTermination::SemanticLimitReached,
        accounting});
  if (exhausted)
    return TechMappingGenerationOutcome(
        ProvenInfeasibleTechMapping{accounting});
  return TechMappingGenerationOutcome(IncompleteTechMappingGeneration{
      IncompleteTechMappingGenerationReason::ProofNotEstablished, accounting});
}

} // namespace loom::mapping
