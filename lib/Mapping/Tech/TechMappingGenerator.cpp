#include "Mapping/Tech/TechMappingGenerator.h"

#include "Common/MappingDebugLog.h"
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
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::InvocationBegin,
      [&](llvm::json::Object &fields) {
        fields["graph_count"] = inputs.covers.size();
        fields["match_row_attempt_limit"] =
            inputs.config.matchRowAttemptLimit();
        fields["partial_cover_expansion_limit"] =
            inputs.config.partialCoverExpansionLimit();
        fields["candidate_publication_limit"] =
            inputs.config.candidatePublicationLimit();
      });
  const auto finish = [&](TechMappingGenerationOutcome outcome) {
    llvm::StringRef status = "internal";
    std::uint64_t publicationCount = 0;
    std::visit(
        [&](const auto &result) {
          using Result = std::decay_t<decltype(result)>;
          if constexpr (std::is_same_v<Result, GeneratedTechMappings>) {
            status = result.termination ==
                             TechMappingGenerationTermination::SearchExhausted
                         ? "search_exhausted"
                         : "semantic_limit_reached";
            publicationCount = result.candidates.size();
          } else if constexpr (std::is_same_v<Result,
                                              ProvenInfeasibleTechMapping>) {
            status = "proven_infeasible";
          } else if constexpr (std::is_same_v<
                                   Result, IncompleteTechMappingGeneration>) {
            status = "incomplete";
          } else if constexpr (std::is_same_v<Result,
                                              InvalidTechMappingGeneration>) {
            status = "invalid";
          }
        },
        outcome);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
        mapping_debug::Event::InvocationEnd,
        [&](llvm::json::Object &fields) {
          fields["closure_status"] = status;
          fields["candidate_publications"] = publicationCount;
          fields["match_row_attempts"] = accounting.matchRowAttempts;
          fields["partial_cover_expansions"] =
              accounting.partialCoverExpansions;
          fields["publication_slots"] = accounting.publicationSlots;
        });
    mapping_debug::MappingRunStatistics statistics;
    statistics.candidateRows = accounting.matchRowAttempts;
    statistics.candidatePublications = publicationCount;
    statistics.emit(mapping_debug::Stage::TechMapping, status);
    return outcome;
  };
  TechMappingInvocationValidation validation = validateInvocation(inputs);
  if (auto *invalid = std::get_if<InvalidTechMappingGeneration>(&validation))
    return finish(TechMappingGenerationOutcome(std::move(*invalid)));
  auto domain = detail::deriveTechMatchDomain(
      inputs,
      std::get<ValidatedTechMappingInvocation>(validation).selectedActors,
      accounting);
  if (!domain)
    return finish(TechMappingGenerationOutcome(internal(
        InternalTechMappingGenerationReason::MatchRowDerivationFailed,
        accounting, domain.takeError())));

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
      return finish(TechMappingGenerationOutcome(internal(
          InternalTechMappingGenerationReason::CandidateFinalizationFailed,
          accounting, candidate.takeError())));
    if (!llvm::is_contained(candidates, *candidate)) {
      candidates.push_back(std::move(*candidate));
      mapping_debug::emit(
          mapping_debug::Level::Decision,
          mapping_debug::Stage::TechMapping,
          mapping_debug::Event::Candidate,
          [&](llvm::json::Object &fields) {
            fields["candidate"] = candidates.size() - 1;
            fields["publication_slot"] = accounting.publicationSlots - 1;
          });
    }
  }

  const bool exhausted = domain->exhausted && search.exhausted;
  if (!candidates.empty())
    return finish(TechMappingGenerationOutcome(GeneratedTechMappings{
        std::move(candidates),
        exhausted ? TechMappingGenerationTermination::SearchExhausted
                  : TechMappingGenerationTermination::SemanticLimitReached,
        accounting}));
  if (exhausted)
    return finish(TechMappingGenerationOutcome(
        ProvenInfeasibleTechMapping{accounting}));
  return finish(TechMappingGenerationOutcome(IncompleteTechMappingGeneration{
      IncompleteTechMappingGenerationReason::ProofNotEstablished,
      accounting}));
}

} // namespace loom::mapping
