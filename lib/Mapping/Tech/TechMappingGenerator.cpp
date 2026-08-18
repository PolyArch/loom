#include "Mapping/Tech/TechMappingGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/MappingDebugLog.h"
#include "TechMappingCandidate.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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

TechMappingInterruptionSnapshot
interruptionSnapshot(TechMappingInterruptionStage stage,
                     const TechMappingGenerationAccounting &accounting,
                     std::optional<std::uint64_t> bestCanonicalRank,
                     std::uint64_t uncoveredActors,
                     std::uint64_t retainedCandidates,
                     const ExecutionResourceTracker &resources) {
  return {
      stage,
      {accounting.matchRowAttempts, accounting.partialCoverExpansions,
       accounting.candidateEvaluations, accounting.publicationSlots},
      bestCanonicalRank,
      {uncoveredActors, retainedCandidates},
      resources.observe(),
  };
}

InterruptedTechMappingGeneration
interrupted(TechMappingInterruptionStage stage,
            std::vector<ArtifactRootReference> candidates,
            const TechMappingGenerationAccounting &accounting,
            std::uint64_t uncoveredActors,
            const ExecutionResourceTracker &resources) {
  const std::uint64_t retainedCandidates = candidates.size();
  const std::optional<std::uint64_t> bestCanonicalRank =
      candidates.empty() ? std::nullopt : std::optional<std::uint64_t>(0);
  return {std::move(candidates), accounting,
          interruptionSnapshot(stage, accounting, bestCanonicalRank,
                               uncoveredActors, retainedCandidates, resources)};
}

llvm::json::Object
interruptionPayload(const TechMappingInterruptionSnapshot &snapshot) {
  llvm::json::Object frontier;
  frontier["match_row_attempts"] = snapshot.frontier.matchRowAttempts;
  frontier["partial_cover_expansions"] =
      snapshot.frontier.partialCoverExpansions;
  frontier["candidate_evaluations"] = snapshot.frontier.candidateEvaluations;
  frontier["publication_slots"] = snapshot.frontier.publicationSlots;

  llvm::json::Object residual;
  residual["uncovered_actors"] = snapshot.closureResidual.uncoveredActors;
  residual["retained_candidates"] = snapshot.closureResidual.retainedCandidates;

  llvm::json::Object resources;
  resources["active_wall_time_ns"] =
      snapshot.resources.activeWallTimeNanoseconds;
  resources["allocated_memory_bytes"] = snapshot.resources.allocatedMemoryBytes;
  if (snapshot.resources.peakResidentMemoryBytes)
    resources["peak_resident_memory_bytes"] =
        *snapshot.resources.peakResidentMemoryBytes;
  else
    resources["peak_resident_memory_bytes"] = nullptr;

  llvm::json::Object payload;
  payload["stage"] = techMappingInterruptionStageSpelling(snapshot.stage);
  payload["frontier"] = std::move(frontier);
  if (snapshot.bestCanonicalRank)
    payload["best_canonical_rank"] = *snapshot.bestCanonicalRank;
  else
    payload["best_canonical_rank"] = nullptr;
  payload["closure_residual"] = std::move(residual);
  payload["resources"] = std::move(resources);
  return payload;
}

void emitTechMatchDomainStatistics(
    const detail::TechMatchDomain &domain,
    const detail::TechMatchDomainStatistics &statistics) {
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "tech_match_domain";
        fields["row_count"] = statistics.rowCount;
        fields["compute_row_count"] = statistics.computeRowCount;
        fields["memory_row_count"] = statistics.memoryRowCount;
        fields["domain_exhausted"] = domain.exhausted;
        llvm::json::Array buckets;
        for (const detail::TechMatchMemoryDomainBucket &bucket :
             statistics.memoryBuckets) {
          llvm::json::Object value;
          value["schedule"] = bucket.schedule == ::fabric::Schedule::Temporal
                                  ? "temporal"
                                  : "spatial";
          value["actor_count"] = bucket.actorCount;
          value["occurrence_domain_width"] = bucket.occurrenceDomainWidth;
          value["row_count"] = bucket.rowCount;
          buckets.push_back(std::move(value));
        }
        fields["memory_buckets"] = std::move(buckets);
      });
}

} // namespace

llvm::StringRef
techMappingInterruptionStageSpelling(TechMappingInterruptionStage stage) {
  switch (stage) {
  case TechMappingInterruptionStage::InputAdmission:
    return "input_admission";
  case TechMappingInterruptionStage::MatchRowDerivation:
    return "match_row_derivation";
  case TechMappingInterruptionStage::CoverSearch:
    return "cover_search";
  case TechMappingInterruptionStage::CandidateFinalization:
    return "candidate_finalization";
  }
  llvm_unreachable("unknown Tech Mapping interruption stage");
}

TechMappingGenerationOutcome
generateTechMappings(const TechMappingGenerationInputs &inputs) {
  const ExecutionResourceTracker resources;
  TechMappingGenerationAccounting accounting;
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::InvocationBegin, [&](llvm::json::Object &fields) {
        fields["graph_count"] = inputs.covers.size();
        fields["match_row_attempt_limit"] =
            inputs.config.matchRowAttemptLimit();
        fields["partial_cover_expansion_limit"] =
            inputs.config.partialCoverExpansionLimit();
        fields["candidate_publication_limit"] =
            inputs.config.candidatePublicationLimit();
        fields["candidate_evaluation_limit"] =
            inputs.config.candidateEvaluationLimit();
      });
  const auto finish = [&](TechMappingGenerationOutcome outcome) {
    mapping_debug::ClosureStatus status =
        mapping_debug::ClosureStatus::Internal;
    std::uint64_t publicationCount = 0;
    const TechMappingInterruptionSnapshot *interruption = nullptr;
    std::visit(
        [&](const auto &result) {
          using Result = std::decay_t<decltype(result)>;
          if constexpr (std::is_same_v<Result, GeneratedTechMappings>) {
            status = result.termination ==
                             TechMappingGenerationTermination::SearchExhausted
                         ? mapping_debug::ClosureStatus::SearchExhausted
                         : mapping_debug::ClosureStatus::SemanticLimitReached;
            publicationCount = result.candidates.size();
          } else if constexpr (std::is_same_v<Result,
                                              ProvenInfeasibleTechMapping>) {
            status = mapping_debug::ClosureStatus::ProvenInfeasible;
          } else if constexpr (std::is_same_v<
                                   Result, IncompleteTechMappingGeneration>) {
            status = mapping_debug::ClosureStatus::ProofNotEstablished;
          } else if constexpr (std::is_same_v<
                                   Result, InterruptedTechMappingGeneration>) {
            status = mapping_debug::ClosureStatus::CancelledOrTimeout;
            publicationCount = result.candidates.size();
            interruption = &result.snapshot;
          } else if constexpr (std::is_same_v<Result,
                                              InvalidTechMappingGeneration>) {
            status = mapping_debug::ClosureStatus::Invalid;
          }
        },
        outcome);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
        mapping_debug::Event::InvocationEnd, [&](llvm::json::Object &fields) {
          fields["closure_status"] =
              mapping_debug::closureStatusSpelling(status);
          fields["candidate_publications"] = publicationCount;
          fields["match_row_attempts"] = accounting.matchRowAttempts;
          fields["match_row_first_visits"] = accounting.matchRowFirstVisits;
          fields["match_row_cursor_resumptions"] =
              accounting.matchRowCursorResumptions;
          fields["match_row_replay_visits"] = accounting.matchRowReplayVisits;
          fields["memory_row_frontier_limits"] =
              accounting.memoryRowFrontierLimits;
          fields["partial_cover_expansions"] =
              accounting.partialCoverExpansions;
          fields["constructive_cover_search_invocations"] =
              accounting.constructiveCoverSearchInvocations;
          fields["constructive_cover_completed_checks"] =
              accounting.constructiveCoverCompletedChecks;
          fields["constructive_cover_publications"] =
              accounting.constructiveCoverPublications;
          fields["compute_context_projection_work"] =
              accounting.computeContextProjectionWork;
          fields["compute_context_matching_checks"] =
              accounting.computeContextMatchingChecks;
          fields["compute_context_rejected_checks"] =
              accounting.computeContextRejectedChecks;
          fields["compute_context_matching_work"] =
              accounting.computeContextMatchingWork;
          fields["memory_supply_projection_work"] =
              accounting.memorySupplyProjectionWork;
          fields["memory_supply_checks"] = accounting.memorySupplyChecks;
          fields["memory_supply_partial_checks"] =
              accounting.memorySupplyPartialChecks;
          fields["memory_supply_full_checks"] =
              accounting.memorySupplyFullChecks;
          fields["memory_supply_rejected_checks"] =
              accounting.memorySupplyRejectedChecks;
          fields["memory_supply_empty_domain_rejections"] =
              accounting.memorySupplyEmptyDomainRejections;
          fields["memory_supply_exclusive_resource_rejections"] =
              accounting.memorySupplyExclusiveResourceRejections;
          fields["memory_supply_spatial_port_rejections"] =
              accounting.memorySupplySpatialPortRejections;
          fields["memory_supply_temporal_ingress_rejections"] =
              accounting.memorySupplyTemporalIngressRejections;
          fields["memory_supply_internal_connection_rejections"] =
              accounting.memorySupplyInternalConnectionRejections;
          fields["memory_supply_resident_capacity_rejections"] =
              accounting.memorySupplyResidentCapacityRejections;
          fields["memory_supply_joint_assignment_rejections"] =
              accounting.memorySupplyJointAssignmentRejections;
          fields["memory_supply_search_work"] =
              accounting.memorySupplySearchWork;
          fields["candidate_evaluations"] = accounting.candidateEvaluations;
          fields["publication_slots"] = accounting.publicationSlots;
          if (interruption)
            fields["interruption"] = interruptionPayload(*interruption);
        });
    mapping_debug::MappingRunStatistics statistics;
    statistics.candidateRows = accounting.matchRowAttempts;
    statistics.candidatePublications = publicationCount;
    statistics.emit(mapping_debug::Stage::TechMapping, status,
                    [&](llvm::json::Object &fields) {
                      if (interruption)
                        fields["interruption"] =
                            interruptionPayload(*interruption);
                    });
    return outcome;
  };
  TechMappingInvocationValidation validation = validateInvocation(inputs);
  if (auto *invalid = std::get_if<InvalidTechMappingGeneration>(&validation))
    return finish(TechMappingGenerationOutcome(std::move(*invalid)));
  const auto &selectedActors =
      std::get<ValidatedTechMappingInvocation>(validation).selectedActors;
  if (inputs.executionControl.stopRequested())
    return finish(TechMappingGenerationOutcome(
        interrupted(TechMappingInterruptionStage::InputAdmission, {},
                    accounting, selectedActors.size(), resources)));
  auto domain =
      detail::deriveTechMatchDomain(inputs, selectedActors, accounting);
  if (!domain)
    return finish(TechMappingGenerationOutcome(
        internal(InternalTechMappingGenerationReason::MatchRowDerivationFailed,
                 accounting, domain.takeError())));
  if (domain->interrupted)
    return finish(TechMappingGenerationOutcome(
        interrupted(TechMappingInterruptionStage::MatchRowDerivation, {},
                    accounting, domain->actors.size(), resources)));
  const detail::TechMatchDomainStatistics domainStatistics =
      detail::summarizeTechMatchDomain(*domain);
  emitTechMatchDomainStatistics(*domain, domainStatistics);

  auto search = detail::searchTechMatchCovers(
      *domain, inputs.config, accounting,
      std::min(inputs.config.candidateEvaluationLimit(),
               inputs.config.candidatePublicationLimit()),
      inputs.executionControl);
  std::vector<ArtifactRootReference> candidates;
  if (search.interrupted) {
    auto outcome = interrupted(
        TechMappingInterruptionStage::CoverSearch, std::move(candidates),
        accounting, search.covers.empty() ? domain->actors.size() : 0,
        resources);
    outcome.feedback = std::move(search.feedback);
    return finish(TechMappingGenerationOutcome(std::move(outcome)));
  }
  for (const auto &cover : search.covers) {
    if (inputs.executionControl.stopRequested()) {
      auto outcome =
          interrupted(TechMappingInterruptionStage::CandidateFinalization,
                      std::move(candidates), accounting, 0, resources);
      outcome.feedback = std::move(search.feedback);
      return finish(TechMappingGenerationOutcome(std::move(outcome)));
    }
    if (accounting.publicationSlots >=
        inputs.config.candidatePublicationLimit()) {
      search.exhausted = false;
      break;
    }
    ++accounting.candidateEvaluations;
    ++accounting.publicationSlots;
    auto candidate = detail::materializeTechMappingCandidate(inputs, cover);
    if (!candidate)
      return finish(TechMappingGenerationOutcome(internal(
          InternalTechMappingGenerationReason::CandidateFinalizationFailed,
          accounting, candidate.takeError())));
    if (!llvm::is_contained(candidates, *candidate)) {
      candidates.push_back(std::move(*candidate));
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::TechMapping,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["candidate"] = candidates.size() - 1;
            fields["publication_slot"] = accounting.publicationSlots - 1;
          });
    }
  }

  if (inputs.executionControl.stopRequested()) {
    auto outcome =
        interrupted(TechMappingInterruptionStage::CandidateFinalization,
                    std::move(candidates), accounting, 0, resources);
    outcome.feedback = std::move(search.feedback);
    return finish(TechMappingGenerationOutcome(std::move(outcome)));
  }

  const bool exhausted = domain->exhausted && search.exhausted;
  if (!candidates.empty())
    return finish(TechMappingGenerationOutcome(GeneratedTechMappings{
        std::move(candidates),
        exhausted ? TechMappingGenerationTermination::SearchExhausted
                  : TechMappingGenerationTermination::SemanticLimitReached,
        accounting, std::move(search.feedback)}));
  if (exhausted)
    return finish(TechMappingGenerationOutcome(
        ProvenInfeasibleTechMapping{accounting, std::move(search.feedback)}));
  return finish(TechMappingGenerationOutcome(IncompleteTechMappingGeneration{
      IncompleteTechMappingGenerationReason::ProofNotEstablished, accounting,
      std::move(search.feedback)}));
}

llvm::Expected<TechMappingCandidateEnumerationResult>
enumerateTechMappingCandidates(
    const TechMappingGenerationInputs &inputs,
    llvm::function_ref<llvm::Expected<TechMappingCandidateEnumerationControl>(
        const ArtifactRootReference &)>
        visitor) {
  const ExecutionResourceTracker resources;
  TechMappingGenerationAccounting accounting;
  TechMappingInvocationValidation validation = validateInvocation(inputs);
  if (const auto *rejected =
          std::get_if<InvalidTechMappingGeneration>(&validation))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_candidate_enumeration_invalid: " + rejected->diagnostic);
  const auto &selectedActors =
      std::get<ValidatedTechMappingInvocation>(validation).selectedActors;
  if (inputs.executionControl.stopRequested())
    return TechMappingCandidateEnumerationResult{
        TechMappingGenerationTermination::SemanticLimitReached, accounting, 0,
        interruptionSnapshot(TechMappingInterruptionStage::InputAdmission,
                             accounting, std::nullopt, selectedActors.size(), 0,
                             resources)};
  auto domain =
      detail::deriveTechMatchDomain(inputs, selectedActors, accounting);
  if (!domain)
    return domain.takeError();
  if (domain->interrupted)
    return TechMappingCandidateEnumerationResult{
        TechMappingGenerationTermination::SemanticLimitReached, accounting, 0,
        interruptionSnapshot(TechMappingInterruptionStage::MatchRowDerivation,
                             accounting, std::nullopt, domain->actors.size(), 0,
                             resources)};
  const detail::TechMatchDomainStatistics domainStatistics =
      detail::summarizeTechMatchDomain(*domain);
  emitTechMatchDomainStatistics(*domain, domainStatistics);
  auto search = detail::searchTechMatchCovers(
      *domain, inputs.config, accounting,
      inputs.config.candidateEvaluationLimit(), inputs.executionControl);
  if (search.interrupted)
    return TechMappingCandidateEnumerationResult{
        TechMappingGenerationTermination::SemanticLimitReached, accounting, 0,
        interruptionSnapshot(
            TechMappingInterruptionStage::CoverSearch, accounting, std::nullopt,
            search.covers.empty() ? domain->actors.size() : 0, 0, resources),
        std::move(search.feedback)};
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)> seen(
      artifactRootReferenceLess);
  std::uint64_t visited = 0;
  bool visitorStopped = false;
  for (const auto &cover : search.covers) {
    if (inputs.executionControl.stopRequested())
      return TechMappingCandidateEnumerationResult{
          TechMappingGenerationTermination::SemanticLimitReached, accounting,
          visited,
          interruptionSnapshot(
              TechMappingInterruptionStage::CandidateFinalization, accounting,
              visited == 0 ? std::nullopt : std::optional<std::uint64_t>(0), 0,
              visited, resources),
          std::move(search.feedback)};
    auto candidate = detail::materializeTechMappingCandidate(inputs, cover);
    if (!candidate)
      return candidate.takeError();
    if (!seen.insert(*candidate).second)
      continue;
    ++accounting.candidateEvaluations;
    ++visited;
    auto control = visitor(*candidate);
    if (!control)
      return control.takeError();
    if (*control == TechMappingCandidateEnumerationControl::Stop) {
      visitorStopped = true;
      break;
    }
  }
  if (inputs.executionControl.stopRequested())
    return TechMappingCandidateEnumerationResult{
        TechMappingGenerationTermination::SemanticLimitReached, accounting,
        visited,
        interruptionSnapshot(
            TechMappingInterruptionStage::CandidateFinalization, accounting,
            visited == 0 ? std::nullopt : std::optional<std::uint64_t>(0), 0,
            visited, resources),
        std::move(search.feedback)};
  const bool exhausted = domain->exhausted && search.exhausted &&
                         !visitorStopped && visited == seen.size();
  return TechMappingCandidateEnumerationResult{
      exhausted ? TechMappingGenerationTermination::SearchExhausted
                : TechMappingGenerationTermination::SemanticLimitReached,
      accounting, visited, std::nullopt, std::move(search.feedback)};
}

} // namespace loom::mapping
