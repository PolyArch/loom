#include "PnR/System/SystemPnrGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/FabricTopologyQualityDiagnostic.h"
#include "PnR/PnrWorkLedger.h"
#include "PnR/System/SystemActionExecutor.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <string>
#include <type_traits>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

enum class FreezeFailureKind : std::uint8_t {
  Invalid,
  ProvenInfeasible,
  Internal,
};

struct FreezeFailure final {
  FreezeFailureKind kind = FreezeFailureKind::Internal;
  std::string diagnostic;
};

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

FreezeFailure classifyFreezeFailure(llvm::Error error) {
  FreezeFailure result;
  llvm::handleAllErrors(
      std::move(error),
      [&](const SystemPnrFreezeFailure &failure) {
        result.kind = failure.kind() == SystemPnrFreezeFailureKind::Invalid
                          ? FreezeFailureKind::Invalid
                          : FreezeFailureKind::ProvenInfeasible;
        result.diagnostic = errorMessage(failure);
      },
      [&](const llvm::ErrorInfoBase &failure) {
        result.kind = FreezeFailureKind::Internal;
        result.diagnostic = errorMessage(failure);
      });
  return result;
}

void emitProvenInfeasibleFreeze(llvm::StringRef scope,
                                llvm::StringRef diagnostic) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = scope;
        fields["closure_status"] = "proven_infeasible";
        fields["diagnostic"] = diagnostic;
      });
}

struct InitializationFailure final {
  SystemCandidateInitializationFailureKind kind =
      SystemCandidateInitializationFailureKind::Internal;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
  std::string diagnostic;
};

InitializationFailure classifyInitializationFailure(llvm::Error error) {
  InitializationFailure result;
  llvm::handleAllErrors(
      std::move(error),
      [&](const SystemCandidateInitializationFailure &failure) {
        result.kind = failure.kind();
        result.assignmentAttempts = failure.assignmentAttempts();
        result.endpointExpansions = failure.endpointExpansions();
        result.negotiationIterations = failure.negotiationIterations();
        result.diagnostic = errorMessage(failure);
      },
      [&](const llvm::ErrorInfoBase &failure) {
        result.kind = SystemCandidateInitializationFailureKind::Internal;
        result.diagnostic = errorMessage(failure);
      });
  return result;
}

InternalSystemPnrGeneration
internal(InternalSystemPnrGenerationReason reason,
         const SystemPnrGenerationAccounting &accounting,
         const llvm::Twine &diagnostic) {
  return {reason, accounting, diagnostic.str()};
}

InternalSystemPnrGeneration
internal(InternalSystemPnrGenerationReason reason,
         const SystemPnrGenerationAccounting &accounting, llvm::Error error) {
  return internal(reason, accounting, llvm::toString(std::move(error)));
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        "System PnR accounting overflow: " + subject);
  target += amount;
  return llvm::Error::success();
}

llvm::Error accumulateAnnealing(const SystemAnnealingStatistics &source,
                                SystemPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.mutationOracleVerificationCount,
                                     target.mutationOracleVerificationAttempts,
                                     "mutation oracle verification attempts"))
    return error;
  return checkedAdd(source.acceptedActionCount, target.annealingAcceptedActions,
                    "annealing accepted Actions");
}

PnrWorkLedgerView
canonicalWorkLedger(SystemPnrGenerationAccounting &accounting) {
  std::array<PnrWorkCounterRef, pnrWorkKindCount> counters{};
  const auto bind = [&](PnrWorkKind kind, std::uint64_t &planned,
                        std::uint64_t &consumed) {
    counters[static_cast<std::size_t>(kind)] = {&planned, &consumed};
  };
  bind(PnrWorkKind::SeedAttempt, accounting.plannedSeedAttemptSlots,
       accounting.seedAttemptSlots);
  bind(PnrWorkKind::InitializerAssignment,
       accounting.plannedInitializerAssignmentAttempts,
       accounting.initializerAssignmentAttempts);
  bind(PnrWorkKind::EndpointExpansion, accounting.plannedEndpointExpansionSlots,
       accounting.endpointExpansionSlots);
  bind(PnrWorkKind::NegotiationIteration,
       accounting.plannedNegotiationIterationSlots,
       accounting.negotiationIterationSlots);
  bind(PnrWorkKind::CalibrationProposal,
       accounting.plannedCalibrationProposalSlots,
       accounting.calibrationProposalSlots);
  bind(PnrWorkKind::AnnealingBaseProposal,
       accounting.plannedAnnealingBaseProposalSlots,
       accounting.annealingBaseProposalSlots);
  bind(PnrWorkKind::AnnealingMovableProposal,
       accounting.plannedAnnealingMovableProposalSlots,
       accounting.annealingMovableProposalSlots);
  bind(PnrWorkKind::ExactRepairRegionDecision,
       accounting.plannedExactRepairRegionDecisions,
       accounting.exactRepairRegionDecisions);
  bind(PnrWorkKind::ExactRepairSolverCall,
       accounting.plannedExactRepairSolverCalls,
       accounting.exactRepairSolverCalls);
  bind(PnrWorkKind::FinalClosureAttempt, accounting.plannedFinalClosureAttempts,
       accounting.finalClosureAttempts);
  return PnrWorkLedgerView(counters);
}

void emitInvocationAccounting(const SystemPnrGenerationAccounting &accounting,
                              mapping_debug::ClosureStatus closureStatus,
                              std::uint64_t candidatePublications) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "system_pnr_invocation";
        fields["closure_status"] =
            mapping_debug::closureStatusSpelling(closureStatus);
        fields["candidate_publications"] = candidatePublications;
        fields["migration_seed_attempt_slots"] =
            accounting.migrationSeedAttemptSlots;
        fields["migration_seed_prepared"] = accounting.migrationSeedPrepared;
        fields["migration_seed_fallbacks"] = accounting.migrationSeedFallbacks;
        fields["migration_preserved_thread_bindings"] =
            accounting.migrationPreservedThreadBindings;
        fields["migration_preserved_graph_bindings"] =
            accounting.migrationPreservedGraphBindings;
        fields["migration_preserved_service_legs"] =
            accounting.migrationPreservedServiceLegs;
        fields["migration_preserved_resource_uses"] =
            accounting.migrationPreservedResourceUses;
        fields["migration_reopened_thread_bindings"] =
            accounting.migrationReopenedThreadBindings;
        fields["migration_reopened_graph_bindings"] =
            accounting.migrationReopenedGraphBindings;
        fields["migration_reopened_service_legs"] =
            accounting.migrationReopenedServiceLegs;
        fields["migration_reopened_resource_uses"] =
            accounting.migrationReopenedResourceUses;
        fields["migration_new_service_legs"] =
            accounting.migrationNewServiceLegs;
        fields["migration_new_resource_uses"] =
            accounting.migrationNewResourceUses;
        fields["planned_seed_attempt_slots"] =
            accounting.plannedSeedAttemptSlots;
        fields["seed_attempt_slots"] = accounting.seedAttemptSlots;
        fields["prepared_seeds"] = accounting.preparedSeeds;
        fields["planned_initializer_assignment_attempts"] =
            accounting.plannedInitializerAssignmentAttempts;
        fields["initializer_assignment_attempts"] =
            accounting.initializerAssignmentAttempts;
        fields["planned_endpoint_expansion_slots"] =
            accounting.plannedEndpointExpansionSlots;
        fields["endpoint_expansion_slots"] = accounting.endpointExpansionSlots;
        fields["planned_negotiation_iteration_slots"] =
            accounting.plannedNegotiationIterationSlots;
        fields["negotiation_iteration_slots"] =
            accounting.negotiationIterationSlots;
        fields["planned_calibration_proposal_slots"] =
            accounting.plannedCalibrationProposalSlots;
        fields["calibration_proposal_slots"] =
            accounting.calibrationProposalSlots;
        fields["planned_annealing_base_proposal_slots"] =
            accounting.plannedAnnealingBaseProposalSlots;
        fields["annealing_base_proposal_slots"] =
            accounting.annealingBaseProposalSlots;
        fields["planned_annealing_movable_proposal_slots"] =
            accounting.plannedAnnealingMovableProposalSlots;
        fields["annealing_movable_proposal_slots"] =
            accounting.annealingMovableProposalSlots;
        fields["annealing_accepted_actions"] =
            accounting.annealingAcceptedActions;
        fields["mutation_oracle_verification_attempts"] =
            accounting.mutationOracleVerificationAttempts;
        fields["exact_repair_invocations"] = accounting.exactRepairInvocations;
        fields["planned_exact_repair_region_decisions"] =
            accounting.plannedExactRepairRegionDecisions;
        fields["exact_repair_region_decisions"] =
            accounting.exactRepairRegionDecisions;
        fields["planned_exact_repair_solver_calls"] =
            accounting.plannedExactRepairSolverCalls;
        fields["exact_repair_solver_calls"] = accounting.exactRepairSolverCalls;
        fields["planned_final_closure_attempts"] =
            accounting.plannedFinalClosureAttempts;
        fields["final_closure_attempts"] = accounting.finalClosureAttempts;
        fields["final_verification_attempts"] =
            accounting.finalVerificationAttempts;
        fields["finalized_restarts"] = accounting.finalizedRestarts;
        fields["publication_slots"] = accounting.publicationSlots;
      });
}

llvm::Expected<ArtifactRootReference>
publishExecutionBindingCheckpoint(const FrozenSystemPnrProblem &problem,
                                  const SystemCapacityOveruseWitness &witness,
                                  llvm::ArrayRef<PnrIndex> choices,
                                  const ArtifactStore &store) {
  const std::size_t threadCount = problem.threadDecisions().size();
  if (choices.size() != threadCount + problem.graphDecisions().size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "System capacity checkpoint has an incomplete choice vector");
  std::vector<::loom::mapping::SystemThreadExecutionCheckpoint> threads;
  threads.reserve(threadCount);
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.threadDecisions())) {
    const auto domain = problem.threadChoiceCatalogOrdinals(decision);
    if (choices[decision] >= domain.size() ||
        domain[choices[decision]] >= problem.accCores().size())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "System capacity checkpoint has a foreign thread choice");
    threads.push_back({frozen.root, frozen.cell,
                       problem.accCores()[domain[choices[decision]]]});
  }
  std::vector<::loom::mapping::SystemGraphExecutionCheckpoint> graphs;
  graphs.reserve(problem.graphDecisions().size());
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.graphDecisions())) {
    const PnrIndex choice = choices[threadCount + decision];
    const auto domain = problem.graphChoiceCatalogOrdinals(decision);
    if (choice >= domain.size() ||
        domain[choice] >= problem.spatialMappings().size())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "System capacity checkpoint has a foreign graph choice");
    graphs.push_back({frozen.launch, frozen.cell,
                      problem.spatialMappings()[domain[choice]]});
  }
  ArtifactRootReference dataflow{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, problem.dataflowIdentity()};
  ArtifactRootReference system{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, problem.fabricIdentity()};
  ArtifactRootReference constraints{
      ::loom::mapping::mappingConstraintSetSchema.identity.str(),
      ::loom::mapping::mappingConstraintSetSchema.version,
      problem.constraintIdentity()};
  if (witness.namespaceOrdinal == 0 ||
      witness.namespaceOrdinal > problem.accCores().size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "System capacity checkpoint has no exact witness AccCore");
  const auto witnessAccCore =
      problem
          .accCores()[static_cast<std::size_t>(witness.namespaceOrdinal - 1)];
  std::vector<::dataflow::RootThreadLaunchRef> dependencyRoots;
  for (const auto &binding : threads)
    if (binding.target == witnessAccCore &&
        !llvm::is_contained(dependencyRoots, binding.root))
      dependencyRoots.push_back(binding.root);
  auto searchDomainDigest =
      ComponentViewDigest::fromBytes(problem.searchDomainDigest().bytes());
  if (!searchDomainDigest)
    return searchDomainDigest.takeError();
  auto finalized = ::loom::mapping::finalizeSystemExecutionBindingCheckpoint(
      std::move(dataflow), std::move(system), std::move(constraints),
      problem.config().digest(), std::move(*searchDomainDigest),
      {::loom::mapping::SystemExecutionBindingCheckpointIncompleteKind::
           ImportedSpatialCapacity,
       witnessAccCore, witness.usage, witness.capacity,
       std::move(dependencyRoots)},
      std::move(threads), std::move(graphs), store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

struct SystemInterruptionBestProjection final {
  std::optional<dse::ObjectiveVector> objective;
  std::optional<
      std::array<std::optional<std::uint64_t>, resolvedPnrViolationKindCount>>
      violationValues;
};

llvm::Error considerInterruptionCandidate(
    const SystemCandidateState &candidate,
    SystemInterruptionBestProjection &best,
    const dse::ObjectiveVector *knownObjective = nullptr) {
  std::optional<dse::ObjectiveVector> evaluatedObjective;
  if (!knownObjective) {
    auto objective = candidate.problem().objectiveProgram().evaluate(candidate);
    if (!objective)
      return objective.takeError();
    evaluatedObjective.emplace(std::move(*objective));
    knownObjective = &*evaluatedObjective;
  }
  bool selected = !best.violationValues;
  if (!best.objective) {
    selected = true;
  } else {
    auto comparison =
        candidate.problem().objectiveProgram().compareSelectedRank(
            *knownObjective, {}, *best.objective, {});
    if (!comparison)
      return comparison.takeError();
    selected = *comparison < 0;
  }
  if (!selected)
    return llvm::Error::success();

  best.objective = *knownObjective;
  std::array<std::optional<std::uint64_t>, resolvedPnrViolationKindCount>
      values{};
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal) {
    auto value = systemMappingViolationValue(
        candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
    if (!value)
      return value.takeError();
    values[ordinal] = *value;
  }
  best.violationValues = std::move(values);
  return llvm::Error::success();
}

SystemPnrInterruptionSnapshot
projectInterruptionSnapshot(SystemPnrInterruptionStage stage,
                            std::optional<std::uint32_t> restartOrdinal,
                            const SystemPnrGenerationAccounting &accounting,
                            std::uint64_t retainedCandidates,
                            const SystemInterruptionBestProjection &best,
                            const ExecutionResourceTracker &resources) {
  SystemPnrInterruptionSnapshot snapshot;
  snapshot.stage = stage;
  snapshot.frontier = {
      restartOrdinal,
      accounting.migrationSeedAttemptSlots,
      accounting.migrationSeedPrepared,
      accounting.migrationSeedFallbacks,
      accounting.migrationPreservedThreadBindings,
      accounting.migrationPreservedGraphBindings,
      accounting.migrationPreservedServiceLegs,
      accounting.migrationPreservedResourceUses,
      accounting.migrationReopenedThreadBindings,
      accounting.migrationReopenedGraphBindings,
      accounting.migrationReopenedServiceLegs,
      accounting.migrationReopenedResourceUses,
      accounting.migrationNewServiceLegs,
      accounting.migrationNewResourceUses,
      accounting.seedAttemptSlots,
      accounting.preparedSeeds,
      accounting.initializerAssignmentAttempts,
      accounting.endpointExpansionSlots,
      accounting.negotiationIterationSlots,
      accounting.calibrationProposalSlots,
      accounting.annealingBaseProposalSlots,
      accounting.annealingMovableProposalSlots,
      accounting.mutationOracleVerificationAttempts,
      accounting.finalClosureAttempts,
      accounting.finalVerificationAttempts,
      accounting.finalizedRestarts,
      accounting.publicationSlots,
  };
  if (best.objective)
    snapshot.bestSelectedRank = std::vector<std::uint64_t>(
        best.objective->codes().begin(), best.objective->codes().end());
  snapshot.closureResidual.violationValues = best.violationValues;
  snapshot.closureResidual.retainedCandidates = retainedCandidates;
  snapshot.resources = resources.observe();
  return snapshot;
}

llvm::json::Object
interruptionPayload(const SystemPnrInterruptionSnapshot &snapshot) {
  llvm::json::Object frontier;
  if (snapshot.frontier.restartOrdinal)
    frontier["restart_ordinal"] = *snapshot.frontier.restartOrdinal;
  else
    frontier["restart_ordinal"] = nullptr;
  frontier["migration_seed_attempt_slots"] =
      snapshot.frontier.migrationSeedAttemptSlots;
  frontier["migration_seed_prepared"] = snapshot.frontier.migrationSeedPrepared;
  frontier["migration_seed_fallbacks"] =
      snapshot.frontier.migrationSeedFallbacks;
  frontier["migration_preserved_thread_bindings"] =
      snapshot.frontier.migrationPreservedThreadBindings;
  frontier["migration_preserved_graph_bindings"] =
      snapshot.frontier.migrationPreservedGraphBindings;
  frontier["migration_preserved_service_legs"] =
      snapshot.frontier.migrationPreservedServiceLegs;
  frontier["migration_preserved_resource_uses"] =
      snapshot.frontier.migrationPreservedResourceUses;
  frontier["migration_reopened_thread_bindings"] =
      snapshot.frontier.migrationReopenedThreadBindings;
  frontier["migration_reopened_graph_bindings"] =
      snapshot.frontier.migrationReopenedGraphBindings;
  frontier["migration_reopened_service_legs"] =
      snapshot.frontier.migrationReopenedServiceLegs;
  frontier["migration_reopened_resource_uses"] =
      snapshot.frontier.migrationReopenedResourceUses;
  frontier["migration_new_service_legs"] =
      snapshot.frontier.migrationNewServiceLegs;
  frontier["migration_new_resource_uses"] =
      snapshot.frontier.migrationNewResourceUses;
  frontier["seed_attempt_slots"] = snapshot.frontier.seedAttemptSlots;
  frontier["prepared_seeds"] = snapshot.frontier.preparedSeeds;
  frontier["initializer_assignment_attempts"] =
      snapshot.frontier.initializerAssignmentAttempts;
  frontier["endpoint_expansion_slots"] =
      snapshot.frontier.endpointExpansionSlots;
  frontier["negotiation_iteration_slots"] =
      snapshot.frontier.negotiationIterationSlots;
  frontier["calibration_proposal_slots"] =
      snapshot.frontier.calibrationProposalSlots;
  frontier["annealing_base_proposal_slots"] =
      snapshot.frontier.annealingBaseProposalSlots;
  frontier["annealing_movable_proposal_slots"] =
      snapshot.frontier.annealingMovableProposalSlots;
  frontier["mutation_oracle_verification_attempts"] =
      snapshot.frontier.mutationOracleVerificationAttempts;
  frontier["final_closure_attempts"] = snapshot.frontier.finalClosureAttempts;
  frontier["final_verification_attempts"] =
      snapshot.frontier.finalVerificationAttempts;
  frontier["finalized_restarts"] = snapshot.frontier.finalizedRestarts;
  frontier["publication_slots"] = snapshot.frontier.publicationSlots;

  llvm::json::Object residual;
  if (snapshot.closureResidual.violationValues) {
    llvm::json::Array values;
    for (const std::optional<std::uint64_t> &value :
         *snapshot.closureResidual.violationValues)
      if (value)
        values.push_back(*value);
      else
        values.push_back(nullptr);
    residual["violation_values"] = std::move(values);
  } else {
    residual["violation_values"] = nullptr;
  }
  residual["retained_candidates"] = snapshot.closureResidual.retainedCandidates;

  llvm::json::Array rank;
  if (snapshot.bestSelectedRank)
    for (std::uint64_t code : *snapshot.bestSelectedRank)
      rank.push_back(code);
  llvm::json::Object resourceValues;
  resourceValues["active_wall_time_ns"] =
      snapshot.resources.activeWallTimeNanoseconds;
  if (snapshot.resources.processCpuTimeDeltaNanoseconds)
    resourceValues["process_cpu_time_delta_ns"] =
        *snapshot.resources.processCpuTimeDeltaNanoseconds;
  else
    resourceValues["process_cpu_time_delta_ns"] = nullptr;
  resourceValues["resource_observation_scope"] = "process";
  resourceValues["allocated_memory_bytes"] =
      snapshot.resources.allocatedMemoryBytes;
  if (snapshot.resources.peakResidentMemoryBytes)
    resourceValues["peak_resident_memory_bytes"] =
        *snapshot.resources.peakResidentMemoryBytes;
  else
    resourceValues["peak_resident_memory_bytes"] = nullptr;

  llvm::json::Object payload;
  payload["stage"] = systemPnrInterruptionStageSpelling(snapshot.stage);
  payload["frontier"] = std::move(frontier);
  payload["best_selected_rank"] = snapshot.bestSelectedRank
                                      ? llvm::json::Value(std::move(rank))
                                      : llvm::json::Value(nullptr);
  payload["closure_residual"] = std::move(residual);
  payload["resources"] = std::move(resourceValues);
  return payload;
}

SystemPnrGenerationOutcome
interruptedOutcome(SystemPnrInterruptionStage stage,
                   std::optional<std::uint32_t> restartOrdinal,
                   SystemPnrGenerationAccounting accounting,
                   std::vector<ArtifactRootReference> candidates,
                   const SystemInterruptionBestProjection &best,
                   const ExecutionResourceTracker &resources) {
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  SystemPnrInterruptionSnapshot snapshot = projectInterruptionSnapshot(
      stage, restartOrdinal, accounting, candidates.size(), best, resources);
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "invocation";
        fields["closure_status"] = "cancelled_or_timeout";
        fields["interruption"] = interruptionPayload(snapshot);
      });
  emitInvocationAccounting(accounting,
                           mapping_debug::ClosureStatus::CancelledOrTimeout,
                           candidates.size());
  return InterruptedSystemPnrGeneration{std::move(candidates), accounting,
                                        std::move(snapshot)};
}

} // namespace

llvm::StringRef
systemPnrInterruptionStageSpelling(SystemPnrInterruptionStage stage) {
  switch (stage) {
  case SystemPnrInterruptionStage::InputAdmission:
    return "input_admission";
  case SystemPnrInterruptionStage::FrozenModelConstruction:
    return "frozen_model_construction";
  case SystemPnrInterruptionStage::CandidateInitialization:
    return "candidate_initialization";
  case SystemPnrInterruptionStage::Annealing:
    return "annealing";
  case SystemPnrInterruptionStage::FinalClosure:
    return "final_closure";
  case SystemPnrInterruptionStage::CandidateVerification:
    return "candidate_verification";
  case SystemPnrInterruptionStage::CandidateFinalization:
    return "candidate_finalization";
  }
  llvm_unreachable("unknown System PnR interruption stage");
}

llvm::Error
verifySystemPnrWorkAccounting(const SystemPnrGenerationAccounting &accounting,
                              bool requireClosedWork) {
  const std::array<std::pair<std::uint64_t, std::uint64_t>, 10> counters = {{
      {accounting.plannedSeedAttemptSlots, accounting.seedAttemptSlots},
      {accounting.plannedInitializerAssignmentAttempts,
       accounting.initializerAssignmentAttempts},
      {accounting.plannedEndpointExpansionSlots,
       accounting.endpointExpansionSlots},
      {accounting.plannedNegotiationIterationSlots,
       accounting.negotiationIterationSlots},
      {accounting.plannedCalibrationProposalSlots,
       accounting.calibrationProposalSlots},
      {accounting.plannedAnnealingBaseProposalSlots,
       accounting.annealingBaseProposalSlots},
      {accounting.plannedAnnealingMovableProposalSlots,
       accounting.annealingMovableProposalSlots},
      {accounting.plannedExactRepairRegionDecisions,
       accounting.exactRepairRegionDecisions},
      {accounting.plannedExactRepairSolverCalls,
       accounting.exactRepairSolverCalls},
      {accounting.plannedFinalClosureAttempts, accounting.finalClosureAttempts},
  }};
  for (const auto [planned, consumed] : counters) {
    if (consumed > planned)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "System PnR consumed work exceeds planned work");
    if (requireClosedWork && planned != consumed)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "System PnR completed with admitted work still live");
  }
  return llvm::Error::success();
}

/// One isolated fresh System restart slot. The slot performs initialization,
/// annealing, the strict final closure and candidate invariant verification;
/// draft materialization, finalization and publication stay in the ordinal
/// reduction so scheduling cannot reorder store effects.
struct SystemRestartResult final {
  enum class Kind : std::uint8_t {
    Candidate,
    Incomplete,
    ProvenInfeasible,
    Interrupted,
    Internal,
    /// The slot finished without a candidate or an incompleteness claim, such
    /// as a migration initializer fallback or a bounded-prefix early stop.
    Skipped,
  };
  Kind kind = Kind::Internal;
  bool completionGoalReached = false;
  SystemPnrGenerationAccounting accounting;
  SystemCandidateStateHandle candidate;
  SystemInterruptionBestProjection best;
  bool semanticLimit = false;
  std::string diagnostic;
  SystemPnrInterruptionStage stage =
      SystemPnrInterruptionStage::CandidateInitialization;
  InternalSystemPnrGenerationReason internalReason =
      InternalSystemPnrGenerationReason::CandidateInitialization;
  SystemPnrInfeasibilityProofKind proofKind =
      SystemPnrInfeasibilityProofKind::InitializerRelation;
};

llvm::Error mergeInterruptionBest(const MappingObjectiveProgram &program,
                                  const SystemInterruptionBestProjection &local,
                                  SystemInterruptionBestProjection &best) {
  if (!local.objective)
    return llvm::Error::success();
  bool selected = !best.violationValues;
  if (!best.objective) {
    selected = true;
  } else {
    auto comparison =
        program.compareSelectedRank(*local.objective, {}, *best.objective, {});
    if (!comparison)
      return comparison.takeError();
    selected = *comparison < 0;
  }
  if (!selected)
    return llvm::Error::success();
  best = local;
  return llvm::Error::success();
}

/// Adds one slot's counters into the invocation totals. The accounting struct
/// is a flat sequence of saturating u64 counters; the layout assertions keep
/// this mechanical accumulation in sync with the field list.
void accumulateRestartAccounting(const SystemPnrGenerationAccounting &slot,
                                 SystemPnrGenerationAccounting &total) {
  static_assert(std::is_standard_layout_v<SystemPnrGenerationAccounting> &&
                    std::is_trivially_copyable_v<SystemPnrGenerationAccounting>,
                "accounting must remain a flat counter block");
  static_assert(sizeof(SystemPnrGenerationAccounting) %
                        sizeof(std::uint64_t) ==
                    0,
                "accounting must remain u64 counters only");
  constexpr std::size_t counterCount =
      sizeof(SystemPnrGenerationAccounting) / sizeof(std::uint64_t);
  const auto *source = reinterpret_cast<const std::uint64_t *>(&slot);
  auto *destination = reinterpret_cast<std::uint64_t *>(&total);
  for (std::size_t counter = 0; counter != counterCount; ++counter)
    destination[counter] =
        source[counter] >
                std::numeric_limits<std::uint64_t>::max() -
                    destination[counter]
            ? std::numeric_limits<std::uint64_t>::max()
            : destination[counter] + source[counter];
}

SystemPnrGenerationOutcome
generateSystemMappingsImpl(const SystemPnrGenerationInputs &inputs) {
  const ExecutionResourceTracker resources;
  SystemPnrGenerationAccounting accounting;
  const PnrWorkLedgerView workLedger = canonicalWorkLedger(accounting);
  SystemInterruptionBestProjection interruptionBest;
  if (inputs.executionControl.stopRequested())
    return interruptedOutcome(SystemPnrInterruptionStage::InputAdmission,
                              std::nullopt, accounting, {}, interruptionBest,
                              resources);
  std::optional<SystemStaticContext> ownedStaticContext;
  const SystemStaticContext *staticContext = inputs.staticContext;
  if (!staticContext) {
    DerivedContextCacheAccess access;
    auto built = buildSystemStaticContext(inputs.fabric, &access);
    if (!built) {
      FreezeFailure failure = classifyFreezeFailure(built.takeError());
      switch (failure.kind) {
      case FreezeFailureKind::Invalid:
        return InvalidSystemPnrGeneration{
            InvalidSystemPnrGenerationReason::FrozenInput, accounting,
            std::move(failure.diagnostic)};
      case FreezeFailureKind::ProvenInfeasible:
        emitProvenInfeasibleFreeze("system_static_context", failure.diagnostic);
        return ProvenInfeasibleSystemMapping{
            accounting, std::move(failure.diagnostic),
            SystemPnrInfeasibilityProofKind::FrozenStaticContext};
      case FreezeFailureKind::Internal:
        return internal(
            InternalSystemPnrGenerationReason::FrozenModelConstruction,
            accounting, failure.diagnostic);
      }
    }
    ownedStaticContext.emplace(std::move(*built));
    staticContext = &*ownedStaticContext;
    emitSystemStaticContextStatistics(*staticContext,
                                      mapping_debug::Stage::SystemPnr,
                                      access.hits, access.misses);
  } else {
    if (llvm::Error error =
            revalidateSystemStaticContext(*staticContext, inputs.fabric)) {
      FreezeFailure failure = classifyFreezeFailure(std::move(error));
      return InvalidSystemPnrGeneration{
          InvalidSystemPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    }
    emitSystemStaticContextStatistics(*staticContext,
                                      mapping_debug::Stage::SystemPnr, 1, 0);
  }
  std::optional<SystemActiveContext> ownedActiveContext;
  const SystemActiveContext *activeContext = inputs.activeContext;
  if (!activeContext) {
    std::vector<ArtifactRootReference> spatialMappings(
        inputs.constraints.view().spatialMappingReferences().begin(),
        inputs.constraints.view().spatialMappingReferences().end());
    for (const SystemSearchBindingDomain &binding :
         inputs.searchDomain.bindings())
      if (std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
        for (const SystemSearchAtom &atom : binding.atoms)
          if (const auto *domain =
                  std::get_if<SystemHierarchicalGraphBindingDomain>(
                      &atom.domain))
            spatialMappings.insert(spatialMappings.end(),
                                   domain->compatibleSpatialMappings.begin(),
                                   domain->compatibleSpatialMappings.end());
    llvm::sort(spatialMappings, artifactRootReferenceLess);
    spatialMappings.erase(
        std::unique(spatialMappings.begin(), spatialMappings.end()),
        spatialMappings.end());
    DerivedContextCacheAccess access;
    auto built = buildSystemActiveContext(
        *staticContext, inputs.dataflow, inputs.fabric,
        inputs.physicalTimingProfiles, inputs.constraints, spatialMappings,
        inputs.store, &access);
    if (!built) {
      FreezeFailure failure = classifyFreezeFailure(built.takeError());
      return InvalidSystemPnrGeneration{
          InvalidSystemPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    }
    ownedActiveContext.emplace(std::move(*built));
    activeContext = &*ownedActiveContext;
    emitSystemActiveContextStatistics(*activeContext,
                                      mapping_debug::Stage::SystemPnr,
                                      access.hits, access.misses);
  } else {
    if (llvm::Error error = revalidateSystemActiveContext(
            *activeContext, *staticContext, inputs.dataflow, inputs.fabric,
            inputs.physicalTimingProfiles, inputs.constraints,
            activeContext->spatialMappings())) {
      FreezeFailure failure = classifyFreezeFailure(std::move(error));
      return InvalidSystemPnrGeneration{
          InvalidSystemPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    }
    emitSystemActiveContextStatistics(*activeContext,
                                      mapping_debug::Stage::SystemPnr, 1, 0);
  }
  if (const auto *topology = staticContext->topologyQualityDiagnostic())
    emitFabricTopologyQuality(*topology, mapping_debug::Stage::SystemPnr);
  auto problem = freezeSystemPnrProblem(
      inputs.dataflow, inputs.fabric, inputs.physicalTimingProfiles,
      inputs.searchDomain, inputs.config, inputs.constraints, inputs.store,
      staticContext, activeContext);
  if (!problem) {
    FreezeFailure failure = classifyFreezeFailure(problem.takeError());
    switch (failure.kind) {
    case FreezeFailureKind::Invalid:
      return InvalidSystemPnrGeneration{
          InvalidSystemPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    case FreezeFailureKind::ProvenInfeasible:
      emitProvenInfeasibleFreeze("system_active_problem", failure.diagnostic);
      return ProvenInfeasibleSystemMapping{
          accounting, std::move(failure.diagnostic),
          SystemPnrInfeasibilityProofKind::FrozenActiveProblem};
    case FreezeFailureKind::Internal:
      return internal(
          InternalSystemPnrGenerationReason::FrozenModelConstruction,
          accounting, failure.diagnostic);
    }
  }
  if (inputs.executionControl.stopRequested())
    return interruptedOutcome(
        SystemPnrInterruptionStage::FrozenModelConstruction, std::nullopt,
        accounting, {}, interruptionBest, resources);

  std::optional<SystemMappingMigrationProjection> migrationProjection;
  std::optional<ArtifactRootReference> rebasedMapping;
  if (inputs.migrationSeed && inputs.checkpointMigrationSeed)
    return InvalidSystemPnrGeneration{
        InvalidSystemPnrGenerationReason::FrozenInput, accounting,
        "System PnR received two migration seed sources"};
  if (inputs.migrationSeed && inputs.migrationSeed->reopenedRoots().empty()) {
    const auto &parent = inputs.migrationSeed->parentMapping();
    auto rebased = ::loom::mapping::rebaseSystemMapping(
        parent, inputs.fabric,
        parent.view().executionBindings().spatialMappingImports(),
        inputs.migrationSeed->correspondence().entities(),
        inputs.migrationSeed->correspondence().transferPatterns(),
        inputs.constraints.view(), inputs.store,
        &(*problem)->spatialMappingImports());
    if (rebased) {
      rebasedMapping = rebased->reference();
      ++accounting.migrationSeedAttemptSlots;
      ++accounting.migrationSeedPrepared;
      accounting.migrationPreservedThreadBindings =
          (*problem)->threadDecisions().size();
      accounting.migrationPreservedGraphBindings =
          (*problem)->graphDecisions().size();
      accounting.migrationPreservedServiceLegs =
          (*problem)->serviceLegs().size();
      accounting.migrationPreservedResourceUses =
          parent.view().resourceUses().size();
      if (llvm::Error error = workLedger.plan(PnrWorkKind::FinalClosureAttempt))
        return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                        accounting, std::move(error));
      if (llvm::Error error =
              workLedger.consume(PnrWorkKind::FinalClosureAttempt))
        return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                        accounting, std::move(error));
      ++accounting.finalVerificationAttempts;
      ++accounting.publicationSlots;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "system_mapping_full_rebase";
            fields["mapping"] =
                formatArtifactIdentityHex(rebased->reference().artifact);
            fields["preserved_thread_bindings"] =
                accounting.migrationPreservedThreadBindings;
            fields["preserved_graph_bindings"] =
                accounting.migrationPreservedGraphBindings;
            fields["preserved_service_legs"] =
                accounting.migrationPreservedServiceLegs;
            fields["preserved_resource_uses"] =
                accounting.migrationPreservedResourceUses;
          });
    } else {
      ++accounting.migrationSeedFallbacks;
      const std::string diagnostic = llvm::toString(rebased.takeError());
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "system_mapping_migration_fallback";
            fields["reason"] = systemMappingMigrationFallbackReasonSpelling(
                SystemMappingMigrationFallbackReason::ChildRebaseRejected);
            fields["diagnostic"] = diagnostic;
          });
    }
  }
  if (!rebasedMapping &&
      (inputs.migrationSeed || inputs.checkpointMigrationSeed)) {
    SystemMappingMigrationProjectionOutcome projected =
        inputs.migrationSeed
            ? projectSystemMappingMigrationSeed(*inputs.migrationSeed,
                                                **problem)
            : projectSystemMappingMigrationSeed(*inputs.checkpointMigrationSeed,
                                                **problem);
    if (const auto *fallback =
            std::get_if<SystemMappingMigrationFallback>(&projected)) {
      ++accounting.migrationSeedFallbacks;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "system_mapping_migration_fallback";
            fields["reason"] =
                systemMappingMigrationFallbackReasonSpelling(fallback->reason);
          });
    } else {
      migrationProjection =
          std::get<SystemMappingMigrationProjection>(std::move(projected));
      accounting.migrationPreservedThreadBindings =
          migrationProjection->preservedThreadBindings;
      accounting.migrationPreservedGraphBindings =
          migrationProjection->preservedGraphBindings;
      if (accounting.migrationPreservedThreadBindings >
              (*problem)->threadDecisions().size() ||
          accounting.migrationPreservedGraphBindings >
              (*problem)->graphDecisions().size())
        return internal(
            InternalSystemPnrGenerationReason::CandidateInitialization,
            accounting,
            "System migration preservation exceeds the child decision domain");
      accounting.migrationReopenedThreadBindings =
          (*problem)->threadDecisions().size() -
          accounting.migrationPreservedThreadBindings;
      accounting.migrationReopenedGraphBindings =
          (*problem)->graphDecisions().size() -
          accounting.migrationPreservedGraphBindings;
      accounting.migrationPreservedServiceLegs =
          migrationProjection->preservedServiceLegs;
      accounting.migrationReopenedServiceLegs =
          migrationProjection->reopenedServiceLegs;
      accounting.migrationNewServiceLegs =
          migrationProjection->preservedServiceLegs == 0 &&
                  migrationProjection->reopenedServiceLegs == 0
              ? (*problem)->serviceLegs().size()
              : migrationProjection->reopenedServiceLegs;
      accounting.migrationNewResourceUses =
          (*problem)->instructionUsePatternDomains().size() +
          (*problem)->consistencyUsePatternDomains().size();
    }
  }

  switch ((*problem)->progressBasis().kind) {
  case ::loom::mapping::MappingDataflowProgressBasisKind::Acyclic:
  case ::loom::mapping::MappingDataflowProgressBasisKind::InitializedFeedback:
    break;
  case ::loom::mapping::MappingDataflowProgressBasisKind::Cyclic:
    ::loom::mapping::emitMappingDataflowProgressBasisDiagnostic(
        (*problem)->progressBasis(), inputs.dataflow,
        mapping_debug::Stage::SystemPnr);
    emitInvocationAccounting(
        accounting, mapping_debug::ClosureStatus::ProofNotEstablished, 0);
    return IncompleteSystemPnrGeneration{
        IncompleteSystemPnrGenerationReason::ProofNotEstablished, accounting,
        "proof_not_established: cyclic System Dataflow progress basis requires "
        "a typed cycle-breaking proof",
        std::nullopt, std::nullopt};
  }

  const auto &search = inputs.config.policy().search;
  bool requireImportedCapacityClosure = false;
  if (search.completionGoal ==
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
    auto capacityFit = searchSystemImportedCapacity(*problem, workLedger);
    if (!capacityFit)
      return internal(
          InternalSystemPnrGenerationReason::CandidateInitialization,
          accounting, capacityFit.takeError());
    const std::uint64_t assignmentAttempts =
        std::visit([](const auto &value) { return value.assignmentAttempts; },
                   *capacityFit);
    if (const auto *pressure =
            std::get_if<SystemImportedCapacityPressure>(&*capacityFit)) {
      auto checkpoint = publishExecutionBindingCheckpoint(
          **problem, pressure->witness, pressure->checkpointChoices,
          inputs.store);
      if (!checkpoint)
        return internal(
            InternalSystemPnrGenerationReason::CandidateInitialization,
            accounting, checkpoint.takeError());
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "imported_spatial_capacity";
            fields["closure_status"] = "proof_not_established";
            fields["assignment_attempts"] = pressure->assignmentAttempts;
            fields["capacity_witness_namespace"] =
                pressure->witness.namespaceOrdinal;
            fields["capacity_witness_usage"] = pressure->witness.usage;
            fields["capacity_witness_capacity"] = pressure->witness.capacity;
          });
      emitInvocationAccounting(
          accounting, mapping_debug::ClosureStatus::ProofNotEstablished, 0);
      return IncompleteSystemPnrGeneration{
          IncompleteSystemPnrGenerationReason::ProofNotEstablished, accounting,
          "proof_not_established: every bounded execution binding retains "
          "imported Spatial capacity pressure",
          *pressure, std::move(*checkpoint)};
    }
    if (std::holds_alternative<SystemImportedCapacitySearchLimit>(
            *capacityFit)) {
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "imported_spatial_capacity_preflight";
            fields["closure_status"] = "semantic_limit_reached";
            fields["assignment_attempts"] = assignmentAttempts;
            fields["diagnostic"] =
                "capacity preflight reached its per-seed assignment bound";
          });
    }
    if (const auto *infeasible =
            std::get_if<SystemImportedCapacityRelationInfeasible>(
                &*capacityFit)) {
      emitProvenInfeasibleFreeze("imported_capacity_relation",
                                 infeasible->diagnostic);
      emitInvocationAccounting(
          accounting, mapping_debug::ClosureStatus::ProvenInfeasible, 0);
      return ProvenInfeasibleSystemMapping{
          accounting, infeasible->diagnostic,
          SystemPnrInfeasibilityProofKind::ImportedCapacityRelation};
    }
    requireImportedCapacityClosure = true;
  }
  std::vector<ArtifactRootReference> candidates;
  if (rebasedMapping)
    candidates.push_back(*rebasedMapping);
  if (rebasedMapping && inputs.config.policy().search.completionGoal ==
                            ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
    emitInvocationAccounting(accounting,
                             mapping_debug::ClosureStatus::SemanticLimitReached,
                             candidates.size());
    return GeneratedSystemMappings{
        std::move(candidates), PnrGenerationTermination::SemanticLimitReached,
        accounting};
  }
  bool semanticLimitReached = false;
  bool proofNotEstablished = false;
  std::string firstIncompleteDiagnostic;
  mlir::MLIRContext context;
  context.loadDialect<::mapping::MappingDialect>();

  const auto rememberIncomplete = [&](llvm::StringRef diagnostic,
                                      bool semanticLimit) {
    semanticLimitReached |= semanticLimit;
    proofNotEstablished |= !semanticLimit;
    if (firstIncompleteDiagnostic.empty())
      firstIncompleteDiagnostic = diagnostic.str();
  };

  const std::uint32_t freshCount = search.initializer.seedAttemptCount;
  const std::uint32_t globalAttemptOffset = migrationProjection ? 1 : 0;
  const bool firstVerifiedGoal =
      inputs.config.policy().search.completionGoal ==
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;

  const auto internalRestart =
      [](SystemRestartResult &result, InternalSystemPnrGenerationReason reason,
         const llvm::Twine &diagnostic) -> SystemRestartResult && {
    result.kind = SystemRestartResult::Kind::Internal;
    result.internalReason = reason;
    result.diagnostic = diagnostic.str();
    return std::move(result);
  };
  const auto interruptedRestart =
      [](SystemRestartResult &result,
         SystemPnrInterruptionStage stage) -> SystemRestartResult && {
    result.kind = SystemRestartResult::Kind::Interrupted;
    result.stage = stage;
    return std::move(result);
  };

  // Annealing, the strict final closure and candidate invariant verification
  // for one initialized slot. Store publication stays outside so slots can run
  // in parallel without reordering store effects.
  const auto computeRestartTail = [&](SystemCandidateStateHandle candidate,
                                      std::uint64_t annealingSeedOrdinal,
                                      const PnrWorkLedgerView &slotLedger,
                                      SystemRestartResult &result)
      -> SystemRestartResult && {
    if (inputs.executionControl.stopRequested())
      return interruptedRestart(result, SystemPnrInterruptionStage::Annealing);
    SystemAnnealingSearchScratch annealing;
    auto annealed = annealing.run(candidate, annealingSeedOrdinal,
                                  inputs.executionControl, slotLedger);
    if (!annealed)
      return internalRestart(result,
                             InternalSystemPnrGenerationReason::Annealing,
                             llvm::toString(annealed.takeError()));
    if (llvm::Error error = accumulateAnnealing(*annealed, result.accounting))
      return internalRestart(
          result, InternalSystemPnrGenerationReason::AccountingOverflow,
          llvm::toString(std::move(error)));
    result.completionGoalReached |= annealed->completionGoalReached;
    if (llvm::Error error =
            considerInterruptionCandidate(*candidate, result.best))
      return internalRestart(result,
                             InternalSystemPnrGenerationReason::Annealing,
                             llvm::toString(std::move(error)));
    if (annealed->interrupted)
      return interruptedRestart(result, SystemPnrInterruptionStage::Annealing);

    if (inputs.executionControl.stopRequested())
      return interruptedRestart(result,
                                SystemPnrInterruptionStage::FinalClosure);
    if (llvm::Error error = slotLedger.plan(PnrWorkKind::FinalClosureAttempt))
      return internalRestart(
          result, InternalSystemPnrGenerationReason::AccountingOverflow,
          llvm::toString(std::move(error)));
    auto currentObjective =
        candidate->problem().objectiveProgram().evaluate(*candidate);
    if (!currentObjective)
      return internalRestart(result,
                             InternalSystemPnrGenerationReason::FinalClosure,
                             llvm::toString(currentObjective.takeError()));
    SystemActionProbeAccounting closureWork;
    auto closed = probeSystemAction(
        candidate, *currentObjective,
        SystemMappingAction{
            SystemTransportRoutingAction{SystemGlobalRoutingAction{}}},
        closureWork, SystemActionExecutionContext::FinalClosure, slotLedger);
    bool transitionFailure = false;
    bool workLimit = false;
    bool upstreamReopen = false;
    std::string closureDiagnostic;
    const bool closureSucceeded = static_cast<bool>(closed);
    if (!closureSucceeded)
      llvm::handleAllErrors(
          closed.takeError(),
          [&](const SystemActionTransitionFailure &failure) {
            transitionFailure = true;
            workLimit =
                failure.kind() == SystemActionTransitionFailureKind::WorkLimit;
            upstreamReopen = failure.reopenWitness().has_value();
            closureDiagnostic = errorMessage(failure);
          },
          [&](const llvm::ErrorInfoBase &failure) {
            closureDiagnostic = errorMessage(failure);
          });
    if (closureSucceeded || transitionFailure)
      if (llvm::Error error =
              slotLedger.consume(PnrWorkKind::FinalClosureAttempt))
        return internalRestart(
            result, InternalSystemPnrGenerationReason::AccountingOverflow,
            llvm::toString(std::move(error)));
    if (inputs.executionControl.stopRequested())
      return interruptedRestart(result,
                                SystemPnrInterruptionStage::FinalClosure);
    if (!closureSucceeded) {
      if (workLimit || upstreamReopen) {
        result.kind = SystemRestartResult::Kind::Incomplete;
        result.semanticLimit = workLimit;
        result.diagnostic = std::move(closureDiagnostic);
        return std::move(result);
      }
      return internalRestart(result,
                             InternalSystemPnrGenerationReason::FinalClosure,
                             closureDiagnostic.empty()
                                 ? "final global Action lost its failure cause"
                                 : closureDiagnostic);
    }
    if (llvm::Error error = considerInterruptionCandidate(
            *closed->candidate, result.best, &closed->objective))
      return internalRestart(result,
                             InternalSystemPnrGenerationReason::FinalClosure,
                             llvm::toString(std::move(error)));
    candidate = std::move(closed->candidate);
    if (candidate->capacityOveruse() != 0) {
      result.kind = SystemRestartResult::Kind::Incomplete;
      result.semanticLimit = false;
      result.diagnostic =
          "strict final global Action retained full CapacityOveruse";
      return std::move(result);
    }

    if (inputs.executionControl.stopRequested())
      return interruptedRestart(
          result, SystemPnrInterruptionStage::CandidateVerification);
    if (llvm::Error error = candidate->verify())
      return internalRestart(
          result, InternalSystemPnrGenerationReason::CandidateVerification,
          llvm::toString(std::move(error)));
    ++result.accounting.finalVerificationAttempts;
    result.kind = SystemRestartResult::Kind::Candidate;
    result.candidate = std::move(candidate);
    return std::move(result);
  };

  const auto runFreshRestart =
      [&](std::uint32_t freshAttempt) -> SystemRestartResult {
    SystemRestartResult result;
    const PnrWorkLedgerView slotLedger = canonicalWorkLedger(result.accounting);
    if (inputs.executionControl.stopRequested())
      return interruptedRestart(
          result, SystemPnrInterruptionStage::CandidateInitialization);
    if (llvm::Error error = slotLedger.plan(PnrWorkKind::SeedAttempt))
      return internalRestart(
          result, InternalSystemPnrGenerationReason::AccountingOverflow,
          llvm::toString(std::move(error)));
    auto initialized =
        requireImportedCapacityClosure
            ? initializeSystemCandidateAttemptWithImportedCapacityClosure(
                  *problem, freshAttempt, slotLedger)
            : initializeSystemCandidateAttempt(*problem, freshAttempt,
                                               slotLedger);
    if (!initialized) {
      InitializationFailure failure =
          classifyInitializationFailure(initialized.takeError());
      if (failure.kind != SystemCandidateInitializationFailureKind::Internal)
        if (llvm::Error error = slotLedger.consume(PnrWorkKind::SeedAttempt))
          return internalRestart(
              result, InternalSystemPnrGenerationReason::AccountingOverflow,
              llvm::toString(std::move(error)));
      if (inputs.executionControl.stopRequested())
        return interruptedRestart(
            result, SystemPnrInterruptionStage::CandidateInitialization);
      switch (failure.kind) {
      case SystemCandidateInitializationFailureKind::ProvenInfeasible:
        result.kind = SystemRestartResult::Kind::ProvenInfeasible;
        result.diagnostic = std::move(failure.diagnostic);
        result.proofKind = SystemPnrInfeasibilityProofKind::InitializerRelation;
        return result;
      case SystemCandidateInitializationFailureKind::SemanticLimitReached:
        result.kind = SystemRestartResult::Kind::Incomplete;
        result.semanticLimit = true;
        result.diagnostic = std::move(failure.diagnostic);
        return result;
      case SystemCandidateInitializationFailureKind::Internal:
        return internalRestart(
            result, InternalSystemPnrGenerationReason::CandidateInitialization,
            failure.diagnostic);
      }
    }
    if (llvm::Error error = slotLedger.consume(PnrWorkKind::SeedAttempt))
      return internalRestart(
          result, InternalSystemPnrGenerationReason::AccountingOverflow,
          llvm::toString(std::move(error)));
    ++result.accounting.preparedSeeds;
    SystemCandidateStateHandle candidate = std::move(initialized->state);
    if (llvm::Error error =
            considerInterruptionCandidate(*candidate, result.best))
      return internalRestart(
          result, InternalSystemPnrGenerationReason::CandidateInitialization,
          llvm::toString(std::move(error)));
    return computeRestartTail(std::move(candidate), freshAttempt, slotLedger,
                              result);
  };

  // Serial ordinal reduction: totals, best-projection merge, incompleteness
  // classification and publication all follow canonical attempt order.
  const auto reduceRestart = [&](SystemRestartResult restart,
                                 std::uint32_t globalAttempt)
      -> std::optional<SystemPnrGenerationOutcome> {
    accumulateRestartAccounting(restart.accounting, accounting);
    if (llvm::Error error = mergeInterruptionBest(
            (*problem)->objectiveProgram(), restart.best, interruptionBest))
      return internal(InternalSystemPnrGenerationReason::CandidateVerification,
                      accounting, std::move(error));
    semanticLimitReached |= restart.completionGoalReached;
    switch (restart.kind) {
    case SystemRestartResult::Kind::Internal:
      return internal(restart.internalReason, accounting, restart.diagnostic);
    case SystemRestartResult::Kind::Interrupted:
      return interruptedOutcome(restart.stage, globalAttempt, accounting,
                                std::move(candidates), interruptionBest,
                                resources);
    case SystemRestartResult::Kind::ProvenInfeasible:
      if (candidates.empty()) {
        emitInvocationAccounting(
            accounting, mapping_debug::ClosureStatus::ProvenInfeasible, 0);
        return SystemPnrGenerationOutcome{ProvenInfeasibleSystemMapping{
            accounting, std::move(restart.diagnostic), restart.proofKind}};
      }
      return internal(
          InternalSystemPnrGenerationReason::CandidateInitialization,
          accounting,
          "an initializer proved infeasibility after a verified candidate "
          "was published");
    case SystemRestartResult::Kind::Incomplete:
      rememberIncomplete(restart.diagnostic, restart.semanticLimit);
      return std::nullopt;
    case SystemRestartResult::Kind::Skipped:
      return std::nullopt;
    case SystemRestartResult::Kind::Candidate:
      break;
    }
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateFinalization, globalAttempt,
          accounting, std::move(candidates), interruptionBest, resources);
    auto draft = materializeSystemCandidateDraft(*restart.candidate, context);
    if (!draft)
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, draft.takeError());
    auto root = mlir::cast<::mapping::SystemOp>(draft->get());
    ++accounting.publicationSlots;
    auto finalized = ::loom::mapping::finalizeSystemMapping(
        root, inputs.dataflow, inputs.fabric, inputs.constraints.view(),
        inputs.store, &restart.candidate->problem().spatialMappingImports(),
        inputs.executionControl);
    if (!finalized) {
      std::optional<std::string> incompleteDiagnostic;
      bool interrupted = false;
      llvm::Error remaining = llvm::handleErrors(
          finalized.takeError(),
          [&](const ::loom::mapping::SystemMappingIncompleteError &error) {
            if (error.reason() ==
                ::loom::mapping::SystemMappingIncompleteReason::
                    CancelledOrTimeout)
              interrupted = true;
            else
              incompleteDiagnostic = error.diagnostic().str();
          },
          [&](const ::loom::mapping::SystemMappingRejectedError &error) {
            incompleteDiagnostic = error.diagnostic().str();
          });
      if (interrupted) {
        if (remaining)
          return internal(
              InternalSystemPnrGenerationReason::CandidateFinalization,
              accounting, std::move(remaining));
        return interruptedOutcome(
            SystemPnrInterruptionStage::CandidateFinalization, globalAttempt,
            accounting, std::move(candidates), interruptionBest, resources);
      }
      if (incompleteDiagnostic) {
        if (remaining)
          return internal(
              InternalSystemPnrGenerationReason::CandidateFinalization,
              accounting, std::move(remaining));
        rememberIncomplete(*incompleteDiagnostic, false);
        return std::nullopt;
      }
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, std::move(remaining));
    }
    ++accounting.finalizedRestarts;
    candidates.push_back(finalized->reference());
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateFinalization, globalAttempt,
          accounting, std::move(candidates), interruptionBest, resources);
    if (firstVerifiedGoal)
      semanticLimitReached = true;
    return std::nullopt;
  };

  bool stopFresh = false;
  if (migrationProjection) {
    // The migration slot runs serially before the fresh slots: its
    // direct-publication trial and its annealed candidate share one state, and
    // publication must stay in canonical order.
    SystemRestartResult slot;
    const PnrWorkLedgerView slotLedger = canonicalWorkLedger(slot.accounting);
    bool tailPending = false;
    SystemCandidateStateHandle migrationCandidate;
    if (inputs.executionControl.stopRequested()) {
      interruptedRestart(slot,
                         SystemPnrInterruptionStage::CandidateInitialization);
    } else if (llvm::Error error = slotLedger.plan(PnrWorkKind::SeedAttempt)) {
      internalRestart(slot,
                      InternalSystemPnrGenerationReason::AccountingOverflow,
                      llvm::toString(std::move(error)));
    } else {
      ++slot.accounting.migrationSeedAttemptSlots;
      auto initialized =
          migrationProjection->releasedChoices.empty()
              ? (migrationProjection->routeSeed
                     ? initializeSystemCandidateWithFixedChoicesAndRoutes(
                           *problem, migrationProjection->fixedChoices,
                           *migrationProjection->routeSeed, slotLedger)
                     : initializeSystemCandidateWithFixedChoices(
                           *problem, migrationProjection->fixedChoices,
                           slotLedger))
              : initializeSystemCandidateWithReleasedChoicesAndImportedCapacityClosure(
                    *problem, migrationProjection->fixedChoices,
                    migrationProjection->releasedChoices, slotLedger);
      if (!initialized) {
        InitializationFailure failure =
            classifyInitializationFailure(initialized.takeError());
        if (llvm::Error error = slotLedger.consume(PnrWorkKind::SeedAttempt)) {
          internalRestart(slot,
                          InternalSystemPnrGenerationReason::AccountingOverflow,
                          llvm::toString(std::move(error)));
        } else if (inputs.executionControl.stopRequested()) {
          interruptedRestart(
              slot, SystemPnrInterruptionStage::CandidateInitialization);
        } else {
          ++slot.accounting.migrationSeedFallbacks;
          mapping_debug::emit(
              mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
              mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
                fields["operation"] = "system_mapping_migration_fallback";
                fields["reason"] = systemMappingMigrationFallbackReasonSpelling(
                    SystemMappingMigrationFallbackReason::
                        ChildInitializerRejected);
                fields["diagnostic"] = failure.diagnostic;
              });
          slot.kind = SystemRestartResult::Kind::Skipped;
        }
      } else if (llvm::Error error =
                     slotLedger.consume(PnrWorkKind::SeedAttempt)) {
        internalRestart(slot,
                        InternalSystemPnrGenerationReason::AccountingOverflow,
                        llvm::toString(std::move(error)));
      } else {
        ++slot.accounting.preparedSeeds;
        ++slot.accounting.migrationSeedPrepared;
        migrationCandidate = std::move(initialized->state);
        if (llvm::Error error = considerInterruptionCandidate(
                *migrationCandidate, slot.best)) {
          internalRestart(
              slot, InternalSystemPnrGenerationReason::CandidateInitialization,
              llvm::toString(std::move(error)));
        } else {
          llvm::Error directVerification = migrationCandidate->verify();
          if (!directVerification) {
            ++slot.accounting.finalVerificationAttempts;
            auto directDraft =
                materializeSystemCandidateDraft(*migrationCandidate, context);
            if (directDraft) {
              ++slot.accounting.publicationSlots;
              auto directFinalized = ::loom::mapping::finalizeSystemMapping(
                  mlir::cast<::mapping::SystemOp>(directDraft->get()),
                  inputs.dataflow, inputs.fabric, inputs.constraints.view(),
                  inputs.store,
                  &migrationCandidate->problem().spatialMappingImports(),
                  inputs.executionControl);
              if (directFinalized) {
                ++slot.accounting.finalizedRestarts;
                candidates.push_back(directFinalized->reference());
                mapping_debug::emit(
                    mapping_debug::Level::Summary,
                    mapping_debug::Stage::SystemPnr,
                    mapping_debug::Event::Candidate,
                    [&](llvm::json::Object &fields) {
                      fields["operation"] =
                          "system_mapping_migration_direct_publication";
                      fields["mapping"] = formatArtifactIdentityHex(
                          directFinalized->reference().artifact);
                      fields["released_choice_count"] =
                          migrationProjection->releasedChoices.size();
                    });
                if (firstVerifiedGoal) {
                  slot.kind = SystemRestartResult::Kind::Skipped;
                  slot.completionGoalReached = true;
                  stopFresh = true;
                }
              } else {
                llvm::consumeError(directFinalized.takeError());
              }
            } else {
              llvm::consumeError(directDraft.takeError());
            }
          } else {
            llvm::consumeError(std::move(directVerification));
          }
          if (!stopFresh)
            tailPending = true;
        }
      }
    }
    if (tailPending)
      computeRestartTail(std::move(migrationCandidate), 0, slotLedger, slot);
    if (auto outcome = reduceRestart(std::move(slot), 0))
      return std::move(*outcome);
  }

  if (!stopFresh) {
    if (firstVerifiedGoal) {
      for (std::uint32_t fresh = 0; fresh != freshCount; ++fresh) {
        const std::size_t publishedBefore = candidates.size();
        if (auto outcome = reduceRestart(runFreshRestart(fresh),
                                         globalAttemptOffset + fresh))
          return std::move(*outcome);
        if (candidates.size() != publishedBefore)
          break;
      }
    } else {
      std::vector<SystemRestartResult> results(freshCount);
      const std::uint32_t workerCount = std::max<std::uint32_t>(
          1, std::min<std::uint32_t>(inputs.candidateWorkerCount,
                                     freshCount == 0 ? 1 : freshCount));
      if (workerCount <= 1) {
        for (std::uint32_t fresh = 0; fresh != freshCount; ++fresh)
          results[fresh] = runFreshRestart(fresh);
      } else {
        llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
            static_cast<unsigned>(workerCount)));
        std::atomic_uint32_t nextRestart{0};
        for (std::uint32_t worker = 0; worker != workerCount; ++worker)
          pool.async([&] {
            while (true) {
              const std::uint32_t fresh =
                  nextRestart.fetch_add(1, std::memory_order_relaxed);
              if (fresh >= freshCount)
                break;
              results[fresh] = runFreshRestart(fresh);
            }
          });
        pool.wait();
      }
      for (std::uint32_t fresh = 0; fresh != freshCount; ++fresh)
        if (auto outcome = reduceRestart(std::move(results[fresh]),
                                         globalAttemptOffset + fresh))
          return std::move(*outcome);
    }
  }

  if (inputs.executionControl.stopRequested())
    return interruptedOutcome(SystemPnrInterruptionStage::CandidateFinalization,
                              std::nullopt, accounting, std::move(candidates),
                              interruptionBest, resources);

  if (!candidates.empty()) {
    llvm::sort(candidates, artifactRootReferenceLess);
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    const mapping_debug::ClosureStatus closureStatus =
        proofNotEstablished ? mapping_debug::ClosureStatus::ProofNotEstablished
        : semanticLimitReached
            ? mapping_debug::ClosureStatus::SemanticLimitReached
            : mapping_debug::ClosureStatus::Closed;
    emitInvocationAccounting(accounting, closureStatus, candidates.size());
    return GeneratedSystemMappings{
        std::move(candidates),
        proofNotEstablished ? PnrGenerationTermination::ProofNotEstablished
        : semanticLimitReached
            ? PnrGenerationTermination::SemanticLimitReached
            : PnrGenerationTermination::FixedAttemptsCompleted,
        accounting};
  }
  emitInvocationAccounting(
      accounting,
      semanticLimitReached ? mapping_debug::ClosureStatus::SemanticLimitReached
                           : mapping_debug::ClosureStatus::ProofNotEstablished,
      0);
  return IncompleteSystemPnrGeneration{
      semanticLimitReached
          ? IncompleteSystemPnrGenerationReason::SemanticLimitReached
          : IncompleteSystemPnrGenerationReason::ProofNotEstablished,
      accounting,
      firstIncompleteDiagnostic.empty()
          ? "no fixed System restart reached independent final verification"
          : std::move(firstIncompleteDiagnostic),
      std::nullopt, std::nullopt};
}

SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs) {
  SystemPnrGenerationOutcome outcome = generateSystemMappingsImpl(inputs);
  const SystemPnrGenerationAccounting &accounting = std::visit(
      [](const auto &value) -> const SystemPnrGenerationAccounting & {
        return value.accounting;
      },
      outcome);
  const bool requireClosedWork =
      !std::holds_alternative<InterruptedSystemPnrGeneration>(outcome) &&
      !std::holds_alternative<InternalSystemPnrGeneration>(outcome);
  if (llvm::Error error =
          verifySystemPnrWorkAccounting(accounting, requireClosedWork))
    return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                    accounting, std::move(error));
  return outcome;
}

} // namespace loom::pnr
