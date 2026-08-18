#include "PnR/System/SystemPnrGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/MappingDebugLog.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/FabricTopologyQualityDiagnostic.h"
#include "PnR/System/SystemActionExecutor.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
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

llvm::Error accumulateInitialization(const InitializationFailure &source,
                                     SystemPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.assignmentAttempts,
                                     target.initializerAssignmentAttempts,
                                     "initializer assignment attempts"))
    return error;
  if (llvm::Error error =
          checkedAdd(source.endpointExpansions, target.endpointExpansionSlots,
                     "initializer endpoint expansions"))
    return error;
  return checkedAdd(source.negotiationIterations,
                    target.negotiationIterationSlots,
                    "initializer negotiation iterations");
}

llvm::Error accumulateInitialization(const InitializedSystemCandidate &source,
                                     SystemPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.assignmentAttempts,
                                     target.initializerAssignmentAttempts,
                                     "initializer assignment attempts"))
    return error;
  if (llvm::Error error =
          checkedAdd(source.endpointExpansions, target.endpointExpansionSlots,
                     "initializer endpoint expansions"))
    return error;
  return checkedAdd(source.negotiationIterations,
                    target.negotiationIterationSlots,
                    "initializer negotiation iterations");
}

llvm::Error accumulateAnnealing(const SystemAnnealingStatistics &source,
                                SystemPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.calibrationProposalSlots,
                                     target.calibrationProposalSlots,
                                     "calibration proposal slots"))
    return error;
  if (llvm::Error error = checkedAdd(source.annealingBaseProposalSlots,
                                     target.annealingBaseProposalSlots,
                                     "base annealing proposal slots"))
    return error;
  if (llvm::Error error =
          checkedAdd(source.annealingMovableProposalSlots,
                     target.annealingMovableProposalSlots,
                     "movable-decision annealing proposal slots"))
    return error;
  if (llvm::Error error = checkedAdd(source.assignmentAttempts,
                                     target.initializerAssignmentAttempts,
                                     "annealing assignment attempts"))
    return error;
  if (llvm::Error error =
          checkedAdd(source.endpointExpansions, target.endpointExpansionSlots,
                     "annealing endpoint expansions"))
    return error;
  if (llvm::Error error = checkedAdd(source.negotiationIterations,
                                     target.negotiationIterationSlots,
                                     "annealing negotiation iterations"))
    return error;
  if (llvm::Error error = checkedAdd(source.mutationOracleVerificationCount,
                                     target.mutationOracleVerificationAttempts,
                                     "mutation oracle verification attempts"))
    return error;
  return checkedAdd(source.acceptedActionCount, target.annealingAcceptedActions,
                    "annealing accepted Actions");
}

llvm::Error accumulateActionProbe(const SystemActionProbeAccounting &source,
                                  SystemPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.assignmentAttempts,
                                     target.initializerAssignmentAttempts,
                                     "Action assignment attempts"))
    return error;
  if (llvm::Error error =
          checkedAdd(source.endpointExpansions, target.endpointExpansionSlots,
                     "Action endpoint expansions"))
    return error;
  return checkedAdd(source.negotiationIterations,
                    target.negotiationIterationSlots,
                    "Action negotiation iterations");
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
        fields["seed_attempt_slots"] = accounting.seedAttemptSlots;
        fields["prepared_seeds"] = accounting.preparedSeeds;
        fields["initializer_assignment_attempts"] =
            accounting.initializerAssignmentAttempts;
        fields["endpoint_expansion_slots"] = accounting.endpointExpansionSlots;
        fields["negotiation_iteration_slots"] =
            accounting.negotiationIterationSlots;
        fields["calibration_proposal_slots"] =
            accounting.calibrationProposalSlots;
        fields["annealing_base_proposal_slots"] =
            accounting.annealingBaseProposalSlots;
        fields["annealing_movable_proposal_slots"] =
            accounting.annealingMovableProposalSlots;
        fields["annealing_accepted_actions"] =
            accounting.annealingAcceptedActions;
        fields["mutation_oracle_verification_attempts"] =
            accounting.mutationOracleVerificationAttempts;
        fields["exact_repair_invocations"] = accounting.exactRepairInvocations;
        fields["exact_repair_region_decisions"] =
            accounting.exactRepairRegionDecisions;
        fields["exact_repair_solver_calls"] = accounting.exactRepairSolverCalls;
        fields["final_closure_attempts"] = accounting.finalClosureAttempts;
        fields["final_verification_attempts"] =
            accounting.finalVerificationAttempts;
        fields["finalized_restarts"] = accounting.finalizedRestarts;
        fields["publication_slots"] = accounting.publicationSlots;
      });
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

SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs) {
  const ExecutionResourceTracker resources;
  SystemPnrGenerationAccounting accounting;
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
        return ProvenInfeasibleSystemMapping{accounting,
                                             std::move(failure.diagnostic)};
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
      return ProvenInfeasibleSystemMapping{accounting,
                                           std::move(failure.diagnostic)};
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
        "a typed cycle-breaking proof"};
  }

  const auto &search = inputs.config.policy().search;
  SystemAnnealingSearchScratch annealing;
  std::vector<ArtifactRootReference> candidates;
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

  for (std::uint32_t attempt = 0;
       attempt != search.initializer.seedAttemptCount; ++attempt) {
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateInitialization, attempt,
          accounting, std::move(candidates), interruptionBest, resources);
    ++accounting.seedAttemptSlots;
    auto initialized = initializeSystemCandidateAttempt(*problem, attempt);
    if (!initialized) {
      InitializationFailure failure =
          classifyInitializationFailure(initialized.takeError());
      if (llvm::Error error = accumulateInitialization(failure, accounting))
        return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                        accounting, std::move(error));
      if (inputs.executionControl.stopRequested())
        return interruptedOutcome(
            SystemPnrInterruptionStage::CandidateInitialization, attempt,
            accounting, std::move(candidates), interruptionBest, resources);
      switch (failure.kind) {
      case SystemCandidateInitializationFailureKind::ProvenInfeasible:
        if (candidates.empty()) {
          emitInvocationAccounting(
              accounting, mapping_debug::ClosureStatus::ProvenInfeasible, 0);
          return ProvenInfeasibleSystemMapping{accounting,
                                               std::move(failure.diagnostic)};
        }
        return internal(
            InternalSystemPnrGenerationReason::CandidateInitialization,
            accounting,
            "an initializer proved infeasibility after a verified candidate "
            "was published");
      case SystemCandidateInitializationFailureKind::SemanticLimitReached:
        rememberIncomplete(failure.diagnostic, true);
        continue;
      case SystemCandidateInitializationFailureKind::Internal:
        return internal(
            InternalSystemPnrGenerationReason::CandidateInitialization,
            accounting, failure.diagnostic);
      }
    }
    if (llvm::Error error = accumulateInitialization(*initialized, accounting))
      return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                      accounting, std::move(error));

    ++accounting.preparedSeeds;
    SystemCandidateStateHandle candidate = std::move(initialized->state);
    if (llvm::Error error =
            considerInterruptionCandidate(*candidate, interruptionBest))
      return internal(
          InternalSystemPnrGenerationReason::CandidateInitialization,
          accounting, std::move(error));
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(SystemPnrInterruptionStage::Annealing, attempt,
                                accounting, std::move(candidates),
                                interruptionBest, resources);
    auto annealed = annealing.run(candidate, attempt, inputs.executionControl);
    if (!annealed)
      return internal(InternalSystemPnrGenerationReason::Annealing, accounting,
                      annealed.takeError());
    if (llvm::Error error = accumulateAnnealing(*annealed, accounting))
      return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                      accounting, std::move(error));
    semanticLimitReached |= annealed->completionGoalReached;
    if (llvm::Error error =
            considerInterruptionCandidate(*candidate, interruptionBest))
      return internal(InternalSystemPnrGenerationReason::Annealing, accounting,
                      std::move(error));
    if (annealed->interrupted)
      return interruptedOutcome(SystemPnrInterruptionStage::Annealing, attempt,
                                accounting, std::move(candidates),
                                interruptionBest, resources);

    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(SystemPnrInterruptionStage::FinalClosure,
                                attempt, accounting, std::move(candidates),
                                interruptionBest, resources);
    ++accounting.finalClosureAttempts;
    auto currentObjective =
        candidate->problem().objectiveProgram().evaluate(*candidate);
    if (!currentObjective)
      return internal(InternalSystemPnrGenerationReason::FinalClosure,
                      accounting, currentObjective.takeError());
    SystemActionProbeAccounting closureWork;
    auto closed = probeSystemAction(
        candidate, *currentObjective,
        SystemMappingAction{
            SystemTransportRoutingAction{SystemGlobalRoutingAction{}}},
        closureWork, SystemActionExecutionContext::FinalClosure);
    if (llvm::Error error = accumulateActionProbe(closureWork, accounting))
      return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                      accounting, std::move(error));
    if (inputs.executionControl.stopRequested()) {
      if (!closed)
        llvm::consumeError(closed.takeError());
      return interruptedOutcome(SystemPnrInterruptionStage::FinalClosure,
                                attempt, accounting, std::move(candidates),
                                interruptionBest, resources);
    }
    if (!closed) {
      bool workLimit = false;
      bool upstreamReopen = false;
      std::string diagnostic;
      llvm::handleAllErrors(
          closed.takeError(),
          [&](const SystemActionTransitionFailure &failure) {
            workLimit =
                failure.kind() == SystemActionTransitionFailureKind::WorkLimit;
            upstreamReopen = failure.reopenWitness().has_value();
            diagnostic = errorMessage(failure);
          },
          [&](const llvm::ErrorInfoBase &failure) {
            diagnostic = errorMessage(failure);
          });
      if (workLimit) {
        rememberIncomplete(diagnostic, true);
        continue;
      }
      if (upstreamReopen) {
        rememberIncomplete(diagnostic, false);
        continue;
      }
      return internal(
          InternalSystemPnrGenerationReason::FinalClosure, accounting,
          diagnostic.empty() ? "final global Action lost its failure cause"
                             : diagnostic);
    }
    if (llvm::Error error = considerInterruptionCandidate(
            *closed->candidate, interruptionBest, &closed->objective))
      return internal(InternalSystemPnrGenerationReason::FinalClosure,
                      accounting, std::move(error));
    candidate = std::move(closed->candidate);
    if (candidate->capacityOveruse() != 0) {
      rememberIncomplete(
          "strict final global Action retained full CapacityOveruse", false);
      continue;
    }

    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateVerification, attempt,
          accounting, std::move(candidates), interruptionBest, resources);
    if (llvm::Error error = candidate->verify())
      return internal(InternalSystemPnrGenerationReason::CandidateVerification,
                      accounting, std::move(error));
    ++accounting.finalVerificationAttempts;
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateFinalization, attempt,
          accounting, std::move(candidates), interruptionBest, resources);
    auto draft = materializeSystemCandidateDraft(*candidate, context);
    if (!draft)
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, draft.takeError());
    auto root = mlir::cast<::mapping::SystemOp>(draft->get());
    ++accounting.publicationSlots;
    auto finalized = ::loom::mapping::finalizeSystemMapping(
        root, inputs.dataflow, inputs.fabric, inputs.constraints.view(),
        inputs.store, &candidate->problem().spatialMappingImports());
    if (!finalized) {
      std::optional<std::string> incompleteDiagnostic;
      llvm::Error remaining = llvm::handleErrors(
          finalized.takeError(),
          [&](const ::loom::mapping::SystemMappingIncompleteError &error) {
            incompleteDiagnostic = error.diagnostic().str();
          },
          [&](const ::loom::mapping::SystemMappingRejectedError &error) {
            incompleteDiagnostic = error.diagnostic().str();
          });
      if (incompleteDiagnostic) {
        if (remaining)
          return internal(
              InternalSystemPnrGenerationReason::CandidateFinalization,
              accounting, std::move(remaining));
        rememberIncomplete(*incompleteDiagnostic, false);
        continue;
      }
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, std::move(remaining));
    }
    ++accounting.finalizedRestarts;
    candidates.push_back(finalized->reference());
    if (inputs.executionControl.stopRequested())
      return interruptedOutcome(
          SystemPnrInterruptionStage::CandidateFinalization, attempt,
          accounting, std::move(candidates), interruptionBest, resources);
    if (inputs.config.policy().search.completionGoal ==
        ResolvedPnrCompletionGoal::FirstVerifiedCandidate) {
      semanticLimitReached = true;
      break;
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
          : std::move(firstIncompleteDiagnostic)};
}

} // namespace loom::pnr
