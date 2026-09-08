#include "PnR/SpatialPnrGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/MappingDebugLog.h"
#include "InitializerRelationSolver.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/FabricTopologyQualityDiagnostic.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialAnnealingSearch.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialGlobalRoutingClosure.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPnrWorkLedger.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialPnrExecution.h"
#include "SpatialProgressIndex.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

enum class AttemptFailureKind : std::uint8_t {
  ProvenInfeasible,
  SemanticLimit,
  Rejected,
  Internal,
  Interrupted,
};

struct AttemptFailure final {
  AttemptFailureKind kind = AttemptFailureKind::Internal;
  std::string diagnostic;
  std::optional<detail::InitializerRelationHallWitness> hallWitness;
};

void emitActiveHandshakeStatistics(
    const HandshakeActiveDemandStatistics &statistics, std::uint32_t attempt) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "spatial_active_handshake";
        fields["candidate_attempt"] = attempt;
        fields["cache_hits"] = 0;
        fields["cache_misses"] = statistics.constructionCount;
        fields["construction_count"] = statistics.constructionCount;
        fields["construction_time_ns"] = statistics.constructionNanoseconds;
        fields["retained_bytes"] = statistics.retainedBytes;
        fields["deterministic_work"] = statistics.deterministicWork;
        fields["active_fragment_count"] = statistics.activeFragmentCount;
        fields["materialized_node_count"] = statistics.materializedNodeCount;
        fields["materialized_arc_count"] = statistics.materializedArcCount;
        fields["fabric_unconditional_arc_count"] =
            statistics.fabricUnconditionalArcCount;
        fields["materialized_contribution_count"] =
            statistics.materializedContributionCount;
        fields["transaction_closure_count"] =
            statistics.transactionClosureCount;
        fields["transaction_inserted_arc_count"] =
            statistics.transactionInsertedArcCount;
        fields["transaction_removed_arc_count"] =
            statistics.transactionRemovedArcCount;
        fields["transaction_affected_node_count"] =
            statistics.transactionAffectedNodeCount;
        fields["transaction_affected_rank_span"] =
            statistics.transactionAffectedRankSpan;
        fields["cached_verification_count"] =
            statistics.cachedVerificationCount;
        fields["cold_verification_construction_count"] =
            statistics.coldVerificationConstructionCount;
        fields["cold_verification_construction_time_ns"] =
            statistics.coldVerificationConstructionNanoseconds;
      });
}

void emitInvocationAccounting(const SpatialPnrGenerationAccounting &accounting,
                              mapping_debug::ClosureStatus closureStatus,
                              std::uint64_t candidatePublications) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
        fields["statistics_kind"] = "spatial_pnr_invocation";
        fields["closure_status"] =
            mapping_debug::closureStatusSpelling(closureStatus);
        fields["candidate_publications"] = candidatePublications;
        fields["planned_seed_attempt_slots"] =
            accounting.plannedSeedAttemptSlots;
        fields["seed_attempt_slots"] = accounting.seedAttemptSlots;
        fields["prepared_seeds"] = accounting.preparedSeeds;
        fields["initializer_assignment_attempts"] =
            accounting.initializerAssignmentAttempts;
        fields["planned_initializer_assignment_attempts"] =
            accounting.plannedInitializerAssignmentAttempts;
        fields["planned_endpoint_expansion_slots"] =
            accounting.plannedEndpointExpansionSlots;
        fields["a_star_expansions"] = accounting.endpointExpansionSlots;
        fields["endpoint_expansion_slots"] = accounting.endpointExpansionSlots;
        fields["negotiation_iterations"] = accounting.negotiationIterationSlots;
        fields["planned_negotiation_iteration_slots"] =
            accounting.plannedNegotiationIterationSlots;
        fields["negotiation_iteration_slots"] =
            accounting.negotiationIterationSlots;
        fields["calibration_proposal_slots"] =
            accounting.calibrationProposalSlots;
        fields["planned_calibration_proposal_slots"] =
            accounting.plannedCalibrationProposalSlots;
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
        fields["exact_repair_invocations"] = accounting.exactRepairInvocations;
        fields["exact_repair_region_decisions"] =
            accounting.exactRepairRegionDecisions;
        fields["planned_exact_repair_region_decisions"] =
            accounting.plannedExactRepairRegionDecisions;
        fields["planned_exact_repair_solver_calls"] =
            accounting.plannedExactRepairSolverCalls;
        fields["exact_repair_solver_calls"] = accounting.exactRepairSolverCalls;
        fields["planned_local_transfer_adoption_probes"] =
            accounting.plannedLocalTransferAdoptionProbes;
        fields["local_transfer_adoption_probes"] =
            accounting.localTransferAdoptionProbes;
        fields["adopted_local_transfers"] = accounting.adoptedLocalTransfers;
        fields["planned_final_closure_attempts"] =
            accounting.plannedFinalClosureAttempts;
        fields["final_closure_attempts"] = accounting.finalClosureAttempts;
        fields["finalized_restarts"] = accounting.finalizedRestarts;
        fields["publication_slots"] = accounting.publicationSlots;
      });
}

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
  return stream.str();
}

FreezeFailure classifyFreezeFailure(llvm::Error error) {
  FreezeFailure result;
  llvm::handleAllErrors(
      std::move(error),
      [&](const SpatialPnrFreezeFailure &failure) {
        result.kind = failure.kind() == SpatialPnrFreezeFailureKind::Invalid
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

AttemptFailure classifyAttemptFailure(llvm::Error error) {
  AttemptFailure result;
  llvm::handleAllErrors(
      std::move(error),
      [&](const detail::InitializerRelationSolveFailure &failure) {
        switch (failure.kind()) {
        case detail::InitializerRelationSolveFailureKind::ProvenInfeasible:
          result.kind = AttemptFailureKind::ProvenInfeasible;
          break;
        case detail::InitializerRelationSolveFailureKind::FixedRootInfeasible:
          result.kind = AttemptFailureKind::Rejected;
          break;
        case detail::InitializerRelationSolveFailureKind::WorkLimit:
          result.kind = AttemptFailureKind::SemanticLimit;
          break;
        }
        result.diagnostic = errorMessage(failure);
        result.hallWitness = failure.hallWitness();
      },
      [&](const EndpointRouteSearchFailure &failure) {
        switch (failure.kind()) {
        case EndpointRouteSearchFailureKind::Unreachable:
          result.kind = AttemptFailureKind::Rejected;
          break;
        case EndpointRouteSearchFailureKind::WorkLimit:
          result.kind = AttemptFailureKind::SemanticLimit;
          break;
        case EndpointRouteSearchFailureKind::Invalid:
          result.kind = AttemptFailureKind::Internal;
          break;
        case EndpointRouteSearchFailureKind::ArithmeticOverflow:
          result.kind = AttemptFailureKind::Internal;
          break;
        }
        result.diagnostic = errorMessage(failure);
      },
      [&](const RoutingNegotiationError &failure) {
        result.kind = AttemptFailureKind::Internal;
        result.diagnostic = errorMessage(failure);
      },
      [&](const SpatialPathFinderClosureFailure &failure) {
        switch (failure.kind()) {
        case SpatialPathFinderClosureFailure::Kind::NonClosure:
        case SpatialPathFinderClosureFailure::Kind::NoProgress:
          result.kind = AttemptFailureKind::SemanticLimit;
          break;
        case SpatialPathFinderClosureFailure::Kind::Interrupted:
          result.kind = AttemptFailureKind::Interrupted;
          break;
        case SpatialPathFinderClosureFailure::Kind::RegionalLimit:
        case SpatialPathFinderClosureFailure::Kind::FixedTerminalCapacityCut:
        case SpatialPathFinderClosureFailure::Kind::
            SelectedCombinationalHandshakeCycle:
          result.kind = AttemptFailureKind::Rejected;
          break;
        }
        result.diagnostic = errorMessage(failure);
      },
      [&](const SpatialActionTransitionFailure &failure) {
        switch (failure.kind()) {
        case SpatialActionTransitionFailureKind::WorkLimit:
          result.kind = AttemptFailureKind::SemanticLimit;
          break;
        case SpatialActionTransitionFailureKind::Interrupted:
          result.kind = AttemptFailureKind::Interrupted;
          break;
        case SpatialActionTransitionFailureKind::IntrinsicInvalid:
          result.kind = AttemptFailureKind::Rejected;
          break;
        }
        result.diagnostic = errorMessage(failure);
      },
      [&](const SpatialGlobalRoutingClosureFailure &failure) {
        result.kind = AttemptFailureKind::Rejected;
        result.diagnostic = errorMessage(failure);
      },
      [&](const llvm::ErrorInfoBase &failure) {
        result.kind = AttemptFailureKind::Internal;
        result.diagnostic = errorMessage(failure);
      });
  return result;
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        "Spatial PnR accounting overflow: " + subject);
  target += amount;
  return llvm::Error::success();
}

SpatialPnrWorkLedgerView
canonicalWorkLedger(SpatialPnrGenerationAccounting &accounting) {
  std::array<SpatialPnrWorkCounterRef, spatialPnrWorkKindCount> counters{};
  const auto bind = [&](SpatialPnrWorkKind kind, std::uint64_t &planned,
                        std::uint64_t &consumed) {
    counters[static_cast<std::size_t>(kind)] = {&planned, &consumed};
  };
  bind(SpatialPnrWorkKind::SeedAttempt, accounting.plannedSeedAttemptSlots,
       accounting.seedAttemptSlots);
  bind(SpatialPnrWorkKind::InitializerAssignment,
       accounting.plannedInitializerAssignmentAttempts,
       accounting.initializerAssignmentAttempts);
  bind(SpatialPnrWorkKind::EndpointExpansion,
       accounting.plannedEndpointExpansionSlots,
       accounting.endpointExpansionSlots);
  bind(SpatialPnrWorkKind::NegotiationIteration,
       accounting.plannedNegotiationIterationSlots,
       accounting.negotiationIterationSlots);
  bind(SpatialPnrWorkKind::CalibrationProposal,
       accounting.plannedCalibrationProposalSlots,
       accounting.calibrationProposalSlots);
  bind(SpatialPnrWorkKind::AnnealingBaseProposal,
       accounting.plannedAnnealingBaseProposalSlots,
       accounting.annealingBaseProposalSlots);
  bind(SpatialPnrWorkKind::AnnealingMovableProposal,
       accounting.plannedAnnealingMovableProposalSlots,
       accounting.annealingMovableProposalSlots);
  bind(SpatialPnrWorkKind::ExactRepairRegionDecision,
       accounting.plannedExactRepairRegionDecisions,
       accounting.exactRepairRegionDecisions);
  bind(SpatialPnrWorkKind::ExactRepairSolverCall,
       accounting.plannedExactRepairSolverCalls,
       accounting.exactRepairSolverCalls);
  bind(SpatialPnrWorkKind::LocalTransferAdoptionProbe,
       accounting.plannedLocalTransferAdoptionProbes,
       accounting.localTransferAdoptionProbes);
  bind(SpatialPnrWorkKind::FinalClosureAttempt,
       accounting.plannedFinalClosureAttempts, accounting.finalClosureAttempts);
  return SpatialPnrWorkLedgerView(counters);
}

InternalSpatialPnrGeneration
internal(InternalSpatialPnrGenerationReason reason,
         const SpatialPnrGenerationAccounting &accounting,
         const llvm::Twine &diagnostic) {
  return {reason, accounting, diagnostic.str()};
}

InternalSpatialPnrGeneration
internal(InternalSpatialPnrGenerationReason reason,
         const SpatialPnrGenerationAccounting &accounting, llvm::Error error) {
  return internal(reason, accounting, llvm::toString(std::move(error)));
}

llvm::Error accumulateAnnealing(const SpatialAnnealingStatistics &source,
                                SpatialPnrGenerationAccounting &target) {
  if (llvm::Error error = checkedAdd(source.acceptedActionCount,
                                     target.annealingAcceptedActions,
                                     "annealing accepted Actions"))
    return error;
  return checkedAdd(source.adoptedLocalTransfers, target.adoptedLocalTransfers,
                    "adopted local transfers");
}

llvm::Expected<std::optional<ResolvedPnrViolationKind>>
firstFinalViolation(const SpatialCandidateState &candidate) {
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal) {
    const auto kind = static_cast<ResolvedPnrViolationKind>(ordinal);
    auto value = spatialMappingViolationValue(candidate, kind);
    if (!value)
      return value.takeError();
    if (*value != 0)
      return std::optional<ResolvedPnrViolationKind>(kind);
  }
  return std::optional<ResolvedPnrViolationKind>();
}

std::string violationDiagnostic(ResolvedPnrViolationKind kind) {
  const auto violations = mappingViolationDescriptors();
  return (llvm::Twine("candidate retained final Mapping violation ") +
          violations[static_cast<std::uint32_t>(kind)].spelling)
      .str();
}

enum class SpatialRestartDisposition : std::uint8_t {
  Candidate,
  ProvenInfeasible,
  Incomplete,
  Interrupted,
  Internal,
};

struct SpatialRestartCandidateSummary final {
  std::optional<dse::ObjectiveVector> objective;
  SpatialPnrClosureResidual residual;
};

struct SpatialFinalizedRestart final {
  ArtifactRootReference reference;
  SpatialRestartCandidateSummary summary;
};

struct SpatialRestartResult final {
  SpatialRestartDisposition disposition = SpatialRestartDisposition::Internal;
  SpatialPnrGenerationAccounting accounting;
  SpatialCandidateStateHandle candidate;
  bool semanticLimitReached = false;
  InternalSpatialPnrGenerationReason internalReason =
      InternalSpatialPnrGenerationReason::SeedConstruction;
  std::string diagnostic;
  SpatialPnrInterruptionStage interruptionStage =
      SpatialPnrInterruptionStage::SeedConstruction;
  std::optional<SpatialGraphBoundaryEndpointHallDeficit>
      graphBoundaryEndpointHall = std::nullopt;
  std::uint64_t workerScratchRetainedBytes = 0;
  std::optional<SpatialFinalizedRestart> finalized;
};

struct SpatialRestartScratch final {
  SpatialAnnealingSearchScratch annealing;
  SpatialExactRepairScratch repair;
  SpatialGlobalRoutingClosureScratch finalClosure;

  std::uint64_t retainedStorageBytes() const {
    std::uint64_t total = 0;
    const auto add = [&](std::size_t bytes) {
      std::uint64_t amount = 0;
      if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
        amount = bytes > std::numeric_limits<std::uint64_t>::max()
                     ? std::numeric_limits<std::uint64_t>::max()
                     : static_cast<std::uint64_t>(bytes);
      else
        amount = static_cast<std::uint64_t>(bytes);
      total = amount > std::numeric_limits<std::uint64_t>::max() - total
                  ? std::numeric_limits<std::uint64_t>::max()
                  : total + amount;
    };
    add(annealing.retainedStorageBytes());
    add(repair.retainedStorageBytes());
    add(finalClosure.retainedStorageBytes());
    return total;
  }
};

llvm::StringRef spelling(SpatialRestartDisposition disposition) {
  switch (disposition) {
  case SpatialRestartDisposition::Candidate:
    return "candidate";
  case SpatialRestartDisposition::ProvenInfeasible:
    return "proven_infeasible";
  case SpatialRestartDisposition::Incomplete:
    return "incomplete";
  case SpatialRestartDisposition::Interrupted:
    return "cancelled_or_timeout";
  case SpatialRestartDisposition::Internal:
    return "internal";
  }
  llvm_unreachable("unknown Spatial restart disposition");
}

llvm::StringRef spelling(InternalSpatialPnrGenerationReason reason) {
  switch (reason) {
  case InternalSpatialPnrGenerationReason::FrozenModelConstruction:
    return "frozen_model_construction";
  case InternalSpatialPnrGenerationReason::SeedConstruction:
    return "seed_construction";
  case InternalSpatialPnrGenerationReason::Annealing:
    return "annealing";
  case InternalSpatialPnrGenerationReason::ExactRepair:
    return "exact_repair";
  case InternalSpatialPnrGenerationReason::FinalClosure:
    return "final_closure";
  case InternalSpatialPnrGenerationReason::CandidateVerification:
    return "candidate_verification";
  case InternalSpatialPnrGenerationReason::CandidateFinalization:
    return "candidate_finalization";
  case InternalSpatialPnrGenerationReason::AccountingOverflow:
    return "accounting_overflow";
  }
  llvm_unreachable("unknown Spatial restart termination owner");
}

void emitRestartFailure(std::uint32_t ordinal,
                        const SpatialRestartResult &restart, bool contributesToResult) {
  if (restart.disposition == SpatialRestartDisposition::Candidate)
    return;
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "restart";
        fields["restart_ordinal"] = ordinal;
        fields["contributes_to_result"] = contributesToResult;
        fields["closure_status"] = spelling(restart.disposition);
        fields["termination_owner"] =
            restart.disposition == SpatialRestartDisposition::Interrupted
                ? spatialPnrInterruptionStageSpelling(restart.interruptionStage)
                : spelling(restart.internalReason);
        fields["semantic_limit_reached"] = restart.semanticLimitReached;
        fields["diagnostic"] = restart.diagnostic;
        fields["prepared_seeds"] = restart.accounting.preparedSeeds;
        fields["initializer_assignment_attempts"] =
            restart.accounting.initializerAssignmentAttempts;
        fields["endpoint_expansions"] =
            restart.accounting.endpointExpansionSlots;
        fields["negotiation_iterations"] =
            restart.accounting.negotiationIterationSlots;
        fields["annealing_accepted_actions"] =
            restart.accounting.annealingAcceptedActions;
        fields["exact_repair_invocations"] =
            restart.accounting.exactRepairInvocations;
        fields["exact_repair_region_decisions"] =
            restart.accounting.exactRepairRegionDecisions;
        fields["exact_repair_solver_calls"] =
            restart.accounting.exactRepairSolverCalls;
        fields["local_transfer_adoption_probes"] =
            restart.accounting.localTransferAdoptionProbes;
        fields["adopted_local_transfers"] =
            restart.accounting.adoptedLocalTransfers;
        fields["final_closure_attempts"] =
            restart.accounting.finalClosureAttempts;
      });
}

void preferPreparedRestart(const SpatialRestartResult &candidate,
                           const SpatialRestartResult *&selected) {
  if (!selected || (selected->accounting.preparedSeeds == 0 &&
                    candidate.accounting.preparedSeeds != 0))
    selected = &candidate;
}

llvm::Expected<std::optional<SpatialFifoCapacitySuggestion>>
projectFifoCapacityShortfall(const SpatialRestartResult *restart) {
  if (!restart || !restart->candidate)
    return std::optional<SpatialFifoCapacitySuggestion>();
  return projectSpatialFifoCapacitySuggestion(*restart->candidate);
}

SpatialRestartResult restartInternal(InternalSpatialPnrGenerationReason reason,
                                     SpatialPnrGenerationAccounting accounting,
                                     const llvm::Twine &diagnostic) {
  return {SpatialRestartDisposition::Internal,
          std::move(accounting),
          nullptr,
          false,
          reason,
          diagnostic.str()};
}

SpatialRestartResult restartInternal(InternalSpatialPnrGenerationReason reason,
                                     SpatialPnrGenerationAccounting accounting,
                                     llvm::Error error) {
  return restartInternal(reason, std::move(accounting),
                         llvm::toString(std::move(error)));
}

SpatialRestartResult
restartInterrupted(SpatialPnrInterruptionStage stage,
                   SpatialPnrGenerationAccounting accounting,
                   SpatialCandidateStateHandle candidate = nullptr) {
  SpatialRestartResult result;
  result.disposition = SpatialRestartDisposition::Interrupted;
  result.accounting = std::move(accounting);
  result.candidate = std::move(candidate);
  result.diagnostic = "execution control requested stop";
  result.interruptionStage = stage;
  return result;
}

/// Emits one place-and-route report per restart on scope exit: per-phase
/// elapsed time plus the final resource occupancy and congestion state of the
/// observed candidate. Restart failure diagnostics carry the typed outcome
/// separately; both join on the restart ordinal.
class SpatialRestartReporter final {
public:
  explicit SpatialRestartReporter(std::uint32_t attempt) : attempt_(attempt) {}
  SpatialRestartReporter(const SpatialRestartReporter &) = delete;
  SpatialRestartReporter &operator=(const SpatialRestartReporter &) = delete;

  ~SpatialRestartReporter() {
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
        mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
          fields["statistics_kind"] = "spatial_restart_report";
          fields["restart"] = attempt_;
          fields["seed_ns"] = seedNanoseconds;
          fields["annealing_ns"] = annealingNanoseconds;
          fields["exact_repair_ns"] = exactRepairNanoseconds;
          fields["final_closure_ns"] = finalClosureNanoseconds;
          fields["verification_ns"] = verificationNanoseconds;
          if (!captured_)
            return;
          for (auto &entry : *captured_)
            fields[entry.first] = std::move(entry.second);
        });
  }

  /// The terminal restart disposition moves the candidate handle into its
  /// result, so the reporter snapshots the candidate-derived fields while the
  /// candidate is still owned instead of reading a handle the caller may have
  /// already emptied.
  void capture(const SpatialCandidateState &candidate,
               const FrozenSpatialPnrProblem &problem) {
    llvm::json::Object fields;
    fields["unrouted_obligations"] = candidate.unroutedObligationCount();
    fields["hard_progress_violation"] = candidate.hardProgressViolation();
    fields["progress_proof_debt_witnesses"] =
        candidate.progressProofDebtWitnessCount();
    fields["progress_capacity_shortfall"] =
        candidate.progressCapacityShortfall();
    fields["progress_route_anchors"] = candidate.progressRouteAnchorCount();
    fields["runtime_counterexample_violations"] =
        candidate.runtimeCounterexampleViolation();
    fields["selected_handshake_violations"] =
        candidate.selectedHandshakeViolation();
    std::uint64_t registerFifoTransfers = 0;
    for (PnrIndex logicalNet = 0;
         logicalNet < problem.transfers().logicalNets().size(); ++logicalNet)
      registerFifoTransfers += candidate.usesRegisterFifo(logicalNet);
    fields["register_fifo_transfers"] = registerFifoTransfers;
    fields["shared_finite_buffer_conflicts"] =
        candidate.progress().sharedFiniteBufferConflictCount();
    fields["route_dependency_violations"] =
        candidate.progress().routeDependencyViolationCount();
    fields["transport_closure_violation"] =
        candidate.hasTransportClosureViolation();
    fields["atomic_capacity_overuse"] = candidate.atomicCapacityOveruse();
    fields["static_schedule_pressure"] = candidate.staticSchedulePressure();
    fields["route_capacity_overuse"] = candidate.routeCapacityOveruse();
    fields["tag_unassigned"] = candidate.tagUnassignedCount();
    fields["tag_conflicts"] = candidate.tagConflictCount();
    fields["tag_resident_overuse"] = candidate.tagResidentCapacityOveruse();
    fields["worst_route_arrival_delay_quanta"] =
        candidate.worstRouteArrivalDelayQuanta();
    fields["total_route_negative_slack_quanta"] =
        candidate.totalRouteNegativeSlackQuanta();
    const auto dimensions = problem.resources().capacityDimensions();
    std::uint64_t usedUnits = 0;
    std::uint64_t totalUnits = 0;
    std::uint64_t overusedDimensions = 0;
    std::uint64_t maximumOveruse = 0;
    for (PnrIndex dimension = 0; dimension < dimensions.size(); ++dimension) {
      usedUnits += candidate.routeCapacityUsageRaw(dimension);
      totalUnits += dimensions[dimension].capacity;
      const std::uint64_t overuse =
          candidate.routeCapacityOveruseRaw(dimension);
      overusedDimensions += overuse != 0;
      maximumOveruse = std::max(maximumOveruse, overuse);
    }
    fields["route_capacity_dimensions"] =
        static_cast<std::uint64_t>(dimensions.size());
    fields["route_capacity_used_units"] = usedUnits;
    fields["route_capacity_total_units"] = totalUnits;
    fields["route_capacity_overused_dimensions"] = overusedDimensions;
    fields["route_capacity_maximum_overuse"] = maximumOveruse;
    const auto contexts = problem.capacity().computeInstructionContextOveruse();
    llvm::DenseSet<PnrIndex> usedContexts;
    const std::size_t realizationCount =
        problem.realizations().computeRealizations().size();
    for (PnrIndex realization = 0; realization < realizationCount;
         ++realization)
      usedContexts.insert(
          candidate.computeBinding(realization).instructionContext);
    fields["compute_realizations"] =
        static_cast<std::uint64_t>(realizationCount);
    fields["compute_contexts_used"] =
        static_cast<std::uint64_t>(usedContexts.size());
    fields["compute_contexts_total"] =
        static_cast<std::uint64_t>(contexts.size());
    fields["finite_buffer_owners"] = static_cast<std::uint64_t>(
        problem.progressIndex().finiteBufferOwners().size());
    if (auto witnesses =
            candidate.progress().finiteBufferConflictWitnesses(candidate)) {
      std::uint64_t routeArcAnchors = 0;
      std::uint64_t attachmentAnchors = 0;
      for (const SpatialFiniteBufferConflictWitness &witness : *witnesses)
        for (const SpatialProgressRouteAnchor &anchor : witness.routeAnchors) {
          if (anchor.kind == SpatialProgressRouteAnchorKind::RouteTreeArc)
            ++routeArcAnchors;
          else
            ++attachmentAnchors;
        }
      fields["conflict_witnesses"] =
          static_cast<std::uint64_t>(witnesses->size());
      fields["conflict_route_arc_anchors"] = routeArcAnchors;
      fields["conflict_attachment_anchors"] = attachmentAnchors;
    } else {
      llvm::consumeError(witnesses.takeError());
    }
    fields["logical_nets"] =
        static_cast<std::uint64_t>(problem.transfers().logicalNets().size());
    captured_ = std::move(fields);
  }

  std::uint64_t &phase(std::uint64_t &slot) {
    const auto now = std::chrono::steady_clock::now();
    slot += std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark_)
                .count();
    mark_ = now;
    return slot;
  }

  void restartClock() { mark_ = std::chrono::steady_clock::now(); }

  std::uint64_t seedNanoseconds = 0;
  std::uint64_t annealingNanoseconds = 0;
  std::uint64_t exactRepairNanoseconds = 0;
  std::uint64_t finalClosureNanoseconds = 0;
  std::uint64_t verificationNanoseconds = 0;

private:
  std::uint32_t attempt_;
  std::optional<llvm::json::Object> captured_;
  std::chrono::steady_clock::time_point mark_ =
      std::chrono::steady_clock::now();
};

SpatialRestartResult runSpatialRestartImpl(
    const FrozenSpatialPnrProblemHandle &problem, std::uint32_t attempt,
    ExecutionControlView executionControl, SpatialRestartScratch &scratch,
    SpatialPathFinderSeedHandoffHandle preparedSeedHandoff = nullptr) {
  SpatialPnrGenerationAccounting accounting;
  if (!preparedSeedHandoff && executionControl.stopRequested())
    return restartInterrupted(SpatialPnrInterruptionStage::SeedConstruction,
                              std::move(accounting));
  const SpatialPnrWorkLedgerView workLedger = canonicalWorkLedger(accounting);
  const auto &search = problem->config().policy().search;
  SpatialAnnealingSearchScratch &annealing = scratch.annealing;
  SpatialExactRepairScratch &repair = scratch.repair;
  SpatialGlobalRoutingClosureScratch &finalClosure = scratch.finalClosure;

  SpatialPathFinderSeedWorkSummary seedWork;
  const auto seedBegin = std::chrono::steady_clock::now();
  llvm::Expected<SpatialPathFinderSeed> seed = [&]() {
    if (!preparedSeedHandoff)
      return createPathFinderSpatialSeed(problem, attempt, seedWork, {},
                                         executionControl);
    if (preparedSeedHandoff->consumed)
      return llvm::Expected<SpatialPathFinderSeed>(llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial seed handoff was consumed more than once"));
    if (preparedSeedHandoff->attemptOrdinal != attempt)
      return llvm::Expected<SpatialPathFinderSeed>(llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial seed handoff ordinal does not match restart ordinal"));
    preparedSeedHandoff->consumed = true;
    seedWork = preparedSeedHandoff->workSummary;
    if (preparedSeedHandoff->seed) {
      auto prepared = llvm::Expected<SpatialPathFinderSeed>(
          std::move(*preparedSeedHandoff->seed));
      preparedSeedHandoff->seed.reset();
      return prepared;
    }
    if (preparedSeedHandoff->failure) {
      auto failure = llvm::Expected<SpatialPathFinderSeed>(
          std::move(*preparedSeedHandoff->failure));
      preparedSeedHandoff->failure.reset();
      return failure;
    }
    return llvm::Expected<SpatialPathFinderSeed>(llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Spatial seed handoff contains neither a seed nor a failure"));
  }();
  SpatialRestartReporter reporter(attempt);
  reporter.seedNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - seedBegin)
          .count();
  std::optional<AttemptFailure> seedFailure;
  if (!seed)
    seedFailure.emplace(classifyAttemptFailure(seed.takeError()));
  const bool seedAttemptCompleted =
      !seedFailure || (seedFailure->kind != AttemptFailureKind::Internal &&
                       seedFailure->kind != AttemptFailureKind::Interrupted);
  if (llvm::Error error = checkedAdd(seedWork.plannedSeedAttempts,
                                     accounting.plannedSeedAttemptSlots,
                                     "planned seed attempt slots"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error =
          checkedAdd(seedWork.seedAttempts, accounting.seedAttemptSlots,
                     "seed attempt slots"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error =
          checkedAdd(seedWork.plannedInitializerAssignmentAttempts,
                     accounting.plannedInitializerAssignmentAttempts,
                     "planned initializer assignment attempts"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.initializerAssignmentAttempts,
                                     accounting.initializerAssignmentAttempts,
                                     "initializer assignment attempts"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.plannedEndpointExpansions,
                                     accounting.plannedEndpointExpansionSlots,
                                     "planned seed endpoint expansions"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.endpointExpansions,
                                     accounting.endpointExpansionSlots,
                                     "seed endpoint expansions"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error =
          checkedAdd(seedWork.plannedNegotiationIterations,
                     accounting.plannedNegotiationIterationSlots,
                     "planned seed negotiation iterations"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.negotiationIterations,
                                     accounting.negotiationIterationSlots,
                                     "seed negotiation iterations"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (seedWork.plannedSeedAttempts != 1 || seedWork.seedAttempts > 1 ||
      (seedWork.seedAttempts != 0) != seedAttemptCompleted)
    return restartInternal(
        InternalSpatialPnrGenerationReason::SeedConstruction,
        std::move(accounting),
        "canonical seed owner completion disagrees with its typed outcome");
  if ((seedFailure && seedFailure->kind == AttemptFailureKind::Interrupted) ||
      executionControl.stopRequested())
    return restartInterrupted(SpatialPnrInterruptionStage::SeedConstruction,
                              std::move(accounting));
  if (seedFailure) {
    AttemptFailure failure = std::move(*seedFailure);
    if (failure.kind == AttemptFailureKind::ProvenInfeasible) {
      std::optional<SpatialGraphBoundaryEndpointHallDeficit> feedback;
      if (failure.hallWitness) {
        auto projected = problem->bindingRelations().projectGraphBoundaryHall(
            *failure.hallWitness);
        if (!projected)
          return restartInternal(
              InternalSpatialPnrGenerationReason::SeedConstruction,
              std::move(accounting), projected.takeError());
        if (*projected)
          feedback = SpatialGraphBoundaryEndpointHallDeficit{
              (**projected).inputDemandCount, (**projected).inputEndpointCount,
              (**projected).outputDemandCount,
              (**projected).outputEndpointCount};
      }
      SpatialRestartResult result{
          SpatialRestartDisposition::ProvenInfeasible,
          std::move(accounting),
          nullptr,
          false,
          InternalSpatialPnrGenerationReason::SeedConstruction,
          std::move(failure.diagnostic)};
      result.graphBoundaryEndpointHall = std::move(feedback);
      return result;
    }
    if (failure.kind == AttemptFailureKind::Internal)
      return restartInternal(
          InternalSpatialPnrGenerationReason::SeedConstruction,
          std::move(accounting), failure.diagnostic);
    return {SpatialRestartDisposition::Incomplete,
            std::move(accounting),
            nullptr,
            failure.kind == AttemptFailureKind::SemanticLimit,
            InternalSpatialPnrGenerationReason::SeedConstruction,
            std::move(failure.diagnostic)};
  }

  accounting.preparedSeeds = 1;
  reporter.restartClock();
  auto annealed = annealing.run(*seed, executionControl, workLedger);
  reporter.phase(reporter.annealingNanoseconds);
  if (!annealed)
    return restartInternal(InternalSpatialPnrGenerationReason::Annealing,
                           std::move(accounting), annealed.takeError());
  if (llvm::Error error = accumulateAnnealing(*annealed, accounting))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (annealed->interrupted)
    return restartInterrupted(SpatialPnrInterruptionStage::Annealing,
                              std::move(accounting),
                              std::move(seed->candidate));

  const bool exactRepairEnabled =
      search.exactRepair.kind != ResolvedPnrExactRepairKind::Disabled;
  DeterministicPnrRandomStream exactRepairStream =
      DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, attempt,
          PnrRandomStreamPurpose::ExactRepair);
  bool transportRepairRequested = annealed->repairReadyHandoff;
  bool finalClosureRequired = true;

  while (true) {
    const bool hasAtomicCapacityOveruse =
        seed->candidate->atomicCapacityOveruse() != 0;
    const bool hasTransportViolation =
        seed->candidate->hasTransportClosureViolation();
    if (executionControl.stopRequested())
      return restartInterrupted(
          hasAtomicCapacityOveruse || transportRepairRequested
              ? SpatialPnrInterruptionStage::ExactRepair
              : SpatialPnrInterruptionStage::FinalClosure,
          std::move(accounting), std::move(seed->candidate));
    if (!hasAtomicCapacityOveruse && !hasTransportViolation &&
        !finalClosureRequired)
      break;
    if (hasAtomicCapacityOveruse && !exactRepairEnabled)
      return {SpatialRestartDisposition::Incomplete,
              std::move(accounting),
              std::move(seed->candidate),
              false,
              InternalSpatialPnrGenerationReason::ExactRepair,
              "candidate retained atomic CapacityOveruse while exact repair "
              "is disabled"};

    if (hasAtomicCapacityOveruse || transportRepairRequested) {
      transportRepairRequested = false;
      if (accounting.exactRepairSolverCalls >=
          search.exactRepair.maxSolverCalls)
        return {SpatialRestartDisposition::Incomplete,
                std::move(accounting),
                std::move(seed->candidate),
                true,
                InternalSpatialPnrGenerationReason::ExactRepair,
                "restart exact repair exhausted its solver-call budget"};
      if (llvm::Error error = checkedAdd(1, accounting.exactRepairInvocations,
                                         "exact-repair invocations"))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
      const std::uint64_t remainingSolverCalls =
          search.exactRepair.maxSolverCalls - accounting.exactRepairSolverCalls;
      reporter.restartClock();
      auto repaired = repair.repair(*seed->candidate, attempt,
                                    remainingSolverCalls, exactRepairStream,
                                    workLedger, std::nullopt, executionControl);
      reporter.phase(reporter.exactRepairNanoseconds);
      if (!repaired)
        return restartInternal(InternalSpatialPnrGenerationReason::ExactRepair,
                               std::move(accounting), repaired.takeError());
      if (repaired->solverCalls > remainingSolverCalls)
        return restartInternal(
            InternalSpatialPnrGenerationReason::ExactRepair,
            std::move(accounting),
            "exact repair exceeded the restart solver-call budget");
      if (executionControl.stopRequested())
        return restartInterrupted(SpatialPnrInterruptionStage::ExactRepair,
                                  std::move(accounting),
                                  std::move(seed->candidate));
      switch (repaired->kind) {
      case SpatialExactRepairResultKind::Repaired:
        if (repaired->solverCalls == 0)
          return restartInternal(
              InternalSpatialPnrGenerationReason::ExactRepair,
              std::move(accounting),
              "successful exact repair consumed no solver call");
        finalClosureRequired = true;
        continue;
      case SpatialExactRepairResultKind::UnknownBudgetExhausted:
      case SpatialExactRepairResultKind::TimedOut:
      case SpatialExactRepairResultKind::RoutingIncomplete:
      case SpatialExactRepairResultKind::RegionTooLarge:
        return {SpatialRestartDisposition::Incomplete,
                std::move(accounting),
                std::move(seed->candidate),
                true,
                InternalSpatialPnrGenerationReason::ExactRepair,
                std::move(repaired->detail)};
      case SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary:
      case SpatialExactRepairResultKind::ProofNotEstablished:
      case SpatialExactRepairResultKind::UnsupportedEncoding:
        return {SpatialRestartDisposition::Incomplete,
                std::move(accounting),
                std::move(seed->candidate),
                false,
                InternalSpatialPnrGenerationReason::ExactRepair,
                std::move(repaired->detail)};
      case SpatialExactRepairResultKind::InternalError:
        return restartInternal(InternalSpatialPnrGenerationReason::ExactRepair,
                               std::move(accounting), repaired->detail);
      }
    }

    if (llvm::Error error =
            workLedger.plan(SpatialPnrWorkKind::FinalClosureAttempt))
      return restartInternal(
          InternalSpatialPnrGenerationReason::AccountingOverflow,
          std::move(accounting), std::move(error));
    reporter.restartClock();
    llvm::Error closureError =
        finalClosure.run(*seed->candidate, workLedger, executionControl);
    reporter.phase(reporter.finalClosureNanoseconds);
    emitFinalClosureHandshakeProjectionStatistics(
        finalClosure.handshakeProjectionStatistics(), attempt,
        accounting.finalClosureAttempts);
    std::optional<AttemptFailure> closureFailure;
    if (closureError)
      closureFailure.emplace(classifyAttemptFailure(std::move(closureError)));
    const bool closureAttemptCompleted =
        !closureFailure ||
        (closureFailure->kind != AttemptFailureKind::Internal &&
         closureFailure->kind != AttemptFailureKind::Interrupted);
    if (closureAttemptCompleted)
      if (llvm::Error error =
              workLedger.consume(SpatialPnrWorkKind::FinalClosureAttempt))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
    if (closureFailure && closureFailure->kind == AttemptFailureKind::Internal)
      return restartInternal(InternalSpatialPnrGenerationReason::FinalClosure,
                             std::move(accounting),
                             std::move(closureFailure->diagnostic));
    if ((closureFailure &&
         closureFailure->kind == AttemptFailureKind::Interrupted) ||
        executionControl.stopRequested()) {
      return restartInterrupted(SpatialPnrInterruptionStage::FinalClosure,
                                std::move(accounting),
                                std::move(seed->candidate));
    }
    if (!closureFailure) {
      finalClosureRequired = false;
      if (seed->candidate->hasTransportClosureViolation()) {
        if (!exactRepairEnabled)
          return {SpatialRestartDisposition::Incomplete,
                  std::move(accounting),
                  std::move(seed->candidate),
                  false,
                  InternalSpatialPnrGenerationReason::FinalClosure,
                  "final routing closure retained a transport violation while "
                  "exact repair is disabled"};
        transportRepairRequested = true;
      }
      continue;
    }

    AttemptFailure failure = std::move(*closureFailure);
    if (!exactRepairEnabled || !seed->candidate->hasTransportClosureViolation())
      return {SpatialRestartDisposition::Incomplete,
              std::move(accounting),
              std::move(seed->candidate),
              failure.kind == AttemptFailureKind::SemanticLimit,
              InternalSpatialPnrGenerationReason::FinalClosure,
              std::move(failure.diagnostic)};
    transportRepairRequested = true;
  }

  if (executionControl.stopRequested())
    return restartInterrupted(
        SpatialPnrInterruptionStage::CandidateVerification,
        std::move(accounting), std::move(seed->candidate));
  reporter.restartClock();
  if (llvm::Error error = seed->candidate->verify())
    return restartInternal(
        InternalSpatialPnrGenerationReason::CandidateVerification,
        std::move(accounting), std::move(error));
  reporter.phase(reporter.verificationNanoseconds);
  reporter.capture(*seed->candidate, *problem);
  emitActiveHandshakeStatistics(
      seed->candidate->handshake().materializationStatistics(), attempt);
  auto violation = firstFinalViolation(*seed->candidate);
  if (!violation)
    return restartInternal(
        InternalSpatialPnrGenerationReason::CandidateVerification,
        std::move(accounting), violation.takeError());
  if (*violation)
    return {SpatialRestartDisposition::Incomplete,
            std::move(accounting),
            std::move(seed->candidate),
            false,
            InternalSpatialPnrGenerationReason::CandidateVerification,
            violationDiagnostic(**violation)};
  if (executionControl.stopRequested())
    return restartInterrupted(
        SpatialPnrInterruptionStage::CandidateVerification,
        std::move(accounting), std::move(seed->candidate));
  return {SpatialRestartDisposition::Candidate,
          std::move(accounting),
          std::move(seed->candidate),
          annealed->completionGoalReached,
          InternalSpatialPnrGenerationReason::CandidateVerification,
          {}};
}

SpatialRestartResult runSpatialRestart(
    const FrozenSpatialPnrProblemHandle &problem, std::uint32_t attempt,
    ExecutionControlView executionControl,
    SpatialPathFinderSeedHandoffHandle preparedSeedHandoff = nullptr) {
  SpatialRestartScratch scratch;
  SpatialRestartResult result =
      runSpatialRestartImpl(problem, attempt, executionControl, scratch,
                            std::move(preparedSeedHandoff));
  result.workerScratchRetainedBytes = scratch.retainedStorageBytes();
  return result;
}

llvm::Error
accumulateRestartAccounting(const SpatialPnrGenerationAccounting &source,
                            SpatialPnrGenerationAccounting &target) {
#define LOOM_ACCUMULATE_SPATIAL_FIELD(Field, Label)                            \
  if (llvm::Error error = checkedAdd(source.Field, target.Field, Label))       \
  return error
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedInitializerAssignmentAttempts,
                                "planned initializer assignment attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedEndpointExpansionSlots,
                                "planned endpoint expansion slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedNegotiationIterationSlots,
                                "planned negotiation iteration slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedCalibrationProposalSlots,
                                "planned calibration proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedAnnealingBaseProposalSlots,
                                "planned base annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedAnnealingMovableProposalSlots,
                                "planned movable annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedExactRepairRegionDecisions,
                                "planned exact repair region decisions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedExactRepairSolverCalls,
                                "planned exact repair solver calls");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedLocalTransferAdoptionProbes,
                                "planned local transfer adoption probes");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedFinalClosureAttempts,
                                "planned final closure attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedSeedAttemptSlots,
                                "planned seed attempt slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(seedAttemptSlots, "seed attempt slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(preparedSeeds, "prepared seeds");
  LOOM_ACCUMULATE_SPATIAL_FIELD(initializerAssignmentAttempts,
                                "initializer assignment attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(endpointExpansionSlots,
                                "endpoint expansion slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(negotiationIterationSlots,
                                "negotiation iteration slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(calibrationProposalSlots,
                                "calibration proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingBaseProposalSlots,
                                "base annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingMovableProposalSlots,
                                "movable annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingAcceptedActions,
                                "annealing accepted Actions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairInvocations,
                                "exact repair invocations");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairRegionDecisions,
                                "exact repair region decisions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairSolverCalls,
                                "exact repair solver calls");
  LOOM_ACCUMULATE_SPATIAL_FIELD(localTransferAdoptionProbes,
                                "local transfer adoption probes");
  LOOM_ACCUMULATE_SPATIAL_FIELD(adoptedLocalTransfers,
                                "adopted local transfers");
  LOOM_ACCUMULATE_SPATIAL_FIELD(finalClosureAttempts, "final closure attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(finalizedRestarts, "finalized restarts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(publicationSlots, "publication slots");
#undef LOOM_ACCUMULATE_SPATIAL_FIELD
  return llvm::Error::success();
}

SpatialRestartCandidateSummary
summarizeCandidate(const SpatialCandidateState &candidate) {
  SpatialRestartCandidateSummary summary;
  auto objective = candidate.problem().objectiveProgram().evaluate(candidate);
  if (objective)
    summary.objective = std::move(*objective);
  else
    llvm::consumeError(objective.takeError());
  std::array<std::optional<std::uint64_t>, resolvedPnrViolationKindCount>
      values{};
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal) {
    auto value = spatialMappingViolationValue(
        candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
    if (value)
      values[ordinal] = *value;
    else
      llvm::consumeError(value.takeError());
  }
  summary.residual.violationValues = values;
  return summary;
}

SpatialPnrInterruptionSnapshot
projectInterruptionSnapshot(SpatialPnrInterruptionStage stage,
                            std::optional<std::uint32_t> restartOrdinal,
                            const SpatialPnrGenerationAccounting &accounting,
                            llvm::ArrayRef<SpatialRestartResult> restarts,
                            std::uint64_t retainedCandidates,
                            const ExecutionResourceTracker &resources,
                            const MappingObjectiveProgram *objectiveProgram) {
  std::optional<SpatialRestartCandidateSummary> best;
  for (const SpatialRestartResult &restart : restarts) {
    if (!restart.finalized && !restart.candidate)
      continue;
    SpatialRestartCandidateSummary summary =
        restart.finalized ? restart.finalized->summary
                          : summarizeCandidate(*restart.candidate);
    if (!summary.objective) {
      if (!best)
        best = std::move(summary);
      continue;
    }
    if (best && best->objective) {
      auto comparison = objectiveProgram->compareSelectedRank(
          *summary.objective, {}, *best->objective, {});
      if (!comparison) {
        llvm::consumeError(comparison.takeError());
        continue;
      }
      if (*comparison >= 0)
        continue;
    }
    best = std::move(summary);
  }

  SpatialPnrInterruptionSnapshot snapshot;
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
      accounting.exactRepairSolverCalls,
      accounting.localTransferAdoptionProbes,
      accounting.finalClosureAttempts,
      accounting.finalizedRestarts,
      accounting.publicationSlots,
  };
  if (best) {
    if (best->objective)
      snapshot.bestSelectedRank = std::vector<std::uint64_t>(
          best->objective->codes().begin(), best->objective->codes().end());
    snapshot.closureResidual = best->residual;
  }
  snapshot.closureResidual.retainedCandidates = retainedCandidates;
  snapshot.resources = resources.observe();
  return snapshot;
}

llvm::json::Object
interruptionPayload(const SpatialPnrInterruptionSnapshot &snapshot) {
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
  frontier["exact_repair_solver_calls"] =
      snapshot.frontier.exactRepairSolverCalls;
  frontier["local_transfer_adoption_probes"] =
      snapshot.frontier.localTransferAdoptionProbes;
  frontier["final_closure_attempts"] = snapshot.frontier.finalClosureAttempts;
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
  payload["stage"] = spatialPnrInterruptionStageSpelling(snapshot.stage);
  payload["frontier"] = std::move(frontier);
  payload["best_selected_rank"] = snapshot.bestSelectedRank
                                      ? llvm::json::Value(std::move(rank))
                                      : llvm::json::Value(nullptr);
  payload["closure_residual"] = std::move(residual);
  payload["resources"] = std::move(resourceValues);
  return payload;
}

SpatialPnrGenerationOutcome
interruptedOutcome(SpatialPnrInterruptionStage stage,
                   std::optional<std::uint32_t> restartOrdinal,
                   SpatialPnrGenerationAccounting accounting,
                   std::vector<ArtifactRootReference> candidates,
                   llvm::ArrayRef<SpatialRestartResult> restarts,
                   const ExecutionResourceTracker &resources,
                   const MappingObjectiveProgram *objectiveProgram = nullptr) {
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  SpatialPnrInterruptionSnapshot snapshot = projectInterruptionSnapshot(
      stage, restartOrdinal, accounting, restarts, candidates.size(), resources,
      objectiveProgram);
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "invocation";
        fields["closure_status"] = "cancelled_or_timeout";
        fields["interruption"] = interruptionPayload(snapshot);
      });
  return InterruptedSpatialPnrGeneration{std::move(candidates), accounting,
                                         std::move(snapshot)};
}

} // namespace

llvm::StringRef
spatialPnrInterruptionStageSpelling(SpatialPnrInterruptionStage stage) {
  switch (stage) {
  case SpatialPnrInterruptionStage::InputAdmission:
    return "input_admission";
  case SpatialPnrInterruptionStage::FrozenModelConstruction:
    return "frozen_model_construction";
  case SpatialPnrInterruptionStage::SeedConstruction:
    return "seed_construction";
  case SpatialPnrInterruptionStage::Annealing:
    return "annealing";
  case SpatialPnrInterruptionStage::ExactRepair:
    return "exact_repair";
  case SpatialPnrInterruptionStage::FinalClosure:
    return "final_closure";
  case SpatialPnrInterruptionStage::CandidateVerification:
    return "candidate_verification";
  case SpatialPnrInterruptionStage::CandidateFinalization:
    return "candidate_finalization";
  }
  llvm_unreachable("unknown Spatial PnR interruption stage");
}

llvm::Error
verifySpatialPnrWorkAccounting(const SpatialPnrGenerationAccounting &accounting,
                               bool requireClosedWork) {
  const std::array<std::pair<std::uint64_t, std::uint64_t>, 11> counters = {{
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
      {accounting.plannedLocalTransferAdoptionProbes,
       accounting.localTransferAdoptionProbes},
      {accounting.plannedFinalClosureAttempts, accounting.finalClosureAttempts},
  }};
  for (const auto [planned, consumed] : counters) {
    if (consumed > planned)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial PnR consumed work exceeds planned work");
    if (requireClosedWork && planned != consumed)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial PnR completed with admitted work still live");
  }
  return llvm::Error::success();
}

namespace {

SpatialPnrGenerationOutcome
generateSpatialMappingsImpl(const SpatialPnrGenerationInputs &inputs,
                            llvm::ThreadPoolInterface *invocationWorkers) {
  const ExecutionResourceTracker resources;
  SpatialPnrGenerationAccounting accounting;
  if (inputs.maximumCandidatePublications &&
      *inputs.maximumCandidatePublications == 0)
    return InvalidSpatialPnrGeneration{
        InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
        "maximum candidate publications must be positive"};
  ExecutionResourceBudget executionBudget = inputs.executionBudget;
  if (executionBudget.cpuCores && *executionBudget.cpuCores == 0)
    executionBudget.cpuCores.reset();
  if (executionBudget.memoryBytes && *executionBudget.memoryBytes == 0)
    executionBudget.memoryBytes.reset();
  if (inputs.executionControl.stopRequested())
    return interruptedOutcome(SpatialPnrInterruptionStage::InputAdmission,
                              std::nullopt, accounting, {}, {}, resources);

  std::optional<FabricDerivedContextBundle> ownedContexts;
  const FabricDerivedContextBundle *derivedContexts = inputs.derivedContexts;
  if (!derivedContexts) {
    DerivedContextCacheAccess staticAccess;
    DerivedContextCacheAccess timingAccess;
    auto built = buildFabricDerivedContextBundle(
        inputs.fabric, inputs.physicalTiming, &staticAccess, &timingAccess);
    if (!built) {
      FreezeFailure failure = classifyFreezeFailure(built.takeError());
      switch (failure.kind) {
      case FreezeFailureKind::Invalid:
        return InvalidSpatialPnrGeneration{
            InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
            std::move(failure.diagnostic)};
      case FreezeFailureKind::ProvenInfeasible:
        return ProvenInfeasibleSpatialMapping{
            accounting, std::move(failure.diagnostic), std::nullopt,
            SpatialPnrInfeasibilityProofKind::FrozenDerivedContext};
      case FreezeFailureKind::Internal:
        return internal(
            InternalSpatialPnrGenerationReason::FrozenModelConstruction,
            accounting, failure.diagnostic);
      }
    }
    ownedContexts.emplace(std::move(*built));
    derivedContexts = &*ownedContexts;
    emitFabricDerivedContextStatistics(
        *derivedContexts, mapping_debug::Stage::SpatialPnr, staticAccess.hits,
        staticAccess.misses, timingAccess.hits, timingAccess.misses);
  } else {
    emitFabricDerivedContextStatistics(
        *derivedContexts, mapping_debug::Stage::SpatialPnr, 1, 0, 1, 0);
  }

  if (inputs.emitTopologyQualityDiagnostic &&
      inputs.topologyQualityDiagnostic) {
    emitFabricTopologyQuality(*inputs.topologyQualityDiagnostic,
                              mapping_debug::Stage::SpatialPnr);
  } else if (inputs.emitTopologyQualityDiagnostic) {
    if (const auto *topology = derivedContexts->topologyQualityDiagnostic())
      emitFabricTopologyQuality(*topology, mapping_debug::Stage::SpatialPnr);
  }
  llvm::Expected<FrozenSpatialPnrProblemHandle> problem =
      inputs.preparedActiveProblem
          ? llvm::Expected<FrozenSpatialPnrProblemHandle>(
                inputs.preparedActiveProblem)
          : freezeSpatialPnrProblem(inputs.dataflow, inputs.techMapping,
                                    inputs.fabric, inputs.physicalTiming,
                                    inputs.config, inputs.constraints,
                                    derivedContexts);
  if (!problem) {
    FreezeFailure failure = classifyFreezeFailure(problem.takeError());
    switch (failure.kind) {
    case FreezeFailureKind::Invalid:
      return InvalidSpatialPnrGeneration{
          InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    case FreezeFailureKind::ProvenInfeasible:
      return ProvenInfeasibleSpatialMapping{
          accounting, std::move(failure.diagnostic), std::nullopt,
          SpatialPnrInfeasibilityProofKind::FrozenActiveProblem};
    case FreezeFailureKind::Internal:
      return internal(
          InternalSpatialPnrGenerationReason::FrozenModelConstruction,
          accounting, failure.diagnostic);
    }
  }
  if (inputs.preparedActiveProblem) {
    if (llvm::Error error = revalidateFrozenSpatialPnrCacheHit(
            **problem, inputs.dataflow, inputs.techMapping, inputs.fabric,
            inputs.physicalTiming, inputs.config, inputs.constraints))
      return InvalidSpatialPnrGeneration{
          InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
          llvm::toString(std::move(error))};
  }
  emitSpatialActiveProblemStatistics(**problem,
                                     mapping_debug::Stage::SpatialPnr,
                                     inputs.preparedActiveProblem ? 1 : 0,
                                     inputs.preparedActiveProblem ? 0 : 1);
  if (inputs.executionControl.stopRequested())
    return interruptedOutcome(
        SpatialPnrInterruptionStage::FrozenModelConstruction, std::nullopt,
        accounting, {}, {}, resources);

  if (!std::holds_alternative<ResolvedPathFinderPolicy>(
          inputs.config.policy().search.routing.negotiation))
    return UnsupportedSpatialPnrGeneration{
        UnsupportedSpatialPnrGenerationReason::RoutingNegotiation, accounting,
        "the selected routing negotiation kernel is not implemented"};

  if (std::optional<std::string> unsupported =
          unsupportedSpatialExactRepairDomain(**problem))
    return UnsupportedSpatialPnrGeneration{
        UnsupportedSpatialPnrGenerationReason::ExactRepairCapability,
        accounting, std::move(*unsupported)};

  switch ((*problem)->progressBasis().kind) {
  case ::loom::mapping::MappingDataflowProgressBasisKind::Acyclic:
  case ::loom::mapping::MappingDataflowProgressBasisKind::InitializedFeedback:
    break;
  case ::loom::mapping::MappingDataflowProgressBasisKind::Cyclic:
    ::loom::mapping::emitMappingDataflowProgressBasisDiagnostic(
        (*problem)->progressBasis(), inputs.dataflow,
        mapping_debug::Stage::SpatialPnr);
    return IncompleteSpatialPnrGeneration{
        IncompleteSpatialPnrGenerationReason::ProofNotEstablished, accounting,
        "proof_not_established: cyclic Dataflow basis requires a typed "
        "progress breaker"};
  }

  const std::uint32_t restartCount =
      inputs.config.policy().search.initializer.seedAttemptCount;
  if (inputs.candidateWorkerCount == 0)
    return InvalidSpatialPnrGeneration{
        InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
        "candidate worker count must be positive"};
  const bool firstVerifiedCandidate =
      inputs.config.policy().search.completionGoal ==
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  if (inputs.preparedCanonicalSeed) {
    const SpatialPathFinderSeedHandoff &handoff = *inputs.preparedCanonicalSeed;
    if (firstVerifiedCandidate || handoff.attemptOrdinal != 0 ||
        !handoff.problemCacheKey ||
        *handoff.problemCacheKey != (*problem)->cacheKey() ||
        (handoff.seed.has_value() == handoff.failure.has_value()))
      return InvalidSpatialPnrGeneration{
          InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
          "prepared Spatial seed handoff is incompatible with the selected "
          "completion goal or ordinal"};
    if (handoff.seed &&
        (!handoff.seed->candidate ||
         handoff.seed->attemptOrdinal != handoff.attemptOrdinal ||
         handoff.seed->candidate->problem().cacheKey() !=
             (*problem)->cacheKey()))
      return InvalidSpatialPnrGeneration{
          InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
          "prepared Spatial seed handoff does not bind the exact frozen "
          "problem"};
  }
  const FrozenSpatialPnrProblemHandle frozenProblem = *problem;
  std::vector<SpatialRestartResult> restartResults(restartCount);
  const auto finalizeRestart = [&](SpatialRestartResult &restart) {
    if (restart.disposition != SpatialRestartDisposition::Candidate)
      return;
    if (inputs.executionControl.stopRequested()) {
      restart.disposition = SpatialRestartDisposition::Interrupted;
      restart.interruptionStage =
          SpatialPnrInterruptionStage::CandidateFinalization;
      restart.diagnostic = "execution control requested stop";
      return;
    }
    const auto failFinalization = [&](llvm::Error error) {
      restart.disposition = SpatialRestartDisposition::Internal;
      restart.internalReason =
          InternalSpatialPnrGenerationReason::CandidateFinalization;
      restart.diagnostic = llvm::toString(std::move(error));
    };
    ++restart.accounting.publicationSlots;
    auto finalized = finalizeSpatialMappingCandidate(
        *restart.candidate, inputs.dataflow, inputs.techMapping, inputs.fabric,
        inputs.constraints, inputs.store, &derivedContexts->handshakeContext());
    if (!finalized)
      return failFinalization(finalized.takeError());
    ++restart.accounting.finalizedRestarts;
    const auto &view = finalized->view();
    auto progress = ::loom::mapping::deriveSpatialMappingProgressClosure(
        inputs.dataflow, inputs.techMapping, inputs.fabric,
        view.computeBindings(), view.registerFifoTransfers(), view.routeTrees(),
        view.resourceUses(), view.physicalTagSegments());
    if (!progress)
      return failFinalization(progress.takeError());
    if (progress->kind !=
        ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet) {
      restart.diagnostic =
          "proof_not_established: " +
          ::loom::mapping::mappingProgressClosureReasonSpelling(
              progress->reason)
              .str();
      return;
    }
    restart.finalized.emplace(SpatialFinalizedRestart{
        finalized->reference(), summarizeCandidate(*restart.candidate)});
    restart.candidate.reset();
  };
  std::atomic_uint32_t resultPrefixEnd{restartCount};
  struct RestartExecutionControl final {
    ExecutionControlView invocation;
    const std::atomic_uint32_t &resultPrefixEnd;
    std::uint32_t ordinal;
  };
  std::vector<std::uint8_t> completedRestarts(restartCount);
  std::mutex publicationMutex;
  std::uint32_t nextPublicationOrdinal = 0;
  std::uint64_t admittedPublications = 0;
  const auto runRestart = [&](std::uint32_t attempt) {
    SpatialPathFinderSeedHandoffHandle handoff;
    if (attempt == 0)
      handoff = inputs.preparedCanonicalSeed;
    const RestartExecutionControl restartControl{
        inputs.executionControl, resultPrefixEnd, attempt};
    const ExecutionControlView control = firstVerifiedCandidate
        ? ExecutionControlView{
              &restartControl,
              [](const void *context) {
                const auto &control =
                    *static_cast<const RestartExecutionControl *>(context);
                return control.invocation.stopRequested() ||
                       control.ordinal >= control.resultPrefixEnd.load(
                                              std::memory_order_acquire);
              },
              [](const void *context) {
                return static_cast<const RestartExecutionControl *>(context)
                    ->invocation.remainingTime();
              }}
        : inputs.executionControl;
    SpatialRestartResult result = runSpatialRestart(
        frozenProblem, attempt, control, std::move(handoff));
    if (!firstVerifiedCandidate && !inputs.maximumCandidatePublications) {
      finalizeRestart(result);
      restartResults[attempt] = std::move(result);
      return;
    }
    // Publication and FirstVerified selection admit only an ordinal prefix.
    // Later completed slots retain their actual work until the prefix closes.
    std::lock_guard<std::mutex> lock(publicationMutex);
    restartResults[attempt] = std::move(result);
    completedRestarts[attempt] = 1;
    while (nextPublicationOrdinal < resultPrefixEnd.load() &&
           completedRestarts[nextPublicationOrdinal]) {
      SpatialRestartResult &ready = restartResults[nextPublicationOrdinal++];
      if (!inputs.maximumCandidatePublications ||
          admittedPublications < *inputs.maximumCandidatePublications) {
        finalizeRestart(ready);
        admittedPublications += ready.accounting.publicationSlots;
      }
      if (firstVerifiedCandidate &&
          (ready.finalized || ready.disposition ==
                                  SpatialRestartDisposition::ProvenInfeasible))
        resultPrefixEnd.store(nextPublicationOrdinal, std::memory_order_release);
    }
  };
  const bool calibrateWorkerMemory =
      executionBudget.memoryBytes && restartCount > 1;
  std::uint32_t firstPendingRestart = 0;
  std::uint64_t workerScratchReservationBytes = 0;
  if (calibrateWorkerMemory) {
    runRestart(0);
    workerScratchReservationBytes =
        restartResults.front().workerScratchRetainedBytes;
    firstPendingRestart = 1;
  }
  detail::SpatialPnrWorkerAllocation workerAllocation =
      detail::resolveWorkerAllocation(
          inputs.candidateWorkerCount, restartCount, (*problem)->statistics(),
          workerScratchReservationBytes, executionBudget, calibrateWorkerMemory);
  if (calibrateWorkerMemory)
    workerAllocation.actualWorkerCount =
        std::max(1U, std::min(workerAllocation.actualWorkerCount,
                              resultPrefixEnd.load() - firstPendingRestart));
  const std::uint32_t workerCount = workerAllocation.actualWorkerCount;
  workerAllocation.serialPrefix = firstVerifiedCandidate && workerCount == 1;
  auto emitExecutionStatisticsOnExit = llvm::scope_exit([&] {
    detail::emitInvocationExecutionStatistics(
        workerAllocation, (*problem)->statistics(), executionBudget, resources,
        inputs.preparedCanonicalSeed != nullptr, invocationWorkers != nullptr);
  });

  if (workerCount == 1) {
    while (firstPendingRestart < resultPrefixEnd.load())
      runRestart(firstPendingRestart++);
  } else {
    std::optional<llvm::DefaultThreadPool> ownedWorkers;
    if (!invocationWorkers)
      ownedWorkers.emplace(llvm::heavyweight_hardware_concurrency(
          static_cast<unsigned>(workerCount)));
    llvm::ThreadPoolTaskGroup restarts(invocationWorkers ? *invocationWorkers
                                                         : *ownedWorkers);
    if (firstVerifiedCandidate) {
      // A wave bounds both active workers and completed unselected state.
      while (firstPendingRestart < resultPrefixEnd.load() &&
             !inputs.executionControl.stopRequested()) {
        const std::uint32_t waveEnd = firstPendingRestart +
            std::min(workerCount, restartCount - firstPendingRestart);
        for (; firstPendingRestart != waveEnd; ++firstPendingRestart)
          restarts.async(runRestart, firstPendingRestart);
        restarts.wait();
      }
    } else {
      std::atomic_uint32_t nextRestart{firstPendingRestart};
      for (std::uint32_t worker = 0; worker != workerCount; ++worker)
        restarts.async([&] {
          while (true) {
            const std::uint32_t attempt =
                nextRestart.fetch_add(1, std::memory_order_relaxed);
            if (attempt >= restartCount)
              break;
            runRestart(attempt);
          }
        });
      // Waiting runs child work in the borrowed pool without an idle parent.
      restarts.wait();
      firstPendingRestart = restartCount;
    }
  }
  restartResults.resize(firstPendingRestart);
  const std::uint32_t resultRestartCount =
      std::min(firstPendingRestart, resultPrefixEnd.load());
  llvm::MutableArrayRef<SpatialRestartResult> resultRestarts(
      restartResults.data(), resultRestartCount);
  if (firstVerifiedCandidate)
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
        mapping_debug::Event::Statistics, [&](llvm::json::Object &fields) {
          fields["statistics_kind"] = "spatial_first_verified_prefix";
          fields["result_prefix_restart_count"] = resultRestartCount;
          fields["started_restart_count"] = firstPendingRestart;
          fields["speculative_suffix_restart_count"] =
              firstPendingRestart - resultRestartCount;
        });
  for (const SpatialRestartResult &restart : restartResults)
    workerAllocation.maximumObservedWorkerScratchBytes =
        std::max(workerAllocation.maximumObservedWorkerScratchBytes,
                 restart.workerScratchRetainedBytes);
  if (!executionBudget.memoryBytes)
    workerAllocation.workerScratchReservationBytes =
        workerAllocation.maximumObservedWorkerScratchBytes;

  std::vector<ArtifactRootReference> candidates;
  bool semanticLimitReached = resultRestartCount != restartCount;
  bool proofNotEstablished = false;
  std::string staticProgressDiagnostic;
  const SpatialRestartResult *incompleteRepresentative = nullptr;
  const SpatialRestartResult *semanticLimitRepresentative = nullptr;
  const SpatialRestartResult *interruptedRepresentative = nullptr;
  std::optional<std::uint32_t> interruptedOrdinal;
  llvm::Error accountingErrors = llvm::Error::success();
  for (const auto indexedRestart : llvm::enumerate(restartResults)) {
    const SpatialRestartResult &restart = indexedRestart.value();
    accountingErrors = llvm::joinErrors(
        std::move(accountingErrors),
        accumulateRestartAccounting(restart.accounting, accounting));
    const bool requireClosedWork =
        restart.disposition != SpatialRestartDisposition::Interrupted &&
        restart.disposition != SpatialRestartDisposition::Internal;
    accountingErrors = llvm::joinErrors(
        std::move(accountingErrors),
        verifySpatialPnrWorkAccounting(restart.accounting, requireClosedWork));
    const bool contributesToResult = indexedRestart.index() < resultRestartCount;
    emitRestartFailure(static_cast<std::uint32_t>(indexedRestart.index()),
                       restart, contributesToResult);
    if (contributesToResult &&
        restart.disposition == SpatialRestartDisposition::Interrupted &&
        (!interruptedRepresentative ||
         (interruptedRepresentative->accounting.preparedSeeds == 0 &&
          restart.accounting.preparedSeeds != 0))) {
      interruptedRepresentative = &restart;
      interruptedOrdinal = static_cast<std::uint32_t>(indexedRestart.index());
    }
  }

  if (accountingErrors)
    return internal(InternalSpatialPnrGenerationReason::AccountingOverflow,
                    accounting, std::move(accountingErrors));

  for (const SpatialRestartResult &restart : restartResults)
    if (restart.disposition == SpatialRestartDisposition::Internal &&
        restart.internalReason ==
            InternalSpatialPnrGenerationReason::AccountingOverflow)
      return internal(restart.internalReason, accounting, restart.diagnostic);

  const bool hasCandidateRestart =
      llvm::any_of(restartResults, [](const SpatialRestartResult &restart) {
        return restart.disposition == SpatialRestartDisposition::Candidate;
      });
  if (hasCandidateRestart &&
      llvm::any_of(restartResults, [](const SpatialRestartResult &restart) {
        return restart.disposition == SpatialRestartDisposition::ProvenInfeasible;
      }))
    return internal(
        InternalSpatialPnrGenerationReason::CandidateVerification, accounting,
        "one restart proved global infeasibility while another produced "
        "a verified candidate");
  for (SpatialRestartResult &restart : resultRestarts) {
    switch (restart.disposition) {
    case SpatialRestartDisposition::Candidate:
      semanticLimitReached |= restart.semanticLimitReached;
      if (restart.finalized)
        candidates.push_back(restart.finalized->reference);
      else if (restart.accounting.finalizedRestarts != 0) {
        proofNotEstablished = true;
        if (staticProgressDiagnostic.empty())
          staticProgressDiagnostic = restart.diagnostic;
      }
      continue;
    case SpatialRestartDisposition::ProvenInfeasible: {
      emitInvocationAccounting(
          accounting, mapping_debug::ClosureStatus::ProvenInfeasible, 0);
      const bool hasGraphBoundaryHall =
          restart.graphBoundaryEndpointHall.has_value();
      return ProvenInfeasibleSpatialMapping{
          accounting, std::move(restart.diagnostic),
          std::move(restart.graphBoundaryEndpointHall),
          hasGraphBoundaryHall
              ? SpatialPnrInfeasibilityProofKind::GraphBoundaryEndpointHall
              : SpatialPnrInfeasibilityProofKind::InitializerRelation};
    }
    case SpatialRestartDisposition::Incomplete:
      semanticLimitReached |= restart.semanticLimitReached;
      proofNotEstablished |= !restart.semanticLimitReached;
      preferPreparedRestart(restart, incompleteRepresentative);
      if (restart.semanticLimitReached)
        preferPreparedRestart(restart, semanticLimitRepresentative);
      continue;
    case SpatialRestartDisposition::Interrupted:
      continue;
    case SpatialRestartDisposition::Internal:
      return internal(restart.internalReason, accounting, restart.diagnostic);
    }
  }

  if (interruptedRepresentative || inputs.executionControl.stopRequested())
    return interruptedOutcome(
        interruptedRepresentative
            ? interruptedRepresentative->interruptionStage
            : SpatialPnrInterruptionStage::CandidateFinalization,
        interruptedOrdinal, accounting, std::move(candidates), resultRestarts,
        resources, &frozenProblem->objectiveProgram());

  if (!candidates.empty()) {
    llvm::sort(candidates, artifactRootReferenceLess);
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    // A published candidate completes the FirstVerified goal even when an
    // earlier restart could not establish progress. Those restart diagnostics
    // remain recorded, but do not invalidate the independently verified result.
    if (firstVerifiedCandidate) {
      proofNotEstablished = false;
      semanticLimitReached = true;
    }
    const mapping_debug::ClosureStatus closureStatus =
        proofNotEstablished ? mapping_debug::ClosureStatus::ProofNotEstablished
        : semanticLimitReached
            ? mapping_debug::ClosureStatus::SemanticLimitReached
            : mapping_debug::ClosureStatus::Closed;
    emitInvocationAccounting(accounting, closureStatus, candidates.size());
    return GeneratedSpatialMappings{
        std::move(candidates),
        proofNotEstablished ? PnrGenerationTermination::ProofNotEstablished
        : semanticLimitReached
            ? PnrGenerationTermination::SemanticLimitReached
            : PnrGenerationTermination::FixedAttemptsCompleted,
        accounting};
  }
  if (accounting.preparedSeeds == 0 && !semanticLimitReached) {
    emitInvocationAccounting(
        accounting, mapping_debug::ClosureStatus::ProofNotEstablished, 0);
    return IncompleteSpatialPnrGeneration{
        IncompleteSpatialPnrGenerationReason::NoPreparedSeed, accounting,
        !incompleteRepresentative
            ? "no fixed initializer slot produced a prepared Spatial candidate"
            : incompleteRepresentative->diagnostic};
  }
  const SpatialRestartResult *representative = semanticLimitReached
                                                   ? semanticLimitRepresentative
                                                   : incompleteRepresentative;
  auto fifoCapacityShortfall =
      projectFifoCapacityShortfall(incompleteRepresentative);
  if (!fifoCapacityShortfall)
    return internal(InternalSpatialPnrGenerationReason::CandidateVerification,
                    accounting, fifoCapacityShortfall.takeError());
  emitInvocationAccounting(
      accounting,
      proofNotEstablished ? mapping_debug::ClosureStatus::ProofNotEstablished
      : semanticLimitReached
          ? mapping_debug::ClosureStatus::SemanticLimitReached
          : mapping_debug::ClosureStatus::ProofNotEstablished,
      0);
  return IncompleteSpatialPnrGeneration{
      proofNotEstablished
          ? IncompleteSpatialPnrGenerationReason::ProofNotEstablished
      : semanticLimitReached
          ? IncompleteSpatialPnrGenerationReason::SemanticLimitReached
          : IncompleteSpatialPnrGenerationReason::ProofNotEstablished,
      accounting,
      !staticProgressDiagnostic.empty() ? std::move(staticProgressDiagnostic)
      : !representative
          ? "no fixed restart reached independent final verification"
          : representative->diagnostic,
      std::move(*fifoCapacityShortfall)};
}

} // namespace

SpatialPnrGenerationOutcome
generateSpatialMappings(const SpatialPnrGenerationInputs &inputs) {
  return generateSpatialMappingsImpl(inputs, nullptr);
}

SpatialPnrGenerationOutcome
generateSpatialMappings(const SpatialPnrGenerationInputs &inputs,
                        llvm::ThreadPoolInterface &workers) {
  return generateSpatialMappingsImpl(inputs, &workers);
}

} // namespace loom::pnr
