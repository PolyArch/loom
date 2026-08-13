#include "PnR/SpatialPnrGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/MappingDebugLog.h"
#include "FabricTopologyQualityDiagnostic.h"
#include "InitializerRelationSolver.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialAnnealingSearch.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialGlobalRoutingClosure.h"
#include "PnR/SpatialMappingMaterializer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <limits>
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
};

struct AttemptFailure final {
  AttemptFailureKind kind = AttemptFailureKind::Internal;
  std::string diagnostic;
};

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
          result.kind = AttemptFailureKind::Rejected;
          break;
        }
        result.diagnostic = errorMessage(failure);
      },
      [&](const RoutingNegotiationError &failure) {
        result.kind =
            failure.kind() == RoutingNegotiationError::Kind::ArithmeticOverflow
                ? AttemptFailureKind::Rejected
                : AttemptFailureKind::Internal;
        result.diagnostic = errorMessage(failure);
      },
      [&](const SpatialPathFinderClosureFailure &failure) {
        result.kind =
            failure.kind() ==
                        SpatialPathFinderClosureFailure::Kind::NonClosure ||
                    failure.kind() ==
                        SpatialPathFinderClosureFailure::Kind::NoProgress
                ? AttemptFailureKind::SemanticLimit
                : AttemptFailureKind::Rejected;
        result.diagnostic = errorMessage(failure);
      },
      [&](const SpatialActionTransitionFailure &failure) {
        result.kind =
            failure.kind() == SpatialActionTransitionFailureKind::WorkLimit
                ? AttemptFailureKind::SemanticLimit
                : AttemptFailureKind::Rejected;
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
  if (llvm::Error error =
          checkedAdd(source.endpointExpansions, target.endpointExpansionSlots,
                     "annealing endpoint expansions"))
    return error;
  if (llvm::Error error = checkedAdd(source.negotiationIterations,
                                     target.negotiationIterationSlots,
                                     "annealing negotiation iterations"))
    return error;
  return checkedAdd(source.acceptedActionCount, target.annealingAcceptedActions,
                    "annealing accepted Actions");
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
  Internal,
};

struct SpatialRestartResult final {
  SpatialRestartDisposition disposition = SpatialRestartDisposition::Internal;
  SpatialPnrGenerationAccounting accounting;
  SpatialCandidateStateHandle candidate;
  bool semanticLimitReached = false;
  InternalSpatialPnrGenerationReason internalReason =
      InternalSpatialPnrGenerationReason::SeedConstruction;
  std::string diagnostic;
};

llvm::StringRef spelling(SpatialRestartDisposition disposition) {
  switch (disposition) {
  case SpatialRestartDisposition::Candidate:
    return "candidate";
  case SpatialRestartDisposition::ProvenInfeasible:
    return "proven_infeasible";
  case SpatialRestartDisposition::Incomplete:
    return "incomplete";
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
                        const SpatialRestartResult &restart) {
  if (restart.disposition == SpatialRestartDisposition::Candidate)
    return;
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "restart";
        fields["restart_ordinal"] = ordinal;
        fields["closure_status"] = spelling(restart.disposition);
        fields["termination_owner"] = spelling(restart.internalReason);
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
runSpatialRestart(const FrozenSpatialPnrProblemHandle &problem,
                  std::uint32_t attempt) {
  SpatialPnrGenerationAccounting accounting;
  accounting.seedAttemptSlots = 1;
  const auto &search = problem->config().policy().search;
  SpatialAnnealingSearchScratch annealing;
  SpatialExactRepairScratch repair;
  SpatialGlobalRoutingClosureScratch finalClosure;

  SpatialPathFinderSeedWorkSummary seedWork;
  auto seed = createPathFinderSpatialSeed(problem, attempt, seedWork);
  if (llvm::Error error = checkedAdd(seedWork.initializerAssignmentAttempts,
                                     accounting.initializerAssignmentAttempts,
                                     "initializer assignment attempts"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.endpointExpansions,
                                     accounting.endpointExpansionSlots,
                                     "seed endpoint expansions"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (llvm::Error error = checkedAdd(seedWork.negotiationIterations,
                                     accounting.negotiationIterationSlots,
                                     "seed negotiation iterations"))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));
  if (!seed) {
    AttemptFailure failure = classifyAttemptFailure(seed.takeError());
    if (failure.kind == AttemptFailureKind::ProvenInfeasible)
      return {SpatialRestartDisposition::ProvenInfeasible,
              std::move(accounting),
              nullptr,
              false,
              InternalSpatialPnrGenerationReason::SeedConstruction,
              std::move(failure.diagnostic)};
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
  auto annealed = annealing.run(*seed->candidate, attempt);
  if (!annealed)
    return restartInternal(InternalSpatialPnrGenerationReason::Annealing,
                           std::move(accounting), annealed.takeError());
  if (llvm::Error error = accumulateAnnealing(*annealed, accounting))
    return restartInternal(
        InternalSpatialPnrGenerationReason::AccountingOverflow,
        std::move(accounting), std::move(error));

  const auto hasTransportClosureViolation = [&]() {
    return seed->candidate->unroutedObligationCount() != 0 ||
           seed->candidate->routeCapacityOveruse() != 0 ||
           seed->candidate->tagResidentCapacityOveruse() != 0 ||
           seed->candidate->tagUnassignedCount() != 0 ||
           seed->candidate->tagConflictCount() != 0;
  };
  const bool exactRepairEnabled =
      search.exactRepair.kind != ResolvedPnrExactRepairKind::Disabled;
  DeterministicPnrRandomStream exactRepairStream =
      DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, attempt,
          PnrRandomStreamPurpose::ExactRepair);
  bool transportRepairRequested = false;
  bool finalClosureRequired = !annealed->exactClosureReached;

  while (true) {
    const bool hasAtomicCapacityOveruse =
        seed->candidate->atomicCapacityOveruse() != 0;
    const bool hasTransportViolation = hasTransportClosureViolation();
    if (!hasAtomicCapacityOveruse && !hasTransportViolation &&
        !finalClosureRequired)
      break;
    if (hasAtomicCapacityOveruse && !exactRepairEnabled)
      return {SpatialRestartDisposition::Incomplete,
              std::move(accounting),
              nullptr,
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
                nullptr,
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
      auto repaired = repair.repair(*seed->candidate, attempt,
                                    remainingSolverCalls, exactRepairStream);
      if (!repaired)
        return restartInternal(InternalSpatialPnrGenerationReason::ExactRepair,
                               std::move(accounting), repaired.takeError());
      if (repaired->solverCalls > remainingSolverCalls)
        return restartInternal(
            InternalSpatialPnrGenerationReason::ExactRepair,
            std::move(accounting),
            "exact repair exceeded the restart solver-call budget");
      if (llvm::Error error = checkedAdd(repaired->regionDecisions,
                                         accounting.exactRepairRegionDecisions,
                                         "exact-repair region decisions"))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
      if (llvm::Error error = checkedAdd(repaired->solverCalls,
                                         accounting.exactRepairSolverCalls,
                                         "exact-repair solver calls"))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
      if (llvm::Error error = checkedAdd(repaired->endpointExpansions,
                                         accounting.endpointExpansionSlots,
                                         "exact-repair endpoint expansions"))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
      if (llvm::Error error = checkedAdd(repaired->negotiationIterations,
                                         accounting.negotiationIterationSlots,
                                         "exact-repair negotiation iterations"))
        return restartInternal(
            InternalSpatialPnrGenerationReason::AccountingOverflow,
            std::move(accounting), std::move(error));
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
      case SpatialExactRepairResultKind::RegionTooLarge:
        return {SpatialRestartDisposition::Incomplete,
                std::move(accounting),
                nullptr,
                true,
                InternalSpatialPnrGenerationReason::ExactRepair,
                std::move(repaired->detail)};
      case SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary:
      case SpatialExactRepairResultKind::UnsupportedEncoding:
        return {SpatialRestartDisposition::Incomplete,
                std::move(accounting),
                nullptr,
                false,
                InternalSpatialPnrGenerationReason::ExactRepair,
                std::move(repaired->detail)};
      case SpatialExactRepairResultKind::InternalError:
        return restartInternal(InternalSpatialPnrGenerationReason::ExactRepair,
                               std::move(accounting), repaired->detail);
      }
    }

    if (llvm::Error error = checkedAdd(1, accounting.finalClosureAttempts,
                                       "final-closure attempts"))
      return restartInternal(
          InternalSpatialPnrGenerationReason::AccountingOverflow,
          std::move(accounting), std::move(error));
    llvm::Error closureError = finalClosure.run(*seed->candidate);
    if (llvm::Error error = checkedAdd(finalClosure.endpointExpansionCount(),
                                       accounting.endpointExpansionSlots,
                                       "final-closure endpoint expansions"))
      return restartInternal(
          InternalSpatialPnrGenerationReason::AccountingOverflow,
          std::move(accounting), std::move(error));
    if (llvm::Error error = checkedAdd(finalClosure.negotiationIterationCount(),
                                       accounting.negotiationIterationSlots,
                                       "final-closure negotiation iterations"))
      return restartInternal(
          InternalSpatialPnrGenerationReason::AccountingOverflow,
          std::move(accounting), std::move(error));
    if (!closureError) {
      finalClosureRequired = false;
      continue;
    }

    AttemptFailure failure = classifyAttemptFailure(std::move(closureError));
    if (failure.kind == AttemptFailureKind::Internal)
      return restartInternal(InternalSpatialPnrGenerationReason::FinalClosure,
                             std::move(accounting), failure.diagnostic);
    if (!exactRepairEnabled || !hasTransportClosureViolation())
      return {SpatialRestartDisposition::Incomplete,
              std::move(accounting),
              nullptr,
              failure.kind == AttemptFailureKind::SemanticLimit,
              InternalSpatialPnrGenerationReason::FinalClosure,
              std::move(failure.diagnostic)};
    transportRepairRequested = true;
  }

  if (llvm::Error error = seed->candidate->verify())
    return restartInternal(
        InternalSpatialPnrGenerationReason::CandidateVerification,
        std::move(accounting), std::move(error));
  auto violation = firstFinalViolation(*seed->candidate);
  if (!violation)
    return restartInternal(
        InternalSpatialPnrGenerationReason::CandidateVerification,
        std::move(accounting), violation.takeError());
  if (*violation)
    return {SpatialRestartDisposition::Incomplete,
            std::move(accounting),
            nullptr,
            false,
            InternalSpatialPnrGenerationReason::CandidateVerification,
            violationDiagnostic(**violation)};
  return {SpatialRestartDisposition::Candidate,
          std::move(accounting),
          std::move(seed->candidate),
          false,
          InternalSpatialPnrGenerationReason::CandidateVerification,
          {}};
}

llvm::Error
accumulateRestartAccounting(const SpatialPnrGenerationAccounting &source,
                            SpatialPnrGenerationAccounting &target) {
#define LOOM_ACCUMULATE_SPATIAL_FIELD(Field, Label)                            \
  if (llvm::Error error = checkedAdd(source.Field, target.Field, Label))       \
  return error
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
  LOOM_ACCUMULATE_SPATIAL_FIELD(focusedClosureProposalSlots,
                                "focused closure proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingAcceptedActions,
                                "annealing accepted Actions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairInvocations,
                                "exact repair invocations");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairRegionDecisions,
                                "exact repair region decisions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairSolverCalls,
                                "exact repair solver calls");
  LOOM_ACCUMULATE_SPATIAL_FIELD(finalClosureAttempts, "final closure attempts");
#undef LOOM_ACCUMULATE_SPATIAL_FIELD
  return llvm::Error::success();
}

} // namespace

SpatialPnrGenerationOutcome
generateSpatialMappings(const SpatialPnrGenerationInputs &inputs) {
  SpatialPnrGenerationAccounting accounting;
  auto topology = analyzeAndEmitFabricTopologyQuality(
      inputs.fabric, mapping_debug::Stage::SpatialPnr);
  if (!topology)
    return InvalidSpatialPnrGeneration{
        InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
        llvm::toString(topology.takeError())};
  auto problem =
      freezeSpatialPnrProblem(inputs.dataflow, inputs.techMapping,
                              inputs.fabric, inputs.config, inputs.constraints);
  if (!problem) {
    FreezeFailure failure = classifyFreezeFailure(problem.takeError());
    switch (failure.kind) {
    case FreezeFailureKind::Invalid:
      return InvalidSpatialPnrGeneration{
          InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    case FreezeFailureKind::ProvenInfeasible:
      return ProvenInfeasibleSpatialMapping{accounting,
                                            std::move(failure.diagnostic)};
    case FreezeFailureKind::Internal:
      return internal(
          InternalSpatialPnrGenerationReason::FrozenModelConstruction,
          accounting, failure.diagnostic);
    }
  }

  if (!std::holds_alternative<ResolvedPathFinderPolicy>(
          inputs.config.policy().search.routing.negotiation))
    return UnsupportedSpatialPnrGeneration{
        UnsupportedSpatialPnrGenerationReason::RoutingNegotiation, accounting,
        "the selected routing negotiation kernel is not implemented"};

  switch ((*problem)->progressClosure().kind) {
  case ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet:
    break;
  case ::loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet:
    return ProvenInfeasibleSpatialMapping{
        accounting, "Dataflow progress proof found a closed wait set"};
  case ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished:
    return IncompleteSpatialPnrGeneration{
        IncompleteSpatialPnrGenerationReason::ProofNotEstablished, accounting,
        "proof_not_established: Spatial progress closure is unavailable"};
  }

  const std::uint32_t restartCount =
      inputs.config.policy().search.initializer.seedAttemptCount;
  const std::uint32_t workerCount =
      std::min(inputs.candidateWorkerCount, restartCount);
  if (workerCount == 0)
    return InvalidSpatialPnrGeneration{
        InvalidSpatialPnrGenerationReason::FrozenInput, accounting,
        "candidate worker count must be positive"};

  std::vector<SpatialRestartResult> restartResults(restartCount);
  const auto runRestart = [&](std::uint32_t attempt) {
    restartResults[attempt] = runSpatialRestart(*problem, attempt);
  };
  if (workerCount == 1) {
    for (std::uint32_t attempt = 0; attempt != restartCount; ++attempt)
      runRestart(attempt);
  } else {
    llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
        static_cast<unsigned>(workerCount)));
    std::atomic_uint32_t nextRestart{0};
    for (std::uint32_t worker = 0; worker != workerCount; ++worker)
      pool.async([&] {
        while (true) {
          const std::uint32_t attempt =
              nextRestart.fetch_add(1, std::memory_order_relaxed);
          if (attempt >= restartCount)
            break;
          runRestart(attempt);
        }
      });
    pool.wait();
  }

  std::vector<ArtifactRootReference> candidates;
  bool semanticLimitReached = false;
  const SpatialRestartResult *incompleteRepresentative = nullptr;
  const SpatialRestartResult *semanticLimitRepresentative = nullptr;
  for (const auto indexedRestart : llvm::enumerate(restartResults)) {
    const SpatialRestartResult &restart = indexedRestart.value();
    if (llvm::Error error =
            accumulateRestartAccounting(restart.accounting, accounting))
      return internal(InternalSpatialPnrGenerationReason::AccountingOverflow,
                      accounting, std::move(error));
    emitRestartFailure(static_cast<std::uint32_t>(indexedRestart.index()),
                       restart);
  }

  for (SpatialRestartResult &restart : restartResults) {
    switch (restart.disposition) {
    case SpatialRestartDisposition::Candidate:
      break;
    case SpatialRestartDisposition::ProvenInfeasible:
      return ProvenInfeasibleSpatialMapping{accounting,
                                            std::move(restart.diagnostic)};
    case SpatialRestartDisposition::Incomplete:
      semanticLimitReached |= restart.semanticLimitReached;
      preferPreparedRestart(restart, incompleteRepresentative);
      if (restart.semanticLimitReached)
        preferPreparedRestart(restart, semanticLimitRepresentative);
      continue;
    case SpatialRestartDisposition::Internal:
      return internal(restart.internalReason, accounting, restart.diagnostic);
    }

    ++accounting.publicationSlots;
    auto finalized = finalizeSpatialMappingCandidate(
        *restart.candidate, inputs.dataflow, inputs.techMapping, inputs.fabric,
        inputs.constraints, inputs.store);
    if (!finalized)
      return internal(InternalSpatialPnrGenerationReason::CandidateFinalization,
                      accounting, finalized.takeError());
    ++accounting.finalizedRestarts;
    candidates.push_back(finalized->reference());
  }

  if (!candidates.empty()) {
    llvm::sort(candidates, artifactRootReferenceLess);
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    return GeneratedSpatialMappings{
        std::move(candidates),
        semanticLimitReached
            ? SpatialPnrGenerationTermination::SemanticLimitReached
            : SpatialPnrGenerationTermination::FixedAttemptsCompleted,
        accounting};
  }
  if (accounting.preparedSeeds == 0 && !semanticLimitReached)
    return IncompleteSpatialPnrGeneration{
        IncompleteSpatialPnrGenerationReason::NoPreparedSeed, accounting,
        !incompleteRepresentative
            ? "no fixed initializer slot produced a prepared Spatial candidate"
            : incompleteRepresentative->diagnostic};
  const SpatialRestartResult *representative =
      semanticLimitReached ? semanticLimitRepresentative
                           : incompleteRepresentative;
  return IncompleteSpatialPnrGeneration{
      semanticLimitReached
          ? IncompleteSpatialPnrGenerationReason::SemanticLimitReached
          : IncompleteSpatialPnrGenerationReason::ProofNotEstablished,
      accounting,
      !representative
          ? "no fixed restart reached independent final verification"
          : representative->diagnostic};
}

} // namespace loom::pnr
