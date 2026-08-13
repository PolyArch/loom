#include "PnR/System/SystemPnrGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/MappingDebugLog.h"
#include "FabricTopologyQualityDiagnostic.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
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

} // namespace

SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs) {
  SystemPnrGenerationAccounting accounting;
  auto topology = analyzeAndEmitFabricTopologyQuality(
      inputs.fabric.artifact(), mapping_debug::Stage::SystemPnr);
  if (!topology)
    return InvalidSystemPnrGeneration{
        InvalidSystemPnrGenerationReason::FrozenInput, accounting,
        llvm::toString(topology.takeError())};
  auto problem = freezeSystemPnrProblem(inputs.dataflow, inputs.fabric,
                                        inputs.searchDomain, inputs.config,
                                        inputs.constraints, inputs.store);
  if (!problem) {
    FreezeFailure failure = classifyFreezeFailure(problem.takeError());
    switch (failure.kind) {
    case FreezeFailureKind::Invalid:
      return InvalidSystemPnrGeneration{
          InvalidSystemPnrGenerationReason::FrozenInput, accounting,
          std::move(failure.diagnostic)};
    case FreezeFailureKind::ProvenInfeasible:
      return ProvenInfeasibleSystemMapping{accounting,
                                           std::move(failure.diagnostic)};
    case FreezeFailureKind::Internal:
      return internal(
          InternalSystemPnrGenerationReason::FrozenModelConstruction,
          accounting, failure.diagnostic);
    }
  }

  switch ((*problem)->progressClosure().kind) {
  case ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet:
    break;
  case ::loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet:
    return ProvenInfeasibleSystemMapping{
        accounting, "Dataflow progress proof found a closed wait set"};
  case ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished:
    return IncompleteSystemPnrGeneration{
        IncompleteSystemPnrGenerationReason::ProofNotEstablished, accounting,
        "proof_not_established: System progress closure is unavailable"};
  }

  const auto &search = inputs.config.policy().search;
  SystemAnnealingSearchScratch annealing;
  std::vector<ArtifactRootReference> candidates;
  bool semanticLimitReached = false;
  std::string firstIncompleteDiagnostic;
  mlir::MLIRContext context;
  context.loadDialect<::mapping::MappingDialect>();

  const auto rememberIncomplete = [&](llvm::StringRef diagnostic,
                                      bool semanticLimit) {
    semanticLimitReached |= semanticLimit;
    if (firstIncompleteDiagnostic.empty())
      firstIncompleteDiagnostic = diagnostic.str();
  };

  for (std::uint32_t attempt = 0;
       attempt != search.initializer.seedAttemptCount; ++attempt) {
    ++accounting.seedAttemptSlots;
    auto initialized = initializeSystemCandidateAttempt(*problem, attempt);
    if (!initialized) {
      InitializationFailure failure =
          classifyInitializationFailure(initialized.takeError());
      if (llvm::Error error = accumulateInitialization(failure, accounting))
        return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                        accounting, std::move(error));
      switch (failure.kind) {
      case SystemCandidateInitializationFailureKind::ProvenInfeasible:
        if (candidates.empty())
          return ProvenInfeasibleSystemMapping{accounting,
                                               std::move(failure.diagnostic)};
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
    auto annealed = annealing.run(candidate, attempt);
    if (!annealed)
      return internal(InternalSystemPnrGenerationReason::Annealing, accounting,
                      annealed.takeError());
    if (llvm::Error error = accumulateAnnealing(*annealed, accounting))
      return internal(InternalSystemPnrGenerationReason::AccountingOverflow,
                      accounting, std::move(error));

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
    if (!closed) {
      bool workLimit = false;
      std::string diagnostic;
      llvm::handleAllErrors(
          closed.takeError(),
          [&](const SystemActionTransitionFailure &failure) {
            workLimit =
                failure.kind() == SystemActionTransitionFailureKind::WorkLimit;
            diagnostic = errorMessage(failure);
          },
          [&](const llvm::ErrorInfoBase &failure) {
            diagnostic = errorMessage(failure);
          });
      if (workLimit) {
        rememberIncomplete(diagnostic, true);
        continue;
      }
      return internal(
          InternalSystemPnrGenerationReason::FinalClosure, accounting,
          diagnostic.empty() ? "final global Action lost its failure cause"
                             : diagnostic);
    }
    candidate = std::move(closed->candidate);
    if (candidate->capacityOveruse() != 0) {
      rememberIncomplete(
          "strict final global Action retained full CapacityOveruse", false);
      continue;
    }

    if (llvm::Error error = candidate->verify())
      return internal(InternalSystemPnrGenerationReason::CandidateVerification,
                      accounting, std::move(error));
    auto draft = materializeSystemCandidateDraft(*candidate, context);
    if (!draft)
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, draft.takeError());
    auto root = mlir::cast<::mapping::SystemOp>(draft->get());
    ++accounting.finalVerificationAttempts;
    const ::loom::mapping::SystemMappingBaseVerification verification =
        ::loom::mapping::verifySystemMappingBase(root, inputs.dataflow,
                                                 inputs.fabric, inputs.store);
    if (const auto *incomplete =
            std::get_if<::loom::mapping::IncompleteSystemMappingBase>(
                &verification)) {
      rememberIncomplete(incomplete->diagnostic, false);
      continue;
    }
    if (const auto *rejected =
            std::get_if<::loom::mapping::RejectedSystemMappingBase>(
                &verification)) {
      rememberIncomplete(rejected->diagnostic, false);
      continue;
    }
    if (const auto *failure =
            std::get_if<::loom::mapping::InternalSystemMappingBaseError>(
                &verification))
      return internal(InternalSystemPnrGenerationReason::CandidateVerification,
                      accounting, failure->diagnostic);

    ++accounting.publicationSlots;
    auto finalized = ::loom::mapping::finalizeSystemMapping(
        root, inputs.dataflow, inputs.fabric, inputs.constraints.view(),
        inputs.store);
    if (!finalized)
      return internal(InternalSystemPnrGenerationReason::CandidateFinalization,
                      accounting, finalized.takeError());
    ++accounting.finalizedRestarts;
    candidates.push_back(finalized->reference());
  }

  if (!candidates.empty()) {
    llvm::sort(candidates, artifactRootReferenceLess);
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    return GeneratedSystemMappings{std::move(candidates), accounting};
  }
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
