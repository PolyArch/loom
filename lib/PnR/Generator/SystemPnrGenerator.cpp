#include "PnR/System/SystemPnrGenerator.h"

#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

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

} // namespace

SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs) {
  SystemPnrGenerationAccounting accounting;
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

  auto initialized = initializeCanonicalSystemCandidate(*problem);
  if (!initialized) {
    InitializationFailure failure =
        classifyInitializationFailure(initialized.takeError());
    accounting.initializerAssignmentAttempts = failure.assignmentAttempts;
    accounting.endpointExpansionSlots = failure.endpointExpansions;
    switch (failure.kind) {
    case SystemCandidateInitializationFailureKind::ProvenInfeasible:
      return ProvenInfeasibleSystemMapping{accounting,
                                           std::move(failure.diagnostic)};
    case SystemCandidateInitializationFailureKind::SemanticLimitReached:
      return IncompleteSystemPnrGeneration{
          IncompleteSystemPnrGenerationReason::SemanticLimitReached, accounting,
          std::move(failure.diagnostic)};
    case SystemCandidateInitializationFailureKind::Internal:
      return internal(
          InternalSystemPnrGenerationReason::CandidateInitialization,
          accounting, failure.diagnostic);
    }
  }
  accounting.initializerAssignmentAttempts = initialized->assignmentAttempts;
  accounting.endpointExpansionSlots = initialized->endpointExpansions;
  if (llvm::Error error = initialized->state->verify())
    return internal(InternalSystemPnrGenerationReason::CandidateVerification,
                    accounting, std::move(error));

  mlir::MLIRContext context;
  context.loadDialect<::mapping::MappingDialect>();
  auto draft = materializeSystemCandidateDraft(*initialized->state, context);
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
              &verification))
    return IncompleteSystemPnrGeneration{
        IncompleteSystemPnrGenerationReason::ProofNotEstablished, accounting,
        incomplete->diagnostic};
  if (const auto *rejected =
          std::get_if<::loom::mapping::RejectedSystemMappingBase>(
              &verification))
    return IncompleteSystemPnrGeneration{
        IncompleteSystemPnrGenerationReason::ProofNotEstablished, accounting,
        rejected->diagnostic};
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
  return GeneratedSystemMappings{{finalized->reference()}, accounting};
}

} // namespace loom::pnr
