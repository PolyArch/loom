#include "DSE/StructuredEvaluationAcquisition.h"

#include "Common/ArtifactStore.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  CandidateInput,
  FabricInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {PromotionAcquisitionInputSlotRef(CandidateInput), "candidates",
         PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {PromotionAcquisitionInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {PromotionAcquisitionInputSlotRef(WorkloadInput), "workload",
         PlanValueRole::CandidateSet, &sim::simulationWorkloadSchema,
         PlanValueCardinality::ExactlyOne},
        {PromotionAcquisitionInputSlotRef(RuntimeInput), "runtime_input",
         PlanValueRole::CandidateSet, &sim::simulationRuntimeInputSchema,
         PlanValueCardinality::ExactlyOne},
    }};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_evaluation_acquisition_invalid: " +
                                     message);
}

const PromotionAcquisitionDescriptor descriptor{
    builtinPromotionAcquisitionKind(
        BuiltinPromotionAcquisition::StructuredEvaluation),
    "compiler.structured_evaluation",
    "loom.compiler.structured_evaluation.acquisition.v1",
    inputSlots,
    PromotionAcquisitionInputSlotRef(CandidateInput),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{
        resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
    &resolveEvidenceObligationSetConfig,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<PromotionAcquisitionInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(const ResolvedPromotionAcquisitionBinding &,
             llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store, const BlobStore &) {
  auto invocation =
      evaluation::models::prepareStructuredFabricAnalyticInvocation(
          inputBindings[CandidateInput].artifacts,
          singleInput(inputBindings, FabricInput),
          singleInput(inputBindings, WorkloadInput),
          singleInput(inputBindings, RuntimeInput), store);
  if (!invocation)
    return invocation.takeError();
  auto resolution = std::make_shared<const evaluation::CaseArtifactResolution>(
      invocation->caseResolution());

  const evaluation::EvaluationModelDescriptorRef analytic =
      evaluation::models::structuredFabricAnalyticModelDescriptorRef();
  const evaluation::EvaluationModelDescriptorRef functional =
      evaluation::models::structuredProgramFunctionalModelDescriptorRef();
  StructuredOwnershipInvocation *ownershipInvocation =
      detail::StructuredOwnershipInvocationAccess::current();
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation)
      return invalid("task has no Evidence obligation");
    const auto model = task.obligation->modelBinding().descriptorRef();
    if (model != analytic && model != functional)
      return invalid("task references a non-Structured Evaluation model");
    if (!llvm::binary_search(inputBindings[CandidateInput].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("task candidate is outside the bound candidate set");
    if (model == analytic && ownershipInvocation)
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              primeAnalyticCandidate(*ownershipInvocation, task.candidate,
                                     store))
        return std::move(error);
    if (model == functional) {
      if (!ownershipInvocation)
        return PromotionAcquisitionResolutionOutcome{
            IncompletePromotionAcquisitionResolution{
                PromotionAcquisitionIncompleteReason::Unsupported}};
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              primeFunctionalReplay(*ownershipInvocation, task.candidate,
                                    store))
        return std::move(error);
    }
    resolved.push_back({0, resolution, std::nullopt});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider provider{descriptor.reference(),
                                            &resolveCases};

} // namespace

const PromotionAcquisitionDescriptor &
structuredEvaluationPromotionAcquisitionDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredEvaluationPromotionAcquisition() {
  if (llvm::Error error =
          evaluation::models::registerStructuredFabricAnalyticModel())
    return error;
  if (llvm::Error error =
          evaluation::models::registerStructuredProgramFunctionalModel())
    return error;
  if (evaluation::models::structuredFabricAnalyticCandidateRole() !=
      evaluation::models::structuredProgramFunctionalCandidateRole())
    return invalid("Structured models disagree on the candidate role");
  if (evaluation::models::structuredFabricAnalyticCandidateRole() !=
      descriptor.candidateRole)
    return invalid("descriptor candidate role differs from the model owner");
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(descriptor))
    return error;
  return registerPromotionAcquisitionProvider(provider);
}

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindStructuredEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  std::vector<ArtifactRootReference> candidates(structuredPrograms.begin(),
                                                structuredPrograms.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<PromotionAcquisitionInputBinding> bindings = {
      {PromotionAcquisitionInputSlotRef(CandidateInput), std::move(candidates)},
      {PromotionAcquisitionInputSlotRef(FabricInput), {fabricReference}},
      {PromotionAcquisitionInputSlotRef(WorkloadInput), {workload}},
      {PromotionAcquisitionInputSlotRef(RuntimeInput), {runtimeInput}},
  };
  if (llvm::Error error = validatePromotionAcquisitionInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveStructuredEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared = evaluation::models::prepareStructuredFabricEvaluation(
      prototypeCandidate, fabricReference, workload, runtimeInput, config,
      store, blobs);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, prepared->candidateRole,
      {{evaluation::models::structuredFabricAnalyticFabricRole(),
        EvidenceAcquisitionInputSlotRef(FabricInput)}});
}

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredProgramFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (llvm::Error error = registerStructuredEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared =
      evaluation::models::prepareStructuredProgramFunctionalEvaluation(
          prototypeCandidate, workload, runtimeInput, config, store, blobs);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(prepared->request,
                                         prepared->candidateRole, {});
}

} // namespace loom::dse
