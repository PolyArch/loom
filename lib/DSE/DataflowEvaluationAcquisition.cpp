#include "DSE/DataflowEvaluationAcquisition.h"

#include "Common/ArtifactStore.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
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
  StructuredParentInput,
  FabricInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {PromotionAcquisitionInputSlotRef(CandidateInput), "candidates",
         PlanValueRole::CandidateSet, &dataflow::canonicalDataflowSchema,
         PlanValueCardinality::FiniteSet},
        {PromotionAcquisitionInputSlotRef(StructuredParentInput),
         "selected_structured_parent", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::ExactlyOne},
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
                                 "dataflow_evaluation_acquisition_invalid: " +
                                     message);
}

const PromotionAcquisitionDescriptor descriptor{
    dataflowEvaluationPromotionAcquisitionKind,
    "compiler.dataflow_evaluation",
    "loom.compiler.dataflow_evaluation.acquisition.v1",
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
             const ArtifactStore &store) {
  const auto analytic =
      evaluation::models::canonicalDataflowFabricAnalyticModelDescriptorRef();
  const auto functional =
      evaluation::models::canonicalDataflowFunctionalModelDescriptorRef();
  const ArtifactRootReference &parent =
      singleInput(inputBindings, StructuredParentInput);
  const ArtifactRootReference &fabric = singleInput(inputBindings, FabricInput);
  const ArtifactRootReference &workload =
      singleInput(inputBindings, WorkloadInput);
  const ArtifactRootReference &runtimeInput =
      singleInput(inputBindings, RuntimeInput);

  std::shared_ptr<const evaluation::CaseArtifactResolution> analyticResolution;
  std::shared_ptr<const evaluation::CaseArtifactResolution>
      functionalResolution;
  StructuredOwnershipInvocation *ownershipInvocation =
      detail::StructuredOwnershipInvocationAccess::current();
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation)
      return invalid("task has no Evidence obligation");
    if (!llvm::binary_search(inputBindings[CandidateInput].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("task candidate is outside the bound candidate set");
    const auto model = task.obligation->modelBinding().descriptorRef();
    if (model == analytic) {
      if (!analyticResolution) {
        auto resolution =
            evaluation::models::resolveCanonicalDataflowFabricEvaluationCases(
                inputBindings[CandidateInput].artifacts, fabric, store);
        if (!resolution)
          return resolution.takeError();
        analyticResolution =
            std::make_shared<const evaluation::CaseArtifactResolution>(
                std::move(*resolution));
      }
      resolved.push_back({0, analyticResolution, std::nullopt});
      continue;
    }
    if (model != functional)
      return invalid("task references a non-Dataflow Evaluation model");
    if (!ownershipInvocation)
      return PromotionAcquisitionResolutionOutcome{
          IncompletePromotionAcquisitionResolution{
              PromotionAcquisitionIncompleteReason::Unsupported}};
    if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
            primeDataflowFunctionalReplay(*ownershipInvocation, parent,
                                          task.candidate, store))
      return std::move(error);
    if (!functionalResolution) {
      auto resolution =
          evaluation::models::resolveCanonicalDataflowFunctionalEvaluationCases(
              inputBindings[CandidateInput].artifacts, parent, workload,
              runtimeInput, store);
      if (!resolution)
        return resolution.takeError();
      functionalResolution =
          std::make_shared<const evaluation::CaseArtifactResolution>(
              std::move(*resolution));
    }
    resolved.push_back({0, functionalResolution, std::nullopt});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider provider{descriptor.reference(),
                                            &resolveCases};

} // namespace

const PromotionAcquisitionDescriptor &
dataflowEvaluationPromotionAcquisitionDescriptor() {
  return descriptor;
}

llvm::Error registerDataflowEvaluationPromotionAcquisition() {
  if (llvm::Error error =
          evaluation::models::registerCanonicalDataflowFabricAnalyticModel())
    return error;
  if (llvm::Error error =
          evaluation::models::registerCanonicalDataflowFunctionalModel())
    return error;
  if (evaluation::models::canonicalDataflowFabricAnalyticCandidateRole() !=
      evaluation::models::canonicalDataflowFunctionalCandidateRole())
    return invalid("Dataflow models disagree on the candidate role");
  if (evaluation::models::canonicalDataflowFabricAnalyticCandidateRole() !=
      descriptor.candidateRole)
    return invalid("descriptor candidate role differs from the model owner");
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(descriptor))
    return error;
  return registerPromotionAcquisitionProvider(provider);
}

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindDataflowEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error = registerDataflowEvaluationPromotionAcquisition())
    return std::move(error);
  std::vector<ArtifactRootReference> candidates(
      canonicalDataflowPrograms.begin(), canonicalDataflowPrograms.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<PromotionAcquisitionInputBinding> bindings = {
      {PromotionAcquisitionInputSlotRef(CandidateInput), std::move(candidates)},
      {PromotionAcquisitionInputSlotRef(StructuredParentInput),
       {structuredParent}},
      {PromotionAcquisitionInputSlotRef(FabricInput), {fabric}},
      {PromotionAcquisitionInputSlotRef(WorkloadInput), {workload}},
      {PromotionAcquisitionInputSlotRef(RuntimeInput), {runtimeInput}},
  };
  if (llvm::Error error = validatePromotionAcquisitionInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveDataflowEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config) {
  if (llvm::Error error = registerDataflowEvaluationPromotionAcquisition())
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<EvidenceObligationTemplate>
prepareCanonicalDataflowFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabric, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (llvm::Error error = registerDataflowEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared = evaluation::models::prepareCanonicalDataflowFabricEvaluation(
      prototypeCandidate, fabric, config, store, blobs);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, prepared->candidateRole,
      {{evaluation::models::canonicalDataflowFabricAnalyticFabricRole(),
        EvidenceAcquisitionInputSlotRef(FabricInput)}});
}

llvm::Expected<EvidenceObligationTemplate>
prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (llvm::Error error = registerDataflowEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared =
      evaluation::models::prepareCanonicalDataflowFunctionalEvaluation(
          prototypeCandidate, structuredParent, workload, runtimeInput, config,
          store, blobs);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, prepared->candidateRole,
      {{evaluation::models::canonicalDataflowFunctionalStructuredParentRole(),
        EvidenceAcquisitionInputSlotRef(StructuredParentInput)}});
}

} // namespace loom::dse
