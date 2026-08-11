#include "DSE/SpatialMappingEvaluationAcquisition.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t {
  CandidateInput,
  DataflowInput,
  FabricInput,
  WorkloadInput,
  RuntimeInput,
  InputSlotCount,
};

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {PromotionAcquisitionInputSlotRef(CandidateInput), "spatial_mappings",
         PlanValueRole::CandidateSet, &mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {PromotionAcquisitionInputSlotRef(DataflowInput), "dataflow_owners",
         PlanValueRole::CandidateSet, &dataflow::canonicalDataflowSchema,
         PlanValueCardinality::NonEmptySet},
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
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "spatial_mapping_evaluation_acquisition_invalid: " + message);
}

const PromotionAcquisitionDescriptor descriptor{
    spatialMappingEvaluationPromotionAcquisitionKind,
    "mapping.spatial_cgra_evaluation",
    "loom.mapping.spatial_cgra_evaluation.acquisition.v1",
    inputSlots,
    PromotionAcquisitionInputSlotRef(CandidateInput),
    evaluation::CaseSubjectRoleRef(2),
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
  const ArtifactRootReference &fabric = singleInput(inputBindings, FabricInput);
  const ArtifactRootReference &workload =
      singleInput(inputBindings, WorkloadInput);
  const ArtifactRootReference &runtimeInput =
      singleInput(inputBindings, RuntimeInput);
  const auto model = evaluation::models::cgraSimulationModelDescriptorRef();

  std::optional<ArtifactRootReference> cachedCandidate;
  std::optional<ArtifactRootReference> cachedDataflow;
  std::shared_ptr<const evaluation::CaseArtifactResolution> cachedResolution;
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation)
      return invalid("task has no Evidence obligation");
    if (task.obligation->modelBinding().descriptorRef() != model)
      return invalid("task references a non-CGRA Evaluation model");
    if (task.obligation->workload() != workload ||
        task.obligation->runtimeInput() != runtimeInput)
      return invalid("task changes the exact workload or runtime input");
    if (!llvm::binary_search(inputBindings[CandidateInput].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("task candidate is outside the bound candidate set");

    if (!cachedCandidate || *cachedCandidate != task.candidate) {
      auto owners = evaluation::models::resolveCgraSimulationCase(
          task.candidate, workload, runtimeInput, store);
      if (!owners)
        return owners.takeError();
      if (owners->fabric != fabric)
        return invalid("SpatialMapping names a foreign Fabric owner");
      if (!llvm::binary_search(inputBindings[DataflowInput].artifacts,
                               owners->canonicalDataflow,
                               artifactRootReferenceLess))
        return invalid("SpatialMapping Dataflow owner is outside the bound "
                       "input set");
      cachedCandidate = task.candidate;
      cachedDataflow = owners->canonicalDataflow;
      cachedResolution =
          std::make_shared<const evaluation::CaseArtifactResolution>(
              std::move(owners->resolution));
    }

    if (!cachedDataflow || !cachedResolution)
      return invalid("candidate case resolution cache is incomplete");
    std::vector<EvidenceAcquisitionInputBinding> selectedInputs = {
        {EvidenceAcquisitionInputSlotRef(DataflowInput), {*cachedDataflow}},
        {EvidenceAcquisitionInputSlotRef(FabricInput), {fabric}},
    };
    resolved.push_back({0, cachedResolution, std::move(selectedInputs)});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider provider{descriptor.reference(),
                                            &resolveCases};

} // namespace

const PromotionAcquisitionDescriptor &
spatialMappingEvaluationPromotionAcquisitionDescriptor() {
  return descriptor;
}

llvm::Error registerSpatialMappingEvaluationPromotionAcquisition() {
  if (llvm::Error error = evaluation::models::registerCgraSimulationModel())
    return error;
  if (evaluation::models::cgraSimulationSpatialMappingRole() !=
      descriptor.candidateRole)
    return invalid("descriptor candidate role differs from the model owner");
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(descriptor))
    return error;
  return registerPromotionAcquisitionProvider(provider);
}

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindSpatialMappingEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  if (llvm::Error error =
          registerSpatialMappingEvaluationPromotionAcquisition())
    return std::move(error);
  std::vector<ArtifactRootReference> candidates(spatialMappings.begin(),
                                                spatialMappings.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<ArtifactRootReference> dataflows(
      canonicalDataflowPrograms.begin(), canonicalDataflowPrograms.end());
  llvm::sort(dataflows, artifactRootReferenceLess);
  dataflows.erase(std::unique(dataflows.begin(), dataflows.end()),
                  dataflows.end());
  std::vector<PromotionAcquisitionInputBinding> bindings = {
      {PromotionAcquisitionInputSlotRef(CandidateInput), std::move(candidates)},
      {PromotionAcquisitionInputSlotRef(DataflowInput), std::move(dataflows)},
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
resolveSpatialMappingEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config) {
  if (llvm::Error error =
          registerSpatialMappingEvaluationPromotionAcquisition())
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<EvidenceObligationTemplate>
prepareCgraSimulationEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeDataflow,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &prototypeSpatialMapping,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (llvm::Error error =
          registerSpatialMappingEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared = evaluation::models::prepareCgraSimulationEvaluation(
      prototypeDataflow, fabric, prototypeSpatialMapping, workload,
      runtimeInput, config, store, blobs);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, evaluation::models::cgraSimulationSpatialMappingRole(),
      {{evaluation::models::cgraSimulationProgramRole(),
        EvidenceAcquisitionInputSlotRef(DataflowInput)},
       {evaluation::models::cgraSimulationHardwareRole(),
        EvidenceAcquisitionInputSlotRef(FabricInput)}});
}

} // namespace loom::dse
