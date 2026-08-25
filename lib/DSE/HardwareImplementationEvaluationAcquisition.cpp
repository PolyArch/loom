#include "DSE/HardwareImplementationEvaluationAcquisition.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/Models/OpenRoadStaticFpa.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Hardware/Implementation/HardwareImplementation.h"

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
  InputSlotCount,
};

constexpr std::array<PromotionAcquisitionInputSlotDescriptor, InputSlotCount>
    inputSlots = {{{PromotionAcquisitionInputSlotRef(CandidateInput),
                    "hardware_implementations", PlanValueRole::CandidateSet,
                    &hardware::hardwareImplementationSchema,
                    PlanValueCardinality::NonEmptySet}}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "hardware_implementation_evaluation_acquisition_invalid: " + message);
}

const PromotionAcquisitionDescriptor descriptor{
    builtinPromotionAcquisitionKind(
        BuiltinPromotionAcquisition::HardwareImplementationEvaluation),
    "hardware.implementation_evaluation",
    "loom.hardware.implementation_evaluation.acquisition.v1",
    inputSlots,
    PromotionAcquisitionInputSlotRef(CandidateInput),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{
        resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
    &resolveEvidenceObligationSetConfig,
};

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(const ResolvedPromotionAcquisitionBinding &,
             llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store, const BlobStore &blobs) {
  const evaluation::EvaluationModelDescriptorRef model =
      evaluation::models::openRoadStaticFpaModelDescriptorRef();
  std::optional<ArtifactRootReference> cachedCandidate;
  std::shared_ptr<const evaluation::CaseArtifactResolution> cachedResolution;
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!externalContracts)
    return externalContracts.takeError();
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation)
      return invalid("task has no Evidence obligation");
    if (task.obligation->modelBinding().descriptorRef() != model)
      return invalid("task references a non-OpenROAD Evaluation model");
    if (!llvm::binary_search(inputBindings[CandidateInput].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("task candidate is outside the bound candidate set");
    if (!cachedCandidate || *cachedCandidate != task.candidate) {
      auto resolution =
          evaluation::models::resolveHardwareImplementationPhysicalCase(
              task.candidate, *externalContracts, store, blobs);
      if (!resolution)
        return resolution.takeError();
      cachedCandidate = task.candidate;
      cachedResolution =
          std::make_shared<const evaluation::CaseArtifactResolution>(
              std::move(*resolution));
    }
    resolved.push_back({0, cachedResolution, std::nullopt});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

const PromotionAcquisitionProvider provider{descriptor.reference(),
                                            &resolveCases};

} // namespace

const PromotionAcquisitionDescriptor &
hardwareImplementationEvaluationPromotionAcquisitionDescriptor() {
  return descriptor;
}

llvm::Error registerHardwareImplementationEvaluationPromotionAcquisition() {
  if (llvm::Error error = evaluation::models::registerOpenRoadStaticFpaModel())
    return error;
  if (evaluation::models::hardwareImplementationPhysicalSubjectRole() !=
      descriptor.candidateRole)
    return invalid("descriptor candidate role differs from the model owner");
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(descriptor))
    return error;
  return registerPromotionAcquisitionProvider(provider);
}

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindHardwareImplementationEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> hardwareImplementations) {
  if (llvm::Error error =
          registerHardwareImplementationEvaluationPromotionAcquisition())
    return std::move(error);
  std::vector<ArtifactRootReference> candidates(hardwareImplementations.begin(),
                                                hardwareImplementations.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<PromotionAcquisitionInputBinding> bindings = {
      {PromotionAcquisitionInputSlotRef(CandidateInput),
       std::move(candidates)}};
  if (llvm::Error error = validatePromotionAcquisitionInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveHardwareImplementationEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config) {
  if (llvm::Error error =
          registerHardwareImplementationEvaluationPromotionAcquisition())
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<EvidenceObligationTemplate>
prepareOpenRoadStaticFpaEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeHardwareImplementation,
    llvm::ArrayRef<evaluation::EvaluationCondition> conditions,
    llvm::ArrayRef<evaluation::MetricKind> metrics,
    std::optional<CalibrationPartitionRole> calibrationPartitionRole,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  if (llvm::Error error =
          registerHardwareImplementationEvaluationPromotionAcquisition())
    return std::move(error);
  auto prepared = evaluation::models::prepareOpenRoadStaticFpaEvaluation(
      prototypeHardwareImplementation, conditions, metrics, config,
      artifactStore, blobStore);
  if (!prepared)
    return prepared.takeError();
  return EvidenceObligationTemplate::get(
      prepared->request, prepared->candidateRole, {}, calibrationPartitionRole);
}

} // namespace loom::dse
