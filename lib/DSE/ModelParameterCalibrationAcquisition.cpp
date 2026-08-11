#include "DSE/ModelParameterCalibrationAcquisition.h"

#include "Config/ResolvedConfig.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr PromotionAcquisitionInputSlotRef kCandidateInput(0);
constexpr PromotionAcquisitionInputSlotRef kEvidenceInput(1);
constexpr evaluation::CaseSubjectRoleRef kCandidateRole(0);
constexpr evaluation::CaseSubjectRoleRef kEvidenceRole(1);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "model_parameter_calibration_acquisition_invalid: " + message);
}

const std::array<PromotionAcquisitionInputSlotDescriptor, 2> &
fpaValidationInputs() {
  static const std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputs = {
      {{kCandidateInput, "parameter_bundles", PlanValueRole::CandidateSet,
        &evaluation::modelParameterBundleSchema,
        PlanValueCardinality::FiniteSet,
        &evaluation::models::fpaModelParameterContractRef()},
       {kEvidenceInput, "validation_evidence", PlanValueRole::EvidenceSet,
        &evaluation::EvaluationEvidence::artifactSchema,
        PlanValueCardinality::NonEmptySet, nullptr,
        CalibrationPartitionRole::Validation}}};
  return inputs;
}

const std::array<PromotionAcquisitionInputSlotDescriptor, 2> &
fpaHeldOutInputs() {
  static const std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputs = {
      {{kCandidateInput, "parameter_bundles", PlanValueRole::CandidateSet,
        &evaluation::modelParameterBundleSchema,
        PlanValueCardinality::FiniteSet,
        &evaluation::models::fpaModelParameterContractRef()},
       {kEvidenceInput, "held_out_evidence", PlanValueRole::EvidenceSet,
        &evaluation::EvaluationEvidence::artifactSchema,
        PlanValueCardinality::NonEmptySet, nullptr,
        CalibrationPartitionRole::HeldOut}}};
  return inputs;
}

const std::array<PromotionAcquisitionInputSlotDescriptor, 2> &
runtimeValidationInputs() {
  static const std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputs = {
      {{kCandidateInput, "parameter_bundles", PlanValueRole::CandidateSet,
        &evaluation::modelParameterBundleSchema,
        PlanValueCardinality::FiniteSet,
        &evaluation::models::systemRuntimeModelParameterContractRef()},
       {kEvidenceInput, "validation_evidence", PlanValueRole::EvidenceSet,
        &evaluation::EvaluationEvidence::artifactSchema,
        PlanValueCardinality::NonEmptySet, nullptr,
        CalibrationPartitionRole::Validation}}};
  return inputs;
}

const std::array<PromotionAcquisitionInputSlotDescriptor, 2> &
runtimeHeldOutInputs() {
  static const std::array<PromotionAcquisitionInputSlotDescriptor, 2> inputs = {
      {{kCandidateInput, "parameter_bundles", PlanValueRole::CandidateSet,
        &evaluation::modelParameterBundleSchema,
        PlanValueCardinality::FiniteSet,
        &evaluation::models::systemRuntimeModelParameterContractRef()},
       {kEvidenceInput, "held_out_evidence", PlanValueRole::EvidenceSet,
        &evaluation::EvaluationEvidence::artifactSchema,
        PlanValueCardinality::NonEmptySet, nullptr,
        CalibrationPartitionRole::HeldOut}}};
  return inputs;
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveCases(ModelParameterCalibrationTarget target,
             const ResolvedPromotionAcquisitionBinding &,
             llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
             llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
             const ArtifactStore &store, const BlobStore &blobs) {
  if (inputBindings.size() != 2 ||
      !(inputBindings[kCandidateInput.ordinal()].slot == kCandidateInput) ||
      !(inputBindings[kEvidenceInput.ordinal()].slot == kEvidenceInput))
    return invalid("provider received a malformed input table");

  const evaluation::EvaluationModelDescriptorRef expectedModel =
      llvm::cantFail(evaluation::builtinEvaluationModelDescriptorRef(
          target == ModelParameterCalibrationTarget::Fpa
              ? evaluation::BuiltinEvaluationModel::FpaModelParameterCalibration
              : evaluation::BuiltinEvaluationModel::
                    SystemRuntimeModelParameterCalibration));
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> resolved;
  resolved.reserve(tasks.size());
  for (const PromotionEvidenceAcquisitionTask &task : tasks) {
    if (!task.obligation ||
        task.obligation->modelBinding().descriptorRef() != expectedModel ||
        !llvm::binary_search(inputBindings[kCandidateInput.ordinal()].artifacts,
                             task.candidate, artifactRootReferenceLess))
      return invalid("provider received a foreign calibration task");

    llvm::Expected<evaluation::CaseArtifactResolution> resolution =
        target == ModelParameterCalibrationTarget::Fpa
            ? evaluation::models::resolveFpaCalibrationCaseArtifactResolution(
                  task.candidate,
                  inputBindings[kEvidenceInput.ordinal()].artifacts, store,
                  blobs)
            : evaluation::models::
                  resolveSystemRuntimeCalibrationCaseArtifactResolution(
                      task.candidate,
                      inputBindings[kEvidenceInput.ordinal()].artifacts, store,
                      blobs);
    if (!resolution)
      return resolution.takeError();
    resolved.push_back(
        {0,
         std::make_shared<const evaluation::CaseArtifactResolution>(
             std::move(*resolution)),
         std::nullopt});
  }
  return PromotionAcquisitionResolutionOutcome{
      CompletedPromotionAcquisitionResolution{std::move(resolved)}};
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveFpaValidation(const ResolvedPromotionAcquisitionBinding &binding,
                     llvm::ArrayRef<PromotionAcquisitionInputBinding> inputs,
                     llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
                     const ArtifactStore &store, const BlobStore &blobs) {
  return resolveCases(ModelParameterCalibrationTarget::Fpa, binding, inputs,
                      tasks, store, blobs);
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveFpaHeldOut(const ResolvedPromotionAcquisitionBinding &binding,
                  llvm::ArrayRef<PromotionAcquisitionInputBinding> inputs,
                  llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
                  const ArtifactStore &store, const BlobStore &blobs) {
  return resolveCases(ModelParameterCalibrationTarget::Fpa, binding, inputs,
                      tasks, store, blobs);
}

llvm::Expected<PromotionAcquisitionResolutionOutcome> resolveRuntimeValidation(
    const ResolvedPromotionAcquisitionBinding &binding,
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputs,
    llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
    const ArtifactStore &store, const BlobStore &blobs) {
  return resolveCases(ModelParameterCalibrationTarget::SystemRuntime, binding,
                      inputs, tasks, store, blobs);
}

llvm::Expected<PromotionAcquisitionResolutionOutcome>
resolveRuntimeHeldOut(const ResolvedPromotionAcquisitionBinding &binding,
                      llvm::ArrayRef<PromotionAcquisitionInputBinding> inputs,
                      llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
                      const ArtifactStore &store, const BlobStore &blobs) {
  return resolveCases(ModelParameterCalibrationTarget::SystemRuntime, binding,
                      inputs, tasks, store, blobs);
}

const std::array<PromotionAcquisitionDescriptor, 4> &descriptors() {
  static const std::array<PromotionAcquisitionDescriptor, 4> values = {{
      {PromotionAcquisitionKind(3),
       "model.fpa_validation",
       "loom.model.fpa_validation.acquisition.v1",
       fpaValidationInputs(),
       kCandidateInput,
       kCandidateRole,
       {resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
       &resolveEvidenceObligationSetConfig},
      {PromotionAcquisitionKind(4),
       "model.fpa_held_out",
       "loom.model.fpa_held_out.acquisition.v1",
       fpaHeldOutInputs(),
       kCandidateInput,
       kCandidateRole,
       {resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
       &resolveEvidenceObligationSetConfig},
      {PromotionAcquisitionKind(5),
       "model.system_runtime_validation",
       "loom.model.system_runtime_validation.acquisition.v1",
       runtimeValidationInputs(),
       kCandidateInput,
       kCandidateRole,
       {resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
       &resolveEvidenceObligationSetConfig},
      {PromotionAcquisitionKind(6),
       "model.system_runtime_held_out",
       "loom.model.system_runtime_held_out.acquisition.v1",
       runtimeHeldOutInputs(),
       kCandidateInput,
       kCandidateRole,
       {resolvedEvidenceObligationSetConfigSchemaBytes(),
        validateResolvedEvidenceObligationSetConfigView},
       &resolveEvidenceObligationSetConfig},
  }};
  return values;
}

std::size_t descriptorIndex(ModelParameterCalibrationTarget target,
                            CalibrationPartitionRole partition) {
  const std::size_t base =
      target == ModelParameterCalibrationTarget::Fpa ? 0 : 2;
  return base + (partition == CalibrationPartitionRole::HeldOut ? 1 : 0);
}

} // namespace

const PromotionAcquisitionDescriptor &
modelParameterCalibrationPromotionAcquisitionDescriptor(
    ModelParameterCalibrationTarget target,
    CalibrationPartitionRole partition) {
  return descriptors()[descriptorIndex(target, partition)];
}

llvm::Error registerModelParameterCalibrationPromotionAcquisitions() {
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return error;
  const std::array<PromotionAcquisitionProviderFunction, 4> providers = {
      &resolveFpaValidation, &resolveFpaHeldOut, &resolveRuntimeValidation,
      &resolveRuntimeHeldOut};
  for (std::size_t index = 0; index != descriptors().size(); ++index) {
    if (llvm::Error error =
            registerPromotionAcquisitionDescriptor(descriptors()[index]))
      return error;
    if (llvm::Error error = registerPromotionAcquisitionProvider(
            {descriptors()[index].reference(), providers[index]}))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<EvidenceObligationTemplate>
prepareModelParameterCalibrationEvidenceObligationTemplate(
    ModelParameterCalibrationTarget target, CalibrationPartitionRole partition,
    evaluation::ExactRatio quantile, const ResolvedConfig &resolvedConfig) {
  if (partition == CalibrationPartitionRole::Training)
    return invalid("Training Evidence cannot be a calibration obligation");
  if (llvm::Error error =
          registerModelParameterCalibrationPromotionAcquisitions())
    return std::move(error);

  const evaluation::BuiltinEvaluationModel model =
      target == ModelParameterCalibrationTarget::Fpa
          ? evaluation::BuiltinEvaluationModel::FpaModelParameterCalibration
          : evaluation::BuiltinEvaluationModel::
                SystemRuntimeModelParameterCalibration;
  auto modelRef = evaluation::builtinEvaluationModelDescriptorRef(model);
  if (!modelRef)
    return modelRef.takeError();
  auto binding =
      evaluation::ResolvedModelBinding::project(*modelRef, {}, resolvedConfig);
  if (!binding)
    return binding.takeError();

  const evaluation::EvaluationCondition quantileCondition{
      evaluation::QuantileCondition{quantile}};
  std::vector<MetricRequestTemplate> metrics;
  if (target == ModelParameterCalibrationTarget::Fpa) {
    for (evaluation::MetricKind metric :
         {evaluation::MetricKind::LimitingClockFrequencyPredictionError,
          evaluation::MetricKind::TotalAreaPredictionError,
          evaluation::MetricKind::DynamicPowerPredictionError,
          evaluation::MetricKind::LeakagePowerPredictionError})
      metrics.push_back(
          {{metric,
            evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
           {quantileCondition}});
  } else {
    metrics.push_back(
        {{evaluation::MetricKind::RuntimePredictionError,
          evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
         {quantileCondition}});
  }
  llvm::sort(metrics, [](const MetricRequestTemplate &lhs,
                         const MetricRequestTemplate &rhs) {
    return evaluation::canonicalMetricRequestKey(lhs.query, lhs.conditions) <
           evaluation::canonicalMetricRequestKey(rhs.query, rhs.conditions);
  });
  return EvidenceObligationTemplate::get(
      std::move(*binding), {}, std::nullopt, std::nullopt, {},
      std::move(metrics), {}, kCandidateRole,
      {{kEvidenceRole, EvidenceAcquisitionInputSlotRef(1)}}, partition);
}

} // namespace loom::dse
