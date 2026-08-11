#include "Evaluation/Models/SystemRuntimeParameterContract.h"

#include "FixedTabularGbdt.h"

#include "Evaluation/OwnerError.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SimulationBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr std::uint32_t kIntegralFeatureCount = 35;
constexpr std::uint32_t kDecimalFeatureCount = 0;
constexpr std::uint32_t kCategoricalFeatureCount = 4;
constexpr std::uint32_t kPresenceFeatureCount = 1;
constexpr std::uint32_t kTargetCount = 1;

constexpr llvm::StringLiteral kParameterSchema =
    "loom.system_runtime.gbdt_parameter_payload.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_runtime_parameter_contract_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> parameterSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kParameterSchema.data()),
          kParameterSchema.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::int64_t> checkedFeature(std::uint64_t value,
                                            llvm::StringRef field) {
  constexpr std::uint64_t limit = std::uint64_t{1} << 40;
  if (value > limit)
    return invalid(field + " exceeds the admitted feature magnitude");
  return static_cast<std::int64_t>(value);
}

llvm::Expected<detail::FixedTabularFeatureView>
fixedFeatures(const SystemRuntimeFeatureView &features) {
  detail::FixedTabularFeatureView fixed;
  const std::array<std::pair<std::uint64_t, llvm::StringRef>,
                   kIntegralFeatureCount>
      integral = {{
          {features.deployment.instructionCoreBinaryCount,
           "instruction-core binary count"},
          {features.deployment.hardwareBindingCount, "hardware-binding count"},
          {features.deployment.configurationImageCount,
           "configuration-image count"},
          {features.deployment.staticMemoryImageCount,
           "static-memory image count"},
          {features.workload.entryValueInputCount, "entry value-input count"},
          {features.workload.runtimeEntryValueInputCount,
           "runtime entry value-input count"},
          {features.workload.externalValueInputCount,
           "external value-input count"},
          {features.workload.runtimeExternalValueInputCount,
           "runtime external value-input count"},
          {features.workload.valueResultCount, "value-result count"},
          {features.workload.externalValueOutputCount,
           "external value-output count"},
          {features.workload.externalStreamOutputCount,
           "external stream-output count"},
          {features.workload.memoryObservableCount, "memory-observable count"},
          {features.runtimeInput.runtimeEntryValueCount,
           "runtime entry-value count"},
          {features.runtimeInput.runtimeExternalValueCount,
           "runtime external-value count"},
          {features.runtimeInput.externalStreamInputCount,
           "external stream-input count"},
          {features.runtimeInput.memoryObjectCount, "memory-object count"},
          {features.runtimeInput.memoryInterfaceBindingCount,
           "memory-interface binding count"},
          {features.runtimeInput.memoryByteCount, "memory-byte count"},
          {features.runtimeInput.streamTokenCount, "stream-token count"},
          {features.mapping.instructionContextDomainCount,
           "instruction-context domain count"},
          {features.mapping.spatialContextDomainCount,
           "spatial-context domain count"},
          {features.mapping.serviceRealizationCount,
           "service-realization count"},
          {features.mapping.capacityCellCount, "capacity-cell count"},
          {features.mapping.resourceActivationCount,
           "resource-activation count"},
          {features.mapping.capacityClaimCount, "capacity-claim count"},
          {features.mapping.causalReleaseCount, "causal-release count"},
          {features.fabric.entityCount, "Fabric entity count"},
          {features.fabric.hostCoreOccurrenceCount,
           "host-core occurrence count"},
          {features.fabric.acceleratorCoreOccurrenceCount,
           "accelerator-core occurrence count"},
          {features.fabric.systemMemoryServiceCount,
           "System memory-service count"},
          {features.fabric.systemTransportResourceCount,
           "System transport-resource count"},
          {features.fabric.hardwareDomainCount, "hardware-domain count"},
          {features.fabric.transportEndpointCount, "transport-endpoint count"},
          {features.fabric.pointConnectionCount, "point-connection count"},
          {features.fabric.admittedTraversalCount, "admitted-traversal count"},
      }};
  fixed.integral.reserve(integral.size());
  for (const auto &[value, field] : integral) {
    auto admitted = checkedFeature(value, field);
    if (!admitted)
      return admitted.takeError();
    fixed.integral.push_back(*admitted);
  }
  fixed.categorical = {
      features.softwarePartitioningKey, features.modeledPlatformKey,
      features.gem5BindingFeatureKey, features.admittedRuntimeConditionKey};
  fixed.presence = {features.deployment.hasSpatialLaunchImage};
  return fixed;
}

llvm::Expected<OwnerValue> adoptParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = adoptSystemRuntimeGbdtParameters(bytes);
  if (!parameters)
    return parameters.takeError();
  return OwnerValue::get(std::move(*parameters));
}

llvm::Expected<std::vector<std::uint8_t>>
encodeParameters(const OwnerValue &value) {
  const auto *parameters = value.getIf<SystemRuntimeGbdtParameters>();
  if (!parameters)
    return invalid("parameter value has a foreign owner type");
  return encodeSystemRuntimeGbdtParameters(*parameters);
}

llvm::Expected<std::vector<std::uint8_t>>
parameterTargetKey(const OwnerValue &value) {
  const auto *parameters = value.getIf<SystemRuntimeGbdtParameters>();
  if (!parameters)
    return invalid("parameter value has a foreign owner type");
  return parameters->groundTruthTargetKey().vec();
}

llvm::Error unavailableGem5Owner() {
  return evaluationOwnerUnavailable(
      runtime::gem5SimulationBindingSchema.identity,
      runtime::gem5SimulationBindingSchema.version);
}

llvm::Expected<OwnerValue> projectFeatures(const EvaluationCase &,
                                           const CaseArtifactResolution &,
                                           const ArtifactStore &,
                                           const BlobStore &) {
  return unavailableGem5Owner();
}

llvm::Expected<ModelParameterInferenceOutcome>
inferParameters(const OwnerValue &parameters, const OwnerValue &features) {
  const auto *typedParameters = parameters.getIf<SystemRuntimeGbdtParameters>();
  const auto *typedFeatures = features.getIf<SystemRuntimeFeatureView>();
  if (!typedParameters || !typedFeatures)
    return invalid("inference received a foreign owner value");
  return inferSystemRuntimeGbdtParameters(*typedParameters, *typedFeatures);
}

llvm::Expected<std::vector<std::uint8_t>>
groundTruthTargetKey(const EvaluationRequest &, const CaseArtifactResolution &,
                     const ArtifactStore &, const BlobStore &) {
  return unavailableGem5Owner();
}

llvm::Expected<std::vector<std::uint8_t>>
calibrationSampleGroupKey(const EvaluationEvidence &, const EvaluationRequest &,
                          const CaseArtifactResolution &, const ArtifactStore &,
                          const BlobStore &) {
  return unavailableGem5Owner();
}

const std::vector<EvaluationCaseSignatureRef> &predictionCases() {
  static const std::vector<EvaluationCaseSignatureRef> values = {
      systemSimulationCaseSignatureRef()};
  return values;
}

const std::vector<EvaluationModelDescriptorRef> &groundTruthModels() {
  static const std::vector<EvaluationModelDescriptorRef> values = {
      llvm::cantFail(builtinEvaluationModelDescriptorRef(
          BuiltinEvaluationModel::Gem5SystemCgra))};
  return values;
}

const std::vector<ModelParameterConditionPatternSet> &conditionTable() {
  static const std::vector<ModelParameterConditionPatternSet> values = {
      {systemSimulationCaseSignatureRef(), systemSimulationCaseSignatureRef()
                                               .descriptor()
                                               ->permittedBaseConditions}};
  return values;
}

const ModelParameterContractDescriptor &descriptor() {
  static const ModelParameterContractDescriptor value{
      systemRuntimeModelParameterContractRef(),
      "Deterministic single-head System Runtime prediction over exact "
      "Deployment, workload, mapping, Fabric, gem5-binding, and runtime "
      "condition features.",
      predictionCases(),
      groundTruthModels(),
      conditionTable(),
      systemRuntimePredictionViewSchemaDescriptorBytes(),
      {18, ModelParameterDecimalRounding::RoundToNearestTiesToEven},
      &adoptParameters,
      &encodeParameters,
      &parameterTargetKey,
      &projectFeatures,
      &inferParameters,
      &groundTruthTargetKey,
      &calibrationSampleGroupKey};
  return value;
}

} // namespace

struct SystemRuntimeGbdtParameters::Storage final {
  detail::FixedTabularGbdtParameters parameters;
};

llvm::ArrayRef<std::uint8_t>
SystemRuntimeGbdtParameters::groundTruthTargetKey() const {
  return storage_ ? llvm::ArrayRef<std::uint8_t>(
                        storage_->parameters.groundTruthTargetKey)
                  : llvm::ArrayRef<std::uint8_t>();
}

const ModelParameterContractRef &systemRuntimeModelParameterContractRef() {
  static const ModelParameterContractRef reference = llvm::cantFail(
      ModelParameterContractRef::get("loom.system_runtime", {1, 0}, 0));
  return reference;
}

llvm::ArrayRef<std::uint8_t>
systemRuntimePredictionViewSchemaDescriptorBytes() {
  static const std::vector<std::uint8_t> bytes = [] {
    std::vector<std::uint8_t> result;
    constexpr llvm::StringLiteral owner = "loom.system_runtime.prediction_view";
    appendU64(result, owner.size());
    result.insert(result.end(), owner.bytes_begin(), owner.bytes_end());
    appendU32(result, 1);
    appendU32(result, 0);
    appendU64(result, 1);
    appendU32(result, static_cast<std::uint32_t>(MetricKind::Runtime));
    return result;
  }();
  return bytes;
}

const ModelParameterContractDescriptor &
systemRuntimeModelParameterContractDescriptor() {
  return descriptor();
}

llvm::Error registerSystemRuntimeModelParameterContract() {
  return registerModelParameterContract(descriptor());
}

llvm::Expected<SystemRuntimeGbdtParameters> trainSystemRuntimeGbdtParameters(
    llvm::ArrayRef<SystemRuntimeTrainingSample> training,
    const SystemRuntimeGbdtTrainingConfig &config,
    const SystemRuntimeGbdtParameters *prior) {
  if (training.empty())
    return invalid("Training partition is empty");
  const std::vector<std::uint8_t> &targetKey =
      training.front().groundTruthTargetKey;
  if (targetKey.empty())
    return invalid("Training target key is empty");
  std::vector<detail::FixedTabularTrainingRow> rows;
  rows.reserve(training.size());
  for (const SystemRuntimeTrainingSample &sample : training) {
    if (sample.groundTruthTargetKey != targetKey)
      return invalid("Training partition mixes ground-truth target keys");
    if (sample.sampleGroupKey.empty())
      return invalid("Training sample-group key is empty");
    auto features = fixedFeatures(sample.features);
    if (!features)
      return features.takeError();
    rows.push_back({std::move(*features), {sample.runtime}});
  }
  detail::DeterministicGbdtConfig trainingConfig{
      config.seed,
      config.treeCount,
      config.maximumDepth,
      config.minimumRowsPerLeaf,
      config.learningRateNumerator,
      config.learningRateDenominator};
  auto parameters = detail::trainFixedTabularGbdt(
      rows, targetKey, trainingConfig,
      prior && prior->storage_ ? &prior->storage_->parameters : nullptr);
  if (!parameters)
    return parameters.takeError();
  auto storage = std::make_shared<SystemRuntimeGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return SystemRuntimeGbdtParameters(std::move(storage));
}

llvm::Expected<SystemRuntimeGbdtParameters> adoptSystemRuntimeGbdtParameters(
    llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes) {
  auto parameters = detail::decodeFixedTabularGbdt(
      canonicalPayloadBytes, parameterSchemaBytes(), kIntegralFeatureCount,
      kDecimalFeatureCount, kCategoricalFeatureCount, kPresenceFeatureCount,
      kTargetCount);
  if (!parameters)
    return parameters.takeError();
  auto storage = std::make_shared<SystemRuntimeGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return SystemRuntimeGbdtParameters(std::move(storage));
}

llvm::Expected<std::vector<std::uint8_t>> encodeSystemRuntimeGbdtParameters(
    const SystemRuntimeGbdtParameters &parameters) {
  if (!parameters.storage_)
    return invalid("parameter storage is empty");
  return detail::encodeFixedTabularGbdt(parameters.storage_->parameters,
                                        parameterSchemaBytes());
}

llvm::Expected<ModelParameterInferenceOutcome>
inferSystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &parameters,
                                 const SystemRuntimeFeatureView &features) {
  if (!parameters.storage_)
    return invalid("parameter storage is empty");
  auto fixed = fixedFeatures(features);
  if (!fixed)
    return fixed.takeError();
  auto prediction =
      detail::inferFixedTabularGbdt(parameters.storage_->parameters, *fixed);
  if (!prediction)
    return prediction.takeError();
  if (!*prediction)
    return ModelParameterInferenceOutcome{UnsupportedModelParameterInference{}};
  if ((**prediction).size() != kTargetCount)
    return invalid("inference returned the wrong target count");
  return ModelParameterInferenceOutcome{ModelParameterPrediction{
      OwnerValue::get(SystemRuntimePredictionView{(**prediction)[0]})}};
}

} // namespace loom::evaluation::models
