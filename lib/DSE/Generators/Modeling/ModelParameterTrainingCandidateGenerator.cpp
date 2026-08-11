#include "DSE/ModelParameterTrainingCandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

using evaluation::models::FpaGbdtParameters;
using evaluation::models::FpaTrainingEvidenceSample;
using evaluation::models::SystemRuntimeGbdtParameters;
using evaluation::models::SystemRuntimeTrainingSample;

constexpr llvm::StringLiteral kConfigSchema =
    "loom.deterministic_gbdt_training.config.1.0";
constexpr std::uint32_t kMaximumTreeCount = 4096;
constexpr std::uint32_t kMaximumDepth = 31;
constexpr std::uint32_t kMaximumLearningRateDenominator = 1000000000;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "model_parameter_training_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kConfigSchema.data()),
          kConfigSchema.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(const llvm::Twine &field) {
    if (remaining() < 4)
      return invalid(field + " is truncated");
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> u64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Error validateConfig(const DeterministicGbdtTrainingConfig &config) {
  if (config.treeCount == 0 || config.treeCount > kMaximumTreeCount)
    return invalid("tree count is outside the admitted range");
  if (config.maximumDepth == 0 || config.maximumDepth > kMaximumDepth)
    return invalid("maximum depth is outside the admitted range");
  if (config.minimumTrainingRowsPerLeaf == 0)
    return invalid("minimum Training rows per leaf must be positive");
  if (config.learningRateNumerator == 0 ||
      config.learningRateDenominator == 0 ||
      config.learningRateNumerator > config.learningRateDenominator ||
      config.learningRateDenominator > kMaximumLearningRateDenominator)
    return invalid("learning rate is outside (0, 1]");
  return llvm::Error::success();
}

std::vector<std::uint8_t>
encodeConfig(const DeterministicGbdtTrainingConfig &config) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(28);
  appendU64(bytes, config.seed);
  appendU32(bytes, config.treeCount);
  appendU32(bytes, config.maximumDepth);
  appendU32(bytes, config.minimumTrainingRowsPerLeaf);
  appendU32(bytes, config.learningRateNumerator);
  appendU32(bytes, config.learningRateDenominator);
  return bytes;
}

llvm::Expected<DeterministicGbdtTrainingConfig>
decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto seed = decoder.u64("seed");
  if (!seed)
    return seed.takeError();
  auto treeCount = decoder.u32("tree count");
  if (!treeCount)
    return treeCount.takeError();
  auto maximumDepth = decoder.u32("maximum depth");
  if (!maximumDepth)
    return maximumDepth.takeError();
  auto minimumRows = decoder.u32("minimum Training rows per leaf");
  if (!minimumRows)
    return minimumRows.takeError();
  auto rateNumerator = decoder.u32("learning-rate numerator");
  if (!rateNumerator)
    return rateNumerator.takeError();
  auto rateDenominator = decoder.u32("learning-rate denominator");
  if (!rateDenominator)
    return rateDenominator.takeError();
  if (decoder.remaining() != 0)
    return invalid("config has trailing bytes");
  DeterministicGbdtTrainingConfig config{*seed,          *treeCount,
                                         *maximumDepth,  *minimumRows,
                                         *rateNumerator, *rateDenominator};
  if (llvm::Error error = validateConfig(config))
    return std::move(error);
  return config;
}

llvm::Error validateResolvedConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                   const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedDeterministicGbdtTrainingConfigView(
      configSchemaBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const std::array<CandidateGeneratorInputSlotDescriptor, 4> &fpaInputSlots() {
  static const std::array<CandidateGeneratorInputSlotDescriptor, 4> slots = {{
      {CandidateGeneratorInputSlotRef(0), "training_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::Training},
      {CandidateGeneratorInputSlotRef(1), "validation_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::Validation},
      {CandidateGeneratorInputSlotRef(2), "held_out_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::HeldOut},
      {CandidateGeneratorInputSlotRef(3), "prior_parameter_bundle",
       PlanValueRole::CandidateSet, &evaluation::modelParameterBundleSchema,
       PlanValueCardinality::ZeroOrOne,
       &evaluation::models::fpaModelParameterContractRef(), std::nullopt},
  }};
  return slots;
}

const std::array<CandidateGeneratorOutputSlotDescriptor, 1> &fpaOutputSlots() {
  static const std::array<CandidateGeneratorOutputSlotDescriptor, 1> slots = {{
      {CandidateGeneratorOutputSlotRef(0), "parameter_bundle",
       PlanValueRole::CandidateSet, &evaluation::modelParameterBundleSchema,
       PlanValueCardinality::ExactlyOne,
       &evaluation::models::fpaModelParameterContractRef(), std::nullopt},
  }};
  return slots;
}

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> kWorkUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "tree_head_fit"},
}};

const CandidateGeneratorDescriptor &fpaDescriptor() {
  static const CandidateGeneratorDescriptor descriptor{
      fpaGbdtTrainingCandidateGeneratorKind,
      "fpa_gbdt_training",
      "loom.fpa.gbdt_training.generator.v1",
      fpaInputSlots(),
      fpaOutputSlots(),
      ResolvedDseConfigViewContract{configSchemaBytes(),
                                    validateResolvedConfig},
      CandidateGeneratorDeterminism::Deterministic,
      kWorkUnits,
      nullptr,
      ProviderForm::InProcess};
  return descriptor;
}

const std::array<CandidateGeneratorInputSlotDescriptor, 4> &
systemRuntimeInputSlots() {
  static const std::array<CandidateGeneratorInputSlotDescriptor, 4> slots = {{
      {CandidateGeneratorInputSlotRef(0), "training_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::Training},
      {CandidateGeneratorInputSlotRef(1), "validation_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::Validation},
      {CandidateGeneratorInputSlotRef(2), "held_out_evidence",
       PlanValueRole::EvidenceSet,
       &evaluation::EvaluationEvidence::artifactSchema,
       PlanValueCardinality::NonEmptySet, nullptr,
       CalibrationPartitionRole::HeldOut},
      {CandidateGeneratorInputSlotRef(3), "prior_parameter_bundle",
       PlanValueRole::CandidateSet, &evaluation::modelParameterBundleSchema,
       PlanValueCardinality::ZeroOrOne,
       &evaluation::models::systemRuntimeModelParameterContractRef(),
       std::nullopt},
  }};
  return slots;
}

const std::array<CandidateGeneratorOutputSlotDescriptor, 1> &
systemRuntimeOutputSlots() {
  static const std::array<CandidateGeneratorOutputSlotDescriptor, 1> slots = {{
      {CandidateGeneratorOutputSlotRef(0), "parameter_bundle",
       PlanValueRole::CandidateSet, &evaluation::modelParameterBundleSchema,
       PlanValueCardinality::ExactlyOne,
       &evaluation::models::systemRuntimeModelParameterContractRef(),
       std::nullopt},
  }};
  return slots;
}

const CandidateGeneratorDescriptor &systemRuntimeDescriptor() {
  static const CandidateGeneratorDescriptor descriptor{
      systemRuntimeGbdtTrainingCandidateGeneratorKind,
      "system_runtime_gbdt_training",
      "loom.system_runtime.gbdt_training.generator.v1",
      systemRuntimeInputSlots(),
      systemRuntimeOutputSlots(),
      ResolvedDseConfigViewContract{configSchemaBytes(),
                                    validateResolvedConfig},
      CandidateGeneratorDeterminism::Deterministic,
      kWorkUnits,
      nullptr,
      ProviderForm::InProcess};
  return descriptor;
}

struct ImportedPartition final {
  std::vector<FpaTrainingEvidenceSample> samples;
  std::set<std::vector<std::uint8_t>> sampleGroups;
};

llvm::Expected<ImportedPartition>
importPartition(llvm::ArrayRef<ArtifactRootReference> evidence,
                const ArtifactStore &store, const BlobStore &blobs) {
  ImportedPartition partition;
  partition.samples.reserve(evidence.size());
  for (const ArtifactRootReference &reference : evidence) {
    auto sample = evaluation::models::importFpaTrainingEvidenceSample(
        reference, store, blobs);
    if (!sample)
      return sample.takeError();
    partition.sampleGroups.insert(sample->sampleGroupKey);
    partition.samples.push_back(std::move(*sample));
  }
  return partition;
}

bool intersects(const std::set<std::vector<std::uint8_t>> &lhs,
                const std::set<std::vector<std::uint8_t>> &rhs) {
  auto left = lhs.begin();
  auto right = rhs.begin();
  while (left != lhs.end() && right != rhs.end()) {
    if (*left < *right)
      ++left;
    else if (*right < *left)
      ++right;
    else
      return true;
  }
  return false;
}

llvm::Error validateTargetKeys(const ImportedPartition &training,
                               const ImportedPartition &validation,
                               const ImportedPartition &heldOut) {
  const std::vector<std::uint8_t> &target =
      training.samples.front().groundTruthTargetKey;
  for (const ImportedPartition *partition : {&training, &validation, &heldOut})
    for (const FpaTrainingEvidenceSample &sample : partition->samples)
      if (sample.groundTruthTargetKey != target)
        return invalid("Evidence partitions mix ground-truth target keys");
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeFpaProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
                  const ResolvedCandidateGeneratorBinding &binding,
                  const ArtifactStore &store, const BlobStore &blobs) {
  auto resolved = adoptResolvedDeterministicGbdtTrainingConfigView(
      configSchemaBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!resolved)
    return resolved.takeError();
  const DeterministicGbdtTrainingConfig &config = resolved->config();
  if (inputBindings[0].artifacts.size() < config.minimumTrainingRowsPerLeaf)
    return invalid("Training partition is smaller than one required leaf");

  auto training = importPartition(inputBindings[0].artifacts, store, blobs);
  if (!training)
    return training.takeError();
  auto validation = importPartition(inputBindings[1].artifacts, store, blobs);
  if (!validation)
    return validation.takeError();
  auto heldOut = importPartition(inputBindings[2].artifacts, store, blobs);
  if (!heldOut)
    return heldOut.takeError();
  if (intersects(training->sampleGroups, validation->sampleGroups) ||
      intersects(training->sampleGroups, heldOut->sampleGroups) ||
      intersects(validation->sampleGroups, heldOut->sampleGroups))
    return invalid("calibration sample groups overlap across partitions");
  if (llvm::Error error = validateTargetKeys(*training, *validation, *heldOut))
    return std::move(error);

  std::optional<evaluation::FinalizedModelParameterBundle> priorBundle;
  const FpaGbdtParameters *prior = nullptr;
  if (!inputBindings[3].artifacts.empty()) {
    auto imported = evaluation::importModelParameterBundle(
        inputBindings[3].artifacts.front(), store, blobs);
    if (!imported)
      return imported.takeError();
    priorBundle = std::move(*imported);
    prior = priorBundle->parametersIf<FpaGbdtParameters>();
    if (!prior)
      return invalid("prior bundle has a foreign parameter owner type");
    if (prior->groundTruthTargetKey() !=
        llvm::ArrayRef<std::uint8_t>(
            training->samples.front().groundTruthTargetKey))
      return invalid("prior bundle has a different ground-truth target key");
  }

  evaluation::models::FpaGbdtTrainingConfig trainingConfig{
      config.seed,
      config.treeCount,
      config.maximumDepth,
      config.minimumTrainingRowsPerLeaf,
      config.learningRateNumerator,
      config.learningRateDenominator};
  auto parameters = evaluation::models::trainFpaGbdtParameters(
      training->samples, trainingConfig, prior);
  if (!parameters)
    return parameters.takeError();
  evaluation::OwnerValue owner =
      evaluation::OwnerValue::get(std::move(*parameters));
  auto bundle = evaluation::finalizeModelParameterBundle(
      evaluation::models::fpaModelParameterContractRef(), owner, store, blobs);
  if (!bundle)
    return bundle.takeError();

  const std::uint64_t work = static_cast<std::uint64_t>(config.treeCount) * 4;
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {bundle->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            bundle->reference(),
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), work, work}}};
}

const CandidateGeneratorProvider &fpaProvider() {
  static const CandidateGeneratorProvider provider{
      fpaDescriptor().reference(),
      CandidateGeneratorInProcessProvider{invokeFpaProvider}};
  return provider;
}

struct ImportedSystemRuntimePartition final {
  std::vector<SystemRuntimeTrainingSample> samples;
  std::set<std::vector<std::uint8_t>> sampleGroups;
};

llvm::Expected<ImportedSystemRuntimePartition>
importSystemRuntimePartition(llvm::ArrayRef<ArtifactRootReference> evidence,
                             const ArtifactStore &store,
                             const BlobStore &blobs) {
  ImportedSystemRuntimePartition partition;
  partition.samples.reserve(evidence.size());
  for (const ArtifactRootReference &reference : evidence) {
    auto sample = evaluation::models::importSystemRuntimeTrainingEvidenceSample(
        reference, store, blobs);
    if (!sample)
      return sample.takeError();
    partition.sampleGroups.insert(sample->sampleGroupKey);
    partition.samples.push_back(std::move(*sample));
  }
  return partition;
}

llvm::Error validateSystemRuntimeTargetKeys(
    const ImportedSystemRuntimePartition &training,
    const ImportedSystemRuntimePartition &validation,
    const ImportedSystemRuntimePartition &heldOut) {
  const std::vector<std::uint8_t> &target =
      training.samples.front().groundTruthTargetKey;
  for (const ImportedSystemRuntimePartition *partition :
       {&training, &validation, &heldOut})
    for (const SystemRuntimeTrainingSample &sample : partition->samples)
      if (sample.groundTruthTargetKey != target)
        return invalid("Evidence partitions mix ground-truth target keys");
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeSystemRuntimeProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto resolved = adoptResolvedDeterministicGbdtTrainingConfigView(
      configSchemaBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!resolved)
    return resolved.takeError();
  const DeterministicGbdtTrainingConfig &config = resolved->config();
  if (inputBindings[0].artifacts.size() < config.minimumTrainingRowsPerLeaf)
    return invalid("Training partition is smaller than one required leaf");

  auto training =
      importSystemRuntimePartition(inputBindings[0].artifacts, store, blobs);
  if (!training)
    return training.takeError();
  auto validation =
      importSystemRuntimePartition(inputBindings[1].artifacts, store, blobs);
  if (!validation)
    return validation.takeError();
  auto heldOut =
      importSystemRuntimePartition(inputBindings[2].artifacts, store, blobs);
  if (!heldOut)
    return heldOut.takeError();
  if (intersects(training->sampleGroups, validation->sampleGroups) ||
      intersects(training->sampleGroups, heldOut->sampleGroups) ||
      intersects(validation->sampleGroups, heldOut->sampleGroups))
    return invalid("calibration sample groups overlap across partitions");
  if (llvm::Error error =
          validateSystemRuntimeTargetKeys(*training, *validation, *heldOut))
    return std::move(error);

  std::optional<evaluation::FinalizedModelParameterBundle> priorBundle;
  const SystemRuntimeGbdtParameters *prior = nullptr;
  if (!inputBindings[3].artifacts.empty()) {
    auto imported = evaluation::importModelParameterBundle(
        inputBindings[3].artifacts.front(), store, blobs);
    if (!imported)
      return imported.takeError();
    priorBundle = std::move(*imported);
    prior = priorBundle->parametersIf<SystemRuntimeGbdtParameters>();
    if (!prior)
      return invalid("prior bundle has a foreign parameter owner type");
    if (prior->groundTruthTargetKey() !=
        llvm::ArrayRef<std::uint8_t>(
            training->samples.front().groundTruthTargetKey))
      return invalid("prior bundle has a different ground-truth target key");
  }

  evaluation::models::SystemRuntimeGbdtTrainingConfig trainingConfig{
      config.seed,
      config.treeCount,
      config.maximumDepth,
      config.minimumTrainingRowsPerLeaf,
      config.learningRateNumerator,
      config.learningRateDenominator};
  auto parameters = evaluation::models::trainSystemRuntimeGbdtParameters(
      training->samples, trainingConfig, prior);
  if (!parameters)
    return parameters.takeError();
  evaluation::OwnerValue owner =
      evaluation::OwnerValue::get(std::move(*parameters));
  auto bundle = evaluation::finalizeModelParameterBundle(
      evaluation::models::systemRuntimeModelParameterContractRef(), owner,
      store, blobs);
  if (!bundle)
    return bundle.takeError();

  const std::uint64_t work = config.treeCount;
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {bundle->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            bundle->reference(),
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), work, work}}};
}

const CandidateGeneratorProvider &systemRuntimeProvider() {
  static const CandidateGeneratorProvider provider{
      systemRuntimeDescriptor().reference(),
      CandidateGeneratorInProcessProvider{invokeSystemRuntimeProvider}};
  return provider;
}

void canonicalizeReferences(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedDeterministicGbdtTrainingConfigSchemaBytes() {
  return configSchemaBytes();
}

llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
resolveDeterministicGbdtTrainingConfig(
    const DeterministicGbdtTrainingConfig &config) {
  if (llvm::Error error = validateConfig(config))
    return std::move(error);
  std::vector<std::uint8_t> bytes = encodeConfig(config);
  auto digest = computeComponentViewDigest(configSchemaBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedDeterministicGbdtTrainingConfigView(config, std::move(bytes),
                                                     *digest);
}

llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
adoptResolvedDeterministicGbdtTrainingConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != configSchemaBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto config = decodeConfig(canonicalViewBytes);
  if (!config)
    return config.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*config);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("config does not re-encode canonically");
  return ResolvedDeterministicGbdtTrainingConfigView(
      *config, std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
fpaGbdtTrainingCandidateGeneratorDescriptor() {
  return fpaDescriptor();
}

llvm::Error registerFpaGbdtTrainingCandidateGenerator() {
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return error;
  if (llvm::Error error = registerCandidateGeneratorDescriptor(fpaDescriptor()))
    return error;
  return registerCandidateGeneratorProvider(fpaProvider());
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFpaGbdtTrainingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> training,
    llvm::ArrayRef<ArtifactRootReference> validation,
    llvm::ArrayRef<ArtifactRootReference> heldOut,
    const std::optional<ArtifactRootReference> &priorBundle) {
  if (llvm::Error error = registerFpaGbdtTrainingCandidateGenerator())
    return std::move(error);
  std::vector<ArtifactRootReference> trainingSet = training.vec();
  std::vector<ArtifactRootReference> validationSet = validation.vec();
  std::vector<ArtifactRootReference> heldOutSet = heldOut.vec();
  canonicalizeReferences(trainingSet);
  canonicalizeReferences(validationSet);
  canonicalizeReferences(heldOutSet);
  std::vector<ArtifactRootReference> prior;
  if (priorBundle)
    prior.push_back(*priorBundle);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(0), std::move(trainingSet)},
      {CandidateGeneratorInputSlotRef(1), std::move(validationSet)},
      {CandidateGeneratorInputSlotRef(2), std::move(heldOutSet)},
      {CandidateGeneratorInputSlotRef(3), std::move(prior)}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          fpaDescriptor().reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFpaGbdtTrainingCandidateGeneratorBinding(
    const ResolvedDeterministicGbdtTrainingConfigView &config) {
  if (llvm::Error error = registerFpaGbdtTrainingCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(fpaDescriptor().reference(),
                                                config.canonicalViewBytes(),
                                                config.digest());
}

const CandidateGeneratorDescriptor &
systemRuntimeGbdtTrainingCandidateGeneratorDescriptor() {
  return systemRuntimeDescriptor();
}

llvm::Error registerSystemRuntimeGbdtTrainingCandidateGenerator() {
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return error;
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(systemRuntimeDescriptor()))
    return error;
  return registerCandidateGeneratorProvider(systemRuntimeProvider());
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSystemRuntimeGbdtTrainingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> training,
    llvm::ArrayRef<ArtifactRootReference> validation,
    llvm::ArrayRef<ArtifactRootReference> heldOut,
    const std::optional<ArtifactRootReference> &priorBundle) {
  if (llvm::Error error = registerSystemRuntimeGbdtTrainingCandidateGenerator())
    return std::move(error);
  std::vector<ArtifactRootReference> trainingSet = training.vec();
  std::vector<ArtifactRootReference> validationSet = validation.vec();
  std::vector<ArtifactRootReference> heldOutSet = heldOut.vec();
  canonicalizeReferences(trainingSet);
  canonicalizeReferences(validationSet);
  canonicalizeReferences(heldOutSet);
  std::vector<ArtifactRootReference> prior;
  if (priorBundle)
    prior.push_back(*priorBundle);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(0), std::move(trainingSet)},
      {CandidateGeneratorInputSlotRef(1), std::move(validationSet)},
      {CandidateGeneratorInputSlotRef(2), std::move(heldOutSet)},
      {CandidateGeneratorInputSlotRef(3), std::move(prior)}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          systemRuntimeDescriptor().reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSystemRuntimeGbdtTrainingCandidateGeneratorBinding(
    const ResolvedDeterministicGbdtTrainingConfigView &config) {
  if (llvm::Error error = registerSystemRuntimeGbdtTrainingCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      systemRuntimeDescriptor().reference(), config.canonicalViewBytes(),
      config.digest());
}

} // namespace loom::dse
