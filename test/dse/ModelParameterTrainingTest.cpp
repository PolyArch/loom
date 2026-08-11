#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "model parameter training test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireError(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error containing '" + expected + "'");
  std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail("unexpected error: " + message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-model-parameter-training", path_);
    if (error)
      fail("cannot create test directory: " + error.message());
    blobPath_ = path_;
    llvm::sys::path::append(blobPath_, "blobs");
    error = llvm::sys::fs::create_directory(blobPath_);
    if (error)
      fail("cannot create test BlobStore: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }
  llvm::StringRef blobPath() const { return blobPath_; }

private:
  llvm::SmallString<128> path_;
  llvm::SmallString<128> blobPath_;
};

loom::evaluation::DecimalValue decimal(std::int64_t coefficient,
                                       std::int64_t exponent = 0) {
  return take(loom::evaluation::DecimalValue::get(coefficient, exponent));
}

loom::evaluation::models::FpaFeatureView feature(std::uint64_t entities) {
  loom::evaluation::models::FpaFeatureView result;
  result.fabric.entityCount = entities;
  result.fabric.peOccurrenceCount = entities;
  result.conditions.processCornerCohortKey = {0x11};
  result.conditions.conditionTargetShapeKey = {0x22};
  result.implementationFamilyKey = {0x33};
  return result;
}

loom::evaluation::models::FpaTrainingEvidenceSample
sample(std::uint64_t entities, std::int64_t offset, std::uint8_t sampleGroup) {
  return {feature(entities),
          {decimal(100 + offset), decimal(200 + offset), decimal(300 + offset),
           decimal(400 + offset)},
          {0x41, 0x42},
          {sampleGroup}};
}

loom::evaluation::models::FpaGbdtTrainingConfig trainingConfig() {
  return {7, 3, 2, 1, 1, 2};
}

loom::evaluation::models::SystemRuntimeFeatureView
systemRuntimeFeature(std::uint64_t entities) {
  loom::evaluation::models::SystemRuntimeFeatureView result;
  result.deployment.hardwareBindingCount = 1;
  result.deployment.hasSpatialLaunchImage = true;
  result.workload.entryValueInputCount = entities;
  result.runtimeInput.runtimeEntryValueCount = entities;
  result.mapping.spatialContextDomainCount = entities;
  result.fabric.entityCount = entities;
  result.fabric.acceleratorCoreOccurrenceCount = 1;
  result.softwarePartitioningKey = {0x51};
  result.modeledPlatformKey = {0x52};
  result.gem5BindingFeatureKey = {0x53};
  result.admittedRuntimeConditionKey = {0x54};
  return result;
}

loom::evaluation::models::SystemRuntimeTrainingSample
systemRuntimeSample(std::uint64_t entities, std::int64_t runtime,
                    std::uint8_t sampleGroup) {
  return {systemRuntimeFeature(entities),
          decimal(runtime, -6),
          {0x61, 0x62},
          {sampleGroup}};
}

loom::evaluation::models::SystemRuntimeGbdtTrainingConfig
systemRuntimeTrainingConfig() {
  return {11, 3, 2, 1, 1, 2};
}

void exactContractAndTrainingKernel() {
  using namespace loom::evaluation;
  using namespace loom::evaluation::models;

  requireSuccess(registerProductionEvaluationRegistry());
  const ModelParameterContractDescriptor *contract =
      findModelParameterContract(fpaModelParameterContractRef());
  require(contract == &fpaModelParameterContractDescriptor(),
          "production registry did not install the FPA parameter contract");
  require(contract->predictionCaseSignatures.size() == 3 &&
              contract->groundTruthModelDescriptors.size() == 1 &&
              contract->predictionDecimalFinalization.significantDigits == 18,
          "FPA contract has the wrong closed capability shape");

  std::vector<FpaTrainingEvidenceSample> rows = {
      sample(1, 0, 1), sample(2, 10, 2), sample(3, 20, 3)};
  auto first = take(trainFpaGbdtParameters(rows, trainingConfig()));
  auto repeated = take(trainFpaGbdtParameters(rows, trainingConfig()));
  const std::vector<std::uint8_t> firstBytes =
      take(encodeFpaGbdtParameters(first));
  const std::vector<std::uint8_t> repeatedBytes =
      take(encodeFpaGbdtParameters(repeated));
  require(firstBytes == repeatedBytes,
          "identical deterministic training did not converge on one payload");

  auto prediction = take(inferFpaGbdtParameters(first, rows[1].features));
  const auto *predicted = std::get_if<ModelParameterPrediction>(&prediction);
  require(predicted && predicted->view.getIf<FpaMetricPredictionView>(),
          "in-envelope inference did not return the typed FPA prediction");
  FpaFeatureView outside = feature(4);
  auto unsupported = take(inferFpaGbdtParameters(first, outside));
  require(
      std::holds_alternative<UnsupportedModelParameterInference>(unsupported),
      "out-of-envelope inference returned a numeric extrapolation");

  auto warm = take(trainFpaGbdtParameters(rows, trainingConfig(), &first));
  const std::vector<std::uint8_t> warmBytes =
      take(encodeFpaGbdtParameters(warm));
  require(warmBytes != firstBytes,
          "warm start did not append to the prior ensemble");
  std::vector<FpaTrainingEvidenceSample> changed = rows;
  changed.back().features.fabric.entityCount = 4;
  requireError(
      trainFpaGbdtParameters(changed, trainingConfig(), &first).takeError(),
      "exact Training support");

  requireError(adoptFpaGbdtParameters(
                   llvm::ArrayRef<std::uint8_t>(firstBytes).drop_back())
                   .takeError(),
               "truncated");
}

void strictBundleAndGeneratorContracts() {
  using namespace loom;
  using namespace loom::dse;
  using namespace loom::evaluation;
  using namespace loom::evaluation::models;

  TemporaryDirectory directory;
  ArtifactStore store(directory.path());
  BlobStore blobs(directory.blobPath());
  std::vector<FpaTrainingEvidenceSample> rows = {sample(1, 0, 1),
                                                 sample(2, 10, 2)};
  auto parameters = take(trainFpaGbdtParameters(rows, trainingConfig()));
  OwnerValue owner = OwnerValue::get(parameters);
  auto finalized = take(finalizeModelParameterBundle(
      fpaModelParameterContractRef(), owner, store, blobs));
  auto imported =
      take(importModelParameterBundle(finalized.reference(), store, blobs));
  require(imported.bundle().parameterContract() ==
                  fpaModelParameterContractRef() &&
              imported.parametersIf<FpaGbdtParameters>(),
          "strict bundle import lost its exact contract or owner type");
  require(take(encodeFpaGbdtParameters(
              *imported.parametersIf<FpaGbdtParameters>())) ==
              take(encodeFpaGbdtParameters(parameters)),
          "bundle import changed canonical parameter bytes");

  requireError(
      resolveDeterministicGbdtTrainingConfig({0, 0, 1, 1, 1, 1}).takeError(),
      "tree count");
  auto config =
      take(resolveDeterministicGbdtTrainingConfig({7, 2, 3, 1, 1, 4}));
  std::vector<std::uint8_t> corrupted = config.canonicalViewBytes().vec();
  corrupted.back() ^= 1;
  requireError(adoptResolvedDeterministicGbdtTrainingConfigView(
                   resolvedDeterministicGbdtTrainingConfigSchemaBytes(),
                   corrupted, config.digest())
                   .takeError(),
               "digest");

  requireSuccess(registerFpaGbdtTrainingCandidateGenerator());
  const CandidateGeneratorDescriptor &descriptor =
      fpaGbdtTrainingCandidateGeneratorDescriptor();
  require(descriptor.kind == fpaGbdtTrainingCandidateGeneratorKind &&
              descriptor.inputSlots.size() == 4 &&
              descriptor.outputSlots.size() == 1,
          "FPA trainer descriptor has the wrong closed slot shape");
  require(descriptor.inputSlots[0].calibrationPartitionRole ==
                  CalibrationPartitionRole::Training &&
              descriptor.inputSlots[1].calibrationPartitionRole ==
                  CalibrationPartitionRole::Validation &&
              descriptor.inputSlots[2].calibrationPartitionRole ==
                  CalibrationPartitionRole::HeldOut &&
              *descriptor.inputSlots[3].modelParameterContract ==
                  fpaModelParameterContractRef() &&
              *descriptor.outputSlots[0].modelParameterContract ==
                  fpaModelParameterContractRef(),
          "FPA trainer descriptor lost a partition or contract refinement");
}

void systemRuntimeIndependentCore() {
  using namespace loom;
  using namespace loom::dse;
  using namespace loom::evaluation;
  using namespace loom::evaluation::models;

  requireSuccess(registerProductionEvaluationRegistry());
  const ModelParameterContractDescriptor *contract =
      findModelParameterContract(systemRuntimeModelParameterContractRef());
  require(contract == &systemRuntimeModelParameterContractDescriptor() &&
              contract->predictionCaseSignatures.size() == 1 &&
              contract->groundTruthModelDescriptors.size() == 1 &&
              contract->groundTruthModelDescriptors.front().modelKind() ==
                  builtinEvaluationModelKind(
                      BuiltinEvaluationModel::Gem5SystemCgra) &&
              contract->predictionDecimalFinalization.significantDigits == 18,
          "System Runtime contract has the wrong closed capability shape");

  llvm::ArrayRef<std::uint8_t> schema =
      systemRuntimePredictionViewSchemaDescriptorBytes();
  constexpr llvm::StringLiteral owner = "loom.system_runtime.prediction_view";
  require(
      schema.size() == 63 && schema[7] == owner.size() &&
          llvm::ArrayRef<std::uint8_t>(schema.data() + 8, owner.size()) ==
              llvm::ArrayRef<std::uint8_t>(owner.bytes_begin(), owner.size()),
      "System Runtime prediction schema bytes are not exact");

  std::vector<SystemRuntimeTrainingSample> rows = {
      systemRuntimeSample(1, 100, 1), systemRuntimeSample(2, 130, 2),
      systemRuntimeSample(3, 170, 3)};
  auto first = take(
      trainSystemRuntimeGbdtParameters(rows, systemRuntimeTrainingConfig()));
  auto repeated = take(
      trainSystemRuntimeGbdtParameters(rows, systemRuntimeTrainingConfig()));
  const std::vector<std::uint8_t> firstBytes =
      take(encodeSystemRuntimeGbdtParameters(first));
  require(firstBytes == take(encodeSystemRuntimeGbdtParameters(repeated)),
          "System Runtime training is not deterministic");

  auto prediction =
      take(inferSystemRuntimeGbdtParameters(first, rows[1].features));
  const auto *predicted = std::get_if<ModelParameterPrediction>(&prediction);
  require(predicted && predicted->view.getIf<SystemRuntimePredictionView>(),
          "System Runtime inference did not return its typed prediction");
  auto unsupported =
      take(inferSystemRuntimeGbdtParameters(first, systemRuntimeFeature(4)));
  require(
      std::holds_alternative<UnsupportedModelParameterInference>(unsupported),
      "System Runtime inference extrapolated outside Training support");
  requireError(adoptSystemRuntimeGbdtParameters(
                   llvm::ArrayRef<std::uint8_t>(firstBytes).drop_back())
                   .takeError(),
               "truncated");
  std::vector<SystemRuntimeTrainingSample> mixed = rows;
  mixed.back().groundTruthTargetKey = {0x63};
  requireError(
      trainSystemRuntimeGbdtParameters(mixed, systemRuntimeTrainingConfig())
          .takeError(),
      "mixes ground-truth target keys");

  TemporaryDirectory directory;
  ArtifactStore store(directory.path());
  BlobStore blobs(directory.blobPath());
  auto bundle = take(
      finalizeModelParameterBundle(systemRuntimeModelParameterContractRef(),
                                   OwnerValue::get(first), store, blobs));
  auto imported =
      take(importModelParameterBundle(bundle.reference(), store, blobs));
  require(imported.parametersIf<SystemRuntimeGbdtParameters>() != nullptr,
          "System Runtime bundle did not strict-import its owner payload");

  requireSuccess(registerSystemRuntimeGbdtTrainingCandidateGenerator());
  const CandidateGeneratorDescriptor &descriptor =
      systemRuntimeGbdtTrainingCandidateGeneratorDescriptor();
  require(descriptor.kind == systemRuntimeGbdtTrainingCandidateGeneratorKind &&
              descriptor.inputSlots.size() == 4 &&
              descriptor.outputSlots.size() == 1 &&
              *descriptor.inputSlots[3].modelParameterContract ==
                  systemRuntimeModelParameterContractRef() &&
              *descriptor.outputSlots[0].modelParameterContract ==
                  systemRuntimeModelParameterContractRef(),
          "System Runtime trainer descriptor lost its fixed contract shape");
}

} // namespace

int main() {
  exactContractAndTrainingKernel();
  strictBundleAndGeneratorContracts();
  systemRuntimeIndependentCore();
  return EXIT_SUCCESS;
}
