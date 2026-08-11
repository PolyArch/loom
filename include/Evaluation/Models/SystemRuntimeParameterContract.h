#ifndef LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPARAMETERCONTRACT_H
#define LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPARAMETERCONTRACT_H

#include "Evaluation/ModelParameter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace loom::evaluation::models {

struct SystemRuntimeDeploymentFeatureView final {
  std::uint64_t instructionCoreBinaryCount = 0;
  std::uint64_t hardwareBindingCount = 0;
  std::uint64_t configurationImageCount = 0;
  std::uint64_t staticMemoryImageCount = 0;
  bool hasSpatialLaunchImage = false;
};

struct SystemRuntimeWorkloadFeatureView final {
  std::uint64_t entryValueInputCount = 0;
  std::uint64_t runtimeEntryValueInputCount = 0;
  std::uint64_t externalValueInputCount = 0;
  std::uint64_t runtimeExternalValueInputCount = 0;
  std::uint64_t valueResultCount = 0;
  std::uint64_t externalValueOutputCount = 0;
  std::uint64_t externalStreamOutputCount = 0;
  std::uint64_t memoryObservableCount = 0;
};

struct SystemRuntimeInputFeatureView final {
  std::uint64_t runtimeEntryValueCount = 0;
  std::uint64_t runtimeExternalValueCount = 0;
  std::uint64_t externalStreamInputCount = 0;
  std::uint64_t memoryObjectCount = 0;
  std::uint64_t memoryInterfaceBindingCount = 0;
  std::uint64_t memoryByteCount = 0;
  std::uint64_t streamTokenCount = 0;
};

struct SystemRuntimeMappingFeatureView final {
  std::uint64_t instructionContextDomainCount = 0;
  std::uint64_t spatialContextDomainCount = 0;
  std::uint64_t serviceRealizationCount = 0;
  std::uint64_t capacityCellCount = 0;
  std::uint64_t resourceActivationCount = 0;
  std::uint64_t capacityClaimCount = 0;
  std::uint64_t causalReleaseCount = 0;
};

struct SystemRuntimeFabricFeatureView final {
  std::uint64_t entityCount = 0;
  std::uint64_t hostCoreOccurrenceCount = 0;
  std::uint64_t acceleratorCoreOccurrenceCount = 0;
  std::uint64_t systemMemoryServiceCount = 0;
  std::uint64_t systemTransportResourceCount = 0;
  std::uint64_t hardwareDomainCount = 0;
  std::uint64_t transportEndpointCount = 0;
  std::uint64_t pointConnectionCount = 0;
  std::uint64_t admittedTraversalCount = 0;
};

/// The contract-owned, ephemeral tabular view. Each categorical key is a
/// canonical projection of the named owner domain, never caller-defined
/// feature metadata or persistent Artifact state.
struct SystemRuntimeFeatureView final {
  SystemRuntimeDeploymentFeatureView deployment;
  SystemRuntimeWorkloadFeatureView workload;
  SystemRuntimeInputFeatureView runtimeInput;
  SystemRuntimeMappingFeatureView mapping;
  SystemRuntimeFabricFeatureView fabric;
  std::vector<std::uint8_t> softwarePartitioningKey;
  std::vector<std::uint8_t> modeledPlatformKey;
  std::vector<std::uint8_t> gem5BindingFeatureKey;
  std::vector<std::uint8_t> admittedRuntimeConditionKey;
};

struct SystemRuntimePredictionView final {
  DecimalValue runtime;
};

struct SystemRuntimeGbdtTrainingConfig final {
  std::uint64_t seed = 0;
  std::uint32_t treeCount = 0;
  std::uint32_t maximumDepth = 0;
  std::uint32_t minimumRowsPerLeaf = 0;
  std::uint32_t learningRateNumerator = 0;
  std::uint32_t learningRateDenominator = 0;
};

struct SystemRuntimeTrainingSample final {
  SystemRuntimeFeatureView features;
  DecimalValue runtime;
  std::vector<std::uint8_t> groundTruthTargetKey;
  std::vector<std::uint8_t> sampleGroupKey;
};

class SystemRuntimeGbdtParameters final {
public:
  SystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &) = default;
  SystemRuntimeGbdtParameters(SystemRuntimeGbdtParameters &&) noexcept =
      default;
  SystemRuntimeGbdtParameters &
  operator=(const SystemRuntimeGbdtParameters &) = default;
  SystemRuntimeGbdtParameters &
  operator=(SystemRuntimeGbdtParameters &&) noexcept = default;

  llvm::ArrayRef<std::uint8_t> groundTruthTargetKey() const;

private:
  struct Storage;
  explicit SystemRuntimeGbdtParameters(std::shared_ptr<const Storage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const Storage> storage_;

  friend llvm::Expected<SystemRuntimeGbdtParameters>
  trainSystemRuntimeGbdtParameters(llvm::ArrayRef<SystemRuntimeTrainingSample>,
                                   const SystemRuntimeGbdtTrainingConfig &,
                                   const SystemRuntimeGbdtParameters *);
  friend llvm::Expected<SystemRuntimeGbdtParameters>
      adoptSystemRuntimeGbdtParameters(llvm::ArrayRef<std::uint8_t>);
  friend llvm::Expected<std::vector<std::uint8_t>>
  encodeSystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &);
  friend llvm::Expected<ModelParameterInferenceOutcome>
  inferSystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &,
                                   const SystemRuntimeFeatureView &);
};

const ModelParameterContractRef &systemRuntimeModelParameterContractRef();
const ModelParameterContractDescriptor &
systemRuntimeModelParameterContractDescriptor();
llvm::ArrayRef<std::uint8_t> systemRuntimePredictionViewSchemaDescriptorBytes();

llvm::Error registerSystemRuntimeModelParameterContract();

llvm::Expected<SystemRuntimeTrainingSample>
importSystemRuntimeTrainingEvidenceSample(
    const ArtifactRootReference &evidenceReference,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

llvm::Expected<SystemRuntimeGbdtParameters> trainSystemRuntimeGbdtParameters(
    llvm::ArrayRef<SystemRuntimeTrainingSample> training,
    const SystemRuntimeGbdtTrainingConfig &config,
    const SystemRuntimeGbdtParameters *prior = nullptr);

llvm::Expected<SystemRuntimeGbdtParameters> adoptSystemRuntimeGbdtParameters(
    llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes);

llvm::Expected<std::vector<std::uint8_t>> encodeSystemRuntimeGbdtParameters(
    const SystemRuntimeGbdtParameters &parameters);

llvm::Expected<ModelParameterInferenceOutcome>
inferSystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &parameters,
                                 const SystemRuntimeFeatureView &features);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEPARAMETERCONTRACT_H
