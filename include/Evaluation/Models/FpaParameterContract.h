#ifndef LOOM_EVALUATION_MODELS_FPAPARAMETERCONTRACT_H
#define LOOM_EVALUATION_MODELS_FPAPARAMETERCONTRACT_H

#include "Evaluation/ModelParameter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {

inline constexpr std::uint64_t maximumFpaModelParameterPayloadBytes =
    10'000'000'000ULL;

struct FpaFabricStructureFeatureView final {
  std::uint64_t entityCount = 0;
  std::uint64_t peOccurrenceCount = 0;
  std::uint64_t fuOccurrenceCount = 0;
  std::uint64_t operationCapabilityCount = 0;
  std::uint64_t memoryOccurrenceCount = 0;
  std::uint64_t memoryOperationPortCount = 0;
  std::uint64_t switchOccurrenceCount = 0;
  std::uint64_t fifoOccurrenceCount = 0;
  std::uint64_t boundaryOccurrenceCount = 0;
  std::uint64_t hostCoreOccurrenceCount = 0;
  std::uint64_t accCoreOccurrenceCount = 0;
  std::uint64_t systemMemoryServiceCount = 0;
  std::uint64_t systemTransportResourceCount = 0;
  std::uint64_t hardwareDomainCount = 0;
  std::uint64_t transportEndpointCount = 0;
  std::uint64_t pointConnectionCount = 0;
  std::uint64_t admittedTraversalCount = 0;
  std::uint64_t importedModuleCount = 0;
};

struct FpaOperatingConditionFeatureView final {
  std::uint64_t processCornerCount = 0;
  std::uint64_t supplyVoltageCount = 0;
  std::uint64_t temperatureCount = 0;
  std::uint64_t requiredClockCount = 0;
  std::uint64_t relativeClockCount = 0;
  std::uint64_t activityBindingCount = 0;

  std::optional<DecimalValue> minimumSupplyVoltage;
  std::optional<DecimalValue> maximumSupplyVoltage;
  std::optional<DecimalValue> minimumTemperature;
  std::optional<DecimalValue> maximumTemperature;
  std::optional<DecimalValue> minimumRequiredClockPeriod;
  std::optional<DecimalValue> maximumRequiredClockPeriod;
  std::optional<DecimalValue> minimumStaticProbability;
  std::optional<DecimalValue> maximumStaticProbability;
  std::optional<DecimalValue> minimumTransitionsPerClock;
  std::optional<DecimalValue> maximumTransitionsPerClock;
  std::optional<DecimalValue> minimumRelativeClockPeriod;
  std::optional<DecimalValue> maximumRelativeClockPeriod;
  std::optional<DecimalValue> minimumRelativeClockPhase;
  std::optional<DecimalValue> maximumRelativeClockPhase;

  std::vector<std::uint8_t> processCornerCohortKey;
  std::vector<std::uint8_t> conditionTargetShapeKey;
};

struct FpaFeatureView final {
  FpaFabricStructureFeatureView fabric;
  FpaOperatingConditionFeatureView conditions;
  std::vector<std::uint8_t> implementationFamilyKey;
};

struct FpaMetricPredictionView final {
  DecimalValue limitingClockFrequency;
  DecimalValue totalArea;
  DecimalValue dynamicPower;
  DecimalValue leakagePower;
};

struct FpaGbdtTrainingConfig final {
  std::uint64_t seed = 0;
  std::uint32_t treeCount = 0;
  std::uint32_t maximumDepth = 0;
  std::uint32_t minimumRowsPerLeaf = 0;
  std::uint32_t learningRateNumerator = 0;
  std::uint32_t learningRateDenominator = 0;
};

struct FpaTrainingEvidenceSample final {
  FpaFeatureView features;
  FpaMetricPredictionView observation;
  std::vector<std::uint8_t> groundTruthTargetKey;
  std::vector<std::uint8_t> sampleGroupKey;
};

class FpaGbdtParameters final {
public:
  FpaGbdtParameters(const FpaGbdtParameters &) = default;
  FpaGbdtParameters(FpaGbdtParameters &&) noexcept = default;
  FpaGbdtParameters &operator=(const FpaGbdtParameters &) = default;
  FpaGbdtParameters &operator=(FpaGbdtParameters &&) noexcept = default;

  llvm::ArrayRef<std::uint8_t> groundTruthTargetKey() const;

private:
  struct Storage;
  explicit FpaGbdtParameters(std::shared_ptr<const Storage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const Storage> storage_;

  friend llvm::Expected<FpaGbdtParameters>
  trainFpaGbdtParameters(llvm::ArrayRef<FpaTrainingEvidenceSample>,
                         const FpaGbdtTrainingConfig &,
                         const FpaGbdtParameters *);
  friend llvm::Expected<FpaGbdtParameters>
      adoptFpaGbdtParameters(llvm::ArrayRef<std::uint8_t>);
  friend llvm::Expected<std::vector<std::uint8_t>>
  encodeFpaGbdtParameters(const FpaGbdtParameters &);
  friend llvm::Expected<ModelParameterInferenceOutcome>
  inferFpaGbdtParameters(const FpaGbdtParameters &, const FpaFeatureView &);
};

const ModelParameterContractRef &fpaModelParameterContractRef();
const ModelParameterContractDescriptor &fpaModelParameterContractDescriptor();
llvm::ArrayRef<std::uint8_t> fpaMetricPredictionViewSchemaDescriptorBytes();

llvm::Error registerFpaModelParameterContract();

llvm::Expected<CaseArtifactResolution>
resolveFpaCalibrationCaseArtifactResolution(
    const ArtifactRootReference &parameterBundle,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

llvm::Expected<FpaTrainingEvidenceSample>
importFpaTrainingEvidenceSample(const ArtifactRootReference &evidence,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore);

llvm::Expected<FpaTrainingEvidenceSample>
importFpaTrainingEvidenceSample(const ArtifactRootReference &evidence,
                                const CaseArtifactResolution &resolution,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore);

/// Derives the exact cross-partition leakage key for one implementation from
/// its Fabric root and implementation-family contract.
llvm::Expected<std::vector<std::uint8_t>>
deriveFpaSampleGroupKey(const ArtifactRootReference &hardwareImplementation,
                        const ArtifactStore &artifactStore,
                        const BlobStore &blobStore);

llvm::Expected<FpaGbdtParameters>
trainFpaGbdtParameters(llvm::ArrayRef<FpaTrainingEvidenceSample> training,
                       const FpaGbdtTrainingConfig &config,
                       const FpaGbdtParameters *prior = nullptr);

llvm::Expected<FpaGbdtParameters>
adoptFpaGbdtParameters(llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes);

llvm::Error validateFpaModelParameterPayloadSize(std::uint64_t byteCount);

llvm::Expected<std::vector<std::uint8_t>>
encodeFpaGbdtParameters(const FpaGbdtParameters &parameters);

llvm::Expected<ModelParameterInferenceOutcome>
inferFpaGbdtParameters(const FpaGbdtParameters &parameters,
                       const FpaFeatureView &features);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_FPAPARAMETERCONTRACT_H
