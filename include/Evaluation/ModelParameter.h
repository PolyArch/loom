#ifndef LOOM_EVALUATION_MODELPARAMETER_H
#define LOOM_EVALUATION_MODELPARAMETER_H

#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/OwnerValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom::evaluation {

enum class ModelParameterDecimalRounding : std::uint8_t {
  RoundToNearestTiesToEven = 0,
};

struct ModelParameterDecimalFinalizationContract final {
  std::uint32_t significantDigits = 0;
  ModelParameterDecimalRounding rounding =
      ModelParameterDecimalRounding::RoundToNearestTiesToEven;
};

struct ModelParameterConditionPatternSet final {
  EvaluationCaseSignatureRef caseSignature;
  llvm::ArrayRef<ConditionApplicabilityPattern> consumedBaseConditions;
};

struct ModelParameterPrediction final {
  OwnerValue view;
};

struct UnsupportedModelParameterInference final {};

using ModelParameterInferenceOutcome =
    std::variant<ModelParameterPrediction, UnsupportedModelParameterInference>;

struct ModelParameterContractDescriptor final {
  ModelParameterContractRef reference;
  llvm::StringRef semanticDefinition;
  llvm::ArrayRef<EvaluationCaseSignatureRef> predictionCaseSignatures;
  llvm::ArrayRef<EvaluationModelDescriptorRef> groundTruthModelDescriptors;
  llvm::ArrayRef<ModelParameterConditionPatternSet>
      consumedBaseConditionPatterns;
  llvm::ArrayRef<std::uint8_t> predictionSchemaDescriptorBytes;
  ModelParameterDecimalFinalizationContract predictionDecimalFinalization;

  llvm::Expected<OwnerValue> (*adopt)(
      llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes);
  llvm::Expected<std::vector<std::uint8_t>> (*encode)(
      const OwnerValue &parameters);
  llvm::Expected<std::vector<std::uint8_t>> (*parameterGroundTruthTargetKey)(
      const OwnerValue &parameters);
  llvm::Expected<OwnerValue> (*projectFeatures)(
      const EvaluationCase &evaluationCase,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
  llvm::Expected<ModelParameterInferenceOutcome> (*infer)(
      const OwnerValue &parameters, const OwnerValue &featureView);
  llvm::Expected<std::vector<std::uint8_t>> (*groundTruthTargetKey)(
      const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
  llvm::Expected<std::vector<std::uint8_t>> (*calibrationSampleGroupKey)(
      const EvaluationEvidence &evidence, const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
};

llvm::Error registerModelParameterContract(
    const ModelParameterContractDescriptor &descriptor);

const ModelParameterContractDescriptor *
findModelParameterContract(const ModelParameterContractRef &reference);

llvm::Expected<OwnerValue>
projectModelFeatures(const ModelParameterContractRef &reference,
                     const EvaluationCase &evaluationCase,
                     const CaseArtifactResolution &resolution,
                     const ArtifactStore &artifactStore,
                     const BlobStore &blobStore);

llvm::Expected<ModelParameterInferenceOutcome>
inferModelParameters(const FinalizedModelParameterBundle &bundle,
                     const OwnerValue &featureView);

llvm::Expected<std::vector<std::uint8_t>> modelParameterGroundTruthTargetKey(
    const ModelParameterContractRef &reference,
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

llvm::Expected<std::vector<std::uint8_t>>
modelParameterCalibrationSampleGroupKey(
    const ModelParameterContractRef &reference,
    const EvaluationEvidence &evidence, const EvaluationRequest &request,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELPARAMETER_H
