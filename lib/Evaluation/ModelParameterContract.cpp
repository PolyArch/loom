#include "Evaluation/ModelParameter.h"

#include "CanonicalSupport.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

std::mutex &contractMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<const ModelParameterContractDescriptor *> &contracts() {
  static std::vector<const ModelParameterContractDescriptor *> values;
  return values;
}

bool modelRefLess(EvaluationModelDescriptorRef lhs,
                  EvaluationModelDescriptorRef rhs) {
  if (lhs.schemaVersion().major != rhs.schemaVersion().major)
    return lhs.schemaVersion().major < rhs.schemaVersion().major;
  if (lhs.schemaVersion().minor != rhs.schemaVersion().minor)
    return lhs.schemaVersion().minor < rhs.schemaVersion().minor;
  return lhs.modelKind() < rhs.modelKind();
}

bool containsCase(llvm::ArrayRef<EvaluationCaseSignatureRef> cases,
                  EvaluationCaseSignatureRef value) {
  return llvm::binary_search(cases, value, evaluationCaseSignatureRefLess);
}

bool containsModel(llvm::ArrayRef<EvaluationModelDescriptorRef> models,
                   EvaluationModelDescriptorRef value) {
  return llvm::binary_search(models, value, modelRefLess);
}

llvm::Error validateCaseSet(llvm::ArrayRef<EvaluationCaseSignatureRef> cases) {
  if (cases.empty())
    return evaluationError(
        "model parameter contract requires prediction cases");
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!cases[index].descriptor())
      return evaluationError(
          "model parameter contract references an unregistered prediction "
          "case");
    if (index != 0 &&
        !evaluationCaseSignatureRefLess(cases[index - 1], cases[index]))
      return evaluationError(
          "model parameter prediction cases must be canonical and unique");
  }
  return llvm::Error::success();
}

llvm::Error
validateModelSet(llvm::ArrayRef<EvaluationModelDescriptorRef> models) {
  if (models.empty())
    return evaluationError(
        "model parameter contract requires ground-truth models");
  for (std::size_t index = 0; index < models.size(); ++index) {
    if (!models[index].descriptor())
      return evaluationError(
          "model parameter contract references an unregistered ground-truth "
          "model");
    if (index != 0 && !modelRefLess(models[index - 1], models[index]))
      return evaluationError(
          "model parameter ground-truth models must be canonical and unique");
  }
  return llvm::Error::success();
}

llvm::Error
validateConditionTable(const ModelParameterContractDescriptor &descriptor) {
  std::vector<EvaluationCaseSignatureRef> required(
      descriptor.predictionCaseSignatures.begin(),
      descriptor.predictionCaseSignatures.end());
  for (EvaluationModelDescriptorRef model :
       descriptor.groundTruthModelDescriptors)
    required.push_back(model.descriptor()->caseSignature);
  llvm::sort(required, evaluationCaseSignatureRefLess);
  required.erase(std::unique(required.begin(), required.end()), required.end());

  if (descriptor.consumedBaseConditionPatterns.size() != required.size())
    return evaluationError(
        "model parameter condition table is not total over its cases");
  for (std::size_t index = 0; index < required.size(); ++index) {
    const ModelParameterConditionPatternSet &entry =
        descriptor.consumedBaseConditionPatterns[index];
    if (entry.caseSignature != required[index])
      return evaluationError(
          "model parameter condition table is not in canonical case order");
    if (llvm::Error error = validateConditionApplicabilityPatternSet(
            "model parameter contract", entry.consumedBaseConditions,
            ConditionLocation::Base))
      return error;
    const EvaluationCaseSignatureDescriptor *signature =
        entry.caseSignature.descriptor();
    for (const ConditionApplicabilityPattern &pattern :
         entry.consumedBaseConditions) {
      if (pattern.targets.caseSignature != entry.caseSignature)
        return evaluationError(
            "model parameter condition pattern names a foreign case");
      if (!llvm::is_contained(signature->permittedBaseConditions, pattern))
        return evaluationError(
            "model parameter contract consumes a condition not permitted by "
            "its case");
    }
  }
  return llvm::Error::success();
}

llvm::Error
validateDescriptor(const ModelParameterContractDescriptor &descriptor) {
  if (descriptor.semanticDefinition.empty())
    return evaluationError(
        "model parameter contract requires a semantic definition");
  if (llvm::Error error = validateCaseSet(descriptor.predictionCaseSignatures))
    return error;
  if (llvm::Error error =
          validateModelSet(descriptor.groundTruthModelDescriptors))
    return error;
  if (llvm::Error error = validateConditionTable(descriptor))
    return error;
  if (descriptor.predictionSchemaDescriptorBytes.empty())
    return evaluationError(
        "model parameter contract requires a prediction schema descriptor");
  if (descriptor.predictionDecimalFinalization.significantDigits == 0 ||
      descriptor.predictionDecimalFinalization.significantDigits > 18 ||
      static_cast<std::uint8_t>(
          descriptor.predictionDecimalFinalization.rounding) >
          static_cast<std::uint8_t>(
              ModelParameterDecimalRounding::RoundToNearestTiesToEven))
    return evaluationError(
        "model parameter contract has invalid decimal finalization");
  if (descriptor.maximumPayloadBytes && *descriptor.maximumPayloadBytes == 0)
    return evaluationError(
        "model parameter contract has an invalid payload bound");
  if (!descriptor.adopt || !descriptor.encode ||
      !descriptor.parameterGroundTruthTargetKey ||
      !descriptor.projectFeatures || !descriptor.infer ||
      !descriptor.groundTruthTargetKey || !descriptor.calibrationSampleGroupKey)
    return evaluationError(
        "model parameter contract has an incomplete typed operation table");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ModelParameterContractRef>
ModelParameterContractRef::get(llvm::StringRef ownerRegistryIdentity,
                               SchemaVersion ownerRegistryVersion,
                               std::uint32_t ownerLocalContractKind) {
  if (ownerRegistryIdentity.empty())
    return evaluationError("model parameter owner is empty");
  for (unsigned char character : ownerRegistryIdentity)
    if (character < 0x21 || character > 0x7e)
      return evaluationError("model parameter owner is not canonical ASCII");
  return ModelParameterContractRef(ownerRegistryIdentity.str(),
                                   ownerRegistryVersion,
                                   ownerLocalContractKind);
}

bool operator<(const ModelParameterContractRef &lhs,
               const ModelParameterContractRef &rhs) {
  if (lhs.ownerRegistryIdentity_ != rhs.ownerRegistryIdentity_)
    return lhs.ownerRegistryIdentity_ < rhs.ownerRegistryIdentity_;
  if (lhs.ownerRegistryVersion_.major != rhs.ownerRegistryVersion_.major)
    return lhs.ownerRegistryVersion_.major < rhs.ownerRegistryVersion_.major;
  if (lhs.ownerRegistryVersion_.minor != rhs.ownerRegistryVersion_.minor)
    return lhs.ownerRegistryVersion_.minor < rhs.ownerRegistryVersion_.minor;
  return lhs.ownerLocalContractKind_ < rhs.ownerLocalContractKind_;
}

std::vector<std::uint8_t> canonicalModelParameterContractReferenceBytes(
    const ModelParameterContractRef &reference) {
  std::vector<std::uint8_t> bytes;
  detail::appendFramedString(bytes, reference.ownerRegistryIdentity());
  detail::appendSchemaVersion(bytes, reference.ownerRegistryVersion());
  detail::appendU32Be(bytes, reference.ownerLocalContractKind());
  return bytes;
}

llvm::Error registerModelParameterContract(
    const ModelParameterContractDescriptor &descriptor) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;
  std::lock_guard<std::mutex> lock(contractMutex());
  for (const ModelParameterContractDescriptor *existing : contracts()) {
    if (existing->reference != descriptor.reference)
      continue;
    if (existing == &descriptor)
      return llvm::Error::success();
    return evaluationError(
        "conflicting registration for model parameter contract '" +
        descriptor.reference.ownerRegistryIdentity() + "'");
  }
  contracts().push_back(&descriptor);
  llvm::sort(contracts(), [](const ModelParameterContractDescriptor *lhs,
                             const ModelParameterContractDescriptor *rhs) {
    return lhs->reference < rhs->reference;
  });
  return llvm::Error::success();
}

const ModelParameterContractDescriptor *
findModelParameterContract(const ModelParameterContractRef &reference) {
  std::lock_guard<std::mutex> lock(contractMutex());
  auto found =
      llvm::lower_bound(contracts(), reference,
                        [](const ModelParameterContractDescriptor *descriptor,
                           const ModelParameterContractRef &requested) {
                          return descriptor->reference < requested;
                        });
  if (found == contracts().end() || (*found)->reference != reference)
    return nullptr;
  return *found;
}

llvm::Expected<OwnerValue>
projectModelFeatures(const ModelParameterContractRef &reference,
                     const EvaluationCase &evaluationCase,
                     const CaseArtifactResolution &resolution,
                     const ArtifactStore &artifactStore,
                     const BlobStore &blobStore) {
  const ModelParameterContractDescriptor *descriptor =
      findModelParameterContract(reference);
  if (!descriptor)
    return evaluationError("model parameter contract is unregistered");
  if (!containsCase(descriptor->predictionCaseSignatures,
                    evaluationCase.signature()))
    return evaluationError(
        "model parameter contract does not accept the prediction case");
  auto features = descriptor->projectFeatures(evaluationCase, resolution,
                                              artifactStore, blobStore);
  if (!features)
    return features.takeError();
  if (!*features)
    return evaluationError(
        "model parameter feature projector returned an empty owner value");
  return features;
}

llvm::Expected<ModelParameterInferenceOutcome>
inferModelParameters(const FinalizedModelParameterBundle &bundle,
                     const OwnerValue &featureView) {
  const ModelParameterContractDescriptor *descriptor =
      findModelParameterContract(bundle.bundle().parameterContract());
  if (!descriptor)
    return evaluationError("model parameter contract is unregistered");
  if (!featureView)
    return evaluationError("model parameter feature view is empty");
  auto outcome = descriptor->infer(bundle.ownerParameters(), featureView);
  if (!outcome)
    return outcome.takeError();
  if (const auto *prediction = std::get_if<ModelParameterPrediction>(&*outcome))
    if (!prediction->view)
      return evaluationError(
          "model parameter inference returned an empty prediction view");
  return outcome;
}

llvm::Expected<std::vector<std::uint8_t>> modelParameterGroundTruthTargetKey(
    const ModelParameterContractRef &reference,
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const ModelParameterContractDescriptor *descriptor =
      findModelParameterContract(reference);
  if (!descriptor)
    return evaluationError("model parameter contract is unregistered");
  if (!containsModel(descriptor->groundTruthModelDescriptors,
                     request.modelBinding().descriptorRef()))
    return evaluationError(
        "request does not select an admitted ground-truth model");
  auto key = descriptor->groundTruthTargetKey(request, resolution,
                                              artifactStore, blobStore);
  if (!key)
    return key.takeError();
  if (key->empty())
    return evaluationError("ground-truth target key is empty");
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
modelParameterCalibrationSampleGroupKey(
    const ModelParameterContractRef &reference,
    const EvaluationEvidence &evidence, const EvaluationRequest &request,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const ModelParameterContractDescriptor *descriptor =
      findModelParameterContract(reference);
  if (!descriptor)
    return evaluationError("model parameter contract is unregistered");
  if (!containsModel(descriptor->groundTruthModelDescriptors,
                     request.modelBinding().descriptorRef()))
    return evaluationError(
        "evidence request does not select an admitted ground-truth model");
  auto key = descriptor->calibrationSampleGroupKey(
      evidence, request, resolution, artifactStore, blobStore);
  if (!key)
    return key.takeError();
  if (key->empty())
    return evaluationError("calibration sample-group key is empty");
  return key;
}

} // namespace loom::evaluation
