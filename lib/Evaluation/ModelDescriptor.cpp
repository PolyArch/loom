#include "Evaluation/ModelDescriptor.h"

#include "CanonicalSupport.h"

#include <algorithm>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

llvm::Error validateDescriptor(const EvaluationModelDescriptor &descriptor) {
  const EvaluationCaseSignatureDescriptor *signature =
      descriptor.caseSignature.descriptor();
  if (!signature)
    return evaluationError("model '" +
                           descriptor.implementationSemanticIdentity +
                           "' references an unregistered case signature");

  for (std::size_t index = 0; index < descriptor.supportedMetrics.size();
       ++index) {
    if (index != 0 && !(descriptor.supportedMetrics[index - 1] <
                        descriptor.supportedMetrics[index]))
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' must declare supported metrics in canonical order without "
          "duplicates");
  }

  for (std::size_t index = 0; index < descriptor.conditionCapabilities.size();
       ++index) {
    const ModelConditionCapability &capability =
        descriptor.conditionCapabilities[index];
    if (index != 0 && !conditionApplicabilityPatternLess(
                          descriptor.conditionCapabilities[index - 1].pattern,
                          capability.pattern))
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' must declare condition capabilities in canonical order without "
          "duplicates");
    if (capability.pattern.targets.caseSignature != descriptor.caseSignature)
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' declares a condition capability for a foreign case signature");

    bool permitted = std::find(signature->permittedBaseConditions.begin(),
                               signature->permittedBaseConditions.end(),
                               capability.pattern) !=
                     signature->permittedBaseConditions.end();
    for (MetricKind metric : descriptor.supportedMetrics) {
      const llvm::ArrayRef<ConditionApplicabilityPattern> metricPatterns =
          metricDescriptor(metric).permittedRequestConditionPatterns;
      permitted =
          permitted || std::find(metricPatterns.begin(), metricPatterns.end(),
                                 capability.pattern) != metricPatterns.end();
    }
    if (!permitted)
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' widens condition applicability beyond its case and metric "
          "owners");
  }
  return llvm::Error::success();
}

llvm::Error requireRecognizedConditions(
    const EvaluationModelDescriptor &descriptor,
    llvm::ArrayRef<EvaluationCondition> conditions,
    EvaluationCaseSignatureRef signature,
    llvm::SmallVectorImpl<ConditionApplicabilityPattern> &present) {
  for (const EvaluationCondition &condition : conditions) {
    const ConditionApplicabilityPattern derived =
        deriveConditionApplicabilityPattern(condition, signature);
    if (!descriptor.findConditionCapability(derived))
      return evaluationError("model '" +
                             descriptor.implementationSemanticIdentity +
                             "' does not recognize condition '" +
                             toString(condition.kind()) + "'");
    present.push_back(derived);
  }
  return llvm::Error::success();
}

} // namespace

const ModelConditionCapability *
EvaluationModelDescriptor::findConditionCapability(
    const ConditionApplicabilityPattern &pattern) const {
  for (const ModelConditionCapability &capability : conditionCapabilities)
    if (capability.pattern == pattern)
      return &capability;
  return nullptr;
}

bool EvaluationModelDescriptor::supportsMetric(MetricKind metric) const {
  return std::find(supportedMetrics.begin(), supportedMetrics.end(), metric) !=
         supportedMetrics.end();
}

llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;
  const EvaluationCaseSignatureRef signature = evaluationCase.signature();
  if (signature != descriptor.caseSignature)
    return evaluationError(
        "model '" + descriptor.implementationSemanticIdentity +
        "' does not evaluate the case's exact case signature");

  for (const MetricRequest &request : metricRequests)
    if (!descriptor.supportsMetric(request.query().metric))
      return evaluationError("model '" +
                             descriptor.implementationSemanticIdentity +
                             "' does not support metric '" +
                             toString(request.query().metric) + "'");

  llvm::SmallVector<ConditionApplicabilityPattern, 4> present;
  if (llvm::Error error = requireRecognizedConditions(
          descriptor, evaluationCase.baseConditions(), signature, present))
    return error;
  for (const MetricRequest &request : metricRequests)
    if (llvm::Error error = requireRecognizedConditions(
            descriptor, request.conditions(), signature, present))
      return error;

  for (const ModelConditionCapability &capability :
       descriptor.conditionCapabilities) {
    if (capability.disposition != ConditionDisposition::Required)
      continue;
    if (std::find(present.begin(), present.end(), capability.pattern) ==
        present.end())
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' requires condition '" + toString(capability.pattern.kind) + "'");
  }
  return llvm::Error::success();
}

} // namespace loom::evaluation
