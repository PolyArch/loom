#include "Evaluation/ModelDescriptor.h"

#include "CanonicalSupport.h"

#include <algorithm>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

llvm::Error requireRecognizedConditions(
    const EvaluationModelDescriptor &descriptor,
    llvm::ArrayRef<EvaluationCondition> conditions,
    EvaluationCaseSignatureRef signature,
    llvm::SmallVectorImpl<ConditionApplicabilityPattern> &present) {
  for (const EvaluationCondition &condition : conditions) {
    const ConditionApplicabilityPattern derived =
        deriveConditionApplicabilityPattern(condition, signature);
    if (!descriptor.findConditionCapability(derived))
      return evaluationError(
          "model '" + descriptor.implementationSemanticIdentity +
          "' does not recognize condition '" + toString(condition.kind()) +
          "'");
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
      return evaluationError("model '" +
                             descriptor.implementationSemanticIdentity +
                             "' requires condition '" +
                             toString(capability.pattern.kind) + "'");
  }
  return llvm::Error::success();
}

} // namespace loom::evaluation
