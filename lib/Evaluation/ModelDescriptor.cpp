#include "Evaluation/ModelDescriptor.h"

#include "CanonicalSupport.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

const ModelConditionCapability analyticalTimingCapabilities[] = {
    {EvaluationConditionKind::RequiredClockPeriod,
     ConditionDisposition::Consumed},
    {EvaluationConditionKind::Quantile, ConditionDisposition::Consumed},
};

const MetricKind analyticalTimingMetrics[] = {
    MetricKind::CycleCount, MetricKind::ClockPeriod, MetricKind::Runtime};

const ModelConditionCapability cycleAccurateSimulatorCapabilities[] = {
    {EvaluationConditionKind::RequiredClockPeriod,
     ConditionDisposition::Required},
};

const MetricKind cycleAccurateSimulatorMetrics[] = {MetricKind::CycleCount,
                                                    MetricKind::Runtime};

/// The registry holds the exact case-signature reference each descriptor
/// evaluates, so no consumer reconstructs it from a bare case kind.
llvm::ArrayRef<EvaluationModelDescriptor> modelDescriptors() {
  static const std::array<EvaluationModelDescriptor, 2> descriptors = {{
      {EvaluationModelKind::AnalyticalTimingModel, "analytical_timing_model",
       SchemaVersion{1, 0}, "loom.evaluation.analytical_timing_model",
       llvm::cantFail(EvaluationCaseSignatureRef::get(
           evaluationSchemaVersion(),
           EvaluationCaseKind::MappedWorkloadExecution)),
       analyticalTimingCapabilities, analyticalTimingMetrics},
      {EvaluationModelKind::CycleAccurateSimulator, "cycle_accurate_simulator",
       SchemaVersion{1, 0}, "loom.evaluation.cycle_accurate_simulator",
       llvm::cantFail(EvaluationCaseSignatureRef::get(
           evaluationSchemaVersion(),
           EvaluationCaseKind::MappedWorkloadExecution)),
       cycleAccurateSimulatorCapabilities, cycleAccurateSimulatorMetrics},
  }};
  return descriptors;
}

llvm::Error
requireRecognizedCondition(const EvaluationModelDescriptor &descriptor,
                           llvm::ArrayRef<EvaluationCondition> conditions) {
  for (const EvaluationCondition &condition : conditions)
    if (!descriptor.findConditionCapability(condition.kind()))
      return evaluationError("model '" + descriptor.spelling +
                             "' does not recognize condition '" +
                             toString(condition.kind()) + "'");
  return llvm::Error::success();
}

bool containsKind(llvm::ArrayRef<EvaluationCondition> conditions,
                  EvaluationConditionKind kind) {
  return std::any_of(conditions.begin(), conditions.end(),
                     [&](const EvaluationCondition &condition) {
                       return condition.kind() == kind;
                     });
}

} // namespace

const ModelConditionCapability *
EvaluationModelDescriptor::findConditionCapability(
    EvaluationConditionKind kind) const {
  for (const ModelConditionCapability &capability : conditionCapabilities)
    if (capability.kind == kind)
      return &capability;
  return nullptr;
}

bool EvaluationModelDescriptor::supportsMetric(MetricKind metric) const {
  return std::find(supportedMetrics.begin(), supportedMetrics.end(), metric) !=
         supportedMetrics.end();
}

const EvaluationModelDescriptor &
modelDescriptor(EvaluationModelKind modelKind) {
  for (const EvaluationModelDescriptor &descriptor : modelDescriptors())
    if (descriptor.modelKind == modelKind)
      return descriptor;
  llvm_unreachable("unknown EvaluationModelKind");
}

llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests) {
  if (evaluationCase.signature() != descriptor.caseSignature)
    return evaluationError(
        "model '" + descriptor.spelling + "' evaluates case signature '" +
        toString(descriptor.caseSignature.caseKind()) + "', not '" +
        toString(evaluationCase.signature().caseKind()) + "'");

  for (const MetricRequest &request : metricRequests)
    if (!descriptor.supportsMetric(request.query().metric))
      return evaluationError("model '" + descriptor.spelling +
                             "' does not support metric '" +
                             toString(request.query().metric) + "'");

  if (llvm::Error error = requireRecognizedCondition(
          descriptor, evaluationCase.baseConditions()))
    return error;
  for (const MetricRequest &request : metricRequests)
    if (llvm::Error error =
            requireRecognizedCondition(descriptor, request.conditions()))
      return error;

  for (const ModelConditionCapability &capability :
       descriptor.conditionCapabilities) {
    if (capability.disposition != ConditionDisposition::Required)
      continue;
    bool present =
        containsKind(evaluationCase.baseConditions(), capability.kind);
    for (const MetricRequest &request : metricRequests)
      present = present || containsKind(request.conditions(), capability.kind);
    if (!present)
      return evaluationError("model '" + descriptor.spelling +
                             "' requires condition '" +
                             toString(capability.kind) + "'");
  }
  return llvm::Error::success();
}

} // namespace loom::evaluation
