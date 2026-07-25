#ifndef LOOM_EVALUATION_MODELDESCRIPTOR_H
#define LOOM_EVALUATION_MODELDESCRIPTOR_H

#include "Evaluation/Case.h"
#include "Evaluation/Metric.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom::evaluation {

/// What a model does with one exact condition pattern its permitting owner
/// already allows. The three dispositions are one closed choice per pattern,
/// so a model can neither redefine a payload nor silently ignore an
/// unrecognized condition.
enum class ConditionDisposition : std::uint8_t {
  Consumed,
  Required,
  Invariant
};

struct ModelConditionCapability {
  ConditionApplicabilityPattern pattern;
  ConditionDisposition disposition;
};

/// An immutable, versioned capability entry owned by the model's
/// implementation. It is not an Artifact, and there is no central model
/// registry: each model owner constructs its own descriptor. The descriptor
/// holds exactly one EvaluationCaseSignatureRef and owns no subject role,
/// schema, cardinality, or subject slot: that authority belongs to the case
/// signature alone.
struct EvaluationModelDescriptor {
  llvm::StringRef implementationSemanticIdentity;
  EvaluationCaseSignatureRef caseSignature;
  llvm::ArrayRef<ModelConditionCapability> conditionCapabilities;
  llvm::ArrayRef<MetricKind> supportedMetrics;

  const ModelConditionCapability *
  findConditionCapability(const ConditionApplicabilityPattern &pattern) const;
  bool supportsMetric(MetricKind metric) const;
};

/// Checks that the model evaluates the case's exact signature, supports every
/// requested metric, recognizes every base and request-specific condition it
/// is given by exact pattern, and receives every condition pattern it
/// requires. The model never widens what the case signature or a Metric
/// descriptor already permitted.
llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELDESCRIPTOR_H
