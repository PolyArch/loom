#ifndef LOOM_EVALUATION_MODELDESCRIPTOR_H
#define LOOM_EVALUATION_MODELDESCRIPTOR_H

#include "Evaluation/Case.h"
#include "Evaluation/Metric.h"
#include "Evaluation/Request.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::evaluation {

enum class EvaluationModelKind : std::uint8_t {
  AnalyticalTimingModel,
  CycleAccurateSimulator,
};

/// What a model does with a condition its permitting owner already allows. The
/// three dispositions are one closed choice per kind, so a model can neither
/// redefine a payload nor silently ignore an unrecognized condition.
enum class ConditionDisposition : std::uint8_t {
  Consumed,
  Required,
  Invariant
};

struct ModelConditionCapability {
  EvaluationConditionKind kind;
  ConditionDisposition disposition;
};

/// An immutable, versioned capability entry in the Evaluation library's static
/// typed registry. It is not an Artifact. It holds exactly one
/// EvaluationCaseSignatureRef and owns no subject role, schema, cardinality, or
/// subject slot: that authority belongs to the case signature alone.
struct EvaluationModelDescriptor {
  EvaluationModelKind modelKind;
  llvm::StringRef spelling;
  SchemaVersion descriptorVersion;
  llvm::StringRef implementationSemanticIdentity;
  EvaluationCaseSignatureRef caseSignature;
  llvm::ArrayRef<ModelConditionCapability> conditionCapabilities;
  llvm::ArrayRef<MetricKind> supportedMetrics;

  const ModelConditionCapability *
  findConditionCapability(EvaluationConditionKind kind) const;
  bool supportsMetric(MetricKind metric) const;
};

const EvaluationModelDescriptor &modelDescriptor(EvaluationModelKind modelKind);

/// Checks that the model evaluates the case's exact signature, supports every
/// requested metric, recognizes every base and request-specific condition it is
/// given, and receives every condition it requires. The model never widens what
/// the case signature or a Metric descriptor already permitted.
llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELDESCRIPTOR_H
