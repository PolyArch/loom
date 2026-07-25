#include "Evaluation/Request.h"

#include "CanonicalSupport.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::appendFramedBytes;
using detail::appendFramedString;
using detail::appendSchemaVersion;
using detail::appendU32Be;
using detail::appendU64Be;

constexpr llvm::StringLiteral baseCaseKeyDomain =
    "loom.evaluation.base_case_key.v1";
constexpr llvm::StringLiteral metricCaseKeyDomain =
    "loom.evaluation.metric_case_key.v1";

constexpr std::uint64_t absentReference = 0;
constexpr std::uint64_t presentReference = 1;

void appendOptionalReference(
    std::vector<std::uint8_t> &preimage,
    const std::optional<ArtifactRootReference> &reference) {
  if (!reference) {
    appendU64Be(preimage, absentReference);
    return;
  }
  appendU64Be(preimage, presentReference);
  appendFramedBytes(preimage, encodeArtifactRootReference(*reference));
}

void appendConditions(std::vector<std::uint8_t> &preimage,
                      llvm::ArrayRef<EvaluationCondition> conditions) {
  appendU64Be(preimage, conditions.size());
  for (const EvaluationCondition &condition : conditions)
    appendFramedBytes(preimage, conditionPayloadKey(condition));
}

} // namespace

llvm::Expected<MetricRequest>
MetricRequest::get(MetricQuery query,
                   llvm::ArrayRef<EvaluationCondition> conditions,
                   const EvaluationCase &evaluationCase,
                   const CaseArtifactResolution &resolution) {
  if (llvm::Error error = validateMetricQuery(query))
    return std::move(error);

  const CaseTargetContext context = evaluationCase.targetContext(resolution);
  if (llvm::Error error = validateEvaluationScopeCase(
          query.scope, metricDescriptor(query.metric).scopeForms, context))
    return std::move(error);

  // The metric descriptor owns which request-specific conditions apply. Every
  // permitted kind is targetless in schema 1.0, so each derives exactly one
  // complete pattern for this exact case signature.
  const MetricDescriptor &descriptor = metricDescriptor(query.metric);
  llvm::SmallVector<ConditionApplicabilityPattern, 1> permittedPatterns;
  for (EvaluationConditionKind kind : descriptor.permittedRequestConditions) {
    if (conditionDescriptor(kind).permitsLocation(ConditionLocation::Base))
      return detail::evaluationError(
          "metric descriptor '" + descriptor.spelling +
          "' permits a base-only condition kind in request conditions");
    permittedPatterns.push_back(ConditionApplicabilityPattern{
        kind, OrderedTargetPattern{context.signatureRef(), {}}});
  }

  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(conditions,
                                       ConditionLocation::MetricRequest,
                                       descriptor.spelling, permittedPatterns,
                                       context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();

  return MetricRequest(std::move(query), std::move(*canonicalConditions));
}

EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase) {
  std::vector<std::uint8_t> preimage;
  appendFramedString(preimage, baseCaseKeyDomain);
  appendSchemaVersion(preimage, evaluationCase.signature().schemaVersion());
  appendU32Be(preimage, evaluationCase.signature().caseKind().ordinal());

  const llvm::ArrayRef<CaseRoleBinding> bindings =
      evaluationCase.subjectBindings().roleBindings();
  appendU64Be(preimage, bindings.size());
  for (const CaseRoleBinding &binding : bindings) {
    appendU32Be(preimage, binding.role.ordinal());
    appendU64Be(preimage, binding.subjects.size());
    for (const ArtifactRootReference &subject : binding.subjects)
      appendFramedBytes(preimage, encodeArtifactRootReference(subject));
  }

  appendOptionalReference(preimage, evaluationCase.workload());
  appendOptionalReference(preimage, evaluationCase.runtimeInput());
  appendConditions(preimage, evaluationCase.baseConditions());

  return EvaluationCaseKey(llvm::SHA256::hash(preimage));
}

EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                const MetricRequest &request) {
  const EvaluationCaseKey base = baseCaseKey(evaluationCase);

  std::vector<std::uint8_t> preimage;
  appendFramedString(preimage, metricCaseKeyDomain);
  preimage.insert(preimage.end(), base.bytes().begin(), base.bytes().end());
  appendFramedString(preimage, toString(request.query().metric));
  appendFramedBytes(preimage, canonicalScopeKey(request.query().scope));
  appendConditions(preimage, request.conditions());

  return EvaluationCaseKey(llvm::SHA256::hash(preimage));
}

} // namespace loom::evaluation
