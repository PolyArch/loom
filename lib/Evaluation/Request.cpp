#include "Evaluation/Request.h"

#include "CanonicalSupport.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::appendArtifactIdentity;
using detail::appendFramedBytes;
using detail::appendFramedString;
using detail::appendSchemaVersion;
using detail::appendU64Be;

constexpr llvm::StringLiteral baseCaseKeyDomain =
    "loom.evaluation.base_case_key.v1";
constexpr llvm::StringLiteral metricCaseKeyDomain =
    "loom.evaluation.metric_case_key.v1";

constexpr std::uint64_t absentReference = 0;
constexpr std::uint64_t presentReference = 1;

void appendOptionalReference(std::vector<std::uint8_t> &preimage,
                             const std::optional<ArtifactIdentity> &reference) {
  if (!reference) {
    appendU64Be(preimage, absentReference);
    return;
  }
  appendU64Be(preimage, presentReference);
  appendArtifactIdentity(preimage, *reference);
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
  if (llvm::Error error = validateEvaluationScopeCase(query.scope, context))
    return std::move(error);

  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(
          conditions,
          metricDescriptor(query.metric).requestConditionApplicability(),
          context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();

  return MetricRequest(std::move(query), std::move(*canonicalConditions));
}

EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase) {
  std::vector<std::uint8_t> preimage;
  appendFramedString(preimage, baseCaseKeyDomain);
  appendSchemaVersion(preimage, evaluationCase.signature().schemaVersion());
  appendFramedString(preimage, toString(evaluationCase.signature().caseKind()));

  const llvm::ArrayRef<CaseRoleBinding> bindings =
      evaluationCase.subjectBindings().roleBindings();
  appendU64Be(preimage, bindings.size());
  for (const CaseRoleBinding &binding : bindings) {
    detail::appendU32Be(preimage, binding.role.ordinal());
    appendU64Be(preimage, binding.subjects.size());
    for (const ArtifactIdentity &subject : binding.subjects)
      appendArtifactIdentity(preimage, subject);
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
