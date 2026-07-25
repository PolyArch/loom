#include "QueryText.h"

#include "CanonicalSupport.h"
#include "Evaluation/CaseText.h"

#include <vector>

namespace loom::evaluation::detail {
namespace {

llvm::Error rejectQueryUnknownFields(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::ArrayRef<llvm::StringRef> payloadFields,
    llvm::ArrayRef<llvm::StringRef> envelopeFields) {
  std::vector<llvm::StringRef> allowed(payloadFields.begin(),
                                       payloadFields.end());
  allowed.insert(allowed.end(), envelopeFields.begin(), envelopeFields.end());
  return rejectUnknownFields(object, context, allowed);
}

} // namespace

void writeMetricQueryPayload(llvm::json::OStream &json,
                             const MetricQuery &query) {
  json.attribute("metric", toString(query.metric));
  json.attributeBegin("scope");
  writeEvaluationScopeJson(json, query.scope);
  json.attributeEnd();
}

llvm::Expected<MetricQuery> parseMetricQueryPayload(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::ArrayRef<llvm::StringRef> envelopeFields) {
  if (llvm::Error error = rejectQueryUnknownFields(
          object, context, {"metric", "scope"}, envelopeFields))
    return std::move(error);
  auto spelling = requireString(object, "metric", context);
  if (!spelling)
    return spelling.takeError();
  auto kind = parseMetricKind(*spelling);
  if (!kind)
    return kind.takeError();
  auto scopeObject = requireObject(object, "scope", context);
  if (!scopeObject)
    return scopeObject.takeError();
  auto scope =
      parseEvaluationScopeJson(**scopeObject, metricDescriptor(*kind).scopeForms);
  if (!scope)
    return scope.takeError();
  return MetricQuery{*kind, std::move(*scope)};
}

void writeFindingQueryPayload(llvm::json::OStream &json,
                              const FindingQuery &query) {
  json.attribute("finding", toString(query.kind));
  json.attributeBegin("scope");
  writeEvaluationScopeJson(json, query.scope);
  json.attributeEnd();
}

llvm::Expected<FindingQuery> parseFindingQueryPayload(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::ArrayRef<llvm::StringRef> envelopeFields) {
  if (llvm::Error error = rejectQueryUnknownFields(
          object, context, {"finding", "scope"}, envelopeFields))
    return std::move(error);
  auto spelling = requireString(object, "finding", context);
  if (!spelling)
    return spelling.takeError();
  auto kind = parseFindingKind(*spelling);
  if (!kind)
    return kind.takeError();
  const FindingDescriptor *descriptor = findFindingDescriptor(*kind);
  if (!descriptor)
    return evaluationError("unregistered finding kind " +
                           std::to_string(kind->ordinal()));
  auto scopeObject = requireObject(object, "scope", context);
  if (!scopeObject)
    return scopeObject.takeError();
  auto scope = parseEvaluationScopeJson(**scopeObject, descriptor->scopeForms);
  if (!scope)
    return scope.takeError();
  return FindingQuery{*kind, std::move(*scope)};
}

} // namespace loom::evaluation::detail
