#include "Evaluation/Finding.h"

#include "CanonicalSupport.h"
#include "Evaluation/CaseText.h"
#include "QueryText.h"

#include "Common/ArtifactText.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::appendFramedBytes;
using detail::appendU32Be;
using detail::evaluationError;
using detail::rejectUnknownFields;
using detail::requireObject;
using detail::requireString;

constexpr llvm::StringLiteral findingQuerySchemaIdentity =
    "evaluation.finding_query";
constexpr SchemaVersion findingQuerySchemaVersion{1, 0};

std::vector<const FindingDescriptor *> &findingDescriptors() {
  static std::vector<const FindingDescriptor *> descriptors;
  return descriptors;
}

std::mutex &findingDescriptorMutex() {
  static std::mutex mutex;
  return mutex;
}

bool findingQueryLess(const FindingQuery &lhs, const FindingQuery &rhs) {
  return canonicalFindingQueryKey(lhs) < canonicalFindingQueryKey(rhs);
}

} // namespace

llvm::Error registerFindingDescriptor(const FindingDescriptor &descriptor) {
  if (descriptor.spelling.empty())
    return evaluationError("a finding descriptor requires a spelling");
  if (descriptor.semanticDefinition.empty())
    return evaluationError("finding '" + descriptor.spelling +
                           "' requires a semantic definition");
  if (descriptor.scopeForms.empty())
    return evaluationError("finding '" + descriptor.spelling +
                           "' requires at least one scope form");
  if (descriptor.occurrenceCodec.occurrenceSchema.identity.empty() ||
      !descriptor.occurrenceCodec.encode ||
      !descriptor.occurrenceCodec.decode ||
      !descriptor.occurrenceCodec.validate)
    return evaluationError("finding '" + descriptor.spelling +
                           "' requires a complete occurrence codec");
  if (descriptor.terminalWitnessCodec &&
      (descriptor.terminalWitnessCodec->witnessSchema.identity.empty() ||
       !descriptor.terminalWitnessCodec->encode ||
       !descriptor.terminalWitnessCodec->decode ||
       !descriptor.terminalWitnessCodec->validate))
    return evaluationError("finding '" + descriptor.spelling +
                           "' requires a complete terminal-witness codec");

  if (llvm::Error error = validateScopeFormDescriptors(descriptor.scopeForms))
    return error;
  for (const ScopeFormDescriptor &form : descriptor.scopeForms)
    if (form.referenceCycleRequirement !=
        ReferenceCycleRequirement::NotRequired)
      return evaluationError("finding '" + descriptor.spelling +
                             "' cannot require a reference-cycle basis");
  if (llvm::Error error = validateConditionApplicabilityPatternSet(
          descriptor.spelling, descriptor.permittedRequestConditionPatterns,
          ConditionLocation::FindingRequest))
    return error;

  std::lock_guard<std::mutex> lock(findingDescriptorMutex());
  for (const FindingDescriptor *existing : findingDescriptors()) {
    if (existing->kind == descriptor.kind) {
      if (existing == &descriptor)
        return llvm::Error::success();
      return evaluationError("conflicting registration for finding kind " +
                             std::to_string(descriptor.kind.ordinal()));
    }
    if (existing->spelling == descriptor.spelling)
      return evaluationError("conflicting registration for finding '" +
                             descriptor.spelling + "'");
  }
  findingDescriptors().push_back(&descriptor);
  std::sort(findingDescriptors().begin(), findingDescriptors().end(),
            [](const FindingDescriptor *lhs, const FindingDescriptor *rhs) {
              return lhs->kind < rhs->kind;
            });
  return llvm::Error::success();
}

const FindingDescriptor *findFindingDescriptor(FindingKind kind) {
  std::lock_guard<std::mutex> lock(findingDescriptorMutex());
  for (const FindingDescriptor *descriptor : findingDescriptors())
    if (descriptor->kind == kind)
      return descriptor;
  return nullptr;
}

llvm::Expected<FindingKind> parseFindingKind(llvm::StringRef spelling) {
  std::lock_guard<std::mutex> lock(findingDescriptorMutex());
  for (const FindingDescriptor *descriptor : findingDescriptors())
    if (descriptor->spelling == spelling)
      return descriptor->kind;
  return evaluationError("unknown FindingKind '" + spelling + "'");
}

llvm::StringRef toString(FindingKind kind) {
  const FindingDescriptor *descriptor = findFindingDescriptor(kind);
  return descriptor ? descriptor->spelling : llvm::StringRef{};
}

llvm::Error validateFindingQuery(const FindingQuery &query) {
  const FindingDescriptor *descriptor = findFindingDescriptor(query.kind);
  if (!descriptor)
    return evaluationError("unregistered finding kind " +
                           std::to_string(query.kind.ordinal()));
  return validateEvaluationScopeForm(descriptor->scopeForms, query.scope);
}

std::vector<std::uint8_t> canonicalFindingQueryKey(const FindingQuery &query) {
  std::vector<std::uint8_t> key;
  appendU32Be(key, query.kind.ordinal());
  appendFramedBytes(key, canonicalScopeKey(query.scope));
  return key;
}

llvm::Expected<std::vector<FindingQuery>>
canonicalizeFindingQueries(llvm::ArrayRef<FindingQuery> queries) {
  std::vector<FindingQuery> canonical(queries.begin(), queries.end());
  for (const FindingQuery &query : canonical)
    if (llvm::Error error = validateFindingQuery(query))
      return std::move(error);
  std::sort(canonical.begin(), canonical.end(), findingQueryLess);
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1] == canonical[index])
      return evaluationError("duplicate finding query for '" +
                             toString(canonical[index].kind) + "'");
  return canonical;
}

llvm::Expected<std::string> serializeFindingQuery(const FindingQuery &query) {
  if (llvm::Error error = validateFindingQuery(query))
    return std::move(error);

  llvm::SmallString<256> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", findingQuerySchemaIdentity);
    json.attribute("schema_version",
                   formatSchemaVersion(findingQuerySchemaVersion));
    detail::writeFindingQueryPayload(json, query);
  });
  return output.str().str();
}

llvm::Expected<FindingQuery> parseFindingQuery(llvm::StringRef jsonText) {
  auto value = llvm::json::parse(jsonText);
  if (!value)
    return value.takeError();
  const llvm::json::Object *root = value->getAsObject();
  if (!root)
    return evaluationError("evaluation.finding_query root must be an object");
  auto schema = requireString(*root, "schema", "evaluation.finding_query root");
  if (!schema)
    return schema.takeError();
  if (*schema != findingQuerySchemaIdentity)
    return evaluationError("unsupported finding query schema '" + *schema +
                           "'");
  auto version =
      requireString(*root, "schema_version", "evaluation.finding_query root");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (*parsedVersion != findingQuerySchemaVersion)
    return evaluationError("unsupported evaluation.finding_query version '" +
                           *version + "'");

  const llvm::StringRef envelopeFields[] = {"schema", "schema_version"};
  auto query = detail::parseFindingQueryPayload(
      *root, "evaluation.finding_query root", envelopeFields);
  if (!query)
    return query.takeError();
  auto canonical = serializeFindingQuery(*query);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != jsonText)
    return evaluationError("finding query JSON is not canonical");
  return *query;
}

} // namespace loom::evaluation
