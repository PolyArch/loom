#include "Evaluation/CaseText.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactText.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;
using detail::rejectUnknownFields;
using detail::requireArray;
using detail::requireObject;
using detail::requireString;
using detail::requireUnsigned;

constexpr llvm::StringLiteral scopeContext = "evaluation scope";
constexpr llvm::StringLiteral targetContext = "evaluation scope target";
constexpr llvm::StringLiteral referenceContext = "artifact reference";

llvm::Expected<std::uint32_t> requireOrdinal(const llvm::json::Object &object,
                                             llvm::StringRef key,
                                             llvm::StringRef context) {
  llvm::Expected<std::uint64_t> value = requireUnsigned(object, key, context);
  if (!value)
    return value.takeError();
  if (*value > std::numeric_limits<std::uint32_t>::max())
    return evaluationError(context + " field '" + key + "' is out of range");
  return static_cast<std::uint32_t>(*value);
}

llvm::Expected<ArtifactIdentity>
requireIdentity(const llvm::json::Object &object, llvm::StringRef key,
                llvm::StringRef context) {
  llvm::Expected<llvm::StringRef> spelling =
      requireString(object, key, context);
  if (!spelling)
    return spelling.takeError();
  return parseArtifactIdentityHex(*spelling);
}

struct ParsedSchema {
  std::string identity;
  SchemaVersion version;
};

llvm::Expected<ParsedSchema> requireSchema(const llvm::json::Object &object,
                                           llvm::StringRef context) {
  llvm::Expected<llvm::StringRef> identity =
      requireString(object, "schema", context);
  if (!identity)
    return identity.takeError();
  llvm::Expected<llvm::StringRef> version =
      requireString(object, "schema_version", context);
  if (!version)
    return version.takeError();
  llvm::Expected<SchemaVersion> parsed = parseSchemaVersion(*version);
  if (!parsed)
    return parsed.takeError();
  return ParsedSchema{identity->str(), *parsed};
}

llvm::Expected<ArtifactRootReference>
parseRootReference(const llvm::json::Object &object, llvm::StringRef context,
                   bool isTarget) {
  if (isTarget) {
    if (llvm::Error error = rejectUnknownFields(
            object, context, {"kind", "schema", "schema_version", "artifact"}))
      return std::move(error);
  } else {
    if (llvm::Error error = rejectUnknownFields(
            object, context, {"schema", "schema_version", "artifact"}))
      return std::move(error);
  }
  llvm::Expected<ParsedSchema> schema = requireSchema(object, context);
  if (!schema)
    return schema.takeError();
  llvm::Expected<ArtifactIdentity> artifact =
      requireIdentity(object, "artifact", context);
  if (!artifact)
    return artifact.takeError();
  return ArtifactRootReference{std::move(schema->identity), schema->version,
                               std::move(*artifact)};
}

llvm::Expected<SubjectTarget> parseTarget(const llvm::json::Object &object) {
  llvm::Expected<llvm::StringRef> kind =
      requireString(object, "kind", targetContext);
  if (!kind)
    return kind.takeError();

  if (*kind == "artifact_root") {
    llvm::Expected<ArtifactRootReference> root =
        parseRootReference(object, targetContext, true);
    if (!root)
      return root.takeError();
    return SubjectTarget{std::move(*root)};
  }

  if (*kind != "artifact_local")
    return evaluationError(targetContext + " has unknown kind '" + *kind + "'");

  if (llvm::Error error =
          rejectUnknownFields(object, targetContext,
                              {"kind", "schema", "schema_version", "artifact",
                               "local_kind", "payload"}))
    return std::move(error);
  llvm::Expected<ParsedSchema> schema = requireSchema(object, targetContext);
  if (!schema)
    return schema.takeError();
  llvm::Expected<ArtifactIdentity> artifact =
      requireIdentity(object, "artifact", targetContext);
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<std::uint32_t> localKind =
      requireOrdinal(object, "local_kind", targetContext);
  if (!localKind)
    return localKind.takeError();
  llvm::Expected<llvm::StringRef> payloadText =
      requireString(object, "payload", targetContext);
  if (!payloadText)
    return payloadText.takeError();
  llvm::Expected<std::vector<std::uint8_t>> payload =
      parseArtifactLocalPayloadHex(*payloadText);
  if (!payload)
    return payload.takeError();

  EncodedArtifactLocalReference local{
      ArtifactRootReference{schema->identity, schema->version,
                            std::move(*artifact)},
      *localKind, std::move(*payload)};
  if (llvm::Error error = validateArtifactLocalReferencePayload(local))
    return error;
  return SubjectTarget{std::move(local)};
}

llvm::Expected<SubjectTargetRef>
parseSubjectTargetRefImpl(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return evaluationError(targetContext + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, targetContext, {"case_subject_role", "anchor", "target"}))
    return std::move(error);

  llvm::Expected<std::uint32_t> caseSubjectRole =
      requireOrdinal(*object, "case_subject_role", targetContext);
  if (!caseSubjectRole)
    return caseSubjectRole.takeError();
  llvm::Expected<const llvm::json::Object *> anchorObject =
      requireObject(*object, "anchor", targetContext);
  if (!anchorObject)
    return anchorObject.takeError();
  llvm::Expected<ArtifactRootReference> anchor =
      parseRootReference(**anchorObject, referenceContext, false);
  if (!anchor)
    return anchor.takeError();
  llvm::Expected<const llvm::json::Object *> targetObject =
      requireObject(*object, "target", targetContext);
  if (!targetObject)
    return targetObject.takeError();
  llvm::Expected<SubjectTarget> target = parseTarget(**targetObject);
  if (!target)
    return target.takeError();

  return SubjectTargetRef{CaseSubjectRoleRef(*caseSubjectRole),
                          std::move(*anchor), std::move(*target)};
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.attribute("schema", reference.schemaIdentity);
  json.attribute("schema_version",
                 formatSchemaVersion(reference.schemaVersion));
  json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
}

} // namespace

void writeArtifactRootReferenceJson(llvm::json::OStream &json,
                                    const ArtifactRootReference &reference) {
  json.object([&] { writeRootReference(json, reference); });
}

llvm::Expected<ArtifactRootReference>
parseArtifactRootReferenceJson(const llvm::json::Object &object) {
  return parseRootReference(object, referenceContext, false);
}

void writeEncodedArtifactLocalReferenceJson(
    llvm::json::OStream &json, const EncodedArtifactLocalReference &reference) {
  json.object([&] {
    writeRootReference(json, reference.artifact);
    json.attribute("local_kind", reference.ownerLocalKind);
    json.attribute("payload", formatArtifactLocalPayloadHex(reference.payload));
  });
}

llvm::Expected<EncodedArtifactLocalReference>
parseEncodedArtifactLocalReferenceJson(const llvm::json::Object &object) {
  if (llvm::Error error = rejectUnknownFields(
          object, referenceContext,
          {"schema", "schema_version", "artifact", "local_kind", "payload"}))
    return std::move(error);
  llvm::Expected<ParsedSchema> schema = requireSchema(object, referenceContext);
  if (!schema)
    return schema.takeError();
  llvm::Expected<ArtifactIdentity> artifact =
      requireIdentity(object, "artifact", referenceContext);
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<std::uint32_t> localKind =
      requireOrdinal(object, "local_kind", referenceContext);
  if (!localKind)
    return localKind.takeError();
  llvm::Expected<llvm::StringRef> payloadText =
      requireString(object, "payload", referenceContext);
  if (!payloadText)
    return payloadText.takeError();
  llvm::Expected<std::vector<std::uint8_t>> payload =
      parseArtifactLocalPayloadHex(*payloadText);
  if (!payload)
    return payload.takeError();
  EncodedArtifactLocalReference reference{
      ArtifactRootReference{std::move(schema->identity), schema->version,
                            std::move(*artifact)},
      *localKind, std::move(*payload)};
  if (llvm::Error error = validateArtifactLocalReferencePayload(reference))
    return std::move(error);
  return reference;
}

void writeSubjectTargetRefJson(llvm::json::OStream &json,
                               const SubjectTargetRef &target) {
  json.object([&] {
    json.attribute("case_subject_role", target.caseSubjectRole.ordinal());
    json.attributeBegin("anchor");
    writeArtifactRootReferenceJson(json, target.anchorSubjectArtifact);
    json.attributeEnd();
    json.attributeBegin("target");
    json.object([&] {
      if (const auto *root =
              std::get_if<ArtifactRootReference>(&target.target)) {
        json.attribute("kind", "artifact_root");
        writeRootReference(json, *root);
        return;
      }
      const auto &local =
          std::get<EncodedArtifactLocalReference>(target.target);
      json.attribute("kind", "artifact_local");
      writeRootReference(json, local.artifact);
      json.attribute("local_kind", local.ownerLocalKind);
      json.attribute("payload", formatArtifactLocalPayloadHex(local.payload));
    });
    json.attributeEnd();
  });
}

llvm::Expected<SubjectTargetRef>
parseSubjectTargetRefJson(const llvm::json::Value &value) {
  return parseSubjectTargetRefImpl(value);
}

void writeEvaluationScopeJson(llvm::json::OStream &json,
                              const EvaluationScope &scope) {
  json.object([&] {
    json.attribute("form", scope.form.ordinal());
    json.attributeArray("targets", [&] {
      for (const SubjectTargetRef &target : scope.targets) {
        writeSubjectTargetRefJson(json, target);
      }
    });
  });
}

llvm::Expected<EvaluationScope>
parseEvaluationScopeJson(const llvm::json::Object &object,
                         llvm::ArrayRef<ScopeFormDescriptor> forms) {
  if (llvm::Error error =
          rejectUnknownFields(object, scopeContext, {"form", "targets"}))
    return std::move(error);
  llvm::Expected<std::uint32_t> formOrdinal =
      requireOrdinal(object, "form", scopeContext);
  if (!formOrdinal)
    return formOrdinal.takeError();
  const ScopeFormDescriptor *descriptor =
      findScopeForm(forms, ScopeFormRef(*formOrdinal));
  if (!descriptor)
    return evaluationError("unknown scope form ordinal " +
                           std::to_string(*formOrdinal));

  llvm::Expected<const llvm::json::Array *> targets =
      requireArray(object, "targets", scopeContext);
  if (!targets)
    return targets.takeError();
  const std::size_t arity = descriptor->roles.size();
  if ((*targets)->size() != arity)
    return evaluationError("scope form " + std::to_string(*formOrdinal) +
                           " requires exactly " + std::to_string(arity) +
                           (arity == 1 ? " target" : " targets"));

  EvaluationScope scope{ScopeFormRef(*formOrdinal), {}};
  scope.targets.reserve(arity);
  for (std::size_t index = 0; index < arity; ++index) {
    llvm::Expected<SubjectTargetRef> target =
        parseSubjectTargetRefImpl((**targets)[index]);
    if (!target)
      return target.takeError();
    scope.targets.push_back(std::move(*target));
  }

  if (llvm::Error error = validateEvaluationScopeForm(forms, scope))
    return std::move(error);
  return scope;
}

} // namespace loom::evaluation
