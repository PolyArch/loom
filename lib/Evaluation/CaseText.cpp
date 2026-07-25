#include "Evaluation/CaseText.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactText.h"

#include <cstdint>
#include <limits>
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

llvm::Expected<SubjectTarget> parseTarget(const llvm::json::Object &object,
                                          const ScopeRoleDescriptor &role) {
  llvm::Expected<llvm::StringRef> kind =
      requireString(object, "kind", targetContext);
  if (!kind)
    return kind.takeError();

  if (*kind == "artifact_root") {
    if (llvm::Error error =
            rejectUnknownFields(object, targetContext, {"kind", "artifact"}))
      return std::move(error);
    llvm::Expected<ArtifactIdentity> artifact =
        requireIdentity(object, "artifact", targetContext);
    if (!artifact)
      return artifact.takeError();
    return SubjectTarget{ArtifactRootTarget{std::move(*artifact)}};
  }

  if (*kind != "artifact_local")
    return evaluationError(targetContext + " has unknown kind '" + *kind + "'");

  if (llvm::Error error =
          rejectUnknownFields(object, targetContext,
                              {"kind", "family", "family_version", "local_kind",
                               "artifact", "payload"}))
    return std::move(error);
  llvm::Expected<llvm::StringRef> family =
      requireString(object, "family", targetContext);
  if (!family)
    return family.takeError();
  llvm::Expected<llvm::StringRef> familyVersion =
      requireString(object, "family_version", targetContext);
  if (!familyVersion)
    return familyVersion.takeError();
  llvm::Expected<SchemaVersion> version = parseSchemaVersion(*familyVersion);
  if (!version)
    return version.takeError();
  llvm::Expected<llvm::StringRef> localKind =
      requireString(object, "local_kind", targetContext);
  if (!localKind)
    return localKind.takeError();
  llvm::Expected<ArtifactIdentity> artifact =
      requireIdentity(object, "artifact", targetContext);
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<llvm::StringRef> payloadText =
      requireString(object, "payload", targetContext);
  if (!payloadText)
    return payloadText.takeError();
  llvm::Expected<std::vector<std::uint8_t>> payloadBytes =
      detail::parsePayloadHex(*payloadText);
  if (!payloadBytes)
    return payloadBytes.takeError();

  // The role's accepted local targets are the only family resolution path, so
  // decoding never consults a global family catalog.
  for (const AcceptedLocalTarget &accepted : role.acceptedLocalTargets) {
    const ArtifactSchemaDescriptor &schema = *accepted.family->artifactSchema;
    if (schema.identity != *family || schema.version != *version)
      continue;
    if (accepted.family->localKindSpelling(accepted.localKind) != *localKind)
      continue;
    // The family validates its own payload, so a decoded local target is
    // never less checked than a composed one.
    llvm::Expected<LocalTargetRef> local = LocalTargetRef::get(
        *accepted.family, accepted.localKind, std::move(*artifact),
        LocalTargetPayload::fromCanonicalBytes(*payloadBytes));
    if (!local)
      return local.takeError();
    return SubjectTarget{std::move(*local)};
  }
  return evaluationError("scope role '" + role.semanticRole +
                         "' does not accept local target kind '" + *localKind +
                         "' from family '" + *family + "'");
}

llvm::Expected<SubjectTargetRef>
parseSubjectTargetRef(const llvm::json::Value &value,
                      const ScopeRoleDescriptor &role) {
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
  llvm::Expected<ArtifactIdentity> anchor =
      requireIdentity(*object, "anchor", targetContext);
  if (!anchor)
    return anchor.takeError();
  llvm::Expected<const llvm::json::Object *> targetObject =
      requireObject(*object, "target", targetContext);
  if (!targetObject)
    return targetObject.takeError();
  llvm::Expected<SubjectTarget> target = parseTarget(**targetObject, role);
  if (!target)
    return target.takeError();

  return SubjectTargetRef{CaseSubjectRoleRef(*caseSubjectRole),
                          std::move(*anchor), std::move(*target)};
}

} // namespace

void writeEvaluationScopeJson(llvm::json::OStream &json,
                              const EvaluationScope &scope) {
  json.object([&] {
    json.attribute("form", scope.form.ordinal());
    json.attributeArray("targets", [&] {
      for (const SubjectTargetRef &target : scope.targets) {
        json.object([&] {
          json.attribute("case_subject_role", target.caseSubjectRole.ordinal());
          json.attribute("anchor",
                         formatArtifactIdentityHex(target.anchorSubject));
          json.attributeBegin("target");
          json.object([&] {
            if (const auto *root =
                    std::get_if<ArtifactRootTarget>(&target.target)) {
              json.attribute("kind", "artifact_root");
              json.attribute("artifact",
                             formatArtifactIdentityHex(root->artifact));
              return;
            }
            const auto &local = std::get<LocalTargetRef>(target.target);
            const ArtifactSchemaDescriptor &schema =
                *local.family().artifactSchema;
            json.attribute("kind", "artifact_local");
            json.attribute("family", schema.identity);
            json.attribute("family_version",
                           formatSchemaVersion(schema.version));
            json.attribute("local_kind",
                           local.family().localKindSpelling(local.localKind()));
            json.attribute("artifact",
                           formatArtifactIdentityHex(local.artifact()));
            json.attribute("payload",
                           detail::formatPayloadHex(local.payload().bytes()));
          });
          json.attributeEnd();
        });
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
        parseSubjectTargetRef((**targets)[index], descriptor->roles[index]);
    if (!target)
      return target.takeError();
    scope.targets.push_back(std::move(*target));
  }

  if (llvm::Error error = validateEvaluationScopeForm(forms, scope))
    return std::move(error);
  return scope;
}

} // namespace loom::evaluation
