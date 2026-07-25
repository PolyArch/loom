#include "Evaluation/Case.h"

#include "CanonicalSupport.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

constexpr std::uint32_t artifactRootDiscriminator = 0;
constexpr std::uint32_t artifactLocalDiscriminator = 1;

std::string localKindText(const LocalTargetRef &local) {
  llvm::StringRef spelling =
      local.family().localKindSpelling(local.localKind());
  if (!spelling.empty())
    return spelling.str();
  return std::to_string(local.localKind().value());
}

bool isSameFamily(const LocalTargetFamilyDescriptor &lhs,
                  const LocalTargetFamilyDescriptor &rhs) {
  return lhs.artifactSchema->identity == rhs.artifactSchema->identity &&
         lhs.artifactSchema->version == rhs.artifactSchema->version;
}

llvm::Error validateAcceptedTargetKind(const ScopeRoleDescriptor &role,
                                       const SubjectTargetRef &target) {
  if (std::holds_alternative<ArtifactRootTarget>(target.target)) {
    if (!role.acceptsArtifactRoot)
      return evaluationError("scope role '" + role.semanticRole +
                             "' does not accept an artifact-root target");
    return llvm::Error::success();
  }

  const auto &local = std::get<LocalTargetRef>(target.target);
  for (const AcceptedLocalTarget &accepted : role.acceptedLocalTargets)
    if (isSameFamily(*accepted.family, local.family()) &&
        accepted.localKind == local.localKind())
      return llvm::Error::success();
  return evaluationError("scope role '" + role.semanticRole +
                         "' does not accept local target kind '" +
                         localKindText(local) + "' from family '" +
                         local.family().artifactSchema->identity + "'");
}

/// A form's role tuple is ordered and nonrepeating: each role ordinal is its
/// own position and no semantic role occurs twice.
llvm::Error validateFormRoles(const ScopeFormDescriptor &descriptor) {
  for (std::size_t index = 0; index < descriptor.roles.size(); ++index) {
    const ScopeRoleDescriptor &role = descriptor.roles[index];
    if (role.role != ScopeRoleRef(static_cast<std::uint32_t>(index)))
      return evaluationError("scope form " +
                             std::to_string(descriptor.form.ordinal()) +
                             " must declare ordered nonrepeating roles");
    for (std::size_t earlier = 0; earlier < index; ++earlier)
      if (descriptor.roles[earlier].semanticRole == role.semanticRole)
        return evaluationError("scope form " +
                               std::to_string(descriptor.form.ordinal()) +
                               " must declare ordered nonrepeating roles");
  }
  return llvm::Error::success();
}

} // namespace

bool operator==(const LocalTargetRef &lhs, const LocalTargetRef &rhs) {
  return isSameFamily(lhs.family(), rhs.family()) &&
         lhs.localKind() == rhs.localKind() &&
         lhs.artifact() == rhs.artifact() && lhs.payload() == rhs.payload();
}

llvm::Expected<LocalTargetRef>
LocalTargetRef::get(const LocalTargetFamilyDescriptor &family,
                    LocalTargetKind kind, ArtifactIdentity artifact,
                    LocalTargetPayload payload) {
  if (llvm::Error error = family.validateLocalTarget(kind, payload))
    return std::move(error);
  return LocalTargetRef(family, kind, std::move(artifact), std::move(payload));
}

const ArtifactIdentity &SubjectTargetRef::targetArtifact() const {
  if (const auto *root = std::get_if<ArtifactRootTarget>(&target))
    return root->artifact;
  return std::get<LocalTargetRef>(target).artifact();
}

const ScopeFormDescriptor *
findScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms, ScopeFormRef form) {
  for (const ScopeFormDescriptor &descriptor : forms)
    if (descriptor.form == form)
      return &descriptor;
  return nullptr;
}

void detail::appendSubjectTargetKey(std::vector<std::uint8_t> &bytes,
                                    const SubjectTargetRef &target) {
  appendU32Be(bytes, target.caseSubjectRole.ordinal());
  appendArtifactIdentity(bytes, target.anchorSubject);
  if (const auto *root = std::get_if<ArtifactRootTarget>(&target.target)) {
    appendU32Be(bytes, artifactRootDiscriminator);
    appendArtifactIdentity(bytes, root->artifact);
    return;
  }
  const auto &local = std::get<LocalTargetRef>(target.target);
  appendU32Be(bytes, artifactLocalDiscriminator);
  appendArtifactIdentity(bytes, local.artifact());
  appendFramedString(bytes, local.family().artifactSchema->identity);
  appendSchemaVersion(bytes, local.family().artifactSchema->version);
  appendU32Be(bytes, local.localKind().value());
  appendFramedBytes(bytes, local.payload().bytes());
}

std::vector<std::uint8_t> canonicalScopeKey(const EvaluationScope &scope) {
  std::vector<std::uint8_t> key;
  detail::appendU32Be(key, scope.form.ordinal());
  detail::appendU64Be(key, scope.targets.size());
  for (const SubjectTargetRef &target : scope.targets)
    detail::appendSubjectTargetKey(key, target);
  return key;
}

llvm::Error
validateEvaluationScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms,
                            const EvaluationScope &scope) {
  const ScopeFormDescriptor *descriptor = findScopeForm(forms, scope.form);
  if (!descriptor)
    return evaluationError("unknown scope form ordinal " +
                           std::to_string(scope.form.ordinal()));
  if (llvm::Error error = validateFormRoles(*descriptor))
    return error;

  const std::size_t arity = descriptor->roles.size();
  if (scope.targets.size() != arity)
    return evaluationError("scope form " +
                           std::to_string(scope.form.ordinal()) +
                           " requires exactly " + std::to_string(arity) +
                           (arity == 1 ? " target" : " targets"));

  for (std::size_t index = 0; index < arity; ++index)
    if (llvm::Error error = validateAcceptedTargetKind(descriptor->roles[index],
                                                       scope.targets[index]))
      return error;

  if (descriptor->verifyRelation)
    return descriptor->verifyRelation(scope.targets);
  return llvm::Error::success();
}

llvm::Error validateSubjectTargetRef(const SubjectTargetRef &target,
                                     const CaseTargetContext &context) {
  const EvaluationCaseSignatureDescriptor &signature = context.signature();
  if (!signature.findSubjectRole(target.caseSubjectRole))
    return evaluationError("case subject role " +
                           std::to_string(target.caseSubjectRole.ordinal()) +
                           " is not a role of case signature '" +
                           signature.spelling + "'");

  if (!context.bindings().isBoundSubject(target.caseSubjectRole,
                                         target.anchorSubject))
    return evaluationError(
        "anchor artifact is not bound to case subject role " +
        std::to_string(target.caseSubjectRole.ordinal()) +
        " of case signature '" + signature.spelling + "'");

  const CaseArtifactResolution::Entry *anchor =
      context.resolution().find(target.anchorSubject);
  if (!anchor)
    return evaluationError("the anchor artifact of case subject role " +
                           std::to_string(target.caseSubjectRole.ordinal()) +
                           " is unresolved");

  const ArtifactIdentity &targetArtifact = target.targetArtifact();
  if (!CaseArtifactResolution::reaches(*anchor, targetArtifact))
    return evaluationError("target artifact is not reachable from its anchor "
                           "subject");

  const auto *local = std::get_if<LocalTargetRef>(&target.target);
  if (!local)
    return llvm::Error::success();

  // A local reference is only meaningful inside an Artifact of its own family.
  const CaseArtifactResolution::Entry *entry =
      context.resolution().find(targetArtifact);
  if (!entry)
    return evaluationError("the target artifact is unresolved");
  const ArtifactSchemaDescriptor &familySchema =
      *local->family().artifactSchema;
  if (entry->schema->identity != familySchema.identity ||
      entry->schema->version != familySchema.version)
    return evaluationError("the target artifact does not belong to family '" +
                           familySchema.identity + "'");
  return llvm::Error::success();
}

llvm::Error validateEvaluationScopeCase(const EvaluationScope &scope,
                                        const CaseTargetContext &context) {
  for (const SubjectTargetRef &target : scope.targets)
    if (llvm::Error error = validateSubjectTargetRef(target, context))
      return error;
  return llvm::Error::success();
}

} // namespace loom::evaluation
