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

int compareSchemaVersion(SchemaVersion lhs, SchemaVersion rhs) {
  if (lhs.major != rhs.major)
    return lhs.major < rhs.major ? -1 : 1;
  if (lhs.minor != rhs.minor)
    return lhs.minor < rhs.minor ? -1 : 1;
  return 0;
}

int compareSchemaDescriptors(const ArtifactSchemaDescriptor &lhs,
                             const ArtifactSchemaDescriptor &rhs) {
  if (lhs.identity != rhs.identity)
    return lhs.identity < rhs.identity ? -1 : 1;
  return compareSchemaVersion(lhs.version, rhs.version);
}

int compareReferenceTypes(const SubjectReferenceType &lhs,
                          const SubjectReferenceType &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index() ? -1 : 1;
  if (const auto *lhsRoot = std::get_if<ArtifactRootType>(&lhs))
    return compareSchemaDescriptors(
        lhsRoot->schema, std::get<ArtifactRootType>(rhs).schema);
  const auto &lhsLocal = std::get<ArtifactLocalType>(lhs).type;
  const auto &rhsLocal = std::get<ArtifactLocalType>(rhs).type;
  if (int order = compareSchemaDescriptors(lhsLocal.ownerSchema,
                                           rhsLocal.ownerSchema))
    return order;
  if (lhsLocal.ownerLocalKind != rhsLocal.ownerLocalKind)
    return lhsLocal.ownerLocalKind < rhsLocal.ownerLocalKind ? -1 : 1;
  return 0;
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

bool evaluationCaseSignatureRefLess(EvaluationCaseSignatureRef lhs,
                                    EvaluationCaseSignatureRef rhs) {
  if (int order = compareSchemaVersion(lhs.schemaVersion(), rhs.schemaVersion()))
    return order < 0;
  return lhs.caseKind().ordinal() < rhs.caseKind().ordinal();
}

bool orderedTargetPatternLess(const OrderedTargetPattern &lhs,
                              const OrderedTargetPattern &rhs) {
  if (lhs.caseSignature != rhs.caseSignature)
    return evaluationCaseSignatureRefLess(lhs.caseSignature, rhs.caseSignature);
  if (lhs.targets.size() != rhs.targets.size())
    return lhs.targets.size() < rhs.targets.size();
  for (std::size_t index = 0; index < lhs.targets.size(); ++index) {
    const SubjectTargetPattern &lhsTarget = lhs.targets[index];
    const SubjectTargetPattern &rhsTarget = rhs.targets[index];
    if (lhsTarget.caseSubjectRole != rhsTarget.caseSubjectRole)
      return lhsTarget.caseSubjectRole.ordinal() <
             rhsTarget.caseSubjectRole.ordinal();
    if (int order =
            compareReferenceTypes(lhsTarget.referenceType, rhsTarget.referenceType))
      return order < 0;
  }
  return false;
}

bool conditionApplicabilityPatternLess(
    const ConditionApplicabilityPattern &lhs,
    const ConditionApplicabilityPattern &rhs) {
  if (lhs.kind != rhs.kind)
    return lhs.kind < rhs.kind;
  return orderedTargetPatternLess(lhs.targets, rhs.targets);
}

llvm::Error validateOrderedTargetPatternSet(
    llvm::StringRef owner, llvm::ArrayRef<OrderedTargetPattern> patterns) {
  for (std::size_t index = 0; index < patterns.size(); ++index) {
    if (index != 0 && !orderedTargetPatternLess(patterns[index - 1],
                                                patterns[index])) {
      if (patterns[index - 1] == patterns[index])
        return evaluationError("'" + owner +
                               "' declares a duplicate target pattern");
      return evaluationError("'" + owner +
                             "' declares target patterns out of canonical "
                             "order");
    }
  }
  return llvm::Error::success();
}

const ScopeFormDescriptor *
findScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms, ScopeFormRef form) {
  for (const ScopeFormDescriptor &descriptor : forms)
    if (descriptor.form == form)
      return &descriptor;
  return nullptr;
}

const ArtifactRootReference &SubjectTargetRef::targetArtifact() const {
  if (const auto *root = std::get_if<ArtifactRootReference>(&target))
    return *root;
  return std::get<EncodedArtifactLocalReference>(target).artifact;
}

SubjectReferenceType subjectReferenceTypeOf(const SubjectTarget &target) {
  if (const auto *root = std::get_if<ArtifactRootReference>(&target))
    return SubjectReferenceType{ArtifactRootType{root->schema}};
  return SubjectReferenceType{
      ArtifactLocalType{std::get<EncodedArtifactLocalReference>(target).type()}};
}

OrderedTargetPattern
deriveOrderedTargetPattern(llvm::ArrayRef<SubjectTargetRef> targets,
                           EvaluationCaseSignatureRef caseSignature) {
  OrderedTargetPattern pattern{caseSignature, {}};
  pattern.targets.reserve(targets.size());
  for (const SubjectTargetRef &target : targets)
    pattern.targets.push_back(SubjectTargetPattern{
        target.caseSubjectRole, subjectReferenceTypeOf(target.target)});
  return pattern;
}

void detail::appendSubjectTargetKey(std::vector<std::uint8_t> &bytes,
                                    const SubjectTargetRef &target) {
  appendU32Be(bytes, target.caseSubjectRole.ordinal());
  appendFramedBytes(bytes,
                    encodeArtifactRootReference(target.anchorSubjectArtifact));
  if (const auto *root = std::get_if<ArtifactRootReference>(&target.target)) {
    appendU32Be(bytes, artifactRootDiscriminator);
    appendFramedBytes(bytes, encodeArtifactRootReference(*root));
    return;
  }
  appendU32Be(bytes, artifactLocalDiscriminator);
  appendFramedBytes(bytes,
                    encodeArtifactLocalReference(
                        std::get<EncodedArtifactLocalReference>(target.target)));
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

  if (arity == 0) {
    if (!descriptor->acceptedTargetPatterns.empty())
      return evaluationError("a zero-role scope form must not declare target "
                             "patterns");
    return llvm::Error::success();
  }
  if (descriptor->acceptedTargetPatterns.empty())
    return evaluationError("scope form " +
                           std::to_string(descriptor->form.ordinal()) +
                           " must declare a nonempty target pattern set");
  if (llvm::Error error = validateOrderedTargetPatternSet(
          descriptor->semanticDefinition, descriptor->acceptedTargetPatterns))
    return error;
  for (const OrderedTargetPattern &pattern : descriptor->acceptedTargetPatterns)
    if (pattern.targets.size() != arity)
      return evaluationError("scope form " +
                             std::to_string(descriptor->form.ordinal()) +
                             " declares a target pattern of the wrong arity");
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
                                         target.anchorSubjectArtifact))
    return evaluationError(
        "anchor artifact is not bound to case subject role " +
        std::to_string(target.caseSubjectRole.ordinal()) + " of case signature '" +
        signature.spelling + "'");

  const CaseArtifactResolution::Entry *anchor =
      context.resolution().find(target.anchorSubjectArtifact);
  if (!anchor)
    return evaluationError("the anchor artifact of case subject role " +
                           std::to_string(target.caseSubjectRole.ordinal()) +
                           " is unresolved");

  const ArtifactRootReference &targetArtifact = target.targetArtifact();
  if (!CaseArtifactResolution::reaches(*anchor, targetArtifact))
    return evaluationError("target artifact is not reachable from its anchor "
                           "subject's exact dependency closure");

  const CaseArtifactResolution::Entry *entry =
      context.resolution().find(targetArtifact);
  if (!entry)
    return evaluationError("the target artifact is unresolved");

  const auto *local =
      std::get_if<EncodedArtifactLocalReference>(&target.target);
  if (!local)
    return llvm::Error::success();

  // A local reference is only meaningful inside an exact Artifact of its own
  // family, imported through the family's own codec and validator. The family
  // resolves its typed importer view through its own owner boundary.
  return validateArtifactLocalReference(*local);
}

llvm::Error validateEvaluationScopeCase(
    const EvaluationScope &scope, llvm::ArrayRef<ScopeFormDescriptor> forms,
    const CaseTargetContext &context) {
  if (llvm::Error error = validateEvaluationScopeForm(forms, scope))
    return error;
  const ScopeFormDescriptor &descriptor = *findScopeForm(forms, scope.form);

  for (const SubjectTargetRef &target : scope.targets)
    if (llvm::Error error = validateSubjectTargetRef(target, context))
      return error;

  if (!descriptor.roles.empty()) {
    const OrderedTargetPattern derived =
        deriveOrderedTargetPattern(scope.targets, context.signatureRef());
    bool accepted = false;
    for (const OrderedTargetPattern &pattern : descriptor.acceptedTargetPatterns)
      accepted = accepted || pattern == derived;
    if (!accepted)
      return evaluationError(
          "scope form " + std::to_string(descriptor.form.ordinal()) +
          " does not accept the derived ordered target pattern of case "
          "signature '" +
          context.signature().spelling + "'");
  }

  if (descriptor.verifyRelation)
    return descriptor.verifyRelation(scope.targets);
  return llvm::Error::success();
}

CaseTargetContext
EvaluationCase::targetContext(const CaseArtifactResolution &resolution) const {
  return CaseTargetContext(*signature_.descriptor(), signature_, bindings_,
                           resolution);
}

} // namespace loom::evaluation
