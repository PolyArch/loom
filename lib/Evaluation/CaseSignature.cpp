#include "Evaluation/Case.h"
#include "Evaluation/ProductionRegistry.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactText.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

constexpr SchemaVersion evaluationSchema30{3, 0};

bool isSupportedEvaluationSchema(SchemaVersion version) {
  return version == evaluationSchema30;
}

bool schemaContainsCaseKind(SchemaVersion version,
                            EvaluationCaseKind) {
  return version == evaluationSchema30;
}

struct CaseSignatureRegistryEntry {
  SchemaVersion version;
  const EvaluationCaseSignatureDescriptor *descriptor;
};

std::vector<CaseSignatureRegistryEntry> &caseSignatures() {
  static std::vector<CaseSignatureRegistryEntry> descriptors;
  return descriptors;
}

std::mutex &caseSignatureMutex() {
  static std::mutex mutex;
  return mutex;
}

llvm::Error unresolvedArtifact(llvm::StringRef reference) {
  return evaluationError(reference + " artifact is unresolved");
}

llvm::Error validateAcceptedSchema(
    llvm::StringRef reference,
    llvm::ArrayRef<const ArtifactSchemaDescriptor *> accepted,
    const ArtifactRootReference &artifact) {
  for (const ArtifactSchemaDescriptor *schema : accepted)
    if (schema->identity == artifact.schemaIdentity &&
        schema->version == artifact.schemaVersion)
      return llvm::Error::success();
  return evaluationError(reference + " does not accept schema '" +
                         artifact.schemaIdentity + " " +
                         formatSchemaVersion(artifact.schemaVersion) + "'");
}

llvm::Error validateReferenceRequirement(
    const EvaluationCaseSignatureDescriptor &signature,
    ArtifactRequirement requirement,
    llvm::ArrayRef<const ArtifactSchemaDescriptor *> accepted,
    llvm::StringRef reference,
    const std::optional<ArtifactRootReference> &value,
    const CaseArtifactResolution &resolution) {
  if (!value) {
    if (requirement == ArtifactRequirement::Required)
      return evaluationError("case signature '" + signature.spelling +
                             "' requires a " + reference + " reference");
    return llvm::Error::success();
  }
  if (requirement == ArtifactRequirement::Forbidden)
    return evaluationError("case signature '" + signature.spelling +
                           "' forbids a " + reference + " reference");
  if (!resolution.find(*value))
    return unresolvedArtifact(reference);
  return validateAcceptedSchema(reference, accepted, *value);
}

llvm::Error
validateRoleDefinitions(const EvaluationCaseSignatureDescriptor &descriptor) {
  for (std::size_t index = 0; index < descriptor.subjectRoles.size(); ++index) {
    const CaseSubjectRoleDescriptor &role = descriptor.subjectRoles[index];
    if (role.role != CaseSubjectRoleRef(static_cast<std::uint32_t>(index)))
      return evaluationError("case signature '" + descriptor.spelling +
                             "' must declare ordered nonrepeating subject "
                             "roles");
    if (role.acceptedSchemas.empty())
      return evaluationError("subject role '" + role.semanticRole +
                             "' must accept at least one artifact schema");
    for (std::size_t earlier = 0; earlier < index; ++earlier)
      if (descriptor.subjectRoles[earlier].semanticRole == role.semanticRole)
        return evaluationError("case signature '" + descriptor.spelling +
                               "' must declare ordered nonrepeating subject "
                               "roles");
  }
  return llvm::Error::success();
}

llvm::Error
validateBasePatternSet(const EvaluationCaseSignatureDescriptor &descriptor) {
  return validateConditionApplicabilityPatternSet(
      descriptor.spelling, descriptor.permittedBaseConditions,
      ConditionLocation::Base, &descriptor);
}

} // namespace

SchemaVersion evaluationSchemaVersion() { return evaluationSchema30; }

const CaseSubjectRoleDescriptor *
EvaluationCaseSignatureDescriptor::findSubjectRole(
    CaseSubjectRoleRef role) const {
  for (const CaseSubjectRoleDescriptor &descriptor : subjectRoles)
    if (descriptor.role == role)
      return &descriptor;
  return nullptr;
}

llvm::Expected<EvaluationCaseSignatureRef>
EvaluationCaseSignatureRef::get(SchemaVersion schemaVersion,
                                EvaluationCaseKind caseKind) {
  if (!isSupportedEvaluationSchema(schemaVersion))
    return evaluationError("unsupported evaluation schema version '" +
                           formatSchemaVersion(schemaVersion) + "'");
  if (!schemaContainsCaseKind(schemaVersion, caseKind))
    return evaluationError(
        "evaluation schema version '" + formatSchemaVersion(schemaVersion) +
        "' does not contain case kind " + std::to_string(caseKind.ordinal()));
  return EvaluationCaseSignatureRef(schemaVersion, caseKind);
}

const EvaluationCaseSignatureDescriptor *
EvaluationCaseSignatureRef::descriptor() const {
  if (!schemaContainsCaseKind(schemaVersion_, caseKind_))
    return nullptr;
  return findEvaluationCaseSignature(schemaVersion_, caseKind_);
}

llvm::Error registerEvaluationCaseSignature(
    const EvaluationCaseSignatureDescriptor &descriptor) {
  if (descriptor.registryVersion != evaluationSchemaVersion())
    return evaluationError(
        "authored case descriptors must use the current registry version");
  llvm::Expected<EvaluationCaseSignatureRef> signatureRef =
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(),
                                      descriptor.caseKind);
  if (!signatureRef)
    return signatureRef.takeError();
  if (descriptor.spelling.empty())
    return evaluationError("a case signature requires a spelling");
  if (llvm::Error error = validateRoleDefinitions(descriptor))
    return error;
  if (descriptor.workload == ArtifactRequirement::Forbidden &&
      !descriptor.acceptedWorkloadSchemas.empty())
    return evaluationError("case signature '" + descriptor.spelling +
                           "' forbids a workload but accepts workload schemas");
  if (descriptor.runtimeInput == ArtifactRequirement::Forbidden &&
      !descriptor.acceptedRuntimeInputSchemas.empty())
    return evaluationError("case signature '" + descriptor.spelling +
                           "' forbids a runtime input but accepts runtime "
                           "input schemas");
  if (const auto *exact =
          std::get_if<ExactSubjectCycle>(&descriptor.wholeCaseCycleBasis)) {
    if (!exact->resolve)
      return evaluationError("case signature '" + descriptor.spelling +
                             "' requires an exact subject-cycle resolver");
    const bool emptyReferenceType = std::visit(
        [](const auto &referenceType) {
          using T = std::decay_t<decltype(referenceType)>;
          if constexpr (std::is_same_v<T, ArtifactRootType>)
            return referenceType.schemaIdentity.empty();
          else
            return referenceType.type.ownerSchemaIdentity.empty();
        },
        exact->acceptedReferenceType);
    if (emptyReferenceType)
      return evaluationError("case signature '" + descriptor.spelling +
                             "' requires an exact reference-cycle type");
  }
  if (llvm::Error error = validateBasePatternSet(descriptor))
    return error;

  std::lock_guard<std::mutex> lock(caseSignatureMutex());
  for (const CaseSignatureRegistryEntry &entry : caseSignatures()) {
    if (entry.version != evaluationSchemaVersion())
      continue;
    const EvaluationCaseSignatureDescriptor *existing = entry.descriptor;
    if (existing->caseKind == descriptor.caseKind) {
      if (existing == &descriptor)
        return llvm::Error::success();
      return evaluationError("conflicting registration for evaluation case "
                             "kind " +
                             std::to_string(descriptor.caseKind.ordinal()));
    }
    if (existing->spelling == descriptor.spelling)
      return evaluationError("conflicting registration for evaluation case "
                             "signature '" +
                             descriptor.spelling + "'");
  }
  caseSignatures().push_back({evaluationSchemaVersion(), &descriptor});

  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor *
findEvaluationCaseSignature(EvaluationCaseKind caseKind) {
  return findEvaluationCaseSignature(evaluationSchemaVersion(), caseKind);
}

const EvaluationCaseSignatureDescriptor *
findEvaluationCaseSignature(SchemaVersion schemaVersion,
                            EvaluationCaseKind caseKind) {
  std::lock_guard<std::mutex> lock(caseSignatureMutex());
  for (const CaseSignatureRegistryEntry &entry : caseSignatures())
    if (entry.version == schemaVersion &&
        entry.descriptor->caseKind == caseKind)
      return entry.descriptor;
  return nullptr;
}

llvm::Expected<ReferenceCycleBasis>
resolveReferenceCycleBasis(const EvaluationCase &evaluationCase,
                           const CaseArtifactResolution &resolution,
                           const ArtifactStore &artifactStore,
                           const BlobStore &blobStore) {
  const EvaluationCaseSignatureDescriptor *descriptor =
      evaluationCase.signature().descriptor();
  if (!descriptor)
    return evaluationError("the EvaluationCase signature is unresolved");
  if (std::holds_alternative<AbsentReferenceCycle>(
          descriptor->wholeCaseCycleBasis))
    return evaluationError("case signature '" + descriptor->spelling +
                           "' has no whole-case reference cycle");
  if (std::holds_alternative<AbstractCaseCycle>(
          descriptor->wholeCaseCycleBasis))
    return ReferenceCycleBasis{AbstractCaseCycle{}};

  const ExactSubjectCycle &exact =
      std::get<ExactSubjectCycle>(descriptor->wholeCaseCycleBasis);
  auto target =
      exact.resolve(evaluationCase, resolution, artifactStore, blobStore);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateSubjectTargetRef(
          *target, evaluationCase.targetContext(resolution, artifactStore)))
    return std::move(error);
  if (subjectReferenceTypeOf(target->target) != exact.acceptedReferenceType)
    return evaluationError("case signature '" + descriptor->spelling +
                           "' resolved a reference cycle of the wrong type");
  return ReferenceCycleBasis{std::move(*target)};
}

llvm::Expected<EvaluationSubjectBindings>
EvaluationSubjectBindings::get(std::vector<CaseRoleBinding> bindings) {
  std::sort(bindings.begin(), bindings.end(),
            [](const CaseRoleBinding &lhs, const CaseRoleBinding &rhs) {
              return lhs.role.ordinal() < rhs.role.ordinal();
            });
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    CaseRoleBinding &binding = bindings[index];
    if (index != 0 && bindings[index - 1].role == binding.role)
      return evaluationError("duplicate subject role " +
                             std::to_string(binding.role.ordinal()) +
                             " in subject bindings");
    if (binding.subjects.empty())
      return evaluationError("subject role " +
                             std::to_string(binding.role.ordinal()) +
                             " must bind at least one subject");
    std::sort(binding.subjects.begin(), binding.subjects.end(),
              artifactRootReferenceLess);
    for (std::size_t subject = 1; subject < binding.subjects.size(); ++subject)
      if (binding.subjects[subject - 1] == binding.subjects[subject])
        return evaluationError("duplicate subject artifact for role " +
                               std::to_string(binding.role.ordinal()));
  }
  return EvaluationSubjectBindings(std::move(bindings));
}

llvm::ArrayRef<ArtifactRootReference>
EvaluationSubjectBindings::subjects(CaseSubjectRoleRef role) const {
  for (const CaseRoleBinding &binding : bindings_)
    if (binding.role == role)
      return binding.subjects;
  return {};
}

bool EvaluationSubjectBindings::isBoundSubject(
    CaseSubjectRoleRef role, const ArtifactRootReference &subject) const {
  llvm::ArrayRef<ArtifactRootReference> bound = subjects(role);
  return std::find(bound.begin(), bound.end(), subject) != bound.end();
}

llvm::Expected<CaseArtifactResolution>
CaseArtifactResolution::get(std::vector<Entry> entries) {
  std::sort(entries.begin(), entries.end(),
            [](const Entry &lhs, const Entry &rhs) {
              return artifactRootReferenceLess(lhs.artifact, rhs.artifact);
            });
  for (std::size_t index = 0; index < entries.size(); ++index) {
    Entry &entry = entries[index];
    if (index != 0 && entries[index - 1].artifact == entry.artifact)
      return evaluationError("duplicate resolution for one artifact");
    std::sort(entry.dependencyClosure.begin(), entry.dependencyClosure.end(),
              artifactRootReferenceLess);
    for (std::size_t member = 1; member < entry.dependencyClosure.size();
         ++member)
      if (entry.dependencyClosure[member - 1] ==
          entry.dependencyClosure[member])
        return evaluationError("duplicate artifact in one dependency closure");
  }
  return CaseArtifactResolution(std::move(entries));
}

const CaseArtifactResolution::Entry *
CaseArtifactResolution::find(const ArtifactRootReference &artifact) const {
  for (const Entry &entry : entries_)
    if (entry.artifact == artifact)
      return &entry;
  return nullptr;
}

bool CaseArtifactResolution::reaches(const Entry &entry,
                                     const ArtifactRootReference &dependency) {
  if (entry.artifact == dependency)
    return true;
  return std::find(entry.dependencyClosure.begin(),
                   entry.dependencyClosure.end(),
                   dependency) != entry.dependencyClosure.end();
}

llvm::Expected<EvaluationCase> EvaluationCase::get(
    EvaluationCaseSignatureRef signature, EvaluationSubjectBindings bindings,
    std::optional<ArtifactRootReference> workload,
    std::optional<ArtifactRootReference> runtimeInput,
    llvm::ArrayRef<EvaluationCondition> baseConditions,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const EvaluationCaseSignatureDescriptor *descriptor = signature.descriptor();
  if (!descriptor)
    return evaluationError("unregistered evaluation case kind " +
                           std::to_string(signature.caseKind().ordinal()));

  for (const CaseRoleBinding &binding : bindings.roleBindings())
    if (!descriptor->findSubjectRole(binding.role))
      return evaluationError(
          "case subject role " + std::to_string(binding.role.ordinal()) +
          " is not a role of case signature '" + descriptor->spelling + "'");

  for (const CaseSubjectRoleDescriptor &role : descriptor->subjectRoles) {
    llvm::ArrayRef<ArtifactRootReference> subjects =
        bindings.subjects(role.role);
    if (subjects.empty())
      return evaluationError("case signature '" + descriptor->spelling +
                             "' requires a binding for subject role " +
                             std::to_string(role.role.ordinal()) + " ('" +
                             role.semanticRole + "')");
    if (role.cardinality == SubjectRoleCardinality::ExactlyOne &&
        subjects.size() != 1)
      return evaluationError("subject role '" + role.semanticRole +
                             "' requires exactly one subject");
    for (const ArtifactRootReference &subject : subjects) {
      const std::string reference =
          ("subject role '" + role.semanticRole + "'").str();
      if (!resolution.find(subject))
        return unresolvedArtifact(reference);
      if (llvm::Error error =
              validateAcceptedSchema(reference, role.acceptedSchemas, subject))
        return std::move(error);
    }
  }

  if (llvm::Error error =
          validateReferenceRequirement(*descriptor, descriptor->workload,
                                       descriptor->acceptedWorkloadSchemas,
                                       "workload", workload, resolution))
    return std::move(error);
  if (llvm::Error error = validateReferenceRequirement(
          *descriptor, descriptor->runtimeInput,
          descriptor->acceptedRuntimeInputSchemas, "runtime input",
          runtimeInput, resolution))
    return std::move(error);

  const CaseTargetContext context(*descriptor, signature, bindings, resolution,
                                  artifactStore);
  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(
          baseConditions, ConditionLocation::Base, descriptor->spelling,
          descriptor->permittedBaseConditions, context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();

  EvaluationCase evaluationCase(signature, std::move(bindings),
                                std::move(workload), std::move(runtimeInput),
                                std::move(*canonicalConditions));
  for (const CaseSubjectRoleDescriptor &role : descriptor->subjectRoles) {
    if (!role.verifyCrossRoleCompatibility)
      continue;
    for (const ArtifactRootReference &subject :
         evaluationCase.subjectBindings().subjects(role.role))
      if (llvm::Error error = role.verifyCrossRoleCompatibility(
              subject, evaluationCase, evaluationCase.subjectBindings(),
              resolution, artifactStore, blobStore))
        return std::move(error);
  }
  if (descriptor->verifyWorkloadCompatibility)
    if (llvm::Error error = descriptor->verifyWorkloadCompatibility(
            evaluationCase, evaluationCase.subjectBindings(),
            evaluationCase.workload(), evaluationCase.runtimeInput(),
            resolution, artifactStore, blobStore))
      return std::move(error);

  return evaluationCase;
}

} // namespace loom::evaluation
