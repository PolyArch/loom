#include "Evaluation/Case.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactText.h"
#include "Fabric/ArtifactSchema.h"
#include "Mapping/ArtifactSchema.h"
#include "Simulator/ArtifactSchema.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace loom::evaluation {
namespace {

using detail::evaluationError;

constexpr SchemaVersion evaluationSchema{1, 0};

bool identityLess(const ArtifactIdentity &lhs, const ArtifactIdentity &rhs) {
  return lhs.bytes() < rhs.bytes();
}

llvm::Error unresolvedArtifact(llvm::StringRef reference) {
  return evaluationError(reference + " artifact is unresolved");
}

llvm::Error validateAcceptedSchema(
    llvm::StringRef reference,
    llvm::ArrayRef<const ArtifactSchemaDescriptor *> accepted,
    const ArtifactIdentity &artifact,
    const CaseArtifactResolution &resolution) {
  const CaseArtifactResolution::Entry *entry = resolution.find(artifact);
  if (!entry)
    return unresolvedArtifact(reference);
  for (const ArtifactSchemaDescriptor *schema : accepted)
    if (schema->identity == entry->schema->identity &&
        schema->version == entry->schema->version)
      return llvm::Error::success();
  return evaluationError(reference + " does not accept schema '" +
                         entry->schema->identity + " " +
                         formatSchemaVersion(entry->schema->version) + "'");
}

llvm::Error validateReferenceRequirement(
    const EvaluationCaseSignatureDescriptor &signature,
    ArtifactRequirement requirement,
    llvm::ArrayRef<const ArtifactSchemaDescriptor *> accepted,
    llvm::StringRef reference, const std::optional<ArtifactIdentity> &value,
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
  return validateAcceptedSchema(reference, accepted, *value, resolution);
}

//===----------------------------------------------------------------------===//
// The MappedWorkloadExecution signature
//===----------------------------------------------------------------------===//

const ArtifactSchemaDescriptor *const acceptedFabricSchemas[] = {
    &fabric::artifactSchema};
const ArtifactSchemaDescriptor *const acceptedMappingSchemas[] = {
    &loom::mapping::artifactSchema};
const ArtifactSchemaDescriptor *const acceptedWorkloadSchemas[] = {
    &loom::sim::workloadSchema};
const ArtifactSchemaDescriptor *const acceptedRuntimeInputSchemas[] = {
    &loom::sim::runtimeInputSchema};

/// A Mapping root binds its exact Fabric upstream, so a bound Mapping subject
/// must depend on every bound Fabric subject of the same case.
llvm::Error verifyMappingBindsFabric(const ArtifactIdentity &subject,
                                     const EvaluationSubjectBindings &bindings,
                                     const CaseArtifactResolution &resolution) {
  const CaseArtifactResolution::Entry *entry = resolution.find(subject);
  if (!entry)
    return unresolvedArtifact("mapping subject");
  for (const ArtifactIdentity &fabricSubject :
       bindings.subjects(CaseSubjectRoleRef(0)))
    if (!CaseArtifactResolution::reaches(*entry, fabricSubject))
      return evaluationError("the bound mapping subject does not depend on the "
                             "bound fabric subject");
  return llvm::Error::success();
}

/// A runtime input references its exact workload, so the two orthogonal
/// references of one case must agree.
llvm::Error verifyRuntimeInputBindsWorkload(
    const EvaluationSubjectBindings &bindings,
    const std::optional<ArtifactIdentity> &workload,
    const std::optional<ArtifactIdentity> &runtimeInput,
    const CaseArtifactResolution &resolution) {
  if (!runtimeInput)
    return llvm::Error::success();
  const CaseArtifactResolution::Entry *entry = resolution.find(*runtimeInput);
  if (!entry)
    return unresolvedArtifact("runtime input");
  if (!workload || !CaseArtifactResolution::reaches(*entry, *workload))
    return evaluationError("the runtime input does not reference the exact "
                           "workload of this case");
  return llvm::Error::success();
}

const CaseSubjectRoleDescriptor mappedWorkloadExecutionRoles[] = {
    {CaseSubjectRoleRef(0), "fabric", SubjectRoleCardinality::ExactlyOne,
     acceptedFabricSchemas, nullptr},
    {CaseSubjectRoleRef(1), "mapping", SubjectRoleCardinality::ExactlyOne,
     acceptedMappingSchemas, &verifyMappingBindsFabric},
};

const CaseSubjectRoleRef clockDomainTargetRoles[] = {CaseSubjectRoleRef(0)};

const ConditionPattern mappedWorkloadExecutionBaseConditions[] = {
    {EvaluationConditionKind::RequiredClockPeriod, clockDomainTargetRoles},
};

const std::array<EvaluationCaseSignatureDescriptor, 1> caseSignatures = {{
    {EvaluationCaseKind::MappedWorkloadExecution, "mapped_workload_execution",
     "One exact Fabric and Mapping executing one exact workload.",
     mappedWorkloadExecutionRoles, ArtifactRequirement::Required,
     acceptedWorkloadSchemas, ArtifactRequirement::Optional,
     acceptedRuntimeInputSchemas, &verifyRuntimeInputBindsWorkload,
     mappedWorkloadExecutionBaseConditions},
}};

} // namespace

SchemaVersion evaluationSchemaVersion() { return evaluationSchema; }

const ConditionPattern *
ConditionApplicability::findPattern(EvaluationConditionKind kind) const {
  for (const ConditionPattern &pattern : permittedPatterns)
    if (pattern.kind == kind)
      return &pattern;
  return nullptr;
}

const CaseSubjectRoleDescriptor *
EvaluationCaseSignatureDescriptor::findSubjectRole(
    CaseSubjectRoleRef role) const {
  for (const CaseSubjectRoleDescriptor &descriptor : subjectRoles)
    if (descriptor.role == role)
      return &descriptor;
  return nullptr;
}

ConditionApplicability
EvaluationCaseSignatureDescriptor::baseConditionApplicability() const {
  return ConditionApplicability{ConditionLocation::Base, spelling,
                                permittedBaseConditions};
}

llvm::Expected<EvaluationCaseSignatureRef>
EvaluationCaseSignatureRef::get(SchemaVersion schemaVersion,
                                EvaluationCaseKind caseKind) {
  if (schemaVersion != evaluationSchema)
    return evaluationError("unsupported evaluation schema version '" +
                           formatSchemaVersion(schemaVersion) + "'");
  return EvaluationCaseSignatureRef(schemaVersion, caseKind);
}

const EvaluationCaseSignatureDescriptor &
EvaluationCaseSignatureRef::descriptor() const {
  return caseSignatureDescriptor(caseKind_);
}

const EvaluationCaseSignatureDescriptor &
caseSignatureDescriptor(EvaluationCaseKind caseKind) {
  for (const EvaluationCaseSignatureDescriptor &descriptor : caseSignatures)
    if (descriptor.caseKind == caseKind)
      return descriptor;
  llvm_unreachable("unknown EvaluationCaseKind");
}

llvm::StringRef toString(EvaluationCaseKind caseKind) {
  return caseSignatureDescriptor(caseKind).spelling;
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
    std::sort(binding.subjects.begin(), binding.subjects.end(), identityLess);
    for (std::size_t subject = 1; subject < binding.subjects.size(); ++subject)
      if (binding.subjects[subject - 1] == binding.subjects[subject])
        return evaluationError("duplicate subject artifact for role " +
                               std::to_string(binding.role.ordinal()));
  }
  return EvaluationSubjectBindings(std::move(bindings));
}

llvm::ArrayRef<ArtifactIdentity>
EvaluationSubjectBindings::subjects(CaseSubjectRoleRef role) const {
  for (const CaseRoleBinding &binding : bindings_)
    if (binding.role == role)
      return binding.subjects;
  return {};
}

bool EvaluationSubjectBindings::isBoundSubject(
    CaseSubjectRoleRef role, const ArtifactIdentity &subject) const {
  llvm::ArrayRef<ArtifactIdentity> bound = subjects(role);
  return std::find(bound.begin(), bound.end(), subject) != bound.end();
}

llvm::Expected<CaseArtifactResolution>
CaseArtifactResolution::get(std::vector<Entry> entries) {
  std::sort(entries.begin(), entries.end(),
            [](const Entry &lhs, const Entry &rhs) {
              return identityLess(lhs.artifact, rhs.artifact);
            });
  for (std::size_t index = 0; index < entries.size(); ++index) {
    Entry &entry = entries[index];
    if (!entry.schema)
      return evaluationError("a resolved artifact must carry its owner's "
                             "schema descriptor");
    if (index != 0 && entries[index - 1].artifact == entry.artifact)
      return evaluationError("duplicate resolution for one artifact");
    std::sort(entry.dependencyClosure.begin(), entry.dependencyClosure.end(),
              identityLess);
    for (std::size_t member = 1; member < entry.dependencyClosure.size();
         ++member)
      if (entry.dependencyClosure[member - 1] ==
          entry.dependencyClosure[member])
        return evaluationError("duplicate artifact in one dependency closure");
  }
  return CaseArtifactResolution(std::move(entries));
}

const CaseArtifactResolution::Entry *
CaseArtifactResolution::find(const ArtifactIdentity &artifact) const {
  for (const Entry &entry : entries_)
    if (entry.artifact == artifact)
      return &entry;
  return nullptr;
}

bool CaseArtifactResolution::reaches(const Entry &entry,
                                     const ArtifactIdentity &dependency) {
  if (entry.artifact == dependency)
    return true;
  return std::find(entry.dependencyClosure.begin(),
                   entry.dependencyClosure.end(),
                   dependency) != entry.dependencyClosure.end();
}

llvm::Expected<EvaluationCase>
EvaluationCase::get(EvaluationCaseSignatureRef signature,
                    EvaluationSubjectBindings bindings,
                    std::optional<ArtifactIdentity> workload,
                    std::optional<ArtifactIdentity> runtimeInput,
                    llvm::ArrayRef<EvaluationCondition> baseConditions,
                    const CaseArtifactResolution &resolution) {
  const EvaluationCaseSignatureDescriptor &descriptor = signature.descriptor();

  for (const CaseRoleBinding &binding : bindings.roleBindings())
    if (!descriptor.findSubjectRole(binding.role))
      return evaluationError(
          "case subject role " + std::to_string(binding.role.ordinal()) +
          " is not a role of case signature '" + descriptor.spelling + "'");

  for (const CaseSubjectRoleDescriptor &role : descriptor.subjectRoles) {
    llvm::ArrayRef<ArtifactIdentity> subjects = bindings.subjects(role.role);
    if (subjects.empty())
      return evaluationError("case signature '" + descriptor.spelling +
                             "' requires a binding for subject role " +
                             std::to_string(role.role.ordinal()) + " ('" +
                             role.semanticRole + "')");
    if (role.cardinality == SubjectRoleCardinality::ExactlyOne &&
        subjects.size() != 1)
      return evaluationError("subject role '" + role.semanticRole +
                             "' requires exactly one subject");
    for (const ArtifactIdentity &subject : subjects) {
      const std::string reference =
          ("subject role '" + role.semanticRole + "'").str();
      if (llvm::Error error = validateAcceptedSchema(
              reference, role.acceptedSchemas, subject, resolution))
        return std::move(error);
    }
  }

  if (llvm::Error error = validateReferenceRequirement(
          descriptor, descriptor.workload, descriptor.acceptedWorkloadSchemas,
          "workload", workload, resolution))
    return std::move(error);
  if (llvm::Error error = validateReferenceRequirement(
          descriptor, descriptor.runtimeInput,
          descriptor.acceptedRuntimeInputSchemas, "runtime input", runtimeInput,
          resolution))
    return std::move(error);

  for (const CaseSubjectRoleDescriptor &role : descriptor.subjectRoles) {
    if (!role.verifyCrossRoleCompatibility)
      continue;
    for (const ArtifactIdentity &subject : bindings.subjects(role.role))
      if (llvm::Error error =
              role.verifyCrossRoleCompatibility(subject, bindings, resolution))
        return std::move(error);
  }
  if (descriptor.verifyWorkloadCompatibility)
    if (llvm::Error error = descriptor.verifyWorkloadCompatibility(
            bindings, workload, runtimeInput, resolution))
      return std::move(error);

  const CaseTargetContext context(descriptor, bindings, resolution);
  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(
          baseConditions, descriptor.baseConditionApplicability(), context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();

  return EvaluationCase(signature, std::move(bindings), std::move(workload),
                        std::move(runtimeInput),
                        std::move(*canonicalConditions));
}

} // namespace loom::evaluation
