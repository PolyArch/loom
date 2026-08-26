#ifndef LOOM_EVALUATION_CASE_H
#define LOOM_EVALUATION_CASE_H

#include "Evaluation/NumericValue.h"

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// The model-independent Evaluation case: one static typed case-signature
// registry, its role-labeled subject bindings, the shared scope algebra used
// by every query kind, and the closed condition union. These atoms are
// mutually defined by specification: a signature permits base-condition
// patterns, a condition carries scope targets, and a scope target names a
// case role and an exact bound anchor. They therefore share one header and
// have no second semantic authority.
//
// Every cross-artifact reference is exact: subject bindings, anchors, and
// targets use the Common ArtifactRootReference and
// EncodedArtifactLocalReference carriers. Each Artifact family owns its local
// kind ordinals, canonical payload bytes, typed decoder, and validation;
// Evaluation owns only case-role, anchor/closure, and pattern verification.

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::evaluation {

/// The exact Evaluation schema version that owns every registry ordinal in
/// this header. Case-signature, scope-form, and role ordinals are stable only
/// within it; a breaking change requires an incompatible schema version.
SchemaVersion evaluationSchemaVersion();

//===----------------------------------------------------------------------===//
// Condition kinds and locations
//===----------------------------------------------------------------------===//

enum class EvaluationConditionKind : std::uint8_t {
  ProcessCorner,
  SupplyVoltage,
  Temperature,
  RequiredClockPeriod,
  RelativeClockSchedule,
  ActivityBinding,
  Quantile,
};

/// The containing field determines a condition's location; location is never
/// copied into a condition payload.
enum class ConditionLocation : std::uint8_t {
  Base,
  MetricRequest,
  FindingRequest,
};

//===----------------------------------------------------------------------===//
// Case subject roles and exact target patterns
//===----------------------------------------------------------------------===//

/// A stable ordinal local to one exact case-signature version. The signature,
/// never a model descriptor, owns which ordinals exist and what they mean.
class CaseSubjectRoleRef {
public:
  explicit constexpr CaseSubjectRoleRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CaseSubjectRoleRef lhs,
                                   CaseSubjectRoleRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CaseSubjectRoleRef lhs,
                                   CaseSubjectRoleRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

/// A stable registry ordinal naming one case signature within the exact
/// Evaluation schema version. Case-signature owners register their descriptor
/// under their own pinned ordinal; it is not an Artifact reference, digest,
/// string name, or model-descriptor-local ordinal.
class EvaluationCaseKind {
public:
  explicit constexpr EvaluationCaseKind(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(EvaluationCaseKind lhs,
                                   EvaluationCaseKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(EvaluationCaseKind lhs,
                                   EvaluationCaseKind rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

/// The persistent registry reference is exactly (Evaluation schema version,
/// EvaluationCaseKind).
class EvaluationCaseSignatureRef {
public:
  static llvm::Expected<EvaluationCaseSignatureRef>
  get(SchemaVersion schemaVersion, EvaluationCaseKind caseKind);

  SchemaVersion schemaVersion() const { return schemaVersion_; }
  EvaluationCaseKind caseKind() const { return caseKind_; }
  /// The registered descriptor, or null when no owner registered this kind.
  const struct EvaluationCaseSignatureDescriptor *descriptor() const;

  friend bool operator==(EvaluationCaseSignatureRef lhs,
                         EvaluationCaseSignatureRef rhs) {
    return lhs.schemaVersion_ == rhs.schemaVersion_ &&
           lhs.caseKind_ == rhs.caseKind_;
  }
  friend bool operator!=(EvaluationCaseSignatureRef lhs,
                         EvaluationCaseSignatureRef rhs) {
    return !(lhs == rhs);
  }

private:
  EvaluationCaseSignatureRef(SchemaVersion schemaVersion,
                             EvaluationCaseKind caseKind)
      : schemaVersion_(schemaVersion), caseKind_(caseKind) {}

  SchemaVersion schemaVersion_;
  EvaluationCaseKind caseKind_;
};

/// Total canonical order over exact case-signature references.
bool evaluationCaseSignatureRefLess(EvaluationCaseSignatureRef lhs,
                                    EvaluationCaseSignatureRef rhs);

/// One exact accepted root type: any Artifact root of this exact schema.
struct ArtifactRootType {
  std::string schemaIdentity;
  SchemaVersion schemaVersion;

  ArtifactRootType(const ArtifactSchemaDescriptor &schema)
      : ArtifactRootType(schema.identity.str(), schema.version) {}

  static ArtifactRootType
  fromRootReference(const ArtifactRootReference &reference) {
    return ArtifactRootType(reference.schemaIdentity, reference.schemaVersion);
  }

  friend bool operator==(const ArtifactRootType &lhs,
                         const ArtifactRootType &rhs) {
    return lhs.schemaIdentity == rhs.schemaIdentity &&
           lhs.schemaVersion == rhs.schemaVersion;
  }
  friend bool operator!=(const ArtifactRootType &lhs,
                         const ArtifactRootType &rhs) {
    return !(lhs == rhs);
  }

private:
  ArtifactRootType(std::string schemaIdentity, SchemaVersion schemaVersion)
      : schemaIdentity(std::move(schemaIdentity)),
        schemaVersion(schemaVersion) {}
};

/// One exact accepted local type: one owner-local kind of one exact Artifact
/// family and schema version.
struct ArtifactLocalType {
  ArtifactLocalReferenceTypeDescriptor type;

  friend bool operator==(const ArtifactLocalType &lhs,
                         const ArtifactLocalType &rhs) {
    return lhs.type == rhs.type;
  }
  friend bool operator!=(const ArtifactLocalType &lhs,
                         const ArtifactLocalType &rhs) {
    return !(lhs == rhs);
  }
};

using SubjectReferenceType = std::variant<ArtifactRootType, ArtifactLocalType>;

/// One positional alternative of a target pattern: the exact case subject
/// role plus the exact root or local reference type accepted at this
/// position. Accepted target types are not independent per-role sets whose
/// accidental Cartesian product admits invalid relations.
struct SubjectTargetPattern {
  CaseSubjectRoleRef caseSubjectRole;
  SubjectReferenceType referenceType;

  friend bool operator==(const SubjectTargetPattern &lhs,
                         const SubjectTargetPattern &rhs) {
    return lhs.caseSubjectRole == rhs.caseSubjectRole &&
           lhs.referenceType == rhs.referenceType;
  }
  friend bool operator!=(const SubjectTargetPattern &lhs,
                         const SubjectTargetPattern &rhs) {
    return !(lhs == rhs);
  }
};

/// One complete positional target-pattern alternative pinned to the exact
/// case signature it serves.
struct OrderedTargetPattern {
  EvaluationCaseSignatureRef caseSignature;
  std::vector<SubjectTargetPattern> targets;

  friend bool operator==(const OrderedTargetPattern &lhs,
                         const OrderedTargetPattern &rhs) {
    return lhs.caseSignature == rhs.caseSignature && lhs.targets == rhs.targets;
  }
  friend bool operator!=(const OrderedTargetPattern &lhs,
                         const OrderedTargetPattern &rhs) {
    return !(lhs == rhs);
  }
};

/// One condition pattern an owner permits: the exact kind and one complete
/// ordered target pattern. Semantic applicability has three nonoverlapping
/// owners: the exact EvaluationCaseSignature owns Base patterns, a Metric or
/// Finding descriptor owns request-specific patterns, and an
/// EvaluationModelDescriptor declares which already-permitted exact patterns
/// it consumes, requires, or proves irrelevant.
struct ConditionApplicabilityPattern {
  EvaluationConditionKind kind;
  OrderedTargetPattern targets;

  friend bool operator==(const ConditionApplicabilityPattern &lhs,
                         const ConditionApplicabilityPattern &rhs) {
    return lhs.kind == rhs.kind && lhs.targets == rhs.targets;
  }
  friend bool operator!=(const ConditionApplicabilityPattern &lhs,
                         const ConditionApplicabilityPattern &rhs) {
    return !(lhs == rhs);
  }
};

/// Canonical pattern ordering: exact case-signature reference, arity, then
/// each positional (case role, root/local discriminant, owner schema,
/// owner-local kind when present) key. Duplicate patterns are invalid.
bool orderedTargetPatternLess(const OrderedTargetPattern &lhs,
                              const OrderedTargetPattern &rhs);
bool conditionApplicabilityPatternLess(
    const ConditionApplicabilityPattern &lhs,
    const ConditionApplicabilityPattern &rhs);

struct EvaluationCaseSignatureDescriptor;

/// Validates one descriptor-owned pattern collection: canonical order, no
/// duplicates, registered exact case signatures, and signature-local roles.
/// The optional descriptor admits its own exact reference during registration.
llvm::Error validateOrderedTargetPatternSet(
    llvm::StringRef owner, llvm::ArrayRef<OrderedTargetPattern> patterns,
    const EvaluationCaseSignatureDescriptor *selfRegisteringSignature =
        nullptr);

//===----------------------------------------------------------------------===//
// Subject bindings and resolved case artifacts
//===----------------------------------------------------------------------===//

struct CaseRoleBinding {
  CaseSubjectRoleRef role;
  std::vector<ArtifactRootReference> subjects;

  friend bool operator==(const CaseRoleBinding &lhs,
                         const CaseRoleBinding &rhs) {
    return lhs.role == rhs.role && lhs.subjects == rhs.subjects;
  }
  friend bool operator!=(const CaseRoleBinding &lhs,
                         const CaseRoleBinding &rhs) {
    return !(lhs == rhs);
  }
};

/// A total table from case-signature role to a canonical collection of exact
/// bound Artifact roots. Collections contain no duplicates and are ordered by
/// complete root-reference canonical key; authoring order has no meaning.
/// Totality relative to the exact signature is enforced by EvaluationCase.
class EvaluationSubjectBindings {
public:
  static llvm::Expected<EvaluationSubjectBindings>
  get(std::vector<CaseRoleBinding> bindings);

  llvm::ArrayRef<CaseRoleBinding> roleBindings() const { return bindings_; }
  llvm::ArrayRef<ArtifactRootReference> subjects(CaseSubjectRoleRef role) const;
  bool isBoundSubject(CaseSubjectRoleRef role,
                      const ArtifactRootReference &subject) const;

  friend bool operator==(const EvaluationSubjectBindings &lhs,
                         const EvaluationSubjectBindings &rhs) {
    return lhs.bindings_ == rhs.bindings_;
  }
  friend bool operator!=(const EvaluationSubjectBindings &lhs,
                         const EvaluationSubjectBindings &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit EvaluationSubjectBindings(std::vector<CaseRoleBinding> bindings)
      : bindings_(std::move(bindings)) {}

  std::vector<CaseRoleBinding> bindings_;
};

/// The facts about exact case Artifacts that only an Artifact store can
/// supply: each Artifact's exact semantic dependency closure. Evaluation
/// resolves no Artifact and persists no resolution; an unresolved Artifact is
/// never silently accepted. The family-owned importer view a local-reference
/// validator needs never enters this structure: each referenced Artifact
/// family resolves its own typed view through its own owner boundary.
class CaseArtifactResolution {
public:
  struct Entry {
    ArtifactRootReference artifact;
    std::vector<ArtifactRootReference> dependencyClosure;

    friend bool operator==(const Entry &lhs, const Entry &rhs) {
      return lhs.artifact == rhs.artifact &&
             lhs.dependencyClosure == rhs.dependencyClosure;
    }
    friend bool operator!=(const Entry &lhs, const Entry &rhs) {
      return !(lhs == rhs);
    }
  };

  static llvm::Expected<CaseArtifactResolution> get(std::vector<Entry> entries);

  const Entry *find(const ArtifactRootReference &artifact) const;
  llvm::ArrayRef<Entry> entries() const { return entries_; }
  /// True when the dependency is the Artifact itself or occurs in its exact
  /// semantic dependency closure.
  static bool reaches(const Entry &entry,
                      const ArtifactRootReference &dependency);

  friend bool operator==(const CaseArtifactResolution &lhs,
                         const CaseArtifactResolution &rhs) {
    return lhs.entries_ == rhs.entries_;
  }
  friend bool operator!=(const CaseArtifactResolution &lhs,
                         const CaseArtifactResolution &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit CaseArtifactResolution(std::vector<Entry> entries)
      : entries_(std::move(entries)) {}

  std::vector<Entry> entries_;
};

/// The exact target of one scope position: an Artifact root or one
/// family-owned local object in heterogeneous framing.
using SubjectTarget =
    std::variant<ArtifactRootReference, EncodedArtifactLocalReference>;

struct SubjectTargetRef {
  CaseSubjectRoleRef caseSubjectRole;
  /// An exact member of the selected case subject-role binding.
  ArtifactRootReference anchorSubjectArtifact;
  SubjectTarget target;

  const ArtifactRootReference &targetArtifact() const;

  friend bool operator==(const SubjectTargetRef &lhs,
                         const SubjectTargetRef &rhs) {
    return lhs.caseSubjectRole == rhs.caseSubjectRole &&
           lhs.anchorSubjectArtifact == rhs.anchorSubjectArtifact &&
           lhs.target == rhs.target;
  }
  friend bool operator!=(const SubjectTargetRef &lhs,
                         const SubjectTargetRef &rhs) {
    return !(lhs == rhs);
  }
};

/// The resolved reference type of one exact target.
SubjectReferenceType subjectReferenceTypeOf(const SubjectTarget &target);

//===----------------------------------------------------------------------===//
// Case signature registry
//===----------------------------------------------------------------------===//

class EvaluationCase;

enum class ArtifactRequirement : std::uint8_t { Forbidden, Optional, Required };

enum class SubjectRoleCardinality : std::uint8_t { ExactlyOne, OneOrMore };

struct AbsentReferenceCycle {};
struct AbstractCaseCycle {};

struct ExactSubjectCycle {
  SubjectReferenceType acceptedReferenceType;
  llvm::Expected<SubjectTargetRef> (*resolve)(
      const EvaluationCase &evaluationCase,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
};

using WholeCaseCycleBasis =
    std::variant<AbsentReferenceCycle, AbstractCaseCycle, ExactSubjectCycle>;
using ReferenceCycleBasis = std::variant<AbstractCaseCycle, SubjectTargetRef>;

struct CaseSubjectRoleDescriptor {
  CaseSubjectRoleRef role;
  llvm::StringRef semanticRole;
  SubjectRoleCardinality cardinality;
  /// Owner-exported Artifact schema descriptors this role accepts. Evaluation
  /// never authors a family schema identity; it references the descriptor the
  /// owning family publishes.
  llvm::ArrayRef<const ArtifactSchemaDescriptor *> acceptedSchemas;
  /// This role's compatibility with the other bound roles, owned by the
  /// signature. Null when the role imposes no cross-role relation.
  llvm::Error (*verifyCrossRoleCompatibility)(
      const ArtifactRootReference &subject,
      const EvaluationCase &evaluationCase,
      const EvaluationSubjectBindings &bindings,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
};

struct EvaluationCaseSignatureDescriptor {
  EvaluationCaseKind caseKind;
  llvm::StringRef spelling;
  llvm::StringRef semanticDefinition;
  llvm::ArrayRef<CaseSubjectRoleDescriptor> subjectRoles;
  ArtifactRequirement workload;
  llvm::ArrayRef<const ArtifactSchemaDescriptor *> acceptedWorkloadSchemas;
  ArtifactRequirement runtimeInput;
  llvm::ArrayRef<const ArtifactSchemaDescriptor *> acceptedRuntimeInputSchemas;
  /// Compatibility between the two orthogonal exact references and the bound
  /// subjects, owned by the signature. Null when the signature imposes no
  /// relation beyond the accepted schemas.
  llvm::Error (*verifyWorkloadCompatibility)(
      const EvaluationCase &evaluationCase,
      const EvaluationSubjectBindings &bindings,
      const std::optional<ArtifactRootReference> &workload,
      const std::optional<ArtifactRootReference> &runtimeInput,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);
  /// Whether this exact case signature defines one semantic reference-cycle
  /// basis for whole-case cycle-count and clock-period queries.
  WholeCaseCycleBasis wholeCaseCycleBasis;
  /// The complete exact Base-condition patterns this signature permits. Every
  /// pattern's case signature must be this signature itself.
  llvm::ArrayRef<ConditionApplicabilityPattern> permittedBaseConditions;

  /// The exact catalog version that owns this immutable descriptor view.
  /// Compatible older views are mechanically derived by the registry.
  SchemaVersion registryVersion = evaluationSchemaVersion();

  const CaseSubjectRoleDescriptor *
  findSubjectRole(CaseSubjectRoleRef role) const;
};

/// Statically registers one case-signature descriptor under its pinned kind
/// ordinal. The descriptor and everything it references must have static
/// storage duration. Re-registering the same descriptor is a no-op; a kind
/// ordinal or spelling conflict, malformed roles, or a noncanonical
/// Base-pattern set is an error.
llvm::Error registerEvaluationCaseSignature(
    const EvaluationCaseSignatureDescriptor &descriptor);

/// The registered descriptor for one case kind, or null.
const EvaluationCaseSignatureDescriptor *
findEvaluationCaseSignature(EvaluationCaseKind caseKind);
const EvaluationCaseSignatureDescriptor *
findEvaluationCaseSignature(SchemaVersion schemaVersion,
                            EvaluationCaseKind caseKind);

/// Resolves the exact signature-owned whole-case cycle basis. Absent basis,
/// resolver failure, a foreign or noncanonical target, or a target of the
/// wrong declared reference type is an error.
llvm::Expected<ReferenceCycleBasis>
resolveReferenceCycleBasis(const EvaluationCase &evaluationCase,
                           const CaseArtifactResolution &resolution,
                           const ArtifactStore &artifactStore,
                           const BlobStore &blobStore);

//===----------------------------------------------------------------------===//
// Evaluation scope
//===----------------------------------------------------------------------===//

/// A stable ordinal local to one query-kind descriptor in the exact
/// Evaluation schema version. The containing MetricKind or FindingKind always
/// resolves it, so it is neither a global relation registry nor a free
/// string.
class ScopeFormRef {
public:
  explicit constexpr ScopeFormRef(std::uint32_t ordinal) : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(ScopeFormRef lhs, ScopeFormRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(ScopeFormRef lhs, ScopeFormRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

/// A stable ordinal within one scope form. Role order is significant: the
/// target tuple uses descriptor role order and is never sorted.
class ScopeRoleRef {
public:
  explicit constexpr ScopeRoleRef(std::uint32_t ordinal) : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(ScopeRoleRef lhs, ScopeRoleRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(ScopeRoleRef lhs, ScopeRoleRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

struct ScopeRoleDescriptor {
  ScopeRoleRef role;
  llvm::StringRef semanticRole;
};

struct WholeExactCaseScope {};

struct ExactTargetPatternsScope {
  llvm::ArrayRef<OrderedTargetPattern> patterns;
};

using ScopeApplicability =
    std::variant<WholeExactCaseScope, ExactTargetPatternsScope>;

enum class ReferenceCycleRequirement : std::uint8_t {
  NotRequired,
  ExactCaseUniqueReferenceCycle,
};

struct ScopeFormDescriptor {
  ScopeFormRef form;
  llvm::StringRef semanticDefinition;
  /// Ordered, nonrepeating role tuple: each role ordinal is its position and
  /// the semantic roles are distinct.
  llvm::ArrayRef<ScopeRoleDescriptor> roles;
  /// A zero-role WholeExactCase form is intrinsic to the Request's exact case.
  /// A targetful form instead owns a canonical nonempty set of complete
  /// positional alternatives pinned to exact case signatures.
  ScopeApplicability applicability;
  /// Relation-specific verification owned by this query form, such as
  /// requiring two distinct endpoints. Null when the accepted patterns are
  /// the whole contract.
  llvm::Error (*verifyRelation)(llvm::ArrayRef<SubjectTargetRef> targets);
  ReferenceCycleRequirement referenceCycleRequirement =
      ReferenceCycleRequirement::NotRequired;
};

const ScopeFormDescriptor *
findScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms, ScopeFormRef form);

/// Validates the complete immutable form table independently of one query:
/// contiguous ordinals, exact whole-case shape, ordered unique roles, and
/// canonical nonempty exact target patterns with valid signature-local roles.
llvm::Error
validateScopeFormDescriptors(llvm::ArrayRef<ScopeFormDescriptor> forms);

/// The one closed scope algebra shared by every MetricKind and FindingKind.
/// The target tuple has exactly the descriptor arity in descriptor role order
/// and is never sorted as an unordered set.
struct EvaluationScope {
  ScopeFormRef form;
  std::vector<SubjectTargetRef> targets;

  friend bool operator==(const EvaluationScope &lhs,
                         const EvaluationScope &rhs) {
    return lhs.form == rhs.form && lhs.targets == rhs.targets;
  }
  friend bool operator!=(const EvaluationScope &lhs,
                         const EvaluationScope &rhs) {
    return !(lhs == rhs);
  }
};

/// The exact ordered target pattern of one validated target tuple under one
/// exact case signature.
OrderedTargetPattern
deriveOrderedTargetPattern(llvm::ArrayRef<SubjectTargetRef> targets,
                           EvaluationCaseSignatureRef caseSignature);

/// The canonical scope key: form ordinal, exact arity framing, and each fully
/// framed target in descriptor role order, including the case role, anchor
/// root reference, target root reference, owner-local type descriptor, and
/// family-owned canonical local payload.
std::vector<std::uint8_t> canonicalScopeKey(const EvaluationScope &scope);

/// Query-descriptor-relative validation: the form exists among the query
/// kind's own forms, its role tuple is ordered and nonrepeating, the target
/// tuple has exactly the descriptor arity, and the form's pattern set is
/// canonical.
llvm::Error
validateEvaluationScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms,
                            const EvaluationScope &scope);

//===----------------------------------------------------------------------===//
// Case-relative target validation
//===----------------------------------------------------------------------===//

/// The case-relative facts a target validator needs: the exact case
/// signature, its bound subjects, the dependency-closure resolution, and the
/// explicit immutable store used to import exact target Artifacts.
class CaseTargetContext {
public:
  CaseTargetContext(const EvaluationCaseSignatureDescriptor &signature,
                    EvaluationCaseSignatureRef signatureRef,
                    const EvaluationSubjectBindings &bindings,
                    const CaseArtifactResolution &resolution,
                    const ArtifactStore &artifactStore)
      : signature_(&signature), signatureRef_(signatureRef),
        bindings_(&bindings), resolution_(&resolution),
        artifactStore_(&artifactStore) {}

  const EvaluationCaseSignatureDescriptor &signature() const {
    return *signature_;
  }
  EvaluationCaseSignatureRef signatureRef() const { return signatureRef_; }
  const EvaluationSubjectBindings &bindings() const { return *bindings_; }
  const CaseArtifactResolution &resolution() const { return *resolution_; }
  const ArtifactStore &artifactStore() const { return *artifactStore_; }

private:
  const EvaluationCaseSignatureDescriptor *signature_;
  EvaluationCaseSignatureRef signatureRef_;
  const EvaluationSubjectBindings *bindings_;
  const CaseArtifactResolution *resolution_;
  const ArtifactStore *artifactStore_;
};

/// Rejects a foreign case role, an anchor that is not an exact member of that
/// role's binding, an unresolved Artifact, a target outside the anchor's
/// exact dependency closure, and a local target whose owner codec or owner
/// validator rejects it.
llvm::Error validateSubjectTargetRef(const SubjectTargetRef &target,
                                     const CaseTargetContext &context);

/// Case-relative scope validation: every target is valid, the derived exact
/// ordered target pattern matches one descriptor-owned pattern of the
/// selected form, and the form's own relation verification holds.
llvm::Error
validateEvaluationScopeCase(const EvaluationScope &scope,
                            llvm::ArrayRef<ScopeFormDescriptor> forms,
                            const CaseTargetContext &context);

//===----------------------------------------------------------------------===//
// Evaluation conditions
//===----------------------------------------------------------------------===//

struct ProcessCornerCondition {
  SubjectTargetRef target;
  /// The exact typed TechnologyCornerRef. Persistent encoders obtain its
  /// heterogeneous eight-byte representation only through the
  /// ImplementationPlatform owner codec.
  platform::TechnologyCornerRef corner;

  friend bool operator==(const ProcessCornerCondition &lhs,
                         const ProcessCornerCondition &rhs) {
    return lhs.target == rhs.target && lhs.corner == rhs.corner;
  }
};

struct SupplyVoltageCondition {
  SubjectTargetRef powerDomain;
  DecimalValue volts;

  friend bool operator==(const SupplyVoltageCondition &lhs,
                         const SupplyVoltageCondition &rhs) {
    return lhs.powerDomain == rhs.powerDomain && lhs.volts == rhs.volts;
  }
};

struct TemperatureCondition {
  SubjectTargetRef thermalDomainOrRoot;
  DecimalValue kelvin;

  friend bool operator==(const TemperatureCondition &lhs,
                         const TemperatureCondition &rhs) {
    return lhs.thermalDomainOrRoot == rhs.thermalDomainOrRoot &&
           lhs.kelvin == rhs.kelvin;
  }
};

struct RequiredClockPeriodCondition {
  SubjectTargetRef clockDomain;
  DecimalValue seconds;

  friend bool operator==(const RequiredClockPeriodCondition &lhs,
                         const RequiredClockPeriodCondition &rhs) {
    return lhs.clockDomain == rhs.clockDomain && lhs.seconds == rhs.seconds;
  }
};

/// Denotes dependent clock edges at phase + k * period_ratio in reference
/// cycles. Absolute clock targets use RequiredClockPeriodCondition instead.
struct RelativeClockScheduleCondition {
  SubjectTargetRef referenceClock;
  SubjectTargetRef dependentClock;
  ExactRatio dependentPeriodPerReferencePeriod;
  ExactRatio dependentPhaseInReferenceCycles;

  friend bool operator==(const RelativeClockScheduleCondition &lhs,
                         const RelativeClockScheduleCondition &rhs) {
    return lhs.referenceClock == rhs.referenceClock &&
           lhs.dependentClock == rhs.dependentClock &&
           lhs.dependentPeriodPerReferencePeriod ==
               rhs.dependentPeriodPerReferencePeriod &&
           lhs.dependentPhaseInReferenceCycles ==
               rhs.dependentPhaseInReferenceCycles;
  }
};

/// One exact typed activity summary owned by a SimulationExecution. That
/// family owns canonical summary order, ordinal range, source-basis coverage,
/// and Request-lineage validation.
struct ExecutionActivitySource {
  ArtifactRootReference simulationExecution;
  std::uint64_t activitySummaryOrdinal;

  friend bool operator==(const ExecutionActivitySource &lhs,
                         const ExecutionActivitySource &rhs) {
    return lhs.simulationExecution == rhs.simulationExecution &&
           lhs.activitySummaryOrdinal == rhs.activitySummaryOrdinal;
  }
};

/// A small uniform vectorless assumption, not an Activity Artifact or an
/// arbitrary per-signal map.
struct ExplicitAssumptionSource {
  SubjectTargetRef clockDomain;
  ExactRatio staticProbability;
  ExactRatio transitionsPerClock;

  friend bool operator==(const ExplicitAssumptionSource &lhs,
                         const ExplicitAssumptionSource &rhs) {
    return lhs.clockDomain == rhs.clockDomain &&
           lhs.staticProbability == rhs.staticProbability &&
           lhs.transitionsPerClock == rhs.transitionsPerClock;
  }
};

using ActivitySource =
    std::variant<ExecutionActivitySource, ExplicitAssumptionSource>;

struct ActivityBindingCondition {
  SubjectTargetRef target;
  ActivitySource source;

  friend bool operator==(const ActivityBindingCondition &lhs,
                         const ActivityBindingCondition &rhs) {
    return lhs.target == rhs.target && lhs.source == rhs.source;
  }
};

struct QuantileCondition {
  ExactRatio probability;

  friend bool operator==(QuantileCondition lhs, QuantileCondition rhs) {
    return lhs.probability == rhs.probability;
  }
};

/// The closed tagged union used by base and request-specific conditions. The
/// variant order is exactly the registry kind order, so the kind is derived
/// rather than stored twice.
using EvaluationConditionPayload =
    std::variant<ProcessCornerCondition, SupplyVoltageCondition,
                 TemperatureCondition, RequiredClockPeriodCondition,
                 RelativeClockScheduleCondition, ActivityBindingCondition,
                 QuantileCondition>;

struct EvaluationCondition {
  EvaluationConditionPayload payload;

  EvaluationConditionKind kind() const;

  friend bool operator==(const EvaluationCondition &lhs,
                         const EvaluationCondition &rhs) {
    return lhs.payload == rhs.payload;
  }
  friend bool operator!=(const EvaluationCondition &lhs,
                         const EvaluationCondition &rhs) {
    return !(lhs == rhs);
  }
};

struct EvaluationConditionDescriptor {
  EvaluationConditionKind kind;
  llvm::StringRef spelling;
  llvm::StringRef semanticDefinition;
  std::uint8_t allowedLocations;

  bool permitsLocation(ConditionLocation location) const;
};

const EvaluationConditionDescriptor &
conditionDescriptor(EvaluationConditionKind kind);

/// Validates one condition owner's complete exact pattern set and delegates
/// every same-kind target collection to validateOrderedTargetPatternSet.
llvm::Error validateConditionApplicabilityPatternSet(
    llvm::StringRef owner,
    llvm::ArrayRef<ConditionApplicabilityPattern> patterns,
    ConditionLocation location,
    const EvaluationCaseSignatureDescriptor *selfRegisteringSignature =
        nullptr);

llvm::StringRef toString(EvaluationConditionKind kind);
llvm::Expected<EvaluationConditionKind>
parseEvaluationConditionKind(llvm::StringRef spelling);
llvm::StringRef toString(ConditionLocation location);

/// The kind-owned typed projection from a validated payload to its exact
/// ordered tuple of SubjectTargetRef values, in semantic payload-field order.
/// It is derived and never serialized as another list.
std::vector<const SubjectTargetRef *>
conditionOrderedTargets(const EvaluationCondition &condition);

/// The exact condition applicability pattern of one validated condition under
/// one exact case signature, derived from the ordered targets and their
/// resolved reference types.
ConditionApplicabilityPattern
deriveConditionApplicabilityPattern(const EvaluationCondition &condition,
                                    EvaluationCaseSignatureRef caseSignature);

/// The kind-owned assignment-key projection. Two values of one kind with the
/// same assignment key but different payloads are a conflict.
std::vector<std::uint8_t>
conditionAssignmentKey(const EvaluationCondition &condition);

/// The complete canonical payload key, used for canonical ordering and exact
/// duplicate detection.
std::vector<std::uint8_t>
conditionPayloadKey(const EvaluationCondition &condition);

/// Validates each condition's location, typed payload, case-bound targets
/// through every target Artifact owner's codec and validator, and exact
/// pattern applicability against the permitting owner's complete patterns,
/// then returns the canonical set ordered by (kind, assignment key, complete
/// payload key). An exact duplicate and an assignment conflict are both
/// invalid; there is no last-wins behavior or override layer.
llvm::Expected<std::vector<EvaluationCondition>>
canonicalizeEvaluationConditions(
    llvm::ArrayRef<EvaluationCondition> conditions, ConditionLocation location,
    llvm::StringRef permittingOwner,
    llvm::ArrayRef<ConditionApplicabilityPattern> permittedPatterns,
    const CaseTargetContext &context);

//===----------------------------------------------------------------------===//
// Evaluation case
//===----------------------------------------------------------------------===//

/// The model-independent persistent case: one exact case signature, its total
/// role-labeled subject bindings, two orthogonal exact references, and the
/// canonical base conditions. It contains no model implementation fact and no
/// derived case key.
class EvaluationCase {
public:
  static llvm::Expected<EvaluationCase>
  get(EvaluationCaseSignatureRef signature, EvaluationSubjectBindings bindings,
      std::optional<ArtifactRootReference> workload,
      std::optional<ArtifactRootReference> runtimeInput,
      llvm::ArrayRef<EvaluationCondition> baseConditions,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);

  EvaluationCaseSignatureRef signature() const { return signature_; }
  const EvaluationSubjectBindings &subjectBindings() const { return bindings_; }
  const std::optional<ArtifactRootReference> &workload() const {
    return workload_;
  }
  const std::optional<ArtifactRootReference> &runtimeInput() const {
    return runtimeInput_;
  }
  llvm::ArrayRef<EvaluationCondition> baseConditions() const {
    return baseConditions_;
  }

  CaseTargetContext targetContext(const CaseArtifactResolution &resolution,
                                  const ArtifactStore &artifactStore) const;

private:
  EvaluationCase(EvaluationCaseSignatureRef signature,
                 EvaluationSubjectBindings bindings,
                 std::optional<ArtifactRootReference> workload,
                 std::optional<ArtifactRootReference> runtimeInput,
                 std::vector<EvaluationCondition> baseConditions)
      : signature_(signature), bindings_(std::move(bindings)),
        workload_(std::move(workload)), runtimeInput_(std::move(runtimeInput)),
        baseConditions_(std::move(baseConditions)) {}

  EvaluationCaseSignatureRef signature_;
  EvaluationSubjectBindings bindings_;
  std::optional<ArtifactRootReference> workload_;
  std::optional<ArtifactRootReference> runtimeInput_;
  std::vector<EvaluationCondition> baseConditions_;
};

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_CASE_H
