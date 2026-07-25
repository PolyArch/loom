#ifndef LOOM_EVALUATION_CASE_H
#define LOOM_EVALUATION_CASE_H

#include "Evaluation/NumericValue.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

// The model-independent Evaluation case: one static typed case-signature
// registry, its role-labeled subject bindings, the shared scope algebra used by
// every query kind, and the closed condition union. These atoms are mutually
// defined by specification: a signature permits base-condition kinds and
// targets, a condition carries scope targets, and a scope target names a case
// role and an exact bound anchor. They therefore share one header and have no
// second semantic authority.

namespace loom::evaluation {

/// The exact Evaluation schema version that owns every registry ordinal in this
/// header. Case-signature, scope-form, and role ordinals are stable only within
/// it; a breaking change requires an incompatible schema version.
SchemaVersion evaluationSchemaVersion();

//===----------------------------------------------------------------------===//
// Condition kinds, locations, and applicability
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

/// One condition pattern an owner permits: the exact kind and the exact case
/// subject roles its targets may name. A targetless kind permits no role.
struct ConditionPattern {
  EvaluationConditionKind kind;
  llvm::ArrayRef<CaseSubjectRoleRef> permittedTargetRoles;
};

/// The exact owner that permits condition patterns at one location: a case
/// signature owns Base applicability, and a Metric or Finding descriptor owns
/// request-specific applicability. A model descriptor separately declares what
/// it consumes, requires, or proves invariant; it never widens either owner or
/// redefines a payload.
struct ConditionApplicability {
  ConditionLocation location;
  llvm::StringRef permittingOwner;
  llvm::ArrayRef<ConditionPattern> permittedPatterns;

  const ConditionPattern *findPattern(EvaluationConditionKind kind) const;
};

//===----------------------------------------------------------------------===//
// Subject bindings and resolved case artifacts
//===----------------------------------------------------------------------===//

struct CaseRoleBinding {
  CaseSubjectRoleRef role;
  std::vector<ArtifactIdentity> subjects;

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
/// bound Artifacts. Collections contain no duplicates and are ordered by
/// complete reference key; authoring order has no meaning. Totality relative
/// to the exact signature is enforced by EvaluationCase.
class EvaluationSubjectBindings {
public:
  static llvm::Expected<EvaluationSubjectBindings>
  get(std::vector<CaseRoleBinding> bindings);

  llvm::ArrayRef<CaseRoleBinding> roleBindings() const { return bindings_; }
  llvm::ArrayRef<ArtifactIdentity> subjects(CaseSubjectRoleRef role) const;
  bool isBoundSubject(CaseSubjectRoleRef role,
                      const ArtifactIdentity &subject) const;

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

/// The facts about exact case Artifacts that only an Artifact store can supply:
/// each Artifact's owner-exported schema descriptor and its exact semantic
/// dependency closure. Evaluation resolves no Artifact and persists no
/// resolution; an unresolved Artifact is never silently accepted.
class CaseArtifactResolution {
public:
  struct Entry {
    ArtifactIdentity artifact;
    const ArtifactSchemaDescriptor *schema;
    std::vector<ArtifactIdentity> dependencyClosure;
  };

  static llvm::Expected<CaseArtifactResolution> get(std::vector<Entry> entries);

  const Entry *find(const ArtifactIdentity &artifact) const;
  /// True when the dependency is the Artifact itself or occurs in its exact
  /// semantic dependency closure.
  static bool reaches(const Entry &entry, const ArtifactIdentity &dependency);

private:
  explicit CaseArtifactResolution(std::vector<Entry> entries)
      : entries_(std::move(entries)) {}

  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// Case signature registry
//===----------------------------------------------------------------------===//

enum class EvaluationCaseKind : std::uint8_t {
  MappedWorkloadExecution,
};

enum class ArtifactRequirement : std::uint8_t { Forbidden, Optional, Required };

enum class SubjectRoleCardinality : std::uint8_t { ExactlyOne, OneOrMore };

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
      const ArtifactIdentity &subject,
      const EvaluationSubjectBindings &bindings,
      const CaseArtifactResolution &resolution);
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
      const EvaluationSubjectBindings &bindings,
      const std::optional<ArtifactIdentity> &workload,
      const std::optional<ArtifactIdentity> &runtimeInput,
      const CaseArtifactResolution &resolution);
  llvm::ArrayRef<ConditionPattern> permittedBaseConditions;

  const CaseSubjectRoleDescriptor *
  findSubjectRole(CaseSubjectRoleRef role) const;
  ConditionApplicability baseConditionApplicability() const;
};

/// The persistent registry reference is exactly
/// (Evaluation schema version, EvaluationCaseKind). It is not an Artifact
/// reference, digest, string name, or model-descriptor-local ordinal.
class EvaluationCaseSignatureRef {
public:
  static llvm::Expected<EvaluationCaseSignatureRef>
  get(SchemaVersion schemaVersion, EvaluationCaseKind caseKind);

  SchemaVersion schemaVersion() const { return schemaVersion_; }
  EvaluationCaseKind caseKind() const { return caseKind_; }
  const EvaluationCaseSignatureDescriptor &descriptor() const;

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

const EvaluationCaseSignatureDescriptor &
caseSignatureDescriptor(EvaluationCaseKind caseKind);

llvm::StringRef toString(EvaluationCaseKind caseKind);

//===----------------------------------------------------------------------===//
// Family-owned local targets
//===----------------------------------------------------------------------===//

/// The discriminator of one local object kind inside an Artifact family. Its
/// values are owned by that family, not by Evaluation.
class LocalTargetKind {
public:
  explicit constexpr LocalTargetKind(std::uint32_t value) : value_(value) {}

  constexpr std::uint32_t value() const { return value_; }

  friend constexpr bool operator==(LocalTargetKind lhs, LocalTargetKind rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(LocalTargetKind lhs, LocalTargetKind rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t value_;
};

/// The canonical payload of one family-owned local reference. The owning family
/// composes it from its own typed entity and structural references and remains
/// the sole authority on its meaning and validity. Evaluation frames the
/// complete payload in canonical keys and text and never interprets it, so
/// there is no Evaluation-owned entity shape, generic path, or property bag.
class LocalTargetPayload {
public:
  /// Family-facing composition from the family's own typed references. Each
  /// reference contributes its ordinal in declaration order, so a single-entity
  /// reference and a structural reference stay distinguishable.
  template <typename... EntityIds>
  static LocalTargetPayload ofEntities(EntityIds... entities) {
    std::vector<std::uint8_t> bytes;
    (appendEntityOrdinal(bytes, entities.value()), ...);
    return LocalTargetPayload(std::move(bytes));
  }

  /// Family-facing decode of bytes this family already framed. The family's
  /// validator remains the authority on whether they are well formed.
  static LocalTargetPayload
  fromCanonicalBytes(llvm::ArrayRef<std::uint8_t> bytes) {
    return LocalTargetPayload(
        std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
  }

  llvm::ArrayRef<std::uint8_t> bytes() const { return bytes_; }

  friend bool operator==(const LocalTargetPayload &lhs,
                         const LocalTargetPayload &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const LocalTargetPayload &lhs,
                         const LocalTargetPayload &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit LocalTargetPayload(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  static void appendEntityOrdinal(std::vector<std::uint8_t> &bytes,
                                  std::uint64_t ordinal) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(ordinal >> shift));
    bytes.push_back(static_cast<std::uint8_t>(ordinal));
  }

  std::vector<std::uint8_t> bytes_;
};

/// One Artifact family's authority over the local objects an Evaluation target
/// may name inside its artifacts. The family owns the closed local kind set,
/// their canonical spellings, and the validity of their payloads. Evaluation
/// imports this descriptor unchanged and defines no global Evaluation entity
/// catalog.
struct LocalTargetFamilyDescriptor {
  const ArtifactSchemaDescriptor *artifactSchema;
  llvm::StringRef (*localKindSpelling)(LocalTargetKind kind);
  llvm::Error (*validateLocalTarget)(LocalTargetKind kind,
                                     const LocalTargetPayload &payload);
};

/// An exact reference to one family-owned local object. Every instance is
/// family-validated, so an unvalidated or caller-invented payload cannot reach
/// a scope, a condition, or a canonical key.
class LocalTargetRef {
public:
  static llvm::Expected<LocalTargetRef>
  get(const LocalTargetFamilyDescriptor &family, LocalTargetKind kind,
      ArtifactIdentity artifact, LocalTargetPayload payload);

  const LocalTargetFamilyDescriptor &family() const { return *family_; }
  LocalTargetKind localKind() const { return kind_; }
  const ArtifactIdentity &artifact() const { return artifact_; }
  const LocalTargetPayload &payload() const { return payload_; }

private:
  LocalTargetRef(const LocalTargetFamilyDescriptor &family,
                 LocalTargetKind kind, ArtifactIdentity artifact,
                 LocalTargetPayload payload)
      : family_(&family), kind_(kind), artifact_(std::move(artifact)),
        payload_(std::move(payload)) {}

  const LocalTargetFamilyDescriptor *family_;
  LocalTargetKind kind_;
  ArtifactIdentity artifact_;
  LocalTargetPayload payload_;
};

/// Family identity is the family's own Artifact schema identity and version, so
/// two descriptors for one family compare equal.
bool operator==(const LocalTargetRef &lhs, const LocalTargetRef &rhs);
inline bool operator!=(const LocalTargetRef &lhs, const LocalTargetRef &rhs) {
  return !(lhs == rhs);
}

//===----------------------------------------------------------------------===//
// Evaluation scope
//===----------------------------------------------------------------------===//

/// A stable ordinal local to one query-kind descriptor in the exact Evaluation
/// schema version. The containing MetricKind or FindingKind always resolves it,
/// so it is neither a global relation registry nor a free string.
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

struct AcceptedLocalTarget {
  const LocalTargetFamilyDescriptor *family;
  LocalTargetKind localKind;
};

struct ScopeRoleDescriptor {
  ScopeRoleRef role;
  llvm::StringRef semanticRole;
  bool acceptsArtifactRoot;
  llvm::ArrayRef<AcceptedLocalTarget> acceptedLocalTargets;
};

struct SubjectTargetRef;

struct ScopeFormDescriptor {
  ScopeFormRef form;
  llvm::StringRef semanticDefinition;
  /// Ordered, nonrepeating role tuple: each role ordinal is its position and
  /// the semantic roles are distinct.
  llvm::ArrayRef<ScopeRoleDescriptor> roles;
  /// Relation-specific verification owned by this query form, such as requiring
  /// two distinct endpoints. Null when per-role acceptance is the whole
  /// contract.
  llvm::Error (*verifyRelation)(llvm::ArrayRef<SubjectTargetRef> targets);
};

const ScopeFormDescriptor *
findScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms, ScopeFormRef form);

struct ArtifactRootTarget {
  ArtifactIdentity artifact;

  friend bool operator==(const ArtifactRootTarget &lhs,
                         const ArtifactRootTarget &rhs) {
    return lhs.artifact == rhs.artifact;
  }
  friend bool operator!=(const ArtifactRootTarget &lhs,
                         const ArtifactRootTarget &rhs) {
    return !(lhs == rhs);
  }
};

using SubjectTarget = std::variant<ArtifactRootTarget, LocalTargetRef>;

struct SubjectTargetRef {
  CaseSubjectRoleRef caseSubjectRole;
  ArtifactIdentity anchorSubject;
  SubjectTarget target;

  const ArtifactIdentity &targetArtifact() const;

  friend bool operator==(const SubjectTargetRef &lhs,
                         const SubjectTargetRef &rhs) {
    return lhs.caseSubjectRole == rhs.caseSubjectRole &&
           lhs.anchorSubject == rhs.anchorSubject && lhs.target == rhs.target;
  }
  friend bool operator!=(const SubjectTargetRef &lhs,
                         const SubjectTargetRef &rhs) {
    return !(lhs == rhs);
  }
};

/// The one closed scope algebra shared by every MetricKind and FindingKind. The
/// target tuple has exactly the descriptor arity in descriptor role order and
/// is never sorted as an unordered set. A zero-role form denotes the entire
/// exact Evaluation case.
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

/// The canonical scope key: form ordinal, exact arity framing, and each fully
/// framed target in descriptor role order, including the case role, anchor
/// identity, target Artifact identity, target-kind discriminator, and the
/// complete family-owned canonical local payload.
std::vector<std::uint8_t> canonicalScopeKey(const EvaluationScope &scope);

/// Query-descriptor-relative validation: the form exists among the query kind's
/// own forms, its role tuple is ordered and nonrepeating, the target tuple has
/// exactly the descriptor arity, every target kind is accepted by its role, and
/// the form's own relation verification holds.
llvm::Error
validateEvaluationScopeForm(llvm::ArrayRef<ScopeFormDescriptor> forms,
                            const EvaluationScope &scope);

//===----------------------------------------------------------------------===//
// Case-relative target validation
//===----------------------------------------------------------------------===//

/// The case-relative facts a target validator needs: the exact case signature,
/// its bound subjects, and the resolved case Artifacts.
class CaseTargetContext {
public:
  CaseTargetContext(const EvaluationCaseSignatureDescriptor &signature,
                    const EvaluationSubjectBindings &bindings,
                    const CaseArtifactResolution &resolution)
      : signature_(&signature), bindings_(&bindings), resolution_(&resolution) {
  }

  const EvaluationCaseSignatureDescriptor &signature() const {
    return *signature_;
  }
  const EvaluationSubjectBindings &bindings() const { return *bindings_; }
  const CaseArtifactResolution &resolution() const { return *resolution_; }

private:
  const EvaluationCaseSignatureDescriptor *signature_;
  const EvaluationSubjectBindings *bindings_;
  const CaseArtifactResolution *resolution_;
};

/// Rejects a foreign case role, an anchor that is not an exact member of that
/// role's binding, an unresolved Artifact, a target outside the anchor's exact
/// dependency closure, and a local target whose Artifact does not belong to the
/// naming family.
llvm::Error validateSubjectTargetRef(const SubjectTargetRef &target,
                                     const CaseTargetContext &context);

llvm::Error validateEvaluationScopeCase(const EvaluationScope &scope,
                                        const CaseTargetContext &context);

//===----------------------------------------------------------------------===//
// Evaluation conditions
//===----------------------------------------------------------------------===//

/// An exact family-owned typed reference into immutable technology data. The
/// provider Artifact owns the corner catalog; a bare corner string is invalid.
class TechnologyCornerRef {
public:
  template <typename CornerId>
  explicit TechnologyCornerRef(const ArtifactReference<CornerId> &reference)
      : provider_(reference.artifact), corner_(reference.entity.value()) {}

  const ArtifactIdentity &provider() const { return provider_; }
  std::uint64_t corner() const { return corner_; }

  friend bool operator==(const TechnologyCornerRef &lhs,
                         const TechnologyCornerRef &rhs) {
    return lhs.provider_ == rhs.provider_ && lhs.corner_ == rhs.corner_;
  }
  friend bool operator!=(const TechnologyCornerRef &lhs,
                         const TechnologyCornerRef &rhs) {
    return !(lhs == rhs);
  }

private:
  ArtifactIdentity provider_;
  std::uint64_t corner_;
};

struct ProcessCornerCondition {
  SubjectTargetRef target;
  TechnologyCornerRef corner;

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

/// One exact typed activity summary owned by a SimulationExecution. That family
/// owns canonical summary order, ordinal range, source-basis coverage, and
/// Request-lineage validation.
struct ExecutionActivitySource {
  ArtifactIdentity simulationExecution;
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

llvm::StringRef toString(EvaluationConditionKind kind);
llvm::StringRef toString(ConditionLocation location);

/// The kind-owned assignment-key projection. Two values of one kind with the
/// same assignment key but different payloads are a conflict.
std::vector<std::uint8_t>
conditionAssignmentKey(const EvaluationCondition &condition);

/// The complete canonical payload key, used for canonical ordering and exact
/// duplicate detection.
std::vector<std::uint8_t>
conditionPayloadKey(const EvaluationCondition &condition);

/// Validates each condition's location, applicability against the exact
/// permitting owner's pattern, typed payload, and case-bound targets, then
/// returns the canonical set ordered by (kind, assignment key, complete payload
/// key). An exact duplicate and an assignment conflict are both invalid; there
/// is no last-wins behavior or override layer.
llvm::Expected<std::vector<EvaluationCondition>>
canonicalizeEvaluationConditions(llvm::ArrayRef<EvaluationCondition> conditions,
                                 const ConditionApplicability &applicability,
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
      std::optional<ArtifactIdentity> workload,
      std::optional<ArtifactIdentity> runtimeInput,
      llvm::ArrayRef<EvaluationCondition> baseConditions,
      const CaseArtifactResolution &resolution);

  EvaluationCaseSignatureRef signature() const { return signature_; }
  const EvaluationSubjectBindings &subjectBindings() const { return bindings_; }
  const std::optional<ArtifactIdentity> &workload() const { return workload_; }
  const std::optional<ArtifactIdentity> &runtimeInput() const {
    return runtimeInput_;
  }
  llvm::ArrayRef<EvaluationCondition> baseConditions() const {
    return baseConditions_;
  }

  CaseTargetContext
  targetContext(const CaseArtifactResolution &resolution) const {
    return CaseTargetContext(signature_.descriptor(), bindings_, resolution);
  }

private:
  EvaluationCase(EvaluationCaseSignatureRef signature,
                 EvaluationSubjectBindings bindings,
                 std::optional<ArtifactIdentity> workload,
                 std::optional<ArtifactIdentity> runtimeInput,
                 std::vector<EvaluationCondition> baseConditions)
      : signature_(signature), bindings_(std::move(bindings)),
        workload_(std::move(workload)), runtimeInput_(std::move(runtimeInput)),
        baseConditions_(std::move(baseConditions)) {}

  EvaluationCaseSignatureRef signature_;
  EvaluationSubjectBindings bindings_;
  std::optional<ArtifactIdentity> workload_;
  std::optional<ArtifactIdentity> runtimeInput_;
  std::vector<EvaluationCondition> baseConditions_;
};

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_CASE_H
