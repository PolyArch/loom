#ifndef LOOM_EVALUATION_MODELDESCRIPTOR_H
#define LOOM_EVALUATION_MODELDESCRIPTOR_H

#include "Evaluation/Case.h"
#include "Evaluation/Finding.h"
#include "Evaluation/Metric.h"
#include "Evaluation/OwnerValue.h"

#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::evaluation {

class MetricRequest;
class FindingRequest;

} // namespace loom::evaluation

namespace loom {
struct ResolvedConfig;
}

namespace loom::evaluation {

class EvaluationModelKind {
public:
  explicit constexpr EvaluationModelKind(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(EvaluationModelKind lhs,
                                   EvaluationModelKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(EvaluationModelKind lhs,
                                   EvaluationModelKind rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(EvaluationModelKind lhs,
                                  EvaluationModelKind rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

enum class ModeledPhenomenon : std::uint32_t {
  StructuredProgram,
  CanonicalDataflow,
  SpatialResources,
  RoutedTransport,
  FiniteBuffering,
  MemoryContention,
  ClockTiming,
  SystemMemoryHierarchy,
  Coherence,
  RTLBehavior,
  PhysicalImplementation,
};

enum class EvaluationExecutionMethod : std::uint32_t {
  Analytic,
  Simulation,
  Emulation,
  ToolMeasurement,
  PhysicalMeasurement,
};

enum class EvaluationInteractionMode : std::uint32_t {
  Incremental,
  Guidance,
};

llvm::StringRef toString(ModeledPhenomenon phenomenon);
llvm::StringRef toString(EvaluationExecutionMethod method);
llvm::StringRef toString(EvaluationInteractionMode mode);

class EvaluationInteractionDomainRef {
public:
  static llvm::Expected<EvaluationInteractionDomainRef>
  get(llvm::StringRef ownerRegistryIdentity, SchemaVersion ownerRegistryVersion,
      std::uint32_t ownerLocalDomainKind);

  llvm::StringRef ownerRegistryIdentity() const {
    return ownerRegistryIdentity_;
  }
  SchemaVersion ownerRegistryVersion() const { return ownerRegistryVersion_; }
  std::uint32_t ownerLocalDomainKind() const { return ownerLocalDomainKind_; }

  friend bool operator==(const EvaluationInteractionDomainRef &lhs,
                         const EvaluationInteractionDomainRef &rhs) {
    return lhs.ownerRegistryIdentity_ == rhs.ownerRegistryIdentity_ &&
           lhs.ownerRegistryVersion_ == rhs.ownerRegistryVersion_ &&
           lhs.ownerLocalDomainKind_ == rhs.ownerLocalDomainKind_;
  }
  friend bool operator!=(const EvaluationInteractionDomainRef &lhs,
                         const EvaluationInteractionDomainRef &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const EvaluationInteractionDomainRef &lhs,
                        const EvaluationInteractionDomainRef &rhs);

private:
  EvaluationInteractionDomainRef(std::string ownerRegistryIdentity,
                                 SchemaVersion ownerRegistryVersion,
                                 std::uint32_t ownerLocalDomainKind)
      : ownerRegistryIdentity_(std::move(ownerRegistryIdentity)),
        ownerRegistryVersion_(ownerRegistryVersion),
        ownerLocalDomainKind_(ownerLocalDomainKind) {}

  std::string ownerRegistryIdentity_;
  SchemaVersion ownerRegistryVersion_;
  std::uint32_t ownerLocalDomainKind_;
};

/// Registry entry supplied by an interaction-domain owner. The validator is
/// the owner's admission gate for its typed in-process protocol; Evaluation
/// does not serialize or reinterpret candidate, delta, query, or value types.
struct EvaluationInteractionDomainDescriptor {
  EvaluationInteractionDomainRef reference;
  llvm::StringRef semanticDefinition;
  llvm::ArrayRef<EvaluationInteractionMode> implementedModes;
  llvm::Error (*validateTypedProtocol)(EvaluationInteractionMode mode);
};

struct EvaluationInteractionCapability {
  EvaluationInteractionDomainRef domain;
  llvm::ArrayRef<EvaluationInteractionMode> modes;
};

llvm::Error registerEvaluationInteractionDomain(
    const EvaluationInteractionDomainDescriptor &descriptor);
const EvaluationInteractionDomainDescriptor *
findEvaluationInteractionDomain(const EvaluationInteractionDomainRef &domain);

/// The persistent model descriptor reference is exactly the Evaluation schema
/// version and one stable model kind.
class EvaluationModelDescriptorRef {
public:
  static llvm::Expected<EvaluationModelDescriptorRef>
  get(SchemaVersion schemaVersion, EvaluationModelKind modelKind);

  SchemaVersion schemaVersion() const { return schemaVersion_; }
  EvaluationModelKind modelKind() const { return modelKind_; }
  const struct EvaluationModelDescriptor *descriptor() const;

  friend bool operator==(EvaluationModelDescriptorRef lhs,
                         EvaluationModelDescriptorRef rhs) {
    return lhs.schemaVersion_ == rhs.schemaVersion_ &&
           lhs.modelKind_ == rhs.modelKind_;
  }
  friend bool operator!=(EvaluationModelDescriptorRef lhs,
                         EvaluationModelDescriptorRef rhs) {
    return !(lhs == rhs);
  }

private:
  EvaluationModelDescriptorRef(SchemaVersion schemaVersion,
                               EvaluationModelKind modelKind)
      : schemaVersion_(schemaVersion), modelKind_(modelKind) {}

  SchemaVersion schemaVersion_;
  EvaluationModelKind modelKind_;
};

class ModelInputSlotRef {
public:
  explicit constexpr ModelInputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(ModelInputSlotRef lhs,
                                   ModelInputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(ModelInputSlotRef lhs,
                                   ModelInputSlotRef rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(ModelInputSlotRef lhs,
                                  ModelInputSlotRef rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class ModelOutputSlotRef {
public:
  explicit constexpr ModelOutputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(ModelOutputSlotRef lhs,
                                   ModelOutputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(ModelOutputSlotRef lhs,
                                   ModelOutputSlotRef rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(ModelOutputSlotRef lhs,
                                  ModelOutputSlotRef rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

enum class ArtifactCollectionCardinality : std::uint8_t {
  Forbidden,
  ZeroOrOne,
  ExactlyOne,
  OneOrMore
};

enum class EvidenceOutcomeKind : std::uint8_t {
  Completed,
  Unsupported,
  ExecutionFailed,
  CancelledOrTimeout
};

llvm::StringRef toString(EvidenceOutcomeKind outcome);
llvm::Error
validateArtifactCollectionCardinality(ArtifactCollectionCardinality cardinality,
                                      std::size_t count, llvm::StringRef owner);

struct ModelInputBinding {
  ModelInputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;

  friend bool operator==(const ModelInputBinding &lhs,
                         const ModelInputBinding &rhs) {
    return lhs.slot == rhs.slot && lhs.artifacts == rhs.artifacts;
  }
};

struct ModelInputSlotDescriptor {
  ModelInputSlotRef slot;
  llvm::StringRef semanticRole;
  llvm::ArrayRef<const ArtifactSchemaDescriptor *> acceptedSchemas;
  ArtifactCollectionCardinality cardinality;
  llvm::Error (*verifyCompatibility)(
      llvm::ArrayRef<ArtifactRootReference> artifacts,
      llvm::ArrayRef<ModelInputBinding> allBindings,
      const ArtifactStore &artifactStore);
};

struct ModelOutputSlotDescriptor {
  ModelOutputSlotRef slot;
  llvm::StringRef semanticRole;
  const ArtifactSchemaDescriptor *schema;
  std::array<ArtifactCollectionCardinality, 4> outcomeCardinalities;

  ArtifactCollectionCardinality cardinality(EvidenceOutcomeKind outcome) const {
    return outcomeCardinalities[static_cast<std::size_t>(outcome)];
  }
};

struct MetricCapability {
  MetricKind kind;
  llvm::ArrayRef<ScopeFormRef> scopeForms;
  std::uint8_t permittedObservationForms;
};

struct FindingCapability {
  FindingKind kind;
  llvm::ArrayRef<ScopeFormRef> scopeForms;
  std::uint8_t permittedResultForms;
};

/// The exact owner contract for the immutable ResolvedConfig component view
/// consumed by one model descriptor.
struct ResolvedModelConfigViewContract {
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes;
  llvm::Expected<OwnerValue> (*project)(const ResolvedConfig &config);
  llvm::Expected<std::vector<std::uint8_t>> (*encode)(
      const OwnerValue &view);
  llvm::Expected<OwnerValue> (*adopt)(
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
      const ComponentViewDigest &digest);
};

/// One descriptor-adopted typed view and its persistent wire. The descriptor
/// reference recovers the schema bytes; they are never copied into Request.
class ResolvedModelConfigView {
public:
  static llvm::Expected<ResolvedModelConfigView>
  project(EvaluationModelDescriptorRef descriptor,
          const ResolvedConfig &config);
  static llvm::Expected<ResolvedModelConfigView>
  adopt(EvaluationModelDescriptorRef descriptor,
        std::vector<std::uint8_t> canonicalViewBytes,
        ComponentViewDigest digest);

  EvaluationModelDescriptorRef descriptorRef() const { return descriptor_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalViewBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

  template <typename T> const std::decay_t<T> *getIf() const {
    return value_.getIf<T>();
  }

private:
  ResolvedModelConfigView(EvaluationModelDescriptorRef descriptor,
                          std::vector<std::uint8_t> canonicalViewBytes,
                          ComponentViewDigest digest, OwnerValue value)
      : descriptor_(descriptor),
        canonicalViewBytes_(std::move(canonicalViewBytes)), digest_(digest),
        value_(std::move(value)) {}

  EvaluationModelDescriptorRef descriptor_;
  std::vector<std::uint8_t> canonicalViewBytes_;
  ComponentViewDigest digest_;
  OwnerValue value_;
};

enum class DeterminismContract : std::uint8_t {
  Deterministic,
  IndependentReplicates
};

/// What a model does with one exact condition pattern its permitting owner
/// already allows.
enum class ConditionDisposition : std::uint8_t {
  Consumed,
  Required,
  Invariant
};

struct ModelConditionCapability {
  ConditionApplicabilityPattern pattern;
  ConditionDisposition disposition;
};

/// One immutable entry in the Evaluation model registry. Subject roles remain
/// owned solely by the referenced case signature.
struct EvaluationModelDescriptor {
  EvaluationModelKind modelKind;
  llvm::StringRef spelling;
  llvm::StringRef implementationSemanticIdentity;
  EvaluationCaseSignatureRef caseSignature;
  llvm::ArrayRef<ModelConditionCapability> conditionCapabilities;
  llvm::ArrayRef<MetricCapability> metricCapabilities;
  llvm::ArrayRef<FindingCapability> findingCapabilities;
  llvm::ArrayRef<ModelInputSlotDescriptor> inputSlots;
  llvm::ArrayRef<ModelOutputSlotDescriptor> outputSlots;
  ResolvedModelConfigViewContract resolvedConfigView;
  llvm::ArrayRef<ModeledPhenomenon> modeledPhenomena;
  EvaluationExecutionMethod executionMethod;
  llvm::ArrayRef<EvaluationInteractionCapability> interactionCapabilities;
  DeterminismContract determinism;
  llvm::ArrayRef<FindingQuery> mandatoryTerminalFindings;

  EvaluationModelDescriptorRef reference() const;
  const ModelConditionCapability *
  findConditionCapability(const ConditionApplicabilityPattern &pattern) const;
  const MetricCapability *findMetricCapability(MetricKind metric) const;
  const FindingCapability *findFindingCapability(FindingKind finding) const;
  const ModelInputSlotDescriptor *findInputSlot(ModelInputSlotRef slot) const;
  const ModelOutputSlotDescriptor *
  findOutputSlot(ModelOutputSlotRef slot) const;
  const ModelOutputSlotDescriptor *
  outputSlotByOrdinal(std::uint32_t ordinal) const;
  bool supportsMetricQuery(const MetricQuery &query) const;
  bool supportsFindingQuery(const FindingQuery &query) const;
};

llvm::Error
registerEvaluationModelDescriptor(const EvaluationModelDescriptor &descriptor);
const EvaluationModelDescriptor *
findEvaluationModelDescriptor(EvaluationModelKind modelKind);

/// Canonical binary projection of the descriptor-owned phenomenon, execution,
/// and interaction capability fields.
std::vector<std::uint8_t> canonicalEvaluationModelCapabilityBytes(
    const EvaluationModelDescriptor &descriptor);

class ResolvedModelBinding {
public:
  static llvm::Expected<ResolvedModelBinding>
  get(EvaluationModelDescriptorRef descriptor,
      std::vector<ModelInputBinding> inputBindings,
      ResolvedModelConfigView resolvedModelConfig);

  static llvm::Expected<ResolvedModelBinding>
  project(EvaluationModelDescriptorRef descriptor,
          std::vector<ModelInputBinding> inputBindings,
          const ResolvedConfig &config);

  static llvm::Expected<ResolvedModelBinding>
  adopt(EvaluationModelDescriptorRef descriptor,
        std::vector<ModelInputBinding> inputBindings,
        std::vector<std::uint8_t> canonicalViewBytes,
        ComponentViewDigest digest);

  EvaluationModelDescriptorRef descriptorRef() const { return descriptor_; }
  llvm::ArrayRef<ModelInputBinding> inputBindings() const {
    return inputBindings_;
  }
  const ResolvedModelConfigView &resolvedModelConfig() const {
    return resolvedModelConfig_;
  }

  const ModelInputBinding *findInputBinding(ModelInputSlotRef slot) const;

private:
  ResolvedModelBinding(EvaluationModelDescriptorRef descriptor,
                       std::vector<ModelInputBinding> inputBindings,
                       ResolvedModelConfigView resolvedModelConfig)
      : descriptor_(descriptor), inputBindings_(std::move(inputBindings)),
        resolvedModelConfig_(std::move(resolvedModelConfig)) {}

  EvaluationModelDescriptorRef descriptor_;
  std::vector<ModelInputBinding> inputBindings_;
  ResolvedModelConfigView resolvedModelConfig_;
};

llvm::Error validateResolvedModelBinding(const ResolvedModelBinding &binding);

llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests,
                        llvm::ArrayRef<FindingRequest> findingRequests = {});

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELDESCRIPTOR_H
