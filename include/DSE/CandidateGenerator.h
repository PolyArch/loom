#ifndef LOOM_DSE_CANDIDATEGENERATOR_H
#define LOOM_DSE_CANDIDATEGENERATOR_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Common/ExecutionControl.h"
#include "Common/ProviderForm.h"
#include "DSE/PlanValue.h"
#include "Evaluation/ModelParameterBundle.h"
#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

/// Host-local execution capacity for independent candidate work. Providers
/// may cap this further by the number of canonical work slots.
std::uint32_t defaultCandidateWorkerCount();

inline constexpr ArtifactSchemaDescriptor candidateGeneratorDescriptorSchema{
    "loom.candidate_generator_descriptor", SchemaVersion{3, 0}};

class CandidateGeneratorKind final {
public:
  explicit constexpr CandidateGeneratorKind(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorKind lhs,
                                   CandidateGeneratorKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorKind lhs,
                                   CandidateGeneratorKind rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(CandidateGeneratorKind lhs,
                                  CandidateGeneratorKind rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorInputSlotRef final {
public:
  explicit constexpr CandidateGeneratorInputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorInputSlotRef lhs,
                                   CandidateGeneratorInputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorInputSlotRef lhs,
                                   CandidateGeneratorInputSlotRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorOutputSlotRef final {
public:
  explicit constexpr CandidateGeneratorOutputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorOutputSlotRef lhs,
                                   CandidateGeneratorOutputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorOutputSlotRef lhs,
                                   CandidateGeneratorOutputSlotRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorWorkUnitRef final {
public:
  explicit constexpr CandidateGeneratorWorkUnitRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorWorkUnitRef lhs,
                                   CandidateGeneratorWorkUnitRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

struct CandidateGeneratorInputBinding final {
  CandidateGeneratorInputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

/// One plan-derived bound for an output slot. A missing maximum means that
/// every provider-owned candidate remains observable by at least one consumer.
struct CandidateGeneratorOutputDemand final {
  CandidateGeneratorOutputSlotRef slot;
  std::optional<std::uint64_t> maximumArtifacts;
};

/// Immutable invocation policy derived from the resolved plan. Output demand
/// is semantic visibility policy owned by that plan and bounds provider work
/// only when the provider contract explicitly permits it. Interruption and
/// resource budgets remain transient execution policy. None is mutable
/// candidate state.
class CandidateGeneratorInvocationView final {
public:
  constexpr CandidateGeneratorInvocationView() = default;
  constexpr CandidateGeneratorInvocationView(
      ExecutionControlView executionControl,
      llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands,
      ExecutionResourceBudget executionBudget = {})
      : executionControl_(executionControl), outputDemands_(outputDemands),
        executionBudget_(executionBudget) {}

  ExecutionControlView executionControl() const { return executionControl_; }
  bool stopRequested() const { return executionControl_.stopRequested(); }
  std::optional<std::uint64_t>
  maximumOutputArtifacts(CandidateGeneratorOutputSlotRef slot) const;
  llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands() const {
    return outputDemands_;
  }
  ExecutionResourceBudget executionBudget() const { return executionBudget_; }

private:
  ExecutionControlView executionControl_;
  llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands_;
  ExecutionResourceBudget executionBudget_;
};

struct CandidateGeneratorInputSlotDescriptor final {
  CandidateGeneratorInputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
  const evaluation::ModelParameterContractRef *modelParameterContract = nullptr;
  std::optional<CalibrationPartitionRole> calibrationPartitionRole =
      std::nullopt;
};

struct CandidateGeneratorOutputSlotDescriptor final {
  CandidateGeneratorOutputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
  const evaluation::ModelParameterContractRef *modelParameterContract = nullptr;
  std::optional<CalibrationPartitionRole> calibrationPartitionRole =
      std::nullopt;
};

enum class CandidateGeneratorDeterminism : std::uint32_t {
  Deterministic,
  IndependentReplicates,
};

struct CandidateGeneratorWorkUnitDescriptor final {
  CandidateGeneratorWorkUnitRef unit;
  llvm::StringRef spelling;
};

/// Invocation-local execution accounting for one descriptor-owned work-unit
/// kind. Counts summarize logical slots, not policy limits or wall time.
struct CandidateGeneratorWorkUnitSummary final {
  CandidateGeneratorWorkUnitRef unit;
  std::uint64_t planned = 0;
  std::uint64_t consumed = 0;

  friend bool operator==(const CandidateGeneratorWorkUnitSummary &lhs,
                         const CandidateGeneratorWorkUnitSummary &rhs) {
    return lhs.unit == rhs.unit && lhs.planned == rhs.planned &&
           lhs.consumed == rhs.consumed;
  }
};

struct CandidateGeneratorOwnerLineagePayloadContract final {
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes;
  llvm::Error (*validateCanonical)(
      llvm::ArrayRef<std::uint8_t>, const ArtifactRootReference &output,
      llvm::ArrayRef<ArtifactRootReference> canonicalParents,
      const ArtifactStore &store);
};

/// Descriptor-owned validation for one invocation-local hardware-demand or
/// other search-feedback payload. The payload is never persisted by the
/// generic controller; a domain-specific bounded reopen may consume it before
/// the invocation ends.
struct CandidateGeneratorOwnerFeedbackPayloadContract final {
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes;
  llvm::Error (*validateCanonical)(
      llvm::ArrayRef<std::uint8_t>,
      llvm::ArrayRef<CandidateGeneratorInputBinding> canonicalInputs,
      const ArtifactStore &store);
};

struct CandidateGeneratorOwnerOutcomeContract;
struct CandidateGeneratorDescriptor;

class CandidateGeneratorDescriptorRef final {
public:
  static llvm::Expected<CandidateGeneratorDescriptorRef>
  get(const ArtifactSchemaDescriptor &descriptorSchema,
      CandidateGeneratorKind kind);

  const ArtifactSchemaDescriptor &descriptorSchema() const {
    return descriptorSchema_;
  }
  CandidateGeneratorKind kind() const { return kind_; }
  const CandidateGeneratorDescriptor *descriptor() const;

  friend bool operator==(const CandidateGeneratorDescriptorRef &lhs,
                         const CandidateGeneratorDescriptorRef &rhs) {
    return lhs.descriptorSchema_ == rhs.descriptorSchema_ &&
           lhs.kind_ == rhs.kind_;
  }
  friend bool operator!=(const CandidateGeneratorDescriptorRef &lhs,
                         const CandidateGeneratorDescriptorRef &rhs) {
    return !(lhs == rhs);
  }

private:
  CandidateGeneratorDescriptorRef(ArtifactSchemaDescriptor descriptorSchema,
                                  CandidateGeneratorKind kind)
      : descriptorSchema_(descriptorSchema), kind_(kind) {}

  ArtifactSchemaDescriptor descriptorSchema_;
  CandidateGeneratorKind kind_;
};

struct CandidateGeneratorDescriptor final {
  CandidateGeneratorKind kind;
  llvm::StringRef spelling;
  llvm::StringRef implementationSemanticIdentity;
  llvm::ArrayRef<CandidateGeneratorInputSlotDescriptor> inputSlots;
  llvm::ArrayRef<CandidateGeneratorOutputSlotDescriptor> outputSlots;
  ResolvedDseConfigViewContract resolvedConfigView;
  CandidateGeneratorDeterminism determinism;
  llvm::ArrayRef<CandidateGeneratorWorkUnitDescriptor> workUnits;
  const CandidateGeneratorOwnerLineagePayloadContract *ownerLineagePayload =
      nullptr;
  /// The closed provider form of this descriptor, recovered from the exact
  /// registry-3.0 descriptor reference before any implementation lookup.
  ProviderForm providerForm;
  const CandidateGeneratorOwnerFeedbackPayloadContract *ownerFeedbackPayload =
      nullptr;
  const CandidateGeneratorOwnerOutcomeContract *ownerOutcome = nullptr;

  CandidateGeneratorDescriptorRef reference() const;
  const CandidateGeneratorInputSlotDescriptor *
  findInputSlot(CandidateGeneratorInputSlotRef slot) const;
  const CandidateGeneratorOutputSlotDescriptor *
  findOutputSlot(CandidateGeneratorOutputSlotRef slot) const;
};

llvm::Error registerCandidateGeneratorDescriptor(
    const CandidateGeneratorDescriptor &descriptor);
const CandidateGeneratorDescriptor *
findCandidateGeneratorDescriptor(CandidateGeneratorKind kind);

llvm::Error validateCandidateGeneratorInputBindings(
    CandidateGeneratorDescriptorRef descriptor,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings);

class ResolvedCandidateGeneratorBinding final {
public:
  static llvm::Expected<ResolvedCandidateGeneratorBinding>
  get(CandidateGeneratorDescriptorRef descriptor,
      llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
      const ComponentViewDigest &configDigest);

  CandidateGeneratorDescriptorRef descriptorRef() const { return descriptor_; }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }

private:
  ResolvedCandidateGeneratorBinding(
      CandidateGeneratorDescriptorRef descriptor,
      std::vector<std::uint8_t> canonicalConfigBytes,
      ComponentViewDigest configDigest)
      : descriptor_(descriptor),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest) {}

  CandidateGeneratorDescriptorRef descriptor_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;
};

struct CandidateGeneratorOutputBinding final {
  CandidateGeneratorOutputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

enum class CandidateGeneratorLineageEdgeKind : std::uint32_t {
  MechanicalDerivation = 0,
  CandidateDecision = 1,
};

struct CandidateGeneratorLineageEdge final {
  CandidateGeneratorLineageEdgeKind kind;
  CandidateGeneratorOutputSlotRef outputSlot;
  ArtifactRootReference output;
  std::vector<ArtifactRootReference> parents;
  std::vector<std::uint8_t> ownerPayload;

  friend bool operator==(const CandidateGeneratorLineageEdge &lhs,
                         const CandidateGeneratorLineageEdge &rhs) {
    return lhs.kind == rhs.kind && lhs.outputSlot == rhs.outputSlot &&
           lhs.output == rhs.output && lhs.parents == rhs.parents &&
           lhs.ownerPayload == rhs.ownerPayload;
  }
};

/// Descriptor-owned closure across all canonical output slots and lineage
/// edges. Per-slot cardinality and per-edge payload validation remain generic;
/// this contract owns semantic relations that span several output artifacts.
struct CandidateGeneratorOwnerOutcomeContract final {
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes;
  llvm::Error (*validateCanonical)(
      llvm::ArrayRef<CandidateGeneratorInputBinding> canonicalInputs,
      llvm::ArrayRef<CandidateGeneratorOutputBinding> canonicalOutputs,
      llvm::ArrayRef<CandidateGeneratorLineageEdge> canonicalLineageEdges,
      bool completed, const ArtifactStore &store);
};

struct CompletedCandidateGeneratorResult final {
  std::vector<CandidateGeneratorOutputBinding> outputBindings;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
};

enum class CandidateGeneratorIncompleteReason : std::uint32_t {
  ProofNotEstablished = 0,
  SemanticLimitReached = 1,
  ProviderUnavailable = 2,
  Unsupported = 3,
  ExecutionFailed = 4,
  CancelledOrTimeout = 5,
};

struct IncompleteCandidateGeneratorResult final {
  CandidateGeneratorIncompleteReason reason;
  std::vector<CandidateGeneratorOutputBinding> retainedOutputBindings;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
};

using CandidateGeneratorProviderOutcome =
    std::variant<CompletedCandidateGeneratorResult,
                 IncompleteCandidateGeneratorResult>;

/// One transient provider report: the outcome variant plus exactly one dense
/// work summary outside it. The descriptor owns the stable work-unit
/// ordinals; the provider is the sole runtime observation source of the
/// planned and consumed counts; InvocationManifest remains the sole
/// persistent owner.
struct CandidateGeneratorProviderResult final {
  CandidateGeneratorProviderOutcome outcome;
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary;
  /// Optional descriptor-owned transient feedback. It is validated against
  /// the exact invocation inputs, excluded from Artifact and journal identity,
  /// and unavailable to terminal replay unless promoted by its semantic owner.
  std::optional<std::vector<std::uint8_t>> ownerFeedback = std::nullopt;
  /// True only when this result came from an actual provider invocation or
  /// external tool execution in the current process. Recovery/import of a
  /// terminal work record leaves it false. This is operational provenance;
  /// it never participates in candidate identity or persisted recovery bytes.
  bool dispatched = false;
};

using CandidateGeneratorProviderFunction =
    llvm::Expected<CandidateGeneratorProviderResult> (*)(
        llvm::ArrayRef<CandidateGeneratorInputBinding>,
        const ResolvedCandidateGeneratorBinding &, const ArtifactStore &,
        const BlobStore &, const CandidateGeneratorInvocationView &);

/// The external prepare callable of one ExternalPrepareImport generator. It
/// materializes one deterministic finalized bundle and never executes a
/// process, publishes an output Artifact, or publishes Evidence.
using CandidateGeneratorPrepareFunction =
    llvm::Expected<external_tool::PreparedExternalToolInvocation> (*)(
        llvm::ArrayRef<CandidateGeneratorInputBinding>,
        const ResolvedCandidateGeneratorBinding &, const ArtifactStore &,
        const BlobStore &,
        const external_tool::ExternalToolPreparationContext &);

/// The external import callable of one ExternalPrepareImport generator. It
/// receives the full typed closure again, validates strict completion against
/// the exact prepared bundle, and returns the provider result.
using CandidateGeneratorImportFunction =
    llvm::Expected<CandidateGeneratorProviderResult> (*)(
        llvm::ArrayRef<CandidateGeneratorInputBinding>,
        const ResolvedCandidateGeneratorBinding &,
        const external_tool::PreparedExternalToolInvocation &,
        const ArtifactStore &, const BlobStore &);

/// The closed provider implementation forms. The registered form must match
/// the descriptor's provider form exactly.
struct CandidateGeneratorInProcessProvider final {
  CandidateGeneratorProviderFunction invoke;

  friend bool operator==(const CandidateGeneratorInProcessProvider &lhs,
                         const CandidateGeneratorInProcessProvider &rhs) {
    return lhs.invoke == rhs.invoke;
  }
};

struct CandidateGeneratorExternalPrepareImportProvider final {
  CandidateGeneratorPrepareFunction prepare;
  CandidateGeneratorImportFunction import;

  friend bool
  operator==(const CandidateGeneratorExternalPrepareImportProvider &lhs,
             const CandidateGeneratorExternalPrepareImportProvider &rhs) {
    return lhs.prepare == rhs.prepare && lhs.import == rhs.import;
  }
};

using CandidateGeneratorProviderImplementation =
    std::variant<CandidateGeneratorInProcessProvider,
                 CandidateGeneratorExternalPrepareImportProvider>;

struct CandidateGeneratorProvider final {
  CandidateGeneratorDescriptorRef descriptor;
  CandidateGeneratorProviderImplementation implementation;
};

llvm::Error
registerCandidateGeneratorProvider(const CandidateGeneratorProvider &provider);

/// Invokes the exact registered in-process provider and canonicalizes every
/// typed output set. Missing implementation is a typed Incomplete outcome
/// with the mechanically derived all-zero work summary; an
/// ExternalPrepareImport provider is never invoked through this facade.
llvm::Expected<CandidateGeneratorProviderResult>
invokeCandidateGenerator(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                         const ResolvedCandidateGeneratorBinding &binding,
                         const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<CandidateGeneratorProviderResult>
invokeCandidateGenerator(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                         const ResolvedCandidateGeneratorBinding &binding,
                         const ArtifactStore &store, const BlobStore &blobs,
                         const ExecutionControlView &executionControl);

llvm::Expected<CandidateGeneratorProviderResult>
invokeCandidateGenerator(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                         const ResolvedCandidateGeneratorBinding &binding,
                         const ArtifactStore &store, const BlobStore &blobs,
                         const CandidateGeneratorInvocationView &invocation);

llvm::Error validateCandidateGeneratorWorkSummary(
    CandidateGeneratorDescriptorRef descriptor,
    llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> summary);

/// Revalidates a complete provider report at an owner-controlled persistence
/// boundary. Canonical set ordering is restored in-place before the caller
/// compares or publishes the result.
llvm::Error validateCandidateGeneratorProviderResult(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    CandidateGeneratorProviderResult &result, const ArtifactStore &store,
    const BlobStore &blobs);

/// Prepares one deterministic finalized invocation bundle through the exact
/// registered ExternalPrepareImport provider. The descriptor form is
/// validated before any provider lookup; the caller alone decides whether,
/// where, and when to execute run.sh.
llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareCandidateGeneratorInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

/// Strictly imports one prepared invocation through the exact registered
/// ExternalPrepareImport provider and validates the provider result against
/// the full typed closure.
llvm::Expected<CandidateGeneratorProviderResult>
importCandidateGeneratorInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &store, const BlobStore &blobs);

/// The canonical key bytes of one descriptor reference under the shared
/// owner-local registry reference framing: u64be identity length, exact
/// registry identity bytes, u32be major and minor schema versions, and the
/// u32be owner-local kind.
std::vector<std::uint8_t> canonicalCandidateGeneratorDescriptorReferenceBytes(
    CandidateGeneratorDescriptorRef reference);

/// The mechanically derived binding identity of one resolved generator
/// binding: SHA-256 over the "loom.candidate_generator_binding.v1\0" domain
/// prefix and the length-framed canonical descriptor-reference and
/// resolved-config-view bytes. Never caller-authored.
BlobDigest deriveCandidateGeneratorBindingIdentity(
    CandidateGeneratorDescriptorRef descriptor,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes);

/// Derives the complete external-tool semantic contract from the exact typed
/// generator invocation. The descriptor must select ExternalPrepareImport;
/// adapters consume the returned value without encoding any semantic field.
llvm::Expected<external_tool::ExternalToolSemanticContract>
deriveExternalToolSemanticContract(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding);

/// Strictly revalidates one immutable invocation record at an external
/// consumption boundary. The check imports every exact input, output, parent,
/// and internal lineage target; it does not create another record authority.
llvm::Error validateCanonicalCandidateGeneratorInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs,
    llvm::ArrayRef<CandidateGeneratorLineageEdge> lineageEdges, bool completed,
    const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_CANDIDATEGENERATOR_H
