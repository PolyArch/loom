#ifndef LOOM_DSE_CANDIDATEGENERATOR_H
#define LOOM_DSE_CANDIDATEGENERATOR_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Common/ProviderForm.h"
#include "DSE/PlanValue.h"
#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

inline constexpr ArtifactSchemaDescriptor candidateGeneratorDescriptorSchema{
    "loom.candidate_generator_descriptor", SchemaVersion{2, 0}};

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

struct CandidateGeneratorInputSlotDescriptor final {
  CandidateGeneratorInputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
};

struct CandidateGeneratorOutputSlotDescriptor final {
  CandidateGeneratorOutputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
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
      llvm::ArrayRef<std::uint8_t>,
      llvm::ArrayRef<ArtifactRootReference> canonicalParents,
      const ArtifactStore &store);
};

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
  /// registry-2.0 descriptor reference before any implementation lookup.
  ProviderForm providerForm;

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
};

using CandidateGeneratorProviderFunction =
    llvm::Expected<CandidateGeneratorProviderResult> (*)(
        llvm::ArrayRef<CandidateGeneratorInputBinding>,
        const ResolvedCandidateGeneratorBinding &, const ArtifactStore &,
        const BlobStore &);

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

  friend bool operator==(
      const CandidateGeneratorExternalPrepareImportProvider &lhs,
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

llvm::Error validateCandidateGeneratorWorkSummary(
    CandidateGeneratorDescriptorRef descriptor,
    llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> summary);

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
