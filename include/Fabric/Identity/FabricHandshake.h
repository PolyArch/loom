#ifndef LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H
#define LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace loom::fabric {

enum class HandshakeSignalKind : std::uint8_t { Valid, Ready };

struct HandshakeSignalRef final {
  FabricTransportEndpointRef endpoint;
  HandshakeSignalKind signal = HandshakeSignalKind::Valid;

  friend bool operator==(const HandshakeSignalRef &lhs,
                         const HandshakeSignalRef &rhs) {
    return lhs.endpoint == rhs.endpoint && lhs.signal == rhs.signal;
  }
  friend bool operator!=(const HandshakeSignalRef &lhs,
                         const HandshakeSignalRef &rhs) {
    return !(lhs == rhs);
  }
};

struct HandshakeDependencyArc final {
  HandshakeSignalRef source;
  HandshakeSignalRef destination;

  friend bool operator==(const HandshakeDependencyArc &lhs,
                         const HandshakeDependencyArc &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

enum class FabricHandshakeOwnerKind : std::uint8_t {
  PointConnection,
  PeOccurrence,
  FuOccurrence,
  MemoryOccurrence,
  SwitchOccurrence,
  FifoOccurrence,
  BoundaryOccurrence,
  TransferPattern,
};

/// A sealed-view owner selector. Every alternative contains an existing
/// Fabric reference or fixed connection; this union has no persistent codec.
class FabricHandshakeOwner final {
public:
  using Payload =
      std::variant<FabricPointConnectionPayload, FabricPeOccurrenceRef,
                   FabricFuOccurrenceRef, FabricMemoryOccurrenceRef,
                   FabricSwitchOccurrenceRef, FabricFifoOccurrenceRef,
                   FabricBoundaryOccurrenceRef, FabricTransferPatternRef>;

  static FabricHandshakeOwner pointConnection(FabricPointConnectionPayload);
  static FabricHandshakeOwner pe(FabricPeOccurrenceRef);
  static FabricHandshakeOwner fu(FabricFuOccurrenceRef);
  static FabricHandshakeOwner memory(FabricMemoryOccurrenceRef);
  static FabricHandshakeOwner switchResource(FabricSwitchOccurrenceRef);
  static FabricHandshakeOwner fifo(FabricFifoOccurrenceRef);
  static FabricHandshakeOwner boundary(FabricBoundaryOccurrenceRef);
  static FabricHandshakeOwner transferPattern(FabricTransferPatternRef);

  FabricHandshakeOwnerKind kind() const {
    return static_cast<FabricHandshakeOwnerKind>(payload_.index());
  }
  const Payload &payload() const { return payload_; }

  friend bool operator==(const FabricHandshakeOwner &lhs,
                         const FabricHandshakeOwner &rhs) {
    return lhs.payload_ == rhs.payload_;
  }
  friend bool operator!=(const FabricHandshakeOwner &lhs,
                         const FabricHandshakeOwner &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit FabricHandshakeOwner(Payload payload)
      : payload_(std::move(payload)) {}

  Payload payload_;
};

enum class HandshakeOwnerNodeKind : std::uint8_t {
  BoundarySignal,
  OwnerLocalJunction,
};

struct HandshakeOwnerNode final {
  HandshakeOwnerNodeKind kind = HandshakeOwnerNodeKind::OwnerLocalJunction;
  std::optional<HandshakeSignalRef> boundarySignal;
};

struct HandshakeOwnerArc final {
  std::uint32_t source = 0;
  std::uint32_t destination = 0;
};

enum class HandshakeActivationKind : std::uint8_t {
  Always,
  AnyTraversal,
  AllTraversals,
  AnySwitchActivationTraversal,
  ExactSwitchActivationTraversal,
  ExactOwnerSelection,
};

/// One Mapping-derived Temporal switch activation identity. The resident row
/// is occurrence-relative and rebuildable from the configured packed rows;
/// the input is the Fabric requester that fires within that row.
struct FabricSwitchHandshakeActivationKey final {
  FabricSwitchOccurrenceRef occurrence;
  FabricOrdinal row = 0;
  FabricOrdinal input = 0;

  friend bool operator==(const FabricSwitchHandshakeActivationKey &lhs,
                         const FabricSwitchHandshakeActivationKey &rhs) {
    return lhs.occurrence == rhs.occurrence && lhs.row == rhs.row &&
           lhs.input == rhs.input;
  }
  friend bool operator!=(const FabricSwitchHandshakeActivationKey &lhs,
                         const FabricSwitchHandshakeActivationKey &rhs) {
    return !(lhs == rhs);
  }
};

struct HandshakeActivationFragment final {
  std::uint32_t contributionOffset = 0;
  std::uint32_t contributionCount = 0;
  HandshakeActivationKind activationKind =
      HandshakeActivationKind::ExactOwnerSelection;
  std::uint32_t witnessOffset = 0;
  std::uint32_t witnessCount = 0;
  std::optional<FabricSwitchHandshakeActivationKey> switchActivation;
};

struct FabricFuOperationHandshakeBinding final {
  FabricFuTemplateNodeRef operation;
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::uint32_t indexBitWidth = 0;
  std::optional<::loom::PointerLayout> pointerLayout;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

struct FabricFuOperationHandshakeSelection final {
  FabricFuOccurrenceNodeRef operation;
  ::dataflow::OperationSchemaId schema =
      ::dataflow::OperationSchemaId::ArithAddI;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;

  friend bool operator==(const FabricFuOperationHandshakeSelection &lhs,
                         const FabricFuOperationHandshakeSelection &rhs) {
    return lhs.operation == rhs.operation && lhs.schema == rhs.schema &&
           lhs.operandPorts == rhs.operandPorts &&
           lhs.resultPorts == rhs.resultPorts;
  }
};

/// One exact FU occurrence configuration derived from a verified TechMapping
/// realization. The factory validates the capability row and every actor's
/// typed ordered port correspondence before this value can reach the owner
/// resolver.
class FabricFuHandshakeSelection final {
public:
  FabricFuOccurrenceRef occurrence() const { return occurrence_; }
  FabricFuCapabilityTemplateRef capability() const { return capability_; }
  llvm::ArrayRef<FabricFuOperationHandshakeSelection> operations() const {
    return operations_;
  }

  friend bool operator==(const FabricFuHandshakeSelection &lhs,
                         const FabricFuHandshakeSelection &rhs) {
    return lhs.occurrence_ == rhs.occurrence_ &&
           lhs.capability_ == rhs.capability_ &&
           lhs.operations_ == rhs.operations_;
  }

private:
  FabricFuHandshakeSelection(
      FabricFuOccurrenceRef occurrence,
      FabricFuCapabilityTemplateRef capability,
      std::vector<FabricFuOperationHandshakeSelection> operations)
      : occurrence_(occurrence), capability_(capability),
        operations_(std::move(operations)) {}

  FabricFuOccurrenceRef occurrence_;
  FabricFuCapabilityTemplateRef capability_;
  std::vector<FabricFuOperationHandshakeSelection> operations_;

  friend llvm::Expected<FabricFuHandshakeSelection>
  makeFuHandshakeSelection(const FabricArtifactView &, FabricFuOccurrenceRef,
                           FabricFuCapabilityTemplateRef,
                           llvm::ArrayRef<FabricFuOperationHandshakeBinding>);
};

llvm::Expected<FabricFuHandshakeSelection> makeFuHandshakeSelection(
    const FabricArtifactView &view, FabricFuOccurrenceRef occurrence,
    FabricFuCapabilityTemplateRef capability,
    llvm::ArrayRef<FabricFuOperationHandshakeBinding> operations);

using FabricMemoryHandshakePlacement =
    std::variant<FabricMemoryOperationPortRef, FabricMemoryOperationContextRef>;

struct FabricMemoryHandshakeExternalRoleSource final {
  FabricOrdinal endpoint = 0;

  friend bool operator==(FabricMemoryHandshakeExternalRoleSource lhs,
                         FabricMemoryHandshakeExternalRoleSource rhs) {
    return lhs.endpoint == rhs.endpoint;
  }
};

struct FabricMemoryHandshakeInternalRoleSource final {
  FabricOrdinal connection = 0;

  friend bool operator==(FabricMemoryHandshakeInternalRoleSource lhs,
                         FabricMemoryHandshakeInternalRoleSource rhs) {
    return lhs.connection == rhs.connection;
  }
};

using FabricMemoryHandshakeRoleSource =
    std::variant<FabricMemoryHandshakeExternalRoleSource,
                 FabricMemoryHandshakeInternalRoleSource>;

struct FabricMemoryHandshakeRoleDestination final {
  std::optional<FabricOrdinal> externalEndpoint;
  std::vector<FabricOrdinal> internalConnections;

  friend bool operator==(const FabricMemoryHandshakeRoleDestination &lhs,
                         const FabricMemoryHandshakeRoleDestination &rhs) {
    return lhs.externalEndpoint == rhs.externalEndpoint &&
           lhs.internalConnections == rhs.internalConnections;
  }
};

/// One exact occurrence-relative memory operation plan. Construction is
/// restricted to the sealed Fabric resolver so a plan cannot name a foreign
/// port, unsupported mask form, or inadmissible use pattern.
class FabricMemoryHandshakeSelection final {
public:
  const FabricMemoryHandshakePlacement &placement() const { return placement_; }
  const FabricMemoryCapabilityAlternativeRef &capability() const {
    return capability_;
  }
  const FabricUsePatternRef &usePattern() const { return usePattern_; }
  ::dataflow::semantics::MemoryMaskForm maskForm() const { return maskForm_; }
  llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleSource>>
  roleSources() const {
    return roleSources_;
  }
  llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleDestination>>
  roleDestinations() const {
    return roleDestinations_;
  }

  friend bool operator==(const FabricMemoryHandshakeSelection &lhs,
                         const FabricMemoryHandshakeSelection &rhs) {
    return lhs.placement_ == rhs.placement_ &&
           lhs.capability_ == rhs.capability_ &&
           lhs.usePattern_ == rhs.usePattern_ &&
           lhs.maskForm_ == rhs.maskForm_ &&
           lhs.roleSources_ == rhs.roleSources_ &&
           lhs.roleDestinations_ == rhs.roleDestinations_;
  }

private:
  FabricMemoryHandshakeSelection(
      FabricMemoryHandshakePlacement placement,
      FabricMemoryCapabilityAlternativeRef capability,
      FabricUsePatternRef usePattern,
      ::dataflow::semantics::MemoryMaskForm maskForm,
      std::vector<std::optional<FabricMemoryHandshakeRoleSource>> roleSources,
      std::vector<std::optional<FabricMemoryHandshakeRoleDestination>>
          roleDestinations)
      : placement_(std::move(placement)), capability_(capability),
        usePattern_(usePattern), maskForm_(maskForm),
        roleSources_(std::move(roleSources)),
        roleDestinations_(std::move(roleDestinations)) {}

  FabricMemoryHandshakePlacement placement_;
  FabricMemoryCapabilityAlternativeRef capability_;
  FabricUsePatternRef usePattern_;
  ::dataflow::semantics::MemoryMaskForm maskForm_;
  std::vector<std::optional<FabricMemoryHandshakeRoleSource>> roleSources_;
  std::vector<std::optional<FabricMemoryHandshakeRoleDestination>>
      roleDestinations_;

  friend llvm::Expected<FabricMemoryHandshakeSelection>
  makeMemoryHandshakeSelection(
      const FabricArtifactView &, FabricMemoryHandshakePlacement,
      FabricMemoryCapabilityAlternativeRef, FabricUsePatternRef,
      ::dataflow::semantics::MemoryMaskForm,
      llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleSource>>,
      llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleDestination>>);
};

llvm::Expected<FabricMemoryHandshakeSelection> makeMemoryHandshakeSelection(
    const FabricArtifactView &view, FabricMemoryHandshakePlacement placement,
    FabricMemoryCapabilityAlternativeRef capability,
    FabricUsePatternRef usePattern,
    ::dataflow::semantics::MemoryMaskForm maskForm,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleSource>> roleSources,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleDestination>>
        roleDestinations);

/// One exact `(resident row, input)` activation and its complete selected
/// crosspoint set. The Fabric resolver validates the relation against the
/// occurrence model; the requester identity remains owned by each traversal's
/// UsePattern and is not an activation-group identity.
struct FabricSwitchHandshakeActivationSelection final {
  FabricSwitchHandshakeActivationKey key;
  std::vector<FabricPhysicalTraversalRef> traversals;

  friend bool operator==(const FabricSwitchHandshakeActivationSelection &lhs,
                         const FabricSwitchHandshakeActivationSelection &rhs) {
    return lhs.key == rhs.key && lhs.traversals == rhs.traversals;
  }
};

/// Exact Mapping-owned choices consumed by the Fabric resolver. Temporal
/// switch traversals occur only inside `switchActivations`; `traversals`
/// carries every other selected physical traversal. Additional owner-specific
/// choices extend this typed record rather than using a bag.
struct FabricHandshakeSelection final {
  std::vector<FabricPhysicalTraversalRef> traversals;
  std::vector<FabricSwitchHandshakeActivationSelection> switchActivations;
  std::vector<FabricFuHandshakeSelection> fuCapabilities;
  std::vector<FabricMemoryHandshakeSelection> memoryOperations;
};

namespace detail {
class HandshakeOwnerModelBuilder;
class HandshakeOwnerModelFactory;
struct HandshakeOwnerModelStorage;

enum class HandshakeFragmentSelectorKind : std::uint8_t {
  Always,
  AnyTraversal,
  AllTraversals,
  FuCapability,
  FuOperationCase,
  FuOperationInputActive,
  FuOperationResultActive,
  MemoryOperationPlan,
  AnySwitchActivationTraversal,
  ExactSwitchActivationTraversal,
};

struct HandshakeFuOperationSelector final {
  FabricFuOccurrenceNodeRef operation;
  ::dataflow::OperationSchemaId schema =
      ::dataflow::OperationSchemaId::ArithAddI;
  std::uint32_t caseOrdinal = 0;
  std::uint64_t physicalPortOrdinal = 0;
};

/// Selection metadata is implementation-only. It remains separate from the
/// semantic owner graph so the sealed public view cannot mutate activation.
struct HandshakeFragmentSelector final {
  HandshakeFragmentSelectorKind kind = HandshakeFragmentSelectorKind::Always;
  std::vector<FabricPhysicalTraversalRef> traversalWitnesses;
  std::optional<FabricFuOccurrenceRef> fuOccurrence;
  std::optional<FabricFuCapabilityTemplateRef> fuCapability;
  std::optional<HandshakeFuOperationSelector> fuOperation;
  std::optional<FabricMemoryCapabilityAlternativeRef> memoryCapability;
  std::optional<FabricUsePatternRef> memoryUsePattern;
  std::optional<::dataflow::semantics::MemoryMaskForm> memoryMaskForm;
  std::vector<::dataflow::semantics::ServiceValueRole>
      requiredExternalMemoryInputRoles;
  std::vector<::dataflow::semantics::ServiceValueRole>
      requiredExternalMemoryOutputRoles;
  std::optional<FabricSwitchHandshakeActivationKey> switchActivation;
  std::optional<std::uint32_t> exclusiveGroup;
};
} // namespace detail

class HandshakeOwnerModel final {
public:
  HandshakeOwnerModel(const HandshakeOwnerModel &) = default;
  HandshakeOwnerModel(HandshakeOwnerModel &&) noexcept = default;
  HandshakeOwnerModel &operator=(const HandshakeOwnerModel &) = default;
  HandshakeOwnerModel &operator=(HandshakeOwnerModel &&) noexcept = default;

  const FabricHandshakeOwner &owner() const;

  std::uint32_t nodeCount() const;
  HandshakeOwnerNode node(std::uint32_t ordinal) const;
  std::uint32_t arcCount() const;
  HandshakeOwnerArc arc(std::uint32_t ordinal) const;
  std::uint32_t fragmentCount() const;
  HandshakeActivationFragment fragment(std::uint32_t ordinal) const;
  std::uint32_t fragmentContributionCount() const;
  std::uint32_t fragmentContributionOrdinal(std::uint32_t ordinal) const;
  std::uint32_t traversalWitnessCount() const;
  FabricPhysicalTraversalRef traversalWitness(std::uint32_t ordinal) const;

  std::optional<std::uint32_t>
  nodeForSignal(const HandshakeSignalRef &signal) const;

private:
  explicit HandshakeOwnerModel(
      std::shared_ptr<const detail::HandshakeOwnerModelStorage> storage)
      : storage_(std::move(storage)) {}

  detail::HandshakeFragmentSelector
  fragmentSelector(std::uint32_t ordinal) const;

  std::shared_ptr<const detail::HandshakeOwnerModelStorage> storage_;

  friend class detail::HandshakeOwnerModelBuilder;
  friend class detail::HandshakeOwnerModelFactory;
  friend llvm::Expected<class ResolvedHandshakeActivation>
  resolveSelectedHandshake(const HandshakeOwnerModel &,
                           const FabricHandshakeSelection &);
  friend llvm::Expected<std::vector<HandshakeDependencyArc>>
  deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &);
};

struct FabricHandshakeContextStatistics final {
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t ownerCount = 0;
  std::uint64_t structuralTemplateCount = 0;
  std::uint64_t bindingInstanceCount = 0;
  std::uint64_t structuralNodeCount = 0;
  std::uint64_t structuralArcCount = 0;
  std::uint64_t structuralFragmentCount = 0;
  std::uint64_t unconditionalArcCount = 0;
  std::uint64_t nodeCount = 0;
  std::uint64_t arcCount = 0;
  std::uint64_t fragmentCount = 0;
};

/// Immutable Fabric-only compilation of shared structural handshake templates
/// and their exact occurrence and row bindings. The complete Fabric identity
/// and algorithm key make reuse explicit and bounded by the lifetime of the
/// owning invocation context.
class FabricHandshakeContext final {
public:
  FabricHandshakeContext(const FabricHandshakeContext &) = default;
  FabricHandshakeContext(FabricHandshakeContext &&) noexcept = default;
  FabricHandshakeContext &operator=(const FabricHandshakeContext &) = default;
  FabricHandshakeContext &
  operator=(FabricHandshakeContext &&) noexcept = default;

  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  const std::array<std::uint8_t, 32> &key() const { return key_; }
  llvm::ArrayRef<HandshakeOwnerModel> ownerModels() const { return *models_; }
  llvm::ArrayRef<HandshakeDependencyArc> unconditionalDependencyArcs() const {
    return *unconditionalArcs_;
  }
  const FabricHandshakeContextStatistics &statistics() const {
    return statistics_;
  }

private:
  FabricHandshakeContext(
      ArtifactIdentity fabricIdentity, std::array<std::uint8_t, 32> key,
      std::shared_ptr<const std::vector<HandshakeOwnerModel>> models,
      std::shared_ptr<const std::vector<HandshakeDependencyArc>>
          unconditionalArcs,
      FabricHandshakeContextStatistics statistics)
      : fabricIdentity_(std::move(fabricIdentity)), key_(key),
        models_(std::move(models)),
        unconditionalArcs_(std::move(unconditionalArcs)),
        statistics_(statistics) {}

  ArtifactIdentity fabricIdentity_;
  std::array<std::uint8_t, 32> key_{};
  std::shared_ptr<const std::vector<HandshakeOwnerModel>> models_;
  std::shared_ptr<const std::vector<HandshakeDependencyArc>> unconditionalArcs_;
  FabricHandshakeContextStatistics statistics_;

  friend llvm::Expected<FabricHandshakeContext>
  buildFabricHandshakeContext(const FabricArtifactView &);
};

class ResolvedHandshakeActivation final {
public:
  llvm::ArrayRef<std::uint32_t> fragmentOrdinals() const {
    return fragmentOrdinals_;
  }
  llvm::ArrayRef<std::uint32_t> arcOrdinals() const { return arcOrdinals_; }

private:
  std::vector<std::uint32_t> fragmentOrdinals_;
  std::vector<std::uint32_t> arcOrdinals_;

  friend llvm::Expected<ResolvedHandshakeActivation>
  resolveSelectedHandshake(const HandshakeOwnerModel &,
                           const FabricHandshakeSelection &);
};

/// Compiles every complete logical owner model in canonical owner order. Equal
/// FU and Memory definitions and equal rows of one switch share immutable
/// structural storage while retaining exact physical bindings.
llvm::Expected<std::vector<HandshakeOwnerModel>>
compileHandshakeOwnerModels(const FabricArtifactView &view);

llvm::Expected<FabricHandshakeContext>
buildFabricHandshakeContext(const FabricArtifactView &view);

llvm::Error
revalidateFabricHandshakeContext(const FabricHandshakeContext &context,
                                 const FabricArtifactView &view);

/// Resolves one owner's exact active fragments. Facts belonging to other
/// owners are ignored; stale or contradictory facts for this owner reject.
llvm::Expected<ResolvedHandshakeActivation>
resolveSelectedHandshake(const HandshakeOwnerModel &model,
                         const FabricHandshakeSelection &selection);

/// Projects the immutable unconditional boundary closure plus the owner-local
/// fragments named by one exact Mapping selection, flattens shared boundary
/// signals and active junctions, and rejects a selected combinational cycle.
/// The active graph owns no persistent identity or mutable candidate state.
llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection);

llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    const FabricHandshakeContext &context);

/// Derives exact reachability between the requested boundary signals under one
/// selected configuration. Owner-local junctions remain private, and an
/// endpoint with no active dependency simply contributes no relation. The
/// same selected graph is first required to be combinationally acyclic.
llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveSelectedHandshakeReachability(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeSignalRef> terminals);

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveSelectedHandshakeReachability(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeSignalRef> terminals,
    const FabricHandshakeContext &context);

/// Derives the root-complete boundary relation that is present in every legal
/// configured view. Internal junctions never escape this projection.
llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &view);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H
