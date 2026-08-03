#ifndef LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H
#define LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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
  ExactOwnerSelection,
};

struct HandshakeActivationFragment final {
  std::uint32_t contributionOffset = 0;
  std::uint32_t contributionCount = 0;
  HandshakeActivationKind activationKind =
      HandshakeActivationKind::ExactOwnerSelection;
  std::uint32_t witnessOffset = 0;
  std::uint32_t witnessCount = 0;
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

  friend bool operator==(const FabricMemoryHandshakeSelection &lhs,
                         const FabricMemoryHandshakeSelection &rhs) {
    return lhs.placement_ == rhs.placement_ &&
           lhs.capability_ == rhs.capability_ &&
           lhs.usePattern_ == rhs.usePattern_ && lhs.maskForm_ == rhs.maskForm_;
  }

private:
  FabricMemoryHandshakeSelection(
      FabricMemoryHandshakePlacement placement,
      FabricMemoryCapabilityAlternativeRef capability,
      FabricUsePatternRef usePattern,
      ::dataflow::semantics::MemoryMaskForm maskForm)
      : placement_(std::move(placement)), capability_(capability),
        usePattern_(usePattern), maskForm_(maskForm) {}

  FabricMemoryHandshakePlacement placement_;
  FabricMemoryCapabilityAlternativeRef capability_;
  FabricUsePatternRef usePattern_;
  ::dataflow::semantics::MemoryMaskForm maskForm_;

  friend llvm::Expected<FabricMemoryHandshakeSelection>
  makeMemoryHandshakeSelection(const FabricArtifactView &,
                               FabricMemoryHandshakePlacement,
                               FabricMemoryCapabilityAlternativeRef,
                               FabricUsePatternRef,
                               ::dataflow::semantics::MemoryMaskForm);
};

llvm::Expected<FabricMemoryHandshakeSelection>
makeMemoryHandshakeSelection(const FabricArtifactView &view,
                             FabricMemoryHandshakePlacement placement,
                             FabricMemoryCapabilityAlternativeRef capability,
                             FabricUsePatternRef usePattern,
                             ::dataflow::semantics::MemoryMaskForm maskForm);

/// Exact Mapping-owned choices consumed by the Fabric resolver. Additional
/// owner-specific choices extend this typed record rather than using a bag.
struct FabricHandshakeSelection final {
  std::vector<FabricPhysicalTraversalRef> traversals;
  std::vector<FabricFuHandshakeSelection> fuCapabilities;
  std::vector<FabricMemoryHandshakeSelection> memoryOperations;
};

namespace detail {
class HandshakeOwnerModelBuilder;

enum class HandshakeFragmentSelectorKind : std::uint8_t {
  Always,
  AnyTraversal,
  AllTraversals,
  FuCapability,
  FuOperationCase,
  FuOperationInputActive,
  FuOperationResultActive,
  MemoryOperationPlan,
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
  std::optional<std::uint32_t> exclusiveGroup;
};
} // namespace detail

class HandshakeOwnerModel final {
public:
  HandshakeOwnerModel(const HandshakeOwnerModel &) = default;
  HandshakeOwnerModel(HandshakeOwnerModel &&) noexcept = default;
  HandshakeOwnerModel &operator=(const HandshakeOwnerModel &) = default;
  HandshakeOwnerModel &operator=(HandshakeOwnerModel &&) noexcept = default;

  const FabricHandshakeOwner &owner() const { return owner_; }
  llvm::ArrayRef<HandshakeOwnerNode> nodes() const { return nodes_; }
  llvm::ArrayRef<HandshakeOwnerArc> arcs() const { return arcs_; }
  llvm::ArrayRef<HandshakeActivationFragment> fragments() const {
    return fragments_;
  }
  llvm::ArrayRef<std::uint32_t> fragmentContributionOrdinals() const {
    return fragmentContributionOrdinals_;
  }
  llvm::ArrayRef<FabricPhysicalTraversalRef> traversalWitnesses() const {
    return traversalWitnesses_;
  }

  std::optional<std::uint32_t>
  nodeForSignal(const HandshakeSignalRef &signal) const;

private:
  explicit HandshakeOwnerModel(FabricHandshakeOwner owner)
      : owner_(std::move(owner)) {}

  FabricHandshakeOwner owner_;
  std::vector<HandshakeOwnerNode> nodes_;
  std::vector<HandshakeOwnerArc> arcs_;
  std::vector<HandshakeActivationFragment> fragments_;
  std::vector<std::uint32_t> fragmentContributionOrdinals_;
  std::vector<FabricPhysicalTraversalRef> traversalWitnesses_;
  std::vector<detail::HandshakeFragmentSelector> fragmentSelectors_;

  friend class detail::HandshakeOwnerModelBuilder;
  friend llvm::Expected<class ResolvedHandshakeActivation>
  resolveSelectedHandshake(const HandshakeOwnerModel &,
                           const FabricHandshakeSelection &);
  friend llvm::Expected<std::vector<HandshakeDependencyArc>>
  deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &);
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

/// Compiles every complete occurrence-level owner model in canonical owner
/// order. The returned models are immutable derived views of `view`.
llvm::Expected<std::vector<HandshakeOwnerModel>>
compileHandshakeOwnerModels(const FabricArtifactView &view);

/// Resolves one owner's exact active fragments. Facts belonging to other
/// owners are ignored; stale or contradictory facts for this owner reject.
llvm::Expected<ResolvedHandshakeActivation>
resolveSelectedHandshake(const HandshakeOwnerModel &model,
                         const FabricHandshakeSelection &selection);

/// Resolves every owner against one exact Mapping selection, flattens shared
/// boundary signals and owner-local junctions, and rejects a selected
/// combinational dependency cycle. The implementation is linear after
/// deterministic arc ordering and owns no persistent graph identity.
llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection);

/// Derives the root-complete boundary relation that is present in every legal
/// configured view. Internal junctions never escape this projection.
llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &view);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICHANDSHAKE_H
