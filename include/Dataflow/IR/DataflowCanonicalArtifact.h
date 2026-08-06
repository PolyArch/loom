#ifndef LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ARTIFACT_H
#define LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowEnums.h"
#include "Dataflow/IR/DataflowInterfaces.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {
class MLIRContext;
class Operation;
} // namespace mlir

namespace dataflow {

class CanonicalDataflowArtifact;
struct FinalizedCanonicalDataflowProjection;
namespace detail {
struct CanonicalLabeling;
} // namespace detail

//===----------------------------------------------------------------------===//
// Read-only entity projections
//
// Each view borrows operations owned by the finalized module the view was
// imported from. A native index is a disposable cache, not a second catalog.
//===----------------------------------------------------------------------===//

struct CanonicalGraphView {
  GraphRef ref;
  mlir::Operation *op = nullptr; // the dataflow.graph definition
};

struct CanonicalActorView {
  ActorRef ref;
  mlir::Operation *op = nullptr; // the actor operation in a graph body
  GraphRef graph;                // the owning graph entity
  CanonicalDataflowActorKind kind = CanonicalDataflowActorKind::Compute;
};

struct CanonicalRootThreadLaunchView {
  RootThreadLaunchRef ref;
  mlir::Operation *op = nullptr;     // the dataflow.thread.launch site
  mlir::Operation *callee = nullptr; // the resolved dataflow.thread definition
};

/// Dataflow-owned logical input signature and may-domain discriminator for one
/// exact root thread launch. Values borrow the imported canonical program.
/// Extents occupy the leading launch-parameter slots in coordinate order.
struct CanonicalRootThreadLogicalDomainView {
  RootThreadLaunchRef launch;
  ThreadDomainKind kind = ThreadDomainKind::DenseRectangular;
  std::uint32_t coordinateRank = 0;
  std::vector<mlir::Value> launchParameters;
};

struct CanonicalStaticGraphLaunchView {
  StaticGraphLaunchRef ref;
  mlir::Operation *op = nullptr; // the dataflow.graph.launch site
  GraphRef callee;               // the resolved launched graph entity
};

struct CanonicalLogicalMemoryRootView {
  LogicalMemoryRootRef ref;
  // The owning dataflow.thread for an imported memory formal (selected by
  // `formalArgIndex`), the dataflow.memory.service operation for an
  // object-scoped service root, or the fresh memref.alloc op for an allocation
  // root. An imported thread formal has no owning-graph entity: its downstream
  // graph role is recovered through the exact graph.launch memory binding.
  mlir::Operation *op = nullptr;
  // Set for an imported thread memory formal: the function-input ordinal, which
  // is also its entry-block-argument index. Absent for an operation-owned root.
  std::optional<unsigned> formalArgIndex;
  // The exact memory-capability type of the root-defining SSA value. This is a
  // borrowed projection of the imported canonical program, not another type
  // catalog.
  mlir::Type type;
};

/// A channel consumer with its exact source_map relation payload. The map is
/// present for a graph stream input and absent for a rank-zero direct receive.
struct ChannelConsumerBinding {
  ChannelConsumerRef consumer;
  std::optional<mlir::AffineMap> sourceMap;
};

/// One exact producer terminal and the payload type of its Canonical Service
/// message-transfer leg. The type borrows the imported canonical program.
struct CanonicalProducerTerminalView {
  CanonicalProducerTerminalRef terminal;
  mlir::Type payloadType;
};

//===----------------------------------------------------------------------===//
// CanonicalDataflowProgramView
//===----------------------------------------------------------------------===//

/// The single Dataflow-owned read-only projection of the five entity kinds and
/// the currently available structural-reference inventories and derived
/// relations in DataflowStructuralRefs.h. Independent import reconstructs the
/// canonical relation graph, verifies every materialized ID, and generates,
/// validates, and resolves the structural references consumers need with no
/// Mapping Artifact. Native lookup indices are disposable caches.
class CanonicalDataflowProgramView {
public:
  /// Reconstruct and verify a finalized program. Rejects any stale, missing,
  /// duplicate, out-of-range, or noncanonical materialized ID, any unresolved
  /// symbol or memory-root relation, and any payload whose Common identity does
  /// not equal `expectedIdentity`.
  static llvm::Expected<CanonicalDataflowProgramView>
  import(mlir::ModuleOp finalizedModule,
         const ::loom::ArtifactIdentity &expectedIdentity,
         const ::loom::CanonicalSemanticBytes &canonicalBytes);

  const ::loom::ArtifactIdentity &identity() const { return identity_; }
  std::uint64_t entityCount() const { return kindOfId_.size(); }

  /// Derives one pointer format from the exact module-owned LLVM DataLayout.
  /// The result is a removable typed projection, not persistent view state.
  llvm::Expected<::loom::PointerLayout>
  pointerLayout(std::uint32_t addressSpace) const;

  // Typed resolution. Each requires the exact artifact identity, so a
  // foreign-artifact or wrong-kind reference is a real runtime rejection.
  llvm::Expected<CanonicalGraphView> resolve(GraphRef ref) const;
  llvm::Expected<CanonicalActorView> resolve(ActorRef ref) const;
  llvm::Expected<CanonicalRootThreadLaunchView>
  resolve(RootThreadLaunchRef ref) const;
  llvm::Expected<CanonicalRootThreadLogicalDomainView>
  projectRootThreadLogicalDomain(RootThreadLaunchRef ref) const;

  /// Project the root launch's complete logical domain for one rooted graph
  /// launch only when the graph invocation is a direct operation in the
  /// thread entry block. Such an invocation executes exactly once for every
  /// root-domain point. A nested invocation returns `std::nullopt`: its exact
  /// conditional or repeated may-domain is not currently published by
  /// Canonical Dataflow.
  llvm::Expected<std::optional<CanonicalRootThreadLogicalDomainView>>
  projectWholeRootedGraphLogicalDomain(RootedGraphLaunchRef ref) const;
  llvm::Expected<CanonicalStaticGraphLaunchView>
  resolve(StaticGraphLaunchRef ref) const;
  llvm::Expected<CanonicalLogicalMemoryRootView>
  resolve(LogicalMemoryRootRef ref) const;

  // Canonical-order enumerations for direct consumers such as DFG-sim.
  llvm::ArrayRef<CanonicalGraphView> graphs() const { return graphs_; }
  llvm::ArrayRef<CanonicalActorView> actors() const { return actors_; }
  llvm::ArrayRef<CanonicalRootThreadLaunchView> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<CanonicalStaticGraphLaunchView> staticGraphLaunches() const {
    return staticGraphLaunches_;
  }
  llvm::ArrayRef<CanonicalLogicalMemoryRootView> logicalMemoryRoots() const {
    return logicalMemoryRoots_;
  }

  //== Closed structural-reference generation, validation, and resolution ==//

  /// Enumerate every rooted graph launch lazily, composing each root/static
  /// pair on demand from the compact grouped thread-owner inventory. No eager
  /// roots-by-launches product is ever materialized.
  void forEachRootedGraphLaunch(
      llvm::function_ref<void(RootedGraphLaunchRef)>) const;

  /// Enumerate the complete Dataflow-owned memory-exposure inventory in
  /// rooted-launch order, then graph memory-result ordinal. The inventory is
  /// derived from the same sealed static-launch relation used by
  /// resolveExposure; consumers must not reconstruct it from graph IR.
  void forEachMemoryExposure(llvm::function_ref<void(MemoryExposureRef)>) const;

  /// Visit the complete producer-terminal inventory reachable from one exact
  /// root thread launch in canonical reference order. This includes root and
  /// graph ABI boundaries plus channel producers. Memory capability crossings
  /// remain MemoryExposureRef values and are deliberately absent.
  llvm::Error forEachProducerTerminal(
      RootThreadLaunchRef root,
      llvm::function_ref<llvm::Error(const CanonicalProducerTerminalView &)>)
      const;

  /// Visit every addressed-memory or fence actor in each graph launch rooted
  /// at `root`. The contextual reference is Dataflow-owned; callers do not
  /// reconstruct actor membership by walking MLIR.
  llvm::Error forEachContextualServiceActor(
      RootThreadLaunchRef root,
      llvm::function_ref<llvm::Error(ContextualActorRef)>) const;

  /// Validate a rooted graph launch: the root launch and the static graph
  /// launch must resolve, and the graph-launch site must belong to the thread
  /// definition reached from the root launch. Returns the launched graph.
  llvm::Expected<GraphRef> resolve(RootedGraphLaunchRef ref) const;

  /// Validate a token-plane endpoint. Rejects an out-of-range ordinal and any
  /// memory-capability operand or result, which is never a token endpoint.
  llvm::Error validate(const CanonicalGraphProducerEndpointRef &endpoint) const;
  llvm::Error validate(const CanonicalGraphConsumerEndpointRef &endpoint) const;

  /// Resolve the exact semantic token type owned by a graph endpoint. The
  /// returned MLIR type borrows the imported canonical program and is a
  /// read-only projection, not a second type catalog.
  llvm::Expected<mlir::Type>
  tokenType(const CanonicalGraphProducerEndpointRef &endpoint) const;
  llvm::Expected<mlir::Type>
  tokenType(const CanonicalGraphConsumerEndpointRef &endpoint) const;

  /// The exact intra-graph software edge relation, generated once at import.
  /// A consumer endpoint has one producer (a near-constant def lookup); a
  /// producer endpoint has its complete canonically sorted consumer set (a
  /// range into the prebuilt edge inventory). Both sides are token-plane only.
  llvm::Expected<CanonicalGraphProducerEndpointRef>
  graphProducer(const CanonicalGraphConsumerEndpointRef &consumer) const;
  llvm::Expected<llvm::ArrayRef<CanonicalGraphConsumerEndpointRef>>
  graphConsumers(const CanonicalGraphProducerEndpointRef &producer) const;

  /// Visits the complete imported intra-graph edge relation in deterministic
  /// producer-key and canonical consumer order. The relation is the same
  /// prebuilt owner used by graphProducer/graphConsumers; callers do not walk
  /// SSA or reconstruct endpoint inventories.
  llvm::Error forEachGraphEdge(
      llvm::function_ref<llvm::Error(const CanonicalGraphProducerEndpointRef &,
                                     const CanonicalGraphConsumerEndpointRef &)>
          callback) const;

  /// Validate a one-message boundary transfer's ordinal against its launch.
  llvm::Error validate(const RootThreadBoundaryTransferRef &transfer) const;
  llvm::Error validate(const GraphLaunchBoundaryTransferRef &transfer) const;

  /// The complete channel consumer relation for one producer, a range into the
  /// prebuilt channel inventory: the complete, non-empty, canonically sorted
  /// consumer set, each with its exact source_map payload (present for a graph
  /// stream input). No per-query topology recomputation.
  llvm::Expected<llvm::ArrayRef<ChannelConsumerBinding>>
  channelConsumers(const ChannelProducerRef &producer) const;

  /// The transfer relation for a producer terminal, delivered by callback: a
  /// boundary source yields its one paired sink; a channel producer yields its
  /// complete prebuilt multicast sink set. No caller scratch and no per-query
  /// allocation.
  llvm::Error
  pairedSinks(const CanonicalProducerTerminalRef &producer,
              llvm::function_ref<void(const CanonicalSinkTerminalRef &)>) const;

  /// Validate a transfer terminal and a static transfer event, rejecting a
  /// foreign-artifact or wrong-owner terminal.
  llvm::Error validate(const CanonicalProducerTerminalRef &terminal) const;
  llvm::Error validate(const CanonicalSinkTerminalRef &terminal) const;
  llvm::Error validate(const StaticTransferEventRef &event) const;

  /// The canonical root-local view inventory of one logical memory root: a
  /// range into the prebuilt view inventory (every admitted root-preserving
  /// view relation), not a per-query scan.
  llvm::Expected<llvm::ArrayRef<LogicalMemoryViewRef>>
  views(LogicalMemoryRootRef root) const;

  /// Resolve the exact memory-capability type owned by one logical root or
  /// root-preserving view. The returned type borrows the imported program.
  llvm::Expected<mlir::Type>
  memoryType(const LogicalMemoryRootOrViewRef &memory) const;

  /// Resolve the exact statically derivable backing extent in bytes. Dynamic,
  /// unranked, or non-identity-layout memories return `std::nullopt`; malformed
  /// and foreign references are errors. A root-preserving view inherits its
  /// root's backing extent even when a memref.cast hides static dimensions.
  llvm::Expected<std::optional<std::uint64_t>>
  staticMemoryByteExtent(const LogicalMemoryRootOrViewRef &memory) const;

  /// Resolve a launch-contextual graph memory result through the graph return
  /// and the exact graph-launch memory-input binding to the upstream logical
  /// root or root-preserving view. A view receives no entity of its own.
  llvm::Expected<LogicalMemoryRootOrViewRef>
  resolveExposure(MemoryExposureRef ref) const;

  /// Validate that a contextual actor belongs to the graph its rooted launch
  /// calls.
  llvm::Error validate(ContextualActorRef ref) const;

  /// Resolve an addressed actor's memory capability through its exact rooted
  /// graph launch. The result is the Dataflow-owned logical root or
  /// root-preserving view selected at that static launch site. The root launch
  /// validates ownership but never multiplies the static composition table.
  llvm::Expected<LogicalMemoryRootOrViewRef>
  resolveAddressedMemory(ContextualActorRef ref) const;

  /// Validate an actor as a `dataflow.fence` family reference.
  llvm::Expected<FenceActorFamilyRef> asFenceFamily(ActorRef ref) const;

  /// The service member of a contextual actor: only a fence actor or one of the
  /// addressed canonical memory actors (classified through the exact Dataflow
  /// service/access schema) is a service member; any other actor is rejected. A
  /// MemoryExposureRef is structurally not a ContextualActorRef and so cannot
  /// even be presented here.
  llvm::Expected<ServiceMemberRef>
  serviceMemberFor(ContextualActorRef ref) const;

  /// The singleton MessageTransfer member of one transfer obligation. Every
  /// valid producer terminal -- a boundary transfer or a channel multicast
  /// producer -- has this one member.
  llvm::Expected<ServiceMemberRef>
  messageTransferMember(const CanonicalProducerTerminalRef &terminal) const;

private:
  explicit CanonicalDataflowProgramView(::loom::ArtifactIdentity identity)
      : identity_(identity) {}

  // Finalization builds and validates the complete closed relation set from the
  // labeling it already computed, so it needs the same private assembly the
  // importer uses.
  friend llvm::Expected<CanonicalDataflowArtifact>
      finalizeCanonicalDataflow(mlir::ModuleOp);
  friend llvm::Expected<FinalizedCanonicalDataflowProjection>
      finalizeCanonicalDataflowWithTrackedStaticGraphLaunches(
          mlir::ModuleOp, llvm::ArrayRef<mlir::Operation *>);

  // Assemble the typed ID maps and every closed structural inventory from an
  // already-computed canonical labeling. The importer calls this after
  // verifying materialized IDs; finalization calls it to validate structural
  // relations before publishing, without recomputing the labeling. Fails on any
  // unresolved owner, symbol, memory-root, or exposure relation.
  static llvm::Expected<CanonicalDataflowProgramView>
  buildView(mlir::ModuleOp module, const ::loom::ArtifactIdentity &identity,
            const detail::CanonicalLabeling &labeling);

  llvm::Expected<std::uint64_t>
  requireKind(const ::loom::ArtifactIdentity &artifact, std::uint64_t id,
              CanonicalDataflowEntityKind kind) const;

  // The collision-free typed key of a channel producer (kind, root slot, static
  // slot, ordinal), validating the full rooted context with O(1) slot and
  // direct count checks. No MLIR walk, no pointer key, no lossy packing.
  using ChannelProducerKey = std::tuple<int, unsigned, unsigned, unsigned>;
  llvm::Expected<ChannelProducerKey>
  channelProducerKey(const ChannelProducerRef &producer) const;

  // Generate the closed structural inventories once, in a bounded number of
  // linear passes over the finalized module, after the entity maps are built.
  llvm::Error buildStructuralInventories(
      mlir::ModuleOp module,
      llvm::ArrayRef<mlir::Operation *> canonicalOperationOrder);

  // Resolve one admitted memory SSA value to its role from the prebuilt cache,
  // or fail. Composition seeds every admitted value, so a query never
  // rederives.
  llvm::Expected<LogicalMemoryRootOrViewRef>
  roleOfValue(mlir::Value value) const;

  ::loom::ArtifactIdentity identity_;
  mlir::ModuleOp module_;
  // Global EntityId -> kind, plus the slot within that kind's vector.
  std::vector<CanonicalDataflowEntityKind> kindOfId_;
  std::vector<std::size_t> slotOfId_;
  std::vector<CanonicalGraphView> graphs_;
  std::vector<CanonicalActorView> actors_;
  std::vector<CanonicalRootThreadLaunchView> rootThreadLaunches_;
  std::vector<CanonicalStaticGraphLaunchView> staticGraphLaunches_;
  std::vector<CanonicalLogicalMemoryRootView> logicalMemoryRoots_;
  // Transient native import caches (owner op/value -> dense EntityId). These
  // are disposable; every persistent typed reference and hot lookup resolves
  // through entity slots and owner-local offsets, never a container pointer
  // key.
  llvm::DenseMap<mlir::Operation *, std::uint64_t> graphIdByOp_;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> actorIdByOp_;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> rootThreadLaunchIdByOp_;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> staticGraphLaunchIdByOp_;
  llvm::DenseMap<mlir::Value, std::uint64_t> memoryRootIdByValue_;

  // Compact thread-owner slots: one slot per distinct thread definition reached
  // by a root launch or static graph launch. Rooted-launch validation is an
  // O(1) slot comparison; enumeration composes root/static pairs lazily.
  std::vector<mlir::Operation *> threadDefs_;
  llvm::DenseMap<mlir::Operation *, unsigned> threadSlotOf_;
  std::vector<unsigned> rootCalleeThreadSlot_;  // by root-launch slot
  std::vector<unsigned> staticOwnerThreadSlot_; // by static-launch slot
  std::vector<llvm::SmallVector<unsigned, 2>> rootsByThreadSlot_;
  std::vector<llvm::SmallVector<unsigned, 2>> staticsByThreadSlot_;

  // Per-thread channel endpoint counts. Hot queries need only the ordinal
  // cardinality; the op-to-ordinal maps are transient import state, so no
  // endpoint pointer inventory is retained.
  std::vector<unsigned> threadSendCount_;
  std::vector<unsigned> threadReceiveCount_;

  // Intra-graph software edge inventory: one flat consumer set addressed by a
  // collision-free typed producer key (kind, owner slot, ordinal). The map is a
  // disposable native cache; there is no lossy dense packing.
  std::vector<CanonicalGraphConsumerEndpointRef> graphEdgeConsumers_;
  std::map<std::tuple<int, unsigned, unsigned>, std::pair<unsigned, unsigned>>
      graphEdgeRange_;

  // Channel multicast inventory: parallel flat consumer bindings and sink
  // terminals addressed by a collision-free typed producer key.
  std::vector<ChannelConsumerBinding> channelBindings_;
  std::vector<CanonicalSinkTerminalRef> channelSinks_;
  std::map<ChannelProducerKey, std::pair<unsigned, unsigned>> channelRange_;
  std::map<ChannelProducerKey, mlir::Type> channelProducerPayloadType_;

  // Static memory composition. Every admitted memory SSA value resolves to a
  // role in `roleTable_` (a disposable value cache); the flat root-local view
  // inventory has per-root ranges; each static graph-launch memory result has a
  // resolved exposure role index. Composition fails finalization on any
  // unresolved relation, so there is never an absent-but-published exposure.
  std::vector<LogicalMemoryRootOrViewRef> roleTable_;
  llvm::DenseMap<mlir::Value, unsigned> roleIndexOf_;
  std::vector<LogicalMemoryViewRef> views_;
  std::vector<mlir::Type> viewTypes_;
  std::vector<std::optional<std::uint64_t>> rootStaticByteExtents_;
  std::vector<std::pair<unsigned, unsigned>> viewsByRootSlot_;
  std::vector<llvm::SmallVector<unsigned, 1>> exposureByStaticSlot_;

  // Addressed actor memory composition, factorized by static graph-launch
  // site. Each site owns one sorted actor-slot range and parallel role-index
  // range. Root launches validate context but do not duplicate this table.
  std::vector<std::pair<unsigned, unsigned>> addressedMemoryRangeByStaticSlot_;
  std::vector<unsigned> addressedMemoryActorSlots_;
  std::vector<unsigned> addressedMemoryRoleIndices_;
  std::vector<llvm::SmallVector<unsigned, 2>> serviceActorSlotsByGraphSlot_;
};

//===----------------------------------------------------------------------===//
// CanonicalDataflowArtifact
//===----------------------------------------------------------------------===//

/// A finalized Canonical Dataflow Artifact: the exact SHA-256 v1 identity, the
/// canonical module carrying materialized entity IDs, and the family-owned
/// canonical bytes the Common finalizer hashed.
class CanonicalDataflowArtifact {
public:
  CanonicalDataflowArtifact(const CanonicalDataflowArtifact &) = delete;
  CanonicalDataflowArtifact(CanonicalDataflowArtifact &&) = default;
  CanonicalDataflowArtifact &
  operator=(const CanonicalDataflowArtifact &) = delete;
  CanonicalDataflowArtifact &operator=(CanonicalDataflowArtifact &&) = default;

  const ::loom::ArtifactIdentity &identity() const { return identity_; }
  mlir::ModuleOp module() const { return module_.get(); }
  const ::loom::CanonicalSemanticBytes &canonicalBytes() const {
    return bytes_;
  }

  /// Returns the read-only projection sealed by finalization or strict import.
  /// This disposable native cache is derived from the canonical bytes and may
  /// never replace them as the persistent authority.
  llvm::Expected<CanonicalDataflowProgramView> view() const { return view_; }

private:
  friend llvm::Expected<CanonicalDataflowArtifact>
      finalizeCanonicalDataflow(mlir::ModuleOp);
  friend llvm::Expected<FinalizedCanonicalDataflowProjection>
      finalizeCanonicalDataflowWithTrackedStaticGraphLaunches(
          mlir::ModuleOp, llvm::ArrayRef<mlir::Operation *>);
  friend llvm::Expected<CanonicalDataflowArtifact>
  importCanonicalDataflow(const ::loom::ArtifactIdentity &,
                          const ::loom::CanonicalSemanticBytes &);
  friend llvm::Expected<CanonicalDataflowArtifact>
  importCanonicalDataflow(const ::loom::ArtifactRootReference &,
                          const ::loom::ArtifactStore &);

  CanonicalDataflowArtifact(
      ::loom::ArtifactIdentity identity,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      ::loom::CanonicalSemanticBytes bytes, CanonicalDataflowProgramView view,
      std::unique_ptr<mlir::MLIRContext> context = nullptr)
      : identity_(identity), context_(std::move(context)),
        module_(std::move(module)), bytes_(std::move(bytes)),
        view_(std::move(view)) {}

  ::loom::ArtifactIdentity identity_;
  // Declaration order makes the module release before its owning context.
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  ::loom::CanonicalSemanticBytes bytes_;
  CanonicalDataflowProgramView view_;
};

/// Finalizer-owned ephemeral correspondence for caller-selected static graph
/// launch operations. The references belong to `artifact`; the correspondence
/// is not serialized and can always be reconstructed by lowering the exact
/// owner again.
struct FinalizedCanonicalDataflowProjection final {
  CanonicalDataflowArtifact artifact;
  std::vector<StaticGraphLaunchRef> trackedStaticGraphLaunches;
};

/// Failure-atomic finalization. Operates on a private clone of `source`, strips
/// every preexisting derived ID, validates the whole canonical program,
/// constructs the canonical relation graph, assigns dense artifact-global IDs
/// by canonical slot, materializes them, and publishes only the complete valid
/// artifact framed by the Common finalizer.
llvm::Expected<CanonicalDataflowArtifact>
finalizeCanonicalDataflow(mlir::ModuleOp source);

/// Finalizes one program while carrying an ordered set of source launch
/// operations through the private clone and canonical labeling transaction.
/// Every tracked operation must be a live dataflow.graph.launch in `source`.
llvm::Expected<FinalizedCanonicalDataflowProjection>
finalizeCanonicalDataflowWithTrackedStaticGraphLaunches(
    mlir::ModuleOp source,
    llvm::ArrayRef<mlir::Operation *> trackedStaticGraphLaunches);

/// Strictly imports one exact stored canonical Dataflow payload. The family
/// importer independently rebuilds canonical labels and materialized entity
/// IDs, then requires an exact byte-for-byte writer round trip.
llvm::Expected<CanonicalDataflowArtifact>
importCanonicalDataflow(const ::loom::ArtifactIdentity &identity,
                        const ::loom::CanonicalSemanticBytes &canonicalBytes);

llvm::Expected<::loom::ArtifactRootReference>
publishCanonicalDataflow(const CanonicalDataflowArtifact &candidate,
                         const ::loom::ArtifactStore &store);

llvm::Expected<CanonicalDataflowArtifact>
importCanonicalDataflow(const ::loom::ArtifactRootReference &reference,
                        const ::loom::ArtifactStore &store);

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ARTIFACT_H
