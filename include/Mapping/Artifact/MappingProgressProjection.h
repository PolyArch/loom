#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/Artifact/SystemPresburger.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::mapping {

enum class MappingProgressClosureKind : std::uint8_t {
  ProvenNoClosedWaitSet,
  ProvenClosedWaitSet,
  ProofNotEstablished,
};

/// Exact reason selected by the shared progress kernel. This is derived
/// diagnostic state, not a persisted proof label or another legality owner.
enum class MappingProgressClosureReason : std::uint8_t {
  None,
  CyclicDataflowBasis,
  MissingDurableBoundary,
  ActivationCapacityExceeded,
  FixedPriorityStarvation,
  PossibleWaitCycle,
  FiniteBufferRecurrenceNotEstablished,
  ClosedBufferDependencyCycle,
  BufferDependencyNotEstablished,
  ReconvergentCapacityShortfall,
  ReconvergentCapacityNotEstablished,
};

enum class MappingProgressWaitNodeKind : std::uint8_t {
  Active,
  Pending,
};

/// One node in a deterministic possible wait-cycle witness. Activation and
/// capacity-cell ordinals address the exact ephemeral projection consumed by
/// the shared kernel; they are diagnostics, not persistent artifact keys.
struct MappingProgressWaitCycleNode final {
  std::uint64_t activationGroupOrdinal = 0;
  MappingProgressWaitNodeKind kind = MappingProgressWaitNodeKind::Active;
  std::vector<std::uint64_t> activationOrdinals;
  std::vector<std::uint64_t> capacityCellOrdinals;
  std::vector<std::uint32_t> triggerEventOrdinals;
  std::vector<std::uint32_t> causalReleaseEventOrdinals;
};

/// The queue partition a static wait fact refers to. A strict FIFO owner has
/// exactly one Global class coupling every resident net; a per-tag virtual
/// channel owner has one PhysicalTag class per selected resident tag value,
/// named by the exact tag bit value.
enum class MappingStaticQueueClassKind : std::uint8_t {
  Global,
  PhysicalTag,
};

struct MappingStaticQueueClass final {
  MappingStaticQueueClassKind kind = MappingStaticQueueClassKind::Global;
  llvm::APInt tagValue = llvm::APInt(1, 0);

  friend bool operator==(const MappingStaticQueueClass &lhs,
                         const MappingStaticQueueClass &rhs) {
    return lhs.kind == rhs.kind && lhs.tagValue == rhs.tagValue;
  }
  friend bool operator!=(const MappingStaticQueueClass &lhs,
                         const MappingStaticQueueClass &rhs) {
    return !(lhs == rhs);
  }
};

/// One physical queue of one selected Buffered FIFO occurrence in the static
/// wait-for graph.
struct MappingStorageQueueProgressNode final {
  ::loom::fabric::FabricFifoOccurrenceRef owner;
  MappingStaticQueueClass queueClass;

  friend bool operator==(const MappingStorageQueueProgressNode &lhs,
                         const MappingStorageQueueProgressNode &rhs) {
    return lhs.owner == rhs.owner && lhs.queueClass == rhs.queueClass;
  }
  friend bool operator!=(const MappingStorageQueueProgressNode &lhs,
                         const MappingStorageQueueProgressNode &rhs) {
    return !(lhs == rhs);
  }
};

/// One logical operand queue of one temporal PE FU in the static wait-for
/// graph.
struct MappingOperandQueueProgressNode final {
  ::fabric::LogicalOperandQueueKey queue;
  ::loom::fabric::FabricFuOccurrenceRef fu;

  friend bool operator==(const MappingOperandQueueProgressNode &lhs,
                         const MappingOperandQueueProgressNode &rhs) {
    return lhs.queue == rhs.queue && lhs.fu == rhs.fu;
  }
  friend bool operator!=(const MappingOperandQueueProgressNode &lhs,
                         const MappingOperandQueueProgressNode &rhs) {
    return !(lhs == rhs);
  }
};

using MappingStaticWaitNode =
    std::variant<MappingStorageQueueProgressNode, ::dataflow::ActorRef,
                 MappingOperandQueueProgressNode>;

/// The mandatory conjunctive wait fact one static buffer-dependency edge
/// quotes. The edge direction is wait-for: the source cannot complete its
/// pending transition until the destination makes progress.
enum class MappingBufferDependencyEdgeKind : std::uint8_t {
  /// A queue class head cannot continue into the downstream storage queue,
  /// whose shared slot pool may be full. Capacity-carrying edge: a cycle that
  /// contains one is proven only together with a capacity proof.
  DownstreamCapacity,
  /// A queue class head reached the route terminal but the consumer actor has
  /// an all-input join on it, or an unbuffered producer's release is joined by
  /// its consumer directly.
  ActorInputJoin,
  /// A producer actor's causal release of an output token waits on the first
  /// buffered storage of that output route accepting it.
  OutputCausalRelease,
  /// An actor firing's input wait joins at the exact temporal operand queue
  /// head owner.
  OperandQueueOwner,
};

/// One edge of the static buffer-dependency graph. The logical net ordinal and
/// the route anchor are witness diagnostics; the wait fact is the node pair
/// and the kind.
struct MappingBufferDependencyEdge final {
  MappingStaticWaitNode from;
  MappingStaticWaitNode to;
  MappingBufferDependencyEdgeKind kind =
      MappingBufferDependencyEdgeKind::ActorInputJoin;
  std::optional<std::uint64_t> logicalNetOrdinal;
  std::optional<::loom::fabric::FabricPhysicalTraversalRef> routeAnchor;
};

struct MappingProgressClosure final {
  MappingProgressClosureKind kind =
      MappingProgressClosureKind::ProofNotEstablished;
  MappingProgressClosureReason reason =
      MappingProgressClosureReason::CyclicDataflowBasis;
  std::vector<MappingProgressWaitCycleNode> possibleWaitCycle;
  /// The exact static wait-for component when the reason is
  /// ClosedBufferDependencyCycle: the closed strongly connected component of
  /// the buffer-dependency graph, in canonical node order.
  std::vector<MappingStaticWaitNode> bufferDependencyCycle;
  /// Exact shortfall of the selected proven capacity witness. Zero for a
  /// non-capacity reason and for ProofNotEstablished.
  std::uint64_t capacityShortfall = 0;
  /// Number of distinct selected physical traversals anchoring the chosen
  /// closed or unestablished witness.
  std::uint64_t routeAnchorCount = 0;
};

/// The one normalized Mapping objective projection derived from a closure.
/// A closure selects one deterministic witness, so its hard/debt count is
/// either zero or one. QoR consumers never reinterpret closure reasons.
struct MappingProgressObjectiveProjection final {
  std::uint64_t hardViolationCount = 0;
  std::uint64_t proofDebtWitnessCount = 0;
  std::uint64_t capacityShortfall = 0;
  std::uint64_t routeAnchorCount = 0;
};

enum class MappingDataflowProgressBasisKind : std::uint8_t {
  Acyclic,
  InitializedFeedback,
  Cyclic,
};

struct MappingDataflowProgressBasis final {
  MappingDataflowProgressBasisKind kind =
      MappingDataflowProgressBasisKind::Cyclic;
  std::uint64_t coveredActorCount = 0;
  std::uint64_t initializedFeedbackEdgeCount = 0;
  std::vector<::dataflow::ActorRef> residualCycle;
};

enum class MappingResourceGrantPolicyKind : std::uint8_t {
  None,
  FixedPriority,
  RoundRobin,
};

struct MappingResourceProgressUse final {
  std::string physicalOwnerKey;
  std::uint32_t requester = 0;
  MappingResourceGrantPolicyKind grantPolicy =
      MappingResourceGrantPolicyKind::None;
};

struct MappingProgressCapacityCellProjection final {
  std::uint64_t capacity = 0;
  std::uint64_t baselineOccupancy = 0;
};

struct MappingProgressCapacityClaimProjection final {
  std::uint64_t capacityCellOrdinal = 0;
  std::uint64_t amount = 0;
};

struct MappingProgressCausalReleaseProjection final {
  std::vector<::dataflow::EventFamilyKey> alternatives;
};

struct MappingProgressActivationProjection final {
  ExecutionContextKey context;
  ::dataflow::RootThreadLaunchRef relationRoot;
  std::vector<SystemPresburgerCell> relationDomain;
  std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
  std::vector<MappingProgressCapacityClaimProjection> capacityClaims;
  std::vector<MappingProgressCausalReleaseProjection> causalRelease;
  MappingResourceProgressUse arbitration;
};

enum class MappingRouteProgressObligationKind : std::uint8_t {
  DurableBoundaryAfterDivergence,
  FiniteBufferRecurrence,
};

enum class MappingReconvergentCapacityProofKind : std::uint8_t {
  Proven,
  ProofNotEstablished,
};

/// One exact capacity obligation of one selected FIFO shared slot pool under
/// the durable-acceptance transfer subdomain. `queueClasses` names every
/// strict-global or tag-local order class sharing this one physical capacity
/// owner; it never partitions `selectedCapacity`. `minimumLegalCapacity` is
/// the number of distinct producer bindings that can each own one active
/// resident token. `routeAnchors` names the selected traversals from which the
/// obligation is rebuilt. The kind is the proof state and the minimum is
/// present exactly when proven. This value is not a persisted proof label.
struct MappingReconvergentCapacityObligation final {
  ::loom::fabric::FabricFifoOccurrenceRef owner;
  std::vector<MappingStaticQueueClass> queueClasses;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> routeAnchors;
  std::uint64_t selectedCapacity = 0;
  std::optional<std::uint64_t> minimumLegalCapacity;
  MappingReconvergentCapacityProofKind kind =
      MappingReconvergentCapacityProofKind::ProofNotEstablished;
};

/// One physical progress obligation rebuilt from the selected route trees and
/// typed Fabric traversals or sink boundaries. An unestablished durable
/// boundary is a concrete closed-wait witness. An unestablished finite-buffer
/// recurrence remains incomplete because a finite replay cannot prove general
/// queue liveness. This value is not a persisted proof label.
struct MappingRouteProgressObligationProjection final {
  MappingRouteProgressObligationKind kind =
      MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence;
  bool established = false;
};

/// The complete removable input to the shared Mapping progress kernel.
/// Spatial and System roots differ only in how they rebuild these fields.
struct MappingProgressProjection final {
  MappingDataflowProgressBasis basis;
  std::vector<MappingRouteProgressObligationProjection> routeObligations;
  std::vector<MappingProgressCapacityCellProjection> capacityCells;
  std::vector<MappingProgressActivationProjection> resourceActivations;
  /// The static buffer-dependency edge set. Engaged and empty when the mapping
  /// carries no buffer wait facts; disengaged when any queue class, tag
  /// residency, or relation domain is indeterminate, which the kernel reports
  /// as BufferDependencyNotEstablished and never as a proven cycle.
  std::optional<std::vector<MappingBufferDependencyEdge>>
      bufferDependencyEdges = std::vector<MappingBufferDependencyEdge>{};
  /// The per-class capacity obligations of the reconvergence proof. Empty when
  /// the mapping selects no Buffered FIFO queue class.
  std::vector<MappingReconvergentCapacityObligation>
      reconvergentCapacityObligations;
};

/// A non-owning projection consumed synchronously by the progress kernel.
/// The owning projection remains the strict-import adapter and source of truth.
struct MappingProgressProjectionView final {
  const MappingDataflowProgressBasis &basis;
  llvm::ArrayRef<MappingRouteProgressObligationProjection> routeObligations;
  llvm::ArrayRef<MappingProgressCapacityCellProjection> capacityCells;
  llvm::ArrayRef<MappingProgressActivationProjection> resourceActivations;
  const std::optional<std::vector<MappingBufferDependencyEdge>>
      &bufferDependencyEdges;
  llvm::ArrayRef<MappingReconvergentCapacityObligation>
      reconvergentCapacityObligations;
};

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H
