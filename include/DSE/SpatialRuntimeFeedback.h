#ifndef LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
#define LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Fabric/IR/FabricEnums.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/CgraClosedWaitCertificate.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::evaluation::models {
class VerifiedCgraClosedWaitEvidence;
}

namespace loom::dse {

enum class SpatialFifoRuntimeFeedbackDisposition : std::uint8_t {
  Exact,
  ProofNotEstablished,
  Unsupported,
};

enum class SpatialFifoRuntimeFeedbackReason : std::uint8_t {
  ExactFullFifoCycle,
  MissingWaitCycle,
  MissingCanonicalFifo,
  AmbiguousFifo,
  StorageNotFull,
  MissingCausalReleaseContext,
  ExactCrossTagGlobalHolCycle,
};

llvm::StringRef spatialFifoRuntimeFeedbackDispositionSpelling(
    SpatialFifoRuntimeFeedbackDisposition disposition);
llvm::StringRef spatialFifoRuntimeFeedbackReasonSpelling(
    SpatialFifoRuntimeFeedbackReason reason);

struct SpatialFifoRuntimeFeedback final {
  SpatialFifoRuntimeFeedback(ArtifactRootReference parent,
                             ArtifactRootReference spatial)
      : parentMapping(std::move(parent)), spatialMapping(std::move(spatial)) {}

  ArtifactRootReference parentMapping;
  ArtifactRootReference spatialMapping;
  SpatialFifoRuntimeFeedbackDisposition disposition =
      SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished;
  SpatialFifoRuntimeFeedbackReason reason =
      SpatialFifoRuntimeFeedbackReason::MissingWaitCycle;
  /// The canonical witness FIFO: the full storage of a full-FIFO cycle, or the
  /// first of `disciplineTargets` for a cross-tag global HOL cycle.
  std::optional<::loom::fabric::FabricFifoOccurrenceRef> fifo;
  /// Every StrictFifo occurrence the certificate proves cross-tag head-of-line
  /// blocking on, in canonical order. The discipline candidate applies to all
  /// of them in one child; the set is empty for every other reason.
  std::vector<::loom::fabric::FabricFifoOccurrenceRef> disciplineTargets;
  std::uint32_t occupancy = 0;
  std::uint32_t capacity = 0;
  std::optional<std::uint32_t> minimumCandidateDepth;
  std::optional<::fabric::FifoQueueDiscipline> currentQueueDiscipline;
  std::optional<::fabric::FifoQueueDiscipline> candidateQueueDiscipline;
  bool bypassCapable = false;
  std::uint64_t transferCycleEdgeCount = 0;
  std::uint64_t actorCycleEdgeCount = 0;
  std::optional<std::uint64_t> causalActorOrdinal;
  std::optional<std::uint64_t> causalActionOrdinal;
  std::optional<std::uint64_t> causalOccurrenceOrdinal;

  friend bool operator==(const SpatialFifoRuntimeFeedback &lhs,
                         const SpatialFifoRuntimeFeedback &rhs) {
    return lhs.parentMapping == rhs.parentMapping &&
           lhs.spatialMapping == rhs.spatialMapping &&
           lhs.disposition == rhs.disposition && lhs.reason == rhs.reason &&
           lhs.fifo == rhs.fifo &&
           lhs.disciplineTargets == rhs.disciplineTargets &&
           lhs.occupancy == rhs.occupancy && lhs.capacity == rhs.capacity &&
           lhs.minimumCandidateDepth == rhs.minimumCandidateDepth &&
           lhs.currentQueueDiscipline == rhs.currentQueueDiscipline &&
           lhs.candidateQueueDiscipline == rhs.candidateQueueDiscipline &&
           lhs.bypassCapable == rhs.bypassCapable &&
           lhs.transferCycleEdgeCount == rhs.transferCycleEdgeCount &&
           lhs.actorCycleEdgeCount == rhs.actorCycleEdgeCount &&
           lhs.causalActorOrdinal == rhs.causalActorOrdinal &&
           lhs.causalActionOrdinal == rhs.causalActionOrdinal &&
           lhs.causalOccurrenceOrdinal == rhs.causalOccurrenceOrdinal;
  }
};

llvm::Expected<SpatialFifoRuntimeFeedback> deriveSpatialFifoRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const ArtifactRootReference &spatialMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts);

void emitSpatialFifoRuntimeFeedback(const SpatialFifoRuntimeFeedback &feedback);

enum class SpatialOperandQueueRuntimeFeedbackDisposition : std::uint8_t {
  Exact,
  ProofNotEstablished,
  Unsupported,
};

enum class SpatialOperandQueueRuntimeFeedbackReason : std::uint8_t {
  ExactClosedWait,
  MissingOwnerReferences,
  OwnerMismatch,
  MissingWaitCycle,
  MissingQueueWaitEdge,
  IncompleteOrderedHead,
  ProjectionMismatch,
  AmbiguousTargetPe,
  CandidateCapacityOverflow,
};

struct SpatialOperandBufferRepairTarget final {
  ::loom::fabric::FabricPeOccurrenceRef pe;
  ::fabric::OperandBufferMode currentMode =
      ::fabric::OperandBufferMode::PerInstruction;
  std::uint32_t currentEntriesPerAllocationUnit = 0;
  std::optional<::fabric::OperandBufferMode> separatedMode;
  std::uint32_t candidateEntriesPerAllocationUnit = 0;

  friend bool operator==(const SpatialOperandBufferRepairTarget &lhs,
                         const SpatialOperandBufferRepairTarget &rhs) {
    return lhs.pe == rhs.pe && lhs.currentMode == rhs.currentMode &&
           lhs.currentEntriesPerAllocationUnit ==
               rhs.currentEntriesPerAllocationUnit &&
           lhs.separatedMode == rhs.separatedMode &&
           lhs.candidateEntriesPerAllocationUnit ==
               rhs.candidateEntriesPerAllocationUnit;
  }
};

struct SpatialOperandQueueRuntimeFeedback final {
  std::optional<ArtifactRootReference> parentMapping;
  SpatialOperandQueueRuntimeFeedbackDisposition disposition =
      SpatialOperandQueueRuntimeFeedbackDisposition::ProofNotEstablished;
  SpatialOperandQueueRuntimeFeedbackReason reason =
      SpatialOperandQueueRuntimeFeedbackReason::MissingOwnerReferences;
  std::optional<sim::CgraExecutionOwnerReferences> owners;
  mapping::SpatialPeOperandRuntimeWitness witness;
  std::optional<SpatialOperandBufferRepairTarget> repairTarget;
  bool admissionPolicyAlternativeUnsupported = false;
  std::uint64_t queueWaitEdgeCount = 0;
  std::uint64_t transferCycleEdgeCount = 0;
  std::uint64_t actorCycleEdgeCount = 0;

  friend bool operator==(const SpatialOperandQueueRuntimeFeedback &lhs,
                         const SpatialOperandQueueRuntimeFeedback &rhs) {
    const auto sameOwners = [](const auto &left, const auto &right) {
      if (left.has_value() != right.has_value())
        return false;
      return !left ||
             (left->dataflow == right->dataflow &&
              left->fabric == right->fabric &&
              left->techMapping == right->techMapping &&
              left->spatialMapping == right->spatialMapping);
    };
    const auto &leftWitness = lhs.witness;
    const auto &rightWitness = rhs.witness;
    return lhs.parentMapping == rhs.parentMapping &&
           lhs.disposition == rhs.disposition && lhs.reason == rhs.reason &&
           sameOwners(lhs.owners, rhs.owners) &&
           leftWitness.status == rightWitness.status &&
           leftWitness.support == rightWitness.support &&
           leftWitness.observedHeadCount == rightWitness.observedHeadCount &&
           leftWitness.exactHeadCount == rightWitness.exactHeadCount &&
           leftWitness.matchedPairingKeyCount ==
               rightWitness.matchedPairingKeyCount &&
           leftWitness.unmatchedPairingKeyCount ==
               rightWitness.unmatchedPairingKeyCount &&
           leftWitness.mismatchedHeadCount ==
               rightWitness.mismatchedHeadCount &&
           leftWitness.fullQueueCount == rightWitness.fullQueueCount &&
           leftWitness.projectionDigest == rightWitness.projectionDigest &&
           lhs.repairTarget == rhs.repairTarget &&
           lhs.admissionPolicyAlternativeUnsupported ==
               rhs.admissionPolicyAlternativeUnsupported &&
           lhs.queueWaitEdgeCount == rhs.queueWaitEdgeCount &&
           lhs.transferCycleEdgeCount == rhs.transferCycleEdgeCount &&
           lhs.actorCycleEdgeCount == rhs.actorCycleEdgeCount;
  }
};

llvm::StringRef spatialOperandQueueRuntimeFeedbackDispositionSpelling(
    SpatialOperandQueueRuntimeFeedbackDisposition disposition);
llvm::StringRef spatialOperandQueueRuntimeFeedbackReasonSpelling(
    SpatialOperandQueueRuntimeFeedbackReason reason);

llvm::Expected<SpatialOperandQueueRuntimeFeedback>
deriveSpatialOperandQueueRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts);

void emitSpatialOperandQueueRuntimeFeedback(
    const SpatialOperandQueueRuntimeFeedback &feedback);

enum class SpatialTransportRuntimeFeedbackDisposition : std::uint8_t {
  Exact,
  ProofNotEstablished,
  Unsupported,
};

enum class SpatialTransportRuntimeFeedbackReason : std::uint8_t {
  ExactClosedStorageWait,
  MissingOwnerReferences,
  OwnerMismatch,
  /// The supplied parent SpatialMapping was not admitted by the exact current
  /// constraint root, so it cannot own a new accumulated counterexample.
  ParentConstraintRejection,
  /// The runtime reported a proof failure, an empty certificate, or a
  /// certificate that is not one closed strongly connected component. There is
  /// nothing to project, and no weaker witness is substituted for it.
  UnprovenWaitCertificate,
  /// One essential certificate edge could not be joined all the way back to the
  /// parent RouteTree. Projection is all-or-nothing, so no clause is published.
  UnjoinedCertificateEdge,
  /// The supplied runtime evidence reference is not a well-formed evaluation
  /// evidence object, or it cannot be bound to the exact parent Mapping.
  UnboundRuntimeEvidence,
  /// The selected SpatialMapping has no exact current constraint-set lineage,
  /// so a persistent accumulated clause cannot be derived safely.
  UnboundConstraintLineage,
  /// The certificate projected cleanly but named no exact Mapping choice, so
  /// there is no non-empty clause to publish.
  EmptyLiteralSet,
  /// Traversal and attachment projection is diagnostic-only until every
  /// Mapping decision needed to preserve the closed wait, including Physical
  /// Tag queue class and durable certificate provenance, is independently
  /// verifiable. No persistent clause is published from a partial causal core.
  CausalCoreNotEstablished,
};

/// Mechanical projection of one canonical `NetUsesTraversal` literal for
/// callers that still consume the older shape. It is never an independent
/// semantic owner: every value is derived from `literals`, and the actor-cycle
/// derivation that used to produce it is gone.
struct SpatialTransportRepairAlternative final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::loom::fabric::FabricPhysicalTraversalRef forbiddenTraversal;

  friend bool operator==(const SpatialTransportRepairAlternative &lhs,
                         const SpatialTransportRepairAlternative &rhs) {
    return lhs.producer == rhs.producer &&
           lhs.forbiddenTraversal == rhs.forbiddenTraversal;
  }
};

/// One mechanical projection of an exact closed wait certificate into the
/// canonical Spatial no-good clause. Every field is derived; the runtime
/// remains the only owner of the dynamic facts, and the published constraint
/// set remains the only persistent artifact.
struct SpatialTransportRuntimeFeedback final {
  /// Separate lineage only, and unchanged in meaning: the parent SystemMapping
  /// an existing SystemMapping-scoped controller selected. No projected literal
  /// and no published clause depends on it, and it may be absent.
  std::optional<ArtifactRootReference> parentMapping;
  /// The exact parent SpatialMapping the certificate was observed on. The
  /// persistent no-good is owned by this and the constraint root's exact
  /// Dataflow/TechMapping/Fabric tuple alone.
  std::optional<ArtifactRootReference> parentSpatialMapping;
  /// The exact constraint set that parent was admitted under.
  std::optional<ArtifactRootReference> parentConstraints;
  /// The exact runtime Evaluation Evidence the certificate came from.
  std::optional<ArtifactRootReference> runtimeEvidence;
  /// The exact Evidence output that owns the durable Halted witness.
  std::optional<ArtifactRootReference> runtimeExecution;
  /// The canonical union of `parentConstraints` with the projected clause.
  /// Engaged exactly when the disposition is Exact.
  std::optional<ArtifactRootReference> constraintSet;
  /// Removable strict-import cache of `constraintSet`. Persistent legality is
  /// still owned solely by the Artifact root; cumulative CEGAR may reuse this
  /// object after requiring exact reference equality.
  std::shared_ptr<const mapping::FinalizedSpatialMappingConstraintSet>
      importedConstraintSet;
  SpatialTransportRuntimeFeedbackDisposition disposition =
      SpatialTransportRuntimeFeedbackDisposition::ProofNotEstablished;
  SpatialTransportRuntimeFeedbackReason reason =
      SpatialTransportRuntimeFeedbackReason::MissingOwnerReferences;
  std::optional<sim::CgraExecutionOwnerReferences> owners;
  /// Deterministic digest of the complete typed certificate. Discovery order
  /// does not affect it, so reprojecting one certificate reproduces it exactly.
  std::optional<sim::CgraClosedWaitCertificateDigest> certificateDigest;
  /// The exact evaluation Request the runtime evidence was produced from,
  /// proven to bind this exact SpatialMapping under the CGRA simulation model.
  std::optional<ArtifactRootReference> evaluationRequest;
  /// The exact projected literals, in canonical order.
  std::vector<mapping::SpatialNoGoodLiteral> literals;
  std::vector<SpatialTransportRepairAlternative> alternatives;
  std::uint64_t certificateEdgeCount = 0;
  std::uint64_t projectedEdgeCount = 0;
  std::uint64_t outputBackpressureEdgeCount = 0;
  std::uint64_t exactBlockedTransferCount = 0;

  friend bool operator==(const SpatialTransportRuntimeFeedback &lhs,
                         const SpatialTransportRuntimeFeedback &rhs) {
    const auto sameOwners = [](const auto &left, const auto &right) {
      if (left.has_value() != right.has_value())
        return false;
      return !left ||
             (left->dataflow == right->dataflow &&
              left->fabric == right->fabric &&
              left->techMapping == right->techMapping &&
              left->spatialMapping == right->spatialMapping);
    };
    return lhs.parentMapping == rhs.parentMapping &&
           lhs.parentSpatialMapping == rhs.parentSpatialMapping &&
           lhs.parentConstraints == rhs.parentConstraints &&
           lhs.runtimeEvidence == rhs.runtimeEvidence &&
           lhs.runtimeExecution == rhs.runtimeExecution &&
           lhs.evaluationRequest == rhs.evaluationRequest &&
           lhs.constraintSet == rhs.constraintSet &&
           lhs.disposition == rhs.disposition && lhs.reason == rhs.reason &&
           sameOwners(lhs.owners, rhs.owners) &&
           lhs.certificateDigest == rhs.certificateDigest &&
           lhs.literals == rhs.literals &&
           lhs.alternatives == rhs.alternatives &&
           lhs.certificateEdgeCount == rhs.certificateEdgeCount &&
           lhs.projectedEdgeCount == rhs.projectedEdgeCount &&
           lhs.outputBackpressureEdgeCount == rhs.outputBackpressureEdgeCount &&
           lhs.exactBlockedTransferCount == rhs.exactBlockedTransferCount;
  }
};

llvm::StringRef spatialTransportRuntimeFeedbackDispositionSpelling(
    SpatialTransportRuntimeFeedbackDisposition disposition);
llvm::StringRef spatialTransportRuntimeFeedbackReasonSpelling(
    SpatialTransportRuntimeFeedbackReason reason);

/// Attempts to promote one verified durable closed wait into a canonical
/// Spatial no-good over `parentConstraints`.
///
/// Promotion is all-or-nothing. Evidence binding, deterministic replay, every
/// essential certificate-edge join, and a complete independently verifiable
/// causal literal core are all required before `constraintSet` may engage.
/// Otherwise the result is typed ProofNotEstablished and publishes nothing;
/// partial literals and repair alternatives are withheld so no caller can
/// execute them as a learned constraint. The promoted no-good is bound by the
/// exact replay Request and its Dataflow/TechMapping/Fabric tuple, never by a
/// SystemMapping. A conservative exact-parent Mapping literal closes the
/// persistent selection core; certificate-derived route, attachment, and tag
/// literals remain independently verified SCC-local repair anchors.
llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentSpatialMapping,
    const ArtifactRootReference &parentConstraints,
    const ::loom::evaluation::models::VerifiedCgraClosedWaitEvidence
        &runtimeEvidence,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> parentSystemMapping = std::nullopt);

/// Incremental equivalent for a parent constraint set that has already passed
/// strict import. The resulting persistent Artifact and feedback are identical
/// to the root-reference overload.
llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentSpatialMapping,
    const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints,
    const ::loom::evaluation::models::VerifiedCgraClosedWaitEvidence
        &runtimeEvidence,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> parentSystemMapping = std::nullopt);

void emitSpatialTransportRuntimeFeedback(
    const SpatialTransportRuntimeFeedback &feedback);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
