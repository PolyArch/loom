#ifndef LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
#define LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Fabric/IR/FabricEnums.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Simulator/CGRASimulator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::evaluation {
class EvaluationRequest;
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
  std::optional<::loom::fabric::FabricFifoOccurrenceRef> fifo;
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
           lhs.fifo == rhs.fifo && lhs.occupancy == rhs.occupancy &&
           lhs.capacity == rhs.capacity &&
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

/// The typed digest of one unified runtime wait certificate. The certificate is
/// dynamic evidence rather than an artifact, so it never carries an
/// ArtifactIdentity; this distinct value type keeps a certificate digest from
/// being compared against, or mistaken for, any other digest in the stack.
/// Only `computeSpatialWaitCertificateDigest` can mint one.
class SpatialWaitCertificateDigest final {
public:
  static constexpr llvm::StringLiteral domain =
      "loom.spatial_wait_certificate.1";

  const ComponentViewDigest &digest() const { return digest_; }

  friend bool operator==(const SpatialWaitCertificateDigest &lhs,
                         const SpatialWaitCertificateDigest &rhs) {
    return lhs.digest_ == rhs.digest_;
  }
  friend bool operator!=(const SpatialWaitCertificateDigest &lhs,
                         const SpatialWaitCertificateDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit SpatialWaitCertificateDigest(ComponentViewDigest digest)
      : digest_(std::move(digest)) {}

  friend llvm::Expected<SpatialWaitCertificateDigest>
  computeSpatialWaitCertificateDigest(
      const sim::CgraClosedWaitSetDiagnostic &closedWait);

  ComponentViewDigest digest_;
};

/// Derives the canonical digest of the complete typed certificate. Certificate
/// edges are canonically ordered first, so discovery order cannot affect it.
llvm::Expected<SpatialWaitCertificateDigest>
computeSpatialWaitCertificateDigest(
    const sim::CgraClosedWaitSetDiagnostic &closedWait);

std::string
formatSpatialWaitCertificateDigest(const SpatialWaitCertificateDigest &digest);

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
  /// The canonical union of `parentConstraints` with the projected clause.
  /// Engaged exactly when the disposition is Exact.
  std::optional<ArtifactRootReference> constraintSet;
  SpatialTransportRuntimeFeedbackDisposition disposition =
      SpatialTransportRuntimeFeedbackDisposition::ProofNotEstablished;
  SpatialTransportRuntimeFeedbackReason reason =
      SpatialTransportRuntimeFeedbackReason::MissingOwnerReferences;
  std::optional<sim::CgraExecutionOwnerReferences> owners;
  /// Deterministic digest of the complete typed certificate. Discovery order
  /// does not affect it, so reprojecting one certificate reproduces it exactly.
  std::optional<SpatialWaitCertificateDigest> certificateDigest;
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

/// The exact runtime evidence one certificate was observed under. The Request
/// root is not carried separately: it is mechanically
/// `evaluationRequestReference(requestView)`, so a second copy would be a
/// duplicate authority the projection would have to reconcile.
struct SpatialTransportRuntimeEvidence final {
  ArtifactRootReference evidence;
  const ::loom::evaluation::EvaluationRequest &requestView;
};

/// Projects one exact closed wait certificate into a canonical Spatial no-good
/// and publishes it by union with `parentConstraints`.
///
/// The projection is all-or-nothing. Any essential certificate edge that cannot
/// be joined back to the exact parent RouteTree yields typed
/// ProofNotEstablished and publishes nothing; there is no first-traversal
/// fallback and no partial clause. The no-good is bound by the constraint
/// root's exact Dataflow/TechMapping/Fabric tuple and never by a SystemMapping.
llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentSpatialMapping,
    const ArtifactRootReference &parentConstraints,
    const SpatialTransportRuntimeEvidence &runtimeEvidence,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> parentSystemMapping = std::nullopt);

void emitSpatialTransportRuntimeFeedback(
    const SpatialTransportRuntimeFeedback &feedback);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
