#ifndef LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
#define LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H

#include "Common/Artifact.h"
#include "Fabric/IR/FabricEnums.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Simulator/CGRASimulator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
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
};

llvm::StringRef spatialFifoRuntimeFeedbackDispositionSpelling(
    SpatialFifoRuntimeFeedbackDisposition disposition);
llvm::StringRef spatialFifoRuntimeFeedbackReasonSpelling(
    SpatialFifoRuntimeFeedbackReason reason);

struct SpatialFifoRuntimeFeedback final {
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
  MissingWaitCycle,
  MissingOutputBackpressure,
  ProjectionMismatch,
  NoAlternativeTraversal,
  CandidateCapacityOverflow,
};

struct SpatialTransportRepairAlternative final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::loom::fabric::FabricPhysicalTraversalRef forbiddenTraversal;

  friend bool operator==(const SpatialTransportRepairAlternative &lhs,
                         const SpatialTransportRepairAlternative &rhs) {
    return lhs.producer == rhs.producer &&
           lhs.forbiddenTraversal == rhs.forbiddenTraversal;
  }
};

struct SpatialTransportRuntimeFeedback final {
  std::optional<ArtifactRootReference> parentMapping;
  SpatialTransportRuntimeFeedbackDisposition disposition =
      SpatialTransportRuntimeFeedbackDisposition::ProofNotEstablished;
  SpatialTransportRuntimeFeedbackReason reason =
      SpatialTransportRuntimeFeedbackReason::MissingOwnerReferences;
  std::optional<sim::CgraExecutionOwnerReferences> owners;
  std::vector<SpatialTransportRepairAlternative> alternatives;
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
           lhs.disposition == rhs.disposition && lhs.reason == rhs.reason &&
           sameOwners(lhs.owners, rhs.owners) &&
           lhs.alternatives == rhs.alternatives &&
           lhs.outputBackpressureEdgeCount ==
               rhs.outputBackpressureEdgeCount &&
           lhs.exactBlockedTransferCount == rhs.exactBlockedTransferCount;
  }
};

llvm::StringRef spatialTransportRuntimeFeedbackDispositionSpelling(
    SpatialTransportRuntimeFeedbackDisposition disposition);
llvm::StringRef spatialTransportRuntimeFeedbackReasonSpelling(
    SpatialTransportRuntimeFeedbackReason reason);

llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts);

void emitSpatialTransportRuntimeFeedback(
    const SpatialTransportRuntimeFeedback &feedback);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALRUNTIMEFEEDBACK_H
