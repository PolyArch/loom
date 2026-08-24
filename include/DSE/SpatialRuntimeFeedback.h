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
