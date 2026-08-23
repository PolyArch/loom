#include "DSE/SpatialRuntimeFeedback.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <vector>

namespace loom::dse {

llvm::StringRef spatialOperandQueueRuntimeFeedbackDispositionSpelling(
    SpatialOperandQueueRuntimeFeedbackDisposition disposition) {
  switch (disposition) {
  case SpatialOperandQueueRuntimeFeedbackDisposition::Exact:
    return "exact";
  case SpatialOperandQueueRuntimeFeedbackDisposition::ProofNotEstablished:
    return "proof_not_established";
  case SpatialOperandQueueRuntimeFeedbackDisposition::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown Spatial operand-queue feedback disposition");
}

llvm::StringRef spatialOperandQueueRuntimeFeedbackReasonSpelling(
    SpatialOperandQueueRuntimeFeedbackReason reason) {
  switch (reason) {
  case SpatialOperandQueueRuntimeFeedbackReason::ExactClosedWait:
    return "exact_closed_wait";
  case SpatialOperandQueueRuntimeFeedbackReason::MissingOwnerReferences:
    return "missing_owner_references";
  case SpatialOperandQueueRuntimeFeedbackReason::OwnerMismatch:
    return "owner_mismatch";
  case SpatialOperandQueueRuntimeFeedbackReason::MissingWaitCycle:
    return "missing_wait_cycle";
  case SpatialOperandQueueRuntimeFeedbackReason::MissingQueueWaitEdge:
    return "missing_queue_wait_edge";
  case SpatialOperandQueueRuntimeFeedbackReason::IncompleteOrderedHead:
    return "incomplete_ordered_head";
  case SpatialOperandQueueRuntimeFeedbackReason::ProjectionMismatch:
    return "projection_mismatch";
  }
  llvm_unreachable("unknown Spatial operand-queue feedback reason");
}

llvm::Expected<SpatialOperandQueueRuntimeFeedback>
deriveSpatialOperandQueueRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts) {
  SpatialOperandQueueRuntimeFeedback result;
  result.parentMapping = parentMapping;
  result.transferCycleEdgeCount = closedWait.transferWaitCycle.size();
  result.actorCycleEdgeCount = closedWait.actorWaitCycle.size();
  if (!closedWait.ownerReferences)
    return result;
  result.owners = closedWait.ownerReferences;
  const sim::CgraExecutionOwnerReferences &owners = *closedWait.ownerReferences;
  auto systemMapping = mapping::importSystemMapping(parentMapping, artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  if (systemMapping->view().dataflowIdentity() != owners.dataflow.artifact) {
    result.disposition =
        SpatialOperandQueueRuntimeFeedbackDisposition::Unsupported;
    result.reason = SpatialOperandQueueRuntimeFeedbackReason::OwnerMismatch;
    return result;
  }
  auto dataflow = dataflow::importCanonicalDataflow(owners.dataflow, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto fabric = fabric::importEntireFabricRoot(owners.fabric, artifacts);
  if (!fabric)
    return fabric.takeError();
  auto tech = mapping::importTechMapping(owners.techMapping, artifacts);
  if (!tech)
    return tech.takeError();
  auto spatial = mapping::importSpatialMapping(owners.spatialMapping, artifacts);
  if (!spatial)
    return spatial.takeError();
  if (tech->view().dataflowIdentity() != owners.dataflow.artifact ||
      tech->view().fabricIdentity() != owners.fabric.artifact ||
      spatial->view().dataflowIdentity() != owners.dataflow.artifact ||
      spatial->view().fabricIdentity() != owners.fabric.artifact ||
      spatial->view().techMappingIdentity() != owners.techMapping.artifact ||
      !llvm::is_contained(
          systemMapping->view().executionBindings().spatialMappingImports(),
          owners.spatialMapping)) {
    result.disposition =
        SpatialOperandQueueRuntimeFeedbackDisposition::Unsupported;
    result.reason = SpatialOperandQueueRuntimeFeedbackReason::OwnerMismatch;
    return result;
  }

  auto groups = mapping::deriveSpatialPeOperandQueueMatchGroups(
      tech->view(), fabric->view(), spatial->view().computeBindings(),
      spatial->view().routeTrees(), spatial->view().resourceUses(),
      spatial->view().physicalTagSegments());
  if (!groups)
    return groups.takeError();
  auto projection = mapping::deriveSpatialPeOperandProgressFeedback(
      *dataflowView, tech->view(), *groups);
  if (!projection)
    return projection.takeError();
  if (projection->groupCount != closedWait.operandQueueGroupCount ||
      projection->potentiallyBlockingGroupCount !=
          closedWait.operandQueuePotentiallyBlockingGroupCount ||
      projection->sharedIngressPressure !=
          closedWait.operandQueueSharedIngressPressure ||
      projection->distinctIngressCount !=
          closedWait.operandQueueDistinctIngressCount ||
      projection->pairingKeyCount != closedWait.operandQueuePairingKeyCount ||
      static_cast<std::uint8_t>(projection->status) !=
          closedWait.operandQueueProgressStatus ||
      static_cast<std::uint8_t>(projection->support) !=
          closedWait.operandQueueProgressSupport ||
      !closedWait.operandQueueProjectionDigest ||
      !projection->projectionDigest ||
      *closedWait.operandQueueProjectionDigest !=
          *projection->projectionDigest) {
    result.disposition =
        SpatialOperandQueueRuntimeFeedbackDisposition::Unsupported;
    result.reason =
        SpatialOperandQueueRuntimeFeedbackReason::ProjectionMismatch;
    return result;
  }

  std::vector<mapping::SpatialPeOperandRuntimeHeadView> heads;
  heads.reserve(closedWait.operandQueueHeads.size());
  for (const auto &head : closedWait.operandQueueHeads)
    heads.push_back({head.queue,
                     head.fu,
                     head.headTag,
                     head.allocationUnit,
                     head.capacity,
                     head.occupancy,
                     head.reservations,
                     head.headBindingOrdinal,
                     head.headOccurrenceOrdinal,
                     head.headProducerSequenceOrdinal,
                     head.exactHead});
  auto witness =
      mapping::deriveSpatialPeOperandRuntimeWitness(*projection, heads);
  if (!witness)
    return witness.takeError();
  result.witness = std::move(*witness);
  if (result.witness.status !=
      mapping::SpatialPeOperandRuntimeWitnessStatus::Exact) {
    result.reason =
        SpatialOperandQueueRuntimeFeedbackReason::IncompleteOrderedHead;
    return result;
  }

  bool queueWaitInCycle = false;
  for (const auto &transfer : closedWait.transfers) {
    result.queueWaitEdgeCount += transfer.operandQueueWaits.size();
    const bool transferCycleMember = llvm::any_of(
        closedWait.transferWaitCycle, [&](const auto &edge) {
          return (edge.waitingBindingOrdinal == transfer.bindingOrdinal &&
                  edge.waitingOccurrenceOrdinal == transfer.occurrenceOrdinal) ||
                 (edge.blockingBindingOrdinal == transfer.bindingOrdinal &&
                  edge.blockingOccurrenceOrdinal == transfer.occurrenceOrdinal);
        });
    const bool actorCycleMember = llvm::any_of(
        closedWait.actorWaitCycle, [&](const auto &edge) {
          return edge.waitingActorOrdinal == transfer.producerActorOrdinal ||
                 edge.blockingActorOrdinal == transfer.producerActorOrdinal ||
                 edge.waitingActorOrdinal == transfer.blockingActorOrdinal ||
                 edge.blockingActorOrdinal == transfer.blockingActorOrdinal;
        });
    for (const auto &wait : transfer.operandQueueWaits) {
      const auto pairing = llvm::find_if(
          projection->pairings, [&](const auto &candidate) {
            return candidate.key.context == wait.queue.context &&
                   candidate.key.fu == wait.fu &&
                   candidate.key.tag.getBitWidth() == wait.tag.getBitWidth() &&
                   candidate.key.tag == wait.tag &&
                   llvm::is_contained(candidate.requiredInputRoles,
                                      wait.queue.fuInput) &&
                   llvm::is_contained(candidate.ingresses, wait.ingress) &&
                   llvm::is_contained(candidate.allocationUnits,
                                      wait.allocationUnit);
          });
      const auto observedHead = llvm::find_if(
          closedWait.operandQueueHeads, [&](const auto &head) {
            return head.queue == wait.queue && head.fu == wait.fu;
          });
      if (pairing == projection->pairings.end() ||
          observedHead == closedWait.operandQueueHeads.end() ||
          wait.occupancy > wait.capacity ||
          wait.reservations > wait.capacity - wait.occupancy ||
          observedHead->capacity != wait.capacity ||
          observedHead->occupancy != wait.occupancy ||
          observedHead->reservations != wait.reservations) {
        result.disposition =
            SpatialOperandQueueRuntimeFeedbackDisposition::Unsupported;
        result.reason =
            SpatialOperandQueueRuntimeFeedbackReason::ProjectionMismatch;
        return result;
      }
      queueWaitInCycle |= (transferCycleMember || actorCycleMember) &&
                          wait.occupancy + wait.reservations == wait.capacity;
    }
  }
  if (result.queueWaitEdgeCount == 0) {
    result.reason =
        SpatialOperandQueueRuntimeFeedbackReason::MissingQueueWaitEdge;
    return result;
  }
  if (!queueWaitInCycle || (closedWait.transferWaitCycle.empty() &&
                            closedWait.actorWaitCycle.empty())) {
    result.reason = SpatialOperandQueueRuntimeFeedbackReason::MissingWaitCycle;
    return result;
  }
  result.disposition = SpatialOperandQueueRuntimeFeedbackDisposition::Exact;
  result.reason = SpatialOperandQueueRuntimeFeedbackReason::ExactClosedWait;
  result.witness.status =
      mapping::SpatialPeOperandRuntimeWitnessStatus::ProvenClosedWait;
  return result;
}

void emitSpatialOperandQueueRuntimeFeedback(
    const SpatialOperandQueueRuntimeFeedback &feedback) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "spatial_operand_queue_runtime_feedback";
        fields["parent_mapping"] =
            feedback.parentMapping
                ? llvm::json::Value(formatArtifactIdentityHex(
                      feedback.parentMapping->artifact))
                : llvm::json::Value(nullptr);
        fields["disposition"] =
            spatialOperandQueueRuntimeFeedbackDispositionSpelling(
                feedback.disposition);
        fields["reason"] = spatialOperandQueueRuntimeFeedbackReasonSpelling(
            feedback.reason);
        fields["queue_wait_edge_count"] = feedback.queueWaitEdgeCount;
        fields["transfer_cycle_edge_count"] =
            feedback.transferCycleEdgeCount;
        fields["actor_cycle_edge_count"] = feedback.actorCycleEdgeCount;
        fields["observed_head_count"] = feedback.witness.observedHeadCount;
        fields["exact_head_count"] = feedback.witness.exactHeadCount;
        fields["matched_pairing_key_count"] =
            feedback.witness.matchedPairingKeyCount;
        fields["unmatched_pairing_key_count"] =
            feedback.witness.unmatchedPairingKeyCount;
        fields["mismatched_head_count"] =
            feedback.witness.mismatchedHeadCount;
        fields["full_queue_count"] = feedback.witness.fullQueueCount;
        fields["runtime_projection_digest"] =
            feedback.witness.projectionDigest
                ? llvm::json::Value(formatComponentViewDigestHex(
                      *feedback.witness.projectionDigest))
                : llvm::json::Value(nullptr);
      });
}

} // namespace loom::dse
