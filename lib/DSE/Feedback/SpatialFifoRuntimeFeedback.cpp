#include "DSE/SpatialRuntimeFeedback.h"

#include "../JointHardwareReopenInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <vector>

namespace loom::dse {

using joint_reopen_detail::invalid;

llvm::StringRef spatialFifoRuntimeFeedbackDispositionSpelling(
    SpatialFifoRuntimeFeedbackDisposition disposition) {
  switch (disposition) {
  case SpatialFifoRuntimeFeedbackDisposition::Exact:
    return "exact";
  case SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished:
    return "proof_not_established";
  case SpatialFifoRuntimeFeedbackDisposition::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown Spatial FIFO runtime feedback disposition");
}

llvm::StringRef spatialFifoRuntimeFeedbackReasonSpelling(
    SpatialFifoRuntimeFeedbackReason reason) {
  switch (reason) {
  case SpatialFifoRuntimeFeedbackReason::ExactFullFifoCycle:
    return "exact_full_fifo_cycle";
  case SpatialFifoRuntimeFeedbackReason::MissingWaitCycle:
    return "missing_wait_cycle";
  case SpatialFifoRuntimeFeedbackReason::MissingCanonicalFifo:
    return "missing_canonical_fifo";
  case SpatialFifoRuntimeFeedbackReason::AmbiguousFifo:
    return "ambiguous_fifo";
  case SpatialFifoRuntimeFeedbackReason::StorageNotFull:
    return "storage_not_full";
  case SpatialFifoRuntimeFeedbackReason::MissingCausalReleaseContext:
    return "missing_causal_release_context";
  }
  llvm_unreachable("unknown Spatial FIFO runtime feedback reason");
}

llvm::Expected<SpatialFifoRuntimeFeedback> deriveSpatialFifoRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const ArtifactRootReference &spatialMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts) {
  auto parent = mapping::importSystemMapping(parentMapping, artifacts);
  if (!parent)
    return parent.takeError();
  if (!llvm::is_contained(
          parent->view().executionBindings().spatialMappingImports(),
          spatialMapping))
    return invalid("FIFO runtime feedback names a foreign SpatialMapping");
  auto spatial = mapping::importSpatialMapping(spatialMapping, artifacts);
  if (!spatial)
    return spatial.takeError();
  ArtifactRootReference moduleReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, spatial->view().fabricIdentity()};
  auto module = fabric::importEntireFabricRoot(moduleReference, artifacts);
  if (!module)
    return module.takeError();
  if (module->view().rootKind() != fabric::FabricRootKind::Module)
    return invalid("FIFO runtime feedback does not bind a Module");

  SpatialFifoRuntimeFeedback feedback{
      parentMapping,
      spatialMapping,
      SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished,
      SpatialFifoRuntimeFeedbackReason::MissingWaitCycle,
      std::nullopt,
      0,
      0,
      std::nullopt,
      false,
      closedWait.transferWaitCycle.size(),
      closedWait.actorWaitCycle.size(),
      std::nullopt,
      std::nullopt,
      std::nullopt};

  const auto inTransferCycle = [&](const auto &transfer) {
    return llvm::any_of(closedWait.transferWaitCycle, [&](const auto &edge) {
      return (edge.waitingBindingOrdinal == transfer.bindingOrdinal &&
              edge.waitingOccurrenceOrdinal == transfer.occurrenceOrdinal) ||
             (edge.blockingBindingOrdinal == transfer.bindingOrdinal &&
              edge.blockingOccurrenceOrdinal == transfer.occurrenceOrdinal);
    });
  };
  const auto inActorCycle = [&](const auto &transfer) {
    return llvm::any_of(closedWait.actorWaitCycle, [&](const auto &edge) {
      return edge.waitingActorOrdinal == transfer.producerActorOrdinal ||
             edge.blockingActorOrdinal == transfer.producerActorOrdinal ||
             edge.waitingActorOrdinal == transfer.blockingActorOrdinal ||
             edge.blockingActorOrdinal == transfer.blockingActorOrdinal;
    });
  };
  const bool hasCycle = !closedWait.transferWaitCycle.empty() ||
                        !closedWait.actorWaitCycle.empty();
  if (!hasCycle)
    return feedback;
  std::vector<const sim::CgraClosedWaitSetDiagnostic::Transfer *> fifoWaits;
  for (const auto &transfer : closedWait.transfers) {
    if (!transfer.blocked || !transfer.blockingFifoOccurrence)
      continue;
    if (!inTransferCycle(transfer) && !inActorCycle(transfer))
      continue;
    fifoWaits.push_back(&transfer);
  }
  if (fifoWaits.empty()) {
    feedback.disposition = SpatialFifoRuntimeFeedbackDisposition::Unsupported;
    feedback.reason = SpatialFifoRuntimeFeedbackReason::MissingCanonicalFifo;
    return feedback;
  }
  llvm::sort(fifoWaits, [](const auto *lhs, const auto *rhs) {
    return fabric::canonicalFabricBytes(*lhs->blockingFifoOccurrence) <
           fabric::canonicalFabricBytes(*rhs->blockingFifoOccurrence);
  });
  const auto fifo = *fifoWaits.front()->blockingFifoOccurrence;
  if (llvm::any_of(fifoWaits, [&](const auto *transfer) {
        return *transfer->blockingFifoOccurrence != fifo;
      })) {
    feedback.reason = SpatialFifoRuntimeFeedbackReason::AmbiguousFifo;
    return feedback;
  }
  if (llvm::Error error = fabric::validateFabricRef(module->view(), fifo))
    return std::move(error);
  if (!mapping::spatialMappingUsesFifoOccurrence(spatial->view(), fifo))
    return invalid("FIFO runtime feedback names an unselected occurrence");
  feedback.fifo = fifo;
  feedback.occupancy = fifoWaits.front()->blockingStorageOccupancy;
  feedback.capacity = fifoWaits.front()->blockingStorageCapacity;
  for (const auto *transfer : fifoWaits)
    if (transfer->blockingStorageOccupancy != feedback.occupancy ||
        transfer->blockingStorageCapacity != feedback.capacity) {
      feedback.reason = SpatialFifoRuntimeFeedbackReason::AmbiguousFifo;
      return feedback;
    }
  for (const auto &traversal : module->view().admittedTraversals()) {
    const auto *candidate =
        std::get_if<fabric::FabricFifoTraversalPayload>(&traversal.payload);
    feedback.bypassCapable |=
        candidate && candidate->owner == fifo &&
        candidate->mode == fabric::FabricFifoTraversalMode::Bypass;
  }
  if (feedback.capacity == 0 || feedback.occupancy != feedback.capacity) {
    feedback.reason = SpatialFifoRuntimeFeedbackReason::StorageNotFull;
    return feedback;
  }

  for (const auto &action : closedWait.physicalActions) {
    if (!action.semanticActorOrdinal || !action.granted ||
        !action.requiresCausalRelease || !action.intrinsicReleaseReached ||
        action.causalReleaseReached)
      continue;
    const auto firing =
        llvm::find_if(closedWait.actorFirings, [&](const auto &candidate) {
          return candidate.semanticActorOrdinal ==
                     *action.semanticActorOrdinal &&
                 candidate.occurrenceOrdinal == action.occurrenceOrdinal &&
                 candidate.physicalComplete &&
                 !candidate.causalReleaseSatisfied;
        });
    if (firing == closedWait.actorFirings.end())
      continue;
    const bool ownsWait = llvm::any_of(fifoWaits, [&](const auto *transfer) {
      return transfer->producerActorOrdinal == *action.semanticActorOrdinal ||
             transfer->blockingActorOrdinal == *action.semanticActorOrdinal;
    });
    if (!ownsWait)
      continue;
    if (feedback.causalActorOrdinal &&
        (*feedback.causalActorOrdinal != *action.semanticActorOrdinal ||
         *feedback.causalActionOrdinal != action.actionOrdinal ||
         *feedback.causalOccurrenceOrdinal != action.occurrenceOrdinal)) {
      feedback.reason =
          SpatialFifoRuntimeFeedbackReason::MissingCausalReleaseContext;
      return feedback;
    }
    feedback.causalActorOrdinal = *action.semanticActorOrdinal;
    feedback.causalActionOrdinal = action.actionOrdinal;
    feedback.causalOccurrenceOrdinal = action.occurrenceOrdinal;
  }
  if (!feedback.causalActorOrdinal) {
    feedback.reason =
        SpatialFifoRuntimeFeedbackReason::MissingCausalReleaseContext;
    return feedback;
  }
  if (feedback.capacity == std::numeric_limits<std::uint32_t>::max())
    return invalid("FIFO runtime feedback depth overflows u32");
  feedback.minimumCandidateDepth = feedback.capacity + 1;
  feedback.disposition = SpatialFifoRuntimeFeedbackDisposition::Exact;
  feedback.reason = SpatialFifoRuntimeFeedbackReason::ExactFullFifoCycle;
  return feedback;
}

void emitSpatialFifoRuntimeFeedback(
    const SpatialFifoRuntimeFeedback &feedback) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "spatial_fifo_runtime_feedback";
        fields["parent_mapping"] =
            formatArtifactIdentityHex(feedback.parentMapping.artifact);
        fields["spatial_mapping"] =
            formatArtifactIdentityHex(feedback.spatialMapping.artifact);
        fields["disposition"] =
            spatialFifoRuntimeFeedbackDispositionSpelling(feedback.disposition);
        fields["reason"] =
            spatialFifoRuntimeFeedbackReasonSpelling(feedback.reason);
        fields["occupancy"] = feedback.occupancy;
        fields["capacity"] = feedback.capacity;
        fields["bypass_capable"] = feedback.bypassCapable;
        fields["transfer_cycle_edge_count"] = feedback.transferCycleEdgeCount;
        fields["actor_cycle_edge_count"] = feedback.actorCycleEdgeCount;
        if (feedback.fifo)
          fields["fifo"] =
              llvm::toHex(fabric::canonicalFabricBytes(*feedback.fifo), true);
        else
          fields["fifo"] = nullptr;
        if (feedback.minimumCandidateDepth)
          fields["minimum_candidate_depth"] = *feedback.minimumCandidateDepth;
        else
          fields["minimum_candidate_depth"] = nullptr;
        if (feedback.causalActorOrdinal)
          fields["causal_actor"] = *feedback.causalActorOrdinal;
        else
          fields["causal_actor"] = nullptr;
        if (feedback.causalActionOrdinal)
          fields["causal_action"] = *feedback.causalActionOrdinal;
        else
          fields["causal_action"] = nullptr;
        if (feedback.causalOccurrenceOrdinal)
          fields["causal_occurrence"] = *feedback.causalOccurrenceOrdinal;
        else
          fields["causal_occurrence"] = nullptr;
        fields["hardware_child_count"] = 0;
      });
}

} // namespace loom::dse
