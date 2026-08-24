#include "DSE/SpatialRuntimeFeedback.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <string>

namespace loom::dse {
namespace {

constexpr std::size_t maximumTransportRepairAlternatives = 32;

std::string
alternativeKey(llvm::ArrayRef<std::uint8_t> producer,
               const ::loom::fabric::FabricPhysicalTraversalRef &traversal) {
  const auto physical = ::loom::fabric::canonicalFabricBytes(traversal);
  std::string key;
  key.reserve(producer.size() + physical.size());
  key.append(reinterpret_cast<const char *>(producer.data()), producer.size());
  key.append(reinterpret_cast<const char *>(physical.data()), physical.size());
  return key;
}

bool routeSelectsTraversal(
    const mapping::SpatialRouteTreeView &route,
    const ::loom::fabric::FabricPhysicalTraversalRef &traversal) {
  return llvm::any_of(route.nodes, [&](const auto &node) {
    return node.incomingTraversal && *node.incomingTraversal == traversal;
  });
}

llvm::Expected<bool> producerMatchesSemanticActor(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    std::uint64_t semanticActorOrdinal) {
  const auto *result = std::get_if<::dataflow::ActorTokenResultRef>(&producer);
  if (!result)
    return false;
  auto actor = dataflow.resolve(result->actor);
  if (!actor)
    return actor.takeError();
  std::uint64_t graphLocalOrdinal = 0;
  for (const auto &candidate : dataflow.actors()) {
    if (candidate.graph != actor->graph)
      continue;
    if (candidate.ref == result->actor)
      return graphLocalOrdinal == semanticActorOrdinal;
    ++graphLocalOrdinal;
  }
  return false;
}

} // namespace

llvm::StringRef spatialTransportRuntimeFeedbackDispositionSpelling(
    SpatialTransportRuntimeFeedbackDisposition disposition) {
  switch (disposition) {
  case SpatialTransportRuntimeFeedbackDisposition::Exact:
    return "exact";
  case SpatialTransportRuntimeFeedbackDisposition::ProofNotEstablished:
    return "proof_not_established";
  case SpatialTransportRuntimeFeedbackDisposition::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown Spatial transport runtime feedback disposition");
}

llvm::StringRef spatialTransportRuntimeFeedbackReasonSpelling(
    SpatialTransportRuntimeFeedbackReason reason) {
  switch (reason) {
  case SpatialTransportRuntimeFeedbackReason::ExactClosedStorageWait:
    return "exact_closed_storage_wait";
  case SpatialTransportRuntimeFeedbackReason::MissingOwnerReferences:
    return "missing_owner_references";
  case SpatialTransportRuntimeFeedbackReason::OwnerMismatch:
    return "owner_mismatch";
  case SpatialTransportRuntimeFeedbackReason::MissingWaitCycle:
    return "missing_wait_cycle";
  case SpatialTransportRuntimeFeedbackReason::MissingOutputBackpressure:
    return "missing_output_backpressure";
  case SpatialTransportRuntimeFeedbackReason::ProjectionMismatch:
    return "projection_mismatch";
  case SpatialTransportRuntimeFeedbackReason::NoAlternativeTraversal:
    return "no_alternative_traversal";
  case SpatialTransportRuntimeFeedbackReason::CandidateCapacityOverflow:
    return "candidate_capacity_overflow";
  }
  llvm_unreachable("unknown Spatial transport runtime feedback reason");
}

llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts) {
  SpatialTransportRuntimeFeedback result;
  result.parentMapping = parentMapping;
  result.owners = closedWait.ownerReferences;
  if (!result.owners)
    return result;

  auto system = mapping::importSystemMapping(parentMapping, artifacts);
  if (!system)
    return system.takeError();
  auto dataflow =
      ::dataflow::importCanonicalDataflow(result.owners->dataflow, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto tech = mapping::importTechMapping(result.owners->techMapping, artifacts);
  if (!tech)
    return tech.takeError();
  auto spatial =
      mapping::importSpatialMapping(result.owners->spatialMapping, artifacts);
  if (!spatial)
    return spatial.takeError();
  auto fabric =
      ::loom::fabric::importEntireFabricRoot(result.owners->fabric, artifacts);
  if (!fabric)
    return fabric.takeError();

  if (system->view().dataflowIdentity() != dataflowView->identity() ||
      spatial->view().dataflowIdentity() != dataflowView->identity() ||
      spatial->view().techMappingIdentity() != tech->view().identity() ||
      spatial->view().fabricIdentity() != fabric->view().identity() ||
      tech->view().dataflowIdentity() != dataflowView->identity() ||
      tech->view().fabricIdentity() != fabric->view().identity() ||
      !llvm::is_contained(
          system->view().executionBindings().spatialMappingImports(),
          result.owners->spatialMapping)) {
    result.disposition =
        SpatialTransportRuntimeFeedbackDisposition::Unsupported;
    result.reason = SpatialTransportRuntimeFeedbackReason::OwnerMismatch;
    return result;
  }
  if (closedWait.actorWaitCycle.empty()) {
    result.reason = SpatialTransportRuntimeFeedbackReason::MissingWaitCycle;
    return result;
  }

  std::map<std::string, SpatialTransportRepairAlternative> alternatives;
  bool projectionMismatch = false;
  for (const auto &edge : closedWait.actorWaitCycle) {
    if (edge.kind !=
        sim::CgraClosedWaitSetDiagnostic::ActorWaitKind::OutputBackpressure)
      continue;
    ++result.outputBackpressureEdgeCount;
    for (const auto &transfer : closedWait.transfers) {
      if (!transfer.blocked ||
          transfer.producerActorOrdinal != edge.waitingActorOrdinal ||
          !transfer.producer)
        continue;
      if (llvm::Error error = dataflowView->validate(*transfer.producer)) {
        llvm::consumeError(std::move(error));
        projectionMismatch = true;
        continue;
      }
      auto producerMatches = producerMatchesSemanticActor(
          *dataflowView, *transfer.producer, transfer.producerActorOrdinal);
      if (!producerMatches)
        return producerMatches.takeError();
      if (!*producerMatches) {
        projectionMismatch = true;
        continue;
      }
      const auto route = llvm::find_if(
          spatial->view().routeTrees(), [&](const auto &candidate) {
            return candidate.logicalNet == *transfer.producer;
          });
      if (route == spatial->view().routeTrees().end()) {
        projectionMismatch = true;
        continue;
      }
      auto encodedProducer = ::dataflow::encodeDataflowReference(
          dataflowView->identity(), *transfer.producer);
      if (!encodedProducer)
        return encodedProducer.takeError();
      bool selectedRuntimeTraversal = false;
      const auto appendTargets =
          [&](llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef>
                  targets) -> llvm::Error {
        for (const auto &traversal : targets) {
          if (llvm::Error error =
                  ::loom::fabric::validateFabricRef(fabric->view(), traversal))
            return error;
          if (!routeSelectsTraversal(*route, traversal))
            continue;
          selectedRuntimeTraversal = true;
          const std::string key = alternativeKey(*encodedProducer, traversal);
          alternatives.try_emplace(key, SpatialTransportRepairAlternative{
                                            *transfer.producer, traversal});
        }
        return llvm::Error::success();
      };
      if (llvm::Error error = appendTargets(transfer.blockingTraversals))
        return std::move(error);
      if (llvm::Error error =
              appendTargets(transfer.blockingDownstreamTraversals))
        return std::move(error);
      if (!selectedRuntimeTraversal) {
        projectionMismatch = true;
        continue;
      }
      ++result.exactBlockedTransferCount;
    }
  }
  if (result.outputBackpressureEdgeCount == 0) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::MissingOutputBackpressure;
    return result;
  }
  if (projectionMismatch) {
    result.disposition =
        SpatialTransportRuntimeFeedbackDisposition::Unsupported;
    result.reason = SpatialTransportRuntimeFeedbackReason::ProjectionMismatch;
    return result;
  }
  if (alternatives.empty() || fabric->view().admittedTraversals().size() < 2) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::NoAlternativeTraversal;
    return result;
  }
  if (alternatives.size() > maximumTransportRepairAlternatives) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::CandidateCapacityOverflow;
    return result;
  }
  for (auto &[key, alternative] : alternatives)
    result.alternatives.push_back(std::move(alternative));
  result.disposition = SpatialTransportRuntimeFeedbackDisposition::Exact;
  result.reason = SpatialTransportRuntimeFeedbackReason::ExactClosedStorageWait;
  return result;
}

void emitSpatialTransportRuntimeFeedback(
    const SpatialTransportRuntimeFeedback &feedback) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "spatial_transport_runtime_feedback";
        fields["parent_mapping"] =
            feedback.parentMapping
                ? llvm::json::Value(formatArtifactIdentityHex(
                      feedback.parentMapping->artifact))
                : llvm::json::Value(nullptr);
        fields["disposition"] =
            spatialTransportRuntimeFeedbackDispositionSpelling(
                feedback.disposition);
        fields["reason"] =
            spatialTransportRuntimeFeedbackReasonSpelling(feedback.reason);
        fields["output_backpressure_edge_count"] =
            feedback.outputBackpressureEdgeCount;
        fields["exact_blocked_transfer_count"] =
            feedback.exactBlockedTransferCount;
        fields["repair_alternative_count"] = feedback.alternatives.size();
      });
}

} // namespace loom::dse
