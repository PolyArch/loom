#include "DSE/SpatialRuntimeFeedback.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/ComponentViewDigest.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace loom::dse {
namespace {

using Diagnostic = sim::CgraClosedWaitSetDiagnostic;
using EdgeKind = Diagnostic::WaitEdgeKind;

constexpr std::uint64_t kAbsent = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kAbsent32 = std::numeric_limits<std::uint32_t>::max();

void appendU32Be(std::string &output, std::uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte)
    output.push_back(static_cast<char>(value >> (8 * (3 - byte))));
}

void appendU64Be(std::string &output, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    output.push_back(static_cast<char>(value >> (8 * (7 - byte))));
}

void appendFramed(std::string &output, llvm::StringRef value) {
  appendU32Be(output, static_cast<std::uint32_t>(value.size()));
  output.append(value.data(), value.size());
}

void appendBytes(std::string &output, llvm::ArrayRef<std::uint8_t> bytes) {
  appendFramed(output,
               llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                               bytes.size()));
}

/// Minimal-width big-endian encoding of one exact tag value. The queue class of
/// a per-tag virtual channel is named by its exact bits, so the digest quotes
/// those bits rather than any plan-local ordinal.
void appendTag(std::string &output, const llvm::APInt &value) {
  const unsigned byteCount = std::max(1u, (value.getActiveBits() + 7) / 8);
  const llvm::APInt extended = value.zextOrTrunc(byteCount * 8);
  std::string bits;
  bits.reserve(byteCount);
  for (unsigned byte = 0; byte < byteCount; ++byte)
    bits.push_back(static_cast<char>(
        extended.extractBitsAsZExtValue(8, 8 * (byteCount - 1 - byte))));
  appendFramed(output, bits);
}

void appendOptionalTag(std::string &output,
                       const std::optional<llvm::APInt> &value) {
  appendU32Be(output, value ? 1 : 0);
  if (value)
    appendTag(output, *value);
}

void appendQueueClass(std::string &output,
                      const Diagnostic::WaitQueueClass &queueClass) {
  appendU32Be(output, queueClass.tagLocal ? 1 : 0);
  appendTag(output, queueClass.tagValue);
}

void appendOwner(std::string &output, const Diagnostic::WaitOwnerKey &owner) {
  appendU32Be(output, static_cast<std::uint32_t>(owner.owner.index()));
  if (const auto *firing =
          std::get_if<Diagnostic::WaitActorFiringKey>(&owner.owner)) {
    appendU64Be(output, firing->semanticActorOrdinal);
    appendU64Be(output, firing->occurrenceOrdinal);
    return;
  }
  const auto &storage = std::get<Diagnostic::WaitStorageQueueKey>(owner.owner);
  appendU32Be(output, static_cast<std::uint32_t>(storage.domain));
  appendU64Be(output, storage.ordinal);
  appendQueueClass(output, storage.queueClass);
}

/// The complete typed content of one certificate edge, in a fixed field order.
std::string edgeKey(const Diagnostic::WaitEdge &edge) {
  std::string key;
  appendOwner(key, edge.from);
  appendOwner(key, edge.to);
  appendU32Be(key, static_cast<std::uint32_t>(edge.kind));
  appendU32Be(key, edge.waitingInputOrdinal);
  appendU64Be(key, edge.waitingChannelOrdinal);
  appendU64Be(key, edge.bindingOrdinal);
  appendU64Be(key, edge.occurrenceOrdinal);
  appendU64Be(key, edge.storageOrdinal);
  appendU32Be(key, edge.fifoOccurrence ? 1 : 0);
  if (edge.fifoOccurrence)
    appendBytes(key,
                ::loom::fabric::canonicalFabricBytes(*edge.fifoOccurrence));
  appendU32Be(key, edge.storageCapacity);
  appendU32Be(key, edge.storageOccupancy);
  appendU32Be(key, edge.awaitedClassPosition);
  appendOptionalTag(key, edge.awaitedTagValue);
  appendOptionalTag(key, edge.headTagValue);
  appendU64Be(key, edge.headBindingOrdinal);
  appendU64Be(key, edge.headOccurrenceOrdinal);
  appendU64Be(key, edge.headDestinationActorOrdinal);
  appendU32Be(key, edge.headDestinationInputOrdinal);
  appendU64Be(key, edge.headDestinationChannelOrdinal);
  return key;
}

using StorageDomain = Diagnostic::WaitStorageDomain;

/// The typed storage owner an edge endpoint names, when it names one at all.
std::optional<Diagnostic::WaitStorageQueueKey>
storageOwner(const Diagnostic::WaitOwnerKey &owner) {
  if (const auto *storage =
          std::get_if<Diagnostic::WaitStorageQueueKey>(&owner.owner))
    return *storage;
  return std::nullopt;
}

bool isTraversalStorage(
    const std::optional<Diagnostic::WaitStorageQueueKey> &key) {
  return key && key->domain == StorageDomain::TraversalStorage;
}

/// The certificate edges this projection treats as essential. Selection is by
/// typed owner domain, never by a bare ordinal: an operand queue owns no
/// traversal, so a `StorageConsumer` out of one is not a transport fact, while
/// a `StorageOrder` into a traversal store stays a transport fact even when its
/// source is an operand queue.
bool crossesTransportStorage(const Diagnostic::WaitEdge &edge) {
  switch (edge.kind) {
  case EdgeKind::StorageOrder:
  case EdgeKind::StorageDownstream:
    // The awaited traversal store is the destination of the order arc.
    return isTraversalStorage(storageOwner(edge.to));
  case EdgeKind::StorageConsumer:
    // The consuming arc leaves the traversal store that holds the token.
    return isTraversalStorage(storageOwner(edge.from));
  case EdgeKind::ActorOutputBackpressure:
    // Backpressure names the same storage relation from the producer side, but
    // only when the store it is blocked on is traversal storage.
    return isTraversalStorage(storageOwner(edge.to));
  case EdgeKind::ActorMissingInput:
  case EdgeKind::OperandQueueWait:
    return false;
  }
  return false;
}

llvm::Expected<std::string>
producerKey(const ArtifactIdentity &dataflow,
            const ::dataflow::CanonicalGraphProducerEndpointRef &ref) {
  auto encoded = ::dataflow::encodeDataflowReference(dataflow, ref);
  if (!encoded)
    return encoded.takeError();
  return std::string(reinterpret_cast<const char *>(encoded->data()),
                     encoded->size());
}

/// Canonical sort key of one projected literal, so the emitted literal sequence
/// does not depend on certificate discovery order.
llvm::Expected<std::string>
literalKey(const ArtifactIdentity &dataflow,
           const mapping::SpatialNoGoodLiteral &literal) {
  std::string key;
  if (const auto *uses =
          std::get_if<mapping::SpatialNetUsesTraversalLiteral>(&literal)) {
    appendU32Be(key, 0);
    auto producer = producerKey(dataflow, uses->producer);
    if (!producer)
      return producer.takeError();
    appendFramed(key, *producer);
    appendU32Be(key, uses->consumer ? 1 : 0);
    if (uses->consumer) {
      auto encoded =
          ::dataflow::encodeDataflowReference(dataflow, *uses->consumer);
      if (!encoded)
        return encoded.takeError();
      appendBytes(key, *encoded);
    }
    appendBytes(key, ::loom::fabric::canonicalFabricBytes(uses->traversal));
    return key;
  }
  const auto &attachment =
      std::get<mapping::SpatialTransferAttachmentEqualsLiteral>(literal);
  appendU32Be(key, 1);
  auto producer = producerKey(dataflow, attachment.terminal.producer);
  if (!producer)
    return producer.takeError();
  appendFramed(key, *producer);
  appendU32Be(key, attachment.terminal.consumer ? 1 : 0);
  if (attachment.terminal.consumer) {
    auto encoded = ::dataflow::encodeDataflowReference(
        dataflow, *attachment.terminal.consumer);
    if (!encoded)
      return encoded.takeError();
    appendBytes(key, *encoded);
  }
  appendBytes(key, ::loom::fabric::canonicalFabricBytes(attachment.endpoint));
  return key;
}

/// The outcome of resolving the destination a certificate edge names. An absent
/// destination is a legitimately unqualified whole-net fact; a named but
/// unresolvable destination is a projection failure and must never silently
/// widen to the whole net.
enum class ConsumerResolution : std::uint8_t {
  Absent,
  Resolved,
  Unresolved,
};

struct ResolvedConsumer final {
  ConsumerResolution resolution = ConsumerResolution::Absent;
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
};

/// Resolves the exact consumer endpoint a certificate edge names, from the
/// graph-local semantic actor ordinal and input ordinal the runtime quotes.
/// Distinguishes "named no destination" from "named a destination that does not
/// resolve"; the caller rejects the latter rather than weakening the claim.
ResolvedConsumer
resolveConsumer(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
                std::uint64_t destinationActorOrdinal,
                std::uint32_t destinationInputOrdinal) {
  if (destinationActorOrdinal == kAbsent &&
      destinationInputOrdinal == kAbsent32)
    return {ConsumerResolution::Absent, std::nullopt};
  // A half-named destination is malformed, not unqualified.
  if (destinationActorOrdinal == kAbsent ||
      destinationInputOrdinal == kAbsent32)
    return {ConsumerResolution::Unresolved, std::nullopt};
  std::optional<::dataflow::GraphRef> graph;
  if (const auto *result =
          std::get_if<::dataflow::ActorTokenResultRef>(&producer)) {
    auto producerActor = dataflow.resolve(result->actor);
    if (!producerActor) {
      llvm::consumeError(producerActor.takeError());
      return {ConsumerResolution::Unresolved, std::nullopt};
    }
    graph = producerActor->graph;
  } else if (const auto *ingress =
                 std::get_if<::dataflow::GraphIngressTokenRef>(&producer)) {
    graph = std::visit([](const auto &value) { return value.graph; }, *ingress);
  } else {
    return {ConsumerResolution::Unresolved, std::nullopt};
  }
  std::uint64_t graphLocalOrdinal = 0;
  for (const auto &candidate : dataflow.actors()) {
    if (candidate.graph != *graph)
      continue;
    if (graphLocalOrdinal == destinationActorOrdinal) {
      return {ConsumerResolution::Resolved,
              ::dataflow::CanonicalGraphConsumerEndpointRef(
                  ::dataflow::ActorTokenOperandRef{
                      candidate.ref,
                      ::dataflow::StructuralOrdinal(destinationInputOrdinal)})};
    }
    ++graphLocalOrdinal;
  }
  return {ConsumerResolution::Unresolved, std::nullopt};
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
  case SpatialTransportRuntimeFeedbackReason::ParentConstraintRejection:
    return "parent_constraint_rejection";
  case SpatialTransportRuntimeFeedbackReason::UnprovenWaitCertificate:
    return "unproven_wait_certificate";
  case SpatialTransportRuntimeFeedbackReason::UnjoinedCertificateEdge:
    return "unjoined_certificate_edge";
  case SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence:
    return "unbound_runtime_evidence";
  case SpatialTransportRuntimeFeedbackReason::UnboundConstraintLineage:
    return "unbound_constraint_lineage";
  case SpatialTransportRuntimeFeedbackReason::EmptyLiteralSet:
    return "empty_literal_set";
  }
  llvm_unreachable("unknown Spatial transport runtime feedback reason");
}

llvm::Expected<SpatialWaitCertificateDigest>
computeSpatialWaitCertificateDigest(
    const sim::CgraClosedWaitSetDiagnostic &closedWait) {
  std::vector<std::string> keys;
  keys.reserve(closedWait.waitCertificate.size());
  for (const auto &edge : closedWait.waitCertificate)
    keys.push_back(edgeKey(edge));
  // Canonical edge order is the sorted order of the complete typed edge
  // content, so the digest cannot depend on the order the runtime emitted the
  // edges in. Duplicate edges are kept: an edge multiset is part of the fact.
  llvm::sort(keys);

  std::string canonical;
  appendU64Be(canonical, static_cast<std::uint64_t>(keys.size()));
  for (const std::string &key : keys)
    appendFramed(canonical, key);

  const llvm::StringRef domain = SpatialWaitCertificateDigest::domain;
  auto digest = computeComponentViewDigest(
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(domain.data()), domain.size()),
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(canonical.data()),
          canonical.size()));
  if (!digest)
    return digest.takeError();
  return SpatialWaitCertificateDigest(*digest);
}

std::string
formatSpatialWaitCertificateDigest(const SpatialWaitCertificateDigest &digest) {
  return formatComponentViewDigestHex(digest.digest());
}

llvm::Expected<SpatialTransportRuntimeFeedback>
deriveSpatialTransportRuntimeFeedback(
    const ArtifactRootReference &parentSpatialMapping,
    const ArtifactRootReference &parentConstraints,
    const SpatialTransportRuntimeEvidence &runtimeEvidence,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> parentSystemMapping) {
  SpatialTransportRuntimeFeedback result;
  result.parentSpatialMapping = parentSpatialMapping;
  result.parentMapping = std::move(parentSystemMapping);
  result.parentConstraints = parentConstraints;
  result.runtimeEvidence = runtimeEvidence.evidence;
  result.owners = closedWait.ownerReferences;
  result.certificateEdgeCount = closedWait.waitCertificate.size();
  if (!result.owners)
    return result;
  const sim::CgraExecutionOwnerReferences &owners = *result.owners;

  // Only an exact, closed, proven certificate is projectable. There is
  // no fallback to actorWaitCycle, transferWaitCycle, or a first blocked FIFO.
  if (closedWait.waitProofFailure || closedWait.waitCertificate.empty() ||
      !sim::verifyClosedWaitCertificateClosure(closedWait)) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnprovenWaitCertificate;
    return result;
  }

  // The evidence reference is validated through its own schema and store owner.
  // This proves the object exists, is an evaluation.evidence root, and that its
  // identity matches its bytes; it also yields the exact Request it came from.
  auto evidenceProjection =
      ::loom::evaluation::importEvaluationEvidenceDependencyProjection(
          runtimeEvidence.evidence, artifacts);
  if (!evidenceProjection) {
    llvm::consumeError(evidenceProjection.takeError());
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence;
    return result;
  }
  // The typed Request view is the only owner that proves which Mapping this
  // certificate was observed under, and under which model. Every check below is
  // a mechanical equality over already-imported owners, so this projection
  // needs no case resolution and no blob store.
  //
  // Role membership is not enough: each role must be a singleton naming exactly
  // the certificate owner it corresponds to. A generic dependency list can name
  // roots for unrelated roles, so it is never consulted here.
  const ::loom::evaluation::EvaluationRequest &request =
      runtimeEvidence.requestView;
  const ArtifactRootReference requestRoot =
      ::loom::evaluation::evaluationRequestReference(request);
  if (evidenceProjection->request != requestRoot) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence;
    return result;
  }
  if (request.modelBinding().descriptorRef() !=
      ::loom::evaluation::models::cgraSimulationModelDescriptorRef()) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence;
    return result;
  }
  const auto exactRole = [&](::loom::evaluation::CaseSubjectRoleRef role,
                             const ArtifactRootReference &expected) {
    const auto subjects = request.subjectBindings().subjects(role);
    return subjects.size() == 1 && subjects.front() == expected;
  };
  if (!exactRole(::loom::evaluation::models::cgraSimulationProgramRole(),
                 owners.dataflow) ||
      !exactRole(::loom::evaluation::models::cgraSimulationHardwareRole(),
                 owners.fabric) ||
      !exactRole(::loom::evaluation::models::cgraSimulationSpatialMappingRole(),
                 owners.spatialMapping) ||
      owners.spatialMapping != parentSpatialMapping) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnboundRuntimeEvidence;
    return result;
  }
  result.evaluationRequest = requestRoot;

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
  auto fabric =
      ::loom::fabric::importEntireFabricRoot(result.owners->fabric, artifacts);
  if (!fabric)
    return fabric.takeError();
  auto spatial = mapping::importSpatialMapping(parentSpatialMapping, artifacts);
  if (!spatial)
    return spatial.takeError();
  auto constraints =
      mapping::importSpatialMappingConstraintSet(parentConstraints, artifacts);
  if (!constraints)
    return constraints.takeError();
  std::optional<mapping::FinalizedSystemMapping> parentSystem;
  if (result.parentMapping) {
    auto imported =
        mapping::importSystemMapping(*result.parentMapping, artifacts);
    if (!imported)
      return imported.takeError();
    parentSystem.emplace(std::move(*imported));
  }

  // The certificate's own owner references, the parent Mapping, and the
  // parent constraint root must name one exact D/T/F/S closure. The no-good is
  // bound by that tuple and never by a SystemMapping.
  if (parentSpatialMapping != result.owners->spatialMapping ||
      spatial->view().dataflowIdentity() != dataflowView->identity() ||
      spatial->view().techMappingIdentity() != tech->view().identity() ||
      spatial->view().fabricIdentity() != fabric->view().identity() ||
      tech->view().dataflowIdentity() != dataflowView->identity() ||
      tech->view().fabricIdentity() != fabric->view().identity() ||
      constraints->view().dataflowIdentity() != dataflowView->identity() ||
      constraints->view().techMappingIdentity() != tech->view().identity() ||
      constraints->view().fabricIdentity() != fabric->view().identity() ||
      (parentSystem &&
       (parentSystem->view().dataflowIdentity() != dataflowView->identity() ||
        !llvm::is_contained(
            parentSystem->view().executionBindings().spatialMappingImports(),
            parentSpatialMapping)))) {
    result.reason = SpatialTransportRuntimeFeedbackReason::OwnerMismatch;
    return result;
  }
  if (llvm::Error error = mapping::admitSpatialMappingConstraints(
          *dataflowView, tech->view(), fabric->view(), constraints->view(),
          spatial->view())) {
    llvm::consumeError(std::move(error));
    result.reason =
        SpatialTransportRuntimeFeedbackReason::ParentConstraintRejection;
    return result;
  }

  // The digest covers the complete typed certificate.
  auto digest = computeSpatialWaitCertificateDigest(closedWait);
  if (!digest)
    return digest.takeError();
  result.certificateDigest = *digest;

  const auto findTransfer =
      [&](std::uint64_t binding,
          std::uint64_t occurrence) -> const Diagnostic::Transfer * {
    for (const auto &transfer : closedWait.transfers)
      if (transfer.bindingOrdinal == binding &&
          transfer.occurrenceOrdinal == occurrence)
        return &transfer;
    return nullptr;
  };

  const auto findRoute =
      [&](const ::dataflow::CanonicalGraphProducerEndpointRef &producer)
      -> const mapping::SpatialRouteTreeView * {
    const auto route =
        llvm::find_if(spatial->view().routeTrees(), [&](const auto &candidate) {
          return candidate.logicalNet == producer;
        });
    return route == spatial->view().routeTrees().end() ? nullptr : &*route;
  };

  /// The semantic actor ordinal an edge endpoint names, when that endpoint is
  /// an actor firing rather than a storage queue.
  const auto firingActor = [](const Diagnostic::WaitOwnerKey &owner)
      -> std::optional<std::uint64_t> {
    if (const auto *firing =
            std::get_if<Diagnostic::WaitActorFiringKey>(&owner.owner))
      return firing->semanticActorOrdinal;
    return std::nullopt;
  };

  /// The one traversal vector an edge is about. A downstream edge is a
  /// statement about the downstream storage, so quoting the upstream vector too
  /// would widen the clause past what the certificate proves. The storage
  /// ordinal must match the transfer's own blocking storage exactly, or the
  /// edge and the transfer are not describing the same queue.
  const auto edgeTraversals = [](const Diagnostic::WaitEdge &edge,
                                 const Diagnostic::Transfer &transfer)
      -> std::optional<
          llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef>> {
    if (edge.kind == EdgeKind::StorageDownstream) {
      if (edge.storageOrdinal != transfer.blockingDownstreamStorageOrdinal)
        return std::nullopt;
      return llvm::ArrayRef(transfer.blockingDownstreamTraversals);
    }
    if (edge.storageOrdinal != transfer.blockingStorageOrdinal)
      return std::nullopt;
    return llvm::ArrayRef(transfer.blockingTraversals);
  };

  std::map<std::string, mapping::SpatialNoGoodLiteral> literals;
  const auto remember =
      [&](mapping::SpatialNoGoodLiteral literal) -> llvm::Error {
    auto key = literalKey(dataflowView->identity(), literal);
    if (!key)
      return key.takeError();
    literals.try_emplace(std::move(*key), std::move(literal));
    return llvm::Error::success();
  };

  /// Projects one side of one edge onto the exact parent RouteTree.
  ///
  /// `destinationActor`/`destinationInput` must be the fields that describe
  /// *this* side's destination; they are never borrowed from the other side. A
  /// side narrows to one branch exactly when the certificate names a consumer
  /// that is a sink of this route, and stays route-wide only when the
  /// certificate carries no consumer owner at all.
  const auto projectSide = [&](const Diagnostic::WaitEdge &edge,
                               const Diagnostic::Transfer &transfer,
                               std::optional<std::uint64_t> destinationActor,
                               std::uint32_t destinationInput, bool requireSink,
                               bool emitAttachment) -> llvm::Expected<bool> {
    if (!transfer.producer)
      return false;
    if (llvm::Error error = dataflowView->validate(*transfer.producer)) {
      llvm::consumeError(std::move(error));
      return false;
    }
    const mapping::SpatialRouteTreeView *route = findRoute(*transfer.producer);
    if (!route)
      return false;

    if (destinationActor.has_value() != (destinationInput != kAbsent32))
      return false;

    ResolvedConsumer resolved{ConsumerResolution::Absent, std::nullopt};
    if (destinationActor)
      resolved = resolveConsumer(*dataflowView, *transfer.producer,
                                 *destinationActor, destinationInput);
    if (resolved.resolution == ConsumerResolution::Unresolved)
      return false;
    const std::optional<::dataflow::CanonicalGraphConsumerEndpointRef>
        &consumer = resolved.consumer;
    if (consumer)
      if (llvm::Error error = dataflowView->validate(*consumer)) {
        llvm::consumeError(std::move(error));
        return false;
      }
    // A terminal consumption fact is a claim about one exact terminal.
    if (requireSink && !consumer)
      return false;

    const mapping::SpatialRouteSinkView *branchSink = nullptr;
    if (consumer)
      for (const auto &sink : route->sinks)
        if (sink.sink == *consumer) {
          branchSink = &sink;
          break;
        }
    if (consumer && !branchSink)
      return false;

    std::vector<::loom::fabric::FabricPhysicalTraversalRef> branch;
    if (branchSink) {
      auto walked = mapping::spatialRouteBranchTraversals(*route, *branchSink);
      if (!walked) {
        llvm::consumeError(walked.takeError());
        return false;
      }
      branch = std::move(*walked);
    }

    auto targets = edgeTraversals(edge, transfer);
    if (!targets)
      return false;

    bool projected = false;
    for (const auto &traversal : *targets) {
      if (llvm::Error error =
              ::loom::fabric::validateFabricRef(fabric->view(), traversal)) {
        llvm::consumeError(std::move(error));
        return false;
      }
      if (!mapping::spatialRouteTreeSelectsTraversal(*route, traversal))
        continue;
      if (branchSink && !llvm::is_contained(branch, traversal))
        continue;
      if (llvm::Error error = remember(mapping::SpatialNetUsesTraversalLiteral{
              *transfer.producer, consumer, traversal}))
        return std::move(error);
      if (route->localTraversal && *route->localTraversal == traversal)
        if (llvm::Error error =
                remember(mapping::SpatialTransferAttachmentEqualsLiteral{
                    mapping::SpatialConstraintTransferTerminal{
                        *transfer.producer, std::nullopt},
                    route->rootEndpoint}))
          return std::move(error);
      if (branchSink && branchSink->localTraversal &&
          *branchSink->localTraversal == traversal)
        if (llvm::Error error =
                remember(mapping::SpatialTransferAttachmentEqualsLiteral{
                    mapping::SpatialConstraintTransferTerminal{
                        *transfer.producer, consumer},
                    route->nodes[branchSink->nodeOrdinal].endpoint}))
          return std::move(error);
      projected = true;
    }

    if (emitAttachment) {
      if (!branchSink)
        return false;
      if (llvm::Error error =
              remember(mapping::SpatialTransferAttachmentEqualsLiteral{
                  mapping::SpatialConstraintTransferTerminal{*transfer.producer,
                                                             consumer},
                  route->nodes[branchSink->nodeOrdinal].endpoint}))
        return std::move(error);
      projected = true;
    }
    return projected;
  };

  // Every essential edge must project completely.
  for (const auto &edge : closedWait.waitCertificate) {
    if (!crossesTransportStorage(edge))
      continue;
    ++result.projectedEdgeCount;
    if (edge.kind == EdgeKind::ActorOutputBackpressure)
      ++result.outputBackpressureEdgeCount;

    const auto reject = [&]() {
      result.reason =
          SpatialTransportRuntimeFeedbackReason::UnjoinedCertificateEdge;
      return result;
    };

    if (edge.bindingOrdinal == kAbsent || edge.occurrenceOrdinal == kAbsent)
      return reject();
    const Diagnostic::Transfer *awaited =
        findTransfer(edge.bindingOrdinal, edge.occurrenceOrdinal);
    if (!awaited)
      return reject();

    // Which certificate fields qualify the edge's own transfer depends
    // on which end of the storage relation that transfer sits at. A consumption
    // or downstream edge is a statement about the token already resident in the
    // storage, so the edge's own ordinals name the head and the head
    // destination names its exact sink. An ordering or output-backpressure edge
    // is a statement about a token still waiting to enter, so the blocked actor
    // firing plus the input ordinal it waits on name the awaited sink.
    const bool terminal = edge.kind == EdgeKind::StorageConsumer;
    const bool edgeTransferIsHead =
        terminal || edge.kind == EdgeKind::StorageDownstream;
    const bool headNamed = edge.headBindingOrdinal != kAbsent &&
                           edge.headOccurrenceOrdinal != kAbsent;
    const bool headIsEdgeTransfer =
        headNamed && edge.headBindingOrdinal == edge.bindingOrdinal &&
        edge.headOccurrenceOrdinal == edge.occurrenceOrdinal;
    const bool requiresHead = edge.kind == EdgeKind::StorageOrder ||
                              edge.kind == EdgeKind::StorageDownstream ||
                              terminal;
    // Order, downstream, and consumption facts all name the resident head.
    // Downstream and consumption edges are about that head itself.
    if ((requiresHead && !headNamed) ||
        ((terminal || edge.kind == EdgeKind::StorageDownstream) &&
         !headIsEdgeTransfer))
      return reject();

    std::optional<std::uint64_t> ownActor;
    std::uint32_t ownInput = kAbsent32;
    if (edgeTransferIsHead) {
      if (edge.headDestinationActorOrdinal != kAbsent)
        ownActor = edge.headDestinationActorOrdinal;
      ownInput = edge.headDestinationInputOrdinal;
    } else if (const std::optional<std::uint64_t> waiting =
                   firingActor(edge.from);
               waiting && edge.waitingInputOrdinal != kAbsent32) {
      ownActor = waiting;
      ownInput = edge.waitingInputOrdinal;
    }

    auto awaitedOutcome = projectSide(edge, *awaited, ownActor, ownInput,
                                      /*requireSink=*/terminal,
                                      /*emitAttachment=*/terminal);
    if (!awaitedOutcome)
      return awaitedOutcome.takeError();
    if (!*awaitedOutcome)
      return reject();
    ++result.exactBlockedTransferCount;

    // A head that is a genuinely different transfer is its own exact Mapping
    // choice and is joined independently. When the head is the edge transfer it
    // was already projected above; projecting it twice would double the
    // accounting for one semantic fact.
    if (headNamed && !headIsEdgeTransfer) {
      const Diagnostic::Transfer *head =
          findTransfer(edge.headBindingOrdinal, edge.headOccurrenceOrdinal);
      if (!head)
        return reject();
      const std::optional<std::uint64_t> headActor =
          edge.headDestinationActorOrdinal == kAbsent
              ? std::nullopt
              : std::optional<std::uint64_t>(edge.headDestinationActorOrdinal);
      auto headOutcome =
          projectSide(edge, *head, headActor, edge.headDestinationInputOrdinal,
                      /*requireSink=*/false, /*emitAttachment=*/false);
      if (!headOutcome)
        return headOutcome.takeError();
      if (!*headOutcome)
        return reject();
    }
  }

  if (result.projectedEdgeCount == 0) {
    result.reason =
        SpatialTransportRuntimeFeedbackReason::UnjoinedCertificateEdge;
    return result;
  }
  // The clause must be non-empty.
  if (literals.empty()) {
    result.reason = SpatialTransportRuntimeFeedbackReason::EmptyLiteralSet;
    return result;
  }

  for (auto &[key, literal] : literals) {
    (void)key;
    result.literals.push_back(literal);
  }

  // The older alternative shape is a mechanical projection of the
  // canonical literals, never an independent derivation.
  for (const auto &literal : result.literals)
    if (const auto *uses =
            std::get_if<mapping::SpatialNetUsesTraversalLiteral>(&literal))
      result.alternatives.push_back(
          SpatialTransportRepairAlternative{uses->producer, uses->traversal});

  auto published = mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
      parentConstraints, result.literals, artifacts);
  if (!published)
    return published.takeError();
  result.constraintSet = published->reference();
  result.disposition = SpatialTransportRuntimeFeedbackDisposition::Exact;
  result.reason = SpatialTransportRuntimeFeedbackReason::ExactClosedStorageWait;
  return result;
}

void emitSpatialTransportRuntimeFeedback(
    const SpatialTransportRuntimeFeedback &feedback) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SpatialPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        const auto reference =
            [](const std::optional<ArtifactRootReference> &value) {
              return value ? llvm::json::Value(
                                 formatArtifactIdentityHex(value->artifact))
                           : llvm::json::Value(nullptr);
            };
        fields["operation"] = "spatial_transport_runtime_feedback";
        fields["parent_mapping"] = reference(feedback.parentMapping);
        fields["parent_spatial_mapping"] =
            reference(feedback.parentSpatialMapping);
        fields["parent_constraints"] = reference(feedback.parentConstraints);
        fields["runtime_evidence"] = reference(feedback.runtimeEvidence);
        fields["evaluation_request"] = reference(feedback.evaluationRequest);
        fields["constraint_set"] = reference(feedback.constraintSet);
        fields["disposition"] =
            spatialTransportRuntimeFeedbackDispositionSpelling(
                feedback.disposition);
        fields["reason"] =
            spatialTransportRuntimeFeedbackReasonSpelling(feedback.reason);
        fields["certificate_digest"] =
            feedback.certificateDigest
                ? llvm::json::Value(formatSpatialWaitCertificateDigest(
                      *feedback.certificateDigest))
                : llvm::json::Value(nullptr);
        if (feedback.owners) {
          fields["dataflow"] =
              formatArtifactIdentityHex(feedback.owners->dataflow.artifact);
          fields["tech_mapping"] =
              formatArtifactIdentityHex(feedback.owners->techMapping.artifact);
          fields["fabric"] =
              formatArtifactIdentityHex(feedback.owners->fabric.artifact);
          fields["spatial_mapping"] = formatArtifactIdentityHex(
              feedback.owners->spatialMapping.artifact);
        }
        fields["certificate_edge_count"] = feedback.certificateEdgeCount;
        fields["projected_edge_count"] = feedback.projectedEdgeCount;
        fields["output_backpressure_edge_count"] =
            feedback.outputBackpressureEdgeCount;
        fields["exact_blocked_transfer_count"] =
            feedback.exactBlockedTransferCount;
        fields["literal_count"] = feedback.literals.size();
        fields["repair_alternative_count"] = feedback.alternatives.size();
      });
}

} // namespace loom::dse
