#include "PnrDerivedContextInternal.h"

#include "SpatialProgressIndex.h"

#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::pnr;

namespace {

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendBytes(std::vector<std::uint8_t> &preimage,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  appendU64Be(preimage, bytes.size());
  preimage.insert(preimage.end(), bytes.begin(), bytes.end());
}

void appendText(std::vector<std::uint8_t> &preimage, llvm::StringRef text) {
  appendBytes(preimage, llvm::ArrayRef<std::uint8_t>(
                            reinterpret_cast<const std::uint8_t *>(text.data()),
                            text.size()));
}

} // namespace

std::array<std::uint8_t, 32> loom::pnr::detail::deriveFabricStaticContextKey(
    const FabricArtifactView &fabric) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, fabricStaticContextAlgorithmIdentity);
  appendText(preimage, ::loom::fabric::fabricArtifactSchema.identity);
  appendU32Be(preimage, ::loom::fabric::fabricArtifactSchema.version.major);
  appendU32Be(preimage, ::loom::fabric::fabricArtifactSchema.version.minor);
  appendU32Be(preimage, static_cast<std::uint32_t>(fabric.rootKind()));
  appendBytes(preimage, fabric.identity().bytes());
  appendU32Be(preimage, sizeof(PnrIndex) * 8);
  return llvm::SHA256::hash(preimage);
}

std::array<std::uint8_t, 32> loom::pnr::detail::deriveFabricTimingContextKey(
    llvm::ArrayRef<std::uint8_t> staticKey,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, fabricTimingContextAlgorithmIdentity);
  appendBytes(preimage, staticKey);
  appendBytes(preimage, physicalTiming.schemaDescriptorBytes());
  appendBytes(preimage, physicalTiming.digest().bytes());
  return llvm::SHA256::hash(preimage);
}

std::uint64_t loom::pnr::detail::elapsedNanoseconds(
    std::chrono::steady_clock::time_point begin) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}

std::uint64_t loom::pnr::detail::staticContextRetainedBytes(
    const FrozenSpatialResourceIndex &resources,
    const FrozenEndpointRoutingTopology &topology,
    const FrozenSpatialTagContinuityIndex &tags,
    const FabricHandshakeContext &handshake,
    const std::optional<FabricTopologyQualityReport> &quality) {
  std::uint64_t bytes = sizeof(FabricStaticContext);
  bytes +=
      resources.resourceOwners().size() * sizeof(FrozenSpatialResourceOwner);
  bytes +=
      resources.resourceStates().size() * sizeof(FrozenSpatialResourceState);
  bytes += resources.capacityDimensions().size() *
           sizeof(FrozenSpatialCapacityDimension);
  bytes += resources.usePatterns().size() * sizeof(FrozenSpatialUsePattern);
  bytes += resources.claims().size() * sizeof(FrozenSpatialResourceClaim);
  bytes += resources.internalTransactions().size() *
           sizeof(FrozenSpatialInternalTransaction);
  bytes += resources.transactionClaims().size() * sizeof(PnrIndex);
  bytes +=
      resources.timingContracts().size() * sizeof(FrozenSpatialTimingContract);
  bytes += resources.eventRanks().size() * sizeof(std::uint32_t);
  bytes += resources.grantRequesterOrder().size() * sizeof(std::uint32_t);
  bytes += topology.endpoints().size() * sizeof(EndpointRoutingEndpoint);
  bytes += topology.traversals().size() * sizeof(EndpointRoutingTraversal);
  bytes += topology.traversalEndpoints().size() * sizeof(PnrIndex);
  bytes += topology.traversalReplicationGroups().size() * sizeof(PnrIndex);
  bytes += topology.arcs().size() * sizeof(EndpointRoutingArc);
  bytes += topology.arcSources().size() * sizeof(PnrIndex);
  bytes += topology.adjacencyOffsets().size() * sizeof(PnrIndex);
  bytes += topology.reverseAdjacencyOffsets().size() * sizeof(PnrIndex);
  bytes += topology.reverseArcOrdinals().size() * sizeof(PnrIndex);
  bytes +=
      topology.capacityCells().size() * sizeof(EndpointRoutingCapacityCell);
  bytes +=
      topology.capacityClaims().size() * sizeof(EndpointRoutingCapacityClaim);
  bytes += tags.points().size() * sizeof(FrozenSpatialTagContinuityPoint);
  bytes += tags.traversalPointOrdinals().size() * sizeof(PnrIndex);
  bytes +=
      tags.matchDomains().size() * sizeof(FabricPhysicalTagMatchDomainView);
  bytes += tags.endpointMatchDomainOrdinals().size() * sizeof(PnrIndex);
  bytes += handshake.statistics().retainedBytes;
  if (quality)
    bytes += topologyQualityRetainedBytes(*quality);
  return bytes;
}

std::uint64_t loom::pnr::detail::topologyQualityRetainedBytes(
    const FabricTopologyQualityReport &report) {
  std::uint64_t bytes = sizeof(FabricTopologyQualityReport);
  bytes += report.owners.capacity() * sizeof(FabricTopologyOwnerQuality);
  for (const FabricTopologyOwnerQuality &owner : report.owners) {
      bytes += owner.ports.capacity() * sizeof(FabricTopologyPortQuality);
      bytes += owner.distinctRoutingResources.capacity() *
               sizeof(FabricTransportEndpointOwnerRef);
      bytes += owner.distinctDirectResources.capacity() *
               sizeof(FabricTransportEndpointOwnerRef);
      for (const FabricTopologyPortQuality &port : owner.ports) {
        bytes += port.routingResources.capacity() *
                 sizeof(FabricTransportEndpointOwnerRef);
        bytes += port.directResources.capacity() *
                 sizeof(FabricTransportEndpointOwnerRef);
      }
  }
  bytes +=
      report.schedules.capacity() * sizeof(FabricTopologyScheduleQuality);
  bytes += report.capabilities.capacity() *
           sizeof(FabricTopologyCapabilityQuality);
  for (const FabricTopologyCapabilityQuality &capability : report.capabilities)
    bytes += capability.supportingPes.capacity() * sizeof(FabricPeOccurrenceRef);
  return bytes;
}

std::uint64_t loom::pnr::detail::topologyQualityDeterministicWork(
    const FabricTopologyQualityReport &report) {
  std::uint64_t work = report.owners.size() + report.schedules.size() +
                       report.capabilities.size();
  for (const FabricTopologyOwnerQuality &owner : report.owners)
    work += owner.ports.size() + owner.distinctRoutingResources.size() +
            owner.distinctDirectResources.size();
  for (const FabricTopologyCapabilityQuality &capability : report.capabilities)
    work += capability.supportingPes.size();
  return work;
}

std::uint64_t loom::pnr::detail::timingContextRetainedBytes(
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialProgressIndex &progressIndex) {
  std::uint64_t bytes = sizeof(FabricTimingContext) + sizeof(routing);
  bytes += routing.traversals().size() * sizeof(FrozenSpatialTraversal);
  bytes += routing.traversalResourceStates().size() * sizeof(PnrIndex);
  bytes += routing.routeClaims().size() * sizeof(FrozenSpatialRouteClaim);
  bytes += routing.traversalClaimKeys().size() * sizeof(PnrIndex);
  bytes += routing.capacityRouteClaimOffsets().size() * sizeof(PnrIndex);
  bytes += routing.capacityRouteClaims().size() * sizeof(PnrIndex);
  bytes += routing.routeClaimTraversalOffsets().size() * sizeof(PnrIndex);
  bytes += routing.routeClaimTraversals().size() * sizeof(PnrIndex);
  bytes += routing.traversalArcOffsets().size() * sizeof(PnrIndex);
  bytes += routing.traversalArcs().size() * sizeof(PnrIndex);
  bytes += progressIndex.finiteBufferOwners().size() *
           sizeof(::loom::fabric::FabricFifoOccurrenceRef);
  bytes += progressIndex.traversalOwnerOrdinals().size() * sizeof(PnrIndex);
  bytes += progressIndex.ownerTraversalOffsets().size() * sizeof(PnrIndex);
  bytes += progressIndex.ownerTraversals().size() * sizeof(PnrIndex);
  return bytes;
}

const ArtifactIdentity &
loom::pnr::FabricDerivedContextBundle::fabricIdentity() const {
  return storage_->staticContext->fabricIdentity;
}

const ComponentViewDigest::Storage &
loom::pnr::FabricDerivedContextBundle::physicalTimingDigestBytes() const {
  return storage_->timingContext->physicalTimingDigestBytes;
}

llvm::ArrayRef<std::uint8_t>
loom::pnr::FabricDerivedContextBundle::staticContextKey() const {
  return storage_->staticContext->key;
}

llvm::ArrayRef<std::uint8_t>
loom::pnr::FabricDerivedContextBundle::timingContextKey() const {
  return storage_->timingContext->key;
}

const FabricDerivedContextStatistics &
loom::pnr::FabricDerivedContextBundle::statistics() const {
  return storage_->statistics;
}

const FabricHandshakeContext &
loom::pnr::FabricDerivedContextBundle::handshakeContext() const {
  return storage_->staticContext->handshake;
}

const FabricTopologyQualityReport *
loom::pnr::FabricDerivedContextBundle::topologyQualityDiagnostic() const {
  const auto &quality = storage_->staticContext->topologyQuality;
  return quality ? &*quality : nullptr;
}

void loom::pnr::emitFabricDerivedContextStatistics(
    const FabricDerivedContextBundle &bundle, mapping_debug::Stage stage,
    std::uint64_t staticHits, std::uint64_t staticMisses,
    std::uint64_t timingHits, std::uint64_t timingMisses) {
  const FabricDerivedContextStatistics &statistics = bundle.statistics();
  const auto emit = [&](llvm::StringRef kind,
                        llvm::ArrayRef<std::uint8_t> key,
                        const DerivedContextConstructionStatistics &context,
                        std::uint64_t hits, std::uint64_t misses) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, stage,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = kind;
          fields["context_key"] = llvm::toHex(key, /*LowerCase=*/true);
          fields["cache_hits"] = hits;
          fields["cache_misses"] = misses;
          fields["construction_count"] = context.constructionCount;
          fields["construction_time_ns"] = context.constructionNanoseconds;
          fields["retained_bytes"] = context.retainedBytes;
          fields["deterministic_work"] = context.deterministicWork;
          fields["resource_owner_count"] = statistics.resourceOwnerCount;
          fields["endpoint_count"] = statistics.endpointCount;
          fields["traversal_count"] = statistics.traversalCount;
          fields["routing_arc_count"] = statistics.routingArcCount;
          fields["handshake_owner_count"] = statistics.handshakeOwnerCount;
          fields["handshake_structural_template_count"] =
              statistics.handshakeStructuralTemplateCount;
          fields["handshake_binding_instance_count"] =
              statistics.handshakeBindingInstanceCount;
          fields["handshake_structural_node_count"] =
              statistics.handshakeStructuralNodeCount;
          fields["handshake_structural_arc_count"] =
              statistics.handshakeStructuralArcCount;
          fields["handshake_structural_fragment_count"] =
              statistics.handshakeStructuralFragmentCount;
          fields["handshake_unconditional_arc_count"] =
              statistics.handshakeUnconditionalArcCount;
          fields["handshake_node_count"] = statistics.handshakeNodeCount;
          fields["handshake_arc_count"] = statistics.handshakeArcCount;
          fields["handshake_fragment_count"] =
              statistics.handshakeFragmentCount;
        });
  };
  emit("fabric_static", bundle.staticContextKey(), statistics.staticContext,
       staticHits, staticMisses);
  emit("fabric_timing", bundle.timingContextKey(), statistics.timingContext,
       timingHits, timingMisses);
}
