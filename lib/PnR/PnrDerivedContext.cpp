#include "PnrDerivedContextInternal.h"

#include "Fabric/Artifact/FabricArtifactCodec.h"

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
    llvm::ArrayRef<HandshakeOwnerModel> models) {
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
  bytes += models.size() * sizeof(HandshakeOwnerModel);
  for (const HandshakeOwnerModel &model : models) {
    bytes += model.nodes().size() * sizeof(HandshakeOwnerNode);
    bytes += model.arcs().size() * sizeof(HandshakeOwnerArc);
    bytes += model.fragments().size() * sizeof(HandshakeActivationFragment);
    bytes +=
        model.fragmentContributionOrdinals().size() * sizeof(std::uint32_t);
    bytes +=
        model.traversalWitnesses().size() * sizeof(FabricPhysicalTraversalRef);
  }
  return bytes;
}

std::uint64_t loom::pnr::detail::timingContextRetainedBytes(
    const FrozenSpatialRoutingGraph &routing) {
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

const FabricDerivedContextStatistics &
loom::pnr::FabricDerivedContextBundle::statistics() const {
  return storage_->statistics;
}

const FabricHandshakeContext &
loom::pnr::FabricDerivedContextBundle::handshakeContext() const {
  return storage_->staticContext->handshake;
}

void loom::pnr::emitFabricDerivedContextStatistics(
    const FabricDerivedContextBundle &bundle, mapping_debug::Stage stage,
    std::uint64_t staticHits, std::uint64_t staticMisses,
    std::uint64_t timingHits, std::uint64_t timingMisses) {
  const FabricDerivedContextStatistics &statistics = bundle.statistics();
  const auto emit = [&](llvm::StringRef kind,
                        const DerivedContextConstructionStatistics &context,
                        std::uint64_t hits, std::uint64_t misses) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, stage,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = kind;
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
          fields["handshake_node_count"] = statistics.handshakeNodeCount;
          fields["handshake_arc_count"] = statistics.handshakeArcCount;
          fields["handshake_fragment_count"] =
              statistics.handshakeFragmentCount;
        });
  };
  emit("fabric_static", statistics.staticContext, staticHits, staticMisses);
  emit("fabric_timing", statistics.timingContext, timingHits, timingMisses);
}
