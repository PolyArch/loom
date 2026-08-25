#ifndef LOOM_LIB_PNR_PNRDERIVEDCONTEXTINTERNAL_H
#define LOOM_LIB_PNR_PNRDERIVEDCONTEXTINTERNAL_H

#include "PnR/PnrDerivedContext.h"
#include "PnR/SpatialPnrProblem.h"

#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Artifact/FabricTopologyQuality.h"

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

inline constexpr llvm::StringLiteral fabricStaticContextAlgorithmIdentity =
    "loom.pnr.fabric_static_context.1";
inline constexpr llvm::StringLiteral fabricTimingContextAlgorithmIdentity =
    "loom.pnr.fabric_timing_context.1";

struct FabricStaticContext final {
  std::array<std::uint8_t, 32> key{};
  ArtifactIdentity fabricIdentity;
  std::shared_ptr<const FrozenSpatialResourceIndex> resources;
  std::shared_ptr<const FrozenEndpointRoutingTopology> routingTopology;
  std::shared_ptr<const FrozenSpatialTagContinuityIndex> tagContinuity;
  ::loom::fabric::FabricHandshakeContext handshake;
  std::optional<::loom::fabric::FabricTopologyQualityReport> topologyQuality;
};

struct FabricTimingContext final {
  std::array<std::uint8_t, 32> key{};
  ArtifactIdentity fabricIdentity;
  ComponentViewDigest::Storage physicalTimingDigestBytes{};
  std::shared_ptr<const FabricStaticContext> staticContext;
  std::shared_ptr<const FrozenSpatialRoutingGraph> routing;
  std::shared_ptr<const FrozenSpatialProgressIndex> progressIndex;
};

struct FabricDerivedContextStorage final {
  std::shared_ptr<const FabricStaticContext> staticContext;
  std::shared_ptr<const FabricTimingContext> timingContext;
  FabricDerivedContextStatistics statistics;
};

std::array<std::uint8_t, 32>
deriveFabricStaticContextKey(const ::loom::fabric::FabricArtifactView &fabric);

std::array<std::uint8_t, 32> deriveFabricTimingContextKey(
    llvm::ArrayRef<std::uint8_t> staticKey,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming);

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin);

std::uint64_t staticContextRetainedBytes(
    const FrozenSpatialResourceIndex &resources,
    const FrozenEndpointRoutingTopology &topology,
    const FrozenSpatialTagContinuityIndex &tags,
    const ::loom::fabric::FabricHandshakeContext &handshake,
    const std::optional<::loom::fabric::FabricTopologyQualityReport> &quality);

std::uint64_t topologyQualityDeterministicWork(
    const ::loom::fabric::FabricTopologyQualityReport &report);

std::uint64_t topologyQualityRetainedBytes(
    const ::loom::fabric::FabricTopologyQualityReport &report);

std::uint64_t timingContextRetainedBytes(
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialProgressIndex &progressIndex);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_PNRDERIVEDCONTEXTINTERNAL_H
