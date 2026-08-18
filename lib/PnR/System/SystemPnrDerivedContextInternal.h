#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXTINTERNAL_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXTINTERNAL_H

#include "PnR/System/SystemPnrDerivedContext.h"
#include "PnR/System/SystemPnrProblem.h"

#include "PnR/EndpointRoutingTopology.h"
#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "SystemSpatialCatalog.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace loom::pnr::detail {

inline constexpr llvm::StringLiteral systemStaticContextAlgorithmIdentity =
    "loom.pnr.system_static_context.1";
inline constexpr llvm::StringLiteral systemActiveContextAlgorithmIdentity =
    "loom.pnr.system_active_context.2";

struct SystemStaticContextStorage final {
  std::array<std::uint8_t, 32> key{};
  ArtifactIdentity systemIdentity;
  std::shared_ptr<const FrozenEndpointRoutingTopology> routingTopology;
  std::shared_ptr<const std::vector<FrozenSystemSpatialTargetClass>>
      targetClasses;
  std::shared_ptr<const std::vector<::loom::fabric::AccCoreOccurrenceRef>>
      accCores;
  std::shared_ptr<const std::vector<PnrIndex>> accCoreTargetClasses;
  std::shared_ptr<const std::vector<FrozenSystemInstructionUsePatternDomain>>
      instructionUsePatterns;
  std::shared_ptr<const std::vector<FrozenSystemConsistencyUsePatternDomain>>
      consistencyUsePatterns;
  std::optional<::loom::fabric::FabricTopologyQualityReport> topologyQuality;
  SystemStaticContextStatistics statistics;
};

struct SystemActiveContextStorage final {
  std::array<std::uint8_t, 32> key{};
  ArtifactIdentity dataflowIdentity;
  ArtifactIdentity systemIdentity;
  ArtifactIdentity constraintIdentity;
  std::vector<ArtifactRootReference> spatialMappings;
  std::shared_ptr<const ::loom::mapping::SpatialMappingImportContext>
      spatialMappingImports;
  std::shared_ptr<const std::vector<SpatialCatalogEntry>> spatialCatalog;
  std::shared_ptr<const std::vector<PnrIndex>> spatialMappingTargetClasses;
  SystemActiveContextStatistics statistics;
};

llvm::Expected<FrozenSystemSpatialTargetClass> deriveSystemSpatialTargetClass(
    const ::loom::fabric::FabricArtifactView &module);

std::string
systemSpatialTargetClassKey(const FrozenSystemSpatialTargetClass &targetClass);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMPNRDERIVEDCONTEXTINTERNAL_H
