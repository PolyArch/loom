#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMSPATIALCATALOG_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMSPATIALCATALOG_H

#include "Common/ComponentViewDigest.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressProjection.h"
#include "PnR/SpatialRecurrenceTiming.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr::detail {

struct SpatialCatalogGraphProgress final {
  ::dataflow::GraphRef graph;
  std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
      routeObligations;
};

struct SpatialCatalogEntry final {
  ArtifactRootReference reference;
  std::shared_ptr<const ::loom::mapping::FinalizedSpatialMapping> mapping;
  std::uint64_t moduleDependencyOrdinal = 0;
  std::vector<::dataflow::GraphRef> covers;
  std::vector<SpatialCatalogGraphProgress> graphProgress;
  std::vector<std::uint64_t> graphStaticSchedulePressures;
  std::vector<SpatialRecurrenceTimingProjection> graphRecurrenceTimings;
  std::uint64_t worstRouteArrivalDelayQuanta = 0;
  std::uint64_t totalRouteNegativeSlackQuanta = 0;
  ComponentViewDigest::Storage physicalTimingProfileDigest{};
  ::loom::fabric::FabricPhysicalTimingProfileKind physicalTimingProfileKind =
      ::loom::fabric::FabricPhysicalTimingProfileKind::NormalizedHeuristic;
};

struct SpatialCatalogImportStatistics final {
  std::uint64_t techMappingImportRequests = 0;
  std::uint64_t techMappingImportHits = 0;
  std::uint64_t techMappingImportMisses = 0;
};

llvm::Expected<std::vector<SpatialCatalogEntry>> importSpatialCatalog(
    llvm::ArrayRef<ArtifactRootReference> references,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    const ArtifactStore &store,
    const ::loom::mapping::SpatialMappingImportContext *imports = nullptr,
    SpatialCatalogImportStatistics *statistics = nullptr);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMSPATIALCATALOG_H
