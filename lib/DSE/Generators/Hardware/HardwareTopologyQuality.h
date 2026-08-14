#ifndef LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H
#define LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H

#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "llvm/Support/Error.h"

namespace loom::dse {

struct HardwareTopologyQualityClosure final {
  std::vector<fabric::FabricTopologyQualityReport> reports;
  fabric::FabricTopologyDseQuality totals;
};

inline llvm::Expected<HardwareTopologyQualityClosure>
analyzeHardwareTopologyQuality(const fabric::FabricArtifactView &fabric) {
  auto reports = fabric::analyzeFabricTopologyQualityClosure(fabric);
  if (!reports)
    return reports.takeError();
  HardwareTopologyQualityClosure result{std::move(*reports), {}};
  for (const fabric::FabricTopologyQualityReport &report : result.reports) {
    const fabric::FabricTopologyDseQuality quality =
        fabric::projectFabricTopologyDseQuality(report);
    result.totals.unscheduledMemoryCount += quality.unscheduledMemoryCount;
    result.totals.scheduleSupplyGap += quality.scheduleSupplyGap;
    result.totals.matchingMemoryUnreachablePeCount +=
        quality.matchingMemoryUnreachablePeCount;
    result.totals.matchingMemoryTotalReachableHops +=
        quality.matchingMemoryTotalReachableHops;
    result.totals.capabilityCoverageUnreachablePeCount +=
        quality.capabilityCoverageUnreachablePeCount;
    result.totals.capabilityCoverageTotalReachableHops +=
        quality.capabilityCoverageTotalReachableHops;
    result.totals.isolatedCapabilitySupportingPeCount +=
        quality.isolatedCapabilitySupportingPeCount;
  }
  return result;
}

inline llvm::Error
validateHardwareTopologyQuality(const fabric::FabricArtifactView &fabric) {
  auto quality = analyzeHardwareTopologyQuality(fabric);
  if (!quality)
    return quality.takeError();
  return llvm::Error::success();
}

} // namespace loom::dse

#endif // LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H
