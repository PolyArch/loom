#ifndef LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H
#define LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H

#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "llvm/Support/Error.h"

namespace loom::dse {

inline llvm::Error
validateHardwareTopologyQuality(const fabric::FabricArtifactView &fabric) {
  auto reports = fabric::analyzeFabricTopologyQualityClosure(fabric);
  if (!reports)
    return reports.takeError();
  return llvm::Error::success();
}

} // namespace loom::dse

#endif // LOOM_DSE_GENERATORS_HARDWARE_HARDWARETOPOLOGYQUALITY_H
