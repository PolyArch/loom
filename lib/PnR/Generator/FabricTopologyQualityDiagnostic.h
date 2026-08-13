#ifndef LOOM_PNR_GENERATOR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H
#define LOOM_PNR_GENERATOR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H

#include "Common/MappingDebugLog.h"
#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

llvm::Expected<fabric::FabricTopologyQualityReport>
analyzeAndEmitFabricTopologyQuality(const fabric::FabricArtifactView &fabric,
                                    mapping_debug::Stage stage);

} // namespace loom::pnr

#endif // LOOM_PNR_GENERATOR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H
