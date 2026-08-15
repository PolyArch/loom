#ifndef LOOM_PNR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H
#define LOOM_PNR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H

#include "Common/MappingDebugLog.h"
#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace loom::pnr {

/// Computes the diagnostic-only report when Summary diagnostics are enabled.
/// A disabled stream performs no topology-quality analysis.
llvm::Expected<std::optional<fabric::FabricTopologyQualityReport>>
analyzeFabricTopologyQualityForDiagnostics(
    const fabric::FabricArtifactView &fabric);

void emitFabricTopologyQuality(
    const fabric::FabricTopologyQualityReport &report,
    mapping_debug::Stage stage);

} // namespace loom::pnr

#endif // LOOM_PNR_FABRICTOPOLOGYQUALITYDIAGNOSTIC_H
