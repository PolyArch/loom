#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMSERVICELEGCARRIER_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMSERVICELEGCARRIER_H

#include "llvm/Support/Error.h"

namespace fabric {
class SystemOp;
}

namespace loom::fabric {
class FabricSystemRootView;
}

namespace loom::fabric::detail {

/// Coalesces authoring rows with the same service-leg key by carrier-set
/// union. Strict import never calls this normalization.
llvm::Error
normalizeSystemServiceLegCarrierAttachments(::fabric::SystemOp root);

/// Validates the complete canonical relation against the sealed System root.
llvm::Error
validateSystemServiceLegCarrierAttachments(const FabricSystemRootView &system);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMSERVICELEGCARRIER_H
