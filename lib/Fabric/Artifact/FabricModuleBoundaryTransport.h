#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEBOUNDARYTRANSPORT_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEBOUNDARYTRANSPORT_H

#include "FabricCanonicalLabeling.h"

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace fabric {
class ModuleOp;
}

namespace loom::fabric::detail {

struct FabricArtifactViewData;

/// Derives every Module-boundary token relation from canonical Module SSA.
/// Resource attachments and direct boundary passthroughs remain disjoint.
llvm::Error appendFabricModuleBoundaryTransportRelations(
    ::fabric::ModuleOp root, FabricModuleTemplateRef module,
    const llvm::DenseMap<::mlir::Operation *, const FabricEntityCarrier *>
        &carrierByOp,
    FabricArtifactViewData &data);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEBOUNDARYTRANSPORT_H
