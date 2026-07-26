#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace mlir {
class Operation;
}

namespace fabric {
class FuOp;
}

namespace loom::fabric::detail {

llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
deriveFabricFuCapabilityTemplates(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
