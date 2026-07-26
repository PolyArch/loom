#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H

#include "Fabric/IR/FuCapabilityDomain.h"
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

/// Returns the owning FU's finite correlated capability domain expressed in
/// `canonicalNodeOrder`. A missing authoring attribute is accepted only when
/// the physical graph has one unambiguous template.
llvm::Expected<::fabric::FuCapabilityDomainRecord>
canonicalizeFabricFuCapabilityDomain(
    ::fabric::FuOp fu, llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder);

llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
deriveFabricFuCapabilityTemplates(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
