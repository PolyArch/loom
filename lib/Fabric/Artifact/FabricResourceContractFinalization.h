#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICRESOURCECONTRACTFINALIZATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICRESOURCECONTRACTFINALIZATION_H

#include "FabricCanonicalLabeling.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace loom::fabric::detail {

llvm::Expected<std::optional<::fabric::ResourceContract>>
validateFabricResourceContract(::mlir::Operation *operation,
                               const FabricCanonicalLabeling &labeling);

llvm::Error
materializeFabricResourceContracts(::fabric::ModuleOp root,
                                   const FabricCanonicalLabeling &labeling);

llvm::Error
validateFabricResourceContracts(::fabric::ModuleOp root,
                                const FabricCanonicalLabeling &labeling);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICRESOURCECONTRACTFINALIZATION_H
