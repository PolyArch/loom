#ifndef LOOM_HARDWARE_RTL_COMMONSKELETON_H
#define LOOM_HARDWARE_RTL_COMMONSKELETON_H

#include "Fabric/Identity/FabricRefImport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::hardware::rtl {

inline constexpr llvm::StringLiteral fabricOperationGeneratorSchemaSymbol =
    "loom_fabric_operation";
inline constexpr llvm::StringLiteral fabricOperationGeneratorDescriptor =
    "loom.fabric.operation";

/// Transient association between one abstract CIRCT leaf and the exact
/// Fabric operation occurrence that owns its capability semantics.
struct FabricOperationLeafAssociation final {
  circt::hw::HWModuleGeneratedOp module;
  fabric::FabricFuOccurrenceNodeRef occurrence;
};

/// One standalone CIRCT container built from an exact Fabric Module root.
/// Operation-leaf associations remain transient handles into `module`.
struct ModuleRootCirctSkeleton final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<FabricOperationLeafAssociation> operationLeaves;
};

/// Builds one target-independent CIRCT module from a finalized Fabric Module
/// root. Validation and construction happen off to the side; failure publishes
/// no partial skeleton.
llvm::Expected<ModuleRootCirctSkeleton>
buildModuleRootCirctSkeleton(mlir::MLIRContext &context,
                             const fabric::FabricArtifactView &fabric);

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves);

/// Verifies a specialized module and rejects any remaining Loom abstract leaf.
llvm::Error verifySpecializedCirctModule(mlir::ModuleOp module);

/// Verifies, lowers Seq to SV, verifies again, and exports SystemVerilog.
/// The input module is consumed by the lowering pipeline and must contain no
/// unresolved Loom Fabric operation leaf.
llvm::Expected<std::string>
lowerAndExportSpecializedSystemVerilog(mlir::ModuleOp module);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_COMMONSKELETON_H
