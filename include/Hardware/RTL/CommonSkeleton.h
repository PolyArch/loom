#ifndef LOOM_HARDWARE_RTL_COMMONSKELETON_H
#define LOOM_HARDWARE_RTL_COMMONSKELETON_H

#include "Fabric/Identity/FabricRefImport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
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

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves);

/// Verifies, lowers Seq to SV, verifies again, and exports SystemVerilog.
/// The input module is consumed by the lowering pipeline and must contain no
/// unresolved Loom Fabric operation leaf.
llvm::Expected<std::string>
lowerAndExportSpecializedSystemVerilog(mlir::ModuleOp module);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_COMMONSKELETON_H
