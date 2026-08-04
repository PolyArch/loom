#include "Hardware/RTL/CommonSkeleton.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SeqToSV.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error skeletonError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_skeleton_invalid: " + message);
}

bool isFabricOperationLeaf(circt::hw::HWModuleGeneratedOp module) {
  return module.getGeneratorKind() == fabricOperationGeneratorSchemaSymbol;
}

llvm::Error verifyNoUnresolvedFabricOperationLeaves(mlir::ModuleOp module) {
  bool unresolved = false;
  module.walk([&](circt::hw::HWModuleGeneratedOp leaf) {
    unresolved |= isFabricOperationLeaf(leaf);
  });
  if (unresolved)
    return skeletonError("unresolved Loom Fabric operation leaf reached "
                         "SystemVerilog export");
  return llvm::Error::success();
}

} // namespace

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("common CIRCT module does not verify");

  std::set<mlir::Operation *> declaredLeaves;
  bool hasInvalidSchema = false;
  module.walk([&](circt::hw::HWModuleGeneratedOp leaf) {
    if (!isFabricOperationLeaf(leaf))
      return;
    auto schema =
        mlir::cast<circt::hw::HWGeneratorSchemaOp>(leaf.getGeneratorKindOp());
    hasInvalidSchema |=
        schema.getDescriptor() != fabricOperationGeneratorDescriptor;
    declaredLeaves.insert(leaf.getOperation());
  });
  if (hasInvalidSchema)
    return skeletonError("Loom Fabric operation schema has an unexpected "
                         "descriptor");

  std::set<mlir::Operation *> associatedLeaves;
  std::set<std::vector<std::uint8_t>> associatedOccurrences;
  for (const FabricOperationLeafAssociation &association : operationLeaves) {
    circt::hw::HWModuleGeneratedOp leaf = association.module;
    if (!leaf || leaf->getParentOfType<mlir::ModuleOp>() != module ||
        !isFabricOperationLeaf(leaf))
      return skeletonError(
          "operation association does not name a Loom leaf in this module");
    if (!associatedLeaves.insert(leaf.getOperation()).second)
      return skeletonError("Loom Fabric operation leaf is associated more than "
                           "once");

    std::vector<std::uint8_t> occurrenceBytes =
        fabric::canonicalFabricBytes(association.occurrence);
    if (!associatedOccurrences.insert(std::move(occurrenceBytes)).second)
      return skeletonError("Fabric operation occurrence is associated more "
                           "than once");
    if (llvm::Error error =
            fabric::validateFabricRef(fabric, association.occurrence)) {
      llvm::consumeError(std::move(error));
      return skeletonError(
          "association does not resolve to a concrete Fabric operation "
          "capability");
    }
    if (!fabric.resolvedFabricOpCapability(association.occurrence))
      return skeletonError(
          "association does not resolve to a concrete Fabric operation "
          "capability");
  }

  if (declaredLeaves != associatedLeaves)
    return skeletonError(
        "Loom Fabric operation leaf has no exact Fabric occurrence "
        "association");
  return llvm::Error::success();
}

llvm::Expected<std::string>
lowerAndExportSpecializedSystemVerilog(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("specialized CIRCT module does not verify");
  if (llvm::Error error = verifyNoUnresolvedFabricOperationLeaves(module))
    return std::move(error);

  circt::LowerSeqToSVOptions loweringOptions;
  loweringOptions.disableRegRandomization = true;
  mlir::PassManager pipeline(module.getContext());
  pipeline.addPass(circt::createLowerSeqToSVPass(loweringOptions));
  if (mlir::failed(pipeline.run(module)))
    return skeletonError("Seq-to-SV lowering failed");
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("lowered HW/SV module does not verify");
  if (llvm::Error error = verifyNoUnresolvedFabricOperationLeaves(module))
    return std::move(error);

  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  if (mlir::failed(circt::exportVerilog(module, output)))
    return skeletonError("ExportVerilog rejected the specialized module");
  return output.str().str();
}

} // namespace loom::hardware::rtl
