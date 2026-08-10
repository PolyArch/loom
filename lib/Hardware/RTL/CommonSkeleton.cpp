#include "Hardware/RTL/CommonSkeleton.h"

#include "Hierarchy/ModuleHierarchy.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Transport.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace loom::hardware::rtl {
char FabricStructuralLoweringUnsupportedError::ID = 0;

void FabricStructuralLoweringUnsupportedError::log(
    llvm::raw_ostream &stream) const {
  stream << "rtl_structural_lowering_unsupported: " << reason_;
}

std::error_code
FabricStructuralLoweringUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

llvm::Error skeletonError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_skeleton_invalid: " + message);
}

bool isFabricOperationLeaf(circt::hw::HWModuleGeneratedOp module) {
  return module.getGeneratorKind() == fabricOperationGeneratorSchemaSymbol;
}

llvm::Expected<std::set<std::vector<std::uint8_t>>>
expectedOperationOccurrences(
    const fabric::FabricSystemRootView &system,
    std::optional<fabric::SpatialCoreOccurrenceRef> spatialCore) {
  std::set<std::vector<std::uint8_t>> result;
  auto operations = enumerateFabricPhysicalOperations(system);
  if (!operations)
    return operations.takeError();
  for (const ResolvedFabricPhysicalOperation &operation : *operations) {
    if (spatialCore) {
      const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
          operation.physicalOccurrence.payload());
      if (internal.spatialCore != *spatialCore)
        continue;
    }
    if (!result
             .insert(fabric::canonicalFabricBytes(operation.physicalOccurrence))
             .second)
      return skeletonError(
          "Fabric operation occurrence inventory is not unique");
  }
  return result;
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

llvm::Error verifyNoUnresolvedStructuralLowering(mlir::ModuleOp module) {
  bool unresolved = false;
  module.walk([&](mlir::UnrealizedConversionCastOp) { unresolved = true; });
  if (unresolved)
    return skeletonError("unresolved structural lowering remains in CIRCT "
                         "module");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ModuleRootCirctSkeleton>
buildModuleRootCirctSkeleton(mlir::MLIRContext &context,
                             fabric::SpatialCoreOccurrenceRef spatialCore,
                             const FinalizedConfigurationABI &finalizedAbi) {
  const ConfigurationABI &configurationAbi = finalizedAbi.abi();
  auto fabricModule = resolveFabricSpatialCoreModule(
      configurationAbi.fabricSystem(), spatialCore);
  if (!fabricModule)
    return fabricModule.takeError();
  const fabric::FabricArtifactView &fabric = *fabricModule;
  const auto root = fabric.moduleRootTemplate();
  if (!root)
    return skeletonError("Module skeleton construction requires a Module "
                         "root");

  mlir::OpBuilder builder(&context);
  auto projections = deriveModuleBoundaryTransportPorts(builder, fabric);
  if (!projections)
    return projections.takeError();
  return hierarchy::buildModuleHierarchySkeleton(
      context, spatialCore, finalizedAbi, fabric, *projections);
}

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const ConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("common CIRCT module does not verify");
  if (llvm::Error error = verifyNoUnresolvedStructuralLowering(module))
    return error;

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
  std::optional<fabric::SpatialCoreOccurrenceRef> associatedSpatialCore;
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
    auto operation = resolveFabricPhysicalOperation(
        configurationAbi.fabricSystem(), association.occurrence);
    if (!operation) {
      llvm::consumeError(operation.takeError());
      return skeletonError(
          "association does not resolve to a concrete Fabric operation "
          "capability");
    }
    const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
        association.occurrence.payload());
    if (associatedSpatialCore && *associatedSpatialCore != internal.spatialCore)
      return skeletonError(
          "one Module skeleton associates multiple SpatialCore occurrences");
    associatedSpatialCore = internal.spatialCore;
    if (llvm::Error error = verifyFabricOperationLeafPorts(
            leaf, association.occurrence, *operation->capability,
            configurationAbi))
      return error;
  }

  if (declaredLeaves != associatedLeaves)
    return skeletonError(
        "Loom Fabric operation leaf has no exact Fabric occurrence "
        "association");
  auto expectedOccurrences = expectedOperationOccurrences(
      configurationAbi.fabricSystem(), associatedSpatialCore);
  if (!expectedOccurrences)
    return expectedOccurrences.takeError();
  if (*expectedOccurrences != associatedOccurrences)
    return skeletonError(
        llvm::Twine("operation association set does not exactly cover Fabric "
                    "operation occurrences: expected ") +
        llvm::Twine(expectedOccurrences->size()) + ", received " +
        llvm::Twine(associatedOccurrences.size()));
  return llvm::Error::success();
}

llvm::Expected<std::string>
lowerAndExportSpecializedSystemVerilog(mlir::ModuleOp module) {
  if (llvm::Error error = verifySpecializedCirctModule(module))
    return std::move(error);

  circt::LowerSeqToSVOptions loweringOptions;
  loweringOptions.disableRegRandomization = true;
  mlir::PassManager pipeline(module.getContext());
  pipeline.addPass(circt::createLowerSeqToSVPass(loweringOptions));
  circt::seq::HWMemSimImplOptions memoryOptions;
  memoryOptions.disableMemRandomization = true;
  memoryOptions.disableRegRandomization = true;
  pipeline.addPass(circt::seq::createHWMemSimImpl(memoryOptions));
  if (mlir::failed(pipeline.run(module)))
    return skeletonError("Seq and memory lowering failed");
  if (llvm::Error error = verifySpecializedCirctModule(module))
    return std::move(error);

  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  if (mlir::failed(circt::exportVerilog(module, output)))
    return skeletonError("ExportVerilog rejected the specialized module");
  return output.str().str();
}

llvm::Error verifySpecializedCirctModule(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("specialized CIRCT module does not verify");
  if (llvm::Error error = verifyNoUnresolvedStructuralLowering(module))
    return error;
  return verifyNoUnresolvedFabricOperationLeaves(module);
}

} // namespace loom::hardware::rtl
