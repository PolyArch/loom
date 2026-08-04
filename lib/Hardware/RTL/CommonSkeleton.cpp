#include "Hardware/RTL/CommonSkeleton.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/Transport.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SeqToSV.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>
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

struct BoundaryPassthroughPlan final {
  const ModuleBoundaryTransportPortProjection *input;
  const ModuleBoundaryTransportPortProjection *output;
  ::fabric::DataPathType inputType;
  ::fabric::DataPathType outputType;
};

void appendBoundaryPorts(
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    const ModuleBoundaryTransportPortProjection &boundary) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  if (boundary.data)
    append(*boundary.data);
  if (boundary.tag)
    append(*boundary.tag);
  append(boundary.valid);
  append(boundary.ready);
}

} // namespace

llvm::Expected<ModuleRootCirctSkeleton>
buildModuleRootCirctSkeleton(mlir::MLIRContext &context,
                             const fabric::FabricArtifactView &fabric) {
  const auto root = fabric.moduleRootTemplate();
  if (!root)
    return skeletonError("Module skeleton construction requires a Module "
                         "root");

  if (!fabric.moduleBoundaryTransportAttachments().empty() ||
      !fabric.pointConnections().empty() || !fabric.peOccurrences().empty() ||
      !fabric.fuOccurrences().empty() || !fabric.memoryOccurrences().empty() ||
      !fabric.switchOccurrences().empty() ||
      !fabric.fifoOccurrences().empty() ||
      !fabric.boundaryOccurrences().empty())
    return skeletonError(
        "Module boundary constructor accepts no internal resource structure");

  mlir::OpBuilder builder(&context);
  auto projections = deriveModuleBoundaryTransportPorts(builder, fabric);
  if (!projections)
    return projections.takeError();

  const std::uint64_t inputCount = fabric.moduleBoundaryEndpointCount(
      *root, fabric::FabricPortDirection::Input);
  const std::uint64_t outputCount = fabric.moduleBoundaryEndpointCount(
      *root, fabric::FabricPortDirection::Output);
  if (projections->size() != inputCount + outputCount)
    return skeletonError(
        "Module boundary constructor accepts no memory-plane boundary");

  std::vector<const ModuleBoundaryTransportPortProjection *> inputs(inputCount);
  std::vector<const ModuleBoundaryTransportPortProjection *> outputs(
      outputCount);
  for (const ModuleBoundaryTransportPortProjection &projection : *projections) {
    if (projection.boundary.module != *root)
      return skeletonError("Module boundary projection names another root");
    auto &index =
        projection.boundary.direction == fabric::FabricPortDirection::Input
            ? inputs
            : outputs;
    if (projection.boundary.ordinal >= index.size() ||
        index[projection.boundary.ordinal])
      return skeletonError("Module boundary projection is not one-to-one");
    index[projection.boundary.ordinal] = &projection;
  }

  std::vector<bool> usedInputs(inputCount, false);
  std::vector<bool> usedOutputs(outputCount, false);
  std::vector<BoundaryPassthroughPlan> passthroughs;
  passthroughs.reserve(fabric.moduleBoundaryTransportPassthroughs().size());
  for (const fabric::FabricModuleBoundaryTransportPassthroughView &passthrough :
       fabric.moduleBoundaryTransportPassthroughs()) {
    if (passthrough.input.module != *root ||
        passthrough.output.module != *root ||
        passthrough.input.direction != fabric::FabricPortDirection::Input ||
        passthrough.output.direction != fabric::FabricPortDirection::Output ||
        passthrough.input.ordinal >= inputs.size() ||
        passthrough.output.ordinal >= outputs.size() ||
        !inputs[passthrough.input.ordinal] ||
        !outputs[passthrough.output.ordinal] ||
        usedInputs[passthrough.input.ordinal] ||
        usedOutputs[passthrough.output.ordinal])
      return skeletonError("Module boundary passthrough is not one-to-one");
    const auto inputType =
        fabric.moduleBoundaryEndpointDataPath(passthrough.input);
    const auto outputType =
        fabric.moduleBoundaryEndpointDataPath(passthrough.output);
    if (!inputType || !outputType)
      return skeletonError("Module boundary passthrough has no token type");
    usedInputs[passthrough.input.ordinal] = true;
    usedOutputs[passthrough.output.ordinal] = true;
    passthroughs.push_back({inputs[passthrough.input.ordinal],
                            outputs[passthrough.output.ordinal], *inputType,
                            *outputType});
  }
  if (llvm::is_contained(usedInputs, false) ||
      llvm::is_contained(usedOutputs, false))
    return skeletonError("Module boundary-only construction requires every "
                         "token port to be connected");

  llvm::SmallVector<circt::hw::PortInfo, 16> inputPorts;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputPorts;
  for (const ModuleBoundaryTransportPortProjection &projection : *projections)
    appendBoundaryPorts(inputPorts, outputPorts, projection);

  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_module"),
      circt::hw::ModulePortInfo(inputPorts, outputPorts),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        for (const BoundaryPassthroughPlan &passthrough : passthroughs) {
          if (materializationError)
            return;
          ForwardTransportSignals source{
              accessor.getInput(passthrough.input->valid.getName()),
              passthrough.input->data
                  ? std::optional<mlir::Value>{accessor.getInput(
                        passthrough.input->data->getName())}
                  : std::nullopt,
              passthrough.input->tag
                  ? std::optional<mlir::Value>{accessor.getInput(
                        passthrough.input->tag->getName())}
                  : std::nullopt};
          auto adapted = adaptForwardTransportSignals(
              bodyBuilder, location, passthrough.inputType,
              passthrough.outputType, std::move(source));
          if (!adapted) {
            materializationError = llvm::toString(adapted.takeError());
            return;
          }
          accessor.setOutput(passthrough.output->valid.getName(),
                             adapted->valid);
          if (passthrough.output->data)
            accessor.setOutput(passthrough.output->data->getName(),
                               *adapted->payload);
          if (passthrough.output->tag)
            accessor.setOutput(passthrough.output->tag->getName(),
                               *adapted->tag);
          accessor.setOutput(
              passthrough.input->ready.getName(),
              accessor.getInput(passthrough.output->ready.getName()));
        }
      });
  if (materializationError)
    return skeletonError(*materializationError);

  ModuleRootCirctSkeleton result{std::move(module), {}};
  if (llvm::Error error = verifyCommonCirctSkeleton(*result.module, fabric,
                                                    result.operationLeaves))
    return std::move(error);
  return result;
}

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
  if (llvm::Error error = verifySpecializedCirctModule(module))
    return std::move(error);

  circt::LowerSeqToSVOptions loweringOptions;
  loweringOptions.disableRegRandomization = true;
  mlir::PassManager pipeline(module.getContext());
  pipeline.addPass(circt::createLowerSeqToSVPass(loweringOptions));
  if (mlir::failed(pipeline.run(module)))
    return skeletonError("Seq-to-SV lowering failed");
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
  return verifyNoUnresolvedFabricOperationLeaves(module);
}

} // namespace loom::hardware::rtl
