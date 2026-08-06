#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

::mapping::ArtifactIdentityAttr identityAttr(mlir::MLIRContext *context,
                                             const ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, identity.bytes()));
}

llvm::Expected<::mapping::RootThreadLaunchRefAttr>
rootThreadLaunchAttr(mlir::MLIRContext *context,
                     const ArtifactIdentity &dataflowIdentity,
                     ::dataflow::RootThreadLaunchRef reference) {
  auto bytes = ::dataflow::encodeDataflowReference(dataflowIdentity, reference);
  if (!bytes)
    return bytes.takeError();
  return ::mapping::RootThreadLaunchRefAttr::get(context,
                                                 denseBytes(context, *bytes));
}

} // namespace

llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeEmptySystemMappingConstraintSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
    const ArtifactStore &store) {
  if (rootThreadLaunches.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "system_mapping_constraint_set_invalid: root launch set is empty");
  mlir::MLIRContext context;
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());

  std::vector<mlir::Attribute> rootAttributes;
  rootAttributes.reserve(rootThreadLaunches.size());
  for (::dataflow::RootThreadLaunchRef reference : rootThreadLaunches) {
    auto attribute =
        rootThreadLaunchAttr(&context, dataflow.identity(), reference);
    if (!attribute)
      return attribute.takeError();
    rootAttributes.push_back(*attribute);
  }

  auto root = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      identityAttr(&context, dataflow.identity()),
      identityAttr(&context, fabric.artifact().identity()),
      builder.getArrayAttr(rootAttributes), builder.getArrayAttr({}));
  root.getBody().emplaceBlock();
  return finalizeSystemMappingConstraintSet(root, dataflow, fabric, store);
}

} // namespace loom::mapping
