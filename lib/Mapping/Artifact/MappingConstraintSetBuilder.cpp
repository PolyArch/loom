#include "Mapping/Artifact/MappingConstraintSet.h"

#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
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

} // namespace

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeEmptySpatialMappingConstraintSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store) {
  mlir::MLIRContext context;
  context.loadDialect<::mapping::MappingDialect>();
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());
  auto root = ::mapping::ConstraintsSpatialOp::create(
      builder, builder.getUnknownLoc(),
      identityAttr(&context, dataflow.identity()),
      identityAttr(&context, techMapping.identity()),
      identityAttr(&context, fabric.identity()));
  root.getBody().emplaceBlock();
  return finalizeSpatialMappingConstraintSet(root, dataflow, techMapping,
                                             fabric, store);
}

} // namespace loom::mapping
