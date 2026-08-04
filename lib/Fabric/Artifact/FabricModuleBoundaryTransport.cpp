#include "FabricModuleBoundaryTransport.h"

#include "../Identity/FabricArtifactViewInternal.h"
#include "FabricOperationTransport.h"

#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

using namespace mlir;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

llvm::Error loom::fabric::detail::appendFabricModuleBoundaryTransportRelations(
    ::fabric::ModuleOp root, FabricModuleTemplateRef module,
    const llvm::DenseMap<Operation *, const FabricEntityCarrier *> &carrierByOp,
    FabricArtifactViewData &data) {
  Block &body = root.getBody().front();
  auto appendDestination = [&](FabricOrdinal signatureOrdinal,
                               Value value) -> llvm::Error {
    if (isa<MemRefType>(value.getType()))
      return llvm::Error::success();
    OpOperand *directUse = nullptr;
    for (OpOperand &use : value.getUses()) {
      if (use.getOwner()->getBlock() != &body)
        continue;
      if (directUse)
        return invalid("a Module token input has multiple direct consumers");
      directUse = &use;
    }
    if (!directUse || isa<::fabric::YieldOp>(directUse->getOwner()))
      return llvm::Error::success();
    auto carrier = carrierByOp.find(directUse->getOwner());
    if (carrier == carrierByOp.end())
      return invalid("a connected Module token input has no occurrence owner");
    auto owner =
        projectFabricTransportOwner(carrier->second->kind, carrier->second->id);
    if (!owner)
      return invalid(
          "a connected Module token input has no transport endpoint owner");
    auto ordinal = resolveFabricTokenInputOrdinal(
        directUse->getOwner(), directUse->getOperandNumber());
    if (!ordinal)
      return ordinal.takeError();
    if (!*ordinal)
      return invalid("a token Module input resolved to a memory endpoint");
    data.moduleBoundaryTransportAttachments.push_back(
        {{module, FabricPortDirection::Input, signatureOrdinal},
         {*owner, **ordinal}});
    return llvm::Error::success();
  };

  for (auto [ordinal, argument] : llvm::enumerate(body.getArguments()))
    if (llvm::Error error = appendDestination(ordinal, argument))
      return error;

  auto yield = dyn_cast<::fabric::YieldOp>(body.getTerminator());
  if (!yield)
    return invalid("a finalized Module has no fabric.yield terminator");
  for (auto [signatureOrdinal, value] : llvm::enumerate(yield.getValues())) {
    if (isa<MemRefType>(value.getType()))
      continue;
    const FabricModuleBoundaryEndpointRef output{
        module, FabricPortDirection::Output,
        static_cast<FabricOrdinal>(signatureOrdinal)};
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() != &body)
        return invalid("a Module token output refers to a foreign argument");
      data.moduleBoundaryTransportPassthroughs.push_back(
          {{module, FabricPortDirection::Input, argument.getArgNumber()},
           output});
      continue;
    }
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return invalid("a Module token output has no canonical SSA source");
    Operation *source = result.getOwner();
    auto carrier = carrierByOp.find(source);
    if (carrier == carrierByOp.end())
      return invalid("a connected Module token output has no occurrence owner");
    auto owner =
        projectFabricTransportOwner(carrier->second->kind, carrier->second->id);
    if (!owner)
      return invalid(
          "a connected Module token output has no transport endpoint owner");
    auto ordinal =
        resolveFabricTokenOutputOrdinal(source, result.getResultNumber());
    if (!ordinal)
      return ordinal.takeError();
    if (!*ordinal)
      return invalid("a token Module output resolved to a memory endpoint");
    data.moduleBoundaryTransportAttachments.push_back(
        {output, {*owner, **ordinal}});
  }
  return llvm::Error::success();
}
