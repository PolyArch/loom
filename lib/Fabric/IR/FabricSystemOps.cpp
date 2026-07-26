#include "Fabric/IR/FabricOps.h"

#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/MemoryServiceContract.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

using namespace mlir;
using namespace fabric;

namespace {

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

LogicalResult verifyClosedAttributes(Operation *operation) {
  for (NamedAttribute attribute : operation->getDiscardableAttrs())
    return operation->emitOpError("has non-canonical discardable attribute '")
           << attribute.getName() << "'";
  return success();
}

LogicalResult verifyInstructionCoreRecords(Operation *operation,
                                           DenseI8ArrayAttr architecture,
                                           DenseI8ArrayAttr microarchitecture,
                                           bool requireSpatialRuntimeServices) {
  auto decodedArchitecture =
      loom::fabric::decodeInstructionCoreArchitecturalContract(
          unsignedBytes(architecture));
  if (!decodedArchitecture)
    return operation->emitOpError("has invalid architecture record: ")
           << llvm::toString(decodedArchitecture.takeError());

  auto decodedMicroarchitecture =
      loom::fabric::decodeInstructionCoreMicroarchitecturalRealization(
          unsignedBytes(microarchitecture));
  if (!decodedMicroarchitecture)
    return operation->emitOpError("has invalid microarchitecture record: ")
           << llvm::toString(decodedMicroarchitecture.takeError());

  if (!requireSpatialRuntimeServices)
    return success();
  llvm::ArrayRef<loom::fabric::InstructionRuntimeService> services =
      decodedArchitecture->runtimeServices();
  if (!llvm::is_contained(
          services, loom::fabric::InstructionRuntimeService::ThreadDispatch) ||
      !llvm::is_contained(
          services, loom::fabric::InstructionRuntimeService::SpatialLaunch))
    return operation->emitOpError(
        "requires ThreadDispatch and SpatialLaunch runtime services");
  return success();
}

std::optional<std::uint64_t> entityId(Operation &operation) {
  auto attribute = operation.getAttrOfType<EntityIdAttr>("entity_id");
  if (!attribute)
    return std::nullopt;
  return attribute.getId();
}

} // namespace

LogicalResult SystemOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one declarative block");
  Block &block = getBody().front();
  if (!block.getArguments().empty())
    return emitOpError("body must not have block arguments");

  llvm::DenseSet<std::uint64_t> entityIds;
  for (Operation &operation : block) {
    if (!isa<SystemHostCoreOp, SystemAccCoreOp, SystemMemoryServiceOp>(
            operation))
      return operation.emitOpError(
          "is not in the closed fabric.system child catalog");
    if (std::optional<std::uint64_t> id = entityId(operation))
      if (!entityIds.insert(*id).second)
        return operation.emitOpError("duplicates Fabric EntityId ") << *id;
  }
  return success();
}

LogicalResult SystemHostCoreOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  return verifyInstructionCoreRecords(getOperation(), getArchitectureAttr(),
                                      getMicroarchitectureAttr(),
                                      /*requireSpatialRuntimeServices=*/false);
}

LogicalResult SystemAccCoreOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (failed(verifyInstructionCoreRecords(
          getOperation(), getArchitectureAttr(), getMicroarchitectureAttr(),
          /*requireSpatialRuntimeServices=*/true)))
    return failure();
  auto spatialCore = loom::fabric::decodeFabricImportedModuleTargetRef(
      unsignedBytes(getSpatialCoreAttr()));
  if (!spatialCore)
    return emitOpError("has invalid spatial_core reference: ")
           << llvm::toString(spatialCore.takeError());
  return success();
}

LogicalResult SystemMemoryServiceOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  MemoryServiceContractAttr contract = getServiceContractAttr();
  auto decoded = decodeMemoryServiceContractRecord(
      unsignedBytes(contract.getRecord()), getContext(),
      MemoryServiceOwnerKind::System);
  if (!decoded)
    return emitOpError("has invalid System memory service contract: ")
           << llvm::toString(decoded.takeError());
  return success();
}
