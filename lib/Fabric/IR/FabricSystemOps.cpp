#include "Fabric/IR/FabricOps.h"

#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

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
    if (!isa<SystemHostCoreOp, SystemAccCoreOp, SystemMemoryServiceOp,
             SystemServiceEndpointOp, SystemServiceTransformOp,
             SystemServiceLegCarrierAttachmentOp, SystemExternalBoundaryOp,
             SystemHardwareDomainOp, SystemTransportResourceOp,
             SystemTransferPatternOp, SystemConnectionOp,
             SystemSpatialAttachmentOp>(operation))
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

LogicalResult SystemServiceEndpointOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto owner = loom::fabric::decodeSystemServiceEndpointOwnerRef(
      unsignedBytes(getOwnerAttr()));
  if (!owner)
    return emitOpError("has invalid owner reference: ")
           << llvm::toString(owner.takeError());
  auto capabilities = loom::fabric::decodeCanonicalServiceCapabilitySet(
      unsignedBytes(getCapabilitiesAttr()), getContext());
  if (!capabilities)
    return emitOpError("has invalid capability set: ")
           << llvm::toString(capabilities.takeError());

  TypeAttr carrierAttribute = getCarrierTypeAttr();
  if (capabilities->plane() ==
      loom::fabric::CanonicalServiceEndpointPlane::Memory) {
    if (carrierAttribute)
      return emitOpError("memory service endpoint must not declare a message "
                         "carrier type");
    return success();
  }
  if (!carrierAttribute)
    return emitOpError("message service endpoint requires a carrier type");
  Type carrier = carrierAttribute.getValue();
  std::optional<unsigned> carrierWidth =
      getFabricTransportPayloadWidth(carrier);
  if (!carrierWidth)
    return emitOpError("has non-transport message carrier type ") << carrier;

  for (const loom::fabric::CanonicalServiceCapabilityRecord &capability :
       capabilities->capabilities()) {
    const auto &domain =
        std::get<loom::fabric::MessageTransferCapabilityDomain>(
            capability.domain());
    for (Type payload : domain.payloadTypes()) {
      std::string error;
      FailureOr<unsigned> width = getSemanticPayloadWidth(payload, error);
      if (failed(width))
        return emitOpError("has unsupported message payload: ") << error;
      if (*width > *carrierWidth)
        return emitOpError("message payload width ")
               << *width << " exceeds carrier width " << *carrierWidth;
    }
    if (domain.fixedVectors() &&
        domain.fixedVectors()->maximumPayloadBits() > *carrierWidth)
      return emitOpError("fixed-vector message payload width ")
             << domain.fixedVectors()->maximumPayloadBits()
             << " exceeds carrier width " << *carrierWidth;
    for (const ::fabric::PointerFormat &format :
         domain.pointerFormats().formats())
      if (format.representationBits > *carrierWidth)
        return emitOpError("pointer message payload width ")
               << format.representationBits << " exceeds carrier width "
               << *carrierWidth;
  }
  return success();
}

LogicalResult SystemServiceLegCarrierAttachmentOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto record = loom::fabric::decodeServiceLegCarrierAttachmentRecord(
      unsignedBytes(getRecordAttr()));
  if (!record)
    return emitOpError("has invalid attachment record: ")
           << llvm::toString(record.takeError());
  return success();
}

LogicalResult SystemServiceTransformOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto contract = loom::fabric::decodeSystemServiceTransformRecord(
      unsignedBytes(getContractAttr()));
  if (!contract)
    return emitOpError("has invalid service transform contract: ")
           << llvm::toString(contract.takeError());
  return success();
}

LogicalResult SystemExternalBoundaryOp::verify() {
  return verifyClosedAttributes(getOperation());
}

LogicalResult SystemHardwareDomainOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto contract = loom::fabric::decodeHardwareDomainContractRecord(
      unsignedBytes(getContractAttr()));
  if (!contract)
    return emitOpError("has invalid hardware-domain contract: ")
           << llvm::toString(contract.takeError());
  return success();
}

LogicalResult SystemTransportResourceOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  FunctionType ports = cast<FunctionType>(getFunctionTypeAttr().getValue());
  if (ports.getNumInputs() == 0 || ports.getNumResults() == 0)
    return emitOpError("requires at least one input and one output port");
  for (ArrayRef<Type> types : {ports.getInputs(), ports.getResults()})
    for (Type type : types)
      if (!isa<BitsType, BitsTagType>(type))
        return emitOpError("has non-transport port type ") << type;

  auto resourceContract =
      decodeResourceContractRecord(unsignedBytes(getResourceContractAttr()));
  if (!resourceContract)
    return emitOpError("has invalid resource contract: ")
           << llvm::toString(resourceContract.takeError());
  if (resourceContract->usePatternCount() == 0)
    return emitOpError("resource contract has no transfer UsePattern");

  if (DenseI8ArrayAttr crossing = getClockCrossingAttr()) {
    auto decoded = loom::fabric::decodeClockCrossingContractRecord(
        unsignedBytes(crossing));
    if (!decoded)
      return emitOpError("has invalid clock crossing contract: ")
             << llvm::toString(decoded.takeError());
  }
  return success();
}

LogicalResult SystemTransferPatternOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto decoded = loom::fabric::decodeSystemTransferPatternRecord(
      unsignedBytes(getContractAttr()));
  if (!decoded)
    return emitOpError("has invalid transfer pattern contract: ")
           << llvm::toString(decoded.takeError());
  return success();
}

LogicalResult SystemConnectionOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  if (getMemoryServiceAttr()) {
    auto source =
        loom::fabric::decodeFabricRef<loom::fabric::FabricMemoryEndpointRef>(
            unsignedBytes(getSourceAttr()));
    if (!source)
      return emitOpError("has invalid memory-service source endpoint: ")
             << llvm::toString(source.takeError());
    auto destination =
        loom::fabric::decodeFabricRef<loom::fabric::FabricMemoryEndpointRef>(
            unsignedBytes(getDestinationAttr()));
    if (!destination)
      return emitOpError("has invalid memory-service destination endpoint: ")
             << llvm::toString(destination.takeError());
    return success();
  }
  auto source =
      loom::fabric::decodeFabricRef<loom::fabric::FabricTransportEndpointRef>(
          unsignedBytes(getSourceAttr()));
  if (!source)
    return emitOpError("has invalid source endpoint: ")
           << llvm::toString(source.takeError());
  auto destination =
      loom::fabric::decodeFabricRef<loom::fabric::FabricTransportEndpointRef>(
          unsignedBytes(getDestinationAttr()));
  if (!destination)
    return emitOpError("has invalid destination endpoint: ")
           << llvm::toString(destination.takeError());
  return success();
}

LogicalResult SystemSpatialAttachmentOp::verify() {
  if (failed(verifyClosedAttributes(getOperation())))
    return failure();
  auto moduleEndpoint =
      loom::fabric::decodeFabricImportedModuleBoundaryEndpointRef(
          unsignedBytes(getModuleEndpointAttr()));
  if (!moduleEndpoint)
    return emitOpError("has invalid module_endpoint reference: ")
           << llvm::toString(moduleEndpoint.takeError());
  auto spatialEndpoint = loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
      unsignedBytes(getSpatialEndpointAttr()));
  if (!spatialEndpoint)
    return emitOpError("has invalid spatial_endpoint reference: ")
           << llvm::toString(spatialEndpoint.takeError());
  DenseI8ArrayAttr serviceAttribute = getServiceEndpointAttr();
  if (spatialEndpoint->plane() ==
      loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport) {
    if (serviceAttribute)
      return emitOpError(
          "transport attachment must not declare a service_endpoint");
    return success();
  }
  if (!serviceAttribute)
    return emitOpError("memory attachment requires a service_endpoint");
  auto serviceEndpoint =
      loom::fabric::decodeFabricRef<loom::fabric::SystemServiceEndpointRef>(
          unsignedBytes(serviceAttribute));
  if (!serviceEndpoint)
    return emitOpError("has invalid service_endpoint reference: ")
           << llvm::toString(serviceEndpoint.takeError());
  return success();
}
