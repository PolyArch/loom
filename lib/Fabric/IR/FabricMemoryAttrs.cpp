#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/MemoryCapabilityFinalization.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryServiceContract.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <vector>

using namespace mlir;
using namespace fabric;

char MemoryCapabilityFinalizationError::ID = 0;

void MemoryCapabilityFinalizationError::log(llvm::raw_ostream &stream) const {
  stream << "Invalid(";
  switch (reason_) {
  case MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract:
    stream << "missing-memory-capability-contract";
    break;
  }
  stream << ')';
}

std::error_code MemoryCapabilityFinalizationError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error
fabric::validateMemoryCapabilityFinalization(MemoryContractAttr contract,
                                             ArrayAttr operationPorts) {
  if (!contract)
    return llvm::Error::success();
  if (LocalMemoryServiceAttr local = contract.getLocalService()) {
    llvm::ArrayRef<std::int8_t> signedBytes =
        local.getServiceContract().getRecord().asArrayRef();
    std::vector<std::uint8_t> bytes;
    bytes.reserve(signedBytes.size());
    for (std::int8_t byte : signedBytes)
      bytes.push_back(static_cast<std::uint8_t>(byte));
    auto service = decodeMemoryServiceContractRecord(
        bytes, contract.getContext(), MemoryServiceOwnerKind::Local);
    if (!service)
      return service.takeError();
    if (llvm::Error error = validateLocalMemoryServiceCapacity(
            *service, local.getCapacityBytes()))
      return error;
  }
  if (contract.getEngine() && (!operationPorts || operationPorts.empty()))
    return llvm::make_error<MemoryCapabilityFinalizationError>(
        MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract);
  if (MemoryEngineAttr engine = contract.getEngine())
    if (engine.getSchedule() == Schedule::Temporal &&
        !engine.getResidentContexts())
      return llvm::make_error<MemoryCapabilityFinalizationError>(
          MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract);
  if (!contract.getConnectivity())
    return llvm::make_error<MemoryCapabilityFinalizationError>(
        MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract);
  return llvm::Error::success();
}

LogicalResult LocalMemoryServiceAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, uint64_t capacityBytes,
    MemoryServiceContractAttr serviceContract) {
  if (capacityBytes == 0)
    return emitError() << "local memory service capacity_bytes must be "
                          "greater than zero";
  if (!serviceContract)
    return emitError() << "local memory service requires an explicit "
                          "memory service contract";
  llvm::ArrayRef<std::int8_t> signedBytes =
      serviceContract.getRecord().asArrayRef();
  std::vector<std::uint8_t> bytes;
  bytes.reserve(signedBytes.size());
  for (std::int8_t byte : signedBytes)
    bytes.push_back(static_cast<std::uint8_t>(byte));
  auto decoded = decodeMemoryServiceContractRecord(
      bytes, serviceContract.getContext(), MemoryServiceOwnerKind::Local);
  if (!decoded)
    return emitError() << llvm::toString(decoded.takeError());
  if (llvm::Error error =
          validateLocalMemoryServiceCapacity(*decoded, capacityBytes))
    return emitError() << llvm::toString(std::move(error));
  return success();
}

LogicalResult MemoryResidentContextsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, uint64_t count) {
  if (count == 0)
    return emitError() << "memory resident-context count must be greater than "
                          "zero";
  return success();
}

LogicalResult MemoryConnectivityContractAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    DenseI8ArrayAttr record) {
  if (!record)
    return emitError() << "memory connectivity requires a canonical record";
  std::vector<std::uint8_t> bytes;
  bytes.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  auto decoded = decodeMemoryConnectivityContractRecord(bytes);
  if (!decoded)
    return emitError() << llvm::toString(decoded.takeError());
  return success();
}

LogicalResult
MemoryEngineAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         Schedule schedule,
                         MemoryResidentContextsAttr residentContexts) {
  if (schedule == Schedule::Spatial && residentContexts)
    return emitError()
           << "spatial memory engine cannot carry resident contexts";
  return success();
}

namespace {

LogicalResult
verifyEndpointOrdinals(llvm::function_ref<InFlightDiagnostic()> emitError,
                       DenseI32ArrayAttr endpoints, StringRef role) {
  if (!endpoints)
    return emitError() << "memory contract requires explicit " << role
                       << " endpoint ordinals";

  int32_t previous = -1;
  for (int32_t endpoint : endpoints.asArrayRef()) {
    if (endpoint < 0)
      return emitError() << role << " endpoint ordinals must be nonnegative";
    if (endpoint <= previous)
      return emitError() << role
                         << " endpoint ordinals must be strictly increasing";
    previous = endpoint;
  }
  return success();
}

} // namespace

LogicalResult
MemoryContractAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           MemoryEngineAttr engine,
                           LocalMemoryServiceAttr localService,
                           MemoryConnectivityContractAttr connectivity,
                           DenseI32ArrayAttr managerEndpoints,
                           DenseI32ArrayAttr subordinateEndpoints) {
  if (failed(verifyEndpointOrdinals(emitError, managerEndpoints, "manager")) ||
      failed(verifyEndpointOrdinals(emitError, subordinateEndpoints,
                                    "subordinate")))
    return failure();
  return success();
}
