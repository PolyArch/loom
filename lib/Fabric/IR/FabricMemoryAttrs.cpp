#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/MemoryCapabilityFinalization.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

char MemoryCapabilityFinalizationError::ID = 0;

void MemoryCapabilityFinalizationError::log(llvm::raw_ostream &stream) const {
  stream << "Invalid(";
  switch (reason_) {
  case MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract:
    stream << "missing-memory-capability-contract";
    break;
  case MemoryCapabilityFinalizationReason::MissingMemoryServiceContract:
    stream << "missing-memory-service-contract";
    break;
  }
  stream << ')';
}

std::error_code MemoryCapabilityFinalizationError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error
fabric::validateMemoryCapabilityFinalization(MemoryContractAttr contract) {
  if (!contract)
    return llvm::Error::success();
  if (contract.getLocalService())
    return llvm::make_error<MemoryCapabilityFinalizationError>(
        MemoryCapabilityFinalizationReason::MissingMemoryServiceContract);
  if (contract.getEngine())
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

LogicalResult MemoryContractAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, MemoryEngineAttr engine,
    LocalMemoryServiceAttr localService, DenseI32ArrayAttr managerEndpoints,
    DenseI32ArrayAttr subordinateEndpoints) {
  if (failed(verifyEndpointOrdinals(emitError, managerEndpoints, "manager")) ||
      failed(verifyEndpointOrdinals(emitError, subordinateEndpoints,
                                    "subordinate")))
    return failure();
  return success();
}
