#ifndef LOOM_FABRIC_IR_MEMORY_CAPABILITY_FINALIZATION_H
#define LOOM_FABRIC_IR_MEMORY_CAPABILITY_FINALIZATION_H

#include "Fabric/IR/FabricAttrs.h"

#include "llvm/Support/Error.h"

#include <system_error>

namespace fabric {

enum class MemoryCapabilityFinalizationReason {
  MissingMemoryCapabilityContract,
  MissingMemoryServiceContract,
};

class MemoryCapabilityFinalizationError final
    : public llvm::ErrorInfo<MemoryCapabilityFinalizationError> {
public:
  static char ID;

  explicit MemoryCapabilityFinalizationError(
      MemoryCapabilityFinalizationReason reason)
      : reason_(reason) {}

  MemoryCapabilityFinalizationReason reason() const { return reason_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  MemoryCapabilityFinalizationReason reason_;
};

llvm::Error validateMemoryCapabilityFinalization(
    MemoryContractAttr contract, mlir::ArrayAttr operationPorts);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_CAPABILITY_FINALIZATION_H
