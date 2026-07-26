#ifndef LOOM_LIB_FABRIC_IR_MEMORY_OPERATION_PORT_INTERNAL_H
#define LOOM_LIB_FABRIC_IR_MEMORY_OPERATION_PORT_INTERNAL_H

#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric::detail {

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryCapabilityAlternativeRecord(
    const MemoryCapabilityAlternativeRecord &alternative);

llvm::Expected<std::vector<std::uint8_t>> encodeMemoryOperationPortDeclaration(
    const MemoryOperationPortDeclaration &declaration);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_MEMORY_OPERATION_PORT_INTERNAL_H
