#ifndef LOOM_LIB_FABRIC_IR_MEMORY_SERVICE_CONTRACT_INTERNAL_H
#define LOOM_LIB_FABRIC_IR_MEMORY_SERVICE_CONTRACT_INTERNAL_H

#include "Fabric/IR/MemoryServiceContract.h"

namespace fabric::detail {

struct MemoryServiceCapabilityPhysicalFacts {
  std::vector<std::uint64_t> serviceRegionOrdinals;
  std::uint64_t serviceBeatWidthBits = 0;
  MemoryServiceConsistencyBinding consistencyBinding;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryServiceCapabilityPhysicalFacts(
    const MemoryServiceCapabilityPhysicalFacts &facts);

llvm::Expected<MemoryServiceCapabilityPhysicalFacts>
decodeMemoryServiceCapabilityPhysicalFacts(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_MEMORY_SERVICE_CONTRACT_INTERNAL_H
