#ifndef FABRIC_IR_FIFORESOURCECONTRACT_H
#define FABRIC_IR_FIFORESOURCECONTRACT_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace fabric {

enum class FifoResourceState : std::uint32_t {
  BufferedQueue = 0,
  BypassTransfer = 1,
};

enum class FifoBufferedCapacity : std::uint32_t {
  QueueSlot = 0,
  EnqueueService = 1,
  DequeueService = 2,
};

enum class FifoResourceTransition : std::uint32_t {
  Append = 0,
  Remove = 1,
  ReplaceHead = 2,
};

enum class FifoUsePattern : std::uint32_t {
  Enqueue = 0,
  Dequeue = 1,
  SimultaneousDequeueEnqueue = 2,
  BypassTransfer = 3,
};

inline UsePatternKey fifoUsePattern(FifoUsePattern pattern) {
  return UsePatternKey(static_cast<std::uint32_t>(pattern));
}

ResourceContractDeclaration declareFifoResourceContract(std::uint32_t maxDepth,
                                                        bool bypassable);

llvm::Expected<ResourceContract>
createFifoResourceContract(std::uint32_t maxDepth, bool bypassable);

} // namespace fabric

#endif // FABRIC_IR_FIFORESOURCECONTRACT_H
