#ifndef FABRIC_IR_FIFORESOURCECONTRACT_H
#define FABRIC_IR_FIFORESOURCECONTRACT_H

#include "Fabric/IR/FabricEnums.h"
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
  /// Moves the dequeue offer cursor past one presented-and-refused virtual
  /// channel. Only a PerTagVirtualChannel queue owns this transition; it is
  /// the single internal arbitration transition of that discipline and
  /// applies atomically at the cycle boundary.
  OfferAdvance = 3,
};

enum class FifoUsePattern : std::uint32_t {
  Enqueue = 0,
  Dequeue = 1,
  SimultaneousDequeueEnqueue = 2,
  BypassTransfer = 3,
  /// Presents the head of one resident virtual channel whose consumer does
  /// not accept it this cycle and rotates the offer cursor at the cycle
  /// boundary. Claims no capacity. Its Physical Tag parameter names the
  /// channel whose head was presented. The enum ordinal names the semantic
  /// pattern; its key inside one declaration is owner-local (see
  /// `fifoVirtualChannelOfferAdvancePattern`).
  OfferAdvance = 4,
};

inline UsePatternKey fifoUsePattern(FifoUsePattern pattern) {
  return UsePatternKey(static_cast<std::uint32_t>(pattern));
}

/// The owner-local key of the offer-advance pattern in one
/// PerTagVirtualChannel declaration. That discipline never declares
/// BypassTransfer, so its closed zero-based pattern domain is Enqueue,
/// Dequeue, SimultaneousDequeueEnqueue, OfferAdvance at keys 0 through 3.
inline UsePatternKey fifoVirtualChannelOfferAdvancePattern() {
  return UsePatternKey(3);
}

/// Declares the complete contract of one FIFO occurrence. A StrictFifo
/// declaration is exactly the pre-discipline contract, so a 7.0 artifact and
/// its 7.1 StrictFifo re-finalization carry identical canonical bytes. A
/// PerTagVirtualChannel declaration requires a positive `tagWidthBits`: its
/// dequeue and offer patterns are qualified by the Physical Tag value they
/// present, its queue slot capacity remains one shared pool, and it owns no
/// bypass alternative.
ResourceContractDeclaration
declareFifoResourceContract(std::uint32_t maxDepth, bool bypassable,
                            FifoQueueDiscipline discipline,
                            std::uint32_t tagWidthBits);

llvm::Expected<ResourceContract>
createFifoResourceContract(std::uint32_t maxDepth, bool bypassable,
                           FifoQueueDiscipline discipline =
                               FifoQueueDiscipline::StrictFifo,
                           std::uint32_t tagWidthBits = 0);

} // namespace fabric

#endif // FABRIC_IR_FIFORESOURCECONTRACT_H
