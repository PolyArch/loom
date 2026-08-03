#ifndef FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
#define FABRIC_IR_TEMPORALPERESOURCECONTRACT_H

#include "Fabric/IR/PhysicalTagResourceContract.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <system_error>

namespace fabric {

enum class TemporalOperandQueueUse : std::uint32_t { Enqueue, Dequeue };

/// The complete resource declaration of one temporal PE. Operand buffering
/// and register FIFOs are parts of the same physical owner and therefore share
/// one ResourceContract and one owner-local key domain.
struct TemporalPeResourceDeclaration final {
  loom::fabric::FabricPeOccurrenceRef pe;
  std::uint32_t contextCount = 0;
  llvm::ArrayRef<std::uint32_t> fuInputCounts;
  OperandBufferMode operandBufferMode{};
  std::uint32_t operandEntriesPerAllocationUnit = 0;
  std::uint32_t registerFifoCount = 0;
  std::uint32_t registerFifoDepth = 0;
  std::uint32_t registerFifoPorts = 1;
};

/// A register FIFO path selects a registered queue and never contributes a
/// combinational ready/valid dependency. The complete temporal-PE contract
/// still owns its queue capacity, port service, transitions, and timing.
class TemporalPeResourceContract final {
public:
  static llvm::Expected<TemporalPeResourceContract>
  create(const TemporalPeResourceDeclaration &declaration);

  const ResourceContract &resourceContract() const { return contract_; }

  std::uint32_t registerFifoCount() const { return registerFifoCount_; }
  StateKey registerFifoState(std::uint32_t fifo) const;
  UsePatternKey registerFifoWritePattern(std::uint32_t fifo) const;
  UsePatternKey registerFifoReadPattern(std::uint32_t fifo) const;

private:
  TemporalPeResourceContract(ResourceContract contract,
                             std::uint32_t registerFifoCount,
                             std::uint32_t registerStateOffset,
                             std::uint32_t registerPatternOffset)
      : contract_(std::move(contract)), registerFifoCount_(registerFifoCount),
        registerStateOffset_(registerStateOffset),
        registerPatternOffset_(registerPatternOffset) {}

  ResourceContract contract_;
  std::uint32_t registerFifoCount_ = 0;
  std::uint32_t registerStateOffset_ = 0;
  std::uint32_t registerPatternOffset_ = 0;
};

/// Resolves the same canonical state layout from an imported temporal-PE
/// contract. This is the sole ordinal projection used by sealed Fabric views.
inline llvm::Expected<StateKey>
resolveTemporalPeRegisterFifoState(const ResourceContract &contract,
                                   std::uint32_t registerFifoCount,
                                   std::uint32_t fifo) {
  if (fifo >= registerFifoCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "register FIFO ordinal is outside its owner domain");
  if (registerFifoCount > contract.stateCount())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "register FIFO states are absent from the PE contract");
  return StateKey(contract.stateCount() - registerFifoCount + fifo);
}

/// Resolves the role-selected register-FIFO use from the same canonical
/// combined temporal-PE contract. Operand-buffer patterns precede the two
/// register-FIFO role ranges; the imported contract remains their sole owner.
inline llvm::Expected<UsePatternKey>
resolveTemporalPeRegisterFifoPattern(const ResourceContract &contract,
                                     std::uint32_t registerFifoCount,
                                     std::uint32_t fifo, bool write) {
  if (fifo >= registerFifoCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "register FIFO ordinal is outside its owner domain");
  const std::uint64_t registerPatternCount =
      static_cast<std::uint64_t>(registerFifoCount) * 2;
  std::uint32_t basePatternCount = contract.usePatternCount();
  while (basePatternCount != 0 &&
         physicalTagAssignmentPatternWidth(
             contract.usePattern(UsePatternKey(basePatternCount - 1))))
    --basePatternCount;
  if (registerPatternCount > basePatternCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "register FIFO patterns are absent from the PE contract");
  const std::uint32_t patternOffset =
      basePatternCount - static_cast<std::uint32_t>(registerPatternCount);
  const std::uint32_t ordinal =
      patternOffset + (write ? 0 : registerFifoCount) + fifo;
  const UsePattern pattern = contract.usePattern(UsePatternKey(ordinal));
  auto state =
      resolveTemporalPeRegisterFifoState(contract, registerFifoCount, fifo);
  if (!state)
    return state.takeError();
  if (pattern.claims.size() != 1 || pattern.claims.front().state != *state)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "register FIFO pattern disagrees with its canonical state");
  return UsePatternKey(ordinal);
}

/// Resolves one exact logical operand queue through a sealed Fabric view. The
/// FU occurrence ordinal and queue ordinal are derived from the canonical
/// occurrence and port inventories used by temporal-PE finalization; Mapping
/// never reproduces that layout.
llvm::Expected<loom::fabric::FabricUsePatternRef>
resolveTemporalPeOperandQueuePattern(
    const loom::fabric::FabricArtifactView &view,
    loom::fabric::InstructionContextRef context,
    loom::fabric::FabricFuOccurrenceRef fu, loom::fabric::FabricOrdinal fuInput,
    TemporalOperandQueueUse use);

} // namespace fabric

#endif // FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
