#ifndef FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
#define FABRIC_IR_TEMPORALPERESOURCECONTRACT_H

#include "Fabric/IR/TemporalOperandBuffer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <system_error>

namespace fabric {

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

} // namespace fabric

#endif // FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
