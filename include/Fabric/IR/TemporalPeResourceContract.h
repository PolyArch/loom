#ifndef FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
#define FABRIC_IR_TEMPORALPERESOURCECONTRACT_H

#include "Fabric/IR/PhysicalTagResourceContract.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <system_error>
#include <vector>

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

/// One member of the closed temporal context-evaluation domain. A grant lets
/// the selected resident configuration drive its FU for one PE clock cycle;
/// it is not an actor transition or a whole-context commit. Candidate order,
/// service sharing, reset, and fairness are derived by the PE contract.
struct TemporalPeDispatchCandidate final {
  loom::fabric::InstructionContextRef context;
  loom::fabric::FabricOrdinal fuOccurrence = 0;
  std::uint32_t allocationUnit = 0;
};

/// A register FIFO path selects a registered queue and never contributes a
/// combinational ready/valid dependency. The complete temporal-PE contract
/// still owns its queue capacity, port service, transitions, and timing.
class TemporalPeResourceContract final {
public:
  static llvm::Expected<TemporalPeResourceContract>
  create(const TemporalPeResourceDeclaration &declaration);

  const ResourceContract &resourceContract() const { return contract_; }

  llvm::ArrayRef<TemporalPeDispatchCandidate> dispatchCandidates() const {
    return dispatchCandidates_;
  }
  std::uint32_t dispatchUnitCount() const {
    return static_cast<std::uint32_t>(dispatchUnitSpans_.size());
  }
  llvm::ArrayRef<std::uint32_t> dispatchCandidatesOf(std::uint32_t unit) const;
  StateKey dispatchState(std::uint32_t unit) const;
  RequesterKey dispatchRequester(std::uint32_t candidate) const;
  UsePatternKey dispatchPattern(std::uint32_t candidate) const;

  std::uint32_t registerFifoCount() const { return registerFifoCount_; }
  StateKey registerFifoState(std::uint32_t fifo) const;
  UsePatternKey registerFifoWritePattern(std::uint32_t fifo) const;
  UsePatternKey registerFifoReadPattern(std::uint32_t fifo) const;

private:
  struct Span final {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
  };

  TemporalPeResourceContract(
      ResourceContract contract,
      std::vector<TemporalPeDispatchCandidate> dispatchCandidates,
      std::vector<std::uint32_t> dispatchUnitCandidates,
      std::vector<Span> dispatchUnitSpans, std::uint32_t dispatchStateOffset,
      std::uint32_t dispatchRequesterOffset,
      std::uint32_t dispatchPatternOffset, std::uint32_t registerFifoCount,
      std::uint32_t registerStateOffset, std::uint32_t registerPatternOffset)
      : contract_(std::move(contract)),
        dispatchCandidates_(std::move(dispatchCandidates)),
        dispatchUnitCandidates_(std::move(dispatchUnitCandidates)),
        dispatchUnitSpans_(std::move(dispatchUnitSpans)),
        dispatchStateOffset_(dispatchStateOffset),
        dispatchRequesterOffset_(dispatchRequesterOffset),
        dispatchPatternOffset_(dispatchPatternOffset),
        registerFifoCount_(registerFifoCount),
        registerStateOffset_(registerStateOffset),
        registerPatternOffset_(registerPatternOffset) {}

  ResourceContract contract_;
  std::vector<TemporalPeDispatchCandidate> dispatchCandidates_;
  std::vector<std::uint32_t> dispatchUnitCandidates_;
  std::vector<Span> dispatchUnitSpans_;
  std::uint32_t dispatchStateOffset_ = 0;
  std::uint32_t dispatchRequesterOffset_ = 0;
  std::uint32_t dispatchPatternOffset_ = 0;
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

/// Resolves the exact context-evaluation service use selected by one temporal
/// compute binding. Its ordinal is derived from the same canonical context/FU
/// inventory used by finalization and RTL lowering.
llvm::Expected<loom::fabric::FabricUsePatternRef>
resolveTemporalPeDispatchPattern(const loom::fabric::FabricArtifactView &view,
                                 loom::fabric::InstructionContextRef context,
                                 loom::fabric::FabricFuOccurrenceRef fu);

} // namespace fabric

#endif // FABRIC_IR_TEMPORALPERESOURCECONTRACT_H
