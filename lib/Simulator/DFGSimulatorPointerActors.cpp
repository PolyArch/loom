//===- DFGSimulatorPointerActors.cpp - LLVM pointer actors ---------------===//
//
// The exact LLVM operation and module DataLayout remain the semantic owners.
// This module derives an immutable scalar GEP execution plan, then applies its
// fixed-width arithmetic while retaining runtime object provenance.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "Common/PointerLayout.h"
#include "Dataflow/IR/OperationSchema.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

bool hasFlag(mlir::LLVM::GEPNoWrapFlags flags,
             mlir::LLVM::GEPNoWrapFlags flag) {
  return mlir::LLVM::bitEnumContainsAny(flags, flag);
}

std::optional<llvm::APInt>
canonicalizeIndex(const llvm::APInt &index, unsigned addressBits,
                  mlir::LLVM::GEPNoWrapFlags flags) {
  if (index.getBitWidth() == addressBits)
    return index;
  if (index.getBitWidth() > addressBits) {
    if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nusw) &&
        !index.isSignedIntN(addressBits))
      return std::nullopt;
    if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nuw) &&
        !index.isIntN(addressBits))
      return std::nullopt;
    return index.trunc(addressBits);
  }
  return index.sext(addressBits);
}

std::optional<llvm::APInt> scaledOffset(const llvm::APInt &index,
                                        const llvm::APInt &scale,
                                        mlir::LLVM::GEPNoWrapFlags flags) {
  bool signedOverflow = false;
  bool unsignedOverflow = false;
  llvm::APInt signedResult = index.smul_ov(scale, signedOverflow);
  llvm::APInt unsignedResult = index.umul_ov(scale, unsignedOverflow);
  if ((hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nusw) && signedOverflow) ||
      (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nuw) && unsignedOverflow))
    return std::nullopt;
  assert(signedResult == unsignedResult &&
         "fixed-width multiplication has one wrapped bit pattern");
  return signedResult;
}

bool addAccumulatedOffset(llvm::APInt &accumulated, const llvm::APInt &offset,
                          mlir::LLVM::GEPNoWrapFlags flags) {
  bool signedOverflow = false;
  bool unsignedOverflow = false;
  llvm::APInt signedResult = accumulated.sadd_ov(offset, signedOverflow);
  llvm::APInt unsignedResult = accumulated.uadd_ov(offset, unsignedOverflow);
  if ((hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nusw) && signedOverflow) ||
      (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nuw) && unsignedOverflow))
    return false;
  assert(signedResult == unsignedResult &&
         "fixed-width addition has one wrapped bit pattern");
  accumulated = std::move(signedResult);
  return true;
}

bool addPointerOffset(PointerValue &pointer, const llvm::APInt &offset,
                      mlir::LLVM::GEPNoWrapFlags flags,
                      llvm::APInt &accumulated) {
  if (offset.isZero())
    return true;

  const unsigned addressBits = pointer.byteOffset.getBitWidth();
  llvm::APInt oldAddress = pointer.representation.trunc(addressBits);
  bool unsignedOverflow = false;
  llvm::APInt newAddress = oldAddress.uadd_ov(offset, unsignedOverflow);
  if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nuw) && unsignedOverflow)
    return false;
  if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nusw) &&
      (offset.isNonNegative() ? newAddress.ult(oldAddress)
                              : newAddress.ugt(oldAddress)))
    return false;

  llvm::APInt newObjectOffset = pointer.byteOffset + offset;
  if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::inboundsFlag)) {
    if (!pointer.memory || newObjectOffset.isNegative() ||
        newObjectOffset.getActiveBits() > 64 ||
        newObjectOffset.getZExtValue() > pointer.memory->bytes.size())
      return false;
  }
  if (!addAccumulatedOffset(accumulated, offset, flags))
    return false;

  llvm::APInt lowMask = llvm::APInt::getLowBitsSet(
      pointer.representation.getBitWidth(), addressBits);
  pointer.representation =
      (pointer.representation & ~lowMask) |
      newAddress.zext(pointer.representation.getBitWidth());
  pointer.byteOffset = std::move(newObjectOffset);
  return true;
}

llvm::Expected<Token> evaluateGep(mlir::LLVM::GEPOp op,
                                  const dataflow::semantics::GepAddressPlan &plan,
                                  llvm::ArrayRef<Token> operands) {
  if (operands.size() != op->getNumOperands())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "LLVM GEP token count does not match operation operands");

  bool hasUndef = false;
  for (const Token &token : operands) {
    if (token.valueState == PrimitiveValueState::Poison)
      return exceptionalValueToken(PrimitiveValueState::Poison,
                                   op.getRes().getType());
    hasUndef |= token.valueState == PrimitiveValueState::Undef;
  }
  if (hasUndef)
    return exceptionalValueToken(PrimitiveValueState::Undef,
                                 op.getRes().getType());

  const PointerValue *base = operands.front().pointerValue();
  if (operands.front().kind != TokenKind::Pointer || !base || !base->memory ||
      base->addressSpace != plan.pointerLayout.addressSpace ||
      base->representation.getBitWidth() !=
          plan.pointerLayout.representationBits ||
      base->byteOffset.getBitWidth() != plan.pointerLayout.addressBits)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "LLVM GEP base does not match its exact pointer layout");

  PointerValue result = *base;
  llvm::APInt accumulated(plan.pointerLayout.addressBits, 0);
  for (const dataflow::semantics::GepOffsetTerm &term : plan.terms) {
    llvm::APInt sourceIndex = term.constantIndex;
    if (term.dynamicOperandOrdinal) {
      const unsigned ordinal = *term.dynamicOperandOrdinal;
      if (ordinal >= operands.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "LLVM GEP execution plan names an unavailable dynamic index");
      auto bits = resolvedTokenBitPattern(
          operands[ordinal], op->getOperand(ordinal).getType(), op);
      if (!bits)
        return bits.takeError();
      sourceIndex = std::move(*bits);
    }
    auto index = canonicalizeIndex(sourceIndex, plan.pointerLayout.addressBits,
                                   plan.noWrapFlags);
    if (!index)
      return exceptionalValueToken(PrimitiveValueState::Poison,
                                   op.getRes().getType());
    std::optional<llvm::APInt> offset =
        scaledOffset(*index, term.scale, plan.noWrapFlags);
    if (!offset ||
        !addPointerOffset(result, *offset, plan.noWrapFlags, accumulated))
      return exceptionalValueToken(PrimitiveValueState::Poison,
                                   op.getRes().getType());
  }

  Token token;
  token.kind = TokenKind::Pointer;
  token.setPointerValue(std::move(result));
  return token;
}

} // namespace

bool fireGetElementPtr(
    mlir::Operation *operation,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state) {
  (void)projection;
  auto op = mlir::cast<mlir::LLVM::GEPOp>(operation);
  if (state.terminalComputeOps.contains(operation))
    return false;
  for (unsigned ordinal = 0; ordinal < operation->getNumOperands(); ++ordinal)
    if (!hasInputToken(state, ordinal))
      return false;
  assert(state.currentActorPlan &&
         state.currentActorPlan->operation == operation &&
         state.currentActorPlan->gep &&
         "admitted LLVM GEP has no execution plan");

  llvm::SmallVector<Token, 4> operands;
  operands.reserve(operation->getNumOperands());
  for (unsigned ordinal = 0; ordinal < operation->getNumOperands(); ++ordinal)
    operands.push_back(peekInputToken(state, ordinal));
  auto result = evaluateGep(op, *state.currentActorPlan->gep, operands);
  if (!result) {
    state.diagnostics.push_back(llvm::toString(result.takeError()));
    state.terminalComputeOps.insert(operation);
    return false;
  }
  for (unsigned ordinal = 0; ordinal < operation->getNumOperands(); ++ordinal)
    (void)popInputToken(state, ordinal);
  emitResultToken(state, 0, *result);
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
