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
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

llvm::Expected<std::uint64_t>
fixedAllocByteSize(const llvm::DataLayout &layout,
                   mlir::LLVM::TypeToLLVMIRTranslator &translator,
                   mlir::Type type) {
  llvm::Type *translated = translator.translateType(type);
  if (!translated)
    return llvm::createStringError(
        std::errc::not_supported,
        "LLVM GEP source type has no LLVM IR projection");
  llvm::TypeSize size = layout.getTypeAllocSize(translated);
  if (size.isScalable() || size.getFixedValue() == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "LLVM GEP source type has no fixed nonzero allocation size");
  return size.getFixedValue();
}

bool hasFlag(mlir::LLVM::GEPNoWrapFlags flags,
             mlir::LLVM::GEPNoWrapFlags flag) {
  return mlir::LLVM::bitEnumContainsAny(flags, flag);
}

llvm::Expected<llvm::APInt>
canonicalizeIndex(const llvm::APInt &index, unsigned addressBits,
                  mlir::LLVM::GEPNoWrapFlags flags) {
  if (index.getBitWidth() == addressBits)
    return index;
  if (index.getBitWidth() > addressBits) {
    if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nusw) &&
        !index.isSignedIntN(addressBits))
      return llvm::createStringError(std::errc::result_out_of_range,
                                     "LLVM GEP nusw index truncation");
    if (hasFlag(flags, mlir::LLVM::GEPNoWrapFlags::nuw) &&
        !index.isIntN(addressBits))
      return llvm::createStringError(std::errc::result_out_of_range,
                                     "LLVM GEP nuw index truncation");
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
                                  const GepExecutionPlan &plan,
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
  for (const GepOffsetTerm &term : plan.terms) {
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

llvm::Expected<GepExecutionPlan> gepExecutionPlan(mlir::LLVM::GEPOp op,
                                                  mlir::Operation *graphScope) {
  auto baseType =
      mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(op.getBase().getType());
  auto resultType =
      mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(op.getRes().getType());
  if (!baseType || !resultType || baseType != resultType)
    return llvm::createStringError(
        std::errc::not_supported,
        "DFG-sim supports scalar GEP with one exact pointer type");
  for (mlir::Value index : op.getDynamicIndices())
    if (!mlir::isa<mlir::IntegerType>(index.getType()))
      return llvm::createStringError(
          std::errc::not_supported,
          "DFG-sim supports scalar integer GEP indices");

  auto pointerLayout =
      ::loom::resolvePointerLayout(graphScope, baseType.getAddressSpace());
  if (!pointerLayout)
    return pointerLayout.takeError();
  if (pointerLayout->kind == ::loom::PointerLayoutKind::Unstable ||
      pointerLayout->kind == ::loom::PointerLayoutKind::ExternalState)
    return llvm::createStringError(
        std::errc::not_supported,
        "DFG-sim has no provider for this LLVM pointer representation kind");
  auto llvmLayout = ::loom::resolveLLVMDataLayout(graphScope);
  if (!llvmLayout)
    return llvmLayout.takeError();

  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator translator(llvmContext);
  GepExecutionPlan plan{*pointerLayout, op.getNoWrapFlags(), {}};
  mlir::Type indexedType = op.getElemType();
  unsigned dynamicOrdinal = 0;
  for (auto [position, rawIndex] :
       llvm::enumerate(op.getRawConstantIndices())) {
    GepOffsetTerm term;
    if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex) {
      if (dynamicOrdinal >= op.getDynamicIndices().size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "LLVM GEP dynamic-index table is malformed");
      term.dynamicOperandOrdinal = 1 + dynamicOrdinal++;
    } else {
      term.constantIndex = llvm::APInt(
          32, static_cast<std::uint64_t>(static_cast<std::int64_t>(rawIndex)),
          /*isSigned=*/true);
    }

    std::uint64_t scale = 0;
    if (position == 0) {
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto array =
                   mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(indexedType)) {
      indexedType = array.getElementType();
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto vector = mlir::dyn_cast<mlir::VectorType>(indexedType)) {
      indexedType = vector.getElementType();
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto structure =
                   mlir::dyn_cast<mlir::LLVM::LLVMStructType>(indexedType)) {
      if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex || rawIndex < 0 ||
          static_cast<std::size_t>(rawIndex) >= structure.getBody().size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "LLVM GEP structure index is not a valid constant field");
      llvm::Type *translated = translator.translateType(structure);
      auto *llvmStruct = llvm::dyn_cast_or_null<llvm::StructType>(translated);
      if (!llvmStruct)
        return llvm::createStringError(
            std::errc::not_supported,
            "LLVM GEP structure has no LLVM IR layout projection");
      const std::uint64_t offset =
          llvmLayout->getStructLayout(llvmStruct)->getElementOffset(rawIndex);
      term.constantIndex = llvm::APInt(pointerLayout->addressBits, offset,
                                       /*isSigned=*/false,
                                       /*implicitTrunc=*/true);
      scale = 1;
      indexedType = structure.getBody()[rawIndex];
    } else {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "LLVM GEP index path does not match its source element type");
    }
    term.scale = llvm::APInt(pointerLayout->addressBits, scale,
                             /*isSigned=*/false,
                             /*implicitTrunc=*/true);
    plan.terms.push_back(std::move(term));
  }
  if (dynamicOrdinal != op.getDynamicIndices().size())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "LLVM GEP has unused dynamic indices");
  return plan;
}

bool fireGetElementPtr(
    mlir::Operation *operation,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state) {
  (void)projection;
  auto op = mlir::cast<mlir::LLVM::GEPOp>(operation);
  if (state.terminalPrimitiveOps.contains(operation))
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
    state.terminalPrimitiveOps.insert(operation);
    return false;
  }
  for (unsigned ordinal = 0; ordinal < operation->getNumOperands(); ++ordinal)
    (void)popInputToken(state, ordinal);
  emitResultToken(state, 0, *result);
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
