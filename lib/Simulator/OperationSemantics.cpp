#include "Simulator/OperationSemantics.h"

#include "DeterministicTranscendental.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>
#include <system_error>
#include <utility>

using namespace loom::sim;

namespace {

using PrimitiveOperationProvider = llvm::Expected<PrimitiveValue> (*)(
    const PrimitiveOperationDescriptor &, llvm::ArrayRef<PrimitiveValue>);

llvm::Expected<PrimitiveValue> evaluateRegisteredPrimitiveOperation(
    const PrimitiveOperationDescriptor &descriptor,
    llvm::ArrayRef<PrimitiveValue> operands);

PrimitiveOperationProvider
primitiveOperationProvider(dataflow::OperationSchemaId schema) {
  using Schema = dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithAddI:
  case Schema::ArithSubI:
  case Schema::ArithMulI:
  case Schema::ArithShLI:
  case Schema::ArithShRSI:
  case Schema::ArithShRUI:
  case Schema::ArithAndI:
  case Schema::ArithOrI:
  case Schema::ArithXOrI:
  case Schema::ArithDivSI:
  case Schema::ArithDivUI:
  case Schema::ArithRemSI:
  case Schema::ArithRemUI:
  case Schema::ArithMinSI:
  case Schema::ArithMaxSI:
  case Schema::ArithMinUI:
  case Schema::ArithMaxUI:
  case Schema::ArithCmpI:
  case Schema::ArithSelect:
  case Schema::ArithExtSI:
  case Schema::ArithExtUI:
  case Schema::ArithTruncI:
  case Schema::ArithIndexCast:
  case Schema::ArithIndexCastUI:
  case Schema::ArithBitcast:
  case Schema::ArithAddF:
  case Schema::ArithSubF:
  case Schema::ArithMulF:
  case Schema::ArithDivF:
  case Schema::ArithRemF:
  case Schema::ArithNegF:
  case Schema::ArithMinimumF:
  case Schema::ArithMaximumF:
  case Schema::ArithMinNumF:
  case Schema::ArithMaxNumF:
  case Schema::ArithCmpF:
  case Schema::ArithExtF:
  case Schema::ArithTruncF:
  case Schema::ArithSIToFP:
  case Schema::ArithUIToFP:
  case Schema::ArithFPToSI:
  case Schema::ArithFPToUI:
  case Schema::LLVMFPToSISat:
  case Schema::LLVMFPToUISat:
  case Schema::MathAbsF:
  case Schema::MathSin:
  case Schema::MathCos:
  case Schema::MathTan:
  case Schema::MathSinh:
  case Schema::MathCosh:
  case Schema::MathTanh:
  case Schema::MathExp:
  case Schema::MathExp2:
  case Schema::MathExpM1:
  case Schema::MathLog:
  case Schema::MathLog2:
  case Schema::MathLog10:
  case Schema::MathLog1p:
  case Schema::MathAbsI:
  case Schema::MathFloor:
  case Schema::MathCeil:
  case Schema::MathRound:
  case Schema::MathTrunc:
  case Schema::MathRoundEven:
  case Schema::MathSqrt:
  case Schema::MathRsqrt:
  case Schema::MathErf:
  case Schema::MathFma:
  case Schema::MathCountLeadingZeros:
  case Schema::MathCountTrailingZeros:
  case Schema::UBPoison:
  case Schema::LLVMFshl:
  case Schema::LLVMByteSwap:
  case Schema::LLVMSAddSat:
  case Schema::LLVMUAddSat:
  case Schema::LLVMSSubSat:
  case Schema::LLVMUSubSat:
  case Schema::LLVMCountLeadingZeros:
  case Schema::LLVMCountTrailingZeros:
  case Schema::LLVMAbs:
  case Schema::LLVMOrDisjoint:
    return &evaluateRegisteredPrimitiveOperation;
  default:
    return nullptr;
  }
}

llvm::StringRef spelling(dataflow::OperationSchemaId schema) {
  return dataflow::operationSchemaSpelling(schema);
}

template <typename Payload>
llvm::Expected<const Payload *>
requirePayload(const PrimitiveOperationDescriptor &descriptor) {
  if (const auto *payload = std::get_if<Payload>(&descriptor.actor.payload))
    return payload;
  return llvm::createStringError(
      std::errc::invalid_argument,
      "%s typed semantic payload does not match operation schema",
      spelling(descriptor.actor.schema).str().c_str());
}

llvm::Error requireArity(dataflow::OperationSchemaId schema,
                         llvm::ArrayRef<PrimitiveValue> operands,
                         unsigned expected) {
  if (operands.size() == expected)
    return llvm::Error::success();
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s expects %u operands, got %u",
                                 spelling(schema).str().c_str(), expected,
                                 static_cast<unsigned>(operands.size()));
}

llvm::Expected<unsigned>
requireResultBitWidth(const PrimitiveOperationDescriptor &descriptor) {
  if (descriptor.resultBitWidth != 0)
    return descriptor.resultBitWidth;
  return llvm::createStringError(
      std::errc::invalid_argument, "%s has no resolved result bit width",
      spelling(descriptor.actor.schema).str().c_str());
}

llvm::Expected<const llvm::APInt *>
requireDefinedBits(dataflow::OperationSchemaId schema,
                   const PrimitiveValue &value) {
  if (value.isDefined())
    return &*value.bits;
  return llvm::createStringError(
      std::errc::invalid_argument,
      "%s provider expected exceptional values to be propagated first",
      spelling(schema).str().c_str());
}

std::optional<PrimitiveValue>
strictExceptionalResult(llvm::ArrayRef<PrimitiveValue> operands) {
  if (llvm::any_of(operands, [](const PrimitiveValue &value) {
        return value.state == PrimitiveValueState::Poison;
      }))
    return PrimitiveValue::poison();
  if (llvm::any_of(operands, [](const PrimitiveValue &value) {
        return value.state == PrimitiveValueState::Undef;
      }))
    return PrimitiveValue::undef();
  return std::nullopt;
}

bool samePrimitiveValue(const PrimitiveValue &lhs, const PrimitiveValue &rhs) {
  if (lhs.state != rhs.state)
    return false;
  if (lhs.state != PrimitiveValueState::Defined)
    return true;
  return lhs.bits == rhs.bits;
}

llvm::Expected<std::pair<const llvm::APInt *, const llvm::APInt *>>
requireIntegerPair(dataflow::OperationSchemaId schema,
                   llvm::ArrayRef<PrimitiveValue> operands) {
  auto lhs = requireDefinedBits(schema, operands[0]);
  if (!lhs)
    return lhs.takeError();
  auto rhs = requireDefinedBits(schema, operands[1]);
  if (!rhs)
    return rhs.takeError();
  if ((*lhs)->getBitWidth() != (*rhs)->getBitWidth())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s operand bit widths do not match",
                                   spelling(schema).str().c_str());
  return std::make_pair(*lhs, *rhs);
}

std::optional<unsigned> checkedShiftAmount(const llvm::APInt &amount,
                                           unsigned bitWidth) {
  if (amount.getActiveBits() > 64 || amount.uge(bitWidth))
    return std::nullopt;
  return static_cast<unsigned>(amount.getZExtValue());
}

llvm::RoundingMode roundingMode(const dataflow::FloatingPointPayload &payload) {
  if (!payload.roundingMode)
    return llvm::RoundingMode::NearestTiesToEven;
  using Mode = mlir::arith::RoundingMode;
  switch (*payload.roundingMode) {
  case Mode::downward:
    return llvm::RoundingMode::TowardNegative;
  case Mode::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  case Mode::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case Mode::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case Mode::upward:
    return llvm::RoundingMode::TowardPositive;
  }
  llvm_unreachable("unknown generated arithmetic rounding mode");
}

mlir::Type scalarElementType(mlir::Type type) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return vector.getElementType();
  return type;
}

llvm::Expected<mlir::FloatType>
floatInputType(const PrimitiveOperationDescriptor &descriptor,
               unsigned operand) {
  if (operand >= descriptor.actor.type.getNumInputs())
    return llvm::createStringError(
        std::errc::invalid_argument, "%s has no input type at ordinal %u",
        spelling(descriptor.actor.schema).str().c_str(), operand);
  auto type = mlir::dyn_cast<mlir::FloatType>(
      scalarElementType(descriptor.actor.type.getInput(operand)));
  if (!type)
    return llvm::createStringError(
        std::errc::invalid_argument, "%s input %u is not floating-point",
        spelling(descriptor.actor.schema).str().c_str(), operand);
  return type;
}

llvm::Expected<mlir::FloatType>
floatResultType(const PrimitiveOperationDescriptor &descriptor) {
  if (descriptor.actor.type.getNumResults() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument, "%s does not have one result type",
        spelling(descriptor.actor.schema).str().c_str());
  auto type = mlir::dyn_cast<mlir::FloatType>(
      scalarElementType(descriptor.actor.type.getResult(0)));
  if (!type)
    return llvm::createStringError(
        std::errc::invalid_argument, "%s result is not floating-point",
        spelling(descriptor.actor.schema).str().c_str());
  return type;
}

llvm::Expected<llvm::APFloat> asFloat(dataflow::OperationSchemaId schema,
                                      const PrimitiveValue &value,
                                      mlir::FloatType type) {
  auto bits = requireDefinedBits(schema, value);
  if (!bits)
    return bits.takeError();
  if ((*bits)->getBitWidth() != type.getWidth())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s floating operand width does not match its projected type",
        spelling(schema).str().c_str());
  return llvm::APFloat(type.getFloatSemantics(), **bits);
}

bool violatesFloatingAssumptions(const dataflow::FloatingPointPayload &payload,
                                 llvm::ArrayRef<llvm::APFloat> operands) {
  using Flags = mlir::arith::FastMathFlags;
  if (mlir::arith::bitEnumContainsAll(payload.flags, Flags::nnan) &&
      llvm::any_of(operands,
                   [](const llvm::APFloat &value) { return value.isNaN(); }))
    return true;
  return mlir::arith::bitEnumContainsAll(payload.flags, Flags::ninf) &&
         llvm::any_of(operands, [](const llvm::APFloat &value) {
           return value.isInfinity();
         });
}

llvm::Expected<PrimitiveValue>
evaluateIntegerArithmetic(const PrimitiveOperationDescriptor &descriptor,
                          llvm::ArrayRef<PrimitiveValue> operands) {
  using Flags = mlir::arith::IntegerOverflowFlags;
  using Schema = dataflow::OperationSchemaId;
  const Schema schema = descriptor.actor.schema;
  auto payload = requirePayload<dataflow::IntegerOverflowPayload>(descriptor);
  if (!payload)
    return payload.takeError();
  if (auto exceptional = strictExceptionalResult(operands))
    return *exceptional;
  auto pair = requireIntegerPair(schema, operands);
  if (!pair)
    return pair.takeError();
  const llvm::APInt &lhs = *pair->first;
  const llvm::APInt &rhs = *pair->second;
  bool unsignedOverflow = false;
  bool signedOverflow = false;
  llvm::APInt result = lhs;
  switch (schema) {
  case Schema::ArithAddI:
    result = lhs.uadd_ov(rhs, unsignedOverflow);
    (void)lhs.sadd_ov(rhs, signedOverflow);
    break;
  case Schema::ArithSubI:
    result = lhs.usub_ov(rhs, unsignedOverflow);
    (void)lhs.ssub_ov(rhs, signedOverflow);
    break;
  case Schema::ArithMulI:
    result = lhs.umul_ov(rhs, unsignedOverflow);
    (void)lhs.smul_ov(rhs, signedOverflow);
    break;
  default:
    llvm_unreachable("integer arithmetic provider received another schema");
  }
  if ((mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nuw) &&
       unsignedOverflow) ||
      (mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nsw) &&
       signedOverflow))
    return PrimitiveValue::poison();
  return PrimitiveValue::integer(std::move(result));
}

llvm::Expected<PrimitiveValue>
evaluateFloatBinary(const PrimitiveOperationDescriptor &descriptor,
                    llvm::ArrayRef<PrimitiveValue> operands) {
  using Schema = dataflow::OperationSchemaId;
  const Schema schema = descriptor.actor.schema;
  auto payload = requirePayload<dataflow::FloatingPointPayload>(descriptor);
  if (!payload)
    return payload.takeError();
  if (auto exceptional = strictExceptionalResult(operands))
    return *exceptional;
  auto type = floatInputType(descriptor, 0);
  if (!type)
    return type.takeError();
  auto lhs = asFloat(schema, operands[0], *type);
  if (!lhs)
    return lhs.takeError();
  auto rhs = asFloat(schema, operands[1], *type);
  if (!rhs)
    return rhs.takeError();
  llvm::APFloat values[] = {*lhs, *rhs};
  if (violatesFloatingAssumptions(**payload, values))
    return PrimitiveValue::poison();
  llvm::APFloat result = *lhs;
  switch (schema) {
  case Schema::ArithAddF:
    (void)result.add(*rhs, roundingMode(**payload));
    break;
  case Schema::ArithSubF:
    (void)result.subtract(*rhs, roundingMode(**payload));
    break;
  case Schema::ArithMulF:
    (void)result.multiply(*rhs, roundingMode(**payload));
    break;
  case Schema::ArithDivF:
    (void)result.divide(*rhs, roundingMode(**payload));
    break;
  case Schema::ArithRemF:
    (void)result.mod(*rhs);
    break;
  case Schema::ArithMinimumF:
    result = llvm::minimum(*lhs, *rhs);
    break;
  case Schema::ArithMaximumF:
    result = llvm::maximum(*lhs, *rhs);
    break;
  case Schema::ArithMinNumF:
    result = llvm::minnum(*lhs, *rhs);
    break;
  case Schema::ArithMaxNumF:
    result = llvm::maxnum(*lhs, *rhs);
    break;
  default:
    llvm_unreachable("floating binary provider received another schema");
  }
  return PrimitiveValue::floating(result);
}

bool compareFloat(mlir::arith::CmpFPredicate predicate,
                  const llvm::APFloat &lhs, const llvm::APFloat &rhs) {
  const llvm::APFloat::cmpResult comparison = lhs.compare(rhs);
  const bool ordered = comparison != llvm::APFloat::cmpUnordered;
  switch (predicate) {
  case mlir::arith::CmpFPredicate::AlwaysFalse:
    return false;
  case mlir::arith::CmpFPredicate::AlwaysTrue:
    return true;
  case mlir::arith::CmpFPredicate::ORD:
    return ordered;
  case mlir::arith::CmpFPredicate::UNO:
    return !ordered;
  case mlir::arith::CmpFPredicate::OEQ:
    return comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::OGT:
    return comparison == llvm::APFloat::cmpGreaterThan;
  case mlir::arith::CmpFPredicate::OGE:
    return comparison == llvm::APFloat::cmpGreaterThan ||
           comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::OLT:
    return comparison == llvm::APFloat::cmpLessThan;
  case mlir::arith::CmpFPredicate::OLE:
    return comparison == llvm::APFloat::cmpLessThan ||
           comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::ONE:
    return ordered && comparison != llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::UEQ:
    return !ordered || comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::UGT:
    return !ordered || comparison == llvm::APFloat::cmpGreaterThan;
  case mlir::arith::CmpFPredicate::UGE:
    return !ordered || comparison == llvm::APFloat::cmpGreaterThan ||
           comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::ULT:
    return !ordered || comparison == llvm::APFloat::cmpLessThan;
  case mlir::arith::CmpFPredicate::ULE:
    return !ordered || comparison == llvm::APFloat::cmpLessThan ||
           comparison == llvm::APFloat::cmpEqual;
  case mlir::arith::CmpFPredicate::UNE:
    return !ordered || comparison != llvm::APFloat::cmpEqual;
  }
  llvm_unreachable("unknown generated floating comparison predicate");
}

llvm::Expected<PrimitiveValue> evaluateRegisteredPrimitiveOperation(
    const PrimitiveOperationDescriptor &descriptor,
    llvm::ArrayRef<PrimitiveValue> operands) {
  using Schema = dataflow::OperationSchemaId;
  const Schema schema = descriptor.actor.schema;

  if (schema == Schema::UBPoison) {
    if (llvm::Error arity = requireArity(schema, operands, 0))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    return PrimitiveValue::poison();
  }

  if (schema == Schema::ArithSelect) {
    if (llvm::Error arity = requireArity(schema, operands, 3))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    const PrimitiveValue &condition = operands[0];
    if (condition.state == PrimitiveValueState::Poison)
      return PrimitiveValue::poison();
    if (condition.state == PrimitiveValueState::Undef)
      return samePrimitiveValue(operands[1], operands[2])
                 ? operands[1]
                 : PrimitiveValue::undef();
    auto bits = requireDefinedBits(schema, condition);
    if (!bits)
      return bits.takeError();
    if ((*bits)->getBitWidth() != 1)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "arith.select condition is not i1");
    return (*bits)->isOne() ? operands[1] : operands[2];
  }

  switch (schema) {
  case Schema::ArithAddI:
  case Schema::ArithSubI:
  case Schema::ArithMulI:
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    return evaluateIntegerArithmetic(descriptor, operands);

  case Schema::ArithAndI:
  case Schema::ArithOrI:
  case Schema::ArithXOrI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    if (schema == Schema::ArithAndI)
      return PrimitiveValue::integer(*pair->first & *pair->second);
    if (schema == Schema::ArithOrI)
      return PrimitiveValue::integer(*pair->first | *pair->second);
    return PrimitiveValue::integer(*pair->first ^ *pair->second);
  }

  case Schema::LLVMOrDisjoint: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    auto payload = requirePayload<dataflow::DisjointPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (!(*payload)->isDisjoint)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "llvm.or canonical actor is missing its disjoint contract");
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    if (!(*pair->first & *pair->second).isZero())
      return PrimitiveValue::poison();
    return PrimitiveValue::integer(*pair->first | *pair->second);
  }

  case Schema::ArithShLI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    auto payload = requirePayload<dataflow::IntegerOverflowPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto lhs = requireDefinedBits(schema, operands[0]);
    auto rhs = requireDefinedBits(schema, operands[1]);
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    auto amount = checkedShiftAmount(**rhs, (*lhs)->getBitWidth());
    if (!amount)
      return PrimitiveValue::poison();
    bool unsignedOverflow = false;
    bool signedOverflow = false;
    llvm::APInt result = (*lhs)->ushl_ov(*amount, unsignedOverflow);
    (void)(*lhs)->sshl_ov(*amount, signedOverflow);
    using Flags = mlir::arith::IntegerOverflowFlags;
    if ((mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nuw) &&
         unsignedOverflow) ||
        (mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nsw) &&
         signedOverflow))
      return PrimitiveValue::poison();
    return PrimitiveValue::integer(std::move(result));
  }

  case Schema::ArithShRSI:
  case Schema::ArithShRUI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    auto payload = requirePayload<dataflow::ExactPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto lhs = requireDefinedBits(schema, operands[0]);
    auto rhs = requireDefinedBits(schema, operands[1]);
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    auto amount = checkedShiftAmount(**rhs, (*lhs)->getBitWidth());
    if (!amount)
      return PrimitiveValue::poison();
    if ((*payload)->isExact && *amount != 0 &&
        !(*lhs)->getLoBits(*amount).isZero())
      return PrimitiveValue::poison();
    return PrimitiveValue::integer(schema == Schema::ArithShRUI
                                       ? (*lhs)->lshr(*amount)
                                       : (*lhs)->ashr(*amount));
  }

  case Schema::ArithDivSI:
  case Schema::ArithDivUI:
  case Schema::ArithRemSI:
  case Schema::ArithRemUI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    if (schema == Schema::ArithDivSI || schema == Schema::ArithDivUI) {
      if (auto payload = requirePayload<dataflow::ExactPayload>(descriptor);
          !payload)
        return payload.takeError();
    } else if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
               !payload) {
      return payload.takeError();
    }
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    const llvm::APInt &lhs = *pair->first;
    const llvm::APInt &rhs = *pair->second;
    if (rhs.isZero())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "%s division by zero is undefined",
                                     spelling(schema).str().c_str());
    if (schema == Schema::ArithDivSI && lhs.isMinSignedValue() &&
        rhs.isAllOnes())
      return llvm::createStringError(std::errc::result_out_of_range,
                                     "%s signed division overflow is undefined",
                                     spelling(schema).str().c_str());
    if (schema == Schema::ArithDivSI || schema == Schema::ArithDivUI) {
      auto payload = requirePayload<dataflow::ExactPayload>(descriptor);
      if (!payload)
        return payload.takeError();
      const llvm::APInt remainder =
          schema == Schema::ArithDivSI ? lhs.srem(rhs) : lhs.urem(rhs);
      if ((*payload)->isExact && !remainder.isZero())
        return PrimitiveValue::poison();
      return PrimitiveValue::integer(
          schema == Schema::ArithDivSI ? lhs.sdiv(rhs) : lhs.udiv(rhs));
    }
    return PrimitiveValue::integer(
        schema == Schema::ArithRemSI ? lhs.srem(rhs) : lhs.urem(rhs));
  }

  case Schema::ArithMinSI:
  case Schema::ArithMaxSI:
  case Schema::ArithMinUI:
  case Schema::ArithMaxUI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    switch (schema) {
    case Schema::ArithMinSI:
      return PrimitiveValue::integer(
          llvm::APIntOps::smin(*pair->first, *pair->second));
    case Schema::ArithMaxSI:
      return PrimitiveValue::integer(
          llvm::APIntOps::smax(*pair->first, *pair->second));
    case Schema::ArithMinUI:
      return PrimitiveValue::integer(
          llvm::APIntOps::umin(*pair->first, *pair->second));
    case Schema::ArithMaxUI:
      return PrimitiveValue::integer(
          llvm::APIntOps::umax(*pair->first, *pair->second));
    default:
      llvm_unreachable("integer min/max provider received another schema");
    }
  }

  case Schema::ArithCmpI: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    auto payload = requirePayload<dataflow::IntegerComparePayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    const llvm::APInt &lhs = *pair->first;
    const llvm::APInt &rhs = *pair->second;
    bool result = false;
    switch ((*payload)->predicate) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhs == rhs;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhs != rhs;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhs.slt(rhs);
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhs.sle(rhs);
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhs.sgt(rhs);
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhs.sge(rhs);
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = lhs.ult(rhs);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = lhs.ule(rhs);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = lhs.ugt(rhs);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = lhs.uge(rhs);
      break;
    }
    return PrimitiveValue::boolean(result);
  }

  case Schema::ArithExtSI:
  case Schema::ArithExtUI:
  case Schema::ArithTruncI:
  case Schema::ArithIndexCast:
  case Schema::ArithIndexCastUI:
  case Schema::ArithBitcast: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    if (schema == Schema::ArithExtUI || schema == Schema::ArithIndexCastUI) {
      if (auto payload =
              requirePayload<dataflow::NonNegativePayload>(descriptor);
          !payload)
        return payload.takeError();
    } else if (schema == Schema::ArithTruncI) {
      if (auto payload =
              requirePayload<dataflow::IntegerOverflowPayload>(descriptor);
          !payload)
        return payload.takeError();
    } else if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
               !payload) {
      return payload.takeError();
    }
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto input = requireDefinedBits(schema, operands[0]);
    if (!input)
      return input.takeError();
    auto resultWidth = requireResultBitWidth(descriptor);
    if (!resultWidth)
      return resultWidth.takeError();
    if (schema == Schema::ArithBitcast) {
      if ((*input)->getBitWidth() != *resultWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "arith.bitcast operand and result widths do not match");
      return PrimitiveValue::integer(**input);
    }
    if (schema == Schema::ArithExtSI) {
      if ((*input)->getBitWidth() >= *resultWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "arith.extsi result must be wider than its operand");
      return PrimitiveValue::integer((*input)->sext(*resultWidth));
    }
    if (schema == Schema::ArithExtUI || schema == Schema::ArithIndexCastUI) {
      auto payload = requirePayload<dataflow::NonNegativePayload>(descriptor);
      if (!payload)
        return payload.takeError();
      if ((*payload)->isNonNegative && (*input)->isNegative())
        return PrimitiveValue::poison();
      if (schema == Schema::ArithExtUI &&
          (*input)->getBitWidth() >= *resultWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "arith.extui result must be wider than its operand");
      return PrimitiveValue::integer((*input)->zextOrTrunc(*resultWidth));
    }
    if (schema == Schema::ArithIndexCast)
      return PrimitiveValue::integer((*input)->sextOrTrunc(*resultWidth));

    auto payload = requirePayload<dataflow::IntegerOverflowPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if ((*input)->getBitWidth() <= *resultWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "arith.trunci result must be narrower than its operand");
    llvm::APInt result = (*input)->trunc(*resultWidth);
    using Flags = mlir::arith::IntegerOverflowFlags;
    if (mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nuw) &&
        result.zext((*input)->getBitWidth()) != **input)
      return PrimitiveValue::poison();
    if (mlir::arith::bitEnumContainsAll((*payload)->flags, Flags::nsw) &&
        result.sext((*input)->getBitWidth()) != **input)
      return PrimitiveValue::poison();
    return PrimitiveValue::integer(std::move(result));
  }

  case Schema::ArithAddF:
  case Schema::ArithSubF:
  case Schema::ArithMulF:
  case Schema::ArithDivF:
  case Schema::ArithRemF:
  case Schema::ArithMinimumF:
  case Schema::ArithMaximumF:
  case Schema::ArithMinNumF:
  case Schema::ArithMaxNumF:
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    return evaluateFloatBinary(descriptor, operands);

  case Schema::ArithNegF:
  case Schema::MathAbsF:
  case Schema::MathSin:
  case Schema::MathCos:
  case Schema::MathTan:
  case Schema::MathSinh:
  case Schema::MathCosh:
  case Schema::MathTanh:
  case Schema::MathExp:
  case Schema::MathExp2:
  case Schema::MathExpM1:
  case Schema::MathLog:
  case Schema::MathLog2:
  case Schema::MathLog10:
  case Schema::MathLog1p:
  case Schema::MathFloor:
  case Schema::MathCeil:
  case Schema::MathRound:
  case Schema::MathTrunc:
  case Schema::MathRoundEven:
  case Schema::MathSqrt:
  case Schema::MathRsqrt:
  case Schema::MathErf: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    auto payload = requirePayload<dataflow::FloatingPointPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto type = floatInputType(descriptor, 0);
    if (!type)
      return type.takeError();
    auto value = asFloat(schema, operands[0], *type);
    if (!value)
      return value.takeError();
    llvm::APFloat values[] = {*value};
    if (violatesFloatingAssumptions(**payload, values))
      return PrimitiveValue::poison();
    switch (schema) {
    case Schema::ArithNegF:
      value->changeSign();
      break;
    case Schema::MathAbsF:
      value->clearSign();
      break;
    case Schema::MathSin:
    case Schema::MathCos:
    case Schema::MathTan:
    case Schema::MathSinh:
    case Schema::MathCosh:
    case Schema::MathTanh:
    case Schema::MathExp:
    case Schema::MathExp2:
    case Schema::MathExpM1:
    case Schema::MathLog:
    case Schema::MathLog2:
    case Schema::MathLog10:
    case Schema::MathLog1p:
    case Schema::MathSqrt:
    case Schema::MathRsqrt:
    case Schema::MathErf: {
      auto result =
          loom::sim::detail::evaluateDeterministicUnaryMath(schema, *value);
      if (!result)
        return result.takeError();
      return PrimitiveValue::floating(*result);
    }
    case Schema::MathFloor:
      (void)value->roundToIntegral(llvm::RoundingMode::TowardNegative);
      break;
    case Schema::MathCeil:
      (void)value->roundToIntegral(llvm::RoundingMode::TowardPositive);
      break;
    case Schema::MathRound:
      (void)value->roundToIntegral(llvm::RoundingMode::NearestTiesToAway);
      break;
    case Schema::MathTrunc:
      (void)value->roundToIntegral(llvm::RoundingMode::TowardZero);
      break;
    case Schema::MathRoundEven:
      (void)value->roundToIntegral(llvm::RoundingMode::NearestTiesToEven);
      break;
    default:
      llvm_unreachable("floating unary provider received another schema");
    }
    return PrimitiveValue::floating(*value);
  }

  case Schema::ArithCmpF: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    auto payload = requirePayload<dataflow::FloatComparePayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto type = floatInputType(descriptor, 0);
    if (!type)
      return type.takeError();
    auto lhs = asFloat(schema, operands[0], *type);
    auto rhs = asFloat(schema, operands[1], *type);
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    dataflow::FloatingPointPayload assumptions{(*payload)->flags, std::nullopt};
    llvm::APFloat values[] = {*lhs, *rhs};
    if (violatesFloatingAssumptions(assumptions, values))
      return PrimitiveValue::poison();
    return PrimitiveValue::boolean(
        compareFloat((*payload)->predicate, *lhs, *rhs));
  }

  case Schema::ArithExtF:
  case Schema::ArithTruncF: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    auto payload = requirePayload<dataflow::FloatingPointPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto sourceType = floatInputType(descriptor, 0);
    auto resultType = floatResultType(descriptor);
    if (!sourceType)
      return sourceType.takeError();
    if (!resultType)
      return resultType.takeError();
    auto value = asFloat(schema, operands[0], *sourceType);
    if (!value)
      return value.takeError();
    bool losesInfo = false;
    (void)value->convert(resultType->getFloatSemantics(),
                         roundingMode(**payload), &losesInfo);
    return PrimitiveValue::floating(*value);
  }

  case Schema::ArithSIToFP:
  case Schema::ArithUIToFP: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    if (schema == Schema::ArithUIToFP) {
      if (auto payload =
              requirePayload<dataflow::NonNegativePayload>(descriptor);
          !payload)
        return payload.takeError();
    } else if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
               !payload) {
      return payload.takeError();
    }
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto input = requireDefinedBits(schema, operands[0]);
    if (!input)
      return input.takeError();
    if (schema == Schema::ArithUIToFP) {
      auto payload = requirePayload<dataflow::NonNegativePayload>(descriptor);
      if (!payload)
        return payload.takeError();
      if ((*payload)->isNonNegative && (*input)->isNegative())
        return PrimitiveValue::poison();
    }
    auto resultType = floatResultType(descriptor);
    if (!resultType)
      return resultType.takeError();
    llvm::APFloat result =
        llvm::APFloat::getZero(resultType->getFloatSemantics());
    (void)result.convertFromAPInt(**input, schema == Schema::ArithSIToFP,
                                  llvm::RoundingMode::NearestTiesToEven);
    return PrimitiveValue::floating(result);
  }

  case Schema::ArithFPToSI:
  case Schema::ArithFPToUI:
  case Schema::LLVMFPToSISat:
  case Schema::LLVMFPToUISat: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto type = floatInputType(descriptor, 0);
    auto width = requireResultBitWidth(descriptor);
    if (!type)
      return type.takeError();
    if (!width)
      return width.takeError();
    auto value = asFloat(schema, operands[0], *type);
    if (!value)
      return value.takeError();
    const bool isUnsigned =
        schema == Schema::ArithFPToUI || schema == Schema::LLVMFPToUISat;
    const bool isSaturating =
        schema == Schema::LLVMFPToSISat || schema == Schema::LLVMFPToUISat;
    llvm::APSInt result(*width, isUnsigned);
    bool exact = false;
    llvm::APFloat::opStatus status =
        value->convertToInteger(result, llvm::RoundingMode::TowardZero, &exact);
    if (!isSaturating && (status & llvm::APFloat::opInvalidOp) != 0)
      return PrimitiveValue::poison();
    return PrimitiveValue::integer(result);
  }

  case Schema::MathFma: {
    if (llvm::Error arity = requireArity(schema, operands, 3))
      return std::move(arity);
    auto payload = requirePayload<dataflow::FloatingPointPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto type = floatInputType(descriptor, 0);
    if (!type)
      return type.takeError();
    auto lhs = asFloat(schema, operands[0], *type);
    auto rhs = asFloat(schema, operands[1], *type);
    auto addend = asFloat(schema, operands[2], *type);
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    if (!addend)
      return addend.takeError();
    llvm::APFloat values[] = {*lhs, *rhs, *addend};
    if (violatesFloatingAssumptions(**payload, values))
      return PrimitiveValue::poison();
    (void)lhs->fusedMultiplyAdd(*rhs, *addend, roundingMode(**payload));
    return PrimitiveValue::floating(*lhs);
  }

  case Schema::MathAbsI: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto value = requireDefinedBits(schema, operands[0]);
    if (!value)
      return value.takeError();
    if ((*value)->isMinSignedValue())
      return PrimitiveValue::poison();
    return PrimitiveValue::integer((*value)->abs());
  }

  case Schema::LLVMFshl: {
    if (llvm::Error arity = requireArity(schema, operands, 3))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto high = requireDefinedBits(schema, operands[0]);
    auto low = requireDefinedBits(schema, operands[1]);
    auto amount = requireDefinedBits(schema, operands[2]);
    if (!high)
      return high.takeError();
    if (!low)
      return low.takeError();
    if (!amount)
      return amount.takeError();
    if ((*high)->getBitWidth() != (*low)->getBitWidth())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "llvm.intr.fshl operand widths differ");
    return PrimitiveValue::integer(
        llvm::APIntOps::fshl(**high, **low, **amount));
  }

  case Schema::LLVMByteSwap: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto value = requireDefinedBits(schema, operands[0]);
    if (!value)
      return value.takeError();
    if ((*value)->getBitWidth() % 16 != 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "llvm.intr.bswap bit width must be a multiple of 16");
    return PrimitiveValue::integer((*value)->byteSwap());
  }

  case Schema::LLVMSAddSat:
  case Schema::LLVMUAddSat:
  case Schema::LLVMSSubSat:
  case Schema::LLVMUSubSat: {
    if (llvm::Error arity = requireArity(schema, operands, 2))
      return std::move(arity);
    if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
        !payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto pair = requireIntegerPair(schema, operands);
    if (!pair)
      return pair.takeError();
    switch (schema) {
    case Schema::LLVMSAddSat:
      return PrimitiveValue::integer(pair->first->sadd_sat(*pair->second));
    case Schema::LLVMUAddSat:
      return PrimitiveValue::integer(pair->first->uadd_sat(*pair->second));
    case Schema::LLVMSSubSat:
      return PrimitiveValue::integer(pair->first->ssub_sat(*pair->second));
    case Schema::LLVMUSubSat:
      return PrimitiveValue::integer(pair->first->usub_sat(*pair->second));
    default:
      llvm_unreachable(
          "saturating arithmetic provider received another schema");
    }
  }

  case Schema::MathCountLeadingZeros:
  case Schema::MathCountTrailingZeros:
  case Schema::LLVMCountLeadingZeros:
  case Schema::LLVMCountTrailingZeros: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    const bool llvmPoisonForm = schema == Schema::LLVMCountLeadingZeros ||
                                schema == Schema::LLVMCountTrailingZeros;
    if (llvmPoisonForm) {
      if (auto payload =
              requirePayload<dataflow::ZeroPoisonPayload>(descriptor);
          !payload)
        return payload.takeError();
    } else if (auto payload = requirePayload<dataflow::NoPayload>(descriptor);
               !payload) {
      return payload.takeError();
    }
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto value = requireDefinedBits(schema, operands[0]);
    if (!value)
      return value.takeError();
    if (llvmPoisonForm) {
      auto payload = requirePayload<dataflow::ZeroPoisonPayload>(descriptor);
      if (!payload)
        return payload.takeError();
      if ((*payload)->isZeroPoison && (*value)->isZero())
        return PrimitiveValue::poison();
    }
    const unsigned count = schema == Schema::MathCountTrailingZeros ||
                                   schema == Schema::LLVMCountTrailingZeros
                               ? (*value)->countr_zero()
                               : (*value)->countl_zero();
    return PrimitiveValue::integer(llvm::APInt((*value)->getBitWidth(), count));
  }

  case Schema::LLVMAbs: {
    if (llvm::Error arity = requireArity(schema, operands, 1))
      return std::move(arity);
    auto payload =
        requirePayload<dataflow::IntegerMinPoisonPayload>(descriptor);
    if (!payload)
      return payload.takeError();
    if (auto exceptional = strictExceptionalResult(operands))
      return *exceptional;
    auto value = requireDefinedBits(schema, operands[0]);
    if (!value)
      return value.takeError();
    if ((*value)->isMinSignedValue())
      return (*payload)->isIntMinPoison ? PrimitiveValue::poison()
                                        : PrimitiveValue::integer(**value);
    return PrimitiveValue::integer((*value)->abs());
  }

  case Schema::ArithSelect:
  case Schema::UBPoison:
    llvm_unreachable("special primitive schema escaped its dispatch");
  default:
    break;
  }

  return llvm::createStringError(std::errc::not_supported,
                                 "%s has no exact primitive provider",
                                 spelling(schema).str().c_str());
}

} // namespace

PrimitiveValue PrimitiveValue::integer(llvm::APInt value) {
  PrimitiveValue result;
  result.state = PrimitiveValueState::Defined;
  result.bits = std::move(value);
  return result;
}

PrimitiveValue PrimitiveValue::floating(const llvm::APFloat &value) {
  return integer(value.bitcastToAPInt());
}

PrimitiveValue PrimitiveValue::boolean(bool value) {
  return integer(llvm::APInt(1, value ? 1 : 0));
}

PrimitiveValue PrimitiveValue::poison() {
  PrimitiveValue result;
  result.state = PrimitiveValueState::Poison;
  return result;
}

PrimitiveValue PrimitiveValue::undef() { return PrimitiveValue{}; }

bool loom::sim::isSupportedPrimitiveOperation(
    dataflow::OperationSchemaId schema) {
  return primitiveOperationProvider(schema) != nullptr;
}

llvm::Expected<PrimitiveValue> loom::sim::evaluatePrimitiveOperation(
    const PrimitiveOperationDescriptor &descriptor,
    llvm::ArrayRef<PrimitiveValue> operands) {
  PrimitiveOperationProvider provider =
      primitiveOperationProvider(descriptor.actor.schema);
  if (!provider)
    return llvm::createStringError(
        std::errc::not_supported, "%s has no exact primitive provider",
        spelling(descriptor.actor.schema).str().c_str());
  return provider(descriptor, operands);
}
