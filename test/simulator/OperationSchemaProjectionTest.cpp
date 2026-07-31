#include "Simulator/OperationSemantics.h"

#include "Dataflow/IR/OperationSchema.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using dataflow::CanonicalActorSchemaProjection;
using dataflow::OperationSchemaId;
using loom::sim::PrimitiveOperationDescriptor;
using loom::sim::PrimitiveValue;
using loom::sim::PrimitiveValueState;

using SupportSignature = bool (*)(OperationSchemaId);
static_assert(
    std::is_same_v<decltype(&loom::sim::isSupportedPrimitiveOperation),
                   SupportSignature>);

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

PrimitiveValue takeValue(llvm::StringRef test,
                         llvm::Expected<PrimitiveValue> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return *value;
}

void requireRejected(llvm::StringRef test, llvm::Expected<PrimitiveValue> value,
                     llvm::StringRef expected) {
  if (value)
    fail(test, "operation unexpectedly succeeded");
  std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

const llvm::APInt &definedBits(llvm::StringRef test,
                               const PrimitiveValue &value) {
  require(test, value.isDefined(), "expected a defined primitive value");
  return *value.bits;
}

PrimitiveValue integer(unsigned width, std::uint64_t value) {
  return PrimitiveValue::integer(llvm::APInt(width, value));
}

PrimitiveValue floating64(std::uint64_t bits) {
  return PrimitiveValue::floating(
      llvm::APFloat(llvm::APFloat::IEEEdouble(), llvm::APInt(64, bits)));
}

PrimitiveValue floating32(std::uint32_t bits) {
  return PrimitiveValue::floating(
      llvm::APFloat(llvm::APFloat::IEEEsingle(), llvm::APInt(32, bits)));
}

PrimitiveOperationDescriptor descriptor(OperationSchemaId schema,
                                        mlir::FunctionType type,
                                        dataflow::SemanticPayload payload,
                                        unsigned resultBitWidth,
                                        unsigned operandBitWidth) {
  return PrimitiveOperationDescriptor{
      CanonicalActorSchemaProjection{schema, type, std::move(payload)},
      resultBitWidth, operandBitWidth};
}

void checkTypedDispatchAndPayload() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i1 = mlir::IntegerType::get(&context, 1);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::Type i16 = mlir::IntegerType::get(&context, 16);
  mlir::Type f64 = mlir::Float64Type::get(&context);

  PrimitiveOperationDescriptor integerCompare = descriptor(
      OperationSchemaId::ArithCmpI,
      mlir::FunctionType::get(&context, {i8, i8}, {i1}),
      dataflow::IntegerComparePayload{mlir::arith::CmpIPredicate::slt}, 1, 8);
  const PrimitiveValue integerOperands[] = {integer(8, 255), integer(8, 1)};
  PrimitiveValue integerResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(integerCompare, integerOperands));
  require(__func__, definedBits(__func__, integerResult).isOne(),
          "typed signed predicate was not applied");

  PrimitiveOperationDescriptor floatCompare = descriptor(
      OperationSchemaId::ArithCmpF,
      mlir::FunctionType::get(&context, {f64, f64}, {i1}),
      dataflow::FloatComparePayload{mlir::arith::CmpFPredicate::UNO,
                                    mlir::arith::FastMathFlags::none},
      1, 0);
  const PrimitiveValue floatOperands[] = {
      PrimitiveValue::floating(
          llvm::APFloat::getNaN(llvm::APFloat::IEEEdouble(), /*Negative=*/false,
                                /*Payload=*/0)),
      floating64(0x3ff0000000000000ULL)};
  PrimitiveValue floatResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(floatCompare,
                                                                floatOperands));
  require(__func__, definedBits(__func__, floatResult).isOne(),
          "typed unordered floating predicate was not applied");

  PrimitiveOperationDescriptor exactShift =
      descriptor(OperationSchemaId::ArithShRUI,
                 mlir::FunctionType::get(&context, {i8, i8}, {i8}),
                 dataflow::ExactPayload{true}, 8, 8);
  const PrimitiveValue shiftOperands[] = {integer(8, 3), integer(8, 1)};
  PrimitiveValue shiftResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(exactShift,
                                                                shiftOperands));
  require(__func__, shiftResult.state == PrimitiveValueState::Poison,
          "exact shift did not produce poison");

  PrimitiveOperationDescriptor truncation = descriptor(
      OperationSchemaId::ArithTruncI,
      mlir::FunctionType::get(&context, {i16}, {i8}),
      dataflow::IntegerOverflowPayload{mlir::arith::IntegerOverflowFlags::nuw},
      8, 16);
  const PrimitiveValue truncationOperands[] = {integer(16, 256)};
  PrimitiveValue truncationResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(truncation, truncationOperands));
  require(__func__, truncationResult.state == PrimitiveValueState::Poison,
          "truncation overflow did not produce poison");

  PrimitiveOperationDescriptor malformed = descriptor(
      OperationSchemaId::ArithAddI,
      mlir::FunctionType::get(&context, {i8, i8}, {i8}),
      dataflow::IntegerComparePayload{mlir::arith::CmpIPredicate::eq}, 8, 8);
  requireRejected(
      __func__,
      loom::sim::evaluatePrimitiveOperation(malformed, integerOperands),
      "does not match operation schema");
}

void checkTypedLLVMExceptionalPolicies() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::FunctionType unaryI8 = mlir::FunctionType::get(&context, {i8}, {i8});

  PrimitiveOperationDescriptor zeroPoison =
      descriptor(OperationSchemaId::LLVMCountLeadingZeros, unaryI8,
                 dataflow::ZeroPoisonPayload{true}, 8, 8);
  const PrimitiveValue zero[] = {integer(8, 0)};
  PrimitiveValue zeroResult = takeValue(
      __func__, loom::sim::evaluatePrimitiveOperation(zeroPoison, zero));
  require(__func__, zeroResult.state == PrimitiveValueState::Poison,
          "LLVM ctlz zero policy did not produce poison");

  PrimitiveOperationDescriptor trailingZeroPoison =
      descriptor(OperationSchemaId::LLVMCountTrailingZeros, unaryI8,
                 dataflow::ZeroPoisonPayload{true}, 8, 8);
  PrimitiveValue trailingZeroResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(trailingZeroPoison, zero));
  require(__func__, trailingZeroResult.state == PrimitiveValueState::Poison,
          "LLVM cttz zero policy did not produce poison");

  PrimitiveOperationDescriptor definedTrailingZeros =
      descriptor(OperationSchemaId::MathCountTrailingZeros, unaryI8,
                 dataflow::NoPayload{}, 8, 8);
  const PrimitiveValue twentyFour[] = {integer(8, 24)};
  PrimitiveValue trailingResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(definedTrailingZeros, twentyFour));
  require(__func__, definedBits(__func__, trailingResult) == llvm::APInt(8, 3),
          "math cttz produced the wrong trailing-zero count");

  const PrimitiveValue signedMinimum[] = {integer(8, 128)};
  PrimitiveOperationDescriptor wrappingAbs =
      descriptor(OperationSchemaId::LLVMAbs, unaryI8,
                 dataflow::IntegerMinPoisonPayload{false}, 8, 8);
  PrimitiveValue wrappingResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(wrappingAbs,
                                                                signedMinimum));
  require(__func__,
          definedBits(__func__, wrappingResult) == llvm::APInt(8, 128),
          "non-poisoning LLVM abs did not preserve signed minimum");

  PrimitiveOperationDescriptor poisonAbs =
      descriptor(OperationSchemaId::LLVMAbs, unaryI8,
                 dataflow::IntegerMinPoisonPayload{true}, 8, 8);
  PrimitiveValue poisonResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(poisonAbs,
                                                                signedMinimum));
  require(__func__, poisonResult.state == PrimitiveValueState::Poison,
          "LLVM abs minimum policy did not produce poison");

  mlir::FunctionType binaryI8 =
      mlir::FunctionType::get(&context, {i8, i8}, {i8});
  PrimitiveOperationDescriptor disjointOr =
      descriptor(OperationSchemaId::LLVMOrDisjoint, binaryI8,
                 dataflow::DisjointPayload{true}, 8, 8);
  const PrimitiveValue disjointOperands[] = {integer(8, 0x30),
                                             integer(8, 0x0c)};
  PrimitiveValue disjointResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(disjointOr, disjointOperands));
  require(__func__,
          definedBits(__func__, disjointResult) == llvm::APInt(8, 0x3c),
          "disjoint LLVM or did not combine non-overlapping operands");

  const PrimitiveValue overlappingOperands[] = {integer(8, 0x30),
                                                integer(8, 0x10)};
  PrimitiveValue overlappingResult = takeValue(
      __func__,
      loom::sim::evaluatePrimitiveOperation(disjointOr, overlappingOperands));
  require(__func__, overlappingResult.state == PrimitiveValueState::Poison,
          "disjoint LLVM or did not poison overlapping operands");
}

void checkTypedIntegerPolicies() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::Type i16 = mlir::IntegerType::get(&context, 16);

  PrimitiveOperationDescriptor wrappingAdd = descriptor(
      OperationSchemaId::ArithAddI,
      mlir::FunctionType::get(&context, {i8, i8}, {i8}),
      dataflow::IntegerOverflowPayload{mlir::arith::IntegerOverflowFlags::nuw},
      8, 8);
  const PrimitiveValue addOperands[] = {integer(8, 255), integer(8, 1)};
  PrimitiveValue addResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(wrappingAdd,
                                                                addOperands));
  require(__func__, addResult.state == PrimitiveValueState::Poison,
          "integer overflow did not produce poison");

  PrimitiveOperationDescriptor nonNegativeExtend =
      descriptor(OperationSchemaId::ArithExtUI,
                 mlir::FunctionType::get(&context, {i8}, {i16}),
                 dataflow::NonNegativePayload{true}, 16, 8);
  const PrimitiveValue negativeOperand[] = {integer(8, 255)};
  PrimitiveValue extendResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(
                              nonNegativeExtend, negativeOperand));
  require(__func__, extendResult.state == PrimitiveValueState::Poison,
          "non-negative assumption did not produce poison");

  PrimitiveOperationDescriptor add =
      descriptor(OperationSchemaId::ArithAddI,
                 mlir::FunctionType::get(&context, {i8, i8}, {i8}),
                 dataflow::IntegerOverflowPayload{}, 8, 8);
  const PrimitiveValue undefOperands[] = {PrimitiveValue::undef(),
                                          integer(8, 1)};
  PrimitiveValue undefResult = takeValue(
      __func__, loom::sim::evaluatePrimitiveOperation(add, undefOperands));
  require(__func__, undefResult.state == PrimitiveValueState::Undef,
          "strict integer operation did not propagate undef");
}

void checkSaturatingIntegerFamily() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::FunctionType binaryI8 =
      mlir::FunctionType::get(&context, {i8, i8}, {i8});
  struct SaturatingCase {
    OperationSchemaId schema;
    std::uint8_t lhs;
    std::uint8_t rhs;
    std::uint8_t expected;
  };
  const std::array cases = {
      SaturatingCase{OperationSchemaId::LLVMSAddSat, 120, 20, 127},
      SaturatingCase{OperationSchemaId::LLVMUAddSat, 250, 10, 255},
      SaturatingCase{OperationSchemaId::LLVMSSubSat, 136, 20, 128},
      SaturatingCase{OperationSchemaId::LLVMUSubSat, 3, 5, 0},
  };
  for (const SaturatingCase &entry : cases) {
    require(__func__, loom::sim::isSupportedPrimitiveOperation(entry.schema),
            "registered saturating arithmetic has no primitive provider");
    PrimitiveOperationDescriptor operation =
        descriptor(entry.schema, binaryI8, dataflow::NoPayload{}, 8, 8);
    const PrimitiveValue operands[] = {integer(8, entry.lhs),
                                       integer(8, entry.rhs)};
    PrimitiveValue result = takeValue(
        __func__, loom::sim::evaluatePrimitiveOperation(operation, operands));
    require(__func__,
            definedBits(__func__, result) == llvm::APInt(8, entry.expected),
            "saturating arithmetic produced the wrong boundary value");
  }
}

void checkSaturatingFloatToInteger() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::FunctionType conversion =
      mlir::FunctionType::get(&context, {f32}, {i8});
  PrimitiveOperationDescriptor signedSaturating =
      descriptor(OperationSchemaId::LLVMFPToSISat, conversion,
                 dataflow::NoPayload{}, 8, 32);
  PrimitiveOperationDescriptor unsignedSaturating =
      descriptor(OperationSchemaId::LLVMFPToUISat, conversion,
                 dataflow::NoPayload{}, 8, 32);

  require(__func__,
          loom::sim::isSupportedPrimitiveOperation(
              OperationSchemaId::LLVMFPToSISat),
          "registered signed saturating conversion has no primitive provider");
  require(
      __func__,
      loom::sim::isSupportedPrimitiveOperation(
          OperationSchemaId::LLVMFPToUISat),
      "registered unsigned saturating conversion has no primitive provider");

  PrimitiveValue signedOverflow =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(
                              signedSaturating, {floating32(0x4302c000U)}));
  require(__func__,
          definedBits(__func__, signedOverflow) == llvm::APInt(8, 127),
          "signed saturating conversion did not clamp its upper bound");

  PrimitiveValue unsignedUnderflow =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(
                              unsignedSaturating, {floating32(0xbf800000U)}));
  require(__func__, definedBits(__func__, unsignedUnderflow).isZero(),
          "unsigned saturating conversion did not clamp a negative input");

  PrimitiveValue nan = PrimitiveValue::floating(
      llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle(), /*Negative=*/false,
                            /*Payload=*/0));
  PrimitiveValue nanResult = takeValue(
      __func__, loom::sim::evaluatePrimitiveOperation(signedSaturating, {nan}));
  require(__func__, definedBits(__func__, nanResult).isZero(),
          "saturating conversion did not map NaN to zero");

  PrimitiveValue poisonResult =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(
                              signedSaturating, {PrimitiveValue::poison()}));
  require(__func__, poisonResult.state == PrimitiveValueState::Poison,
          "saturating conversion did not propagate poison");
}

void checkLazySelectionAndFusedFma() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i1 = mlir::IntegerType::get(&context, 1);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::Type f64 = mlir::Float64Type::get(&context);

  PrimitiveOperationDescriptor select =
      descriptor(OperationSchemaId::ArithSelect,
                 mlir::FunctionType::get(&context, {i1, i8, i8}, {i8}),
                 dataflow::NoPayload{}, 8, 8);
  const PrimitiveValue selectOperands[] = {
      PrimitiveValue::boolean(true), integer(8, 7), PrimitiveValue::poison()};
  PrimitiveValue selected = takeValue(
      __func__, loom::sim::evaluatePrimitiveOperation(select, selectOperands));
  require(__func__, definedBits(__func__, selected) == llvm::APInt(8, 7),
          "select observed its unselected poison operand");

  PrimitiveOperationDescriptor fma =
      descriptor(OperationSchemaId::MathFma,
                 mlir::FunctionType::get(&context, {f64, f64, f64}, {f64}),
                 dataflow::FloatingPointPayload{}, 0, 0);
  const PrimitiveValue fmaOperands[] = {floating64(0x3ff0000000000001ULL),
                                        floating64(0x3fffffffffffffffULL),
                                        floating64(0xbfffffffffffffffULL)};
  PrimitiveValue fused = takeValue(
      __func__, loom::sim::evaluatePrimitiveOperation(fma, fmaOperands));
  require(__func__,
          definedBits(__func__, fused) ==
              llvm::APInt(64, 0x3cbfffffffffffffULL),
          "math.fma was rounded as separate multiply and add operations");
}

void checkTypedProviderAvailability() {
  require(
      __func__,
      loom::sim::isSupportedPrimitiveOperation(OperationSchemaId::ArithAddI),
      "registered integer addition has no primitive provider");
  require(__func__,
          !loom::sim::isSupportedPrimitiveOperation(
              OperationSchemaId::DataflowStream),
          "control actor was classified as a primitive provider");
  require(
      __func__,
      !loom::sim::isSupportedPrimitiveOperation(OperationSchemaId::LLVMFreeze),
      "freeze was accepted without its deterministic execution key");
  require(__func__,
          loom::sim::isSupportedPrimitiveOperation(OperationSchemaId::MathSin),
          "registered sine has no deterministic primitive provider");
  require(__func__,
          loom::sim::isSupportedPrimitiveOperation(OperationSchemaId::UBPoison),
          "registered poison value has no primitive provider");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  PrimitiveOperationDescriptor poison = descriptor(
      OperationSchemaId::UBPoison, mlir::FunctionType::get(&context, {}, {i8}),
      dataflow::NoPayload{}, 8, 0);
  PrimitiveValue poisonValue =
      takeValue(__func__, loom::sim::evaluatePrimitiveOperation(poison, {}));
  require(__func__, poisonValue.state == PrimitiveValueState::Poison,
          "ub.poison did not produce a poison value");
}

void checkDeterministicCosineProvider() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const struct CosineCase {
    mlir::Type type;
    PrimitiveValue zero;
    llvm::APInt one;
  } cases[] = {
      {mlir::Float32Type::get(&context), floating32(0),
       llvm::APInt(32, 0x3f800000U)},
      {mlir::Float64Type::get(&context), floating64(0),
       llvm::APInt(64, 0x3ff0000000000000ULL)},
  };

  require(__func__,
          loom::sim::isSupportedPrimitiveOperation(OperationSchemaId::MathCos),
          "typed cosine has no deterministic primitive provider");
  for (const CosineCase &entry : cases) {
    PrimitiveOperationDescriptor cosine = descriptor(
        OperationSchemaId::MathCos,
        mlir::FunctionType::get(&context, {entry.type}, {entry.type}),
        dataflow::FloatingPointPayload{}, entry.one.getBitWidth(),
        entry.one.getBitWidth());
    PrimitiveValue result = takeValue(
        __func__, loom::sim::evaluatePrimitiveOperation(cosine, {entry.zero}));
    require(__func__, definedBits(__func__, result) == entry.one,
            "typed cosine produced the wrong exact value at zero");
  }
}

void checkDeterministicElementaryMathProvider() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::FunctionType unaryF32 = mlir::FunctionType::get(&context, {f32}, {f32});
  const struct MathCase {
    OperationSchemaId schema;
    std::uint32_t input;
    std::uint32_t expected;
  } cases[] = {
      {OperationSchemaId::MathSqrt, 0x40000000U, 0x3fb504f3U},
      {OperationSchemaId::MathExp, 0x3f800000U, 0x402df854U},
      {OperationSchemaId::MathLog, 0x40000000U, 0x3f317218U},
  };

  for (const MathCase &entry : cases) {
    require(__func__, loom::sim::isSupportedPrimitiveOperation(entry.schema),
            "registered elementary math has no deterministic provider");
    PrimitiveOperationDescriptor operation = descriptor(
        entry.schema, unaryF32, dataflow::FloatingPointPayload{}, 32, 32);
    PrimitiveValue result =
        takeValue(__func__, loom::sim::evaluatePrimitiveOperation(
                                operation, {floating32(entry.input)}));
    require(__func__,
            definedBits(__func__, result) == llvm::APInt(32, entry.expected),
            "elementary math produced the wrong IEEE result");
  }
}

} // namespace

int main() {
  checkTypedDispatchAndPayload();
  checkTypedLLVMExceptionalPolicies();
  checkTypedIntegerPolicies();
  checkSaturatingIntegerFamily();
  checkSaturatingFloatToInteger();
  checkLazySelectionAndFusedFma();
  checkTypedProviderAvailability();
  checkDeterministicCosineProvider();
  checkDeterministicElementaryMathProvider();
  return EXIT_SUCCESS;
}
