//===- ImplementationFamilyScalarFloatBehaviorTest.cpp ------------------===//

#include "ImplementationFamilyScalarFloatBehavior.h"
#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace fabric;
using dataflow::CanonicalActorSchemaProjection;
using dataflow::OperationSchemaId;
using detail::ImplementationFamilyBehaviorKeyComponent;

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(const char *test, llvm::Expected<T> value,
                 llvm::StringRef fragment) {
  if (value)
    fail(test, "invalid scalar floating relation was accepted");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(fragment),
          "unexpected rejection: " + message);
}

mlir::Type floatType(mlir::MLIRContext &context, FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
    return mlir::Float16Type::get(&context);
  case FloatFormat::BF16:
    return mlir::BFloat16Type::get(&context);
  case FloatFormat::F32:
    return mlir::Float32Type::get(&context);
  case FloatFormat::F64:
    return mlir::Float64Type::get(&context);
  }
  llvm_unreachable("unknown floating format");
}

mlir::Type integerType(mlir::MLIRContext &context, IntegerWidth width) {
  return mlir::IntegerType::get(&context, getBitWidth(width));
}

FloatBehaviorProfile behaviorWith(RoundingModeSet roundingModes) {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.roundingModes = roundingModes;
  return behavior;
}

CanonicalActorSchemaProjection uniformFloatActor(
    mlir::MLIRContext &context, OperationSchemaId schema, FloatFormat format,
    std::optional<mlir::arith::RoundingMode> roundingMode = std::nullopt,
    mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none) {
  unsigned inputCount = 0;
  switch (schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    inputCount = 1;
    break;
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
  case OperationSchemaId::ArithDivF:
  case OperationSchemaId::ArithRemF:
    inputCount = 2;
    break;
  case OperationSchemaId::MathFma:
    inputCount = 3;
    break;
  default:
    llvm_unreachable("schema is not a uniform scalar floating actor");
  }
  mlir::Type type = floatType(context, format);
  return {schema,
          mlir::FunctionType::get(
              &context, std::vector<mlir::Type>(inputCount, type), {type}),
          dataflow::FloatingPointPayload{flags, roundingMode}};
}

CanonicalActorSchemaProjection floatCastActor(
    mlir::MLIRContext &context, OperationSchemaId schema, FloatFormat source,
    FloatFormat destination,
    std::optional<mlir::arith::RoundingMode> roundingMode = std::nullopt) {
  return {schema,
          mlir::FunctionType::get(&context, {floatType(context, source)},
                                  {floatType(context, destination)}),
          dataflow::FloatingPointPayload{mlir::arith::FastMathFlags::none,
                                         roundingMode}};
}

CanonicalActorSchemaProjection conversionActor(mlir::MLIRContext &context,
                                               OperationSchemaId schema,
                                               IntegerWidth integer,
                                               FloatFormat format,
                                               bool nonNegative = false) {
  mlir::Type integerEndpoint = integerType(context, integer);
  mlir::Type floatEndpoint = floatType(context, format);
  dataflow::SemanticPayload payload = dataflow::NoPayload{};
  mlir::Type input = integerEndpoint;
  mlir::Type result = floatEndpoint;
  switch (schema) {
  case OperationSchemaId::ArithSIToFP:
    break;
  case OperationSchemaId::ArithUIToFP:
    payload = dataflow::NonNegativePayload{nonNegative};
    break;
  case OperationSchemaId::ArithFPToSI:
  case OperationSchemaId::ArithFPToUI:
  case OperationSchemaId::LLVMFPToSISat:
  case OperationSchemaId::LLVMFPToUISat:
    input = floatEndpoint;
    result = integerEndpoint;
    break;
  default:
    llvm_unreachable("schema is not a scalar floating conversion");
  }
  return {schema, mlir::FunctionType::get(&context, {input}, {result}),
          std::move(payload)};
}

::loom::CanonicalSemanticBytes
expectedKey(const char *test, ImplementationFamilyId family,
            llvm::StringRef role,
            std::vector<ImplementationFamilyBehaviorKeyComponent> components) {
  return take(test, detail::encodeImplementationFamilyBehaviorKey(family, role,
                                                                  components));
}

::loom::CanonicalSemanticBytes expectedFormat(const char *test,
                                              mlir::MLIRContext &context,
                                              FloatFormat format) {
  return take(test, dataflow::encodeCanonicalType(floatType(context, format)));
}

::loom::CanonicalSemanticBytes
expectedRounding(const char *test, mlir::arith::RoundingMode mode) {
  return take(test, dataflow::encodeRoundingMode(mode));
}

bool sameBytes(const ::loom::CanonicalSemanticBytes &lhs,
               const ::loom::CanonicalSemanticBytes &rhs) {
  return lhs.bytes().equals(rhs.bytes());
}

std::vector<FiniteImplementationFamilyBehaviorPoint>
resolve(const char *test, ImplementationFamilyId family,
        const FamilyCapabilityParams &params,
        llvm::ArrayRef<OperationSchemaId> schemas,
        llvm::ArrayRef<std::uint32_t> inputWidths,
        llvm::ArrayRef<std::uint32_t> resultWidths,
        mlir::MLIRContext &context) {
  auto points = take(
      test, detail::resolveScalarFloatBehaviorDomain(
                family, params, schemas, inputWidths, resultWidths, context));
  for (const auto &point : points) {
    require(test,
            point.operandPorts.size() ==
                    point.representativeActor.type.getNumInputs() &&
                point.resultPorts.size() ==
                    point.representativeActor.type.getNumResults(),
            "floating behavior witness has the wrong correspondence arity");
    for (auto [ordinal, port] : llvm::enumerate(point.operandPorts))
      require(test, port == ordinal,
              "floating behavior witness has a non-identity input image");
    for (auto [ordinal, port] : llvm::enumerate(point.resultPorts))
      require(test, port == ordinal,
              "floating behavior witness has a non-identity result image");
  }
  if (points.size() == 1) {
    require(test, !points.front().semanticConfiguration,
            "singleton floating quotient did not collapse to None");
  } else {
    for (const auto &point : points)
      require(test, point.semanticConfiguration.has_value(),
              "non-singleton floating quotient omitted a key");
    for (std::size_t ordinal = 1; ordinal != points.size(); ++ordinal) {
      llvm::ArrayRef<std::uint8_t> previous =
          points[ordinal - 1].semanticConfiguration->bytes();
      llvm::ArrayRef<std::uint8_t> current =
          points[ordinal].semanticConfiguration->bytes();
      require(test,
              std::lexicographical_compare(previous.begin(), previous.end(),
                                           current.begin(), current.end()),
              "floating behavior keys are not strictly sorted and unique");
    }
  }
  return points;
}

void expectProjection(
    const char *test, ImplementationFamilyId family,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
    const CanonicalActorSchemaProjection &actor,
    const ::loom::CanonicalSemanticBytes &expected) {
  auto actual =
      take(test, detail::projectScalarFloatBehavior(family, actor, domain));
  require(test, sameBytes(actual, expected),
          "floating actor projected to the wrong behavior key");
}

const FiniteImplementationFamilyBehaviorPoint &
findPoint(const char *test,
          llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
          const ::loom::CanonicalSemanticBytes &key) {
  auto found = llvm::find_if(domain, [&](const auto &point) {
    return point.semanticConfiguration &&
           point.semanticConfiguration->bytes().equals(key.bytes());
  });
  if (found == domain.end())
    fail(test, "expected floating behavior key is absent");
  return *found;
}

void ownershipIsClosed() {
  const char *test = __func__;
  constexpr std::array families = {
      ImplementationFamilyId::ScalarFloatSign,
      ImplementationFamilyId::ScalarFloatAddSub,
      ImplementationFamilyId::ScalarFloatWidthCast,
      ImplementationFamilyId::ScalarIntegerToFloat,
      ImplementationFamilyId::ScalarFloatToInteger,
      ImplementationFamilyId::ScalarFloatMultiply,
      ImplementationFamilyId::ScalarFloatFma,
      ImplementationFamilyId::ScalarFloatDivide,
      ImplementationFamilyId::ScalarFloatRemainder,
  };
  for (ImplementationFamilyId family : families)
    require(test, detail::ownsScalarFloatBehaviorRelation(family),
            "scalar floating owner rejected one of its families");
  require(test,
          !detail::ownsScalarFloatBehaviorRelation(
              ImplementationFamilyId::ScalarFloatCompareMinMax) &&
              !detail::ownsScalarFloatBehaviorRelation(
                  ImplementationFamilyId::ScalarIntegerAddSub),
          "scalar floating owner claimed a foreign family");
}

void signCollapsesEquivalentRepresentationWidths() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatParams{
      FloatFormatSet::get(
          {FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32}),
      FloatBehaviorProfile::strictIEEE()};
  constexpr std::array schemas = {OperationSchemaId::MathAbsF,
                                  OperationSchemaId::ArithNegF};
  constexpr std::array inputs = {32U};
  constexpr std::array results = {32U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarFloatSign, params,
                        schemas, inputs, results, context);
  require(test, domain.size() == 4,
          "sign relation did not collapse f16 and bf16 by width");

  auto negate16 = expectedKey(test, ImplementationFamilyId::ScalarFloatSign,
                              "Negate", {std::uint32_t{16}});
  auto absolute16 = expectedKey(test, ImplementationFamilyId::ScalarFloatSign,
                                "Absolute", {std::uint32_t{16}});
  expectProjection(test, ImplementationFamilyId::ScalarFloatSign, domain,
                   uniformFloatActor(context, OperationSchemaId::ArithNegF,
                                     FloatFormat::F16),
                   negate16);
  expectProjection(test, ImplementationFamilyId::ScalarFloatSign, domain,
                   uniformFloatActor(context, OperationSchemaId::ArithNegF,
                                     FloatFormat::BF16),
                   negate16);
  expectProjection(test, ImplementationFamilyId::ScalarFloatSign, domain,
                   uniformFloatActor(context, OperationSchemaId::MathAbsF,
                                     FloatFormat::BF16),
                   absolute16);
}

void addSubUsesExactFormatsAndCanonicalRounding() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = behaviorWith(
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::downward}));
  const FamilyCapabilityParams params = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}), behavior};
  constexpr std::array schemas = {OperationSchemaId::ArithSubF,
                                  OperationSchemaId::ArithAddF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarFloatAddSub, params,
                        schemas, inputs, results, context);
  require(test, domain.size() == 8,
          "add/sub relation lost a role, format, or rounding behavior");

  auto f16 = expectedFormat(test, context, FloatFormat::F16);
  auto bf16 = expectedFormat(test, context, FloatFormat::BF16);
  auto nearest =
      expectedRounding(test, mlir::arith::RoundingMode::to_nearest_even);
  auto downward = expectedRounding(test, mlir::arith::RoundingMode::downward);
  auto f16Nearest =
      expectedKey(test, ImplementationFamilyId::ScalarFloatAddSub, "Add",
                  {ImplementationFamilyBehaviorKeyComponent{f16},
                   ImplementationFamilyBehaviorKeyComponent{nearest}});
  auto bf16Nearest =
      expectedKey(test, ImplementationFamilyId::ScalarFloatAddSub, "Add",
                  {ImplementationFamilyBehaviorKeyComponent{bf16},
                   ImplementationFamilyBehaviorKeyComponent{nearest}});
  auto f16Downward =
      expectedKey(test, ImplementationFamilyId::ScalarFloatAddSub, "Add",
                  {ImplementationFamilyBehaviorKeyComponent{f16},
                   ImplementationFamilyBehaviorKeyComponent{downward}});
  require(test, !sameBytes(f16Nearest, bf16Nearest),
          "equal-width floating formats collapsed in arithmetic");
  expectProjection(test, ImplementationFamilyId::ScalarFloatAddSub, domain,
                   uniformFloatActor(context, OperationSchemaId::ArithAddF,
                                     FloatFormat::F16),
                   f16Nearest);
  expectProjection(
      test, ImplementationFamilyId::ScalarFloatAddSub, domain,
      uniformFloatActor(context, OperationSchemaId::ArithAddF, FloatFormat::F16,
                        mlir::arith::RoundingMode::to_nearest_even),
      f16Nearest);
  expectProjection(test, ImplementationFamilyId::ScalarFloatAddSub, domain,
                   uniformFloatActor(context, OperationSchemaId::ArithAddF,
                                     FloatFormat::F16,
                                     mlir::arith::RoundingMode::downward),
                   f16Downward);
}

void widthCastUsesDirectedPairsAndTruncationRounding() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FloatFormatRelation pairs =
      FloatFormatRelation::get({{FloatFormat::F16, FloatFormat::F32},
                                {FloatFormat::F32, FloatFormat::F16}});
  FloatBehaviorProfile behavior = behaviorWith(
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::toward_zero}));
  const FamilyCapabilityParams params =
      ScalarFloatWidthCastParams{pairs, behavior};
  constexpr std::array schemas = {OperationSchemaId::ArithTruncF,
                                  OperationSchemaId::ArithExtF};
  constexpr std::array inputs = {32U};
  constexpr std::array results = {32U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarFloatWidthCast,
                        params, schemas, inputs, results, context);
  require(test, domain.size() == 3,
          "floating cast relation did not retain one extension and two "
          "truncation behaviors");

  auto f16 = expectedFormat(test, context, FloatFormat::F16);
  auto f32 = expectedFormat(test, context, FloatFormat::F32);
  auto nearest =
      expectedRounding(test, mlir::arith::RoundingMode::to_nearest_even);
  auto towardZero =
      expectedRounding(test, mlir::arith::RoundingMode::toward_zero);
  auto extension =
      expectedKey(test, ImplementationFamilyId::ScalarFloatWidthCast, "",
                  {ImplementationFamilyBehaviorKeyComponent{f16},
                   ImplementationFamilyBehaviorKeyComponent{f32}});
  auto truncNearest =
      expectedKey(test, ImplementationFamilyId::ScalarFloatWidthCast, "",
                  {ImplementationFamilyBehaviorKeyComponent{f32},
                   ImplementationFamilyBehaviorKeyComponent{f16},
                   ImplementationFamilyBehaviorKeyComponent{nearest}});
  auto truncTowardZero =
      expectedKey(test, ImplementationFamilyId::ScalarFloatWidthCast, "",
                  {ImplementationFamilyBehaviorKeyComponent{f32},
                   ImplementationFamilyBehaviorKeyComponent{f16},
                   ImplementationFamilyBehaviorKeyComponent{towardZero}});
  expectProjection(test, ImplementationFamilyId::ScalarFloatWidthCast, domain,
                   floatCastActor(context, OperationSchemaId::ArithExtF,
                                  FloatFormat::F16, FloatFormat::F32),
                   extension);
  expectProjection(test, ImplementationFamilyId::ScalarFloatWidthCast, domain,
                   floatCastActor(context, OperationSchemaId::ArithTruncF,
                                  FloatFormat::F32, FloatFormat::F16),
                   truncNearest);
  expectProjection(test, ImplementationFamilyId::ScalarFloatWidthCast, domain,
                   floatCastActor(context, OperationSchemaId::ArithTruncF,
                                  FloatFormat::F32, FloatFormat::F16,
                                  mlir::arith::RoundingMode::toward_zero),
                   truncTowardZero);
}

void integerToFloatReusesTheStrictUnsignedRepresentative() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerFloatConversionParams{
      IntegerFloatFormatRelation::get({{IntegerWidth::I16, FloatFormat::F32},
                                       {IntegerWidth::I32, FloatFormat::F64}})};
  constexpr std::array schemas = {OperationSchemaId::ArithUIToFP,
                                  OperationSchemaId::ArithSIToFP};
  constexpr std::array inputs = {32U};
  constexpr std::array results = {64U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarIntegerToFloat,
                        params, schemas, inputs, results, context);
  require(test, domain.size() == 4,
          "integer-to-float relation added or lost a conversion behavior");

  auto f32 = expectedFormat(test, context, FloatFormat::F32);
  auto unsignedI16F32 = expectedKey(
      test, ImplementationFamilyId::ScalarIntegerToFloat, "Unsigned",
      {std::uint32_t{16}, ImplementationFamilyBehaviorKeyComponent{f32}});
  auto strict = conversionActor(context, OperationSchemaId::ArithUIToFP,
                                IntegerWidth::I16, FloatFormat::F32, false);
  auto nonNegative = conversionActor(context, OperationSchemaId::ArithUIToFP,
                                     IntegerWidth::I16, FloatFormat::F32, true);
  expectProjection(test, ImplementationFamilyId::ScalarIntegerToFloat, domain,
                   strict, unsignedI16F32);
  expectProjection(test, ImplementationFamilyId::ScalarIntegerToFloat, domain,
                   nonNegative, unsignedI16F32);

  const auto &representative = findPoint(test, domain, unsignedI16F32);
  require(test,
          representative.representativeActor.schema ==
              OperationSchemaId::ArithUIToFP,
          "nneg conversion replaced its existing unsigned mode");
  const auto *payload = std::get_if<dataflow::NonNegativePayload>(
      &representative.representativeActor.payload);
  require(test, payload && !payload->isNonNegative,
          "nneg witness displaced the stronger unsigned representative");
}

void saturatingFloatToIntegerRefinesOrdinaryConversions() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerFloatConversionParams{
      IntegerFloatFormatRelation::get({{IntegerWidth::I16, FloatFormat::F32}})};
  constexpr std::array schemas = {
      OperationSchemaId::ArithFPToUI,
      OperationSchemaId::LLVMFPToSISat,
      OperationSchemaId::ArithFPToSI,
      OperationSchemaId::LLVMFPToUISat,
  };
  constexpr std::array inputs = {32U};
  constexpr std::array results = {16U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarFloatToInteger,
                        params, schemas, inputs, results, context);
  require(test, domain.size() == 2,
          "ordinary and saturating conversions created separate modes");

  auto signedKey = expectedKey(
      test, ImplementationFamilyId::ScalarFloatToInteger, "Signed", {});
  auto unsignedKey = expectedKey(
      test, ImplementationFamilyId::ScalarFloatToInteger, "Unsigned", {});
  const auto signedOrdinary =
      conversionActor(context, OperationSchemaId::ArithFPToSI,
                      IntegerWidth::I16, FloatFormat::F32);
  const auto signedSaturating =
      conversionActor(context, OperationSchemaId::LLVMFPToSISat,
                      IntegerWidth::I16, FloatFormat::F32);
  expectProjection(test, ImplementationFamilyId::ScalarFloatToInteger, domain,
                   signedOrdinary, signedKey);
  expectProjection(test, ImplementationFamilyId::ScalarFloatToInteger, domain,
                   signedSaturating, signedKey);
  require(test,
          findPoint(test, domain, signedKey).representativeActor.schema ==
              OperationSchemaId::LLVMFPToSISat,
          "ordinary conversion displaced the stronger saturating witness");
  require(test,
          findPoint(test, domain, unsignedKey).representativeActor.schema ==
              OperationSchemaId::LLVMFPToUISat,
          "ordinary unsigned conversion displaced the saturating witness");
}

void singleRoleArithmeticUsesAnEmptyRoleAndExactComponents() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = behaviorWith(
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::upward}));
  const FamilyCapabilityParams params = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}), behavior};
  auto f16 = expectedFormat(test, context, FloatFormat::F16);
  auto upward = expectedRounding(test, mlir::arith::RoundingMode::upward);
  constexpr std::array results = {16U};

  struct Case final {
    ImplementationFamilyId family;
    OperationSchemaId schema;
    unsigned inputCount;
  };
  constexpr std::array cases = {
      Case{ImplementationFamilyId::ScalarFloatMultiply,
           OperationSchemaId::ArithMulF, 2},
      Case{ImplementationFamilyId::ScalarFloatFma, OperationSchemaId::MathFma,
           3},
      Case{ImplementationFamilyId::ScalarFloatDivide,
           OperationSchemaId::ArithDivF, 2},
  };
  for (const Case &entry : cases) {
    const std::vector<std::uint32_t> inputs(entry.inputCount, 16);
    const std::array schemas = {entry.schema};
    auto domain =
        resolve(test, entry.family, params, schemas, inputs, results, context);
    require(test, domain.size() == 4,
            "single-role arithmetic lost format or rounding behavior");
    auto expected =
        expectedKey(test, entry.family, "",
                    {ImplementationFamilyBehaviorKeyComponent{f16},
                     ImplementationFamilyBehaviorKeyComponent{upward}});
    expectProjection(test, entry.family, domain,
                     uniformFloatActor(context, entry.schema, FloatFormat::F16,
                                       mlir::arith::RoundingMode::upward),
                     expected);
  }

  const FamilyCapabilityParams remainderParams = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      FloatBehaviorProfile::strictIEEE()};
  constexpr std::array remainderSchemas = {OperationSchemaId::ArithRemF};
  constexpr std::array remainderInputs = {16U, 16U};
  auto remainder = resolve(test, ImplementationFamilyId::ScalarFloatRemainder,
                           remainderParams, remainderSchemas, remainderInputs,
                           results, context);
  require(test, remainder.size() == 2,
          "floating remainder exposed a rounding mode");
  auto expected =
      expectedKey(test, ImplementationFamilyId::ScalarFloatRemainder, "",
                  {ImplementationFamilyBehaviorKeyComponent{f16}});
  expectProjection(test, ImplementationFamilyId::ScalarFloatRemainder,
                   remainder,
                   uniformFloatActor(context, OperationSchemaId::ArithRemF,
                                     FloatFormat::F16),
                   expected);
}

void invalidDomainsFailClosed() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  constexpr std::array signSchemas = {OperationSchemaId::ArithNegF};
  constexpr std::array unaryInputs = {32U};
  constexpr std::array results = {32U};

  FloatBehaviorProfile orphanRounding = behaviorWith(
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::downward}));
  const FamilyCapabilityParams signParams = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F32}), orphanRounding};
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatSign, signParams,
                  signSchemas, unaryInputs, results, context),
              "rounding");

  FloatBehaviorProfile orphanNaN = FloatBehaviorProfile::strictIEEE();
  orphanNaN.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams addParams =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), orphanNaN};
  constexpr std::array addSchemas = {OperationSchemaId::ArithAddF};
  constexpr std::array addInputs = {32U, 32U};
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatAddSub, addParams,
                  addSchemas, addInputs, results, context),
              "NaN");

  const FamilyCapabilityParams strictAdd =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}),
                        FloatBehaviorProfile::strictIEEE()};
  constexpr std::array narrowInputs = {31U, 31U};
  constexpr std::array narrowResults = {31U};
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatAddSub, strictAdd,
                  addSchemas, narrowInputs, narrowResults, context),
              "physically reachable");

  constexpr std::array duplicateSchemas = {OperationSchemaId::ArithAddF,
                                           OperationSchemaId::ArithAddF};
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatAddSub, strictAdd,
                  duplicateSchemas, addInputs, results, context),
              "twice");
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatAddSub, strictAdd,
                  signSchemas, addInputs, results, context),
              "foreign");

  const FamilyCapabilityParams castParams = ScalarFloatWidthCastParams{
      FloatFormatRelation::get({{FloatFormat::F16, FloatFormat::F32}}),
      FloatBehaviorProfile::strictIEEE()};
  constexpr std::array truncSchemas = {OperationSchemaId::ArithTruncF};
  expectError(test,
              detail::resolveScalarFloatBehaviorDomain(
                  ImplementationFamilyId::ScalarFloatWidthCast, castParams,
                  truncSchemas, unaryInputs, results, context),
              "no admitted behavior");
}

void singletonQuotientHasNoSemanticField() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}),
                        FloatBehaviorProfile::strictIEEE()};
  constexpr std::array schemas = {OperationSchemaId::MathFma};
  constexpr std::array inputs = {32U, 32U, 32U};
  constexpr std::array results = {32U};
  auto domain = resolve(test, ImplementationFamilyId::ScalarFloatFma, params,
                        schemas, inputs, results, context);
  require(test, domain.size() == 1 && !domain.front().semanticConfiguration,
          "hardwired scalar FMA retained a semantic field");
}

void publicRelationUsesTheSealedScalarFloatOwner() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = behaviorWith(
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::downward}));
  const FamilyCapabilityParams params = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}), behavior};
  constexpr std::array schemas = {OperationSchemaId::ArithSubF,
                                  OperationSchemaId::ArithAddF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto expected = resolve(test, ImplementationFamilyId::ScalarFloatAddSub,
                          params, schemas, inputs, results, context);
  auto relation = take(test, resolveFabricOpSemanticFieldRelation(
                                 ImplementationFamilyId::ScalarFloatAddSub,
                                 params, schemas, inputs, results, context));
  require(test,
          relation.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == expected.size(),
          "public relation did not select the scalar floating owner");
  for (const auto &expectedPoint : expected) {
    require(test, expectedPoint.semanticConfiguration.has_value(),
            "expected scalar floating point has no key");
    const bool present = llvm::any_of(
        relation.finiteBehaviorDomain(), [&](const auto &actualPoint) {
          return actualPoint.semanticConfiguration &&
                 actualPoint.semanticConfiguration->bytes().equals(
                     expectedPoint.semanticConfiguration->bytes());
        });
    require(test, present,
            "public relation changed a sealed scalar floating key");
  }

  auto actor =
      uniformFloatActor(context, OperationSchemaId::ArithAddF, FloatFormat::F16,
                        mlir::arith::RoundingMode::downward);
  auto expectedValue = take(
      test, detail::projectScalarFloatBehavior(
                ImplementationFamilyId::ScalarFloatAddSub, actor, expected));
  auto actualValue = take(test, relation.projectSemanticValue(
                                    actor, std::array<std::uint64_t, 2>{0, 1},
                                    std::array<std::uint64_t, 1>{0}));
  require(test, sameBytes(actualValue, expectedValue),
          "public scalar floating projection bypassed its sealed owner");
}

} // namespace

int main() {
  ownershipIsClosed();
  signCollapsesEquivalentRepresentationWidths();
  addSubUsesExactFormatsAndCanonicalRounding();
  widthCastUsesDirectedPairsAndTruncationRounding();
  integerToFloatReusesTheStrictUnsignedRepresentative();
  saturatingFloatToIntegerRefinesOrdinaryConversions();
  singleRoleArithmeticUsesAnEmptyRoleAndExactComponents();
  invalidDomainsFailClosed();
  singletonQuotientHasNoSemanticField();
  publicRelationUsesTheSealedScalarFloatOwner();
  return EXIT_SUCCESS;
}
