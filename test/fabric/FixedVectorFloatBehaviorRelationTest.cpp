#include "ImplementationFamilyBehaviorInternal.h"
#include "ImplementationFamilyVectorFloatBehavior.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;
using ::mlir::arith::CmpFPredicate;
using ::mlir::arith::FastMathFlags;
using ::mlir::arith::RoundingMode;

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
    fail(test, "expected relation rejection");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(fragment),
          "unexpected rejection: " + message);
}

FloatBehaviorProfile
profile(std::initializer_list<RoundingMode> roundingModes,
        std::initializer_list<FloatNaNBehavior> nanBehaviors,
        FloatSignedZeroBehavior signedZero = FloatSignedZeroBehavior::Preserve,
        FastMathFlags required = FastMathFlags::none) {
  return {RoundingModeSet::get(roundingModes),
          FloatNaNBehaviorSet::get(nanBehaviors),
          FloatSubnormalBehaviorSet::get({FloatSubnormalBehavior::Preserve}),
          FloatSignedZeroBehaviorSet::get({signedZero}), required};
}

std::vector<FiniteImplementationFamilyBehaviorPoint>
resolve(const char *test, ImplementationFamilyId family,
        const FamilyCapabilityParams &params,
        llvm::ArrayRef<OperationSchemaId> schemas,
        llvm::ArrayRef<std::uint32_t> inputs,
        llvm::ArrayRef<std::uint32_t> results, mlir::MLIRContext &context) {
  return take(test, detail::resolveFixedVectorFloatBehaviorDomain(
                        family, params, schemas, inputs, results, context));
}

bool isComparison(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithCmpF;
}

unsigned inputCount(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    return 1;
  case OperationSchemaId::MathFma:
    return 3;
  default:
    return 2;
  }
}

dataflow::CanonicalActorSchemaProjection
makeActor(mlir::MLIRContext &context, OperationSchemaId schema,
          mlir::Type element, llvm::ArrayRef<std::int64_t> shape,
          FastMathFlags flags,
          std::optional<RoundingMode> rounding = std::nullopt,
          std::optional<CmpFPredicate> predicate = std::nullopt) {
  mlir::Type values = mlir::VectorType::get(shape, element);
  std::vector<mlir::Type> inputs(inputCount(schema), values);
  mlir::Type result = values;
  dataflow::SemanticPayload payload =
      dataflow::FloatingPointPayload{flags, rounding};
  if (isComparison(schema)) {
    result = mlir::VectorType::get(shape, mlir::IntegerType::get(&context, 1));
    payload = dataflow::FloatComparePayload{
        predicate.value_or(CmpFPredicate::AlwaysFalse), flags};
  }
  return {schema, mlir::FunctionType::get(&context, inputs, {result}),
          std::move(payload)};
}

std::vector<std::vector<std::uint8_t>>
keys(llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  std::vector<std::vector<std::uint8_t>> result;
  for (const auto &point : domain) {
    if (!point.semanticConfiguration)
      continue;
    result.emplace_back(point.semanticConfiguration->bytes().begin(),
                        point.semanticConfiguration->bytes().end());
  }
  return result;
}

void requireCanonicalDomain(
    const char *test,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  const auto encoded = keys(domain);
  require(test, llvm::is_sorted(encoded),
          "behavior keys are not canonically ordered");
  require(test,
          std::adjacent_find(encoded.begin(), encoded.end()) == encoded.end(),
          "behavior keys are not unique");
  for (const auto &point : domain) {
    require(test,
            point.operandPorts.size() ==
                point.representativeActor.type.getNumInputs(),
            "operand correspondence has the wrong arity");
    require(test,
            point.resultPorts.size() ==
                point.representativeActor.type.getNumResults(),
            "result correspondence has the wrong arity");
    for (auto [ordinal, port] : llvm::enumerate(point.operandPorts))
      require(test, ordinal == port,
              "operand correspondence is not the identity image");
    for (auto [ordinal, port] : llvm::enumerate(point.resultPorts))
      require(test, ordinal == port,
              "result correspondence is not the identity image");
  }
}

std::uint32_t
elementWidth(const FiniteImplementationFamilyBehaviorPoint &point) {
  const auto vector =
      llvm::cast<mlir::VectorType>(point.representativeActor.type.getInput(0));
  return vector.getElementTypeBitWidth();
}

std::int64_t laneCount(const FiniteImplementationFamilyBehaviorPoint &point) {
  const auto vector =
      llvm::cast<mlir::VectorType>(point.representativeActor.type.getInput(0));
  return vector.getNumElements();
}

const FiniteImplementationFamilyBehaviorPoint &
findPoint(const char *test,
          llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
          OperationSchemaId schema, mlir::Type element,
          std::optional<CmpFPredicate> predicate = std::nullopt,
          std::optional<RoundingMode> rounding = std::nullopt) {
  auto found = llvm::find_if(domain, [&](const auto &point) {
    if (point.representativeActor.schema != schema)
      return false;
    const auto vector = llvm::cast<mlir::VectorType>(
        point.representativeActor.type.getInput(0));
    if (vector.getElementType() != element)
      return false;
    if (predicate) {
      const auto *payload = std::get_if<dataflow::FloatComparePayload>(
          &point.representativeActor.payload);
      if (!payload || payload->predicate != *predicate)
        return false;
    }
    if (rounding) {
      const auto *payload = std::get_if<dataflow::FloatingPointPayload>(
          &point.representativeActor.payload);
      if (!payload || payload->roundingMode.value_or(
                          RoundingMode::to_nearest_even) != *rounding)
        return false;
    }
    return true;
  });
  if (found == domain.end())
    fail(test, "expected behavior point is absent");
  return *found;
}

void ownsOnlyTheRegisteredFixedVectorFloatFamilies() {
  const char *test = __func__;
  constexpr std::array owned = {
      ImplementationFamilyId::FixedVectorFloatSign,
      ImplementationFamilyId::FixedVectorFloatAddSub,
      ImplementationFamilyId::FixedVectorFloatCompareMinMax,
      ImplementationFamilyId::FixedVectorFloatMultiply,
      ImplementationFamilyId::FixedVectorFloatFma};
  for (ImplementationFamilyId family : owned)
    require(test, detail::ownsFixedVectorFloatBehaviorRelation(family),
            implementationFamilyKeyword(family).str() + " has no owner");
  require(test,
          !detail::ownsFixedVectorFloatBehaviorRelation(
              ImplementationFamilyId::FixedVectorIntegerAddSub),
          "the floating owner claimed a foreign family");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      FixedVectorFloatParams{FloatFormatSet::get({FloatFormat::F16}),
                             FloatBehaviorProfile::strictIEEE(), 64};
  constexpr std::array inputs = {64U};
  constexpr std::array results = {64U};
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatSign, params, {},
                  inputs, results, context),
              "no enabled schema");
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatSign, params,
                  std::array{OperationSchemaId::ArithAddF}, inputs, results,
                  context),
              "foreign schema");
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatSign, params,
                  std::array{OperationSchemaId::ArithNegF,
                             OperationSchemaId::ArithNegF},
                  inputs, results, context),
              "schema twice");
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatSign,
                  FamilyCapabilityParams{FixedVectorFloatCompareMinMaxParams{
                      FloatFormatSet::get({FloatFormat::F16}),
                      FloatBehaviorProfile::strictIEEE(),
                      FloatPredicateSet::get({CmpFPredicate::OEQ}), 64}},
                  std::array{OperationSchemaId::ArithNegF}, inputs, results,
                  context),
              "wrong parameter schema");

  const FamilyCapabilityParams incompleteCompare =
      FixedVectorFloatCompareMinMaxParams{
          FloatFormatSet::get({FloatFormat::F16}),
          FloatBehaviorProfile::strictIEEE(),
          FloatPredicateSet::get({CmpFPredicate::OEQ}), 64};
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                  incompleteCompare,
                  std::array{OperationSchemaId::ArithCmpF,
                             OperationSchemaId::ArithMinimumF},
                  std::array{64U, 64U}, results, context),
              "predicate");
}

void physicalReachabilityPrecedesTheQuotient() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams sign = FixedVectorFloatParams{
      FloatFormatSet::get(
          {FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32}),
      FloatBehaviorProfile::strictIEEE(), 128};
  constexpr std::array signSchemas = {OperationSchemaId::ArithNegF,
                                      OperationSchemaId::MathAbsF};
  constexpr std::array narrow = {16U};
  const auto signDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatSign, sign,
              signSchemas, narrow, narrow, context);
  require(test, signDomain.size() == 2,
          "equal-width sign behaviors did not collapse by role");
  requireCanonicalDomain(test, signDomain);
  for (const auto &point : signDomain) {
    require(test, elementWidth(point) == 16 && laneCount(point) == 1,
            "unreachable format or lane count survived physical filtering");
  }
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatSign, sign,
                  signSchemas, std::array{8U}, std::array{8U}, context),
              "no reachable behavior");

  const FamilyCapabilityParams compare = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}),
      FloatBehaviorProfile::strictIEEE(),
      FloatPredicateSet::get({CmpFPredicate::OEQ}), 128};
  constexpr std::array compareSchemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array compareInputs = {64U, 64U};
  constexpr std::array compareResults = {2U};
  const auto compareDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatCompareMinMax,
              compare, compareSchemas, compareInputs, compareResults, context);
  require(test,
          compareDomain.size() == 1 && laneCount(compareDomain.front()) == 2,
          "comparison result width did not bound the representative lanes");

  const FamilyCapabilityParams compareAndMinimum =
      FixedVectorFloatCompareMinMaxParams{
          FloatFormatSet::get({FloatFormat::F16}),
          FloatBehaviorProfile::strictIEEE(),
          FloatPredicateSet::get({CmpFPredicate::OEQ, CmpFPredicate::OLT}),
          128};
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                  compareAndMinimum,
                  std::array{OperationSchemaId::ArithCmpF,
                             OperationSchemaId::ArithMinimumF},
                  compareInputs, std::array{1U}, context),
              "enabled schema");
}

void canonicalKeysPreserveFormatRoleAndRounding() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto behavior =
      profile({RoundingMode::to_nearest_even, RoundingMode::upward},
              {FloatNaNBehavior::IEEE});
  const FamilyCapabilityParams params = FixedVectorFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}), behavior, 128};
  constexpr std::array schemas = {OperationSchemaId::ArithAddF,
                                  OperationSchemaId::ArithSubF};
  constexpr std::array inputs = {128U, 128U};
  constexpr std::array results = {128U};
  const auto domain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatAddSub, params,
              schemas, inputs, results, context);
  require(test, domain.size() == 8,
          "add/sub format and rounding product was not the reachable image");
  requireCanonicalDomain(test, domain);

  mlir::Type f16 = mlir::Float16Type::get(&context);
  const auto &point = findPoint(test, domain, OperationSchemaId::ArithAddF, f16,
                                std::nullopt, RoundingMode::to_nearest_even);
  require(test, point.semanticConfiguration.has_value(),
          "multi-point relation omitted its key");
  const auto format = take(test, dataflow::encodeCanonicalType(f16));
  const auto rounding =
      take(test, dataflow::encodeRoundingMode(RoundingMode::to_nearest_even));
  const auto expected = take(
      test, detail::encodeImplementationFamilyBehaviorKey(
                ImplementationFamilyId::FixedVectorFloatAddSub, "Add",
                std::array<detail::ImplementationFamilyBehaviorKeyComponent, 2>{
                    format, rounding}));
  require(test, point.semanticConfiguration->bytes().equals(expected.bytes()),
          "add key escaped the canonical format/rounding component order");

  const auto absent =
      makeActor(context, OperationSchemaId::ArithAddF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::none);
  const auto explicitNearest =
      makeActor(context, OperationSchemaId::ArithAddF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::none,
                RoundingMode::to_nearest_even);
  const auto absentKey =
      take(test,
           detail::projectFixedVectorFloatBehavior(
               ImplementationFamilyId::FixedVectorFloatAddSub, absent, domain));
  const auto explicitKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatAddSub,
                     explicitNearest, domain));
  require(test,
          absentKey.bytes().equals(explicitKey.bytes()) &&
              explicitKey.bytes().equals(expected.bytes()),
          "absent rounding did not canonicalize to nearest-even");
}

void refinementCoverIsCapabilityGlobalAndDeterministic() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Type f16 = mlir::Float16Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);
  constexpr std::array inputs = {128U, 128U};
  constexpr std::array results = {128U};

  const FamilyCapabilityParams mixed = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      FloatBehaviorProfile::strictIEEE(),
      FloatPredicateSet::get({CmpFPredicate::OEQ, CmpFPredicate::UEQ}), 128};
  constexpr std::array compareSchema = {OperationSchemaId::ArithCmpF};
  const auto mixedDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatCompareMinMax,
              mixed, compareSchema, inputs, results, context);
  require(test, mixedDomain.size() == 4,
          "relaxed comparisons added modes beside strict representatives");
  requireCanonicalDomain(test, mixedDomain);
  const auto relaxedUeq =
      makeActor(context, OperationSchemaId::ArithCmpF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::nnan,
                std::nullopt, CmpFPredicate::UEQ);
  const auto strictUeq =
      makeActor(context, OperationSchemaId::ArithCmpF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::none,
                std::nullopt, CmpFPredicate::UEQ);
  const auto strictOeq =
      makeActor(context, OperationSchemaId::ArithCmpF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::none,
                std::nullopt, CmpFPredicate::OEQ);
  const auto relaxedKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     relaxedUeq, mixedDomain));
  const auto strictSameKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     strictUeq, mixedDomain));
  const auto strictNormalizedKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     strictOeq, mixedDomain));
  require(test, relaxedKey.bytes().equals(strictSameKey.bytes()),
          "relaxed comparison did not prefer its strict role");
  require(test, !relaxedKey.bytes().equals(strictNormalizedKey.bytes()),
          "same-role refinement preference was lost");

  const auto allRelaxedBehavior = profile({RoundingMode::to_nearest_even},
                                          {FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams allRelaxed = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      allRelaxedBehavior,
      FloatPredicateSet::get({CmpFPredicate::UEQ, CmpFPredicate::UGT}), 128};
  const auto relaxedDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatCompareMinMax,
              allRelaxed, compareSchema, inputs, results, context);
  require(test, relaxedDomain.size() == 2,
          "all-nnan equal-width comparisons retained exact formats");
  requireCanonicalDomain(test, relaxedDomain);
  const auto f16Ueq =
      makeActor(context, OperationSchemaId::ArithCmpF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::nnan,
                std::nullopt, CmpFPredicate::UEQ);
  const auto bf16Ueq =
      makeActor(context, OperationSchemaId::ArithCmpF, bf16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::nnan,
                std::nullopt, CmpFPredicate::UEQ);
  const auto f16Key =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     f16Ueq, relaxedDomain));
  const auto bf16Key =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     bf16Ueq, relaxedDomain));
  require(test, f16Key.bytes().equals(bf16Key.bytes()),
          "all-nnan equal-width formats did not share one width behavior");

  const FamilyCapabilityParams minNumber = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}),
      FloatBehaviorProfile::strictIEEE(),
      FloatPredicateSet::get({CmpFPredicate::OLT}), 128};
  constexpr std::array minSchemas = {OperationSchemaId::ArithMinimumF,
                                     OperationSchemaId::ArithMinNumF};
  const auto minDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatCompareMinMax,
              minNumber, minSchemas, inputs, results, context);
  require(test, minDomain.size() == 2,
          "relaxed minnum did not reuse strict minimum by exact format");
  const auto minimum =
      makeActor(context, OperationSchemaId::ArithMinimumF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::none);
  const auto relaxedMinNum =
      makeActor(context, OperationSchemaId::ArithMinNumF, f16,
                std::array<std::int64_t, 1>{2}, FastMathFlags::nnan);
  const auto minimumKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     minimum, minDomain));
  const auto minNumKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                     relaxedMinNum, minDomain));
  require(test, minimumKey.bytes().equals(minNumKey.bytes()),
          "uncovered minnum did not reuse compatible minimum behavior");

  const auto numberPreferred = profile({RoundingMode::to_nearest_even},
                                       {FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams reverseMin = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), numberPreferred,
      FloatPredicateSet::get({CmpFPredicate::OLT}), 128};
  const auto reverseMinDomain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatCompareMinMax,
              reverseMin, minSchemas, inputs, results, context);
  require(test,
          reverseMinDomain.size() == 1 &&
              !reverseMinDomain.front().semanticConfiguration,
          "strict minnum did not cover relaxed minimum");
}

void contextualProfilesRejectOrphanMembers() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  constexpr std::array unary = {128U};
  constexpr std::array binary = {128U, 128U};
  constexpr std::array results = {128U};
  const auto multipleRounding =
      profile({RoundingMode::to_nearest_even, RoundingMode::upward},
              {FloatNaNBehavior::IEEE});
  expectError(
      test,
      detail::resolveFixedVectorFloatBehaviorDomain(
          ImplementationFamilyId::FixedVectorFloatSign,
          FamilyCapabilityParams{FixedVectorFloatParams{
              FloatFormatSet::get({FloatFormat::F16}), multipleRounding, 128}},
          std::array{OperationSchemaId::ArithNegF}, unary, results, context),
      "rounding");

  const auto multipleNaN =
      profile({RoundingMode::to_nearest_even},
              {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  expectError(
      test,
      detail::resolveFixedVectorFloatBehaviorDomain(
          ImplementationFamilyId::FixedVectorFloatAddSub,
          FamilyCapabilityParams{FixedVectorFloatParams{
              FloatFormatSet::get({FloatFormat::F16}), multipleNaN, 128}},
          std::array{OperationSchemaId::ArithAddF}, binary, results, context),
      "NaN");
  const FamilyCapabilityParams compare = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), multipleNaN,
      FloatPredicateSet::get({CmpFPredicate::OLT}), 128};
  expectError(test,
              detail::resolveFixedVectorFloatBehaviorDomain(
                  ImplementationFamilyId::FixedVectorFloatCompareMinMax,
                  compare, std::array{OperationSchemaId::ArithMinimumF}, binary,
                  results, context),
              "NaN");
  const auto closed = resolve(
      test, ImplementationFamilyId::FixedVectorFloatCompareMinMax, compare,
      std::array{OperationSchemaId::ArithMinimumF,
                 OperationSchemaId::ArithMinNumF},
      binary, results, context);
  require(test, closed.size() == 2,
          "contextually complete NaN profile lost a strict role");
}

void allFamiliesOwnStableDomainsAndSingletonsCollapse() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto nanRoles =
      profile({RoundingMode::to_nearest_even},
              {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams ordinary =
      FixedVectorFloatParams{FloatFormatSet::get({FloatFormat::F16}),
                             FloatBehaviorProfile::strictIEEE(), 128};
  const FamilyCapabilityParams compare = FixedVectorFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), nanRoles,
      FloatPredicateSet::get(
          {CmpFPredicate::OEQ, CmpFPredicate::OLT, CmpFPredicate::OGT}),
      128};
  constexpr std::array unary = {128U};
  constexpr std::array binary = {128U, 128U};
  constexpr std::array ternary = {128U, 128U, 128U};
  constexpr std::array results = {128U};
  struct Case final {
    ImplementationFamilyId family;
    const FamilyCapabilityParams *params;
    llvm::ArrayRef<OperationSchemaId> schemas;
    llvm::ArrayRef<std::uint32_t> inputs;
    std::size_t count;
  };
  constexpr std::array signSchemas = {OperationSchemaId::ArithNegF,
                                      OperationSchemaId::MathAbsF};
  constexpr std::array addSchemas = {OperationSchemaId::ArithAddF,
                                     OperationSchemaId::ArithSubF};
  constexpr std::array compareSchemas = {
      OperationSchemaId::ArithCmpF, OperationSchemaId::ArithMinimumF,
      OperationSchemaId::ArithMaximumF, OperationSchemaId::ArithMinNumF,
      OperationSchemaId::ArithMaxNumF};
  constexpr std::array multiplySchema = {OperationSchemaId::ArithMulF};
  constexpr std::array fmaSchema = {OperationSchemaId::MathFma};
  const std::array cases = {
      Case{ImplementationFamilyId::FixedVectorFloatSign, &ordinary, signSchemas,
           unary, 2},
      Case{ImplementationFamilyId::FixedVectorFloatAddSub, &ordinary,
           addSchemas, binary, 2},
      Case{ImplementationFamilyId::FixedVectorFloatCompareMinMax, &compare,
           compareSchemas, binary, 7},
      Case{ImplementationFamilyId::FixedVectorFloatMultiply, &ordinary,
           multiplySchema, binary, 1},
      Case{ImplementationFamilyId::FixedVectorFloatFma, &ordinary, fmaSchema,
           ternary, 1}};
  for (const Case &entry : cases) {
    const auto domain = resolve(test, entry.family, *entry.params,
                                entry.schemas, entry.inputs, results, context);
    require(test, domain.size() == entry.count,
            implementationFamilyKeyword(entry.family).str() +
                " exposed the wrong behavior image");
    requireCanonicalDomain(test, domain);
    if (entry.count == 1)
      require(test, !domain.front().semanticConfiguration,
              implementationFamilyKeyword(entry.family).str() +
                  " retained a singleton configuration key");
  }

  const FamilyCapabilityParams sameWidthSign = FixedVectorFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      FloatBehaviorProfile::strictIEEE(), 128};
  const auto singleton = resolve(
      test, ImplementationFamilyId::FixedVectorFloatSign, sameWidthSign,
      std::array{OperationSchemaId::ArithNegF}, unary, results, context);
  require(test,
          singleton.size() == 1 && !singleton.front().semanticConfiguration,
          "same-width sign aliases did not collapse to one witness");
}

void projectionValidatesTheClosedImageAndIgnoresShape() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto behavior =
      profile({RoundingMode::to_nearest_even, RoundingMode::upward},
              {FloatNaNBehavior::IEEE});
  const FamilyCapabilityParams params = FixedVectorFloatParams{
      FloatFormatSet::get({FloatFormat::F16}), behavior, 128};
  constexpr std::array schemas = {OperationSchemaId::ArithAddF};
  constexpr std::array inputs = {128U, 128U};
  constexpr std::array results = {128U};
  const auto domain =
      resolve(test, ImplementationFamilyId::FixedVectorFloatAddSub, params,
              schemas, inputs, results, context);
  mlir::Type f16 = mlir::Float16Type::get(&context);
  const auto rankOne = makeActor(context, OperationSchemaId::ArithAddF, f16,
                                 std::array<std::int64_t, 1>{4},
                                 FastMathFlags::none, RoundingMode::upward);
  const auto rankTwo = makeActor(context, OperationSchemaId::ArithAddF, f16,
                                 std::array<std::int64_t, 2>{2, 2},
                                 FastMathFlags::none, RoundingMode::upward);
  const auto rankOneKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatAddSub, rankOne,
                     domain));
  const auto rankTwoKey =
      take(test, detail::projectFixedVectorFloatBehavior(
                     ImplementationFamilyId::FixedVectorFloatAddSub, rankTwo,
                     domain));
  require(test, rankOneKey.bytes().equals(rankTwoKey.bytes()),
          "incidental fixed-vector shape changed configured behavior");

  expectError(test,
              detail::projectFixedVectorFloatBehavior(
                  ImplementationFamilyId::FixedVectorFloatAddSub,
                  makeActor(context, OperationSchemaId::ArithSubF, f16,
                            std::array<std::int64_t, 1>{4}, FastMathFlags::none,
                            RoundingMode::upward),
                  domain),
              "outside");
  expectError(test,
              detail::projectFixedVectorFloatBehavior(
                  ImplementationFamilyId::FixedVectorFloatAddSub,
                  makeActor(context, OperationSchemaId::ArithAddF,
                            mlir::Float32Type::get(&context),
                            std::array<std::int64_t, 1>{2}, FastMathFlags::none,
                            RoundingMode::upward),
                  domain),
              "outside");

  const auto multiplyDomain = resolve(
      test, ImplementationFamilyId::FixedVectorFloatMultiply, params,
      std::array{OperationSchemaId::ArithMulF}, inputs, results, context);
  expectError(test,
              detail::projectFixedVectorFloatBehavior(
                  ImplementationFamilyId::FixedVectorFloatMultiply,
                  makeActor(context, OperationSchemaId::MathFma, f16,
                            std::array<std::int64_t, 1>{4}, FastMathFlags::none,
                            RoundingMode::upward),
                  multiplyDomain),
              "schema");

  auto malformed = rankOne;
  malformed.payload = dataflow::NoPayload{};
  expectError(
      test,
      detail::projectFixedVectorFloatBehavior(
          ImplementationFamilyId::FixedVectorFloatAddSub, malformed, domain),
      "floating payload");
}

} // namespace

int main() {
  ownsOnlyTheRegisteredFixedVectorFloatFamilies();
  physicalReachabilityPrecedesTheQuotient();
  canonicalKeysPreserveFormatRoleAndRounding();
  refinementCoverIsCapabilityGlobalAndDeterministic();
  contextualProfilesRejectOrphanMembers();
  allFamiliesOwnStableDomainsAndSingletonsCollapse();
  projectionValidatesTheClosedImageAndIgnoresShape();
  return EXIT_SUCCESS;
}
