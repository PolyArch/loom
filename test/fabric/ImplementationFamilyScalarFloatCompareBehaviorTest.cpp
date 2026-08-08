//===- ImplementationFamilyScalarFloatCompareBehaviorTest.cpp -----------===//

#include "ImplementationFamilyScalarFloatCompareBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;

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
void expectRejected(const char *test, llvm::Expected<T> value,
                    llvm::StringRef fragment) {
  if (value)
    fail(test, "invalid floating compare relation was accepted");
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

::dataflow::CanonicalActorSchemaProjection
compareActor(mlir::MLIRContext &context, FloatFormat format,
             mlir::arith::CmpFPredicate predicate,
             mlir::arith::FastMathFlags flags) {
  mlir::Type operand = floatType(context, format);
  mlir::Type result = mlir::IntegerType::get(&context, 1);
  return {OperationSchemaId::ArithCmpF,
          mlir::FunctionType::get(&context, {operand, operand}, {result}),
          ::dataflow::FloatComparePayload{predicate, flags}};
}

FloatBehaviorProfile strictCompareBehavior() {
  return FloatBehaviorProfile::strictIEEE();
}

FloatBehaviorProfile requiredNnanBehavior(FloatNaNBehavior nanBehavior) {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.nanBehaviors = FloatNaNBehaviorSet::get({nanBehavior});
  behavior.requiredFastMath = mlir::arith::FastMathFlags::nnan;
  return behavior;
}

::dataflow::CanonicalActorSchemaProjection
minMaxActor(mlir::MLIRContext &context, FloatFormat format,
            OperationSchemaId schema, mlir::arith::FastMathFlags flags,
            std::optional<mlir::arith::RoundingMode> rounding = std::nullopt) {
  mlir::Type type = floatType(context, format);
  return {schema, mlir::FunctionType::get(&context, {type, type}, {type}),
          ::dataflow::FloatingPointPayload{flags, rounding}};
}

std::vector<std::uint8_t> bytes(const ::loom::CanonicalSemanticBytes &value) {
  return {value.bytes().begin(), value.bytes().end()};
}

std::vector<std::vector<std::uint8_t>>
keyBytes(llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  std::vector<std::vector<std::uint8_t>> result;
  for (const auto &point : domain) {
    if (point.semanticConfiguration)
      result.push_back(bytes(*point.semanticConfiguration));
  }
  return result;
}

void requireCanonicalKeySet(
    const char *test,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
    std::vector<std::vector<std::uint8_t>> expected) {
  std::vector<std::vector<std::uint8_t>> actual = keyBytes(domain);
  llvm::sort(expected);
  require(test, actual == expected, "behavior key bytes are not canonical");
  require(test, llvm::is_sorted(actual), "behavior keys are not sorted");
  require(test,
          std::adjacent_find(actual.begin(), actual.end()) == actual.end(),
          "behavior keys are not unique");
}

::loom::CanonicalSemanticBytes
expectedKey(const char *test, llvm::StringRef role,
            llvm::ArrayRef<detail::ImplementationFamilyBehaviorKeyComponent>
                components = {}) {
  return take(test, detail::encodeImplementationFamilyBehaviorKey(
                        ImplementationFamilyId::ScalarFloatCompareMinMax, role,
                        components));
}

void mixedStrictAndNnanActorsReuseExactFormatModes() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};

  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "mixed strict/nnan image did not retain exactly two exact modes");
  require(test,
          llvm::all_of(domain,
                       [](const auto &point) {
                         return point.semanticConfiguration.has_value() &&
                                point.operandPorts ==
                                    std::vector<std::uint64_t>({0, 1}) &&
                                point.resultPorts ==
                                    std::vector<std::uint64_t>({0});
                       }),
          "mixed domain lost a key or exact identity correspondence");

  std::vector<std::vector<std::uint8_t>> keys;
  for (const auto &point : domain)
    keys.emplace_back(point.semanticConfiguration->bytes().begin(),
                      point.semanticConfiguration->bytes().end());
  require(test, llvm::is_sorted(keys), "behavior keys are not sorted");
  require(test, std::adjacent_find(keys.begin(), keys.end()) == keys.end(),
          "behavior keys are not unique");

  std::vector<std::vector<std::uint8_t>> expected;
  for (FloatFormat format : {FloatFormat::F16, FloatFormat::BF16}) {
    auto encodedType =
        take(test, ::dataflow::encodeCanonicalType(floatType(context, format)));
    std::array<detail::ImplementationFamilyBehaviorKeyComponent, 2> components =
        {std::uint32_t{1}, std::move(encodedType)};
    expected.push_back(bytes(expectedKey(test, "", components)));
  }
  requireCanonicalKeySet(test, domain, std::move(expected));

  for (FloatFormat format : {FloatFormat::F16, FloatFormat::BF16}) {
    auto strict = compareActor(context, format, mlir::arith::CmpFPredicate::UGT,
                               mlir::arith::FastMathFlags::none);
    auto relaxed =
        compareActor(context, format, mlir::arith::CmpFPredicate::UGT,
                     mlir::arith::FastMathFlags::nnan);
    auto strictKey =
        take(test, detail::projectScalarFloatCompareBehavior(
                       ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                       schemas, strict, domain));
    auto relaxedKey =
        take(test, detail::projectScalarFloatCompareBehavior(
                       ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                       schemas, relaxed, domain));
    require(test, strictKey.bytes().equals(relaxedKey.bytes()),
            "nnan actor created a mode beside its exact-format refinement");
  }
}

void predicatePrecedesTaggedNumericFormat() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      strictCompareBehavior(),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 4,
          "predicate and exact format did not form four strict modes");

  std::vector<std::vector<std::uint8_t>> expected;
  for (auto predicate :
       {mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::UGT}) {
    for (FloatFormat format : {FloatFormat::F16, FloatFormat::BF16}) {
      auto encodedPredicate =
          take(test, ::dataflow::encodeFloatComparePredicate(predicate));
      auto encodedType = take(
          test, ::dataflow::encodeCanonicalType(floatType(context, format)));
      std::array<detail::ImplementationFamilyBehaviorKeyComponent, 3>
          components = {std::move(encodedPredicate), std::uint32_t{1},
                        std::move(encodedType)};
      expected.push_back(bytes(expectedKey(test, "", components)));
    }
  }
  requireCanonicalKeySet(test, domain, std::move(expected));
}

void allNnanEqualWidthsCollapseToRepresentationWidth() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get(
          {FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32}),
      requiredNnanBehavior(FloatNaNBehavior::IEEE),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {1U};

  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "all-nnan f16/bf16 did not share the 16-bit behavior");
  std::vector<std::vector<std::uint8_t>> expected;
  for (std::uint32_t width : {16U, 32U}) {
    std::array<detail::ImplementationFamilyBehaviorKeyComponent, 2> components =
        {std::uint32_t{2}, width};
    expected.push_back(bytes(expectedKey(test, "", components)));
  }
  requireCanonicalKeySet(test, domain, std::move(expected));

  auto f16 = take(test, detail::projectScalarFloatCompareBehavior(
                            ImplementationFamilyId::ScalarFloatCompareMinMax,
                            params, schemas,
                            compareActor(context, FloatFormat::F16,
                                         mlir::arith::CmpFPredicate::UGT,
                                         mlir::arith::FastMathFlags::nnan),
                            domain));
  auto bf16 = take(test, detail::projectScalarFloatCompareBehavior(
                             ImplementationFamilyId::ScalarFloatCompareMinMax,
                             params, schemas,
                             compareActor(context, FloatFormat::BF16,
                                          mlir::arith::CmpFPredicate::UGT,
                                          mlir::arith::FastMathFlags::nnan),
                             domain));
  auto f32 = take(test, detail::projectScalarFloatCompareBehavior(
                            ImplementationFamilyId::ScalarFloatCompareMinMax,
                            params, schemas,
                            compareActor(context, FloatFormat::F32,
                                         mlir::arith::CmpFPredicate::UGT,
                                         mlir::arith::FastMathFlags::nnan),
                            domain));
  require(test, f16.bytes().equals(bf16.bytes()),
          "equal-width nnan actors selected different modes");
  require(test, !f16.bytes().equals(f32.bytes()),
          "different representation widths selected one mode");
}

void singletonAllNnanImageCollapsesToNone() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      requiredNnanBehavior(FloatNaNBehavior::IEEE),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 1 && !domain.front().semanticConfiguration,
          "singleton width cover did not collapse to None");
}

void mixedImagePreservesUnrefinedPredicatePreference() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F32}), strictCompareBehavior(),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "mixed predicate image gained a normalized behavior mode");

  auto strictUgt =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F32,
                                  mlir::arith::CmpFPredicate::UGT,
                                  mlir::arith::FastMathFlags::none),
                     domain));
  auto weakUgt =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F32,
                                  mlir::arith::CmpFPredicate::UGT,
                                  mlir::arith::FastMathFlags::nnan),
                     domain));
  auto strictOgt =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F32,
                                  mlir::arith::CmpFPredicate::OGT,
                                  mlir::arith::FastMathFlags::none),
                     domain));
  require(test, strictUgt.bytes().equals(weakUgt.bytes()),
          "weak UGT did not prefer its exact strict representative");
  require(test, !weakUgt.bytes().equals(strictOgt.bytes()),
          "mixed weak UGT was actor-locally normalized to OGT");
}

void uncoveredAliasesNormalizeWithoutExtraModes() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}),
      requiredNnanBehavior(FloatNaNBehavior::IEEE),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "ordered/unordered aliases did not collapse by width");
  for (FloatFormat format : {FloatFormat::F16, FloatFormat::F32}) {
    auto ordered = take(
        test,
        detail::projectScalarFloatCompareBehavior(
            ImplementationFamilyId::ScalarFloatCompareMinMax, params, schemas,
            compareActor(context, format, mlir::arith::CmpFPredicate::OGT,
                         mlir::arith::FastMathFlags::nnan),
            domain));
    auto unordered = take(
        test,
        detail::projectScalarFloatCompareBehavior(
            ImplementationFamilyId::ScalarFloatCompareMinMax, params, schemas,
            compareActor(context, format, mlir::arith::CmpFPredicate::UGT,
                         mlir::arith::FastMathFlags::nnan),
            domain));
    require(test, ordered.bytes().equals(unordered.bytes()),
            "nnan ordered/unordered aliases selected different modes");
  }
}

void uncoveredOrdUnoOmitNumericFormat() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}),
      requiredNnanBehavior(FloatNaNBehavior::IEEE),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::ORD, mlir::arith::CmpFPredicate::UNO})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "constant normalized predicates retained format modes");

  std::vector<std::vector<std::uint8_t>> expected;
  for (auto predicate : {mlir::arith::CmpFPredicate::AlwaysFalse,
                         mlir::arith::CmpFPredicate::AlwaysTrue}) {
    auto encoded =
        take(test, ::dataflow::encodeFloatComparePredicate(predicate));
    std::array<detail::ImplementationFamilyBehaviorKeyComponent, 1> components =
        {std::move(encoded)};
    expected.push_back(bytes(expectedKey(test, "", components)));
  }
  requireCanonicalKeySet(test, domain, std::move(expected));

  for (auto predicate :
       {mlir::arith::CmpFPredicate::ORD, mlir::arith::CmpFPredicate::UNO}) {
    auto f16 = take(test, detail::projectScalarFloatCompareBehavior(
                              ImplementationFamilyId::ScalarFloatCompareMinMax,
                              params, schemas,
                              compareActor(context, FloatFormat::F16, predicate,
                                           mlir::arith::FastMathFlags::nnan),
                              domain));
    auto f32 = take(test, detail::projectScalarFloatCompareBehavior(
                              ImplementationFamilyId::ScalarFloatCompareMinMax,
                              params, schemas,
                              compareActor(context, FloatFormat::F32, predicate,
                                           mlir::arith::FastMathFlags::nnan),
                              domain));
    require(test, f16.bytes().equals(f32.bytes()),
            "constant predicate retained numeric format");
  }
}

void minNumberReusesMinimumBeforeNormalization() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), strictCompareBehavior(),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithMinimumF,
                                  OperationSchemaId::ArithMaximumF,
                                  OperationSchemaId::ArithMinNumF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "weak minnum introduced a behavior beside minimum/maximum");
  auto minimum =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinimumF,
                                 mlir::arith::FastMathFlags::none),
                     domain));
  auto minNumber =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinNumF,
                                 mlir::arith::FastMathFlags::nnan),
                     domain));
  require(test, minimum.bytes().equals(minNumber.bytes()),
          "uncovered minnum did not reuse a compatible minimum");
}

void allNnanNumberRolesNormalizeSymmetrically() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = requiredNnanBehavior(FloatNaNBehavior::IEEE);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), behavior,
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array schemas = {
      OperationSchemaId::ArithMinimumF, OperationSchemaId::ArithMaximumF,
      OperationSchemaId::ArithMinNumF, OperationSchemaId::ArithMaxNumF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "nnan number/minimum roles did not form two symmetric modes");
  requireCanonicalKeySet(test, domain,
                         {bytes(expectedKey(test, "Minimum")),
                          bytes(expectedKey(test, "Maximum"))});

  const auto project = [&](OperationSchemaId schema) {
    return take(test, detail::projectScalarFloatCompareBehavior(
                          ImplementationFamilyId::ScalarFloatCompareMinMax,
                          params, schemas,
                          minMaxActor(context, FloatFormat::F16, schema,
                                      mlir::arith::FastMathFlags::nnan),
                          domain));
  };
  auto minimum = project(OperationSchemaId::ArithMinimumF);
  auto minNumber = project(OperationSchemaId::ArithMinNumF);
  auto maximum = project(OperationSchemaId::ArithMaximumF);
  auto maxNumber = project(OperationSchemaId::ArithMaxNumF);
  require(test, minimum.bytes().equals(minNumber.bytes()),
          "nnan minnum did not normalize to minimum");
  require(test, maximum.bytes().equals(maxNumber.bytes()),
          "nnan maxnum did not normalize to maximum");
  require(test, !minimum.bytes().equals(maximum.bytes()),
          "minimum and maximum collapsed to one behavior");
}

void numberPreferredOnlyRetainsStrictNumberRoles() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = strictCompareBehavior();
  behavior.nanBehaviors =
      FloatNaNBehaviorSet::get({FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), behavior,
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithMinNumF,
                                  OperationSchemaId::ArithMaxNumF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "number-preferred profile did not retain two strict number roles");
  requireCanonicalKeySet(test, domain,
                         {bytes(expectedKey(test, "MinNumber")),
                          bytes(expectedKey(test, "MaxNumber"))});

  for (OperationSchemaId schema : schemas) {
    auto key = take(test, detail::projectScalarFloatCompareBehavior(
                              ImplementationFamilyId::ScalarFloatCompareMinMax,
                              params, schemas,
                              minMaxActor(context, FloatFormat::F16, schema,
                                          mlir::arith::FastMathFlags::none),
                              domain));
    require(test, llvm::is_contained(keyBytes(domain), bytes(key)),
            "strict number actor projected outside its exact role");
  }
}

void fallbackUsesLexicographicallySmallestCompatibleKey() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      strictCompareBehavior(),
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithMinimumF,
                                  OperationSchemaId::ArithMaximumF,
                                  OperationSchemaId::ArithMinNumF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 4,
          "weak minnum changed the four exact min/max modes");

  auto minimumF16 =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinimumF,
                                 mlir::arith::FastMathFlags::none),
                     domain));
  auto minimumBF16 =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::BF16,
                                 OperationSchemaId::ArithMinimumF,
                                 mlir::arith::FastMathFlags::none),
                     domain));
  auto minNumberF16 =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinNumF,
                                 mlir::arith::FastMathFlags::nnan),
                     domain));
  const auto &lexicographicMinimum =
      std::lexicographical_compare(
          minimumF16.bytes().begin(), minimumF16.bytes().end(),
          minimumBF16.bytes().begin(), minimumBF16.bytes().end())
          ? minimumF16
          : minimumBF16;
  require(test, minNumberF16.bytes().equals(lexicographicMinimum.bytes()),
          "fallback did not select the lexicographically smallest key");

  auto minNumberBF16 =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::BF16,
                                 OperationSchemaId::ArithMinNumF,
                                 mlir::arith::FastMathFlags::nnan),
                     domain));
  require(test, minNumberBF16.bytes().equals(minimumBF16.bytes()),
          "f16 strict mode incorrectly refined a bf16 nnan actor");
}

void enabledSchemaOrderDoesNotAffectTheCover() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  FloatBehaviorProfile behavior = strictCompareBehavior();
  behavior.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), behavior,
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array forward = {OperationSchemaId::ArithMinimumF,
                                  OperationSchemaId::ArithMaxNumF};
  constexpr std::array reverse = {OperationSchemaId::ArithMaxNumF,
                                  OperationSchemaId::ArithMinimumF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {16U};
  auto first = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                              ImplementationFamilyId::ScalarFloatCompareMinMax,
                              params, forward, inputs, results, context));
  auto second = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, reverse, inputs, results, context));
  require(test, keyBytes(first) == keyBytes(second),
          "enabled schema order changed canonical behavior keys");
  require(test, first.size() == second.size(),
          "enabled schema order changed cover cardinality");
  for (auto [lhs, rhs] : llvm::zip(first, second)) {
    auto lhsActor = take(test, ::dataflow::encodeCanonicalActorSchemaProjection(
                                   lhs.representativeActor));
    auto rhsActor = take(test, ::dataflow::encodeCanonicalActorSchemaProjection(
                                   rhs.representativeActor));
    require(test, lhsActor.bytes().equals(rhsActor.bytes()),
            "enabled schema order changed representative witness");
  }
}

void physicalReachabilityFiltersExactFormats() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get(
          {FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32}),
      strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));
  require(test, domain.size() == 2,
          "physical width did not filter the unreachable f32 format");
  require(test,
          llvm::all_of(domain,
                       [](const auto &point) {
                         return point.operandPorts ==
                                    std::vector<std::uint64_t>({0, 1}) &&
                                point.resultPorts ==
                                    std::vector<std::uint64_t>({0}) &&
                                !point.resolvedIndexWidth;
                       }),
          "physical role correspondence is not exact identity");

  constexpr std::array narrowInputs = {15U, 15U};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas, narrowInputs, results, context),
                 "physically reachable");
}

void invalidCapabilityContextsAreRejected() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const ScalarFloatCompareMinMaxParams typed{
      FloatFormatSet::get({FloatFormat::F16}), strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::OLT})};
  const FamilyCapabilityParams params = typed;
  constexpr std::array compare = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array compareResults = {1U};

  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatAddSub, params, compare,
                     inputs, compareResults, context),
                 "family");
  const FamilyCapabilityParams wrongParams = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16}), strictCompareBehavior()};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     wrongParams, compare, inputs, compareResults, context),
                 "parameter schema");
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     llvm::ArrayRef<OperationSchemaId>{}, inputs,
                     compareResults, context),
                 "enabled schema");
  constexpr std::array duplicates = {OperationSchemaId::ArithCmpF,
                                     OperationSchemaId::ArithCmpF};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     duplicates, inputs, compareResults, context),
                 "twice");
  constexpr std::array foreign = {OperationSchemaId::ArithAddF};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     foreign, inputs, compareResults, context),
                 "foreign");

  FloatBehaviorProfile orphanNaN = strictCompareBehavior();
  orphanNaN.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams orphanNaNParams = ScalarFloatCompareMinMaxParams{
      typed.formats, orphanNaN, typed.predicates};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     orphanNaNParams, compare, inputs, compareResults, context),
                 "number-preferred");

  const FamilyCapabilityParams orphanPredicate = ScalarFloatCompareMinMaxParams{
      typed.formats, typed.behavior,
      FloatPredicateSet::get(
          {mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OGT})};
  constexpr std::array minimum = {OperationSchemaId::ArithMinimumF};
  constexpr std::array floatResults = {16U};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     orphanPredicate, minimum, inputs, floatResults, context),
                 "predicate");

  FloatBehaviorProfile rounded = strictCompareBehavior();
  rounded.roundingModes =
      RoundingModeSet::get({mlir::arith::RoundingMode::to_nearest_even,
                            mlir::arith::RoundingMode::downward});
  const FamilyCapabilityParams roundedParams =
      ScalarFloatCompareMinMaxParams{typed.formats, rounded, typed.predicates};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     roundedParams, compare, inputs, compareResults, context),
                 "rounding");

  FloatBehaviorProfile physicallyOrphaned = strictCompareBehavior();
  physicallyOrphaned.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  const FamilyCapabilityParams physicallyOrphanedParams =
      ScalarFloatCompareMinMaxParams{typed.formats, physicallyOrphaned,
                                     typed.predicates};
  constexpr std::array compareAndMinNumber = {OperationSchemaId::ArithCmpF,
                                              OperationSchemaId::ArithMinNumF};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     physicallyOrphanedParams, compareAndMinNumber, inputs,
                     compareResults, context),
                 "number-preferred");

  const FamilyCapabilityParams unreachableSchemaParams =
      ScalarFloatCompareMinMaxParams{typed.formats, typed.behavior,
                                     typed.predicates};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     unreachableSchemaParams, compareAndMinNumber, inputs,
                     compareResults, context),
                 "enabled schema");

  FloatBehaviorProfile unobservable = physicallyOrphaned;
  unobservable.requiredFastMath = mlir::arith::FastMathFlags::nnan;
  const FamilyCapabilityParams unobservableParams =
      ScalarFloatCompareMinMaxParams{typed.formats, unobservable,
                                     typed.predicates};
  constexpr std::array minimumAndMinNumber = {OperationSchemaId::ArithMinimumF,
                                              OperationSchemaId::ArithMinNumF};
  expectRejected(test,
                 detail::resolveScalarFloatCompareBehaviorDomain(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     unobservableParams, minimumAndMinNumber, inputs,
                     floatResults, context),
                 "observable");
}

void projectorRejectsDisabledSchemasAndNoncanonicalDomains() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};
  auto domain = take(test, detail::resolveScalarFloatCompareBehaviorDomain(
                               ImplementationFamilyId::ScalarFloatCompareMinMax,
                               params, schemas, inputs, results, context));

  auto swappedActors = domain;
  std::swap(swappedActors[0].representativeActor,
            swappedActors[1].representativeActor);
  expectRejected(test,
                 detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F16,
                                  mlir::arith::CmpFPredicate::UGT,
                                  mlir::arith::FastMathFlags::none),
                     swappedActors),
                 "binding");

  expectRejected(test,
                 detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinimumF,
                                 mlir::arith::FastMathFlags::none),
                     domain),
                 "not enabled");

  std::reverse(domain.begin(), domain.end());
  expectRejected(test,
                 detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F16,
                                  mlir::arith::CmpFPredicate::UGT,
                                  mlir::arith::FastMathFlags::none),
                     domain),
                 "noncanonical");

  constexpr std::array minimum = {OperationSchemaId::ArithMinimumF};
  const FamilyCapabilityParams minimumParams = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16}), strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::OLT})};
  auto minimumDomain = take(
      test, detail::resolveScalarFloatCompareBehaviorDomain(
                ImplementationFamilyId::ScalarFloatCompareMinMax, minimumParams,
                minimum, inputs, std::array<std::uint32_t, 1>{16U}, context));
  expectRejected(test,
                 detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax,
                     minimumParams, minimum,
                     minMaxActor(context, FloatFormat::F16,
                                 OperationSchemaId::ArithMinimumF,
                                 mlir::arith::FastMathFlags::none,
                                 mlir::arith::RoundingMode::downward),
                     minimumDomain),
                 "noncanonical payload");
}

void projectorRejectsRedundantWeakRepresentatives() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}),
      strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};

  std::vector<FiniteImplementationFamilyBehaviorPoint> redundant;
  for (FloatFormat format : {FloatFormat::F16, FloatFormat::F32}) {
    auto encodedPredicate = take(test, ::dataflow::encodeFloatComparePredicate(
                                           mlir::arith::CmpFPredicate::UGT));
    auto encodedType =
        take(test, ::dataflow::encodeCanonicalType(floatType(context, format)));
    std::array<detail::ImplementationFamilyBehaviorKeyComponent, 3> components =
        {std::move(encodedPredicate), std::uint32_t{1}, std::move(encodedType)};
    redundant.emplace_back(
        compareActor(context, format, mlir::arith::CmpFPredicate::UGT,
                     mlir::arith::FastMathFlags::none),
        expectedKey(test, "", components), std::nullopt,
        std::vector<std::uint64_t>{0, 1}, std::vector<std::uint64_t>{0});
  }
  auto encodedOrdered = take(test, ::dataflow::encodeFloatComparePredicate(
                                       mlir::arith::CmpFPredicate::OGT));
  std::array<detail::ImplementationFamilyBehaviorKeyComponent, 3>
      widthComponents = {std::move(encodedOrdered), std::uint32_t{2},
                         std::uint32_t{16}};
  redundant.emplace_back(
      compareActor(context, FloatFormat::F16, mlir::arith::CmpFPredicate::UGT,
                   mlir::arith::FastMathFlags::nnan),
      expectedKey(test, "", widthComponents), std::nullopt,
      std::vector<std::uint64_t>{0, 1}, std::vector<std::uint64_t>{0});
  llvm::sort(redundant, [](const auto &lhs, const auto &rhs) {
    return std::lexicographical_compare(
        lhs.semanticConfiguration->bytes().begin(),
        lhs.semanticConfiguration->bytes().end(),
        rhs.semanticConfiguration->bytes().begin(),
        rhs.semanticConfiguration->bytes().end());
  });

  expectRejected(test,
                 detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas,
                     compareActor(context, FloatFormat::F16,
                                  mlir::arith::CmpFPredicate::UGT,
                                  mlir::arith::FastMathFlags::nnan),
                     redundant),
                 "redundant");
}

void publicRelationUsesTheScalarFloatCompareOwner() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16}),
      strictCompareBehavior(),
      FloatPredicateSet::get({mlir::arith::CmpFPredicate::UGT})};
  constexpr std::array schemas = {OperationSchemaId::ArithCmpF};
  constexpr std::array inputs = {16U, 16U};
  constexpr std::array results = {1U};
  const auto relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas, inputs, results, context));
  require(test,
          relation.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "public relation bypassed the scalar floating compare owner");

  const auto actor =
      compareActor(context, FloatFormat::F16, mlir::arith::CmpFPredicate::UGT,
                   mlir::arith::FastMathFlags::none);
  const auto publicKey =
      take(test, relation.projectSemanticValue(actor, {0, 1}, {0}));
  const auto ownerKey =
      take(test, detail::projectScalarFloatCompareBehavior(
                     ImplementationFamilyId::ScalarFloatCompareMinMax, params,
                     schemas, actor, relation.finiteBehaviorDomain()));
  require(test, publicKey.bytes().equals(ownerKey.bytes()),
          "public relation used a competing scalar floating compare codec");
}

} // namespace

int main() {
  mixedStrictAndNnanActorsReuseExactFormatModes();
  predicatePrecedesTaggedNumericFormat();
  allNnanEqualWidthsCollapseToRepresentationWidth();
  singletonAllNnanImageCollapsesToNone();
  mixedImagePreservesUnrefinedPredicatePreference();
  uncoveredAliasesNormalizeWithoutExtraModes();
  uncoveredOrdUnoOmitNumericFormat();
  minNumberReusesMinimumBeforeNormalization();
  allNnanNumberRolesNormalizeSymmetrically();
  numberPreferredOnlyRetainsStrictNumberRoles();
  fallbackUsesLexicographicallySmallestCompatibleKey();
  enabledSchemaOrderDoesNotAffectTheCover();
  physicalReachabilityFiltersExactFormats();
  invalidCapabilityContextsAreRejected();
  projectorRejectsDisabledSchemasAndNoncanonicalDomains();
  projectorRejectsRedundantWeakRepresentatives();
  publicRelationUsesTheScalarFloatCompareOwner();
  return EXIT_SUCCESS;
}
