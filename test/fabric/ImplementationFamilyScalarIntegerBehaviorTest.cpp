#include "ImplementationFamilyScalarIntegerBehavior.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {

using dataflow::OperationSchemaId;
using fabric::FamilyCapabilityParams;
using fabric::FiniteImplementationFamilyBehaviorPoint;
using fabric::ImplementationFamilyId;
using fabric::IntegerCastRelation;
using fabric::IntegerPredicateSet;
using fabric::IntegerWidth;
using fabric::IntegerWidthRelation;
using fabric::IntegerWidthSet;
using fabric::ResolvedIndexWidth;
using fabric::ResolvedIndexWidthSet;
using fabric::ScalarIntegerCastParams;
using fabric::ScalarIntegerCompareMinMaxParams;
using fabric::ScalarIntegerParams;

[[noreturn]] void fail(const char *test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(const char *test, bool condition, const llvm::Twine &message) {
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

unsigned integerWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  return integer ? integer.getWidth() : 0;
}

const FiniteImplementationFamilyBehaviorPoint &
findPoint(const char *test,
          llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> points,
          OperationSchemaId schema, unsigned inputWidth,
          std::optional<mlir::arith::CmpIPredicate> predicate = std::nullopt) {
  for (const FiniteImplementationFamilyBehaviorPoint &point : points) {
    if (point.representativeActor.schema != schema ||
        point.representativeActor.type.getNumInputs() == 0 ||
        integerWidth(point.representativeActor.type.getInput(0)) != inputWidth)
      continue;
    if (!predicate)
      return point;
    const auto *payload = std::get_if<dataflow::IntegerComparePayload>(
        &point.representativeActor.payload);
    if (payload && payload->predicate == *predicate)
      return point;
  }
  fail(test, "expected behavior point is absent");
}

void expectKey(const char *test,
               const FiniteImplementationFamilyBehaviorPoint &point,
               llvm::ArrayRef<std::uint8_t> expected) {
  require(test, point.semanticConfiguration.has_value(),
          "behavior point has no finite key");
  if (point.semanticConfiguration->bytes().equals(expected))
    return;
  llvm::errs() << test << ": expected " << expected.size() << " bytes, got "
               << point.semanticConfiguration->bytes().size() << ":";
  for (std::uint8_t byte : point.semanticConfiguration->bytes())
    llvm::errs() << ' ' << static_cast<unsigned>(byte);
  llvm::errs() << '\n';
  fail(test, "behavior point has unexpected canonical bytes");
}

std::vector<FiniteImplementationFamilyBehaviorPoint>
resolve(const char *test, ImplementationFamilyId family,
        const FamilyCapabilityParams &params,
        llvm::ArrayRef<OperationSchemaId> schemas,
        llvm::ArrayRef<std::uint32_t> inputWidths,
        llvm::ArrayRef<std::uint32_t> resultWidths,
        mlir::MLIRContext &context) {
  auto points = take(
      test, fabric::detail::resolveScalarIntegerBehaviorDomain(
                family, params, schemas, inputWidths, resultWidths, context));
  for (const FiniteImplementationFamilyBehaviorPoint &point : points) {
    require(test,
            point.operandPorts.size() ==
                    point.representativeActor.type.getNumInputs() &&
                point.resultPorts.size() ==
                    point.representativeActor.type.getNumResults(),
            "behavior witness has the wrong correspondence arity");
    for (auto [ordinal, port] : llvm::enumerate(point.operandPorts))
      require(test, port == ordinal,
              "behavior witness has a non-identity operand correspondence");
    for (auto [ordinal, port] : llvm::enumerate(point.resultPorts))
      require(test, port == ordinal,
              "behavior witness has a non-identity result correspondence");
  }
  if (points.size() > 1) {
    for (const FiniteImplementationFamilyBehaviorPoint &point : points)
      require(test, point.semanticConfiguration.has_value(),
              "non-singleton quotient omitted a behavior key");
    for (std::size_t ordinal = 1; ordinal != points.size(); ++ordinal) {
      const auto previous = points[ordinal - 1].semanticConfiguration->bytes();
      const auto current = points[ordinal].semanticConfiguration->bytes();
      require(test,
              std::lexicographical_compare(previous.begin(), previous.end(),
                                           current.begin(), current.end()),
              "behavior keys are not strictly sorted and unique");
    }
  }
  return points;
}

void addSubUsesRolesAndRejectsGep() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};

  constexpr std::array singletonSchemas = {OperationSchemaId::ArithAddI};
  auto singleton = resolve(test, ImplementationFamilyId::ScalarIntegerAddSub,
                           params, singletonSchemas, inputs, results, context);
  require(test,
          singleton.size() == 1 && !singleton.front().semanticConfiguration,
          "singleton add relation did not collapse to None");
  require(test,
          singleton.front().operandPorts ==
                  std::vector<std::uint64_t>({0, 1}) &&
              singleton.front().resultPorts == std::vector<std::uint64_t>({0}),
          "add relation lost its exact identity correspondence");

  constexpr std::array schemas = {OperationSchemaId::ArithSubI,
                                  OperationSchemaId::ArithAddI};
  auto points = resolve(test, ImplementationFamilyId::ScalarIntegerAddSub,
                        params, schemas, inputs, results, context);
  require(test, points.size() == 2,
          "add/sub relation did not expose both physical behaviors");
  constexpr std::array<std::uint8_t, 73> expectedAdd = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   19,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'A', 'd', 'd', 'S', 'u', 'b', 0,   0,   0,   3,   'A', 'd', 'd'};
  expectKey(test, findPoint(test, points, OperationSchemaId::ArithAddI, 32),
            expectedAdd);

  constexpr std::array gepSchemas = {OperationSchemaId::ArithAddI,
                                     OperationSchemaId::LLVMGetElementPtr};
  expectError(test,
              fabric::detail::resolveScalarIntegerBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerAddSub, params,
                  gepSchemas, inputs, results, context),
              "GEP");
}

void publicRelationProjectsItsOwnedBehaviorKey() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithSubI,
                                  OperationSchemaId::ArithAddI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto relation = take(test, fabric::resolveFabricOpSemanticFieldRelation(
                                 ImplementationFamilyId::ScalarIntegerAddSub,
                                 params, schemas, inputs, results, context));
  require(test,
          relation.kind() ==
                  fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "public scalar relation does not own the exact behavior quotient");

  mlir::Type integer = mlir::IntegerType::get(&context, 32);
  const dataflow::CanonicalActorSchemaProjection actor{
      OperationSchemaId::ArithAddI,
      mlir::FunctionType::get(&context, {integer, integer}, {integer}),
      dataflow::IntegerOverflowPayload{}};
  const auto projected = take(
      test,
      relation.projectSemanticValue(actor, std::array<std::uint64_t, 2>{0, 1},
                                    std::array<std::uint64_t, 1>{0}));
  const auto &expected = findPoint(test, relation.finiteBehaviorDomain(),
                                   OperationSchemaId::ArithAddI, 32);
  require(test,
          expected.semanticConfiguration &&
              projected.bytes().equals(expected.semanticConfiguration->bytes()),
          "public scalar projector diverges from its sealed relation");
}

void publicRelationRejectsDisabledAliases() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithAndI,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto relation = take(test, fabric::resolveFabricOpSemanticFieldRelation(
                                 ImplementationFamilyId::ScalarIntegerLogic,
                                 params, schemas, inputs, results, context));

  mlir::Type integer = mlir::IntegerType::get(&context, 32);
  const dataflow::CanonicalActorSchemaProjection disabledAlias{
      OperationSchemaId::LLVMOrDisjoint,
      mlir::FunctionType::get(&context, {integer, integer}, {integer}),
      dataflow::DisjointPayload{true}};
  expectError(test,
              relation.projectSemanticValue(disabledAlias,
                                            std::array<std::uint64_t, 2>{0, 1},
                                            std::array<std::uint64_t, 1>{0}),
              "enabled");
}

void publicRelationRejectsMalformedActorsAndInvalidSingletons() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithAddI,
                                  OperationSchemaId::ArithSubI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto relation = take(test, fabric::resolveFabricOpSemanticFieldRelation(
                                 ImplementationFamilyId::ScalarIntegerAddSub,
                                 params, schemas, inputs, results, context));
  mlir::Type integer = mlir::IntegerType::get(&context, 32);
  const dataflow::CanonicalActorSchemaProjection malformed{
      OperationSchemaId::ArithAddI,
      mlir::FunctionType::get(&context, {integer, integer}, {integer}),
      dataflow::NoPayload{}};
  expectError(test,
              relation.projectSemanticValue(malformed,
                                            std::array<std::uint64_t, 2>{0, 1},
                                            std::array<std::uint64_t, 1>{0}),
              "payload");

  const FamilyCapabilityParams emptyParams =
      ScalarIntegerParams{IntegerWidthSet::get({})};
  constexpr std::array singletonSchema = {OperationSchemaId::ArithAddI};
  expectError(test,
              fabric::resolveFabricOpSemanticFieldRelation(
                  ImplementationFamilyId::ScalarIntegerAddSub, emptyParams,
                  singletonSchema, inputs, results, context),
              "width domain");
}

void logicAliasesCollapse() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::LLVMOrDisjoint,
                                  OperationSchemaId::ArithAndI,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto points = resolve(test, ImplementationFamilyId::ScalarIntegerLogic,
                        params, schemas, inputs, results, context);
  require(test, points.size() == 2,
          "logic aliases did not collapse to And and Or");
  constexpr std::array<std::uint8_t, 71> expectedOr = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   18,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'L', 'o', 'g', 'i', 'c', 0,   0,   0,   2,   'O', 'r'};
  const auto &point = findPoint(test, points, OperationSchemaId::ArithOrI, 8);
  expectKey(test, point, expectedOr);
  require(test,
          std::holds_alternative<dataflow::NoPayload>(
              point.representativeActor.payload),
          "logic alias did not select the canonical arith witness");
}

void shiftsRetainOnlyVariableSignedWidths() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithShLI,
                                  OperationSchemaId::ArithShRSI,
                                  OperationSchemaId::ArithShRUI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto points = resolve(test, ImplementationFamilyId::ScalarIntegerShift,
                        params, schemas, inputs, results, context);
  require(test, points.size() == 4,
          "shift quotient retained width-insensitive duplicates");
  constexpr std::array<std::uint8_t, 88> expectedArithmeticRight32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   18,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'S', 'h', 'i', 'f', 't', 0,   0,   0,   15,  'A', 'r', 'i', 't', 'h', 'm',
      'e', 't', 'i', 'c', 'R', 'i', 'g', 'h', 't', 0,   0,   0,   32};
  expectKey(test, findPoint(test, points, OperationSchemaId::ArithShRSI, 32),
            expectedArithmeticRight32);
}

void compareUsesRegisteredPredicates() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32}),
      IntegerPredicateSet::get({mlir::arith::CmpIPredicate::eq,
                                mlir::arith::CmpIPredicate::slt,
                                mlir::arith::CmpIPredicate::ult})};
  constexpr std::array schemas = {OperationSchemaId::ArithMinUI,
                                  OperationSchemaId::ArithCmpI,
                                  OperationSchemaId::ArithMinSI};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  auto points =
      resolve(test, ImplementationFamilyId::ScalarIntegerCompareMinMax, params,
              schemas, inputs, results, context);
  require(test, points.size() == 7,
          "compare/min/max quotient has the wrong behavior cardinality");
  const auto &signedCompare =
      findPoint(test, points, OperationSchemaId::ArithCmpI, 32,
                mlir::arith::CmpIPredicate::slt);
  constexpr std::array<std::uint8_t, 144> expectedSignedCompare32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   26,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'C', 'o', 'm', 'p', 'a', 'r', 'e', 'M', 'i', 'n', 'M', 'a', 'x', 0,   0,
      0,   7,   'C', 'o', 'm', 'p', 'a', 'r', 'e', 0,   0,   0,   52,  'l', 'o',
      'o', 'm', '.', 'd', 'a', 't', 'a', 'f', 'l', 'o', 'w', '.', 'i', 'n', 't',
      'e', 'g', 'e', 'r', '-', 'c', 'o', 'm', 'p', 'a', 'r', 'e', '-', 'p', 'r',
      'e', 'd', 'i', 'c', 'a', 't', 'e', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   3,   0,   0,   0,   32};
  expectKey(test, signedCompare, expectedSignedCompare32);
}

void castsCollapseAliasesAndRetainIndexWitnesses() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32},
                                     {IntegerWidth::I8, IntegerWidth::I64},
                                     {IntegerWidth::I32, IntegerWidth::I8},
                                     {IntegerWidth::I64, IntegerWidth::I8}}),
          ResolvedIndexWidthSet::get(
              {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})}};
  constexpr std::array schemas = {
      OperationSchemaId::ArithIndexCastUI, OperationSchemaId::ArithExtUI,
      OperationSchemaId::ArithTruncI, OperationSchemaId::ArithExtSI,
      OperationSchemaId::ArithIndexCast};
  constexpr std::array inputs = {64U};
  constexpr std::array results = {64U};
  auto points = resolve(test, ImplementationFamilyId::ScalarIntegerCast, params,
                        schemas, inputs, results, context);
  require(test, points.size() == 6,
          "integer cast aliases did not collapse by width transform");
  constexpr std::array<std::uint8_t, 82> expectedSignExtend32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p',
      'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i',
      'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   17,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't',
      'e', 'g', 'e', 'r', 'C', 'a', 's', 't', 0,   0,   0,   10,  'S', 'i',
      'g', 'n', 'E', 'x', 't', 'e', 'n', 'd', 0,   0,   0,   32};
  expectKey(test, findPoint(test, points, OperationSchemaId::ArithExtSI, 8),
            expectedSignExtend32);

  const FamilyCapabilityParams identityParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I32, IntegerWidth::I32}}),
          ResolvedIndexWidthSet::get({ResolvedIndexWidth::I32})}};
  constexpr std::array identitySchemas = {OperationSchemaId::ArithIndexCast,
                                          OperationSchemaId::ArithIndexCastUI};
  constexpr std::array identityPorts = {32U};
  auto identity =
      resolve(test, ImplementationFamilyId::ScalarIntegerCast, identityParams,
              identitySchemas, identityPorts, identityPorts, context);
  require(test,
          identity.size() == 1 && !identity.front().semanticConfiguration &&
              identity.front().resolvedIndexWidth == ResolvedIndexWidth::I32 &&
              llvm::isa<mlir::IndexType>(
                  identity.front().representativeActor.type.getInput(0)),
          "identity index cast lost its resolved-width witness");
}

void divRemWidthsAreSignednessSpecific() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32})};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};

  constexpr std::array signedSchemas = {OperationSchemaId::ArithDivSI,
                                        OperationSchemaId::ArithRemSI};
  auto signedPoints =
      resolve(test, ImplementationFamilyId::ScalarSignedIntegerDivRem, params,
              signedSchemas, inputs, results, context);
  require(test, signedPoints.size() == 4,
          "signed div/rem did not retain its active width dimension");
  constexpr std::array<std::uint8_t, 88> expectedSignedQuotient32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   25,  'S', 'c', 'a', 'l', 'a', 'r', 'S', 'i', 'g', 'n', 'e', 'd', 'I',
      'n', 't', 'e', 'g', 'e', 'r', 'D', 'i', 'v', 'R', 'e', 'm', 0,   0,   0,
      8,   'Q', 'u', 'o', 't', 'i', 'e', 'n', 't', 0,   0,   0,   32};
  expectKey(test,
            findPoint(test, signedPoints, OperationSchemaId::ArithDivSI, 32),
            expectedSignedQuotient32);

  constexpr std::array unsignedSchemas = {OperationSchemaId::ArithDivUI,
                                          OperationSchemaId::ArithRemUI};
  auto unsignedPoints =
      resolve(test, ImplementationFamilyId::ScalarUnsignedIntegerDivRem, params,
              unsignedSchemas, inputs, results, context);
  require(test, unsignedPoints.size() == 4,
          "unsigned div/rem did not retain its active width dimension");
  constexpr std::array<std::uint8_t, 90> expectedUnsignedQuotient32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   27,  'S', 'c', 'a', 'l', 'a', 'r', 'U', 'n', 's', 'i', 'g', 'n', 'e',
      'd', 'I', 'n', 't', 'e', 'g', 'e', 'r', 'D', 'i', 'v', 'R', 'e', 'm', 0,
      0,   0,   8,   'Q', 'u', 'o', 't', 'i', 'e', 'n', 't', 0,   0,   0,   32};
  expectKey(test,
            findPoint(test, unsignedPoints, OperationSchemaId::ArithDivUI, 32),
            expectedUnsignedQuotient32);

  constexpr std::array narrowInputs = {8U, 8U};
  constexpr std::array narrowResults = {8U};
  auto narrow =
      resolve(test, ImplementationFamilyId::ScalarSignedIntegerDivRem, params,
              signedSchemas, narrowInputs, narrowResults, context);
  require(test, narrow.size() == 2,
          "physical filtering did not remove unreachable signed widths");
  constexpr std::array<std::uint8_t, 84> expectedNarrowQuotient = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p',
      'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i',
      'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   25,  'S', 'c', 'a', 'l', 'a', 'r', 'S', 'i', 'g',
      'n', 'e', 'd', 'I', 'n', 't', 'e', 'g', 'e', 'r', 'D', 'i', 'v', 'R',
      'e', 'm', 0,   0,   0,   8,   'Q', 'u', 'o', 't', 'i', 'e', 'n', 't'};
  expectKey(test, findPoint(test, narrow, OperationSchemaId::ArithDivSI, 8),
            expectedNarrowQuotient);
}

void saturatingAndCountZeroRolesRetainWidths() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32})};
  constexpr std::array inputs = {32U, 32U};
  constexpr std::array results = {32U};
  constexpr std::array saturatingSchemas = {
      OperationSchemaId::LLVMUSubSat, OperationSchemaId::LLVMSAddSat,
      OperationSchemaId::LLVMUAddSat, OperationSchemaId::LLVMSSubSat};
  auto saturating =
      resolve(test, ImplementationFamilyId::ScalarIntegerSaturatingAddSub,
              params, saturatingSchemas, inputs, results, context);
  require(test, saturating.size() == 8,
          "saturating add/sub quotient has the wrong cardinality");
  constexpr std::array<std::uint8_t, 93> expectedSignedAdd32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p',
      'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i',
      'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   29,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't',
      'e', 'g', 'e', 'r', 'S', 'a', 't', 'u', 'r', 'a', 't', 'i', 'n', 'g',
      'A', 'd', 'd', 'S', 'u', 'b', 0,   0,   0,   9,   'S', 'i', 'g', 'n',
      'e', 'd', 'A', 'd', 'd', 0,   0,   0,   32};
  expectKey(test,
            findPoint(test, saturating, OperationSchemaId::LLVMSAddSat, 32),
            expectedSignedAdd32);

  constexpr std::array countInputs = {32U};
  constexpr std::array countSchemas = {
      OperationSchemaId::LLVMCountTrailingZeros,
      OperationSchemaId::MathCountLeadingZeros,
      OperationSchemaId::LLVMCountLeadingZeros,
      OperationSchemaId::MathCountTrailingZeros};
  auto count = resolve(test, ImplementationFamilyId::ScalarIntegerCountZeros,
                       params, countSchemas, countInputs, results, context);
  require(test, count.size() == 4,
          "count-zero aliases did not collapse by direction and width");
  constexpr std::array<std::uint8_t, 85> expectedLeading32 = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   23,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'C', 'o', 'u', 'n', 't', 'Z', 'e', 'r', 'o', 's', 0,   0,   0,   7,   'L',
      'e', 'a', 'd', 'i', 'n', 'g', 0,   0,   0,   32};
  const auto &leading =
      findPoint(test, count, OperationSchemaId::MathCountLeadingZeros, 32);
  expectKey(test, leading, expectedLeading32);
}

} // namespace

int main() {
  addSubUsesRolesAndRejectsGep();
  publicRelationProjectsItsOwnedBehaviorKey();
  publicRelationRejectsDisabledAliases();
  publicRelationRejectsMalformedActorsAndInvalidSingletons();
  logicAliasesCollapse();
  shiftsRetainOnlyVariableSignedWidths();
  compareUsesRegisteredPredicates();
  castsCollapseAliasesAndRetainIndexWitnesses();
  divRemWidthsAreSignednessSpecific();
  saturatingAndCountZeroRolesRetainWidths();
  return EXIT_SUCCESS;
}
