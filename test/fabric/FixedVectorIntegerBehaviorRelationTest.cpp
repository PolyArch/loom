#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
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
void expectError(const char *test, llvm::Expected<T> value,
                 llvm::StringRef fragment) {
  if (value)
    fail(test, "expected relation rejection");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(fragment),
          "unexpected rejection: " + message);
}

FabricOpSemanticFieldRelation resolve(const char *test,
                                      ImplementationFamilyId family,
                                      const FamilyCapabilityParams &params,
                                      llvm::ArrayRef<OperationSchemaId> schemas,
                                      llvm::ArrayRef<std::uint32_t> inputs,
                                      llvm::ArrayRef<std::uint32_t> results,
                                      mlir::MLIRContext &context) {
  return take(test, resolveFabricOpSemanticFieldRelation(
                        family, params, schemas, inputs, results, context));
}

std::uint32_t
elementWidth(const FiniteImplementationFamilyBehaviorPoint &point) {
  for (mlir::Type type : point.representativeActor.type.getInputs()) {
    if (auto vector = llvm::dyn_cast<mlir::VectorType>(type)) {
      if (!vector.getElementType().isInteger(1))
        return vector.getElementTypeBitWidth();
    }
  }
  auto vector =
      llvm::cast<mlir::VectorType>(point.representativeActor.type.getResult(0));
  return vector.getElementTypeBitWidth();
}

void requireExactIdentityPorts(const char *test,
                               const FabricOpSemanticFieldRelation &relation) {
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test,
            point.operandPorts.size() ==
                point.representativeActor.type.getNumInputs(),
            "operand correspondence has the wrong arity");
    require(test,
            point.resultPorts.size() ==
                point.representativeActor.type.getNumResults(),
            "result correspondence has the wrong arity");
    for (auto [ordinal, port] : llvm::enumerate(point.operandPorts))
      require(test, port == ordinal,
              "operand correspondence is not the exact identity image");
    for (auto [ordinal, port] : llvm::enumerate(point.resultPorts))
      require(test, port == ordinal,
              "result correspondence is not the exact identity image");
  }
}

void requireCanonicalDomain(const char *test,
                            const FabricOpSemanticFieldRelation &relation) {
  std::vector<std::vector<std::uint8_t>> keys;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    if (!point.semanticConfiguration)
      continue;
    keys.emplace_back(point.semanticConfiguration->bytes().begin(),
                      point.semanticConfiguration->bytes().end());
  }
  require(test, llvm::is_sorted(keys),
          "behavior keys are not canonically ordered");
  require(test, std::adjacent_find(keys.begin(), keys.end()) == keys.end(),
          "behavior keys are not unique");
}

void physicalFilteringPrecedesQuotientEncoding() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  constexpr std::array schemas = {OperationSchemaId::ArithMulI};
  constexpr std::array ports = {64U, 64U};
  constexpr std::array results = {64U};
  const auto relation =
      resolve(test, ImplementationFamilyId::FixedVectorIntegerMultiply, params,
              schemas, ports, results, context);

  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::Finite,
          "two reachable element widths did not form a finite relation");
  require(test, relation.finiteBehaviorDomain().size() == 2,
          "equal flattened widths collapsed distinct element widths");
  requireExactIdentityPorts(test, relation);
  requireCanonicalDomain(test, relation);

  std::vector<std::uint32_t> widths;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "multi-point relation omitted a canonical key");
    auto vector = llvm::cast<mlir::VectorType>(
        point.representativeActor.type.getInput(0));
    widths.push_back(vector.getElementTypeBitWidth());
    require(test,
            vector.getNumElements() * vector.getElementTypeBitWidth() == 64,
            "representative actor did not use reachable physical capacity");
  }
  llvm::sort(widths);
  require(test, widths == std::vector<std::uint32_t>({8, 16}),
          "finite relation lost a reachable element width");

  const auto point =
      llvm::find_if(relation.finiteBehaviorDomain(), [](const auto &candidate) {
        return elementWidth(candidate) == 8;
      });
  constexpr std::array<std::uint8_t, 81> expected = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p',
      'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i',
      'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   26,  'F', 'i', 'x', 'e', 'd', 'V', 'e', 'c', 't',
      'o', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r', 'M', 'u', 'l', 't', 'i',
      'p', 'l', 'y', 0,   0,   0,   0,   0,   0,   0,   8};
  require(test,
          point != relation.finiteBehaviorDomain().end() &&
              point->semanticConfiguration &&
              point->semanticConfiguration->bytes().equals(expected),
          "multiply key escaped the canonical behavior-key codec");
}

void unreachableWidthsCollapseToNone() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  constexpr std::array schemas = {OperationSchemaId::ArithMulI};
  constexpr std::array ports = {8U, 8U};
  constexpr std::array results = {8U};
  const auto relation =
      resolve(test, ImplementationFamilyId::FixedVectorIntegerMultiply, params,
              schemas, ports, results, context);

  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::None,
          "one reachable behavior retained a semantic field");
  require(test, relation.finiteBehaviorDomain().size() == 1,
          "singleton relation lost its projection witness");
  require(test, !relation.finiteBehaviorDomain().front().semanticConfiguration,
          "singleton relation retained a placeholder key");
  require(test, elementWidth(relation.finiteBehaviorDomain().front()) == 8,
          "physical filtering retained an unreachable element width");
}

void aliasesCollapseBeforeQuotienting() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  constexpr std::array schemas = {OperationSchemaId::ArithAndI,
                                  OperationSchemaId::LLVMOrDisjoint,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array ports = {64U, 64U};
  constexpr std::array results = {64U};
  const auto relation =
      resolve(test, ImplementationFamilyId::FixedVectorIntegerLogic, params,
              schemas, ports, results, context);

  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::Finite,
          "two logic behaviors did not form a finite relation");
  require(test, relation.finiteBehaviorDomain().size() == 2,
          "logic aliases or element widths escaped quotienting");
  requireExactIdentityPorts(test, relation);
  requireCanonicalDomain(test, relation);

  const auto andPoint =
      llvm::find_if(relation.finiteBehaviorDomain(), [](const auto &point) {
        return point.representativeActor.schema == OperationSchemaId::ArithAndI;
      });
  constexpr std::array<std::uint8_t, 77> expectedAnd = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o',
      'p', 'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a',
      'v', 'i', 'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,
      0,   0,   0,   0,   0,   0,   0,   23,  'F', 'i', 'x', 'e', 'd',
      'V', 'e', 'c', 't', 'o', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'L', 'o', 'g', 'i', 'c', 0,   0,   0,   3,   'A', 'n', 'd'};
  require(test,
          andPoint != relation.finiteBehaviorDomain().end() &&
              andPoint->semanticConfiguration &&
              andPoint->semanticConfiguration->bytes().equals(expectedAnd),
          "logic role escaped the canonical behavior-key codec");

  const auto makeActor = [&](OperationSchemaId schema, IntegerWidth width,
                             dataflow::SemanticPayload payload) {
    mlir::Type vector = mlir::VectorType::get(
        {2}, mlir::IntegerType::get(&context, getBitWidth(width)));
    return dataflow::CanonicalActorSchemaProjection{
        schema, mlir::FunctionType::get(&context, {vector, vector}, {vector}),
        std::move(payload)};
  };
  const auto arithOr = makeActor(OperationSchemaId::ArithOrI, IntegerWidth::I8,
                                 dataflow::NoPayload{});
  const auto llvmOr =
      makeActor(OperationSchemaId::LLVMOrDisjoint, IntegerWidth::I16,
                dataflow::DisjointPayload{true});
  const auto arithKey = take(
      test,
      relation.projectSemanticValue(arithOr, std::array<std::uint64_t, 2>{0, 1},
                                    std::array<std::uint64_t, 1>{0}));
  const auto llvmKey = take(
      test,
      relation.projectSemanticValue(llvmOr, std::array<std::uint64_t, 2>{0, 1},
                                    std::array<std::uint64_t, 1>{0}));
  require(test, arithKey.bytes().equals(llvmKey.bytes()),
          "equivalent logic aliases projected to different behavior keys");
}

void concreteRelationRejectsDisabledAliasesAndInvalidSingletons() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      FixedVectorIntegerParams{IntegerWidthSet::get({IntegerWidth::I8}), 64};
  constexpr std::array schemas = {OperationSchemaId::ArithAndI,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array inputs = {64U, 64U};
  constexpr std::array results = {64U};
  const auto relation =
      resolve(test, ImplementationFamilyId::FixedVectorIntegerLogic, params,
              schemas, inputs, results, context);
  mlir::Type vector = mlir::VectorType::get(
      {2}, mlir::IntegerType::get(&context, getBitWidth(IntegerWidth::I8)));
  const dataflow::CanonicalActorSchemaProjection disabledAlias{
      OperationSchemaId::LLVMOrDisjoint,
      mlir::FunctionType::get(&context, {vector, vector}, {vector}),
      dataflow::DisjointPayload{true}};
  expectError(test,
              relation.projectSemanticValue(disabledAlias,
                                            std::array<std::uint64_t, 2>{0, 1},
                                            std::array<std::uint64_t, 1>{0}),
              "enabled");

  const FamilyCapabilityParams invalid =
      FixedVectorIntegerParams{IntegerWidthSet::get({IntegerWidth::I8}), 0};
  constexpr std::array multiply = {OperationSchemaId::ArithMulI};
  expectError(test,
              resolveFabricOpSemanticFieldRelation(
                  ImplementationFamilyId::FixedVectorIntegerMultiply, invalid,
                  multiply, inputs, results, context),
              "invalid parameters");
}

void everyVectorIntegerFamilyOwnsItsExactImage() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams integers = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  const FamilyCapabilityParams compare = FixedVectorIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}),
      IntegerPredicateSet::get({mlir::arith::CmpIPredicate::eq,
                                mlir::arith::CmpIPredicate::slt,
                                mlir::arith::CmpIPredicate::ugt}),
      128};
  const FamilyCapabilityParams select = FixedVectorValueSelectParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}),
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}), 128};
  constexpr std::array binaryPorts = {64U, 64U};
  constexpr std::array unaryPorts = {64U};
  constexpr std::array selectPorts = {64U, 64U, 64U};
  constexpr std::array results = {64U};

  constexpr std::array addSub = {OperationSchemaId::ArithAddI,
                                 OperationSchemaId::ArithSubI};
  constexpr std::array shifts = {OperationSchemaId::ArithShLI,
                                 OperationSchemaId::ArithShRSI,
                                 OperationSchemaId::ArithShRUI};
  constexpr std::array comparisons = {OperationSchemaId::ArithCmpI,
                                      OperationSchemaId::ArithMinSI,
                                      OperationSchemaId::ArithMaxUI};
  constexpr std::array selects = {OperationSchemaId::ArithSelect};
  constexpr std::array saturating = {
      OperationSchemaId::LLVMSAddSat, OperationSchemaId::LLVMUAddSat,
      OperationSchemaId::LLVMSSubSat, OperationSchemaId::LLVMUSubSat};
  constexpr std::array countZeros = {OperationSchemaId::MathCountLeadingZeros,
                                     OperationSchemaId::LLVMCountLeadingZeros,
                                     OperationSchemaId::MathCountTrailingZeros,
                                     OperationSchemaId::LLVMCountTrailingZeros};

  struct Case final {
    ImplementationFamilyId family;
    const FamilyCapabilityParams *params;
    llvm::ArrayRef<OperationSchemaId> schemas;
    llvm::ArrayRef<std::uint32_t> inputs;
    std::size_t behaviorCount;
  };
  const std::array cases = {
      Case{ImplementationFamilyId::FixedVectorIntegerAddSub, &integers, addSub,
           binaryPorts, 4},
      Case{ImplementationFamilyId::FixedVectorIntegerShift, &integers, shifts,
           binaryPorts, 6},
      Case{ImplementationFamilyId::FixedVectorIntegerCompareMinMax, &compare,
           comparisons, binaryPorts, 10},
      Case{ImplementationFamilyId::FixedVectorValueSelect, &select, selects,
           selectPorts, 3},
      Case{ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub,
           &integers, saturating, binaryPorts, 8},
      Case{ImplementationFamilyId::FixedVectorIntegerCountZeros, &integers,
           countZeros, unaryPorts, 4},
  };
  for (const Case &entry : cases) {
    const auto relation =
        resolve(test, entry.family, *entry.params, entry.schemas, entry.inputs,
                results, context);
    require(test, relation.kind() == FabricOpSemanticFieldRelationKind::Finite,
            implementationFamilyKeyword(entry.family).str() +
                " did not resolve a finite relation");
    require(test, relation.finiteBehaviorDomain().size() == entry.behaviorCount,
            implementationFamilyKeyword(entry.family).str() +
                " exposed the wrong behavior quotient");
    requireExactIdentityPorts(test, relation);
    requireCanonicalDomain(test, relation);
  }
}

void projectionIgnoresIncidentalVectorShape() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  constexpr std::array schemas = {OperationSchemaId::ArithMulI};
  constexpr std::array ports = {64U, 64U};
  constexpr std::array results = {64U};
  const auto relation =
      resolve(test, ImplementationFamilyId::FixedVectorIntegerMultiply, params,
              schemas, ports, results, context);

  mlir::Type vector =
      mlir::VectorType::get({2}, mlir::IntegerType::get(&context, 8));
  const dataflow::CanonicalActorSchemaProjection actor{
      OperationSchemaId::ArithMulI,
      mlir::FunctionType::get(&context, {vector, vector}, {vector}),
      dataflow::IntegerOverflowPayload{}};
  const auto projected = take(
      test,
      relation.projectSemanticValue(actor, std::array<std::uint64_t, 2>{0, 1},
                                    std::array<std::uint64_t, 1>{0}));
  const auto witness =
      llvm::find_if(relation.finiteBehaviorDomain(),
                    [](const auto &point) { return elementWidth(point) == 8; });
  require(test,
          witness != relation.finiteBehaviorDomain().end() &&
              witness->semanticConfiguration &&
              projected.bytes().equals(witness->semanticConfiguration->bytes()),
          "projector treated vector shape as configured behavior");
}

} // namespace

int main() {
  physicalFilteringPrecedesQuotientEncoding();
  unreachableWidthsCollapseToNone();
  aliasesCollapseBeforeQuotienting();
  concreteRelationRejectsDisabledAliasesAndInvalidSingletons();
  everyVectorIntegerFamilyOwnsItsExactImage();
  projectionIgnoresIncidentalVectorShape();
  return EXIT_SUCCESS;
}
