#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
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

void expectError(const char *test, llvm::Error error,
                 llvm::StringRef expected) {
  require(test, static_cast<bool>(error), "expected failure");
  std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "failure did not contain '" + expected.str() + "': " + message);
}

void setPackedField(std::vector<std::uint8_t> &bytes, std::uint32_t offset,
                    std::uint32_t width, std::uint64_t value) {
  for (std::uint32_t bit = 0; bit != width; ++bit)
    if (((value >> bit) & 1U) != 0)
      bytes[(offset + bit) / 8] |=
          static_cast<std::uint8_t>(1U << ((offset + bit) % 8));
}

void concreteCapabilitiesResolveOneSealedRelation() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);

  const FamilyCapabilityParams noneParams =
      ScalarBitReinterpretParams{IntegerWidthSet::get({IntegerWidth::I32}),
                                 FloatFormatSet::get({FloatFormat::F32})};
  constexpr std::array noneSchemas = {OperationSchemaId::ArithBitcast};
  constexpr std::array noneInputs = {32U};
  constexpr std::array noneResults = {32U};
  const FabricOpSemanticFieldRelation none =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarBitReinterpret, noneParams,
                     noneSchemas, noneInputs, noneResults, context));
  require(test, none.kind() == FabricOpSemanticFieldRelationKind::None,
          "bit-preserving capability did not resolve to None");
  require(test, !none.hasConfigurationField(),
          "None relation exposed a configuration field");
  require(test, !none.directEncodedBitCount(),
          "None relation exposed a direct carrier");

  const FamilyCapabilityParams finiteParams =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array finiteSchemas = {OperationSchemaId::ArithAndI,
                                        OperationSchemaId::ArithOrI};
  constexpr std::array finiteInputs = {32U, 32U};
  constexpr std::array finiteResults = {32U};
  const FabricOpSemanticFieldRelation finite =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarIntegerLogic, finiteParams,
                     finiteSchemas, finiteInputs, finiteResults, context));
  require(test, finite.kind() == FabricOpSemanticFieldRelationKind::Finite,
          "multi-operation logic capability did not resolve to Finite");
  require(test, finite.hasConfigurationField(),
          "Finite relation omitted its configuration field");
  require(test, !finite.directEncodedBitCount(),
          "Finite relation exposed a direct carrier");
  require(test, finite.finiteBehaviorDomain().size() == 2,
          "Finite relation did not own both canonical behaviors");
  for (const FiniteImplementationFamilyBehaviorPoint &point :
       finite.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "Finite relation contains a behavior without a key");
    if (llvm::Error error =
            finite.validateSemanticValue(point.semanticConfiguration->bytes()))
      fail(test, llvm::toString(std::move(error)));
  }
  expectError(test, finite.validateSemanticValue({0xff}), "domain");

  const FamilyCapabilityParams directParams =
      FixedVectorShuffleParams{IntegerWidthSet::get({IntegerWidth::I16}),
                               FloatFormatSet{},
                               128,
                               128,
                               32,
                               5,
                               4};
  constexpr std::array directSchemas = {OperationSchemaId::VectorShuffle};
  constexpr std::array directInputs = {128U, 128U};
  constexpr std::array directResults = {128U};
  const FabricOpSemanticFieldRelation direct =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::FixedVectorShuffle, directParams,
                     directSchemas, directInputs, directResults, context));
  const auto *layout = direct.fixedVectorShuffleLayout();
  require(test, layout != nullptr,
          "Direct shuffle relation did not own its layout");
  require(test, direct.kind() == FabricOpSemanticFieldRelationKind::Direct,
          "structural shuffle capability did not resolve to Direct");
  require(test, direct.hasConfigurationField(),
          "Direct relation omitted its configuration field");
  require(test, direct.directEncodedBitCount() == layout->encodedBitCount,
          "Direct relation does not expose the Fabric-owned carrier width");

  std::vector<std::uint8_t> invalidSelector((layout->encodedBitCount + 7) / 8,
                                            0);
  setPackedField(invalidSelector, layout->blockWidthBitOffset,
                 layout->blockWidthBitCount, 15);
  setPackedField(invalidSelector, layout->resultBlockCountBitOffset,
                 layout->resultBlockCountBitCount, 0);
  const std::uint32_t selector = 5;
  setPackedField(invalidSelector, layout->selectorBitOffset,
                 layout->selectorBitCount, selector);
  expectError(test, direct.validateSemanticValue(invalidSelector), "selector");

  std::vector<std::uint8_t> activeTrailingZero(
      (layout->encodedBitCount + 7) / 8, 0);
  setPackedField(activeTrailingZero, layout->blockWidthBitOffset,
                 layout->blockWidthBitCount, 15);
  setPackedField(activeTrailingZero, layout->resultBlockCountBitOffset,
                 layout->resultBlockCountBitCount, 1);
  setPackedField(activeTrailingZero, layout->selectorBitOffset,
                 layout->selectorBitCount, 1);
  if (llvm::Error error = direct.validateSemanticValue(activeTrailingZero))
    fail(test, llvm::toString(std::move(error)));

  std::vector<std::uint8_t> nonzeroPadding((layout->encodedBitCount + 7) / 8,
                                           0);
  setPackedField(nonzeroPadding, layout->blockWidthBitOffset,
                 layout->blockWidthBitCount, 15);
  setPackedField(nonzeroPadding, layout->selectorBitOffset,
                 layout->selectorBitCount, 1);
  setPackedField(nonzeroPadding,
                 layout->selectorBitOffset + layout->selectorBitCount,
                 layout->selectorBitCount, 1);
  expectError(test, direct.validateSemanticValue(nonzeroPadding), "padding");

  const FamilyCapabilityParams constantParams = PayloadCapacityParams{32};
  constexpr std::array constantSchemas = {OperationSchemaId::DataflowConstant};
  constexpr std::array constantInputs = {1U};
  constexpr std::array constantResults = {32U};
  const FabricOpSemanticFieldRelation constant = take(
      test, resolveFabricOpSemanticFieldRelation(
                ImplementationFamilyId::TokenConstant, constantParams,
                constantSchemas, constantInputs, constantResults, context));
  require(test,
          constant.kind() == FabricOpSemanticFieldRelationKind::Direct &&
              constant.directEncodedBitCount() == 32,
          "constant capability did not resolve to its exact Direct carrier");
  if (llvm::Error error =
          constant.validateSemanticValue({0xff, 0xff, 0xff, 0xff}))
    fail(test, llvm::toString(std::move(error)));
}

void directDomainsRejectUnreachablePackedValues() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);

  const FamilyCapabilityParams constantParams = PayloadCapacityParams{9};
  constexpr std::array constantSchemas = {OperationSchemaId::DataflowConstant};
  constexpr std::array constantInputs = {1U};
  constexpr std::array constantResults = {9U};
  const FabricOpSemanticFieldRelation constant = take(
      test, resolveFabricOpSemanticFieldRelation(
                ImplementationFamilyId::TokenConstant, constantParams,
                constantSchemas, constantInputs, constantResults, context));
  if (llvm::Error error = constant.validateSemanticValue({0xff, 0x01}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, constant.validateSemanticValue({0xff, 0x02}), "padding");
  expectError(test, constant.validateSemanticValue({0xff}), "byte count");

  const FamilyCapabilityParams sliceParams = FixedVectorSliceAlignMergeParams{
      IntegerWidthSet::get({IntegerWidth::I8}),
      FloatFormatSet{},
      64,
      8,
      1,
      ResolvedIndexWidthSet::get({ResolvedIndexWidth::I32})};
  constexpr std::array sliceSchemas = {OperationSchemaId::VectorExtract};
  constexpr std::array sliceInputs = {24U, 1U, 32U};
  constexpr std::array sliceResults = {8U};
  const FabricOpSemanticFieldRelation slice = take(
      test, resolveFabricOpSemanticFieldRelation(
                ImplementationFamilyId::FixedVectorSliceAlignMerge, sliceParams,
                sliceSchemas, sliceInputs, sliceResults, context));
  const auto *layout = slice.fixedVectorSliceAlignMergeLayout();
  require(test, layout != nullptr, "slice relation did not own its layout");
  std::vector<std::uint8_t> value((layout->encodedBitCount + 7) / 8, 0);
  setPackedField(value, layout->staticOffsetBitOffset, layout->offsetBitCount,
                 16);
  setPackedField(value, layout->sliceWidthBitOffset, layout->sliceWidthBitCount,
                 7);
  setPackedField(value, layout->dynamicStrideBitOffset,
                 layout->dynamicStrideBitCount, 16);
  expectError(test, slice.validateSemanticValue(value), "container");
}

void compatibilityQueryDerivesFromTheRelation() {
  const char *test = __func__;
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithAndI,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array inputWidths = {32U, 32U};
  constexpr std::array resultWidths = {32U};
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarIntegerLogic, params,
                     schemas, inputWidths, resultWidths, context));
  const bool requiresField =
      take(test, requiresSemanticConfigurationField(
                     ImplementationFamilyId::ScalarIntegerLogic, params,
                     schemas, 2, 1));
  require(test, requiresField == relation.hasConfigurationField(),
          "compatibility query disagrees with the sealed relation");
}

void finiteRelationKeysRemainCanonical() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  constexpr std::array schemas = {OperationSchemaId::ArithAndI,
                                  OperationSchemaId::ArithOrI};
  constexpr std::array inputWidths = {32U, 32U};
  constexpr std::array resultWidths = {32U};
  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarIntegerLogic, params,
                     schemas, inputWidths, resultWidths, context));
  auto points = relation.finiteBehaviorDomain();
  require(test, points.size() == 2,
          "finite logic relation did not expose both behavior keys");

  std::vector<std::vector<std::uint8_t>> keys;
  for (const FiniteImplementationFamilyBehaviorPoint &point : points) {
    require(test, point.semanticConfiguration.has_value(),
            "finite behavior point omitted its canonical key");
    keys.emplace_back(point.semanticConfiguration->bytes().begin(),
                      point.semanticConfiguration->bytes().end());
  }
  require(test, llvm::is_sorted(keys),
          "finite behavior keys are not canonically ordered");
  require(test, std::adjacent_find(keys.begin(), keys.end()) == keys.end(),
          "finite behavior keys are not unique");

  constexpr std::array<std::uint8_t, 72> expectedAnd = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   18,  'S', 'c', 'a', 'l', 'a', 'r', 'I', 'n', 't', 'e', 'g', 'e', 'r',
      'L', 'o', 'g', 'i', 'c', 0,   0,   0,   3,   'A', 'n', 'd'};
  const auto andPoint = llvm::find_if(points, [](const auto &point) {
    return point.representativeActor.schema == OperationSchemaId::ArithAndI;
  });
  require(test,
          andPoint != points.end() && andPoint->semanticConfiguration &&
              andPoint->semanticConfiguration->bytes().equals(expectedAnd),
          "finite behavior key escaped the registered canonical codec");
}

void physicalCapacityEliminatesRedundantBehavior() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = ScalarIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I32, IntegerWidth::I64}),
      IntegerPredicateSet::get({mlir::arith::CmpIPredicate::slt})};
  constexpr std::array schemas = {OperationSchemaId::ArithMinSI};
  constexpr std::array inputWidths = {32U, 32U};
  constexpr std::array resultWidths = {32U};
  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::ScalarIntegerCompareMinMax, params,
                     schemas, inputWidths, resultWidths, context));
  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::None,
          "narrow physical ports retained an unreachable width selector");
}

} // namespace

int main() {
  concreteCapabilitiesResolveOneSealedRelation();
  directDomainsRejectUnreachablePackedValues();
  compatibilityQueryDerivesFromTheRelation();
  finiteRelationKeysRemainCanonical();
  physicalCapacityEliminatesRedundantBehavior();
  return EXIT_SUCCESS;
}
