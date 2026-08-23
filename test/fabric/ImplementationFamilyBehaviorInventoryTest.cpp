#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilyBehaviorInternal.h"
#include "ImplementationFamilyFixedBehavior.h"
#include "ImplementationFamilyScalarFloatBehavior.h"
#include "ImplementationFamilyScalarFloatCompareBehavior.h"
#include "ImplementationFamilyScalarIntegerBehavior.h"
#include "ImplementationFamilySpecialMath.h"
#include "ImplementationFamilyVectorFloatBehavior.h"
#include "ImplementationFamilyVectorIntegerBehavior.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
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

template <typename T>
void expectError(const char *test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  require(test, !value, "expected failure");
  expectError(test, value.takeError(), expected);
}

void expectRelationError(const char *test,
                         llvm::Expected<FabricOpSemanticFieldRelation> relation,
                         llvm::StringRef expected) {
  require(test, !relation, "expected relation resolution failure");
  const std::string message = llvm::toString(relation.takeError());
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

void everyFamilyHasOneBehaviorRelationOwner() {
  const char *test = __func__;
  for (std::uint32_t ordinal = 0; ordinal != implementationFamilyCount();
       ++ordinal) {
    const auto family = static_cast<ImplementationFamilyId>(ordinal);
    const bool direct =
        family == ImplementationFamilyId::TokenConstant ||
        family == ImplementationFamilyId::FixedVectorSliceAlignMerge ||
        family == ImplementationFamilyId::FixedVectorShuffle;
    const unsigned ownerCount =
        static_cast<unsigned>(direct) +
        static_cast<unsigned>(detail::ownsFixedBehaviorRelation(family)) +
        static_cast<unsigned>(
            detail::ownsScalarFloatCompareBehaviorRelation(family)) +
        static_cast<unsigned>(detail::ownsScalarFloatBehaviorRelation(family)) +
        static_cast<unsigned>(
            detail::ownsFixedVectorFloatBehaviorRelation(family)) +
        static_cast<unsigned>(
            detail::ownsScalarIntegerBehaviorRelation(family)) +
        static_cast<unsigned>(
            detail::ownsFixedVectorIntegerBehaviorRelation(family)) +
        static_cast<unsigned>(detail::ownsControlBehaviorRelation(family)) +
        static_cast<unsigned>(
            detail::ownsScalarSpecialMathBehaviorRelation(family));
    require(test, ownerCount == 1,
            implementationFamilyKeyword(family).str() + " has " +
                std::to_string(ownerCount) + " behavior relation owners");
  }
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
  constexpr std::array constantInputs = {0U};
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
  constexpr std::array constantInputs = {0U};
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
  std::vector<std::uint8_t> staticSlice((layout->encodedBitCount + 7) / 8, 0);
  setPackedField(staticSlice, layout->sliceWidthBitOffset,
                 layout->sliceWidthBitCount, 7);
  if (llvm::Error error = slice.validateSemanticValue(staticSlice))
    fail(test, llvm::toString(std::move(error)));

  std::vector<std::uint8_t> value((layout->encodedBitCount + 7) / 8, 0);
  setPackedField(value, layout->staticOffsetBitOffset, layout->offsetBitCount,
                 16);
  setPackedField(value, layout->sliceWidthBitOffset, layout->sliceWidthBitCount,
                 7);
  setPackedField(value, layout->dynamicStrideBitOffset,
                 layout->dynamicStrideBitCount, 16);
  expectError(test, slice.validateSemanticValue(value), "container");
}

void directRelationsRejectForeignAndDuplicateSchemas() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams shuffleParams =
      FixedVectorShuffleParams{IntegerWidthSet::get({IntegerWidth::I16}),
                               FloatFormatSet{},
                               128,
                               128,
                               32,
                               5,
                               4};
  constexpr std::array inputs = {128U, 128U};
  constexpr std::array results = {128U};

  auto foreign = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorShuffle, shuffleParams,
      std::array{OperationSchemaId::VectorExtract}, inputs, results, context);
  require(test, !foreign, "Direct relation accepted a foreign schema");
  const std::string foreignMessage = llvm::toString(foreign.takeError());
  require(test, llvm::StringRef(foreignMessage).contains("not admitted"),
          "foreign schema reported an unrelated failure: " + foreignMessage);

  auto duplicate = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorShuffle, shuffleParams,
      std::array{OperationSchemaId::VectorShuffle,
                 OperationSchemaId::VectorShuffle},
      inputs, results, context);
  require(test, !duplicate, "Direct relation accepted a duplicate schema");
  const std::string duplicateMessage = llvm::toString(duplicate.takeError());
  require(test, llvm::StringRef(duplicateMessage).contains("duplicate"),
          "duplicate schema reported an unrelated failure: " +
              duplicateMessage);

  const FamilyCapabilityParams constantParams = PayloadCapacityParams{8};
  const FamilyCapabilityParams sliceParams =
      FixedVectorSliceAlignMergeParams{IntegerWidthSet::get({IntegerWidth::I8}),
                                       FloatFormatSet{},
                                       8,
                                       8,
                                       0,
                                       ResolvedIndexWidthSet{}};
  constexpr std::array sliceInputs = {8U, 8U};
  constexpr std::array emptySchemas = std::array<OperationSchemaId, 0>{};
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::TokenConstant, constantParams,
                          emptySchemas, std::array{1U}, std::array{8U},
                          context),
                      "no enabled");
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::FixedVectorSliceAlignMerge,
                          sliceParams, emptySchemas, sliceInputs,
                          std::array{8U}, context),
                      "no enabled");
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::FixedVectorShuffle,
                          shuffleParams, emptySchemas, inputs, results,
                          context),
                      "no enabled");
}

void directRelationsRequireAReachableWitness() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  constexpr std::array constantSchemas = {OperationSchemaId::DataflowConstant};
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::TokenConstant,
                          FamilyCapabilityParams{PayloadCapacityParams{8}},
                          constantSchemas, std::array<std::uint32_t, 0>{},
                          std::array{8U}, context),
                      "physical role");

  const FamilyCapabilityParams shuffleParams =
      FixedVectorShuffleParams{IntegerWidthSet::get({IntegerWidth::I8}),
                               FloatFormatSet{},
                               8,
                               8,
                               8,
                               2,
                               1};
  constexpr std::array shuffleSchemas = {OperationSchemaId::VectorShuffle};
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::FixedVectorShuffle,
                          shuffleParams, shuffleSchemas, std::array{8U},
                          std::array{8U}, context),
                      "physically reachable");
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::FixedVectorShuffle,
                          shuffleParams, shuffleSchemas, std::array{1U, 1U},
                          std::array{1U}, context),
                      "physically reachable");

  const FamilyCapabilityParams sliceParams =
      FixedVectorSliceAlignMergeParams{IntegerWidthSet::get({IntegerWidth::I8}),
                                       FloatFormatSet{},
                                       8,
                                       8,
                                       0,
                                       ResolvedIndexWidthSet{}};
  expectRelationError(test,
                      resolveFabricOpSemanticFieldRelation(
                          ImplementationFamilyId::FixedVectorSliceAlignMerge,
                          sliceParams,
                          std::array{OperationSchemaId::VectorExtract,
                                     OperationSchemaId::VectorInsert},
                          std::array{8U, 1U}, std::array{8U}, context),
                      "physically reachable");
}

void constantDirectProjectionPreservesRawBits() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::Builder builder(&context);
  constexpr std::array schemas = {OperationSchemaId::DataflowConstant};
  constexpr std::array inputs = {0U};

  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenConstant,
                     FamilyCapabilityParams{PayloadCapacityParams{16}}, schemas,
                     inputs, std::array{12U}, context));
  require(test, relation.directEncodedBitCount() == 12,
          "constant carrier ignored physical result narrowing");
  const auto project = [&](mlir::Type type, mlir::TypedAttr value) {
    const ::dataflow::CanonicalActorSchemaProjection actor{
        OperationSchemaId::DataflowConstant,
        builder.getFunctionType({builder.getNoneType()}, {type}),
        ::dataflow::ConstantValuePayload{value}};
    const loom::CanonicalSemanticBytes projected = take(
        test,
        relation.projectSemanticValue(actor, std::array<std::uint64_t, 1>{0},
                                      std::array<std::uint64_t, 1>{0}));
    return std::vector<std::uint8_t>(projected.bytes().begin(),
                                     projected.bytes().end());
  };

  require(test,
          project(builder.getI8Type(), builder.getI8IntegerAttr(0xa5)) ==
              std::vector<std::uint8_t>({0xa5, 0x00}),
          "scalar integer constant changed low-bit-first packing");

  const ::dataflow::CanonicalActorSchemaProjection mismatchedActor{
      OperationSchemaId::DataflowConstant,
      builder.getFunctionType({builder.getNoneType()}, {builder.getI8Type()}),
      ::dataflow::ConstantValuePayload{builder.getIntegerAttr(
          builder.getIntegerType(4), llvm::APInt(4, 0x5))}};
  expectError(test,
              relation.projectSemanticValue(mismatchedActor,
                                            std::array<std::uint64_t, 1>{0},
                                            std::array<std::uint64_t, 1>{0}),
              "does not match actor result type");

  const mlir::VectorType vector =
      mlir::VectorType::get({2}, builder.getIntegerType(4));
  const std::array denseValues = {llvm::APInt(4, 0x5), llvm::APInt(4, 0xa)};
  require(
      test,
      project(vector, mlir::DenseIntElementsAttr::get(vector, denseValues)) ==
          std::vector<std::uint8_t>({0xa5, 0x00}),
      "dense constant changed lane-zero-least-significant packing");

  const FabricOpSemanticFieldRelation floatRelation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenConstant,
                     FamilyCapabilityParams{PayloadCapacityParams{32}}, schemas,
                     inputs, std::array{32U}, context));
  const llvm::APInt nanBits(32, 0x7fc01234);
  const mlir::FloatAttr nan =
      mlir::FloatAttr::get(builder.getF32Type(),
                           llvm::APFloat(llvm::APFloat::IEEEsingle(), nanBits));
  const ::dataflow::CanonicalActorSchemaProjection nanActor{
      OperationSchemaId::DataflowConstant,
      builder.getFunctionType({builder.getNoneType()}, {builder.getF32Type()}),
      ::dataflow::ConstantValuePayload{nan}};
  const loom::CanonicalSemanticBytes projectedNan =
      take(test, floatRelation.projectSemanticValue(
                     nanActor, std::array<std::uint64_t, 1>{0},
                     std::array<std::uint64_t, 1>{0}));
  require(test,
          projectedNan.bytes().equals(
              std::array<std::uint8_t, 4>{0x34, 0x12, 0xc0, 0x7f}),
          "floating constant lost its NaN payload bits");

  const mlir::VectorType floatVector =
      mlir::VectorType::get({2}, builder.getF16Type());
  const std::array denseFloatValues = {
      llvm::APFloat(llvm::APFloat::IEEEhalf(), llvm::APInt(16, 0x8000)),
      llvm::APFloat(llvm::APFloat::IEEEhalf(), llvm::APInt(16, 0x7e55))};
  const mlir::DenseFPElementsAttr denseFloats =
      mlir::DenseFPElementsAttr::get(floatVector, denseFloatValues);
  const ::dataflow::CanonicalActorSchemaProjection denseFloatActor{
      OperationSchemaId::DataflowConstant,
      builder.getFunctionType({builder.getNoneType()}, {floatVector}),
      ::dataflow::ConstantValuePayload{denseFloats}};
  const loom::CanonicalSemanticBytes projectedDenseFloats =
      take(test, floatRelation.projectSemanticValue(
                     denseFloatActor, std::array<std::uint64_t, 1>{0},
                     std::array<std::uint64_t, 1>{0}));
  require(test,
          projectedDenseFloats.bytes().equals(
              std::array<std::uint8_t, 4>{0x00, 0x80, 0x55, 0x7e}),
          "dense floating constant lost lane order or special-value bits");
}

void zeroBitDirectCarrierCollapsesToNone() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      FixedVectorSliceAlignMergeParams{IntegerWidthSet::get({IntegerWidth::I1}),
                                       FloatFormatSet{},
                                       1,
                                       1,
                                       0,
                                       ResolvedIndexWidthSet{}};
  constexpr std::array schemas = {OperationSchemaId::VectorExtract};
  constexpr std::array inputs = {1U, 1U};
  constexpr std::array results = {1U};
  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::FixedVectorSliceAlignMerge, params,
                     schemas, inputs, results, context));
  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::None,
          "zero-bit singleton slice did not collapse to None");
  require(test, !relation.hasConfigurationField(),
          "zero-bit singleton slice exposed an ABI field");
  require(
      test,
      relation.finiteBehaviorDomain().size() == 1 &&
          !relation.finiteBehaviorDomain().front().semanticConfiguration &&
          relation.finiteBehaviorDomain().front().representativeActor.schema ==
              OperationSchemaId::VectorExtract,
      "zero-bit singleton slice lost its unique behavior witness");
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

void loopControlRelationOwnsTheReachableQuotient() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params = LoopStreamParams{
      IntegerWidthSet::get({IntegerWidth::I16, IntegerWidth::I32}),
      ::dataflow::StreamStepKind::Add,
      IntegerPredicateSet::get(
          {mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::ult})};
  constexpr std::array schemas = {OperationSchemaId::DataflowStream};
  constexpr std::array inputWidths = {32U, 32U, 32U};
  constexpr std::array resultWidths = {32U, 1U};
  const FabricOpSemanticFieldRelation relation =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::LoopStream, params, schemas,
                     inputWidths, resultWidths, context));
  require(test, relation.kind() == FabricOpSemanticFieldRelationKind::Finite,
          "loop stream did not resolve to a finite relation");
  require(test, relation.finiteBehaviorDomain().size() == 4,
          "loop stream quotient lost a width or predicate behavior");

  constexpr std::array narrowInputs = {16U, 16U, 16U};
  constexpr std::array narrowResults = {16U, 1U};
  const FabricOpSemanticFieldRelation narrow =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::LoopStream, params, schemas,
                     narrowInputs, narrowResults, context));
  require(test,
          narrow.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              narrow.finiteBehaviorDomain().size() == 2,
          "physical filtering retained an unreachable stream width");
  for (const auto &point : narrow.finiteBehaviorDomain()) {
    require(
        test,
        point.representativeActor.type.getInput(0).getIntOrFloatBitWidth() ==
            16,
        "narrow stream relation retained a wide witness");
    require(test, point.semanticConfiguration.has_value(),
            "non-singleton stream behavior omitted its key");
  }
}

void vectorAdaptersQuotientByElementWidthAndLaneCount() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams params =
      FixedVectorAdapterParams{IntegerWidthSet::get({IntegerWidth::I16}),
                               FloatFormatSet::get({FloatFormat::F16}), 64};

  constexpr std::array parallelizeSchemas = {
      OperationSchemaId::DataflowParallelize};
  constexpr std::array parallelizeInputs = {16U, 1U};
  constexpr std::array parallelizeResults = {64U, 4U, 1U};
  const FabricOpSemanticFieldRelation parallelize =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::FixedVectorParallelize, params,
                     parallelizeSchemas, parallelizeInputs, parallelizeResults,
                     context));
  require(test,
          parallelize.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              parallelize.finiteBehaviorDomain().size() == 4,
          "parallelize did not collapse equal-width integer and float actors");

  constexpr std::array serializeSchemas = {
      OperationSchemaId::DataflowSerialize};
  constexpr std::array serializeInputs = {64U, 4U, 1U};
  constexpr std::array serializeResults = {16U, 1U};
  const FabricOpSemanticFieldRelation serialize = take(
      test, resolveFabricOpSemanticFieldRelation(
                ImplementationFamilyId::FixedVectorSerialize, params,
                serializeSchemas, serializeInputs, serializeResults, context));
  require(test,
          serialize.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              serialize.finiteBehaviorDomain().size() == 4,
          "serialize did not own every reachable lane count");

  const FamilyCapabilityParams integerOnly = FixedVectorAdapterParams{
      IntegerWidthSet::get({IntegerWidth::I16}), FloatFormatSet{}, 64};
  const FabricOpSemanticFieldRelation laneOnly =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::FixedVectorParallelize,
                     integerOnly, parallelizeSchemas, parallelizeInputs,
                     parallelizeResults, context));
  require(test,
          laneOnly.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              laneOnly.finiteBehaviorDomain().size() == 4,
          "single-element adapter omitted its lane-count quotient");
}

void routedTokensOwnObservablePhysicalLaneImages() {
  const char *test = __func__;
  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  const FamilyCapabilityParams params = RoutedTokenParams{32, 3};

  constexpr std::array syncSchemas = {OperationSchemaId::DataflowSync};
  constexpr std::array syncWidths = {32U, 32U, 32U};
  const FabricOpSemanticFieldRelation sync =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenSync, params, syncSchemas,
                     syncWidths, syncWidths, context));
  require(test,
          sync.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              sync.finiteBehaviorDomain().size() == 3,
          "sync did not quotient symmetric lane subsets");

  const auto i1 = mlir::IntegerType::get(&context, 1);
  const auto i32 = mlir::IntegerType::get(&context, 32);
  const ::dataflow::CanonicalActorSchemaProjection twoLaneSync{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {i32, i32}, {i32, i32}),
      ::dataflow::NoPayload{}};
  constexpr std::array<std::uint64_t, 2> canonicalSync = {0, 1};
  const ::loom::CanonicalSemanticBytes syncKey =
      take(test, sync.projectSemanticValue(twoLaneSync, canonicalSync,
                                           canonicalSync));
  if (llvm::Error error = sync.validateSemanticValue(syncKey.bytes()))
    fail(test, llvm::toString(std::move(error)));
  constexpr std::array<std::uint8_t, 80> expectedSyncKey = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p',
      'e', 'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i',
      'o', 'r', '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,
      0,   0,   0,   0,   9,   'T', 'o', 'k', 'e', 'n', 'S', 'y', 'n', 'c',
      0,   0,   0,   0,   0,   0,   0,   2,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   1};
  require(test, syncKey.bytes().equals(expectedSyncKey),
          "sync embedding escaped the canonical literal codec");

  constexpr std::array pointerWidths = {64U, 64U};
  const FabricOpSemanticFieldRelation pointerSync =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenSync,
                     FamilyCapabilityParams(RoutedTokenParams{64, 2}),
                     syncSchemas, pointerWidths, pointerWidths, context));
  const auto pointer = mlir::LLVM::LLVMPointerType::get(&context);
  const ::dataflow::CanonicalActorSchemaProjection pointerActor{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {pointer}, {pointer}),
      ::dataflow::NoPayload{}};
  constexpr std::array<std::uint64_t, 1> pointerPorts = {0};
  auto missingPointerLayout = pointerSync.projectSemanticValue(
      pointerActor, pointerPorts, pointerPorts, ResolvedIndexWidth::I64);
  expectError(test, std::move(missingPointerLayout), "exact pointer layout");
  const ::loom::PointerLayout pointerLayout{
      0, 64, 64, ::loom::PointerLayoutKind::StableIntegral};
  take(test, pointerSync.projectSemanticValue(
                 pointerActor, pointerPorts, pointerPorts,
                 ResolvedIndexWidth::I64, &pointerLayout));

  constexpr std::array<std::uint64_t, 2> noncanonical = {0, 2};
  auto redundant =
      sync.projectSemanticValue(twoLaneSync, noncanonical, noncanonical);
  require(test, !redundant,
          "sync accepted a redundant symmetric-lane embedding");
  llvm::consumeError(redundant.takeError());
  constexpr std::array<std::uint64_t, 2> reversed = {1, 0};
  auto invalid = sync.projectSemanticValue(twoLaneSync, reversed, reversed);
  require(test, !invalid, "sync accepted a reversed physical lane image");
  const std::string invalidMessage = llvm::toString(invalid.takeError());
  require(test, llvm::StringRef(invalidMessage).contains("canonical"),
          "sync rejected a reversed lane image for an unrelated reason");

  constexpr std::array muxSchemas = {OperationSchemaId::DataflowMux};
  constexpr std::array muxInputs = {32U, 32U, 32U, 32U};
  constexpr std::array muxResults = {32U};
  const FabricOpSemanticFieldRelation mux =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenMux, params, muxSchemas,
                     muxInputs, muxResults, context));
  require(test,
          mux.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              mux.finiteBehaviorDomain().size() == 2,
          "mux exposed " + std::to_string(mux.finiteBehaviorDomain().size()) +
              " rather than two canonical lane-count embeddings");
  const ::dataflow::CanonicalActorSchemaProjection twoChoiceMux{
      OperationSchemaId::DataflowMux,
      mlir::FunctionType::get(&context, {i1, i32, i32}, {i32}),
      ::dataflow::NoPayload{}};
  constexpr std::array<std::uint64_t, 3> canonicalMux = {0, 1, 2};
  constexpr std::array<std::uint64_t, 1> fixedResult = {0};
  take(test, mux.projectSemanticValue(twoChoiceMux, canonicalMux, fixedResult));
  auto redundantMux =
      mux.projectSemanticValue(twoChoiceMux, {0, 1, 3}, fixedResult);
  require(test, !redundantMux,
          "mux accepted a redundant symmetric-lane embedding");
  llvm::consumeError(redundantMux.takeError());

  constexpr std::array demuxSchemas = {OperationSchemaId::DataflowDemux};
  constexpr std::array demuxInputs = {32U, 32U};
  constexpr std::array demuxResults = {32U, 32U, 32U};
  const FabricOpSemanticFieldRelation demux =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenDemux, params, demuxSchemas,
                     demuxInputs, demuxResults, context));
  require(test,
          demux.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              demux.finiteBehaviorDomain().size() == 2,
          "demux exposed " +
              std::to_string(demux.finiteBehaviorDomain().size()) +
              " rather than two canonical lane-count embeddings");
  const ::dataflow::CanonicalActorSchemaProjection twoChoiceDemux{
      OperationSchemaId::DataflowDemux,
      mlir::FunctionType::get(&context, {i1, i32}, {i32, i32}),
      ::dataflow::NoPayload{}};
  constexpr std::array<std::uint64_t, 2> fixedOperands = {0, 1};
  constexpr std::array<std::uint64_t, 2> canonicalDemux = {0, 1};
  take(test, demux.projectSemanticValue(twoChoiceDemux, fixedOperands,
                                        canonicalDemux));
  constexpr std::array indexDemuxInputs = {64U, 64U};
  constexpr std::array indexDemuxResults = {64U, 64U, 64U};
  const FabricOpSemanticFieldRelation indexDemux =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenDemux,
                     FamilyCapabilityParams(RoutedTokenParams{64, 3}),
                     demuxSchemas,
                     indexDemuxInputs, indexDemuxResults, context));
  const ::dataflow::CanonicalActorSchemaProjection twoChoiceIndexDemux{
      OperationSchemaId::DataflowDemux,
      mlir::FunctionType::get(&context,
                              {i1, mlir::IndexType::get(&context)},
                              {mlir::IndexType::get(&context),
                               mlir::IndexType::get(&context)}),
      ::dataflow::NoPayload{}};
  take(test, indexDemux.projectSemanticValue(
                 twoChoiceIndexDemux, fixedOperands, canonicalDemux,
                 ResolvedIndexWidth::I64));
  auto redundantDemux =
      demux.projectSemanticValue(twoChoiceDemux, fixedOperands, {0, 2});
  require(test, !redundantDemux,
          "demux accepted a redundant symmetric-lane embedding");
  llvm::consumeError(redundantDemux.takeError());

  constexpr std::array asymmetricWidths = {16U, 32U, 32U};
  const FabricOpSemanticFieldRelation asymmetric =
      take(test, resolveFabricOpSemanticFieldRelation(
                     ImplementationFamilyId::TokenSync, params, syncSchemas,
                     asymmetricWidths, asymmetricWidths, context));
  require(test,
          asymmetric.kind() == FabricOpSemanticFieldRelationKind::Finite &&
              asymmetric.finiteBehaviorDomain().size() == 5,
          "sync collapsed physically asymmetric lane classes");
  const auto i16 = mlir::IntegerType::get(&context, 16);
  const ::dataflow::CanonicalActorSchemaProjection narrowThenWide{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {i16, i32}, {i16, i32}),
      ::dataflow::NoPayload{}};
  const ::dataflow::CanonicalActorSchemaProjection wideThenNarrow{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {i32, i16}, {i32, i16}),
      ::dataflow::NoPayload{}};
  const ::loom::CanonicalSemanticBytes narrowWideKey = take(
      test, asymmetric.projectSemanticValue(narrowThenWide, {0, 1}, {0, 1}));
  const ::loom::CanonicalSemanticBytes wideNarrowKey = take(
      test, asymmetric.projectSemanticValue(wideThenNarrow, {1, 0}, {1, 0}));
  require(test, narrowWideKey.bytes().equals(wideNarrowKey.bytes()),
          "sync distinguished lane orders that activate the same hardware");
  const ::dataflow::CanonicalActorSchemaProjection narrowLane{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {i16}, {i16}), ::dataflow::NoPayload{}};
  const ::dataflow::CanonicalActorSchemaProjection wideLane{
      OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context, {i32}, {i32}), ::dataflow::NoPayload{}};
  const ::loom::CanonicalSemanticBytes narrowKey =
      take(test, asymmetric.projectSemanticValue(narrowLane, {0}, {0}));
  const ::loom::CanonicalSemanticBytes wideKey =
      take(test, asymmetric.projectSemanticValue(wideLane, {1}, {1}));
  require(test, !narrowKey.bytes().equals(wideKey.bytes()),
          "sync merged physically distinct active lane sets");
}

void singletonControlRelationsStillValidateTheirCapability() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  constexpr std::array streamSchemas = {OperationSchemaId::DataflowStream};
  constexpr std::array streamInputs = {16U, 16U, 16U};
  constexpr std::array streamResults = {16U, 1U};
  const FamilyCapabilityParams invalidStream = LoopStreamParams{
      IntegerWidthSet::get({IntegerWidth::I16}),
      static_cast<::dataflow::StreamStepKind>(255),
      IntegerPredicateSet::get({mlir::arith::CmpIPredicate::slt})};
  auto stream = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::LoopStream, invalidStream, streamSchemas,
      streamInputs, streamResults, context);
  require(test, !stream, "invalid singleton stream capability was accepted");
  const std::string streamError = llvm::toString(stream.takeError());
  require(test, llvm::StringRef(streamError).contains("step kind"),
          "invalid singleton stream reported an unrelated failure");

  constexpr std::array syncSchemas = {OperationSchemaId::DataflowSync};
  constexpr std::array syncWidths = {32U};
  const FamilyCapabilityParams invalidSync = RoutedTokenParams{32, 0};
  auto sync = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::TokenSync, invalidSync, syncSchemas, syncWidths,
      syncWidths, context);
  require(test, !sync, "zero-fan singleton sync capability was accepted");
  llvm::consumeError(sync.takeError());
}

} // namespace

int main() {
  everyFamilyHasOneBehaviorRelationOwner();
  concreteCapabilitiesResolveOneSealedRelation();
  directDomainsRejectUnreachablePackedValues();
  directRelationsRejectForeignAndDuplicateSchemas();
  directRelationsRequireAReachableWitness();
  constantDirectProjectionPreservesRawBits();
  zeroBitDirectCarrierCollapsesToNone();
  finiteRelationKeysRemainCanonical();
  physicalCapacityEliminatesRedundantBehavior();
  loopControlRelationOwnsTheReachableQuotient();
  vectorAdaptersQuotientByElementWidthAndLaneCount();
  routedTokensOwnObservablePhysicalLaneImages();
  singletonControlRelationsStillValidateTheirCapability();
  return EXIT_SUCCESS;
}
