//===- ImplementationFamilyTest.cpp - HSG family admission anchors -------===//
//
// Anchors the generated family registry and one discriminating example for
// each closed capability schema. Pair relations deliberately reject a pair
// whose source and destination both occur in other admitted pairs.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace fabric;
using namespace mlir;
using dataflow::OperationSchemaId;

namespace {

struct OpFixture {
  explicit OpFixture(MLIRContext &context)
      : builder(&context), loc(builder.getUnknownLoc()) {
    builder.setInsertionPointToEnd(&block);
  }

  Value poison(Type type) { return ub::PoisonOp::create(builder, loc, type); }

  Block block;
  OpBuilder builder;
  Location loc;
};

std::optional<dataflow::CanonicalActorSchemaProjection>
projectActor(Operation *op, bool &ok) {
  auto actor = llvm::dyn_cast<dataflow::CanonicalDataflowActorOpInterface>(op);
  if (!actor) {
    llvm::errs() << op->getName().getStringRef()
                 << " has no canonical actor interface\n";
    ok = false;
    return std::nullopt;
  }
  llvm::Expected<dataflow::CanonicalActorSchemaProjection> projection =
      actor.projectCanonicalActorSchemaProjection();
  if (!projection) {
    llvm::errs() << llvm::toString(projection.takeError()) << '\n';
    ok = false;
    return std::nullopt;
  }
  return *projection;
}

bool expectAdmission(ImplementationFamilyId family,
                     const FamilyCapabilityParams *params,
                     const dataflow::CanonicalActorSchemaProjection &actor,
                     bool admitted, llvm::StringRef semanticReason) {
  llvm::Error error =
      verifyImplementationFamilyAdmission(family, params, actor);
  if (admitted) {
    if (!error)
      return true;
    llvm::errs() << implementationFamilyKeyword(family)
                 << " rejected an admitted actor: "
                 << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (!error) {
    llvm::errs() << implementationFamilyKeyword(family)
                 << " admitted an actor that must fail " << semanticReason
                 << '\n';
    return false;
  }
  std::string message = llvm::toString(std::move(error));
  if (llvm::StringRef(message).contains(semanticReason))
    return true;
  llvm::errs() << implementationFamilyKeyword(family)
               << " reported an unspecific rejection: " << message << '\n';
  return false;
}

bool expectAdmissionAtIndexWidth(
    ImplementationFamilyId family, const FamilyCapabilityParams *params,
    const dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, bool admitted, llvm::StringRef semanticReason) {
  llvm::Error error =
      verifyImplementationFamilyAdmission(family, params, actor, indexBitWidth);
  if (admitted) {
    if (!error)
      return true;
    llvm::errs() << implementationFamilyKeyword(family)
                 << " rejected an admitted resolved-index actor: "
                 << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (!error) {
    llvm::errs() << implementationFamilyKeyword(family)
                 << " admitted a resolved-index actor that must fail "
                 << semanticReason << '\n';
    return false;
  }
  std::string message = llvm::toString(std::move(error));
  if (llvm::StringRef(message).contains(semanticReason))
    return true;
  llvm::errs() << implementationFamilyKeyword(family)
               << " reported an unspecific resolved-index rejection: "
               << message << '\n';
  return false;
}

bool expectAdmissionAtPointerLayout(
    ImplementationFamilyId family, const FamilyCapabilityParams *params,
    const dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, const loom::PointerLayout &pointerLayout,
    bool admitted, llvm::StringRef semanticReason) {
  llvm::Error error = verifyImplementationFamilyAdmission(
      family, params, actor, indexBitWidth, pointerLayout);
  if (admitted) {
    if (!error)
      return true;
    llvm::errs() << implementationFamilyKeyword(family)
                 << " rejected an admitted pointer actor: "
                 << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (!error) {
    llvm::errs() << implementationFamilyKeyword(family)
                 << " admitted a pointer actor that must fail "
                 << semanticReason << '\n';
    return false;
  }
  std::string message = llvm::toString(std::move(error));
  if (llvm::StringRef(message).contains(semanticReason))
    return true;
  llvm::errs() << implementationFamilyKeyword(family)
               << " reported an unspecific pointer rejection: " << message
               << '\n';
  return false;
}

bool checkDescriptorRelations() {
  bool ok = true;
  llvm::StringSet<> keywords;
  const std::uint32_t families = implementationFamilyCount();
  const std::uint32_t schemas = dataflow::operationSchemaCount();
  const std::uint32_t expectedFamilies =
      static_cast<std::uint32_t>(ImplementationFamilyId::FixedVectorShuffle) +
      1;
  if (families != expectedFamilies) {
    llvm::errs() << "the generated family registry is not dense, found "
                 << families << " entries through ordinal "
                 << expectedFamilies - 1 << '\n';
    ok = false;
  }

  for (std::uint32_t index = 0; index < families; ++index) {
    auto family = static_cast<ImplementationFamilyId>(index);
    const ImplementationFamilyDescriptor &descriptor =
        implementationFamily(family);
    auto [familyId, admittedSchemas, capabilitySchema, admissionProvider] =
        descriptor;
    if (familyId != family || admittedSchemas.empty()) {
      llvm::errs() << "family " << index << " has an invalid descriptor\n";
      ok = false;
    }
    for (OperationSchemaId schema : admittedSchemas) {
      if (static_cast<std::uint32_t>(schema) >= schemas ||
          !admitsOperationSchema(family, schema)) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " has an invalid generated member\n";
        ok = false;
      }
    }

    llvm::StringRef keyword = implementationFamilyKeyword(family);
    std::optional<ImplementationFamilyId> resolved =
        findImplementationFamily(keyword);
    if (keyword.empty() || !keywords.insert(keyword).second || !resolved ||
        *resolved != family ||
        capabilityParamsSchemaKeyword(capabilitySchema).empty() ||
        typedAdmissionProviderKeyword(admissionProvider).empty()) {
      llvm::errs() << "family " << index
                   << " does not round-trip through generated vocabularies\n";
      ok = false;
    }
  }
  return ok;
}

bool checkMembership() {
  bool ok = true;
  const std::optional<ImplementationFamilyId> scalarSaturating =
      findImplementationFamily("ScalarIntegerSaturatingAddSub");
  const std::optional<ImplementationFamilyId> vectorSaturating =
      findImplementationFamily("FixedVectorIntegerSaturatingAddSub");
  const std::optional<ImplementationFamilyId> scalarCountZeros =
      findImplementationFamily("ScalarIntegerCountZeros");
  const std::optional<ImplementationFamilyId> vectorCountZeros =
      findImplementationFamily("FixedVectorIntegerCountZeros");
  const std::optional<ImplementationFamilyId> vectorSlice =
      findImplementationFamily("FixedVectorSliceAlignMerge");
  const std::optional<ImplementationFamilyId> vectorShuffle =
      findImplementationFamily("FixedVectorShuffle");
  const std::array saturatingSpellings = {
      "llvm.intr.sadd.sat", "llvm.intr.uadd.sat", "llvm.intr.ssub.sat",
      "llvm.intr.usub.sat"};
  if (!scalarSaturating || !vectorSaturating) {
    llvm::errs() << "the integer saturation families are not registered\n";
    ok = false;
  }
  for (llvm::StringRef spelling : saturatingSpellings) {
    std::optional<OperationSchemaId> schema =
        dataflow::findOperationSchema(spelling);
    if (!schema ||
        (scalarSaturating &&
         !admitsOperationSchema(*scalarSaturating, *schema)) ||
        (vectorSaturating &&
         !admitsOperationSchema(*vectorSaturating, *schema)) ||
        admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                              schema.value_or(OperationSchemaId::ArithAddI))) {
      llvm::errs() << spelling
                   << " is not owned exclusively by the saturation families\n";
      ok = false;
    }
  }
  const std::array zeroCountSpellings = {"math.ctlz", "math.cttz",
                                         "llvm.intr.ctlz", "llvm.intr.cttz"};
  if (!scalarCountZeros || !vectorCountZeros) {
    llvm::errs() << "the integer zero-count families are not registered\n";
    ok = false;
  }
  for (llvm::StringRef spelling : zeroCountSpellings) {
    std::optional<OperationSchemaId> schema =
        dataflow::findOperationSchema(spelling);
    if (!schema ||
        (scalarCountZeros &&
         !admitsOperationSchema(*scalarCountZeros, *schema)) ||
        (vectorCountZeros &&
         !admitsOperationSchema(*vectorCountZeros, *schema)) ||
        admitsOperationSchema(ImplementationFamilyId::ScalarIntegerLogic,
                              schema.value_or(OperationSchemaId::ArithAndI))) {
      llvm::errs() << spelling
                   << " is not owned exclusively by the zero-count families\n";
      ok = false;
    }
  }
  if (!vectorSlice || !vectorShuffle ||
      !admitsOperationSchema(*vectorSlice, OperationSchemaId::VectorExtract) ||
      !admitsOperationSchema(*vectorSlice, OperationSchemaId::VectorInsert) ||
      admitsOperationSchema(*vectorSlice, OperationSchemaId::VectorShuffle) ||
      !admitsOperationSchema(*vectorShuffle,
                             OperationSchemaId::VectorShuffle) ||
      admitsOperationSchema(*vectorShuffle, OperationSchemaId::VectorExtract)) {
    llvm::errs() << "the fixed-vector structural families are incorrect\n";
    ok = false;
  }
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithAddI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithSubI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::LLVMGetElementPtr) ||
      admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                            OperationSchemaId::ArithMulI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarMathPow,
                             OperationSchemaId::MathPowF) ||
      admitsOperationSchema(ImplementationFamilyId::ScalarFloatMultiply,
                            OperationSchemaId::MathPowF) ||
      !admitsOperationSchema(ImplementationFamilyId::LoopStream,
                             OperationSchemaId::DataflowStream) ||
      admitsOperationSchema(ImplementationFamilyId::LoopStream,
                            OperationSchemaId::DataflowCarry)) {
    llvm::errs() << "a generated family/member relation is incorrect\n";
    ok = false;
  }
  if (findImplementationFamily("NoSuchFamily") ||
      findImplementationFamily("UniversalCompute")) {
    llvm::errs() << "an unregistered or speculative family is representable\n";
    ok = false;
  }
  return ok;
}

bool checkCapabilityCodec(MLIRContext &context) {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.roundingModes = RoundingModeSet::get(
      {arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward});
  FamilyCapabilityParams capability = ScalarFloatWidthCastParams{
      FloatFormatRelation::get({{FloatFormat::F16, FloatFormat::F32},
                                {FloatFormat::F32, FloatFormat::F64}}),
      behavior};
  DictionaryAttr encoded = getFamilyCapabilityParamsAttr(&context, capability);
  llvm::Expected<FamilyCapabilityParams> decoded = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarFloatWidthCast, encoded);
  if (!decoded) {
    llvm::errs() << "canonical typed hw_params failed to decode: "
                 << llvm::toString(decoded.takeError()) << '\n';
    return false;
  }
  if (getFamilyCapabilityParamsAttr(&context, *decoded) != encoded) {
    llvm::errs() << "canonical typed hw_params did not round-trip\n";
    return false;
  }

  const FamilyCapabilityParams conversionCapability =
      ScalarIntegerFloatConversionParams{IntegerFloatFormatRelation::get(
          {{IntegerWidth::I32, FloatFormat::F32}})};
  DictionaryAttr encodedConversion =
      getFamilyCapabilityParamsAttr(&context, conversionCapability);
  bool conversionCodecValid = true;
  if (encodedConversion.size() != 1 || !encodedConversion.get("format_pairs") ||
      encodedConversion.get("behavior")) {
    llvm::errs() << "conversion hw_params retained an orphan behavior field\n";
    conversionCodecValid = false;
  }
  auto decodedConversion = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarIntegerToFloat, encodedConversion);
  if (!decodedConversion ||
      getFamilyCapabilityParamsAttr(&context, *decodedConversion) !=
          encodedConversion) {
    if (!decodedConversion)
      llvm::errs() << llvm::toString(decodedConversion.takeError()) << '\n';
    llvm::errs() << "canonical conversion hw_params did not round-trip\n";
    conversionCodecValid = false;
  }

  OpBuilder conversionBuilder(&context);
  DictionaryAttr legacyConversion = conversionBuilder.getDictionaryAttr({
      conversionBuilder.getNamedAttr("format_pairs",
                                     encodedConversion.get("format_pairs")),
      conversionBuilder.getNamedAttr("behavior", encoded.get("behavior")),
  });
  auto rejectedLegacy = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarIntegerToFloat, legacyConversion);
  if (rejectedLegacy) {
    llvm::errs() << "legacy conversion behavior field was accepted\n";
    conversionCodecValid = false;
  } else if (!llvm::StringRef(llvm::toString(rejectedLegacy.takeError()))
                  .contains("unknown field 'behavior'")) {
    llvm::errs() << "legacy conversion behavior field was misclassified\n";
    conversionCodecValid = false;
  }
  if (!conversionCodecValid)
    return false;

  FamilyCapabilityParams pointerCapability = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I32, IntegerWidth::I64}),
      PointerFormatRelation::get(
          {{0, 64, 64, loom::PointerLayoutKind::StableIntegral}})};
  DictionaryAttr encodedPointer =
      getFamilyCapabilityParamsAttr(&context, pointerCapability);
  auto decodedPointer = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarIntegerAddSub, encodedPointer);
  if (!decodedPointer || getFamilyCapabilityParamsAttr(
                             &context, *decodedPointer) != encodedPointer) {
    if (!decodedPointer)
      llvm::errs() << llvm::toString(decodedPointer.takeError()) << '\n';
    llvm::errs() << "pointer-format capability did not round-trip\n";
    return false;
  }

  FamilyCapabilityParams vectorCapability = FixedVectorAdapterParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I32}),
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}), 128};
  DictionaryAttr encodedVector =
      getFamilyCapabilityParamsAttr(&context, vectorCapability);
  auto decodedVector = parseFamilyCapabilityParams(
      ImplementationFamilyId::FixedVectorPack, encodedVector);
  if (!decodedVector || getFamilyCapabilityParamsAttr(
                            &context, *decodedVector) != encodedVector) {
    if (!decodedVector)
      llvm::errs() << llvm::toString(decodedVector.takeError()) << '\n';
    llvm::errs() << "fixed-vector typed hw_params did not round-trip\n";
    return false;
  }

  FamilyCapabilityParams structuralCapability =
      FixedVectorSliceAlignMergeParams{
          IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}),
          FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}),
          130,
          64,
          3,
          ResolvedIndexWidthSet::get(
              {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})};
  DictionaryAttr encodedStructural =
      getFamilyCapabilityParamsAttr(&context, structuralCapability);
  auto decodedStructural = parseFamilyCapabilityParams(
      ImplementationFamilyId::FixedVectorSliceAlignMerge, encodedStructural);
  if (!decodedStructural ||
      getFamilyCapabilityParamsAttr(&context, *decodedStructural) !=
          encodedStructural) {
    if (!decodedStructural)
      llvm::errs() << llvm::toString(decodedStructural.takeError()) << '\n';
    llvm::errs() << "fixed-vector structural hw_params did not round-trip\n";
    return false;
  }

  FamilyCapabilityParams malformedShuffle =
      FixedVectorShuffleParams{IntegerWidthSet::get({IntegerWidth::I16}),
                               FloatFormatSet{},
                               64,
                               64,
                               65,
                               2,
                               1};
  auto rejectedShuffle = parseFamilyCapabilityParams(
      ImplementationFamilyId::FixedVectorShuffle,
      getFamilyCapabilityParamsAttr(&context, malformedShuffle));
  if (rejectedShuffle) {
    llvm::errs() << "accepted malformed fixed-vector shuffle hw_params\n";
    return false;
  }
  llvm::consumeError(rejectedShuffle.takeError());

  OpBuilder builder(&context);
  DictionaryAttr duplicateDomain =
      builder.getDictionaryAttr({builder.getNamedAttr(
          "integer_widths",
          builder.getArrayAttr({builder.getI32IntegerAttr(32),
                                builder.getI32IntegerAttr(32)}))});
  llvm::Expected<FamilyCapabilityParams> duplicateDomainResult =
      parseFamilyCapabilityParams(ImplementationFamilyId::ScalarIntegerAddSub,
                                  duplicateDomain);
  if (duplicateDomainResult) {
    llvm::errs() << "duplicate typed capability domain was accepted\n";
    return false;
  }
  if (!llvm::StringRef(llvm::toString(duplicateDomainResult.takeError()))
           .contains("duplicate")) {
    llvm::errs() << "duplicate typed capability domain was misclassified\n";
    return false;
  }

  ArrayAttr pair = builder.getArrayAttr(
      {builder.getI32IntegerAttr(8), builder.getI32IntegerAttr(32)});
  DictionaryAttr duplicateRelation = builder.getDictionaryAttr(
      {builder.getNamedAttr("width_pairs", builder.getArrayAttr({pair, pair})),
       builder.getNamedAttr(
           "resolved_index_widths",
           builder.getArrayAttr({builder.getI32IntegerAttr(64)}))});
  llvm::Expected<FamilyCapabilityParams> duplicateRelationResult =
      parseFamilyCapabilityParams(ImplementationFamilyId::ScalarIntegerCast,
                                  duplicateRelation);
  if (duplicateRelationResult) {
    llvm::errs() << "duplicate typed capability relation was accepted\n";
    return false;
  }
  if (!llvm::StringRef(llvm::toString(duplicateRelationResult.takeError()))
           .contains("duplicate")) {
    llvm::errs() << "duplicate typed capability relation was misclassified\n";
    return false;
  }
  return true;
}

bool checkIntegerAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  Type i7 = fixture.builder.getIntegerType(7);
  Type i8 = fixture.builder.getI8Type();
  Type i32 = fixture.builder.getI32Type();
  Operation *add8 = arith::AddIOp::create(
      fixture.builder, fixture.loc, fixture.poison(i8), fixture.poison(i8));
  Operation *saturatingAdd = LLVM::SAddSat::create(
      fixture.builder, fixture.loc, fixture.poison(i8), fixture.poison(i8));
  Operation *add7 = arith::AddIOp::create(
      fixture.builder, fixture.loc, fixture.poison(i7), fixture.poison(i7));
  Operation *signedLess = arith::CmpIOp::create(
      fixture.builder, fixture.loc, arith::CmpIPredicate::slt,
      fixture.poison(i32), fixture.poison(i32));
  Operation *equal = arith::CmpIOp::create(
      fixture.builder, fixture.loc, arith::CmpIPredicate::eq,
      fixture.poison(i32), fixture.poison(i32));

  FamilyCapabilityParams ordinary = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64})};
  FamilyCapabilityParams comparisons = ScalarIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I32}),
      IntegerPredicateSet::get({arith::CmpIPredicate::slt})};
  FamilyCapabilityParams wrongSchema =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}),
                        FloatBehaviorProfile::strictIEEE()};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams *params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, params, *projection, admitted, reason);
  };
  check(add8, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary, true, {});
  check(saturatingAdd, ImplementationFamilyId::ScalarIntegerSaturatingAddSub,
        &ordinary, true, {});
  check(saturatingAdd, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary,
        false, "actor schema is not admitted");
  check(add7, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary, false,
        "integer width");
  check(signedLess, ImplementationFamilyId::ScalarIntegerCompareMinMax,
        &comparisons, true, {});
  check(equal, ImplementationFamilyId::ScalarIntegerCompareMinMax, &comparisons,
        false, "integer predicate");
  check(add8, ImplementationFamilyId::ScalarIntegerAddSub, nullptr, false,
        "capability parameters");
  check(add8, ImplementationFamilyId::ScalarIntegerAddSub, &wrongSchema, false,
        "capability parameter schema");
  check(add8, ImplementationFamilyId::ScalarFloatAddSub, &ordinary, false,
        "actor schema is not admitted");

  std::optional<dataflow::CanonicalActorSchemaProjection> mismatch =
      projectActor(add8, ok);
  if (mismatch) {
    Type f32 = fixture.builder.getF32Type();
    mismatch->type = fixture.builder.getFunctionType({f32, f32}, {f32});
    ok &=
        expectAdmission(ImplementationFamilyId::ScalarIntegerAddSub, &ordinary,
                        *mismatch, false, "requires a scalar signless integer");
  }
  return ok;
}

bool checkPointerAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  Type pointer = LLVM::LLVMPointerType::get(&context);
  Type i32 = fixture.builder.getI32Type();
  Type i64 = fixture.builder.getI64Type();
  SmallVector<LLVM::GEPArg, 1> indices{fixture.poison(i64)};
  Operation *gep = LLVM::GEPOp::create(fixture.builder, fixture.loc, pointer,
                                       i32, fixture.poison(pointer), indices,
                                       LLVM::GEPNoWrapFlags::none);

  bool ok = true;
  std::optional<dataflow::CanonicalActorSchemaProjection> projection =
      projectActor(gep, ok);
  if (!projection)
    return false;

  FamilyCapabilityParams integerOnly = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I32, IntegerWidth::I64})};
  FamilyCapabilityParams pointerCapable = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I32, IntegerWidth::I64}),
      PointerFormatRelation::get(
          {{0, 64, 64, loom::PointerLayoutKind::StableIntegral}})};
  const loom::PointerLayout exact{0, 64, 64,
                                  loom::PointerLayoutKind::StableIntegral};
  const loom::PointerLayout wrongWidth{0, 32, 32,
                                       loom::PointerLayoutKind::StableIntegral};

  ok &= expectAdmission(ImplementationFamilyId::ScalarIntegerAddSub,
                        &pointerCapable, *projection, false,
                        "exact pointer layout");
  ok &= expectAdmissionAtPointerLayout(
      ImplementationFamilyId::ScalarIntegerAddSub, &integerOnly, *projection,
      64, exact, false, "pointer format");
  ok &= expectAdmissionAtPointerLayout(
      ImplementationFamilyId::ScalarIntegerAddSub, &pointerCapable, *projection,
      64, wrongWidth, false, "pointer format");
  ok &= expectAdmissionAtPointerLayout(
      ImplementationFamilyId::ScalarIntegerAddSub, &pointerCapable, *projection,
      64, exact, true, {});
  return ok;
}

bool checkFloatingAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  Type f32 = fixture.builder.getF32Type();
  Type f64 = fixture.builder.getF64Type();
  Value lhs = fixture.poison(f32);
  Value rhs = fixture.poison(f32);
  Operation *plain = arith::AddFOp::create(fixture.builder, fixture.loc, lhs,
                                           rhs, arith::FastMathFlags::none,
                                           arith::RoundingModeAttr{});
  Operation *noNaNs = arith::AddFOp::create(fixture.builder, fixture.loc, lhs,
                                            rhs, arith::FastMathFlags::nnan,
                                            arith::RoundingModeAttr{});
  Operation *downward = arith::AddFOp::create(
      fixture.builder, fixture.loc, lhs, rhs, arith::FastMathFlags::none,
      arith::RoundingModeAttr::get(&context, arith::RoundingMode::downward));
  Operation *wide = arith::AddFOp::create(
      fixture.builder, fixture.loc, fixture.poison(f64), fixture.poison(f64),
      arith::FastMathFlags::none, arith::RoundingModeAttr{});
  Operation *orderedLess = arith::CmpFOp::create(
      fixture.builder, fixture.loc, arith::CmpFPredicate::OLT, lhs, rhs,
      arith::FastMathFlags::none);
  Operation *orderedEqual = arith::CmpFOp::create(
      fixture.builder, fixture.loc, arith::CmpFPredicate::OEQ, lhs, rhs,
      arith::FastMathFlags::none);
  Operation *minNum = arith::MinNumFOp::create(
      fixture.builder, fixture.loc, lhs, rhs, arith::FastMathFlags::none);

  FloatBehaviorProfile strict = FloatBehaviorProfile::strictIEEE();
  FloatBehaviorProfile relaxed = strict;
  relaxed.requiredFastMath = arith::FastMathFlags::nnan;
  FloatBehaviorProfile rounded = strict;
  rounded.roundingModes = RoundingModeSet::get(
      {arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward});
  FloatBehaviorProfile numberPreferred = strict;
  numberPreferred.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});

  FamilyCapabilityParams strictFloat =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), strict};
  FamilyCapabilityParams relaxedFloat =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), relaxed};
  FamilyCapabilityParams roundedFloat =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), rounded};
  FamilyCapabilityParams comparisons = ScalarFloatCompareMinMaxParams{
      FloatFormatSet::get({FloatFormat::F32}), strict,
      FloatPredicateSet::get({arith::CmpFPredicate::OLT})};
  FamilyCapabilityParams numberPreferredComparisons =
      ScalarFloatCompareMinMaxParams{
          FloatFormatSet::get({FloatFormat::F32}), numberPreferred,
          FloatPredicateSet::get({arith::CmpFPredicate::OLT})};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(plain, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, true,
        {});
  check(wide, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, false,
        "floating format");
  check(noNaNs, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, true,
        {});
  check(plain, ImplementationFamilyId::ScalarFloatAddSub, relaxedFloat, false,
        "fast-math behavior");
  check(noNaNs, ImplementationFamilyId::ScalarFloatAddSub, relaxedFloat, true,
        {});
  check(downward, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, false,
        "rounding behavior");
  check(downward, ImplementationFamilyId::ScalarFloatAddSub, roundedFloat, true,
        {});
  check(orderedLess, ImplementationFamilyId::ScalarFloatCompareMinMax,
        comparisons, true, {});
  check(orderedEqual, ImplementationFamilyId::ScalarFloatCompareMinMax,
        comparisons, false, "floating predicate");
  check(minNum, ImplementationFamilyId::ScalarFloatCompareMinMax, comparisons,
        false, "NaN behavior");
  check(minNum, ImplementationFamilyId::ScalarFloatCompareMinMax,
        numberPreferredComparisons, true, {});
  return ok;
}

bool checkCastRelations(MLIRContext &context) {
  OpFixture fixture(context);
  Type i8 = fixture.builder.getI8Type();
  Type i16 = fixture.builder.getI16Type();
  Type i32 = fixture.builder.getI32Type();
  Type i64 = fixture.builder.getI64Type();
  Type f16 = fixture.builder.getF16Type();
  Type f32 = fixture.builder.getF32Type();
  Type f64 = fixture.builder.getF64Type();

  Operation *i8ToI32 = arith::ExtSIOp::create(fixture.builder, fixture.loc, i32,
                                              fixture.poison(i8));
  Operation *i16ToI64 = arith::ExtSIOp::create(fixture.builder, fixture.loc,
                                               i64, fixture.poison(i16));
  Operation *i16ToI32 = arith::ExtSIOp::create(fixture.builder, fixture.loc,
                                               i32, fixture.poison(i16));
  Operation *f16ToF32 = arith::ExtFOp::create(fixture.builder, fixture.loc, f32,
                                              fixture.poison(f16));
  Operation *f32ToF64 = arith::ExtFOp::create(fixture.builder, fixture.loc, f64,
                                              fixture.poison(f32));
  Operation *f16ToF64 = arith::ExtFOp::create(fixture.builder, fixture.loc, f64,
                                              fixture.poison(f16));
  Operation *i32ToF32 = arith::SIToFPOp::create(fixture.builder, fixture.loc,
                                                f32, fixture.poison(i32));
  Operation *i64ToF64 = arith::SIToFPOp::create(fixture.builder, fixture.loc,
                                                f64, fixture.poison(i64));
  Operation *i32ToF64 = arith::SIToFPOp::create(fixture.builder, fixture.loc,
                                                f64, fixture.poison(i32));
  Operation *f64ToI64 = arith::FPToSIOp::create(fixture.builder, fixture.loc,
                                                i64, fixture.poison(f64));
  Operation *f64ToI32 = arith::FPToSIOp::create(fixture.builder, fixture.loc,
                                                i32, fixture.poison(f64));
  Operation *saturatingF32ToI16 = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptosi.sat.i16.f32"),
      ValueRange{fixture.poison(f32)});
  Operation *saturatingF16ToI16 = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptosi.sat.i16.f16"),
      ValueRange{fixture.poison(f16)});
  Operation *bitcast = arith::BitcastOp::create(fixture.builder, fixture.loc,
                                                f32, fixture.poison(i32));
  Operation *wideBitcast = arith::BitcastOp::create(
      fixture.builder, fixture.loc, f64, fixture.poison(i32));
  Operation *i16ToIndex = arith::IndexCastOp::create(
      fixture.builder, fixture.loc, fixture.builder.getIndexType(),
      fixture.poison(i16));

  FamilyCapabilityParams integerCasts =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32},
                                     {IntegerWidth::I16, IntegerWidth::I64},
                                     {IntegerWidth::I32, IntegerWidth::I16}}),
          ResolvedIndexWidthSet::get(
              {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})}};
  FamilyCapabilityParams floatCasts = ScalarFloatWidthCastParams{
      FloatFormatRelation::get({{FloatFormat::F16, FloatFormat::F32},
                                {FloatFormat::F32, FloatFormat::F64}}),
      FloatBehaviorProfile::strictIEEE()};
  IntegerFloatFormatRelation conversionPairs =
      IntegerFloatFormatRelation::get({{IntegerWidth::I16, FloatFormat::F32},
                                       {IntegerWidth::I32, FloatFormat::F32},
                                       {IntegerWidth::I64, FloatFormat::F64}});
  FamilyCapabilityParams conversions =
      ScalarIntegerFloatConversionParams{conversionPairs};
  FamilyCapabilityParams reinterpretation = ScalarBitReinterpretParams{
      IntegerWidthSet::get({IntegerWidth::I32}),
      FloatFormatSet::get({FloatFormat::F32, FloatFormat::F64})};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(i8ToI32, ImplementationFamilyId::ScalarIntegerCast, integerCasts, true,
        {});
  check(i16ToI64, ImplementationFamilyId::ScalarIntegerCast, integerCasts, true,
        {});
  check(i16ToI32, ImplementationFamilyId::ScalarIntegerCast, integerCasts,
        false, "integer cast relation");
  check(f16ToF32, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        true, {});
  check(f32ToF64, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        true, {});
  check(f16ToF64, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        false, "floating cast relation");
  check(i32ToF32, ImplementationFamilyId::ScalarIntegerToFloat, conversions,
        true, {});
  check(i64ToF64, ImplementationFamilyId::ScalarIntegerToFloat, conversions,
        true, {});
  check(i32ToF64, ImplementationFamilyId::ScalarIntegerToFloat, conversions,
        false, "integer and floating relation");
  check(f64ToI64, ImplementationFamilyId::ScalarFloatToInteger, conversions,
        true, {});
  check(f64ToI32, ImplementationFamilyId::ScalarFloatToInteger, conversions,
        false, "integer and floating relation");
  check(saturatingF32ToI16, ImplementationFamilyId::ScalarFloatToInteger,
        conversions, true, {});
  check(saturatingF16ToI16, ImplementationFamilyId::ScalarFloatToInteger,
        conversions, false, "integer and floating relation");
  check(bitcast, ImplementationFamilyId::ScalarBitReinterpret, reinterpretation,
        true, {});
  check(wideBitcast, ImplementationFamilyId::ScalarBitReinterpret,
        reinterpretation, false, "equal semantic width");

  if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
          projectActor(i16ToIndex, ok)) {
    ok &= expectAdmissionAtIndexWidth(ImplementationFamilyId::ScalarIntegerCast,
                                      &integerCasts, *projection, 64, true, {});
    ok &= expectAdmissionAtIndexWidth(ImplementationFamilyId::ScalarIntegerCast,
                                      &integerCasts, *projection, 32, false,
                                      "exact resolved index endpoint pair");
  }
  return ok;
}

bool checkLoopAndTokenAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  Type i7 = fixture.builder.getIntegerType(7);
  Type i32 = fixture.builder.getI32Type();
  auto stream = [&](Type type, dataflow::StreamStepKind step,
                    arith::CmpIPredicate predicate) -> Operation * {
    return dataflow::StreamOp::create(
        fixture.builder, fixture.loc, type, fixture.builder.getI1Type(),
        fixture.poison(type), fixture.poison(type), fixture.poison(type), step,
        predicate);
  };
  Operation *addStream =
      stream(i32, dataflow::StreamStepKind::Add, arith::CmpIPredicate::slt);
  Operation *subStream =
      stream(i32, dataflow::StreamStepKind::Sub, arith::CmpIPredicate::slt);
  Operation *equalStream =
      stream(i32, dataflow::StreamStepKind::Add, arith::CmpIPredicate::eq);
  Operation *narrowStream =
      stream(i7, dataflow::StreamStepKind::Add, arith::CmpIPredicate::slt);
  FamilyCapabilityParams streamParams = LoopStreamParams{
      IntegerWidthSet::get({IntegerWidth::I32}), dataflow::StreamStepKind::Add,
      IntegerPredicateSet::get({arith::CmpIPredicate::slt})};

  auto carry = [&](Type payload) -> Operation * {
    return dataflow::CarryOp::create(
        fixture.builder, fixture.loc, payload,
        fixture.poison(fixture.builder.getI1Type()), fixture.poison(payload),
        fixture.poison(payload));
  };
  Operation *scalarCarry = carry(i32);
  Type pointer = LLVM::LLVMPointerType::get(&context);
  Operation *pointerInvariant = dataflow::InvariantOp::create(
      fixture.builder, fixture.loc, pointer,
      fixture.poison(fixture.builder.getI1Type()), fixture.poison(pointer));
  Operation *memoryCarry =
      carry(MemRefType::get({4}, fixture.builder.getI32Type()));
  FamilyCapabilityParams tokenPlane = TokenPlaneParams{};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(addStream, ImplementationFamilyId::LoopStream, streamParams, true, {});
  check(subStream, ImplementationFamilyId::LoopStream, streamParams, false,
        "fixed stream step kind");
  check(equalStream, ImplementationFamilyId::LoopStream, streamParams, false,
        "continuation predicate");
  check(narrowStream, ImplementationFamilyId::LoopStream, streamParams, false,
        "integer width");
  check(scalarCarry, ImplementationFamilyId::LoopCarry, tokenPlane, true, {});
  check(memoryCarry, ImplementationFamilyId::LoopCarry, tokenPlane, false,
        "token-plane payload");
  if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
          projectActor(pointerInvariant, ok)) {
    const loom::PointerLayout exact{0, 64, 64,
                                    loom::PointerLayoutKind::StableIntegral};
    ok &= expectAdmission(ImplementationFamilyId::LoopInvariant, &tokenPlane,
                          *projection, false, "exact pointer layout");
    ok &= expectAdmissionAtPointerLayout(ImplementationFamilyId::LoopInvariant,
                                         &tokenPlane, *projection, 64, exact,
                                         true, {});
  }
  return ok;
}

bool checkFixedVectorAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  Type i32 = fixture.builder.getI32Type();
  VectorType vector4 = VectorType::get({4}, i32);
  VectorType vector8 = VectorType::get({8}, i32);
  Operation *vectorAdd =
      arith::AddIOp::create(fixture.builder, fixture.loc,
                            fixture.poison(vector4), fixture.poison(vector4));
  Operation *wideVectorAdd =
      arith::AddIOp::create(fixture.builder, fixture.loc,
                            fixture.poison(vector8), fixture.poison(vector8));
  Operation *scalarAdd = arith::AddIOp::create(
      fixture.builder, fixture.loc, fixture.poison(i32), fixture.poison(i32));

  FamilyCapabilityParams vectorParams = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64}),
      128};
  FamilyCapabilityParams scalarParams = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64})};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(vectorAdd, ImplementationFamilyId::FixedVectorIntegerAddSub,
        vectorParams, true, {});
  check(scalarAdd, ImplementationFamilyId::FixedVectorIntegerAddSub,
        vectorParams, false, "fixed vector");
  check(vectorAdd, ImplementationFamilyId::ScalarIntegerAddSub, scalarParams,
        false, "scalar");
  check(wideVectorAdd, ImplementationFamilyId::FixedVectorIntegerAddSub,
        vectorParams, false, "payload capacity");
  return ok;
}

std::uint64_t readPackedBits(llvm::ArrayRef<std::uint8_t> bytes,
                             std::uint32_t offset, std::uint32_t count) {
  std::uint64_t value = 0;
  for (std::uint32_t bit = 0; bit != count; ++bit)
    value |=
        std::uint64_t((bytes[(offset + bit) / 8] >> ((offset + bit) % 8)) & 1U)
        << bit;
  return value;
}

bool checkFixedVectorStructuralAdmission(MLIRContext &context) {
  Type i16 = IntegerType::get(&context, 16);
  Type index = IndexType::get(&context);
  VectorType container = VectorType::get({4, 2}, i16);
  VectorType slice = VectorType::get({2}, i16);
  VectorType lhs = VectorType::get({2, 2}, i16);
  VectorType rhs = VectorType::get({1, 2}, i16);
  VectorType shuffled = VectorType::get({3, 2}, i16);
  VectorType indexContainer = VectorType::get({2}, index);
  VectorType indexLeft = VectorType::get({1}, index);
  VectorType indexResult = VectorType::get({2}, index);

  const FamilyCapabilityParams sliceParams = FixedVectorSliceAlignMergeParams{
      IntegerWidthSet::get(
          {IntegerWidth::I16, IntegerWidth::I32, IntegerWidth::I64}),
      FloatFormatSet{},
      130,
      64,
      3,
      ResolvedIndexWidthSet::get(
          {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})};
  const FamilyCapabilityParams shuffleParams = FixedVectorShuffleParams{
      IntegerWidthSet::get(
          {IntegerWidth::I16, IntegerWidth::I32, IntegerWidth::I64}),
      FloatFormatSet{},
      130,
      130,
      32,
      4,
      4};

  const dataflow::CanonicalActorSchemaProjection staticExtract{
      OperationSchemaId::VectorExtract,
      FunctionType::get(&context, {container}, {slice}),
      dataflow::VectorStaticPositionPayload{{2}}};
  const dataflow::CanonicalActorSchemaProjection dynamicExtract{
      OperationSchemaId::VectorExtract,
      FunctionType::get(&context, {container, index}, {slice}),
      dataflow::VectorStaticPositionPayload{{ShapedType::kDynamic}}};
  const dataflow::CanonicalActorSchemaProjection dynamicInsert{
      OperationSchemaId::VectorInsert,
      FunctionType::get(&context, {slice, container, index}, {container}),
      dataflow::VectorStaticPositionPayload{{ShapedType::kDynamic}}};
  const dataflow::CanonicalActorSchemaProjection shuffle{
      OperationSchemaId::VectorShuffle,
      FunctionType::get(&context, {lhs, rhs}, {shuffled}),
      dataflow::VectorShuffleMaskPayload{{0, 2, -1}}};
  const dataflow::CanonicalActorSchemaProjection indexStaticExtract{
      OperationSchemaId::VectorExtract,
      FunctionType::get(&context, {indexContainer}, {index}),
      dataflow::VectorStaticPositionPayload{{1}}};
  const dataflow::CanonicalActorSchemaProjection indexDynamicExtract{
      OperationSchemaId::VectorExtract,
      FunctionType::get(&context, {indexContainer, index}, {index}),
      dataflow::VectorStaticPositionPayload{{ShapedType::kDynamic}}};
  const dataflow::CanonicalActorSchemaProjection indexStaticInsert{
      OperationSchemaId::VectorInsert,
      FunctionType::get(&context, {index, indexContainer}, {indexContainer}),
      dataflow::VectorStaticPositionPayload{{0}}};
  const dataflow::CanonicalActorSchemaProjection indexShuffle{
      OperationSchemaId::VectorShuffle,
      FunctionType::get(&context, {indexLeft, indexLeft}, {indexResult}),
      dataflow::VectorShuffleMaskPayload{{1, 0}}};

  bool ok = true;
  ok &= expectAdmission(ImplementationFamilyId::FixedVectorSliceAlignMerge,
                        &sliceParams, staticExtract, true, {});
  ok &= expectAdmissionAtIndexWidth(
      ImplementationFamilyId::FixedVectorSliceAlignMerge, &sliceParams,
      dynamicExtract, 64, true, {});
  ok &= expectAdmissionAtIndexWidth(
      ImplementationFamilyId::FixedVectorSliceAlignMerge, &sliceParams,
      dynamicInsert, 32, true, {});
  ok &= expectAdmission(ImplementationFamilyId::FixedVectorShuffle,
                        &shuffleParams, shuffle, true, {});

  const std::array<std::uint64_t, 2> extractPorts = {0, 2};
  const std::array<std::uint64_t, 3> insertPorts = {0, 1, 2};
  const std::array<std::uint64_t, 1> resultPort = {0};
  if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
          ImplementationFamilyId::FixedVectorSliceAlignMerge, dynamicExtract,
          extractPorts, resultPort)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    ok = false;
  }
  if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
          ImplementationFamilyId::FixedVectorSliceAlignMerge, dynamicInsert,
          insertPorts, resultPort)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    ok = false;
  }

  constexpr std::array structuralSchemas = {OperationSchemaId::VectorExtract,
                                            OperationSchemaId::VectorInsert};
  constexpr std::array sliceInputWidths = {130U, 130U, 64U, 64U, 64U};
  constexpr std::array sliceResultWidths = {130U};
  constexpr std::array shuffleSchema = {OperationSchemaId::VectorShuffle};
  constexpr std::array shuffleInputWidths = {130U, 130U};
  constexpr std::array shuffleResultWidths = {130U};
  auto sliceRelation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorSliceAlignMerge, sliceParams,
      structuralSchemas, sliceInputWidths, sliceResultWidths, context);
  auto shuffleRelation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorShuffle, shuffleParams, shuffleSchema,
      shuffleInputWidths, shuffleResultWidths, context);
  if (!sliceRelation || !shuffleRelation) {
    if (!sliceRelation)
      llvm::errs() << llvm::toString(sliceRelation.takeError()) << '\n';
    if (!shuffleRelation)
      llvm::errs() << llvm::toString(shuffleRelation.takeError()) << '\n';
    return false;
  }
  const auto *sliceLayout = sliceRelation->fixedVectorSliceAlignMergeLayout();
  const auto *shuffleLayout = shuffleRelation->fixedVectorShuffleLayout();
  if (sliceRelation->kind() != FabricOpSemanticFieldRelationKind::Direct ||
      shuffleRelation->kind() != FabricOpSemanticFieldRelationKind::Direct ||
      !sliceLayout || !shuffleLayout) {
    llvm::errs() << "fixed-vector structural relation lost its direct layout\n";
    return false;
  }

  auto staticConfiguration = sliceRelation->projectSemanticValue(
      staticExtract, std::array<std::uint64_t, 1>{0}, resultPort,
      ResolvedIndexWidth::I64);
  auto dynamicConfiguration = sliceRelation->projectSemanticValue(
      dynamicExtract, extractPorts, resultPort, ResolvedIndexWidth::I64);
  auto shuffleConfiguration = shuffleRelation->projectSemanticValue(
      shuffle, std::array<std::uint64_t, 2>{0, 1}, resultPort);
  auto indexStaticConfiguration = sliceRelation->projectSemanticValue(
      indexStaticExtract, std::array<std::uint64_t, 1>{0}, resultPort,
      ResolvedIndexWidth::I32);
  auto indexDynamicConfiguration = sliceRelation->projectSemanticValue(
      indexDynamicExtract, extractPorts, resultPort, ResolvedIndexWidth::I32);
  auto indexInsertConfiguration = sliceRelation->projectSemanticValue(
      indexStaticInsert, std::array<std::uint64_t, 2>{0, 1}, resultPort,
      ResolvedIndexWidth::I32);
  auto indexShuffleConfiguration = shuffleRelation->projectSemanticValue(
      indexShuffle, std::array<std::uint64_t, 2>{0, 1}, resultPort,
      ResolvedIndexWidth::I32);
  if (!staticConfiguration || !dynamicConfiguration || !shuffleConfiguration) {
    if (!staticConfiguration)
      llvm::errs() << llvm::toString(staticConfiguration.takeError()) << '\n';
    if (!dynamicConfiguration)
      llvm::errs() << llvm::toString(dynamicConfiguration.takeError()) << '\n';
    if (!shuffleConfiguration)
      llvm::errs() << llvm::toString(shuffleConfiguration.takeError()) << '\n';
    return false;
  }
  bool indexConfigurationsValid = true;
  const auto checkConfiguration = [&](auto &configuration) {
    if (configuration)
      return;
    llvm::errs() << llvm::toString(configuration.takeError()) << '\n';
    indexConfigurationsValid = false;
  };
  checkConfiguration(indexStaticConfiguration);
  checkConfiguration(indexDynamicConfiguration);
  checkConfiguration(indexInsertConfiguration);
  checkConfiguration(indexShuffleConfiguration);
  if (!indexConfigurationsValid)
    return false;

  const auto staticBytes = staticConfiguration->bytes();
  const auto dynamicBytes = dynamicConfiguration->bytes();
  const auto shuffleBytes = shuffleConfiguration->bytes();
  const auto indexStaticBytes = indexStaticConfiguration->bytes();
  const auto indexDynamicBytes = indexDynamicConfiguration->bytes();
  const auto indexInsertBytes = indexInsertConfiguration->bytes();
  const auto indexShuffleBytes = indexShuffleConfiguration->bytes();
  ok &= readPackedBits(staticBytes, sliceLayout->modeBitOffset, 1) == 0;
  ok &= readPackedBits(staticBytes, sliceLayout->staticOffsetBitOffset,
                       sliceLayout->offsetBitCount) == 64;
  ok &= readPackedBits(staticBytes, sliceLayout->sliceWidthBitOffset,
                       sliceLayout->sliceWidthBitCount) == 31;
  ok &= readPackedBits(dynamicBytes, sliceLayout->dynamicStrideBitOffset,
                       sliceLayout->dynamicStrideBitCount) == 32;
  ok &= readPackedBits(indexStaticBytes, sliceLayout->staticOffsetBitOffset,
                       sliceLayout->offsetBitCount) == 32;
  ok &= readPackedBits(indexStaticBytes, sliceLayout->sliceWidthBitOffset,
                       sliceLayout->sliceWidthBitCount) == 31;
  ok &= readPackedBits(indexDynamicBytes, sliceLayout->dynamicStrideBitOffset,
                       sliceLayout->dynamicStrideBitCount) == 32;
  ok &= readPackedBits(indexInsertBytes, sliceLayout->modeBitOffset, 1) == 1;
  ok &= readPackedBits(indexShuffleBytes, shuffleLayout->blockWidthBitOffset,
                       shuffleLayout->blockWidthBitCount) == 31;
  const FamilyCapabilityParams powerOfTwoParams =
      FixedVectorSliceAlignMergeParams{
          IntegerWidthSet::get({IntegerWidth::I16}),
          FloatFormatSet{},
          128,
          64,
          1,
          ResolvedIndexWidthSet::get({ResolvedIndexWidth::I64})};
  constexpr std::array powerOfTwoInputWidths = {128U, 128U, 64U};
  constexpr std::array powerOfTwoResultWidths = {128U};
  auto powerOfTwoRelation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorSliceAlignMerge, powerOfTwoParams,
      structuralSchemas, powerOfTwoInputWidths, powerOfTwoResultWidths,
      context);
  if (!powerOfTwoRelation ||
      !powerOfTwoRelation->fixedVectorSliceAlignMergeLayout()) {
    if (!powerOfTwoRelation)
      llvm::errs() << llvm::toString(powerOfTwoRelation.takeError()) << '\n';
    return false;
  }
  const auto *powerOfTwoLayout =
      powerOfTwoRelation->fixedVectorSliceAlignMergeLayout();
  ok &= powerOfTwoLayout->offsetBitCount == 7 &&
        powerOfTwoLayout->dynamicStrideBitCount == 8;
  ok &= readPackedBits(shuffleBytes, shuffleLayout->blockWidthBitOffset,
                       shuffleLayout->blockWidthBitCount) == 31;
  ok &= readPackedBits(shuffleBytes, shuffleLayout->leftBlockCountBitOffset,
                       shuffleLayout->blockCountBitCount) == 1;
  ok &= readPackedBits(shuffleBytes, shuffleLayout->resultBlockCountBitOffset,
                       shuffleLayout->resultBlockCountBitCount) == 2;
  ok &= readPackedBits(shuffleBytes, shuffleLayout->selectorBitOffset,
                       shuffleLayout->selectorBitCount) == 0;
  ok &= readPackedBits(shuffleBytes,
                       shuffleLayout->selectorBitOffset +
                           shuffleLayout->selectorBitCount,
                       shuffleLayout->selectorBitCount) == 2;
  ok &= readPackedBits(shuffleBytes,
                       shuffleLayout->selectorBitOffset +
                           2 * shuffleLayout->selectorBitCount,
                       shuffleLayout->selectorBitCount) == 0;
  if (!ok)
    llvm::errs() << "fixed-vector structural projection is incorrect\n";
  return ok;
}

bool checkAdapterAndTokenAdmission(MLIRContext &context) {
  OpFixture fixture(context);
  VectorType vector4 = VectorType::get({4}, fixture.builder.getF32Type());
  VectorType vector2 = VectorType::get({2}, fixture.builder.getF64Type());
  Type i128 = fixture.builder.getIntegerType(128);
  Operation *pack4 = dataflow::PackOp::create(fixture.builder, fixture.loc,
                                              i128, fixture.poison(vector4));
  Operation *pack2 = dataflow::PackOp::create(fixture.builder, fixture.loc,
                                              i128, fixture.poison(vector2));

  FamilyCapabilityParams adapter = FixedVectorAdapterParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64}),
      FloatFormatSet::get(
          {FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32}),
      128};
  FamilyCapabilityParams routed = RoutedTokenParams{128, 4};

  Value selector = fixture.poison(fixture.builder.getI1Type());
  Value lane0 = fixture.poison(fixture.builder.getI32Type());
  Value lane1 = fixture.poison(fixture.builder.getI32Type());
  Operation *mux = dataflow::MuxOp::create(fixture.builder, fixture.loc,
                                           fixture.builder.getI32Type(),
                                           selector, ValueRange{lane0, lane1});

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(pack4, ImplementationFamilyId::FixedVectorPack, adapter, true, {});
  check(pack2, ImplementationFamilyId::FixedVectorPack, adapter, false,
        "element type");
  check(mux, ImplementationFamilyId::TokenMux, routed, true, {});
  return ok;
}

bool checkPortCorrespondenceEnumeration(MLIRContext &context) {
  Type i32 = IntegerType::get(&context, 32);
  Type i1 = IntegerType::get(&context, 1);
  const auto projection = [&](OperationSchemaId schema,
                              llvm::ArrayRef<Type> inputs,
                              llvm::ArrayRef<Type> results) {
    return dataflow::CanonicalActorSchemaProjection{
        schema, FunctionType::get(&context, inputs, results),
        dataflow::NoPayload{}};
  };
  const auto countDomain =
      [&](ImplementationFamilyId family,
          const dataflow::CanonicalActorSchemaProjection &actor,
          llvm::ArrayRef<std::uint64_t> inputs,
          llvm::ArrayRef<std::uint64_t> results) -> std::optional<unsigned> {
    unsigned count = 0;
    if (llvm::Error error = forEachImplementationFamilyPortCorrespondence(
            family, actor, inputs, results,
            [&](llvm::ArrayRef<std::uint64_t> operandPorts,
                llvm::ArrayRef<std::uint64_t> resultPorts)
                -> llvm::Expected<bool> {
              if (llvm::Error invalid =
                      verifyImplementationFamilyPortCorrespondence(
                          family, actor, operandPorts, resultPorts))
                return std::move(invalid);
              ++count;
              return true;
            })) {
      llvm::errs() << llvm::toString(std::move(error)) << '\n';
      return std::nullopt;
    }
    return count;
  };

  const std::array<Type, 2> syncTypes = {i32, i32};
  const auto sync =
      projection(OperationSchemaId::DataflowSync, syncTypes, syncTypes);
  const std::array<std::uint64_t, 3> syncPorts = {0, 2, 4};
  const auto syncCount = countDomain(ImplementationFamilyId::TokenSync, sync,
                                     syncPorts, syncPorts);

  const std::array<Type, 3> muxInputs = {i1, i32, i32};
  const std::array<Type, 1> muxResults = {i32};
  const auto mux =
      projection(OperationSchemaId::DataflowMux, muxInputs, muxResults);
  const std::array<std::uint64_t, 4> muxInputPorts = {0, 2, 4, 6};
  const std::array<std::uint64_t, 2> muxResultPorts = {0, 1};
  const auto muxCount = countDomain(ImplementationFamilyId::TokenMux, mux,
                                    muxInputPorts, muxResultPorts);

  const std::array<Type, 2> demuxInputs = {i1, i32};
  const std::array<Type, 3> demuxResults = {i32, i32, i32};
  const auto demux =
      projection(OperationSchemaId::DataflowDemux, demuxInputs, demuxResults);
  const std::array<std::uint64_t, 2> demuxInputPorts = {0, 1};
  const std::array<std::uint64_t, 4> demuxResultPorts = {1, 3, 5, 7};
  const auto demuxCount = countDomain(ImplementationFamilyId::TokenDemux, demux,
                                      demuxInputPorts, demuxResultPorts);

  const std::array<Type, 2> subtractInputs = {i32, i32};
  const std::array<Type, 1> subtractResults = {i32};
  const auto subtract =
      projection(OperationSchemaId::ArithSubI, subtractInputs, subtractResults);
  const std::array<std::uint64_t, 3> subtractInputPorts = {0, 1, 2};
  const std::array<std::uint64_t, 2> subtractResultPorts = {0, 1};
  const auto subtractCount =
      countDomain(ImplementationFamilyId::ScalarIntegerAddSub, subtract,
                  subtractInputPorts, subtractResultPorts);

  if (syncCount != 3 || muxCount != 3 || demuxCount != 4 ||
      subtractCount != 1) {
    llvm::errs() << "implementation-family port domain has the wrong size\n";
    return false;
  }
  llvm::Error reversed = verifyImplementationFamilyPortCorrespondence(
      ImplementationFamilyId::TokenSync, sync, {2, 0}, {2, 0});
  if (!reversed) {
    llvm::errs() << "port-domain point query accepted a reversed lane image\n";
    return false;
  }
  llvm::consumeError(std::move(reversed));
  return true;
}

bool checkScalarFloatFmaBehaviorDomain(MLIRContext &context) {
  const FamilyCapabilityParams params = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16,
                           FloatFormat::F32, FloatFormat::F64}),
      FloatBehaviorProfile::strictIEEE()};
  constexpr std::array enabled = {OperationSchemaId::MathFma};
  constexpr std::array inputWidths = {64U, 64U, 64U};
  constexpr std::array resultWidths = {64U};
  auto relation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::ScalarFloatFma, params, enabled, inputWidths,
      resultWidths, context);
  if (!relation) {
    llvm::errs() << "scalar FMA behavior domain did not resolve: "
                 << llvm::toString(relation.takeError()) << '\n';
    return false;
  }
  const auto domain = relation->finiteBehaviorDomain();
  if (relation->kind() != FabricOpSemanticFieldRelationKind::Finite ||
      domain.size() != 4) {
    llvm::errs() << "scalar FMA behavior domain has " << domain.size()
                 << " points instead of four formats\n";
    return false;
  }

  bool sawF16 = false;
  bool sawBF16 = false;
  bool sawF32 = false;
  bool sawF64 = false;
  std::vector<std::vector<std::uint8_t>> semanticValues;
  for (const auto &point : domain) {
    if (!point.semanticConfiguration) {
      llvm::errs() << "configured scalar FMA behavior has no semantic value\n";
      return false;
    }
    semanticValues.emplace_back(point.semanticConfiguration->bytes().begin(),
                                point.semanticConfiguration->bytes().end());
    if (point.representativeActor.schema != OperationSchemaId::MathFma ||
        point.representativeActor.type.getNumInputs() != 3 ||
        point.representativeActor.type.getNumResults() != 1) {
      llvm::errs() << "scalar FMA behavior has the wrong actor shape\n";
      return false;
    }
    Type type = point.representativeActor.type.getInput(0);
    sawF16 |= isa<Float16Type>(type);
    sawBF16 |= isa<BFloat16Type>(type);
    sawF32 |= isa<Float32Type>(type);
    sawF64 |= isa<Float64Type>(type);
  }
  llvm::sort(semanticValues);
  const bool unique =
      std::adjacent_find(semanticValues.begin(), semanticValues.end()) ==
      semanticValues.end();
  if (!sawF16 || !sawBF16 || !sawF32 || !sawF64 || !unique) {
    llvm::errs() << "scalar FMA formats did not remain semantically distinct\n";
    return false;
  }
  return true;
}

bool checkIntegerLogicBehaviorDomains(MLIRContext &context) {
  constexpr std::array enabled = {
      OperationSchemaId::ArithAndI, OperationSchemaId::ArithOrI,
      OperationSchemaId::ArithXOrI, OperationSchemaId::LLVMOrDisjoint};
  const auto check = [&](ImplementationFamilyId family,
                         const FamilyCapabilityParams &params,
                         llvm::ArrayRef<std::uint32_t> inputWidths,
                         llvm::ArrayRef<std::uint32_t> resultWidths) {
    auto relation = resolveFabricOpSemanticFieldRelation(
        family, params, enabled, inputWidths, resultWidths, context);
    if (!relation) {
      llvm::errs() << implementationFamilyKeyword(family)
                   << " behavior domain did not resolve: "
                   << llvm::toString(relation.takeError()) << '\n';
      return false;
    }
    const auto domain = relation->finiteBehaviorDomain();
    if (relation->kind() != FabricOpSemanticFieldRelationKind::Finite ||
        domain.size() != 3) {
      llvm::errs() << implementationFamilyKeyword(family)
                   << " did not collapse to AND/OR/XOR behaviors\n";
      return false;
    }
    llvm::SmallDenseSet<OperationSchemaId, 4> projectedSchemas;
    std::vector<std::vector<std::uint8_t>> semanticValues;
    for (const auto &point : domain) {
      if (!point.semanticConfiguration) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " logic behavior has no schema selection value\n";
        return false;
      }
      projectedSchemas.insert(point.representativeActor.schema);
      semanticValues.emplace_back(point.semanticConfiguration->bytes().begin(),
                                  point.semanticConfiguration->bytes().end());
    }
    llvm::sort(semanticValues);
    const bool hasOrRepresentative =
        projectedSchemas.contains(OperationSchemaId::ArithOrI) ||
        projectedSchemas.contains(OperationSchemaId::LLVMOrDisjoint);
    return projectedSchemas.size() == 3 &&
           projectedSchemas.contains(OperationSchemaId::ArithAndI) &&
           projectedSchemas.contains(OperationSchemaId::ArithXOrI) &&
           hasOrRepresentative &&
           std::adjacent_find(semanticValues.begin(), semanticValues.end()) ==
               semanticValues.end();
  };

  const auto checkEquivalentOrSchemas =
      [&](ImplementationFamilyId family, const FamilyCapabilityParams &params,
          llvm::ArrayRef<std::uint32_t> inputWidths,
          llvm::ArrayRef<std::uint32_t> resultWidths) {
        constexpr std::array equivalentOrSchemas = {
            OperationSchemaId::ArithOrI, OperationSchemaId::LLVMOrDisjoint};
        auto relation = resolveFabricOpSemanticFieldRelation(
            family, params, equivalentOrSchemas, inputWidths, resultWidths,
            context);
        if (!relation) {
          llvm::errs() << implementationFamilyKeyword(family)
                       << " equivalent OR domain did not resolve: "
                       << llvm::toString(relation.takeError()) << '\n';
          return false;
        }
        const auto domain = relation->finiteBehaviorDomain();
        return relation->kind() == FabricOpSemanticFieldRelationKind::None &&
               !relation->hasConfigurationField() && domain.size() == 1 &&
               !domain.front().semanticConfiguration;
      };

  const FamilyCapabilityParams scalar =
      ScalarIntegerParams{IntegerWidthSet::get(
          {IntegerWidth::I1, IntegerWidth::I8, IntegerWidth::I16,
           IntegerWidth::I32, IntegerWidth::I64})};
  const FamilyCapabilityParams vector = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I1, IntegerWidth::I8,
                            IntegerWidth::I16, IntegerWidth::I32,
                            IntegerWidth::I64}),
      128};
  constexpr std::array scalarInputs = {64U, 64U};
  constexpr std::array scalarResults = {64U};
  constexpr std::array vectorInputs = {128U, 128U};
  constexpr std::array vectorResults = {128U};
  const bool scalarOk = check(ImplementationFamilyId::ScalarIntegerLogic,
                              scalar, scalarInputs, scalarResults);
  const bool vectorOk = check(ImplementationFamilyId::FixedVectorIntegerLogic,
                              vector, vectorInputs, vectorResults);
  const bool scalarEquivalentOrOk =
      checkEquivalentOrSchemas(ImplementationFamilyId::ScalarIntegerLogic,
                               scalar, scalarInputs, scalarResults);
  const bool vectorEquivalentOrOk =
      checkEquivalentOrSchemas(ImplementationFamilyId::FixedVectorIntegerLogic,
                               vector, vectorInputs, vectorResults);
  return scalarOk && vectorOk && scalarEquivalentOrOk && vectorEquivalentOrOk;
}

bool checkFixedVectorMultiplyBehaviorDomain(MLIRContext &context) {
  const FamilyCapabilityParams params = FixedVectorIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}), 128};
  constexpr std::array enabled = {OperationSchemaId::ArithMulI};
  constexpr std::array inputWidths = {128U, 128U};
  constexpr std::array resultWidths = {128U};
  auto relation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorIntegerMultiply, params, enabled,
      inputWidths, resultWidths, context);
  if (!relation) {
    llvm::errs() << "fixed-vector multiply behavior domain did not resolve: "
                 << llvm::toString(relation.takeError()) << '\n';
    return false;
  }
  const auto domain = relation->finiteBehaviorDomain();
  if (relation->kind() != FabricOpSemanticFieldRelationKind::Finite)
    return false;
  llvm::SmallDenseSet<unsigned, 2> widths;
  std::vector<std::vector<std::uint8_t>> semanticValues;
  for (const auto &point : domain) {
    if (!point.semanticConfiguration ||
        point.representativeActor.schema != OperationSchemaId::ArithMulI)
      return false;
    auto vector =
        dyn_cast<VectorType>(point.representativeActor.type.getInput(0));
    auto element =
        vector ? dyn_cast<IntegerType>(vector.getElementType()) : IntegerType{};
    if (!element)
      return false;
    widths.insert(element.getWidth());
    semanticValues.emplace_back(point.semanticConfiguration->bytes().begin(),
                                point.semanticConfiguration->bytes().end());
  }
  llvm::sort(semanticValues);
  const auto project = [&](VectorType vector) {
    return relation->projectSemanticValue(
        dataflow::CanonicalActorSchemaProjection{
            OperationSchemaId::ArithMulI,
            FunctionType::get(&context, {vector, vector}, {vector}),
            dataflow::IntegerOverflowPayload{}},
        std::array<std::uint64_t, 2>{0, 1}, std::array<std::uint64_t, 1>{0});
  };
  auto i16x4 = project(VectorType::get({4}, IntegerType::get(&context, 16)));
  auto i16x8 = project(VectorType::get({8}, IntegerType::get(&context, 16)));
  auto i8x8 = project(VectorType::get({8}, IntegerType::get(&context, 8)));
  if (!i16x4 || !i16x8 || !i8x8) {
    if (!i16x4)
      llvm::consumeError(i16x4.takeError());
    if (!i16x8)
      llvm::consumeError(i16x8.takeError());
    if (!i8x8)
      llvm::consumeError(i8x8.takeError());
    llvm::errs() << "fixed-vector multiply configuration did not encode\n";
    return false;
  }
  return domain.size() == 2 && widths.contains(8) && widths.contains(16) &&
         i16x4->bytes().equals(i16x8->bytes()) &&
         !i16x4->bytes().equals(i8x8->bytes()) &&
         std::adjacent_find(semanticValues.begin(), semanticValues.end()) ==
             semanticValues.end();
}

bool checkFixedVectorValueSelectBehaviorDomain(MLIRContext &context) {
  const FamilyCapabilityParams params = FixedVectorValueSelectParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16}),
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::F32}), 128};
  constexpr std::array enabled = {OperationSchemaId::ArithSelect};
  constexpr std::array inputWidths = {128U, 128U, 128U};
  constexpr std::array resultWidths = {128U};
  auto relation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::FixedVectorValueSelect, params, enabled,
      inputWidths, resultWidths, context);
  if (!relation) {
    llvm::errs() << "fixed-vector select behavior domain did not resolve: "
                 << llvm::toString(relation.takeError()) << '\n';
    return false;
  }
  const auto domain = relation->finiteBehaviorDomain();
  if (relation->kind() != FabricOpSemanticFieldRelationKind::Finite)
    return false;
  llvm::SmallDenseSet<unsigned, 4> widths;
  std::vector<std::vector<std::uint8_t>> semanticValues;
  for (const auto &point : domain) {
    if (!point.semanticConfiguration ||
        point.representativeActor.schema != OperationSchemaId::ArithSelect)
      return false;
    auto values =
        dyn_cast<VectorType>(point.representativeActor.type.getInput(1));
    auto condition =
        dyn_cast<VectorType>(point.representativeActor.type.getInput(0));
    if (!values || !condition || condition.getShape() != values.getShape() ||
        !condition.getElementType().isInteger(1))
      return false;
    widths.insert(values.getElementTypeBitWidth());
    semanticValues.emplace_back(point.semanticConfiguration->bytes().begin(),
                                point.semanticConfiguration->bytes().end());
  }
  llvm::sort(semanticValues);
  const auto project = [&](Type element, std::int64_t lanes) {
    VectorType values = VectorType::get({lanes}, element);
    VectorType condition =
        VectorType::get({lanes}, IntegerType::get(&context, 1));
    return relation->projectSemanticValue(
        dataflow::CanonicalActorSchemaProjection{
            OperationSchemaId::ArithSelect,
            FunctionType::get(&context, {condition, values, values}, {values}),
            dataflow::NoPayload{}},
        std::array<std::uint64_t, 3>{0, 1, 2}, std::array<std::uint64_t, 1>{0});
  };
  auto i16x4 = project(IntegerType::get(&context, 16), 4);
  auto f16x4 = project(Float16Type::get(&context), 4);
  auto f16x8 = project(Float16Type::get(&context), 8);
  auto i8x8 = project(IntegerType::get(&context, 8), 8);
  if (!i16x4 || !f16x4 || !f16x8 || !i8x8) {
    if (!i16x4)
      llvm::consumeError(i16x4.takeError());
    if (!f16x4)
      llvm::consumeError(f16x4.takeError());
    if (!f16x8)
      llvm::consumeError(f16x8.takeError());
    if (!i8x8)
      llvm::consumeError(i8x8.takeError());
    llvm::errs() << "fixed-vector select configuration did not encode\n";
    return false;
  }
  return domain.size() == 3 && widths.contains(8) && widths.contains(16) &&
         widths.contains(32) && i16x4->bytes().equals(f16x4->bytes()) &&
         f16x4->bytes().equals(f16x8->bytes()) &&
         !i16x4->bytes().equals(i8x8->bytes()) &&
         std::adjacent_find(semanticValues.begin(), semanticValues.end()) ==
             semanticValues.end();
}

dataflow::CanonicalActorSchemaProjection
makeScalarIntegerCastActor(MLIRContext &context, OperationSchemaId schema,
                           Type source, Type destination) {
  dataflow::SemanticPayload payload = dataflow::NoPayload{};
  switch (dataflow::semanticsCase(schema)) {
  case dataflow::OperationSemanticsCase::NoSemanticPayload:
    break;
  case dataflow::OperationSemanticsCase::ArithNonNegative:
    payload = dataflow::NonNegativePayload{};
    break;
  case dataflow::OperationSemanticsCase::ArithIntegerOverflow:
    payload = dataflow::IntegerOverflowPayload{};
    break;
  default:
    llvm_unreachable("integer cast has an unexpected semantic payload");
  }
  return {schema, FunctionType::get(&context, {source}, {destination}),
          std::move(payload)};
}

bool checkScalarIntegerCastRelationClosure(MLIRContext &context) {
  constexpr std::array indexEnabled = {OperationSchemaId::ArithIndexCast};
  constexpr std::array inputWidths = {64U};
  constexpr std::array resultWidths = {64U};
  const auto expectOrphanRejected =
      [&](const FamilyCapabilityParams &orphanParams,
          llvm::ArrayRef<OperationSchemaId> enabled,
          llvm::StringRef expectedReason) {
        auto orphan = resolveFabricOpSemanticFieldRelation(
            ImplementationFamilyId::ScalarIntegerCast, orphanParams, enabled,
            inputWidths, resultWidths, context);
        if (orphan) {
          llvm::errs() << "orphan integer cast relation was accepted\n";
          return false;
        }
        const std::string error = llvm::toString(orphan.takeError());
        return llvm::StringRef(error).contains(expectedReason);
      };

  const FamilyCapabilityParams orphanIndexWidthParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32}}),
          ResolvedIndexWidthSet::get(
              {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})}};
  if (!expectOrphanRejected(orphanIndexWidthParams, indexEnabled,
                            "orphan index width"))
    return false;

  const FamilyCapabilityParams orphanWidthPairParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32},
                                     {IntegerWidth::I16, IntegerWidth::I64}}),
          ResolvedIndexWidthSet::get({ResolvedIndexWidth::I32})}};
  if (!expectOrphanRejected(orphanWidthPairParams, indexEnabled,
                            "orphan width pair"))
    return false;

  constexpr std::array extensionEnabled = {OperationSchemaId::ArithExtSI};
  const FamilyCapabilityParams extensionWithIndexParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32}}),
          ResolvedIndexWidthSet::get({ResolvedIndexWidth::I32})}};
  return expectOrphanRejected(extensionWithIndexParams, extensionEnabled,
                              "orphan index width");
}

bool checkScalarIntegerCastBehaviorDomain(MLIRContext &context) {
  const FamilyCapabilityParams params =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32},
                                     {IntegerWidth::I8, IntegerWidth::I64},
                                     {IntegerWidth::I32, IntegerWidth::I8},
                                     {IntegerWidth::I64, IntegerWidth::I8}}),
          ResolvedIndexWidthSet::get(
              {ResolvedIndexWidth::I32, ResolvedIndexWidth::I64})}};
  constexpr std::array enabled = {
      OperationSchemaId::ArithExtSI, OperationSchemaId::ArithExtUI,
      OperationSchemaId::ArithTruncI, OperationSchemaId::ArithIndexCast,
      OperationSchemaId::ArithIndexCastUI};
  constexpr std::array inputWidths = {64U};
  constexpr std::array resultWidths = {64U};
  auto relation = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::ScalarIntegerCast, params, enabled, inputWidths,
      resultWidths, context);
  if (!relation) {
    llvm::errs() << "scalar integer cast behavior domain did not resolve: "
                 << llvm::toString(relation.takeError()) << '\n';
    return false;
  }
  const auto domain = relation->finiteBehaviorDomain();
  if (relation->kind() != FabricOpSemanticFieldRelationKind::Finite ||
      domain.size() != 6)
    return false;
  for (const auto &point : domain)
    if (!point.semanticConfiguration)
      return false;

  Type i8 = IntegerType::get(&context, 8);
  Type i32 = IntegerType::get(&context, 32);
  Type i64 = IntegerType::get(&context, 64);
  Type index = IndexType::get(&context);
  const auto project =
      [&](OperationSchemaId schema, Type source, Type destination,
          std::optional<ResolvedIndexWidth> resolved = std::nullopt) {
        return relation->projectSemanticValue(
            makeScalarIntegerCastActor(context, schema, source, destination),
            std::array<std::uint64_t, 1>{0}, std::array<std::uint64_t, 1>{0},
            resolved);
      };
  auto signExtend = project(OperationSchemaId::ArithExtSI, i8, i32);
  auto indexSignExtend = project(OperationSchemaId::ArithIndexCast, i8, index,
                                 ResolvedIndexWidth::I32);
  auto zeroExtend = project(OperationSchemaId::ArithExtUI, i8, i32);
  auto indexZeroExtend = project(OperationSchemaId::ArithIndexCastUI, i8, index,
                                 ResolvedIndexWidth::I32);
  auto truncate = project(OperationSchemaId::ArithTruncI, i32, i8);
  auto indexTruncate = project(OperationSchemaId::ArithIndexCast, index, i8,
                               ResolvedIndexWidth::I32);
  auto indexUiTruncate = project(OperationSchemaId::ArithIndexCastUI, index, i8,
                                 ResolvedIndexWidth::I32);
  auto wideSignExtend = project(OperationSchemaId::ArithExtSI, i8, i64);
  if (!signExtend || !indexSignExtend || !zeroExtend || !indexZeroExtend ||
      !truncate || !indexTruncate || !indexUiTruncate || !wideSignExtend) {
    if (!signExtend)
      llvm::consumeError(signExtend.takeError());
    if (!indexSignExtend)
      llvm::consumeError(indexSignExtend.takeError());
    if (!zeroExtend)
      llvm::consumeError(zeroExtend.takeError());
    if (!indexZeroExtend)
      llvm::consumeError(indexZeroExtend.takeError());
    if (!truncate)
      llvm::consumeError(truncate.takeError());
    if (!indexTruncate)
      llvm::consumeError(indexTruncate.takeError());
    if (!indexUiTruncate)
      llvm::consumeError(indexUiTruncate.takeError());
    if (!wideSignExtend)
      llvm::consumeError(wideSignExtend.takeError());
    return false;
  }
  if (!signExtend->bytes().equals(indexSignExtend->bytes()) ||
      !zeroExtend->bytes().equals(indexZeroExtend->bytes()) ||
      !truncate->bytes().equals(indexTruncate->bytes()) ||
      !truncate->bytes().equals(indexUiTruncate->bytes()) ||
      signExtend->bytes().equals(zeroExtend->bytes()) ||
      signExtend->bytes().equals(wideSignExtend->bytes()))
    return false;

  auto missingIndexWitness = relation->projectSemanticValue(
      makeScalarIntegerCastActor(context, OperationSchemaId::ArithIndexCast, i8,
                                 index),
      std::array<std::uint64_t, 1>{0}, std::array<std::uint64_t, 1>{0});
  if (missingIndexWitness)
    return false;
  const std::string missingIndexWitnessError =
      llvm::toString(missingIndexWitness.takeError());
  if (!llvm::StringRef(missingIndexWitnessError)
           .contains("no resolved index width"))
    return false;

  const FamilyCapabilityParams singletonParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32}}),
          ResolvedIndexWidthSet::get({ResolvedIndexWidth::I32})}};
  constexpr std::array singletonEnabled = {OperationSchemaId::ArithIndexCast};
  auto singleton = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::ScalarIntegerCast, singletonParams,
      singletonEnabled, inputWidths, resultWidths, context);
  if (!singleton ||
      singleton->kind() != FabricOpSemanticFieldRelationKind::None ||
      singleton->hasConfigurationField() ||
      singleton->finiteBehaviorDomain().size() != 1 ||
      singleton->finiteBehaviorDomain().front().semanticConfiguration ||
      singleton->finiteBehaviorDomain().front().resolvedIndexWidth !=
          ResolvedIndexWidth::I32)
    return false;

  const FamilyCapabilityParams missingIndexParams =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32}}),
          ResolvedIndexWidthSet::get({})}};
  auto missingIndexDomain = resolveFabricOpSemanticFieldRelation(
      ImplementationFamilyId::ScalarIntegerCast, missingIndexParams,
      singletonEnabled, inputWidths, resultWidths, context);
  if (missingIndexDomain)
    return false;
  const std::string missingIndexDomainError =
      llvm::toString(missingIndexDomain.takeError());
  return llvm::StringRef(missingIndexDomainError)
      .contains("schema has no admitted behavior");
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, LLVM::LLVMDialect, ub::UBDialect,
                  dataflow::DataflowDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkDescriptorRelations();
  ok &= checkMembership();
  ok &= checkCapabilityCodec(context);
  ok &= checkIntegerAdmission(context);
  ok &= checkPointerAdmission(context);
  ok &= checkFloatingAdmission(context);
  ok &= checkCastRelations(context);
  ok &= checkLoopAndTokenAdmission(context);
  ok &= checkFixedVectorAdmission(context);
  ok &= checkFixedVectorStructuralAdmission(context);
  ok &= checkAdapterAndTokenAdmission(context);
  ok &= checkPortCorrespondenceEnumeration(context);
  ok &= checkScalarFloatFmaBehaviorDomain(context);
  ok &= checkIntegerLogicBehaviorDomains(context);
  ok &= checkFixedVectorMultiplyBehaviorDomain(context);
  ok &= checkFixedVectorValueSelectBehaviorDomain(context);
  ok &= checkScalarIntegerCastRelationClosure(context);
  ok &= checkScalarIntegerCastBehaviorDomain(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
