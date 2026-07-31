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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>

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
  if (families != 68) {
    llvm::errs() << "the registry must contain exactly 68 families, found "
                 << families << '\n';
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
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithAddI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithSubI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::LLVMGetElementPtr) ||
      admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                            OperationSchemaId::ArithMulI) ||
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
  FamilyCapabilityParams conversions = ScalarIntegerFloatConversionParams{
      conversionPairs, FloatBehaviorProfile::strictIEEE()};
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
  ok &= checkAdapterAndTokenAdmission(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
