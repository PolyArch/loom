//===- ImplementationFamilyTest.cpp - HSG family registry anchors ---------===//
//
// Anchors the normative implementation-family registry:
//
//   * a family descriptor exposes exactly the four generated facts, and every
//     admitted member is a registered operation schema;
//   * a family admits its own shared members and rejects an operation of
//     another family, so membership is a real relation rather than a name
//     bag; and
//   * the one keyword of a family is derived from its generated identity and
//     round-trips, so a diagnostic never needs a descriptor name field.
//
// The anchor deliberately does not restate the registry: it checks relations
// the generated source must satisfy, plus the few normative memberships the
// hardware-sharing specification fixes by name.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>

using namespace fabric;
using namespace mlir;
using dataflow::OperationSchemaId;

namespace {

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
  if (!llvm::StringRef(message).contains(semanticReason)) {
    llvm::errs() << implementationFamilyKeyword(family)
                 << " reported an unspecific rejection: " << message << '\n';
    return false;
  }
  return true;
}

bool checkDescriptorRelations() {
  bool ok = true;
  llvm::StringSet<> keywords;
  const std::uint32_t families = implementationFamilyCount();
  const std::uint32_t schemas = dataflow::operationSchemaCount();
  if (families != 20) {
    llvm::errs() << "the registry must contain exactly 20 implementation "
                    "families, found "
                 << families << '\n';
    ok = false;
  }
  for (std::uint32_t index = 0; index < families; ++index) {
    auto family = static_cast<ImplementationFamilyId>(index);
    const ImplementationFamilyDescriptor &descriptor =
        implementationFamily(family);
    auto [familyId, admittedSchemas, capabilityParamsSchema,
          typedAdmissionProvider] = descriptor;
    (void)admittedSchemas;
    (void)capabilityParamsSchema;
    (void)typedAdmissionProvider;
    if (familyId != family) {
      llvm::errs() << "family " << index << " reports another identity\n";
      ok = false;
    }
    if (descriptor.admittedSchemas.empty()) {
      llvm::errs() << implementationFamilyKeyword(family)
                   << " admits no operation schema\n";
      ok = false;
    }
    for (OperationSchemaId member : descriptor.admittedSchemas) {
      if (static_cast<std::uint32_t>(member) >= schemas) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " admits an unregistered operation schema\n";
        ok = false;
        continue;
      }
      if (!admitsOperationSchema(family, member)) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " does not admit its own declared member\n";
        ok = false;
      }
    }

    llvm::StringRef keyword = implementationFamilyKeyword(family);
    if (keyword.empty() || !keywords.insert(keyword).second) {
      llvm::errs() << "family " << index << " has no unique keyword\n";
      ok = false;
    }
    std::optional<ImplementationFamilyId> resolved =
        findImplementationFamily(keyword);
    if (!resolved || *resolved != family) {
      llvm::errs() << "keyword '" << keyword << "' does not resolve back\n";
      ok = false;
    }
    if (capabilityParamsSchemaKeyword(descriptor.capabilityParamsSchema)
            .empty() ||
        typedAdmissionProviderKeyword(descriptor.typedAdmissionProvider)
            .empty()) {
      llvm::errs() << keyword << " selects an unspellable vocabulary member\n";
      ok = false;
    }
  }
  return ok;
}

/// One genuinely shared datapath family admits its own members and nothing
/// else. An adder and a multiplier are separate datapaths, so the add/subtract
/// family must reject a multiply.
bool checkMembershipAndWrongFamily() {
  bool ok = true;
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithAddI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithSubI)) {
    llvm::errs() << "the integer add/subtract family lost a shared member\n";
    ok = false;
  }
  if (admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                            OperationSchemaId::ArithMulI)) {
    llvm::errs() << "the integer add/subtract family admitted a multiply\n";
    ok = false;
  }
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerMultiply,
                             OperationSchemaId::ArithMulI)) {
    llvm::errs() << "the integer multiply family lost its member\n";
    ok = false;
  }

  // The four loop-control families are distinct physical families: none of
  // them admits another's operation.
  const ImplementationFamilyId loopControl[] = {
      ImplementationFamilyId::LoopStream, ImplementationFamilyId::LoopCarry,
      ImplementationFamilyId::LoopInvariant, ImplementationFamilyId::LoopGate};
  const OperationSchemaId loopMembers[] = {
      OperationSchemaId::DataflowStream, OperationSchemaId::DataflowCarry,
      OperationSchemaId::DataflowInvariant, OperationSchemaId::DataflowGate};
  for (unsigned family = 0; family < 4; ++family)
    for (unsigned member = 0; member < 4; ++member) {
      const bool admitted =
          admitsOperationSchema(loopControl[family], loopMembers[member]);
      if (admitted != (family == member)) {
        llvm::errs() << implementationFamilyKeyword(loopControl[family])
                     << " has the wrong relation to "
                     << dataflow::operationSchemaSpelling(loopMembers[member])
                     << '\n';
        ok = false;
      }
    }

  if (findImplementationFamily("NoSuchFamily")) {
    llvm::errs() << "an unregistered keyword resolved to a family\n";
    ok = false;
  }
  if (findImplementationFamily("ScalarFloatDivide")) {
    llvm::errs() << "a speculative divider family remains representable\n";
    ok = false;
  }
  return ok;
}

bool checkIntegerCapabilityAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  llvm::SmallVector<Operation *> values;
  llvm::SmallVector<Operation *> actors;
  auto poison = [&](Type type) -> Value {
    auto value = ub::PoisonOp::create(builder, loc, type);
    values.push_back(value);
    return value;
  };

  Type i1 = builder.getI1Type();
  Type i7 = builder.getIntegerType(7);
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();
  VectorType vectorI32 = VectorType::get({2}, i32);

  Operation *add8 = arith::AddIOp::create(builder, loc, poison(i8), poison(i8));
  Operation *add7 = arith::AddIOp::create(builder, loc, poison(i7), poison(i7));
  Operation *add1 = arith::AddIOp::create(builder, loc, poison(i1), poison(i1));
  Operation *and1 = arith::AndIOp::create(builder, loc, poison(i1), poison(i1));
  Operation *vectorAdd =
      arith::AddIOp::create(builder, loc, poison(vectorI32), poison(vectorI32));
  Operation *signedLess = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, poison(i32), poison(i32));
  Operation *equal = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, poison(i32), poison(i32));
  Operation *select1 =
      arith::SelectOp::create(builder, loc, poison(i1), poison(i1), poison(i1));
  actors.append(
      {add8, add7, add1, and1, vectorAdd, signedLess, equal, select1});

  FamilyCapabilityParams ordinary = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64})};
  FamilyCapabilityParams logic = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I1, IntegerWidth::I8})};
  FamilyCapabilityParams comparisons = ScalarIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                            IntegerWidth::I32, IntegerWidth::I64}),
      IntegerPredicateSet::get({arith::CmpIPredicate::slt})};
  FamilyCapabilityParams invalidComparisons = ScalarIntegerCompareMinMaxParams{
      IntegerWidthSet::get({IntegerWidth::I32}),
      IntegerPredicateSet::get({static_cast<arith::CmpIPredicate>(99)})};
  FamilyCapabilityParams selection = ScalarValueSelectParams{
      IntegerWidthSet::get({IntegerWidth::I1, IntegerWidth::I8}),
      FloatFormatSet::get({FloatFormat::F32})};
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
  check(add7, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary, false,
        "integer width");
  check(add1, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary, false,
        "integer width");
  check(and1, ImplementationFamilyId::ScalarIntegerLogic, &logic, true, {});
  check(vectorAdd, ImplementationFamilyId::ScalarIntegerAddSub, &ordinary,
        false, "scalar actor");
  check(signedLess, ImplementationFamilyId::ScalarIntegerCompareMinMax,
        &comparisons, true, {});
  check(equal, ImplementationFamilyId::ScalarIntegerCompareMinMax, &comparisons,
        false, "integer predicate");
  check(signedLess, ImplementationFamilyId::ScalarIntegerCompareMinMax,
        &invalidComparisons, false, "invalid integer predicate set");
  check(select1, ImplementationFamilyId::ScalarValueSelect, &selection, true,
        {});
  check(add8, ImplementationFamilyId::ScalarIntegerAddSub, nullptr, false,
        "capability parameters");
  check(add8, ImplementationFamilyId::ScalarIntegerAddSub, &wrongSchema, false,
        "capability parameter schema");
  check(add8, ImplementationFamilyId::ScalarFloatAddSub, nullptr, false,
        "actor schema is not admitted");

  for (Operation *actor : llvm::reverse(actors))
    actor->erase();
  for (Operation *value : llvm::reverse(values))
    value->erase();
  return ok;
}

bool checkFloatingCapabilityAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  llvm::SmallVector<Operation *> values;
  llvm::SmallVector<Operation *> actors;
  auto poison = [&](Type type) -> Value {
    auto value = ub::PoisonOp::create(builder, loc, type);
    values.push_back(value);
    return value;
  };

  Type f32 = builder.getF32Type();
  Type f80 = builder.getF80Type();
  Value f32Lhs = poison(f32);
  Value f32Rhs = poison(f32);
  Operation *plain = arith::AddFOp::create(builder, loc, f32Lhs, f32Rhs,
                                           arith::FastMathFlags::none,
                                           arith::RoundingModeAttr{});
  Operation *noNaNs = arith::AddFOp::create(builder, loc, f32Lhs, f32Rhs,
                                            arith::FastMathFlags::nnan,
                                            arith::RoundingModeAttr{});
  Operation *noSignedZeros = arith::AddFOp::create(builder, loc, f32Lhs, f32Rhs,
                                                   arith::FastMathFlags::nsz,
                                                   arith::RoundingModeAttr{});
  Operation *downward = arith::AddFOp::create(
      builder, loc, f32Lhs, f32Rhs, arith::FastMathFlags::none,
      arith::RoundingModeAttr::get(&context, arith::RoundingMode::downward));
  Operation *wide = arith::AddFOp::create(
      builder, loc, poison(f80), poison(f80), arith::FastMathFlags::none,
      arith::RoundingModeAttr{});
  Operation *orderedLess =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT, f32Lhs,
                            f32Rhs, arith::FastMathFlags::none);
  Operation *orderedEqual =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, f32Lhs,
                            f32Rhs, arith::FastMathFlags::none);
  Operation *minNum = arith::MinNumFOp::create(builder, loc, f32Lhs, f32Rhs,
                                               arith::FastMathFlags::none);
  actors.append({plain, noNaNs, noSignedZeros, downward, wide, orderedLess,
                 orderedEqual, minNum});

  FloatBehaviorProfile strict = FloatBehaviorProfile::strictIEEE();
  FloatBehaviorProfile relaxed = strict;
  relaxed.admittedFastMath = arith::FastMathFlags::nnan;
  FloatBehaviorProfile downwardBehavior = strict;
  downwardBehavior.roundingModes = RoundingModeSet::get(
      {arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward});
  FloatBehaviorProfile numberPreferred = strict;
  numberPreferred.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  FloatBehaviorProfile signedZeroRelaxed = strict;
  signedZeroRelaxed.signedZeroBehaviors =
      FloatSignedZeroBehaviorSet::get({FloatSignedZeroBehavior::IgnoreSign});
  signedZeroRelaxed.admittedFastMath = arith::FastMathFlags::nsz;
  FloatBehaviorProfile flushSubnormals = strict;
  flushSubnormals.subnormalBehaviors =
      FloatSubnormalBehaviorSet::get({FloatSubnormalBehavior::FlushToZero});
  FloatBehaviorProfile invalidRounding = strict;
  invalidRounding.roundingModes =
      RoundingModeSet::get({static_cast<arith::RoundingMode>(99)});

  FamilyCapabilityParams strictFloat = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16,
                           FloatFormat::F32, FloatFormat::F64}),
      strict};
  FamilyCapabilityParams relaxedFloat =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), relaxed};
  FamilyCapabilityParams downwardFloat = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F32}), downwardBehavior};
  FamilyCapabilityParams signedZeroFloat = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F32}), signedZeroRelaxed};
  FamilyCapabilityParams flushSubnormalFloat = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F32}), flushSubnormals};
  FamilyCapabilityParams invalidRoundingFloat = ScalarFloatParams{
      FloatFormatSet::get({FloatFormat::F32}), invalidRounding};
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
  check(noNaNs, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, false,
        "fast-math behavior");
  check(noNaNs, ImplementationFamilyId::ScalarFloatAddSub, relaxedFloat, true,
        {});
  check(noSignedZeros, ImplementationFamilyId::ScalarFloatAddSub, strictFloat,
        false, "fast-math behavior");
  check(noSignedZeros, ImplementationFamilyId::ScalarFloatAddSub,
        signedZeroFloat, true, {});
  check(downward, ImplementationFamilyId::ScalarFloatAddSub, strictFloat, false,
        "rounding behavior");
  check(downward, ImplementationFamilyId::ScalarFloatAddSub, downwardFloat,
        true, {});
  check(plain, ImplementationFamilyId::ScalarFloatAddSub, invalidRoundingFloat,
        false, "invalid rounding mode set");
  check(plain, ImplementationFamilyId::ScalarFloatAddSub, flushSubnormalFloat,
        false, "subnormal behavior");
  check(orderedLess, ImplementationFamilyId::ScalarFloatCompareMinMax,
        comparisons, true, {});
  check(orderedEqual, ImplementationFamilyId::ScalarFloatCompareMinMax,
        comparisons, false, "floating predicate");
  check(minNum, ImplementationFamilyId::ScalarFloatCompareMinMax, comparisons,
        false, "NaN behavior");
  check(minNum, ImplementationFamilyId::ScalarFloatCompareMinMax,
        numberPreferredComparisons, true, {});

  for (Operation *actor : llvm::reverse(actors))
    actor->erase();
  for (Operation *value : llvm::reverse(values))
    value->erase();
  return ok;
}

bool checkCastCapabilityAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  llvm::SmallVector<Operation *> values;
  llvm::SmallVector<Operation *> actors;
  auto poison = [&](Type type) -> Value {
    auto value = ub::PoisonOp::create(builder, loc, type);
    values.push_back(value);
    return value;
  };

  Type i1 = builder.getI1Type();
  Type i7 = builder.getIntegerType(7);
  Type i8 = builder.getI8Type();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type f16 = builder.getF16Type();
  Type f32 = builder.getF32Type();
  Type f64 = builder.getF64Type();
  Operation *extend = arith::ExtSIOp::create(builder, loc, i32, poison(i8));
  Operation *extend16To64 =
      arith::ExtSIOp::create(builder, loc, i64, poison(builder.getI16Type()));
  Operation *unrelatedIntegerPair =
      arith::ExtSIOp::create(builder, loc, i64, poison(i8));
  Operation *wrongDirectionExtend =
      arith::ExtSIOp::create(builder, loc, i8, poison(i32));
  Operation *truncate = arith::TruncIOp::create(builder, loc, i8, poison(i32));
  Operation *wrongDirectionTruncate =
      arith::TruncIOp::create(builder, loc, i32, poison(i8));
  Operation *booleanExtend =
      arith::ExtUIOp::create(builder, loc, i8, poison(i1));
  Operation *extend7 = arith::ExtSIOp::create(builder, loc, i32, poison(i7));
  Operation *indexCast = arith::IndexCastOp::create(
      builder, loc, i64, poison(builder.getIndexType()));
  Operation *bitcast = arith::BitcastOp::create(builder, loc, f32, poison(i32));
  Operation *wideBitcast =
      arith::BitcastOp::create(builder, loc, f64, poison(i32));
  Operation *floatExtend =
      arith::ExtFOp::create(builder, loc, f32, poison(f16));
  Operation *bf16ToF64 =
      arith::ExtFOp::create(builder, loc, f64, poison(builder.getBF16Type()));
  Operation *unrelatedFloatPair =
      arith::ExtFOp::create(builder, loc, f64, poison(f16));
  Operation *floatTruncate =
      arith::TruncFOp::create(builder, loc, f16, poison(f32));
  Operation *wrongDirectionFloatExtend =
      arith::ExtFOp::create(builder, loc, f16, poison(f32));
  Operation *signedToFloat =
      arith::SIToFPOp::create(builder, loc, f32, poison(i32));
  Operation *wideSignedToFloat =
      arith::SIToFPOp::create(builder, loc, f64, poison(i64));
  Operation *unrelatedSignedToFloat =
      arith::SIToFPOp::create(builder, loc, f64, poison(i32));
  Operation *floatToSigned =
      arith::FPToSIOp::create(builder, loc, i32, poison(f32));
  Operation *wideFloatToSigned =
      arith::FPToSIOp::create(builder, loc, i64, poison(f64));
  Operation *unrelatedFloatToSigned =
      arith::FPToSIOp::create(builder, loc, i32, poison(f64));
  Operation *booleanToFloat =
      arith::SIToFPOp::create(builder, loc, f32, poison(i1));
  actors.append({extend,
                 extend16To64,
                 unrelatedIntegerPair,
                 wrongDirectionExtend,
                 truncate,
                 wrongDirectionTruncate,
                 booleanExtend,
                 extend7,
                 indexCast,
                 bitcast,
                 wideBitcast,
                 floatExtend,
                 bf16ToF64,
                 unrelatedFloatPair,
                 floatTruncate,
                 wrongDirectionFloatExtend,
                 signedToFloat,
                 wideSignedToFloat,
                 unrelatedSignedToFloat,
                 floatToSigned,
                 wideFloatToSigned,
                 unrelatedFloatToSigned,
                 booleanToFloat});

  IntegerCastRelation integerRelation{
      IntegerWidthRelation::get({{IntegerWidth::I8, IntegerWidth::I32},
                                 {IntegerWidth::I16, IntegerWidth::I64},
                                 {IntegerWidth::I32, IntegerWidth::I8},
                                 {IntegerWidth::I1, IntegerWidth::I8},
                                 {IntegerWidth::I64, IntegerWidth::I64}}),
      ResolvedIndexWidth::I64};
  FamilyCapabilityParams integerCasts =
      ScalarIntegerCastParams{integerRelation};
  integerRelation.resolvedIndexWidth = std::nullopt;
  FamilyCapabilityParams unresolvedIndexCasts =
      ScalarIntegerCastParams{integerRelation};
  FamilyCapabilityParams emptyIntegerCasts = ScalarIntegerCastParams{
      IntegerCastRelation{IntegerWidthRelation{}, ResolvedIndexWidth::I64}};
  FamilyCapabilityParams invalidIntegerCasts =
      ScalarIntegerCastParams{IntegerCastRelation{
          IntegerWidthRelation::get(
              {{static_cast<IntegerWidth>(99), IntegerWidth::I32}}),
          ResolvedIndexWidth::I64}};
  FamilyCapabilityParams bitReinterpretation = ScalarBitReinterpretParams{
      IntegerWidthSet::get({IntegerWidth::I32, IntegerWidth::I64}),
      FloatFormatSet::get({FloatFormat::F32, FloatFormat::F64})};
  FamilyCapabilityParams floatCasts = ScalarFloatWidthCastParams{
      FloatFormatRelation::get({{FloatFormat::F16, FloatFormat::F32},
                                {FloatFormat::BF16, FloatFormat::F64},
                                {FloatFormat::F32, FloatFormat::F16}}),
      FloatBehaviorProfile::strictIEEE()};
  FamilyCapabilityParams emptyFloatCasts = ScalarFloatWidthCastParams{
      FloatFormatRelation{}, FloatBehaviorProfile::strictIEEE()};
  IntegerFloatFormatRelation conversionPairs =
      IntegerFloatFormatRelation::get({{IntegerWidth::I32, FloatFormat::F32},
                                       {IntegerWidth::I64, FloatFormat::F64}});
  FamilyCapabilityParams integerFloatConversions =
      ScalarIntegerFloatConversionParams{conversionPairs,
                                         FloatBehaviorProfile::strictIEEE()};
  FloatBehaviorProfile conversionBehavior = FloatBehaviorProfile::strictIEEE();
  conversionBehavior.roundingModes =
      RoundingModeSet::get({arith::RoundingMode::downward});
  FamilyCapabilityParams conversionWithoutArithmeticRounding =
      ScalarIntegerFloatConversionParams{conversionPairs, conversionBehavior};
  FamilyCapabilityParams emptyConversions = ScalarIntegerFloatConversionParams{
      IntegerFloatFormatRelation{}, FloatBehaviorProfile::strictIEEE()};

  bool ok = true;
  auto check = [&](Operation *op, ImplementationFamilyId family,
                   const FamilyCapabilityParams &params, bool admitted,
                   llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(family, &params, *projection, admitted, reason);
  };
  check(extend, ImplementationFamilyId::ScalarIntegerCast, integerCasts, true,
        {});
  check(extend16To64, ImplementationFamilyId::ScalarIntegerCast, integerCasts,
        true, {});
  check(unrelatedIntegerPair, ImplementationFamilyId::ScalarIntegerCast,
        integerCasts, false, "integer cast relation");
  check(wrongDirectionExtend, ImplementationFamilyId::ScalarIntegerCast,
        integerCasts, false, "extension must widen");
  check(truncate, ImplementationFamilyId::ScalarIntegerCast, integerCasts, true,
        {});
  check(wrongDirectionTruncate, ImplementationFamilyId::ScalarIntegerCast,
        integerCasts, false, "truncation must narrow");
  check(booleanExtend, ImplementationFamilyId::ScalarIntegerCast, integerCasts,
        true, {});
  check(extend7, ImplementationFamilyId::ScalarIntegerCast, integerCasts, false,
        "integer cast relation");
  check(extend, ImplementationFamilyId::ScalarIntegerCast, emptyIntegerCasts,
        false, "non-empty integer cast relation");
  check(extend, ImplementationFamilyId::ScalarIntegerCast, invalidIntegerCasts,
        false, "invalid integer cast relation");
  check(indexCast, ImplementationFamilyId::ScalarIntegerCast, integerCasts,
        true, {});
  check(indexCast, ImplementationFamilyId::ScalarIntegerCast,
        unresolvedIndexCasts, false, "resolved index width");
  check(bitcast, ImplementationFamilyId::ScalarBitReinterpret,
        bitReinterpretation, true, {});
  check(wideBitcast, ImplementationFamilyId::ScalarBitReinterpret,
        bitReinterpretation, false, "equal semantic width");
  check(floatExtend, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        true, {});
  check(bf16ToF64, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        true, {});
  check(unrelatedFloatPair, ImplementationFamilyId::ScalarFloatWidthCast,
        floatCasts, false, "floating cast relation");
  check(floatTruncate, ImplementationFamilyId::ScalarFloatWidthCast, floatCasts,
        true, {});
  check(wrongDirectionFloatExtend, ImplementationFamilyId::ScalarFloatWidthCast,
        floatCasts, false, "extension must widen");
  check(floatExtend, ImplementationFamilyId::ScalarFloatWidthCast,
        emptyFloatCasts, false, "non-empty floating cast relation");
  check(signedToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        integerFloatConversions, true, {});
  check(wideSignedToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        integerFloatConversions, true, {});
  check(unrelatedSignedToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        integerFloatConversions, false, "integer and floating relation");
  check(floatToSigned, ImplementationFamilyId::ScalarFloatToInteger,
        integerFloatConversions, true, {});
  check(wideFloatToSigned, ImplementationFamilyId::ScalarFloatToInteger,
        integerFloatConversions, true, {});
  check(unrelatedFloatToSigned, ImplementationFamilyId::ScalarFloatToInteger,
        integerFloatConversions, false, "integer and floating relation");
  check(signedToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        conversionWithoutArithmeticRounding, true, {});
  check(floatToSigned, ImplementationFamilyId::ScalarFloatToInteger,
        conversionWithoutArithmeticRounding, true, {});
  check(signedToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        emptyConversions, false, "non-empty integer and floating relation");
  check(booleanToFloat, ImplementationFamilyId::ScalarIntegerToFloat,
        integerFloatConversions, false, "integer width");

  for (Operation *actor : llvm::reverse(actors))
    actor->erase();
  for (Operation *value : llvm::reverse(values))
    value->erase();
  return ok;
}

bool checkLoopCapabilityAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  llvm::SmallVector<Operation *> values;
  llvm::SmallVector<Operation *> actors;
  auto poison = [&](Type type) -> Value {
    auto value = ub::PoisonOp::create(builder, loc, type);
    values.push_back(value);
    return value;
  };

  Type i7 = builder.getIntegerType(7);
  Type i32 = builder.getI32Type();
  auto makeStream = [&](Type type, dataflow::StreamStepKind stepKind,
                        arith::CmpIPredicate predicate) -> Operation * {
    Operation *stream = dataflow::StreamOp::create(
        builder, loc, type, builder.getI1Type(), poison(type), poison(type),
        poison(type), stepKind, predicate);
    actors.push_back(stream);
    return stream;
  };
  Operation *addStream =
      makeStream(i32, dataflow::StreamStepKind::Add, arith::CmpIPredicate::slt);
  Operation *subStream =
      makeStream(i32, dataflow::StreamStepKind::Sub, arith::CmpIPredicate::slt);
  Operation *equalStream =
      makeStream(i32, dataflow::StreamStepKind::Add, arith::CmpIPredicate::eq);
  Operation *narrowStream =
      makeStream(i7, dataflow::StreamStepKind::Add, arith::CmpIPredicate::slt);

  FamilyCapabilityParams stream = LoopStreamParams{
      IntegerWidthSet::get({IntegerWidth::I32}), dataflow::StreamStepKind::Add,
      IntegerPredicateSet::get({arith::CmpIPredicate::slt})};
  FamilyCapabilityParams noWidths =
      LoopStreamParams{IntegerWidthSet{}, dataflow::StreamStepKind::Add,
                       IntegerPredicateSet::get({arith::CmpIPredicate::slt})};
  FamilyCapabilityParams noPredicates =
      LoopStreamParams{IntegerWidthSet::get({IntegerWidth::I32}),
                       dataflow::StreamStepKind::Add, IntegerPredicateSet{}};
  FamilyCapabilityParams invalidStep =
      LoopStreamParams{IntegerWidthSet::get({IntegerWidth::I32}),
                       static_cast<dataflow::StreamStepKind>(99),
                       IntegerPredicateSet::get({arith::CmpIPredicate::slt})};

  bool ok = true;
  auto check = [&](Operation *op, const FamilyCapabilityParams &params,
                   bool admitted, llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(ImplementationFamilyId::LoopStream, &params,
                            *projection, admitted, reason);
  };
  check(addStream, stream, true, {});
  check(subStream, stream, false, "fixed stream step kind");
  check(equalStream, stream, false, "continuation predicate");
  check(narrowStream, stream, false, "integer width");
  check(addStream, noWidths, false, "non-empty integer width set");
  check(addStream, noPredicates, false, "non-empty continuation predicate set");
  check(addStream, invalidStep, false, "fixed stream step kind is invalid");

  for (Operation *actor : llvm::reverse(actors))
    actor->erase();
  for (Operation *value : llvm::reverse(values))
    value->erase();
  return ok;
}

bool checkTokenPlaneCapabilityAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  llvm::SmallVector<Operation *> values;
  llvm::SmallVector<Operation *> actors;
  auto poison = [&](Type type) -> Value {
    auto value = ub::PoisonOp::create(builder, loc, type);
    values.push_back(value);
    return value;
  };
  auto makeCarry = [&](Type payload) -> Operation * {
    Operation *carry = dataflow::CarryOp::create(
        builder, loc, payload, poison(builder.getI1Type()), poison(payload),
        poison(payload));
    actors.push_back(carry);
    return carry;
  };

  Operation *scalar = makeCarry(builder.getI32Type());
  Operation *floating = makeCarry(builder.getF32Type());
  Operation *fixedVector =
      makeCarry(VectorType::get({2, 3}, builder.getI16Type()));
  Operation *rankZeroVector =
      makeCarry(VectorType::get({}, builder.getI16Type()));
  Operation *scalableVector =
      makeCarry(VectorType::get({2}, builder.getI16Type(), {true}));
  Operation *indexVector =
      makeCarry(VectorType::get({2}, builder.getIndexType()));
  Operation *none = makeCarry(builder.getNoneType());
  Operation *memory = makeCarry(MemRefType::get({4}, builder.getI32Type()));
  FamilyCapabilityParams tokenPlane = TokenPlaneParams{};

  bool ok = true;
  auto check = [&](Operation *op, bool admitted, llvm::StringRef reason) {
    if (std::optional<dataflow::CanonicalActorSchemaProjection> projection =
            projectActor(op, ok))
      ok &= expectAdmission(ImplementationFamilyId::LoopCarry, &tokenPlane,
                            *projection, admitted, reason);
  };
  check(scalar, true, {});
  check(floating, true, {});
  check(fixedVector, true, {});
  check(rankZeroVector, false, "token-plane payload");
  check(scalableVector, false, "token-plane payload");
  check(indexVector, false, "token-plane payload");
  check(none, true, {});
  check(memory, false, "token-plane payload");

  for (Operation *actor : llvm::reverse(actors))
    actor->erase();
  for (Operation *value : llvm::reverse(values))
    value->erase();
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, math::MathDialect, ub::UBDialect,
                  dataflow::DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkDescriptorRelations();
  ok &= checkMembershipAndWrongFamily();
  ok &= checkIntegerCapabilityAdmission(context);
  ok &= checkFloatingCapabilityAdmission(context);
  ok &= checkCastCapabilityAdmission(context);
  ok &= checkLoopCapabilityAdmission(context);
  ok &= checkTokenPlaneCapabilityAdmission(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
