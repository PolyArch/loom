//===- OperationSchemaTest.cpp - Canonical operation schema anchors -------===//
//
// Anchors the generated operation registry and the closed typed projection
// owned by CanonicalDataflowActorOpInterface. Each semantic category has one
// discriminating example; broad operation matrices belong to dialect tests.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/OperationSchema.h"

#include "Common/SpecialMathAccuracy.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>

using namespace mlir;
using namespace dataflow;

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

template <typename OpTy>
bool nativeSemanticInterfacesAreClassified(OperationSemanticsCase semantics) {
  if constexpr (OpTy::template hasTrait<
                    arith::ArithRoundingModeInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithFloatingPoint;
  if constexpr (OpTy::template hasTrait<arith::ArithFastMathInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithFloatingPoint ||
           semantics == OperationSemanticsCase::SpecialMathAccuracy ||
           semantics == OperationSemanticsCase::ArithFloatCompare;
  if constexpr (OpTy::template hasTrait<
                    arith::ArithIntegerOverflowFlagsInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithIntegerOverflow;
  if constexpr (OpTy::template hasTrait<
                    arith::ArithNonNegFlagInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithNonNegative;
  return true;
}

bool checkRegistry(MLIRContext &context) {
  bool ok = true;
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase, SelectorKind, SelectorValue,      \
                              ElementwiseDecomposable)                         \
  if (!nativeSemanticInterfacesAreClassified<OpClass>(                         \
          OperationSemanticsCase::SemanticsCase)) {                            \
    llvm::errs() << OpClass::getOperationName()                                \
                 << " leaves a native semantic interface unclassified\n";      \
    ok = false;                                                                \
  }
#include "Dataflow/IR/OperationSchemas.inc"

  llvm::StringSet<> spellings;
  const std::uint32_t count = operationSchemaCount();
  if (count == 0) {
    llvm::errs() << "the registry declares no operation schema\n";
    return false;
  }
  for (std::uint32_t index = 0; index < count; ++index) {
    auto schema = static_cast<OperationSchemaId>(index);
    llvm::StringRef spelling = operationSchemaSpelling(schema);
    if (spelling.empty() || !spellings.insert(spelling).second) {
      llvm::errs() << "schema " << index << " has no unique spelling\n";
      ok = false;
      continue;
    }
    std::optional<OperationSchemaId> resolved = findOperationSchema(spelling);
    if (!resolved || *resolved != schema) {
      llvm::errs() << "spelling '" << spelling << "' does not round-trip\n";
      ok = false;
    }
  }
  if (findOperationSchema("arith.no_such_op")) {
    llvm::errs() << "an unregistered spelling resolved to a schema\n";
    ok = false;
  }

  OpFixture fixture(context);
  Value lhs = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                        fixture.builder.getI32IntegerAttr(1));
  Value rhs = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                        fixture.builder.getI32IntegerAttr(2));
  Operation *add =
      arith::AddIOp::create(fixture.builder, fixture.loc, lhs, rhs);
  std::optional<OperationSchemaId> schema = operationSchemaOf(add);
  if (!schema || *schema != OperationSchemaId::ArithAddI ||
      operationSchemaSpelling(*schema) != add->getName().getStringRef() ||
      actorKind(*schema) != CanonicalDataflowActorKind::Compute) {
    llvm::errs() << "arith.addi disagrees with its generated identity\n";
    ok = false;
  }

  Operation *unregistered =
      func::FuncOp::create(fixture.builder, fixture.loc, "not_an_actor",
                           fixture.builder.getFunctionType({}, {}));
  if (operationSchemaOf(unregistered)) {
    llvm::errs() << "an unregistered operation resolved to a schema\n";
    ok = false;
  }
  return ok;
}

std::optional<CanonicalActorSchemaProjection> projectActor(Operation *op,
                                                           bool &ok) {
  auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
  if (!actor) {
    llvm::errs() << op->getName().getStringRef()
                 << " has no canonical actor interface\n";
    ok = false;
    return std::nullopt;
  }

  llvm::Expected<CanonicalActorSchemaProjection> projection =
      actor.projectCanonicalActorSchemaProjection();
  if (!projection) {
    llvm::errs() << llvm::toString(projection.takeError()) << '\n';
    ok = false;
    return std::nullopt;
  }
  std::optional<OperationSchemaId> schema = operationSchemaOf(op);
  if (!schema || actor.getOperationSchema() != *schema ||
      actor.getCanonicalActorKind() != actorKind(*schema)) {
    llvm::errs() << op->getName().getStringRef()
                 << " interface identity disagrees with the registry\n";
    ok = false;
  }
  if (failed(actor.verifyCanonicalActorInstance())) {
    llvm::errs() << op->getName().getStringRef()
                 << " interface verifier rejected its projection\n";
    ok = false;
  }

  llvm::Expected<CanonicalActorSchemaProjection> direct =
      projectRegisteredActorSchemaProjection(op);
  if (!direct) {
    llvm::errs() << llvm::toString(direct.takeError()) << '\n';
    ok = false;
  } else if (*direct != *projection) {
    llvm::errs() << op->getName().getStringRef()
                 << " interface projection disagrees with the registry\n";
    ok = false;
  }

  llvm::Expected<TransitionDescriptorIdentity> descriptor =
      actor.getTransitionDescriptorIdentity();
  if (!descriptor) {
    llvm::errs() << llvm::toString(descriptor.takeError()) << '\n';
    ok = false;
  } else if (descriptor->projection != *projection) {
    llvm::errs() << op->getName().getStringRef()
                 << " transition identity disagrees with the projection\n";
    ok = false;
  }
  return *projection;
}

void expectProjectionDelta(Operation *plain, Operation *qualified,
                           llvm::StringRef state, bool &ok) {
  std::optional<CanonicalActorSchemaProjection> first = projectActor(plain, ok);
  std::optional<CanonicalActorSchemaProjection> second =
      projectActor(qualified, ok);
  if (first && second && *first == *second) {
    llvm::errs() << state << " did not change the actor projection\n";
    ok = false;
  }
}

bool checkSemanticProjection(MLIRContext &context) {
  OpFixture fixture(context);
  Value one = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                        fixture.builder.getI32IntegerAttr(1));
  Value two = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                        fixture.builder.getI32IntegerAttr(2));
  Value oneF = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                         fixture.builder.getF32FloatAttr(1.0));
  Value twoF = arith::ConstantOp::create(fixture.builder, fixture.loc,
                                         fixture.builder.getF32FloatAttr(2.0));

  Operation *signedLess = arith::CmpIOp::create(
      fixture.builder, fixture.loc, arith::CmpIPredicate::slt, one, two);
  Operation *equal = arith::CmpIOp::create(fixture.builder, fixture.loc,
                                           arith::CmpIPredicate::eq, one, two);
  Operation *signedLessAgain = arith::CmpIOp::create(
      fixture.builder, fixture.loc, arith::CmpIPredicate::slt, one, two);

  Operation *plainFloat = arith::AddFOp::create(
      fixture.builder, fixture.loc, oneF, twoF, arith::FastMathFlags::none,
      arith::RoundingModeAttr{});
  Operation *fastFloat = arith::AddFOp::create(
      fixture.builder, fixture.loc, oneF, twoF, arith::FastMathFlags::nnan,
      arith::RoundingModeAttr{});
  Operation *roundedFloat = arith::AddFOp::create(
      fixture.builder, fixture.loc, oneF, twoF, arith::FastMathFlags::none,
      arith::RoundingModeAttr::get(&context, arith::RoundingMode::downward));

  auto plainOverflow =
      arith::AddIOp::create(fixture.builder, fixture.loc, one, two);
  auto signedOverflow =
      arith::AddIOp::create(fixture.builder, fixture.loc, one, two);
  signedOverflow.setOverflowFlags(arith::IntegerOverflowFlags::nsw);
  Operation *plainDivision =
      arith::DivSIOp::create(fixture.builder, fixture.loc, one, two, false);
  Operation *exactDivision =
      arith::DivSIOp::create(fixture.builder, fixture.loc, one, two, true);
  Operation *plainExtension = arith::ExtUIOp::create(
      fixture.builder, fixture.loc, fixture.builder.getI64Type(), one, false);
  Operation *nonNegativeExtension = arith::ExtUIOp::create(
      fixture.builder, fixture.loc, fixture.builder.getI64Type(), one, true);

  bool ok = true;
  std::optional<CanonicalActorSchemaProjection> first =
      projectActor(signedLess, ok);
  std::optional<CanonicalActorSchemaProjection> second =
      projectActor(equal, ok);
  std::optional<CanonicalActorSchemaProjection> third =
      projectActor(signedLessAgain, ok);
  if (first && second && third) {
    if (first->schema != second->schema || first->type != second->type ||
        *first == *second) {
      llvm::errs() << "integer predicate identity is not exact\n";
      ok = false;
    }
    if (*first != *third) {
      llvm::errs() << "equal actor semantics produced different projections\n";
      ok = false;
    }
  }

  expectProjectionDelta(plainFloat, fastFloat, "fast-math state", ok);
  expectProjectionDelta(plainFloat, roundedFloat, "rounding mode", ok);
  expectProjectionDelta(plainOverflow, signedOverflow, "integer overflow state",
                        ok);
  expectProjectionDelta(plainDivision, exactDivision, "exact state", ok);
  expectProjectionDelta(plainExtension, nonNegativeExtension, "nneg state", ok);
  return ok;
}

bool expectProjectionFailure(Operation *op, llvm::StringRef state) {
  llvm::Expected<CanonicalActorSchemaProjection> projection =
      projectRegisteredActorSchemaProjection(op);
  if (!projection) {
    llvm::consumeError(projection.takeError());
    return true;
  }
  llvm::errs() << op->getName().getStringRef() << " projected " << state
               << '\n';
  return false;
}

bool checkSpecialMathAccuracyProjection(MLIRContext &context) {
  OpFixture fixture(context);
  Value input = fixture.poison(fixture.builder.getF32Type());
  auto makeSin = [&](arith::FastMathFlags flags,
                     std::optional<llvm::StringRef> accuracy) -> Operation * {
    Operation *op =
        math::SinOp::create(fixture.builder, fixture.loc, input, flags);
    if (accuracy)
      op->setDiscardableAttr(
          fixture.builder.getStringAttr(loom::kSpecialMathAccuracyAttrName),
          fixture.builder.getStringAttr(*accuracy));
    return op;
  };

  Operation *missing = makeSin(arith::FastMathFlags::none, std::nullopt);
  Operation *strict = makeSin(arith::FastMathFlags::none, "CorrectlyRounded");
  Operation *relaxed = makeSin(arith::FastMathFlags::afn, "Max2Ulp");
  Operation *unauthorized = makeSin(arith::FastMathFlags::none, "Max1Ulp");
  Operation *malformed =
      makeSin(arith::FastMathFlags::afn, "approximately_two_ulp");

  bool ok = true;
  ok &= expectProjectionFailure(missing, "a missing special-math tier");
  ok &= expectProjectionFailure(unauthorized, "a relaxed tier without afn");
  ok &= expectProjectionFailure(malformed, "an unknown special-math tier");

  std::optional<CanonicalActorSchemaProjection> strictProjection =
      projectActor(strict, ok);
  std::optional<CanonicalActorSchemaProjection> relaxedProjection =
      projectActor(relaxed, ok);
  const auto *strictPayload =
      strictProjection
          ? std::get_if<SpecialMathPayload>(&strictProjection->payload)
          : nullptr;
  const auto *relaxedPayload =
      relaxedProjection
          ? std::get_if<SpecialMathPayload>(&relaxedProjection->payload)
          : nullptr;
  if (!strictPayload ||
      strictPayload->accuracy !=
          loom::SpecialMathAccuracyTier::CorrectlyRounded ||
      strictPayload->flags != arith::FastMathFlags::none || !relaxedPayload ||
      relaxedPayload->accuracy != loom::SpecialMathAccuracyTier::Max2Ulp ||
      relaxedPayload->flags != arith::FastMathFlags::afn ||
      (strictProjection && relaxedProjection &&
       *strictProjection == *relaxedProjection)) {
    llvm::errs() << "special-math accuracy was not projected exactly\n";
    ok = false;
  }

  Operation *ordinary = arith::AddFOp::create(
      fixture.builder, fixture.loc, input, input, arith::FastMathFlags::afn,
      arith::RoundingModeAttr{});
  ordinary->setDiscardableAttr(
      fixture.builder.getStringAttr(loom::kSpecialMathAccuracyAttrName),
      fixture.builder.getStringAttr("Max2Ulp"));
  ok &= expectProjectionFailure(ordinary,
                                "a special-math tier on an ordinary actor");
  return ok;
}

bool expectRejectedActor(Operation *op, llvm::StringRef state) {
  bool ok = true;
  if (operationSchemaOf(op) || classifyCanonicalDataflowActor(op) ||
      isCanonicalDataflowActor(op)) {
    llvm::errs() << op->getName().getStringRef() << " admitted " << state
                 << '\n';
    ok = false;
  }
  llvm::Expected<CanonicalActorSchemaProjection> direct =
      projectRegisteredActorSchemaProjection(op);
  if (direct) {
    llvm::errs() << op->getName().getStringRef() << " projected " << state
                 << '\n';
    ok = false;
  } else {
    llvm::consumeError(direct.takeError());
  }
  auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
  if (!actor) {
    llvm::errs() << op->getName().getStringRef()
                 << " has no interface for fail-closed admission\n";
    return false;
  }
  if (succeeded(actor.verifyCanonicalActorInstance())) {
    llvm::errs() << op->getName().getStringRef() << " verified " << state
                 << '\n';
    ok = false;
  }
  return ok;
}

bool checkVectorStructure(MLIRContext &context) {
  OpFixture fixture(context);
  VectorType fixed = VectorType::get({2}, fixture.builder.getI32Type());
  Value lhs = fixture.poison(fixed);
  Value rhs = fixture.poison(fixed);
  Value scalar = fixture.poison(fixture.builder.getI32Type());
  Value dynamicIndex =
      arith::ConstantIndexOp::create(fixture.builder, fixture.loc, 0);

  Operation *extractFirst =
      vector::ExtractOp::create(fixture.builder, fixture.loc, lhs, 0);
  Operation *extractSecond =
      vector::ExtractOp::create(fixture.builder, fixture.loc, lhs, 1);
  Operation *insertSecond =
      vector::InsertOp::create(fixture.builder, fixture.loc, scalar, lhs, 1);
  SmallVector<OpFoldResult> dynamicPosition = {dynamicIndex};
  Operation *dynamicExtract = vector::ExtractOp::create(
      fixture.builder, fixture.loc, lhs, dynamicPosition);
  Operation *poisonShuffle =
      vector::ShuffleOp::create(fixture.builder, fixture.loc, fixed, lhs, rhs,
                                llvm::ArrayRef<int64_t>{0, -1});

  bool ok = true;
  std::optional<CanonicalActorSchemaProjection> first =
      projectActor(extractFirst, ok);
  std::optional<CanonicalActorSchemaProjection> second =
      projectActor(extractSecond, ok);
  std::optional<CanonicalActorSchemaProjection> inserted =
      projectActor(insertSecond, ok);
  std::optional<CanonicalActorSchemaProjection> dynamic =
      projectActor(dynamicExtract, ok);
  std::optional<CanonicalActorSchemaProjection> shuffled =
      projectActor(poisonShuffle, ok);

  VectorType indexVector = VectorType::get({2}, fixture.builder.getIndexType());
  Value indexLhs = fixture.poison(indexVector);
  Value indexRhs = fixture.poison(indexVector);
  Value indexScalar = fixture.poison(fixture.builder.getIndexType());
  Operation *indexExtract =
      vector::ExtractOp::create(fixture.builder, fixture.loc, indexLhs, 0);
  Operation *indexInsert = vector::InsertOp::create(
      fixture.builder, fixture.loc, indexScalar, indexLhs, 1);
  Operation *indexShuffle = vector::ShuffleOp::create(
      fixture.builder, fixture.loc, indexVector, indexLhs, indexRhs,
      llvm::ArrayRef<int64_t>{1, 0});
  std::optional<CanonicalActorSchemaProjection> projectedIndexExtract =
      projectActor(indexExtract, ok);
  std::optional<CanonicalActorSchemaProjection> projectedIndexInsert =
      projectActor(indexInsert, ok);
  std::optional<CanonicalActorSchemaProjection> projectedIndexShuffle =
      projectActor(indexShuffle, ok);
  if (!projectedIndexExtract || !projectedIndexInsert ||
      !projectedIndexShuffle ||
      projectedIndexExtract->schema != OperationSchemaId::VectorExtract ||
      projectedIndexInsert->schema != OperationSchemaId::VectorInsert ||
      projectedIndexShuffle->schema != OperationSchemaId::VectorShuffle) {
    llvm::errs() << "vector<index> structural actors were not registered\n";
    ok = false;
  }
  if (first && second && *first == *second) {
    llvm::errs() << "vector positions share one actor projection\n";
    ok = false;
  }
  if (inserted) {
    const auto *position =
        std::get_if<VectorStaticPositionPayload>(&inserted->payload);
    if (!position || position->position != std::vector<std::int64_t>{1}) {
      llvm::errs() << "vector.insert lost its static position\n";
      ok = false;
    }
  }
  if (dynamic) {
    const auto *position =
        std::get_if<VectorStaticPositionPayload>(&dynamic->payload);
    if (!position ||
        position->position != std::vector<std::int64_t>{ShapedType::kDynamic} ||
        dynamic->type.getNumInputs() != 2) {
      llvm::errs() << "vector.extract lost its dynamic-position structure\n";
      ok = false;
    }
  }
  if (shuffled) {
    const auto *mask =
        std::get_if<VectorShuffleMaskPayload>(&shuffled->payload);
    if (!mask || mask->mask != std::vector<std::int64_t>{0, -1}) {
      llvm::errs() << "vector.shuffle lost its poison lane\n";
      ok = false;
    }
  }

  VectorType scalable =
      VectorType::get({2}, fixture.builder.getI32Type(), {true});
  Operation *scalableExtract = vector::ExtractOp::create(
      fixture.builder, fixture.loc, fixture.poison(scalable), 0);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  ok &= expectRejectedActor(scalableExtract, "a scalable structural vector");
  return ok;
}

bool checkPoisonAndAggregateState(MLIRContext &context) {
  OpFixture fixture(context);
  Type i32 = fixture.builder.getI32Type();
  Value input = fixture.poison(i32);
  Operation *poisonOnZero = LLVM::CountLeadingZerosOp::create(
      fixture.builder, fixture.loc, i32, input, true);
  Operation *definedOnZero = LLVM::CountLeadingZerosOp::create(
      fixture.builder, fixture.loc, i32, input, false);
  Operation *poisonOnMin =
      LLVM::AbsOp::create(fixture.builder, fixture.loc, i32, input, true);
  Operation *definedOnMin =
      LLVM::AbsOp::create(fixture.builder, fixture.loc, i32, input, false);
  Operation *disjointOr =
      LLVM::OrOp::create(fixture.builder, fixture.loc, i32, input, input, true);
  Operation *ordinaryOr = LLVM::OrOp::create(fixture.builder, fixture.loc, i32,
                                             input, input, false);

  bool ok = true;
  std::optional<CanonicalActorSchemaProjection> ctlz =
      projectActor(poisonOnZero, ok);
  std::optional<CanonicalActorSchemaProjection> abs =
      projectActor(poisonOnMin, ok);
  std::optional<CanonicalActorSchemaProjection> disjoint =
      projectActor(disjointOr, ok);
  const auto *zeroPoison =
      ctlz ? std::get_if<ZeroPoisonPayload>(&ctlz->payload) : nullptr;
  const auto *minPoison =
      abs ? std::get_if<IntegerMinPoisonPayload>(&abs->payload) : nullptr;
  const auto *disjointPolicy =
      disjoint ? std::get_if<DisjointPayload>(&disjoint->payload) : nullptr;
  if (!zeroPoison || !zeroPoison->isZeroPoison || !minPoison ||
      !minPoison->isIntMinPoison || !disjointPolicy ||
      !disjointPolicy->isDisjoint) {
    llvm::errs() << "LLVM poison controls were not projected exactly\n";
    ok = false;
  }
  if (operationSchemaOf(definedOnZero) || operationSchemaOf(definedOnMin) ||
      operationSchemaOf(ordinaryOr)) {
    llvm::errs() << "a poison-free LLVM alias was admitted\n";
    ok = false;
  }

  Type aggregateType = LLVM::LLVMStructType::getLiteral(&context, {i32, i32});
  Value aggregate =
      LLVM::UndefOp::create(fixture.builder, fixture.loc, aggregateType);
  Operation *extractFirst = LLVM::ExtractValueOp::create(
      fixture.builder, fixture.loc, aggregate, {0});
  Operation *extractSecond = LLVM::ExtractValueOp::create(
      fixture.builder, fixture.loc, aggregate, {1});
  Operation *insertSecond =
      LLVM::InsertValueOp::create(fixture.builder, fixture.loc, aggregate,
                                  input, llvm::ArrayRef<int64_t>{1});
  std::optional<CanonicalActorSchemaProjection> first =
      projectActor(extractFirst, ok);
  std::optional<CanonicalActorSchemaProjection> second =
      projectActor(extractSecond, ok);
  std::optional<CanonicalActorSchemaProjection> inserted =
      projectActor(insertSecond, ok);
  if (first && second && *first == *second) {
    llvm::errs() << "aggregate positions share one actor projection\n";
    ok = false;
  }
  const auto *position =
      inserted ? std::get_if<AggregatePositionPayload>(&inserted->payload)
               : nullptr;
  if (!position || position->position != std::vector<std::int64_t>{1}) {
    llvm::errs() << "llvm.insertvalue lost its aggregate position\n";
    ok = false;
  }
  return ok;
}

bool checkRegisteredLLVMIntrinsicSelectors(MLIRContext &context) {
  OpFixture fixture(context);
  Type f32 = fixture.builder.getF32Type();
  Type i16 = fixture.builder.getI16Type();
  Value input = fixture.poison(f32);

  Operation *signedSaturating = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptosi.sat.i16.f32"),
      ValueRange{input});
  Operation *unsignedSaturating = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptoui.sat.i16.f32"),
      ValueRange{input});
  Operation *wrongOverload = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptosi.sat.i32.f32"),
      ValueRange{input});
  Operation *relaxed = LLVM::CallIntrinsicOp::create(
      fixture.builder, fixture.loc, i16,
      fixture.builder.getStringAttr("llvm.fptosi.sat.i16.f32"),
      ValueRange{input});
  llvm::cast<LLVM::CallIntrinsicOp>(relaxed).setFastmathFlags(
      LLVM::FastmathFlags::nnan);

  bool ok = true;
  std::optional<OperationSchemaId> signedSchema =
      findOperationSchema("llvm.fptosi.sat");
  std::optional<OperationSchemaId> unsignedSchema =
      findOperationSchema("llvm.fptoui.sat");
  if (!signedSchema || !unsignedSchema || signedSchema == unsignedSchema) {
    llvm::errs() << "saturating conversion selectors are not distinct "
                    "registered schemas\n";
    ok = false;
  }
  if (signedSchema && operationSchemaOf(signedSaturating) != signedSchema) {
    llvm::errs() << "llvm.fptosi.sat did not resolve through its exact "
                    "intrinsic selector\n";
    ok = false;
  }
  if (unsignedSchema &&
      operationSchemaOf(unsignedSaturating) != unsignedSchema) {
    llvm::errs() << "llvm.fptoui.sat did not resolve through its exact "
                    "intrinsic selector\n";
    ok = false;
  }
  if (operationSchemaOf(wrongOverload)) {
    llvm::errs() << "a noncanonical overloaded intrinsic spelling was "
                    "admitted\n";
    ok = false;
  }
  if (operationSchemaOf(relaxed)) {
    llvm::errs() << "a saturating intrinsic with unregistered semantic state "
                    "was admitted\n";
    ok = false;
  }

  if (signedSchema) {
    std::optional<CanonicalActorSchemaProjection> projection =
        projectActor(signedSaturating, ok);
    if (projection &&
        (projection->schema != *signedSchema ||
         projection->type != fixture.builder.getFunctionType({f32}, {i16}) ||
         !std::holds_alternative<NoPayload>(projection->payload))) {
      llvm::errs() << "llvm.fptosi.sat lost its exact typed projection\n";
      ok = false;
    }
  }
  return ok;
}

bool checkPointerProjection(MLIRContext &context) {
  OpFixture fixture(context);
  Type pointer = LLVM::LLVMPointerType::get(&context);
  Type i32 = fixture.builder.getI32Type();
  Type i64 = fixture.builder.getI64Type();
  Value base = fixture.poison(pointer);
  Value dynamicIndex = fixture.poison(i64);

  const SmallVector<LLVM::GEPArg, 1> dynamicIndices{dynamicIndex};
  const SmallVector<LLVM::GEPArg, 1> constantIndices{3};
  Operation *dynamic =
      LLVM::GEPOp::create(fixture.builder, fixture.loc, pointer, i32, base,
                          dynamicIndices, LLVM::GEPNoWrapFlags::none);
  Operation *constant =
      LLVM::GEPOp::create(fixture.builder, fixture.loc, pointer, i32, base,
                          constantIndices, LLVM::GEPNoWrapFlags::none);
  Operation *inbounds =
      LLVM::GEPOp::create(fixture.builder, fixture.loc, pointer, i32, base,
                          constantIndices, LLVM::GEPNoWrapFlags::inbounds);

  bool ok = true;
  auto dynamicProjection = projectActor(dynamic, ok);
  auto constantProjection = projectActor(constant, ok);
  auto inboundsProjection = projectActor(inbounds, ok);
  if (!dynamicProjection || !constantProjection || !inboundsProjection)
    return false;

  const auto *dynamicPayload =
      std::get_if<GetElementPtrPayload>(&dynamicProjection->payload);
  const auto *constantPayload =
      std::get_if<GetElementPtrPayload>(&constantProjection->payload);
  const auto *inboundsPayload =
      std::get_if<GetElementPtrPayload>(&inboundsProjection->payload);
  if (dynamicProjection->schema != OperationSchemaId::LLVMGetElementPtr ||
      !dynamicPayload || dynamicPayload->sourceElementType != i32 ||
      dynamicPayload->rawConstantIndices !=
          std::vector<std::int32_t>{LLVM::GEPOp::kDynamicIndex} ||
      dynamicPayload->noWrapFlags != LLVM::GEPNoWrapFlags::none) {
    llvm::errs() << "dynamic GEP lost its exact typed projection\n";
    ok = false;
  }
  if (!constantPayload ||
      constantPayload->rawConstantIndices != std::vector<std::int32_t>{3} ||
      !inboundsPayload ||
      inboundsPayload->noWrapFlags != LLVM::GEPNoWrapFlags::inbounds ||
      *dynamicProjection == *constantProjection ||
      *constantProjection == *inboundsProjection) {
    llvm::errs()
        << "GEP index shape or no-wrap flags did not affect identity\n";
    ok = false;
  }
  return ok;
}

bool checkFailClosedProjection(MLIRContext &context) {
  OpFixture fixture(context);
  Value lhs = fixture.poison(fixture.builder.getI32Type());
  Value rhs = fixture.poison(fixture.builder.getI32Type());
  Operation *bitAnd =
      arith::AndIOp::create(fixture.builder, fixture.loc, lhs, rhs);
  bitAnd->setDiscardableAttr(
      fixture.builder.getStringAttr("dataflow.unclassified_firing_state"),
      fixture.builder.getUnitAttr());

  bool ok = true;
  llvm::Expected<CanonicalActorSchemaProjection> projection =
      projectRegisteredActorSchemaProjection(bitAnd);
  if (projection) {
    llvm::errs() << "an open attribute escaped the closed projection\n";
    ok = false;
  } else {
    llvm::consumeError(projection.takeError());
  }

  OperationState state(fixture.loc, ConstantOp::getOperationName());
  state.addTypes(fixture.builder.getI32Type());
  state.addAttribute("const_value", fixture.builder.getUnitAttr());
  Operation *malformed = fixture.builder.create(state);
  auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(malformed);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  if (!actor) {
    llvm::errs() << "dataflow.constant has no canonical actor interface\n";
    ok = false;
  } else {
    llvm::Expected<CanonicalActorSchemaProjection> malformedProjection =
        actor.projectCanonicalActorSchemaProjection();
    if (malformedProjection) {
      llvm::errs() << "a semantic type mismatch produced a projection\n";
      ok = false;
    } else {
      llvm::consumeError(malformedProjection.takeError());
    }
    if (succeeded(actor.verifyCanonicalActorInstance())) {
      llvm::errs() << "a semantic type mismatch passed actor verification\n";
      ok = false;
    }
  }
  return ok;
}

template <typename ContractProjection, typename AlignmentFn>
bool checkMemoryAlignmentPair(OpFixture &fixture, llvm::StringRef operationName,
                              Attribute fourByteContract,
                              Attribute eightByteContract,
                              AlignmentFn alignment) {
  auto create = [&](Attribute contract) {
    OperationState state(fixture.loc, operationName);
    state.addAttribute("contract", contract);
    return fixture.builder.create(state);
  };
  Operation *fourByteActor = create(fourByteContract);
  Operation *eightByteActor = create(eightByteContract);

  bool ok = true;
  auto project =
      [&](Operation *op) -> std::optional<CanonicalActorSchemaProjection> {
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        projectRegisteredActorSchemaProjection(op);
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };
  auto exactAlignment = [&](const CanonicalActorSchemaProjection &projection,
                            std::uint64_t expected) {
    const auto *memory =
        std::get_if<MemoryContractPayload>(&projection.payload);
    const auto *contract =
        memory ? std::get_if<ContractProjection>(memory) : nullptr;
    if (!contract || alignment(*contract) != expected) {
      llvm::errs() << operationName << " lost source_alignment_bytes "
                   << expected << '\n';
      ok = false;
    }
  };

  std::optional<CanonicalActorSchemaProjection> four = project(fourByteActor);
  std::optional<CanonicalActorSchemaProjection> eight = project(eightByteActor);
  if (four && eight) {
    if (*four == *eight) {
      llvm::errs() << operationName
                   << " erased source_alignment_bytes from identity\n";
      ok = false;
    }
    exactAlignment(*four, 4);
    exactAlignment(*eight, 8);
  }
  return ok;
}

bool checkMemorySourceAlignmentIdentity(MLIRContext &context) {
  OpFixture fixture(context);
  SyncScopeRefAttr scope = SyncScopeRefAttr::get(
      &context, SyncScopeKind::System, StringAttr{}, StringAttr{});
  auto atomic = [&](std::uint64_t alignment) {
    return AtomicAccessContractAttr::get(&context, AtomicOrdering::Acquire,
                                         scope, alignment, std::nullopt,
                                         /*is_volatile=*/false);
  };
  AtomicAccessContractAttr atomicFour = atomic(4);
  AtomicAccessContractAttr atomicEight = atomic(8);

  bool ok = true;
  auto atomicAlignment = [](const AtomicAccessProjection &projection) {
    return projection.sourceAlignmentBytes;
  };
  ok &= checkMemoryAlignmentPair<AtomicAccessProjection>(
      fixture, LoadOp::getOperationName(), atomicFour, atomicEight,
      atomicAlignment);
  ok &= checkMemoryAlignmentPair<AtomicAccessProjection>(
      fixture, StoreOp::getOperationName(), atomicFour, atomicEight,
      atomicAlignment);

  AtomicRmwContractAttr rmwFour =
      AtomicRmwContractAttr::get(&context, AtomicRmwKind::Add, atomicFour);
  AtomicRmwContractAttr rmwEight =
      AtomicRmwContractAttr::get(&context, AtomicRmwKind::Add, atomicEight);
  ok &= checkMemoryAlignmentPair<AtomicRmwProjection>(
      fixture, AtomicRmwOp::getOperationName(), rmwFour, rmwEight,
      [](const AtomicRmwProjection &projection) {
        return projection.access.sourceAlignmentBytes;
      });

  auto compareExchange = [&](std::uint64_t alignment) {
    return CompareExchangeContractAttr::get(
        &context, AtomicOrdering::SeqCst, AtomicOrdering::Monotonic, scope,
        alignment, std::nullopt, /*weak=*/false, /*is_volatile=*/false);
  };
  ok &= checkMemoryAlignmentPair<CompareExchangeProjection>(
      fixture, CmpXchgOp::getOperationName(), compareExchange(4),
      compareExchange(8), [](const CompareExchangeProjection &projection) {
        return projection.sourceAlignmentBytes;
      });
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                  math::MathDialect, ub::UBDialect, vector::VectorDialect,
                  DataflowDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkRegistry(context);
  ok &= checkSemanticProjection(context);
  ok &= checkSpecialMathAccuracyProjection(context);
  ok &= checkVectorStructure(context);
  ok &= checkPoisonAndAggregateState(context);
  ok &= checkRegisteredLLVMIntrinsicSelectors(context);
  ok &= checkPointerProjection(context);
  ok &= checkFailClosedProjection(context);
  ok &= checkMemorySourceAlignmentIdentity(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
