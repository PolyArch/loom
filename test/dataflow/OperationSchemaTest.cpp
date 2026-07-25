//===- OperationSchemaTest.cpp - Canonical operation schema anchors -------===//
//
// Anchors the one generated operation authority:
//
//   * every `OperationSchemaId` round-trips through its stable spelling, and
//     the spelling is the operation's own registered name rather than a second
//     string table;
//   * an operation that is not registered has no schema, so a consumer cannot
//     invent one from a name; and
//   * the identity-critical typed projection, reached through the typed
//     `CanonicalDataflowActorOpInterface`, separates two actors that differ
//     only in an exact semantic attribute and equates two actors that differ
//     in nothing semantic;
//   * poison-free LLVM aliases fail actor admission while the poison-flagged
//     forms project their exact flag through a closed typed case; and
//   * an instance that does not carry the typed state its declared semantic
//     case owns fails closed instead of projecting an empty payload.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/OperationSchema.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace dataflow;

namespace {

template <typename OpTy>
bool nativeSemanticInterfacesAreClassified(OperationSemanticsCase semantics) {
  if constexpr (OpTy::template hasTrait<
                    arith::ArithRoundingModeInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithFloatingPoint;
  if constexpr (OpTy::template hasTrait<arith::ArithFastMathInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithFloatingPoint ||
           semantics == OperationSemanticsCase::ArithFloatCompare;
  if constexpr (OpTy::template hasTrait<
                    arith::ArithIntegerOverflowFlagsInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithIntegerOverflow;
  if constexpr (OpTy::template hasTrait<
                    arith::ArithNonNegFlagInterface::Trait>())
    return semantics == OperationSemanticsCase::ArithNonNegative;
  return true;
}

bool checkNativeSemanticInterfaceCoverage() {
  bool ok = true;
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
  if (!nativeSemanticInterfacesAreClassified<OpClass>(                         \
          OperationSemanticsCase::SemanticsCase)) {                            \
    llvm::errs() << OpClass::getOperationName()                                \
                 << " leaves a native semantic interface unclassified\n";      \
    ok = false;                                                                \
  }
#include "Dataflow/IR/OperationSchemas.inc"
  return ok;
}

bool checkSpellingRoundTrip() {
  bool ok = true;
  llvm::StringSet<> spellings;
  const std::uint32_t count = operationSchemaCount();
  if (count == 0) {
    llvm::errs() << "the registry declares no operation schema\n";
    return false;
  }
  for (std::uint32_t index = 0; index < count; ++index) {
    auto schema = static_cast<OperationSchemaId>(index);
    llvm::StringRef spelling = operationSchemaSpelling(schema);
    if (spelling.empty()) {
      llvm::errs() << "schema " << index << " has no stable spelling\n";
      ok = false;
      continue;
    }
    if (!spellings.insert(spelling).second) {
      llvm::errs() << "spelling '" << spelling << "' names two schemas\n";
      ok = false;
    }
    std::optional<OperationSchemaId> resolved = findOperationSchema(spelling);
    if (!resolved || *resolved != schema) {
      llvm::errs() << "spelling '" << spelling << "' does not resolve back to "
                   << "schema " << index << '\n';
      ok = false;
    }
  }
  return ok;
}

bool checkUnregisteredSpelling() {
  if (findOperationSchema("arith.no_such_op")) {
    llvm::errs() << "an unregistered spelling resolved to a schema\n";
    return false;
  }
  return true;
}

/// The spelling is the operation's own registered name. Reading it from a
/// live operation and from the registry must agree.
bool checkOperationIdentity(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(2));
  Operation *add = arith::AddIOp::create(builder, loc, lhs, rhs);

  bool ok = true;
  std::optional<OperationSchemaId> schema = operationSchemaOf(add);
  if (!schema || *schema != OperationSchemaId::ArithAddI) {
    llvm::errs() << "arith.addi did not resolve to its registered schema\n";
    ok = false;
  } else if (operationSchemaSpelling(*schema) !=
             add->getName().getStringRef()) {
    llvm::errs() << "the registry spelling disagrees with the operation name\n";
    ok = false;
  }
  if (schema && actorKind(*schema) != CanonicalDataflowActorKind::Compute) {
    llvm::errs() << "arith.addi is not classified as a compute actor\n";
    ok = false;
  }

  Operation *unregistered = func::FuncOp::create(
      builder, loc, "not_an_actor", builder.getFunctionType({}, {}));
  if (operationSchemaOf(unregistered)) {
    llvm::errs() << "an unregistered operation resolved to a schema\n";
    ok = false;
  }
  unregistered->erase();
  add->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

/// Two comparisons that agree on schema and function type but differ in their
/// exact predicate are different semantic points; two that agree on everything
/// are the same one.
bool checkTypedSemanticDelta(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(2));
  Operation *signedLess =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, lhs, rhs);
  Operation *equal =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, lhs, rhs);
  Operation *alsoSignedLess =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, lhs, rhs);

  bool ok = true;
  auto project =
      [&ok](Operation *op) -> std::optional<CanonicalActorSchemaProjection> {
    auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
    if (!actor) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not project through the typed actor interface\n";
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
                   << " interface verifier rejected its generated projection\n";
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
    return *projection;
  };

  std::optional<CanonicalActorSchemaProjection> first = project(signedLess);
  std::optional<CanonicalActorSchemaProjection> second = project(equal);
  std::optional<CanonicalActorSchemaProjection> third = project(alsoSignedLess);
  if (first && second && third) {
    if (first->schema != second->schema || first->type != second->type) {
      llvm::errs() << "the two comparisons differ in more than the predicate\n";
      ok = false;
    }
    if (*first == *second) {
      llvm::errs() << "two comparison predicates share one projection\n";
      ok = false;
    }
    if (*first != *third) {
      llvm::errs() << "one comparison predicate produced two projections\n";
      ok = false;
    }
    auto descriptor = llvm::cast<CanonicalDataflowActorOpInterface>(signedLess)
                          .getTransitionDescriptorIdentity();
    auto other = llvm::cast<CanonicalDataflowActorOpInterface>(equal)
                     .getTransitionDescriptorIdentity();
    if (!descriptor || !other) {
      llvm::errs() << "a registered actor produced no transition descriptor\n";
      ok = false;
    } else if (*descriptor == *other) {
      llvm::errs() << "two comparison predicates share one transition "
                      "descriptor\n";
      ok = false;
    }
  }

  alsoSignedLess->erase();
  equal->erase();
  signedLess->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

bool checkFloatingSemanticState(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(1.0));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(2.0));
  Operation *plain =
      arith::AddFOp::create(builder, loc, lhs, rhs, arith::FastMathFlags::none,
                            arith::RoundingModeAttr{});
  Operation *fast =
      arith::AddFOp::create(builder, loc, lhs, rhs, arith::FastMathFlags::nnan,
                            arith::RoundingModeAttr{});
  Operation *rounded = arith::AddFOp::create(
      builder, loc, lhs, rhs, arith::FastMathFlags::none,
      arith::RoundingModeAttr::get(&context, arith::RoundingMode::downward));
  Operation *plainSin =
      math::SinOp::create(builder, loc, lhs, arith::FastMathFlags::none);
  Operation *fastSin =
      math::SinOp::create(builder, loc, lhs, arith::FastMathFlags::nnan);
  Operation *plainFma = math::FmaOp::create(builder, loc, lhs, rhs, lhs,
                                            arith::FastMathFlags::none,
                                            arith::RoundingModeAttr{});
  Operation *roundedFma = math::FmaOp::create(
      builder, loc, lhs, rhs, lhs, arith::FastMathFlags::none,
      arith::RoundingModeAttr::get(&context, arith::RoundingMode::upward));

  bool ok = true;
  auto project =
      [&ok](Operation *op) -> std::optional<CanonicalActorSchemaProjection> {
    auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
    if (!actor) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not project through the typed actor interface\n";
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
    return *projection;
  };

  std::optional<CanonicalActorSchemaProjection> plainProjection =
      project(plain);
  std::optional<CanonicalActorSchemaProjection> fastProjection = project(fast);
  std::optional<CanonicalActorSchemaProjection> roundedProjection =
      project(rounded);
  std::optional<CanonicalActorSchemaProjection> plainSinProjection =
      project(plainSin);
  std::optional<CanonicalActorSchemaProjection> fastSinProjection =
      project(fastSin);
  std::optional<CanonicalActorSchemaProjection> plainFmaProjection =
      project(plainFma);
  std::optional<CanonicalActorSchemaProjection> roundedFmaProjection =
      project(roundedFma);
  if (plainProjection && fastProjection &&
      *plainProjection == *fastProjection) {
    llvm::errs() << "fast-math state did not change the projection\n";
    ok = false;
  }
  if (plainProjection && roundedProjection &&
      *plainProjection == *roundedProjection) {
    llvm::errs() << "rounding mode did not change the projection\n";
    ok = false;
  }
  if (plainSinProjection && fastSinProjection &&
      *plainSinProjection == *fastSinProjection) {
    llvm::errs()
        << "math unary fast-math state did not change the projection\n";
    ok = false;
  }
  if (plainFmaProjection && roundedFmaProjection &&
      *plainFmaProjection == *roundedFmaProjection) {
    llvm::errs() << "math.fma rounding mode did not change the projection\n";
    ok = false;
  }

  roundedFma->erase();
  plainFma->erase();
  fastSin->erase();
  plainSin->erase();
  rounded->erase();
  fast->erase();
  plain->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

bool checkExactAndNonNegativeSemanticState(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(8));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(2));
  Operation *plainDivision =
      arith::DivSIOp::create(builder, loc, lhs, rhs, false);
  Operation *exactDivision =
      arith::DivSIOp::create(builder, loc, lhs, rhs, true);
  Operation *plainExtension =
      arith::ExtUIOp::create(builder, loc, builder.getI64Type(), lhs, false);
  Operation *nonNegativeExtension =
      arith::ExtUIOp::create(builder, loc, builder.getI64Type(), lhs, true);

  bool ok = true;
  auto expectDelta = [&ok](Operation *plain, Operation *qualified,
                           llvm::StringRef state) {
    llvm::Expected<CanonicalActorSchemaProjection> plainProjection =
        projectRegisteredActorSchemaProjection(plain);
    llvm::Expected<CanonicalActorSchemaProjection> qualifiedProjection =
        projectRegisteredActorSchemaProjection(qualified);
    if (!plainProjection) {
      llvm::errs() << llvm::toString(plainProjection.takeError()) << '\n';
      ok = false;
      if (!qualifiedProjection)
        llvm::consumeError(qualifiedProjection.takeError());
      return;
    }
    if (!qualifiedProjection) {
      llvm::errs() << llvm::toString(qualifiedProjection.takeError()) << '\n';
      ok = false;
      return;
    }
    if (*plainProjection == *qualifiedProjection) {
      llvm::errs() << state << " state did not change the projection\n";
      ok = false;
    }
  };

  expectDelta(plainDivision, exactDivision, "exact");
  expectDelta(plainExtension, nonNegativeExtension, "nneg");

  nonNegativeExtension->erase();
  plainExtension->erase();
  exactDivision->erase();
  plainDivision->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

bool checkVectorStructuralSemanticState(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  VectorType vectorType = VectorType::get({2}, builder.getI32Type());
  SmallVector<Attribute> lhsElements = {builder.getI32IntegerAttr(1),
                                        builder.getI32IntegerAttr(2)};
  SmallVector<Attribute> rhsElements = {builder.getI32IntegerAttr(3),
                                        builder.getI32IntegerAttr(4)};
  Value lhs = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(vectorType, lhsElements));
  Value rhs = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(vectorType, rhsElements));
  Value scalar =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(7));

  Operation *extractFirst = vector::ExtractOp::create(builder, loc, lhs, 0);
  Operation *extractSecond = vector::ExtractOp::create(builder, loc, lhs, 1);
  Operation *insertFirst =
      vector::InsertOp::create(builder, loc, scalar, lhs, 0);
  Operation *insertSecond =
      vector::InsertOp::create(builder, loc, scalar, lhs, 1);
  Operation *identityShuffle = vector::ShuffleOp::create(
      builder, loc, vectorType, lhs, rhs, llvm::ArrayRef<int64_t>{0, 1});
  Operation *reverseShuffle = vector::ShuffleOp::create(
      builder, loc, vectorType, lhs, rhs, llvm::ArrayRef<int64_t>{1, 0});
  Operation *poisonShuffle = vector::ShuffleOp::create(
      builder, loc, vectorType, lhs, rhs, llvm::ArrayRef<int64_t>{0, -1});
  Value dynamicIndex = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<OpFoldResult> dynamicPosition = {dynamicIndex};
  Operation *dynamicExtract =
      vector::ExtractOp::create(builder, loc, lhs, dynamicPosition);

  bool ok = true;
  auto project =
      [&ok](Operation *op) -> std::optional<CanonicalActorSchemaProjection> {
    auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
    if (!actor) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not project through the typed actor interface\n";
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
    return *projection;
  };

  std::optional<CanonicalActorSchemaProjection> firstExtract =
      project(extractFirst);
  std::optional<CanonicalActorSchemaProjection> secondExtract =
      project(extractSecond);
  std::optional<CanonicalActorSchemaProjection> firstInsert =
      project(insertFirst);
  std::optional<CanonicalActorSchemaProjection> secondInsert =
      project(insertSecond);
  std::optional<CanonicalActorSchemaProjection> identity =
      project(identityShuffle);
  std::optional<CanonicalActorSchemaProjection> reverse =
      project(reverseShuffle);
  std::optional<CanonicalActorSchemaProjection> poison = project(poisonShuffle);
  std::optional<CanonicalActorSchemaProjection> dynamic =
      project(dynamicExtract);
  if (firstExtract && secondExtract && *firstExtract == *secondExtract) {
    llvm::errs() << "two vector.extract positions share one projection\n";
    ok = false;
  }
  if (firstInsert && secondInsert && *firstInsert == *secondInsert) {
    llvm::errs() << "two vector.insert positions share one projection\n";
    ok = false;
  }
  if (identity && reverse && *identity == *reverse) {
    llvm::errs() << "two vector.shuffle masks share one projection\n";
    ok = false;
  }
  if (firstExtract) {
    auto *position =
        std::get_if<VectorStaticPositionPayload>(&firstExtract->payload);
    if (!position || position->position != std::vector<std::int64_t>{0}) {
      llvm::errs() << "vector.extract lost its static position\n";
      ok = false;
    }
  }
  if (poison) {
    auto *mask = std::get_if<VectorShuffleMaskPayload>(&poison->payload);
    if (!mask || mask->mask != std::vector<std::int64_t>{0, -1}) {
      llvm::errs() << "vector.shuffle lost its poison mask lane\n";
      ok = false;
    }
  }
  if (dynamic) {
    auto *position =
        std::get_if<VectorStaticPositionPayload>(&dynamic->payload);
    if (!position ||
        position->position != std::vector<std::int64_t>{ShapedType::kDynamic} ||
        dynamic->type.getNumInputs() != 2) {
      llvm::errs() << "vector.extract did not separate dynamic operands from "
                      "static position structure\n";
      ok = false;
    }
  }

  dynamicExtract->erase();
  dynamicIndex.getDefiningOp()->erase();
  poisonShuffle->erase();
  reverseShuffle->erase();
  identityShuffle->erase();
  insertSecond->erase();
  insertFirst->erase();
  extractSecond->erase();
  extractFirst->erase();
  scalar.getDefiningOp()->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

bool expectStructuralVectorRejection(Operation *op, llvm::StringRef state) {
  bool ok = true;
  auto reportAdmission = [&](llvm::StringRef surface) {
    llvm::errs() << op->getName().getStringRef() << " admitted " << state
                 << " through " << surface << '\n';
    ok = false;
  };

  if (operationSchemaOf(op))
    reportAdmission("operationSchemaOf");
  if (classifyCanonicalDataflowActor(op))
    reportAdmission("classifyCanonicalDataflowActor");
  if (isCanonicalDataflowActor(op))
    reportAdmission("isCanonicalDataflowActor");
  if (isCanonicalDataflowActor(op, CanonicalDataflowActorKind::Compute))
    reportAdmission("kind-specific isCanonicalDataflowActor");

  llvm::Expected<CanonicalActorSchemaProjection> direct =
      projectRegisteredActorSchemaProjection(op);
  if (direct)
    reportAdmission("projectRegisteredActorSchemaProjection");
  else
    llvm::consumeError(direct.takeError());

  auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
  if (!actor) {
    llvm::errs() << op->getName().getStringRef()
                 << " has no canonical actor interface for rejecting " << state
                 << '\n';
    return false;
  }
  llvm::Expected<CanonicalActorSchemaProjection> projected =
      actor.projectCanonicalActorSchemaProjection();
  if (projected)
    reportAdmission("CanonicalDataflowActorOpInterface projection");
  else
    llvm::consumeError(projected.takeError());
  if (succeeded(actor.verifyCanonicalActorInstance()))
    reportAdmission("CanonicalDataflowActorOpInterface verifier");
  return ok;
}

bool checkStructuralVectorAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  VectorType scalableType = VectorType::get({2}, builder.getI32Type(), {true});
  VectorType fixedType = VectorType::get({2}, builder.getI32Type());
  VectorType rankZeroType = VectorType::get({}, builder.getI32Type());
  VectorType indexVectorType = VectorType::get({2}, builder.getIndexType());
  Value scalable = ub::PoisonOp::create(builder, loc, scalableType);
  Value fixed = ub::PoisonOp::create(builder, loc, fixedType);
  Value rankZeroLhs = ub::PoisonOp::create(builder, loc, rankZeroType);
  Value rankZeroRhs = ub::PoisonOp::create(builder, loc, rankZeroType);
  Value indexLhs = ub::PoisonOp::create(builder, loc, indexVectorType);
  Value indexRhs = ub::PoisonOp::create(builder, loc, indexVectorType);
  Value scalar =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(7));
  Value indexScalar = arith::ConstantIndexOp::create(builder, loc, 0);

  Operation *scalableExtract =
      vector::ExtractOp::create(builder, loc, scalable, 0);
  Operation *mixedScalableExtract =
      vector::ExtractOp::create(builder, loc, scalableType, fixed, ValueRange{},
                                builder.getDenseI64ArrayAttr({}));
  Operation *scalableInsert =
      vector::InsertOp::create(builder, loc, scalar, scalable, 0);
  Operation *scalableShuffle =
      vector::ShuffleOp::create(builder, loc, scalableType, scalable, scalable,
                                llvm::ArrayRef<int64_t>{0, 1});

  Operation *rankZeroExtract =
      vector::ExtractOp::create(builder, loc, rankZeroLhs);
  Operation *rankZeroInsert =
      vector::InsertOp::create(builder, loc, scalar, rankZeroLhs);
  Operation *rankZeroShuffle =
      vector::ShuffleOp::create(builder, loc, fixedType, rankZeroLhs,
                                rankZeroRhs, llvm::ArrayRef<int64_t>{0, 1});

  Operation *indexExtract =
      vector::ExtractOp::create(builder, loc, indexLhs, 0);
  Operation *indexInsert =
      vector::InsertOp::create(builder, loc, indexScalar, indexLhs, 0);
  Operation *indexShuffle =
      vector::ShuffleOp::create(builder, loc, indexVectorType, indexLhs,
                                indexRhs, llvm::ArrayRef<int64_t>{0, 1});

  bool ok = true;
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  ok &= expectStructuralVectorRejection(scalableExtract,
                                        "a scalable structural vector");
  ok &= expectStructuralVectorRejection(mixedScalableExtract,
                                        "a scalable structural vector result");
  ok &= expectStructuralVectorRejection(scalableInsert,
                                        "a scalable structural vector");
  ok &= expectStructuralVectorRejection(scalableShuffle,
                                        "a scalable structural vector");
  ok &= expectStructuralVectorRejection(rankZeroExtract,
                                        "a rank-zero structural vector");
  ok &= expectStructuralVectorRejection(rankZeroInsert,
                                        "a rank-zero structural vector");
  ok &= expectStructuralVectorRejection(rankZeroShuffle,
                                        "a rank-zero structural vector");
  ok &= expectStructuralVectorRejection(indexExtract,
                                        "an index-element structural vector");
  ok &= expectStructuralVectorRejection(indexInsert,
                                        "an index-element structural vector");
  ok &= expectStructuralVectorRejection(indexShuffle,
                                        "an index-element structural vector");

  indexShuffle->erase();
  indexInsert->erase();
  indexExtract->erase();
  rankZeroShuffle->erase();
  rankZeroInsert->erase();
  rankZeroExtract->erase();
  scalableShuffle->erase();
  scalableInsert->erase();
  mixedScalableExtract->erase();
  scalableExtract->erase();
  indexScalar.getDefiningOp()->erase();
  scalar.getDefiningOp()->erase();
  indexRhs.getDefiningOp()->erase();
  indexLhs.getDefiningOp()->erase();
  rankZeroRhs.getDefiningOp()->erase();
  rankZeroLhs.getDefiningOp()->erase();
  fixed.getDefiningOp()->erase();
  scalable.getDefiningOp()->erase();
  return ok;
}

/// Only the poison-flagged LLVM forms are canonical actors. The flag-free
/// forms normalize to math operations and must fail admission before a
/// projection can be formed.
bool checkPoisonFlagAdmission(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value input =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Type i32 = builder.getI32Type();

  Operation *poisonOnZero =
      LLVM::CountLeadingZerosOp::create(builder, loc, i32, input, true);
  Operation *definedOnZero =
      LLVM::CountLeadingZerosOp::create(builder, loc, i32, input, false);
  Operation *alsoPoisonOnZero =
      LLVM::CountLeadingZerosOp::create(builder, loc, i32, input, true);
  Operation *poisonOnMin = LLVM::AbsOp::create(builder, loc, i32, input, true);
  Operation *definedOnMin =
      LLVM::AbsOp::create(builder, loc, i32, input, false);

  bool ok = true;
  auto project = [&ok](Operation *op, OperationSchemaId schema,
                       OperationSemanticsCase semantics)
      -> std::optional<CanonicalActorSchemaProjection> {
    std::optional<OperationSchemaId> resolved = operationSchemaOf(op);
    if (!resolved || *resolved != schema ||
        semanticsCase(*resolved) != semantics) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not resolve to its registered schema and case\n";
      ok = false;
      return std::nullopt;
    }
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        projectRegisteredActorSchemaProjection(op);
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };

  std::optional<CanonicalActorSchemaProjection> ctlzPoison =
      project(poisonOnZero, OperationSchemaId::LLVMCountLeadingZeros,
              OperationSemanticsCase::LLVMZeroPoison);
  std::optional<CanonicalActorSchemaProjection> ctlzPoisonAgain =
      project(alsoPoisonOnZero, OperationSchemaId::LLVMCountLeadingZeros,
              OperationSemanticsCase::LLVMZeroPoison);
  std::optional<CanonicalActorSchemaProjection> absPoison =
      project(poisonOnMin, OperationSchemaId::LLVMAbs,
              OperationSemanticsCase::LLVMIntegerMinPoison);

  if (ctlzPoison && ctlzPoisonAgain) {
    if (*ctlzPoison != *ctlzPoisonAgain) {
      llvm::errs() << "one zero-poison flag produced two projections\n";
      ok = false;
    }
  }
  if (!absPoison) {
    llvm::errs() << "the poison-flagged absolute value did not project\n";
    ok = false;
  }
  if (operationSchemaOf(definedOnZero) ||
      classifyCanonicalDataflowActor(definedOnZero) ||
      isCanonicalDataflowActor(definedOnZero)) {
    llvm::errs() << "the poison-free llvm.ctlz alias was admitted\n";
    ok = false;
  }
  if (operationSchemaOf(definedOnMin) ||
      classifyCanonicalDataflowActor(definedOnMin) ||
      isCanonicalDataflowActor(definedOnMin)) {
    llvm::errs() << "the poison-free llvm.abs alias was admitted\n";
    ok = false;
  }

  definedOnMin->erase();
  poisonOnMin->erase();
  alsoPoisonOnZero->erase();
  definedOnZero->erase();
  poisonOnZero->erase();
  input.getDefiningOp()->erase();
  return ok;
}

/// LLVM aggregate access has no exact standard-dialect spelling. Its static
/// aggregate position is firing semantics and must therefore survive in the
/// registered actor projection.
bool checkAggregatePositionProjection(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Type i32 = builder.getI32Type();
  Type aggregateType = LLVM::LLVMStructType::getLiteral(&context, {i32, i32});
  Value aggregate = LLVM::UndefOp::create(builder, loc, aggregateType);
  Value value =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(7));

  Operation *extractFirst =
      LLVM::ExtractValueOp::create(builder, loc, aggregate, {0});
  Operation *extractSecond =
      LLVM::ExtractValueOp::create(builder, loc, aggregate, {1});
  Operation *extractFirstAgain =
      LLVM::ExtractValueOp::create(builder, loc, aggregate, {0});
  Operation *insertFirst = LLVM::InsertValueOp::create(
      builder, loc, aggregate, value, llvm::ArrayRef<int64_t>{0});
  Operation *insertSecond = LLVM::InsertValueOp::create(
      builder, loc, aggregate, value, llvm::ArrayRef<int64_t>{1});

  bool ok = true;
  auto project =
      [&ok](Operation *op) -> std::optional<CanonicalActorSchemaProjection> {
    if (!operationSchemaOf(op)) {
      llvm::errs() << op->getName().getStringRef()
                   << " has no registered aggregate schema\n";
      ok = false;
      return std::nullopt;
    }
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        projectRegisteredActorSchemaProjection(op);
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };

  std::optional<CanonicalActorSchemaProjection> first = project(extractFirst);
  std::optional<CanonicalActorSchemaProjection> second = project(extractSecond);
  std::optional<CanonicalActorSchemaProjection> firstAgain =
      project(extractFirstAgain);
  std::optional<CanonicalActorSchemaProjection> insertedFirst =
      project(insertFirst);
  std::optional<CanonicalActorSchemaProjection> insertedSecond =
      project(insertSecond);
  if (first && second && firstAgain) {
    if (*first == *second) {
      llvm::errs() << "two extractvalue positions share one projection\n";
      ok = false;
    }
    if (*first != *firstAgain) {
      llvm::errs() << "one extractvalue position produced two projections\n";
      ok = false;
    }
  }
  if (insertedFirst && insertedSecond && *insertedFirst == *insertedSecond) {
    llvm::errs() << "two insertvalue positions share one projection\n";
    ok = false;
  }

  insertSecond->erase();
  insertFirst->erase();
  extractFirstAgain->erase();
  extractSecond->erase();
  extractFirst->erase();
  value.getDefiningOp()->erase();
  aggregate.getDefiningOp()->erase();
  return ok;
}

/// Any state outside a schema's closed typed payload must fail closed rather
/// than being captured in an open attribute bag or silently ignored.
bool checkUnclassifiedStateFailsClosed(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(2));
  Operation *bitAnd = arith::AndIOp::create(builder, loc, lhs, rhs);
  Operation *poison = ub::PoisonOp::create(builder, loc, builder.getI32Type());
  Value floatLhs =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(1.0));
  Value floatRhs =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(2.0));
  Operation *floating = arith::AddFOp::create(builder, loc, floatLhs, floatRhs,
                                              arith::FastMathFlags::none,
                                              arith::RoundingModeAttr{});

  bool ok = true;
  llvm::Expected<CanonicalActorSchemaProjection> defaultPoison =
      projectRegisteredActorSchemaProjection(poison);
  if (!defaultPoison) {
    llvm::errs() << llvm::toString(defaultPoison.takeError()) << '\n';
    ok = false;
  }

  StringAttr stateName =
      builder.getStringAttr("dataflow.unclassified_firing_state");
  bitAnd->setDiscardableAttr(stateName, builder.getUnitAttr());
  poison->setDiscardableAttr(stateName, builder.getUnitAttr());
  floating->setDiscardableAttr(stateName, builder.getUnitAttr());

  auto expectFailure = [&ok](Operation *op) {
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        projectRegisteredActorSchemaProjection(op);
    if (projection) {
      llvm::errs() << op->getName().getStringRef()
                   << " ignored unclassified firing state\n";
      ok = false;
      return;
    }
    llvm::consumeError(projection.takeError());
  };
  expectFailure(bitAnd);
  expectFailure(poison);
  expectFailure(floating);

  floating->erase();
  floatRhs.getDefiningOp()->erase();
  floatLhs.getDefiningOp()->erase();
  poison->erase();
  bitAnd->erase();
  rhs.getDefiningOp()->erase();
  lhs.getDefiningOp()->erase();
  return ok;
}

/// A declared semantic case owns exact typed state. An instance that does not
/// carry it must fail closed in both the projection and the instance verifier;
/// it must never degrade to an empty payload.
bool checkDeclaredCaseFailsClosed(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  OperationState state(loc, ConstantOp::getOperationName());
  state.addTypes(builder.getI32Type());
  state.addAttribute("const_value", builder.getUnitAttr());
  Operation *malformed = builder.create(state);

  bool ok = true;
  auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(malformed);
  if (!actor) {
    llvm::errs() << "dataflow.constant does not project through the typed "
                    "actor interface\n";
    ok = false;
  } else {
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        actor.projectCanonicalActorSchemaProjection();
    if (projection) {
      llvm::errs() << "an untyped constant value still produced a "
                      "projection\n";
      ok = false;
    } else {
      llvm::consumeError(projection.takeError());
    }
    context.getDiagEngine().registerHandler([](Diagnostic &) {});
    if (succeeded(actor.verifyCanonicalActorInstance())) {
      llvm::errs() << "the instance verifier accepted an untyped constant "
                      "value\n";
      ok = false;
    }
  }

  malformed->erase();
  return ok;
}

template <typename ContractProjection, typename AlignmentFn>
bool checkMemorySourceAlignmentPair(MLIRContext &context,
                                    llvm::StringRef operationName,
                                    Attribute fourByteContract,
                                    Attribute eightByteContract,
                                    AlignmentFn alignment) {
  OpBuilder builder(&context);
  auto create = [&](Attribute contract) {
    OperationState state(builder.getUnknownLoc(), operationName);
    state.addAttribute("contract", contract);
    return builder.create(state);
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
  auto checkAlignment = [&](const CanonicalActorSchemaProjection &projection,
                            std::uint64_t expected) {
    const auto *memory =
        std::get_if<MemoryContractPayload>(&projection.payload);
    if (!memory) {
      llvm::errs() << operationName
                   << " did not project a memory contract payload\n";
      ok = false;
      return;
    }
    const auto *contract = std::get_if<ContractProjection>(memory);
    if (!contract) {
      llvm::errs() << operationName
                   << " projected the wrong memory contract alternative\n";
      ok = false;
      return;
    }
    if (alignment(*contract) != expected) {
      llvm::errs() << operationName << " lost source_alignment_bytes "
                   << expected << '\n';
      ok = false;
    }
  };

  std::optional<CanonicalActorSchemaProjection> fourByte =
      project(fourByteActor);
  std::optional<CanonicalActorSchemaProjection> eightByte =
      project(eightByteActor);
  if (fourByte && eightByte) {
    if (*fourByte == *eightByte) {
      llvm::errs() << operationName
                   << " erased source_alignment_bytes from actor identity\n";
      ok = false;
    }
    checkAlignment(*fourByte, 4);
    checkAlignment(*eightByte, 8);
  }

  eightByteActor->erase();
  fourByteActor->erase();
  return ok;
}

bool checkMemorySourceAlignmentIdentity(MLIRContext &context) {
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
  ok &= checkMemorySourceAlignmentPair<AtomicAccessProjection>(
      context, LoadOp::getOperationName(), atomicFour, atomicEight,
      atomicAlignment);
  ok &= checkMemorySourceAlignmentPair<AtomicAccessProjection>(
      context, StoreOp::getOperationName(), atomicFour, atomicEight,
      atomicAlignment);

  AtomicRmwContractAttr rmwFour =
      AtomicRmwContractAttr::get(&context, AtomicRmwKind::Add, atomicFour);
  AtomicRmwContractAttr rmwEight =
      AtomicRmwContractAttr::get(&context, AtomicRmwKind::Add, atomicEight);
  ok &= checkMemorySourceAlignmentPair<AtomicRmwProjection>(
      context, AtomicRmwOp::getOperationName(), rmwFour, rmwEight,
      [](const AtomicRmwProjection &projection) {
        return projection.access.sourceAlignmentBytes;
      });

  auto compareExchange = [&](std::uint64_t alignment) {
    return CompareExchangeContractAttr::get(
        &context, AtomicOrdering::SeqCst, AtomicOrdering::Monotonic, scope,
        alignment, std::nullopt, /*weak=*/false, /*is_volatile=*/false);
  };
  ok &= checkMemorySourceAlignmentPair<CompareExchangeProjection>(
      context, CmpXchgOp::getOperationName(), compareExchange(4),
      compareExchange(8), [](const CompareExchangeProjection &projection) {
        return projection.sourceAlignmentBytes;
      });
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                  LLVM::LLVMDialect, ub::UBDialect, vector::VectorDialect,
                  DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkNativeSemanticInterfaceCoverage();
  ok &= checkSpellingRoundTrip();
  ok &= checkUnregisteredSpelling();
  ok &= checkOperationIdentity(context);
  ok &= checkTypedSemanticDelta(context);
  ok &= checkFloatingSemanticState(context);
  ok &= checkExactAndNonNegativeSemanticState(context);
  ok &= checkVectorStructuralSemanticState(context);
  ok &= checkStructuralVectorAdmission(context);
  ok &= checkPoisonFlagAdmission(context);
  ok &= checkAggregatePositionProjection(context);
  ok &= checkUnclassifiedStateFailsClosed(context);
  ok &= checkDeclaredCaseFailsClosed(context);
  ok &= checkMemorySourceAlignmentIdentity(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
