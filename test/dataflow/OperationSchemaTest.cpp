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
      [&ok](Operation *op) -> std::optional<CanonicalActorSemantics> {
    auto actor = llvm::dyn_cast<CanonicalDataflowActorOpInterface>(op);
    if (!actor) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not project through the typed actor interface\n";
      ok = false;
      return std::nullopt;
    }
    llvm::Expected<CanonicalActorSemantics> projection =
        actor.projectCanonicalActorSemantics();
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };

  std::optional<CanonicalActorSemantics> first = project(signedLess);
  std::optional<CanonicalActorSemantics> second = project(equal);
  std::optional<CanonicalActorSemantics> third = project(alsoSignedLess);
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
      -> std::optional<CanonicalActorSemantics> {
    std::optional<OperationSchemaId> resolved = operationSchemaOf(op);
    if (!resolved || *resolved != schema ||
        semanticsCase(*resolved) != semantics) {
      llvm::errs() << op->getName().getStringRef()
                   << " does not resolve to its registered schema and case\n";
      ok = false;
      return std::nullopt;
    }
    llvm::Expected<CanonicalActorSemantics> projection =
        projectRegisteredActorSemantics(op);
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };

  std::optional<CanonicalActorSemantics> ctlzPoison =
      project(poisonOnZero, OperationSchemaId::LLVMCountLeadingZeros,
              OperationSemanticsCase::LLVMZeroPoison);
  std::optional<CanonicalActorSemantics> ctlzPoisonAgain =
      project(alsoPoisonOnZero, OperationSchemaId::LLVMCountLeadingZeros,
              OperationSemanticsCase::LLVMZeroPoison);
  std::optional<CanonicalActorSemantics> absPoison =
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
      [&ok](Operation *op) -> std::optional<CanonicalActorSemantics> {
    if (!operationSchemaOf(op)) {
      llvm::errs() << op->getName().getStringRef()
                   << " has no registered aggregate schema\n";
      ok = false;
      return std::nullopt;
    }
    llvm::Expected<CanonicalActorSemantics> projection =
        projectRegisteredActorSemantics(op);
    if (!projection) {
      llvm::errs() << llvm::toString(projection.takeError()) << '\n';
      ok = false;
      return std::nullopt;
    }
    return *projection;
  };

  std::optional<CanonicalActorSemantics> first = project(extractFirst);
  std::optional<CanonicalActorSemantics> second = project(extractSecond);
  std::optional<CanonicalActorSemantics> firstAgain =
      project(extractFirstAgain);
  std::optional<CanonicalActorSemantics> insertedFirst = project(insertFirst);
  std::optional<CanonicalActorSemantics> insertedSecond = project(insertSecond);
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

/// A no-payload schema states that function type is the complete firing
/// semantics. Any otherwise unclassified operation state must fail closed
/// rather than being captured in an open attribute bag or silently ignored.
bool checkNoPayloadFailsClosed(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Value lhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(1));
  Value rhs =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(2));
  Operation *bitAnd = arith::AndIOp::create(builder, loc, lhs, rhs);
  Operation *exactDivision =
      arith::DivSIOp::create(builder, loc, lhs, rhs, true);
  Operation *poison = ub::PoisonOp::create(builder, loc, builder.getI32Type());

  bool ok = true;
  llvm::Expected<CanonicalActorSemantics> defaultPoison =
      projectRegisteredActorSemantics(poison);
  if (!defaultPoison) {
    llvm::errs() << llvm::toString(defaultPoison.takeError()) << '\n';
    ok = false;
  }

  StringAttr stateName =
      builder.getStringAttr("dataflow.unclassified_firing_state");
  bitAnd->setDiscardableAttr(stateName, builder.getUnitAttr());
  poison->setDiscardableAttr(stateName, builder.getUnitAttr());

  auto expectFailure = [&ok](Operation *op) {
    llvm::Expected<CanonicalActorSemantics> projection =
        projectRegisteredActorSemantics(op);
    if (projection) {
      llvm::errs() << op->getName().getStringRef()
                   << " ignored unclassified firing state\n";
      ok = false;
      return;
    }
    llvm::consumeError(projection.takeError());
  };
  expectFailure(bitAnd);
  expectFailure(exactDivision);
  expectFailure(poison);

  poison->erase();
  exactDivision->erase();
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
    llvm::Expected<CanonicalActorSemantics> projection =
        actor.projectCanonicalActorSemantics();
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

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                  LLVM::LLVMDialect, ub::UBDialect, DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkSpellingRoundTrip();
  ok &= checkUnregisteredSpelling();
  ok &= checkOperationIdentity(context);
  ok &= checkTypedSemanticDelta(context);
  ok &= checkPoisonFlagAdmission(context);
  ok &= checkAggregatePositionProjection(context);
  ok &= checkNoPayloadFailsClosed(context);
  ok &= checkDeclaredCaseFailsClosed(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
