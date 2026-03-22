// SCF post-processing: uplift scf.while loops to scf.for when possible.
// Handles iter-arg-based induction patterns produced by LLVM's lift-cf-to-scf.
// General-purpose: supports various induction ops (add/sub) and comparison
// predicates (ne/slt/sle/sgt/sge/ult/ule/ugt/uge).

#include "loom/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// Strip arith cast chains to find the underlying value.
Value stripCasts(Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<arith::ExtUIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::ExtSIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::TruncIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    break;
  }
  return value;
}

bool getConstantInt(Value value, int64_t &out) {
  if (!value)
    return false;
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue())) {
      out = intAttr.getInt();
      return true;
    }
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cst.value();
    return true;
  }
  return false;
}

bool isIndexLike(Type type) {
  return type && (type.isIndex() || llvm::isa<IntegerType>(type));
}

bool isSideEffectFree(Operation &op) {
  if (op.getNumRegions() != 0)
    return false;
  if (auto memEffect = llvm::dyn_cast<MemoryEffectOpInterface>(&op))
    return memEffect.hasNoEffect();
  return false;
}

bool isDefinedIn(Operation *root, Value value) {
  if (!value)
    return false;
  if (auto *def = value.getDefiningOp())
    return root->isAncestor(def);
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    return owner && root->isAncestor(owner);
  }
  return false;
}

// Check if a block's yield just passes through its block arguments
// (possibly via side-effect-free casts).
bool isPassThroughYield(Block &block) {
  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(block.getTerminator());
  if (!yieldOp)
    return false;
  if (yieldOp.getNumOperands() != block.getNumArguments())
    return false;
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (stripCasts(yieldOp.getOperand(i)) != block.getArgument(i))
      return false;
  }
  for (Operation &op : block) {
    if (llvm::isa<scf::YieldOp>(op))
      continue;
    if (!isSideEffectFree(op))
      return false;
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner() != yieldOp)
          return false;
      }
    }
  }
  return true;
}

Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (!value)
    return {};
  if (value.getType().isIndex())
    return value;
  if (llvm::isa<IntegerType>(value.getType()))
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      value);
  return {};
}

Value castIndexToType(OpBuilder &builder, Location loc, Value value,
                      Type targetType) {
  if (!value)
    return {};
  if (value.getType() == targetType)
    return value;
  if (value.getType().isIndex() && llvm::isa<IntegerType>(targetType))
    return arith::IndexCastOp::create(builder, loc, targetType, value);
  return {};
}

// Clone a loop-invariant value to before the loop.
Value cloneLoopInvariant(Value value, Operation *loop,
                         PatternRewriter &rewriter,
                         DenseMap<Value, Value> &cache) {
  if (!value)
    return {};
  if (!isDefinedIn(loop, value))
    return value;
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;
  if (llvm::isa<BlockArgument>(value))
    return {};

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  if (def->getNumRegions() != 0)
    return {};
  if (auto memEffect = llvm::dyn_cast<MemoryEffectOpInterface>(def)) {
    if (!memEffect.hasNoEffect())
      return {};
  }

  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    Value hoisted = cloneLoopInvariant(operand, loop, rewriter, cache);
    if (!hoisted)
      return {};
    mapping.map(operand, hoisted);
  }

  Operation *clone = rewriter.clone(*def, mapping);
  for (auto [orig, repl] :
       llvm::zip(def->getResults(), clone->getResults()))
    cache[orig] = repl;

  return cache.lookup(value);
}

// ---------------------------------------------------------------------------
// Induction variable analysis
// ---------------------------------------------------------------------------

struct StepInfo {
  int64_t constant = 0;
  Value value;
  bool isConst = false;
};

// Match: value = inductionArg +/- step
bool matchInductionUpdate(Value value, BlockArgument inductionArg,
                          StepInfo &step) {
  if (!value)
    return false;
  value = stripCasts(value);
  if (auto addi = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = stripCasts(addi.getLhs());
    Value rhs = stripCasts(addi.getRhs());
    Value other;
    if (lhs == inductionArg)
      other = rhs;
    else if (rhs == inductionArg)
      other = lhs;
    else
      return false;
    int64_t c = 0;
    if (getConstantInt(other, c)) {
      step.constant = c;
      step.isConst = true;
    } else {
      step.value = other;
      step.isConst = false;
    }
    return true;
  }
  if (auto subi = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = stripCasts(subi.getLhs());
    Value rhs = stripCasts(subi.getRhs());
    if (lhs == inductionArg) {
      int64_t c = 0;
      if (getConstantInt(rhs, c)) {
        step.constant = -c;
        step.isConst = true;
        return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Comparison predicate helpers
// ---------------------------------------------------------------------------

enum class CmpKind {
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  NotEqual,
};

bool getCmpKind(arith::CmpIPredicate pred, CmpKind &kind) {
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    kind = CmpKind::Less;
    return true;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    kind = CmpKind::LessEqual;
    return true;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    kind = CmpKind::Greater;
    return true;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    kind = CmpKind::GreaterEqual;
    return true;
  case arith::CmpIPredicate::ne:
    kind = CmpKind::NotEqual;
    return true;
  default:
    return false;
  }
}

bool isUnsignedPredicate(arith::CmpIPredicate pred) {
  return pred == arith::CmpIPredicate::ult ||
         pred == arith::CmpIPredicate::ule ||
         pred == arith::CmpIPredicate::ugt ||
         pred == arith::CmpIPredicate::uge;
}

arith::CmpIPredicate swapPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return pred;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  return pred;
}

// ---------------------------------------------------------------------------
// Core conversion: iter-arg scf.while -> scf.for
// ---------------------------------------------------------------------------
//
// Handles two layouts:
//   Layout A ("body in after"): before has condition logic only,
//             after has body + yield with induction update.
//   Layout B ("body in before"): before has body + condition,
//             after is pass-through yield.
//
// The vecadd pattern is Layout B with compare-on-update:
//   before: body ops, %next = addi %iv, step, cmpi ne %next, bound,
//           condition(%cmp) %next
//   after:  yield %arg (pass-through)

LogicalResult tryUpliftIterArgWhile(scf::WhileOp loop,
                                    PatternRewriter &rewriter) {
  if (loop.getInits().empty() || loop.getResults().empty())
    return failure();
  if (loop.getInits().size() != loop.getResults().size())
    return failure();
  if (!loop.getBefore().hasOneBlock() || !loop.getAfter().hasOneBlock())
    return failure();

  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();
  auto conditionOp = llvm::dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!conditionOp)
    return failure();
  if (conditionOp.getArgs().size() != before.getNumArguments())
    return failure();

  Value condValue = stripCasts(conditionOp.getCondition());
  auto cmpOp = condValue.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return failure();

  auto pred = cmpOp.getPredicate();
  CmpKind cmpKind;
  if (!getCmpKind(pred, cmpKind))
    return failure();

  // Determine layout.
  bool afterIsPassThrough = isPassThroughYield(after);

  bool condArgsPassThrough = true;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (stripCasts(conditionOp.getArgs()[i]) != before.getArgument(i)) {
      condArgsPassThrough = false;
      break;
    }
  }

  bool beforeSideEffectFree = true;
  for (Operation &op : before) {
    if (llvm::isa<scf::ConditionOp>(op))
      continue;
    if (!isSideEffectFree(op)) {
      beforeSideEffectFree = false;
      break;
    }
  }

  bool bodyInBefore;
  if (condArgsPassThrough && beforeSideEffectFree)
    bodyInBefore = false;
  else if (afterIsPassThrough)
    bodyInBefore = true;
  else
    return failure();

  // Find induction variable among block args.
  Value lhs = stripCasts(cmpOp.getLhs());
  Value rhs = stripCasts(cmpOp.getRhs());

  auto matchOperand = [&](Value operand, int &idx, bool &usesUpdate,
                          StepInfo &step) -> bool {
    for (unsigned i = 0; i < before.getNumArguments(); ++i) {
      if (operand == before.getArgument(i)) {
        idx = static_cast<int>(i);
        usesUpdate = false;
        return true;
      }
      StepInfo candidateStep;
      if (matchInductionUpdate(operand, before.getArgument(i), candidateStep)) {
        idx = static_cast<int>(i);
        usesUpdate = true;
        step = candidateStep;
        return true;
      }
    }
    return false;
  };

  int lhsIndex = -1, rhsIndex = -1;
  bool lhsUsesUpdate = false, rhsUsesUpdate = false;
  StepInfo lhsStep, rhsStep;
  bool lhsMatch = matchOperand(lhs, lhsIndex, lhsUsesUpdate, lhsStep);
  bool rhsMatch = matchOperand(rhs, rhsIndex, rhsUsesUpdate, rhsStep);

  // Exactly one side should be the induction variable.
  if (lhsMatch && rhsMatch)
    return failure();

  int inductionIndex = -1;
  bool compareUsesUpdate = false;
  StepInfo cmpStepInfo;

  if (rhsMatch) {
    std::swap(lhs, rhs);
    pred = swapPredicate(pred);
    getCmpKind(pred, cmpKind);
    inductionIndex = rhsIndex;
    compareUsesUpdate = rhsUsesUpdate;
    cmpStepInfo = rhsStep;
  } else if (lhsMatch) {
    inductionIndex = lhsIndex;
    compareUsesUpdate = lhsUsesUpdate;
    cmpStepInfo = lhsStep;
  } else {
    return failure();
  }

  if (!isIndexLike(before.getArgument(inductionIndex).getType()))
    return failure();
  if (!loop.getResult(inductionIndex).use_empty())
    return failure();

  // Find the induction step from yield/condition args.
  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != loop.getInits().size())
    return failure();

  StepInfo stepInfo;
  if (bodyInBefore) {
    Value updateVal = stripCasts(conditionOp.getArgs()[inductionIndex]);
    if (!matchInductionUpdate(updateVal,
                              before.getArgument(inductionIndex), stepInfo))
      return failure();
  } else {
    if (!matchInductionUpdate(yieldOp.getOperand(inductionIndex),
                              after.getArgument(inductionIndex), stepInfo))
      return failure();
  }

  if (stepInfo.isConst && stepInfo.constant == 0)
    return failure();

  // When compare uses the updated value, verify step consistency.
  if (compareUsesUpdate) {
    if (cmpStepInfo.isConst != stepInfo.isConst)
      return failure();
    if (cmpStepInfo.isConst && cmpStepInfo.constant != stepInfo.constant)
      return failure();
    if (!cmpStepInfo.isConst &&
        stripCasts(cmpStepInfo.value) != stripCasts(stepInfo.value))
      return failure();
  }

  if (!stepInfo.isConst && isDefinedIn(loop, stepInfo.value))
    return failure();
  if (!stepInfo.isConst && !isIndexLike(stepInfo.value.getType()))
    return failure();

  // Determine loop direction.
  bool directionAscending = false;
  switch (cmpKind) {
  case CmpKind::Less:
  case CmpKind::LessEqual:
    directionAscending = true;
    break;
  case CmpKind::Greater:
  case CmpKind::GreaterEqual:
    directionAscending = false;
    break;
  case CmpKind::NotEqual:
    if (!stepInfo.isConst)
      return failure();
    directionAscending = stepInfo.constant > 0;
    break;
  }

  if (directionAscending) {
    if (stepInfo.isConst && stepInfo.constant <= 0)
      return failure();
  } else {
    if (!stepInfo.isConst || stepInfo.constant >= 0)
      return failure();
  }

  // Build the scf.for operation.
  Location loc = loop.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  Value initValue = loop.getInits()[inductionIndex];
  Value boundValue = rhs;

  // Hoist bound if defined inside the loop.
  Value boundOutside = boundValue;
  if (isDefinedIn(loop, boundValue)) {
    DenseMap<Value, Value> hoisted;
    boundOutside = cloneLoopInvariant(boundValue, loop, rewriter, hoisted);
    if (!boundOutside)
      return failure();
  }

  // Hoist non-constant step if needed.
  Value stepOutside;
  if (!stepInfo.isConst) {
    stepOutside = stepInfo.value;
    if (!stepOutside)
      return failure();
    if (isDefinedIn(loop, stepOutside)) {
      DenseMap<Value, Value> hoisted;
      stepOutside = cloneLoopInvariant(stepOutside, loop, rewriter, hoisted);
      if (!stepOutside)
        return failure();
    }
  }

  // Compute lower/upper/step in index type.
  Value initIndex = castToIndex(rewriter, loc, initValue);
  if (!initIndex)
    return failure();
  Value boundIndex = castToIndex(rewriter, loc, boundOutside);
  if (!boundIndex)
    return failure();

  Value lowerIndex, upperIndex;
  if (directionAscending) {
    lowerIndex = initIndex;
    upperIndex = boundIndex;
    if (cmpKind == CmpKind::LessEqual) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      upperIndex = arith::AddIOp::create(rewriter, loc, upperIndex, one);
    }
  } else {
    // Descending: map to ascending [0, init - bound) with positive step.
    Value diff =
        arith::SubIOp::create(rewriter, loc, initIndex, boundIndex);
    upperIndex = diff;
    if (cmpKind == CmpKind::GreaterEqual) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      upperIndex = arith::AddIOp::create(rewriter, loc, upperIndex, one);
    }
    lowerIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  }

  Value stepIndex;
  if (stepInfo.isConst) {
    int64_t stepAbs =
        stepInfo.constant > 0 ? stepInfo.constant : -stepInfo.constant;
    stepIndex = arith::ConstantIndexOp::create(rewriter, loc, stepAbs);
  } else {
    stepIndex = castToIndex(rewriter, loc, stepOutside);
    if (!stepIndex)
      return failure();
  }

  // Iter args for scf.for: all while iter args except the induction variable.
  SmallVector<Value, 4> initArgs;
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    initArgs.push_back(loop.getInits()[i]);
  }

  auto forOp = scf::ForOp::create(rewriter, loc, lowerIndex, upperIndex,
                                   stepIndex, initArgs);
  scf::ForOp::ensureTerminator(forOp.getRegion(), rewriter, loc);
  if (isUnsignedPredicate(pred))
    forOp.setUnsignedCmp(true);

  // Helper: get the actual IV value inside the for body (handles descending).
  Value mappedIvCache;
  auto getMappedIv = [&](OpBuilder &b) -> Value {
    if (directionAscending)
      return forOp.getInductionVar();
    if (!mappedIvCache)
      mappedIvCache = arith::SubIOp::create(b, loc, initIndex,
                                            forOp.getInductionVar());
    return mappedIvCache;
  };

  Block *forBody = forOp.getBody();

  if (bodyInBefore) {
    // Layout B: body ops are in the before-region.
    // Clone everything except the induction-related ops.
    Operation *terminator = forBody->getTerminator();
    if (!terminator)
      return failure();
    OpBuilder bodyBuilder(rewriter.getContext());
    bodyBuilder.setInsertionPoint(terminator);

    // Identify which ops are induction-related (should not be cloned).
    llvm::SmallPtrSet<Operation *, 8> inductionOps;
    inductionOps.insert(conditionOp.getOperation());
    inductionOps.insert(cmpOp.getOperation());
    // The update expression (e.g., addi %iv, %step).
    Value rawUpdate = stripCasts(conditionOp.getArgs()[inductionIndex]);
    if (auto *updateDef = rawUpdate.getDefiningOp()) {
      if (updateDef->getBlock() == &before)
        inductionOps.insert(updateDef);
    }
    // Casts between cmpOp result and conditionOp condition.
    Value castChain = conditionOp.getCondition();
    while (castChain != cmpOp.getResult()) {
      if (auto *def = castChain.getDefiningOp()) {
        if (def->getBlock() == &before)
          inductionOps.insert(def);
      }
      if (auto cast = castChain.getDefiningOp<arith::ExtUIOp>())
        castChain = cast.getIn();
      else if (auto cast = castChain.getDefiningOp<arith::ExtSIOp>())
        castChain = cast.getIn();
      else if (auto cast = castChain.getDefiningOp<arith::TruncIOp>())
        castChain = cast.getIn();
      else if (auto cast = castChain.getDefiningOp<arith::IndexCastOp>())
        castChain = cast.getIn();
      else
        break;
    }

    // Build mapping for block args.
    IRMapping mapping;
    auto newIterArgs = forOp.getRegionIterArgs();
    unsigned iterIdx = 0;
    for (unsigned i = 0; i < before.getNumArguments(); ++i) {
      BlockArgument arg = before.getArgument(i);
      if (static_cast<int>(i) == inductionIndex) {
        Value iv = getMappedIv(bodyBuilder);
        Value casted = castIndexToType(bodyBuilder, loc, iv, arg.getType());
        if (!casted)
          return failure();
        mapping.map(arg, casted);
      } else {
        mapping.map(arg, newIterArgs[iterIdx++]);
      }
    }

    // Clone body ops.
    for (Operation &op : before) {
      if (inductionOps.contains(&op))
        continue;
      bodyBuilder.clone(op, mapping);
    }

    // Build yield operands for the remaining iter args.
    SmallVector<Value, 4> newYieldOperands;
    for (unsigned i = 0; i < conditionOp.getArgs().size(); ++i) {
      if (static_cast<int>(i) == inductionIndex)
        continue;
      Value operand = conditionOp.getArgs()[i];
      Value mapped = mapping.lookupOrDefault(operand);
      if (mapped == operand && isDefinedIn(loop, operand))
        return failure();
      newYieldOperands.push_back(mapped);
    }
    auto oldYield = llvm::dyn_cast<scf::YieldOp>(terminator);
    if (!oldYield)
      return failure();
    rewriter.modifyOpInPlace(
        oldYield, [&]() { oldYield->setOperands(newYieldOperands); });

  } else {
    // Layout A: body ops are in the after-region.
    auto oldYield = llvm::dyn_cast<scf::YieldOp>(forBody->getTerminator());
    if (!oldYield)
      return failure();
    rewriter.setInsertionPoint(oldYield);

    DenseMap<Type, Value> ivCasts;
    auto getIvAsType = [&](Type type) -> Value {
      auto it = ivCasts.find(type);
      if (it != ivCasts.end())
        return it->second;
      Value iv = getMappedIv(rewriter);
      Value casted = castIndexToType(rewriter, loc, iv, type);
      ivCasts[type] = casted;
      return casted;
    };

    IRMapping mapping;
    auto newIterArgs = forOp.getRegionIterArgs();
    unsigned iterIdx = 0;
    for (unsigned i = 0; i < after.getNumArguments(); ++i) {
      BlockArgument arg = after.getArgument(i);
      if (static_cast<int>(i) == inductionIndex) {
        Value casted = getIvAsType(arg.getType());
        if (!casted)
          return failure();
        mapping.map(arg, casted);
      } else {
        mapping.map(arg, newIterArgs[iterIdx++]);
      }
    }

    for (Operation &op : after) {
      if (llvm::isa<scf::YieldOp>(op))
        continue;
      rewriter.clone(op, mapping);
    }

    SmallVector<Value, 4> newYieldOperands;
    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      if (static_cast<int>(i) == inductionIndex)
        continue;
      Value mapped = mapping.lookupOrDefault(yieldOp.getOperand(i));
      newYieldOperands.push_back(mapped);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, newYieldOperands);
  }

  // Verify the generated for op before committing.
  if (failed(verify(forOp))) {
    rewriter.eraseOp(forOp);
    return failure();
  }

  // Replace results (the induction variable result is unused).
  unsigned resultIdx = 0;
  for (unsigned i = 0; i < loop.getResults().size(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    loop.getResult(i).replaceAllUsesWith(forOp.getResult(resultIdx++));
  }

  rewriter.eraseOp(loop);
  return success();
}

// ---------------------------------------------------------------------------
// Pattern and pass definitions
// ---------------------------------------------------------------------------

struct WhileToForPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    return tryUpliftIterArgWhile(loop, rewriter);
  }
};

struct UpliftWhileToForPass
    : public PassWrapper<UpliftWhileToForPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UpliftWhileToForPass)

  StringRef getArgument() const final { return "loom-uplift-while-to-for"; }
  StringRef getDescription() const final {
    return "Convert scf.while with induction pattern to scf.for";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<WhileToForPattern>(context);
    scf::populateUpliftWhileToForPatterns(patterns);
    FrozenRewritePatternSet frozen(std::move(patterns));

    for (auto func : getOperation().getOps<func::FuncOp>())
      (void)applyPatternsGreedily(func, frozen);
  }
};

struct EliminateSubviewBumpsPass
    : public PassWrapper<EliminateSubviewBumpsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateSubviewBumpsPass)

  StringRef getArgument() const override {
    return "loom-eliminate-subview-bumps";
  }
  StringRef getDescription() const override {
    return "Eliminate redundant subview offset bumps";
  }
  void runOnOperation() override {
    // TODO: implementation
  }
};

} // namespace

std::unique_ptr<Pass> loom::createUpliftWhileToForPass() {
  return std::make_unique<UpliftWhileToForPass>();
}

std::unique_ptr<Pass> loom::createEliminateSubviewBumpsPass() {
  return std::make_unique<EliminateSubviewBumpsPass>();
}
