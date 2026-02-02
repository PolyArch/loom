#include "loom/Conversion/SCFPostProcess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

constexpr StringRef kLoopMarkerPrefix = "__loom_loop_";

bool IsZeroIndex(Value value) {
  if (!value)
    return false;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (cst.getType().isIndex()) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        return intAttr.getInt() == 0;
    }
  }
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return IsZeroIndex(cast.getIn());
  return false;
}

bool GetConstantInt(Value value, int64_t &out) {
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

Value StripCasts(Value value) {
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

memref::LoadOp GetZeroIndexLoad(Value value) {
  auto load = StripCasts(value).getDefiningOp<memref::LoadOp>();
  if (!load)
    return {};
  if (load.getIndices().size() != 1)
    return {};
  if (!IsZeroIndex(load.getIndices().front()))
    return {};
  return load;
}

bool IsTrivialElseBlock(Block &block) {
  for (Operation &op : block) {
    if (llvm::isa<scf::YieldOp>(op))
      return true;
    if (op.hasTrait<OpTrait::ConstantLike>())
      continue;
    return false;
  }
  return false;
}

bool MatchInductionUpdate(memref::StoreOp store, int64_t &step,
                          Value &inductionLoadValue) {
  Value stored = store.getValue();
  if (auto addi = stored.getDefiningOp<arith::AddIOp>()) {
    int64_t constant = 0;
    if (GetConstantInt(addi.getRhs(), constant)) {
      step = constant;
      inductionLoadValue = addi.getLhs();
      return true;
    }
    if (GetConstantInt(addi.getLhs(), constant)) {
      step = constant;
      inductionLoadValue = addi.getRhs();
      return true;
    }
  }
  if (auto subi = stored.getDefiningOp<arith::SubIOp>()) {
    int64_t constant = 0;
    if (GetConstantInt(subi.getRhs(), constant)) {
      step = -constant;
      inductionLoadValue = subi.getLhs();
      return true;
    }
  }
  return false;
}

bool MatchInductionUpdateValue(Value value, BlockArgument inductionArg,
                               int64_t &step) {
  if (!value)
    return false;
  value = StripCasts(value);
  if (auto addi = value.getDefiningOp<arith::AddIOp>()) {
    int64_t constant = 0;
    Value lhs = StripCasts(addi.getLhs());
    Value rhs = StripCasts(addi.getRhs());
    if (lhs == inductionArg && GetConstantInt(rhs, constant)) {
      step = constant;
      return true;
    }
    if (rhs == inductionArg && GetConstantInt(lhs, constant)) {
      step = constant;
      return true;
    }
  }
  if (auto subi = value.getDefiningOp<arith::SubIOp>()) {
    int64_t constant = 0;
    Value lhs = StripCasts(subi.getLhs());
    Value rhs = StripCasts(subi.getRhs());
    if (lhs == inductionArg && GetConstantInt(rhs, constant)) {
      step = -constant;
      return true;
    }
  }
  return false;
}

bool IsDefinedIn(Operation *root, Value value) {
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

bool HasStoreToMemref(Operation *root, Value memref) {
  bool found = false;
  root->walk([&](memref::StoreOp store) {
    if (store.getMemref() == memref)
      found = true;
  });
  return found;
}

Value CloneLoopInvariantValue(Value value, Operation *loop,
                              PatternRewriter &rewriter,
                              DenseMap<Value, Value> &cache) {
  if (!value)
    return {};
  if (!IsDefinedIn(loop, value))
    return value;
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;
  if (llvm::isa<BlockArgument>(value))
    return {};

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  Location loc = def->getLoc();
  if (auto load = llvm::dyn_cast<memref::LoadOp>(def)) {
    if (load.getIndices().size() != 1 ||
        !IsZeroIndex(load.getIndices().front()))
      return {};
    Value memref = load.getMemref();
    if (HasStoreToMemref(loop, memref))
      return {};
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value hoisted = rewriter.create<memref::LoadOp>(loc, memref, zeroIndex);
    cache[value] = hoisted;
    return hoisted;
  }

  if (def->getNumRegions() != 0)
    return {};

  if (auto memEffect = llvm::dyn_cast<MemoryEffectOpInterface>(def)) {
    if (!memEffect.hasNoEffect())
      return {};
  }

  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    Value hoisted = CloneLoopInvariantValue(operand, loop, rewriter, cache);
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

Value CastToIndex(OpBuilder &builder, Location loc, Value value) {
  if (!value)
    return {};
  if (value.getType().isIndex())
    return value;
  if (llvm::isa<IntegerType>(value.getType()))
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              value);
  return {};
}

Value CastIndexToType(OpBuilder &builder, Location loc, Value value,
                      Type targetType) {
  if (!value)
    return {};
  if (value.getType() == targetType)
    return value;
  if (value.getType().isIndex() && llvm::isa<IntegerType>(targetType))
    return builder.create<arith::IndexCastOp>(loc, targetType, value);
  return {};
}

LogicalResult TryUpliftIterArgWhile(scf::WhileOp loop,
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

  Value condValue = StripCasts(conditionOp.getCondition());
  auto cmpOp = condValue.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return failure();

  auto pred = cmpOp.getPredicate();
  bool predIsLess = pred == arith::CmpIPredicate::slt ||
                    pred == arith::CmpIPredicate::ult ||
                    pred == arith::CmpIPredicate::sle ||
                    pred == arith::CmpIPredicate::ule;
  if (!predIsLess)
    return failure();

  Value lhs = StripCasts(cmpOp.getLhs());
  int inductionIndex = -1;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (lhs == before.getArgument(i)) {
      inductionIndex = static_cast<int>(i);
      break;
    }
  }
  if (inductionIndex < 0)
    return failure();

  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != loop.getInits().size())
    return failure();

  int64_t step = 0;
  if (!MatchInductionUpdateValue(yieldOp.getOperand(inductionIndex),
                                 after.getArgument(inductionIndex), step))
    return failure();
  if (step <= 0)
    return failure();

  if (!loop.getResult(inductionIndex).use_empty())
    return failure();

  Location loc = loop.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  Value upperValue = cmpOp.getRhs();
  Value upperOutside = upperValue;
  if (IsDefinedIn(loop, upperValue)) {
    DenseMap<Value, Value> hoisted;
    upperOutside =
        CloneLoopInvariantValue(upperValue, loop, rewriter, hoisted);
    if (!upperOutside)
      return failure();
  }

  Value lowerIndex = CastToIndex(rewriter, loc, loop.getInits()[inductionIndex]);
  Value upperIndex = CastToIndex(rewriter, loc, upperOutside);
  if (!lowerIndex || !upperIndex)
    return failure();

  if (pred == arith::CmpIPredicate::sle ||
      pred == arith::CmpIPredicate::ule) {
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    upperIndex = rewriter.create<arith::AddIOp>(loc, upperIndex, one);
  }

  Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);
  SmallVector<Value, 4> initArgs;
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    initArgs.push_back(loop.getInits()[i]);
  }

  auto forOp =
      rewriter.create<scf::ForOp>(loc, lowerIndex, upperIndex, stepIndex,
                                  initArgs);
  if (pred == arith::CmpIPredicate::ult ||
      pred == arith::CmpIPredicate::ule)
    forOp.setUnsignedCmp(true);

  Block *forBody = forOp.getBody();
  scf::YieldOp oldYield;
  if (!forBody->empty()) {
    oldYield = llvm::dyn_cast<scf::YieldOp>(forBody->getTerminator());
    if (!oldYield)
      return failure();
    rewriter.setInsertionPoint(oldYield);
  } else {
    rewriter.setInsertionPointToStart(forBody);
  }

  DenseMap<Type, Value> ivCasts;
  auto getIvAsType = [&](Type type) -> Value {
    auto it = ivCasts.find(type);
    if (it != ivCasts.end())
      return it->second;
    Value casted =
        CastIndexToType(rewriter, loc, forOp.getInductionVar(), type);
    ivCasts[type] = casted;
    return casted;
  };

  IRMapping mapping;
  auto newIterArgs = forOp.getRegionIterArgs();
  unsigned iterIndex = 0;
  for (unsigned i = 0; i < after.getNumArguments(); ++i) {
    BlockArgument arg = after.getArgument(i);
    if (static_cast<int>(i) == inductionIndex) {
      Value casted = getIvAsType(arg.getType());
      if (!casted)
        return failure();
      mapping.map(arg, casted);
    } else {
      mapping.map(arg, newIterArgs[iterIndex++]);
    }
  }

  for (Operation &op : after) {
    if (llvm::isa<scf::YieldOp>(op))
      continue;
    rewriter.clone(op, mapping);
  }

  SmallVector<Value, 4> newYieldOperands;
  newYieldOperands.reserve(initArgs.size());
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    Value mapped = mapping.lookupOrDefault(yieldOp.getOperand(i));
    newYieldOperands.push_back(mapped);
  }
  if (oldYield)
    rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, newYieldOperands);
  else
    rewriter.create<scf::YieldOp>(loc, newYieldOperands);

  unsigned resultIndex = 0;
  for (unsigned i = 0; i < loop.getResults().size(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    loop.getResult(i).replaceAllUsesWith(forOp.getResult(resultIndex++));
  }

  rewriter.eraseOp(loop);
  return success();
}

void AppendLoomAnnotation(Operation *op, StringRef value, Builder &builder) {
  if (!op || value.empty())
    return;
  auto attr = op->getAttrOfType<ArrayAttr>("loom.annotations");
  SmallVector<Attribute, 4> values;
  if (attr)
    values.append(attr.begin(), attr.end());
  values.push_back(builder.getStringAttr(value));
  op->setAttr("loom.annotations", builder.getArrayAttr(values));
}

bool IsLoopMarkerCallee(StringRef callee) {
  return callee.starts_with(kLoopMarkerPrefix);
}

bool CollectLoopMarkerAnnotations(Operation *op,
                                  SmallVectorImpl<StringAttr> &out) {
  StringRef callee;
  if (auto call = llvm::dyn_cast<func::CallOp>(op)) {
    callee = call.getCallee();
  } else if (auto call = llvm::dyn_cast<LLVM::CallOp>(op)) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return false;
    callee = calleeAttr.getValue();
  } else {
    return false;
  }

  if (!IsLoopMarkerCallee(callee))
    return false;

  if (auto ann = op->getAttrOfType<ArrayAttr>("loom.annotations")) {
    for (Attribute attr : ann) {
      if (auto strAttr = llvm::dyn_cast<StringAttr>(attr))
        out.push_back(strAttr);
    }
  }
  return true;
}

bool IsLoopLike(Operation *op) {
  return llvm::isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp>(
      op);
}

void ProcessRegion(Region &region) {
  for (Block &block : region) {
    SmallVector<StringAttr, 4> pending;
    for (auto it = block.begin(); it != block.end();) {
      Operation *op = &*it++;
      SmallVector<StringAttr, 4> collected;
      if (CollectLoopMarkerAnnotations(op, collected)) {
        pending.append(collected.begin(), collected.end());
        op->erase();
        continue;
      }

      if (IsLoopLike(op) && !pending.empty()) {
        Builder builder(op->getContext());
        for (StringAttr attr : pending)
          AppendLoomAnnotation(op, attr.getValue(), builder);
        pending.clear();
      }

      for (Region &nested : op->getRegions())
        ProcessRegion(nested);
    }
  }
}

struct WhileMemrefToForPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    if (!loop.getInits().empty() || !loop.getResults().empty())
      return TryUpliftIterArgWhile(loop, rewriter);

    if (!loop.getBefore().hasOneBlock() || !loop.getAfter().hasOneBlock())
      return failure();

    Block &after = loop.getAfter().front();
    if (after.empty())
      return failure();
    if (!llvm::isa<scf::YieldOp>(after.back()))
      return failure();

    Block &before = loop.getBefore().front();
    auto *terminator = before.getTerminator();
    auto conditionOp = llvm::dyn_cast<scf::ConditionOp>(terminator);
    if (!conditionOp || !conditionOp.getArgs().empty())
      return failure();

    Value condValue = StripCasts(conditionOp.getCondition());
    scf::IfOp ifOp = condValue.getDefiningOp<scf::IfOp>();
    Block *bodyBlock = nullptr;
    arith::CmpIOp cmpOp;
    if (ifOp) {
      if (ifOp->getBlock() != &before)
        return failure();
      if (!ifOp.getThenRegion().hasOneBlock() ||
          !ifOp.getElseRegion().hasOneBlock())
        return failure();

      unsigned condResultIndex = 0;
      if (auto result = llvm::dyn_cast<OpResult>(condValue))
        condResultIndex = result.getResultNumber();
      else
        return failure();

      Block &thenBlock = ifOp.getThenRegion().front();
      Block &elseBlock = ifOp.getElseRegion().front();
      auto thenYield = llvm::dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
      auto elseYield = llvm::dyn_cast<scf::YieldOp>(elseBlock.getTerminator());
      if (!thenYield || !elseYield)
        return failure();

      int64_t thenCondValue = 0;
      int64_t elseCondValue = 0;
      if (condResultIndex >= thenYield.getNumOperands() ||
          condResultIndex >= elseYield.getNumOperands())
        return failure();
      if (!GetConstantInt(thenYield.getOperand(condResultIndex),
                          thenCondValue))
        return failure();
      if (!GetConstantInt(elseYield.getOperand(condResultIndex),
                          elseCondValue))
        return failure();
      if (!(thenCondValue == 1 && elseCondValue == 0))
        return failure();

      if (!IsTrivialElseBlock(elseBlock))
        return failure();

      if (after.begin() != std::prev(after.end()))
        return failure();

      cmpOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        return failure();
      bodyBlock = &thenBlock;
    } else {
      cmpOp = condValue.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        return failure();
      auto bodyYield = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
      if (!bodyYield || bodyYield.getNumOperands() != 0)
        return failure();
      bodyBlock = &after;
    }

    auto pred = cmpOp.getPredicate();
    bool predIsLess = pred == arith::CmpIPredicate::slt ||
                      pred == arith::CmpIPredicate::ult ||
                      pred == arith::CmpIPredicate::sle ||
                      pred == arith::CmpIPredicate::ule;
    if (!predIsLess)
      return failure();

    memref::StoreOp updateStore;
    int64_t step = 0;
    Value inductionMemref;
    for (auto store : bodyBlock->getOps<memref::StoreOp>()) {
      if (store.getIndices().size() != 1 ||
          !IsZeroIndex(store.getIndices().front()))
        continue;
      int64_t candidateStep = 0;
      Value candidateInduction;
      if (!MatchInductionUpdate(store, candidateStep, candidateInduction))
        continue;
      auto load = GetZeroIndexLoad(candidateInduction);
      if (!load || load.getMemref() != store.getMemref())
        continue;
      if (updateStore)
        return failure();
      updateStore = store;
      step = candidateStep;
      inductionMemref = store.getMemref();
    }
    if (!updateStore || !inductionMemref)
      return failure();
    if (step <= 0)
      return failure();

    for (auto store : bodyBlock->getOps<memref::StoreOp>()) {
      if (store.getMemref() != inductionMemref)
        continue;
      if (store.getIndices().size() != 1 ||
          !IsZeroIndex(store.getIndices().front()))
        return failure();
      if (store != updateStore)
        return failure();
    }

    auto lhsLoad = GetZeroIndexLoad(cmpOp.getLhs());
    if (!lhsLoad || lhsLoad.getMemref() != inductionMemref)
      return failure();

    Location loc = loop.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);
    Value upperValue = cmpOp.getRhs();
    Value upperOutside = upperValue;
    if (IsDefinedIn(loop, upperValue)) {
      DenseMap<Value, Value> hoisted;
      upperOutside =
          CloneLoopInvariantValue(upperValue, loop, rewriter, hoisted);
      if (!upperOutside)
        return failure();
    }

    memref::StoreOp initStore;
    Block *parent = loop->getBlock();
    for (auto it = loop->getIterator(); it != parent->begin();) {
      --it;
      if (auto store = llvm::dyn_cast<memref::StoreOp>(&*it)) {
        if (store.getMemref() == inductionMemref &&
            store.getIndices().size() == 1 &&
            IsZeroIndex(store.getIndices().front())) {
          initStore = store;
          break;
        }
      }
    }
    if (!initStore)
      return failure();

    Value lowerIndex = CastToIndex(rewriter, loc, initStore.getValue());
    Value upperIndex = CastToIndex(rewriter, loc, upperOutside);
    if (!lowerIndex || !upperIndex)
      return failure();

    if (pred == arith::CmpIPredicate::sle ||
        pred == arith::CmpIPredicate::ule) {
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      upperIndex = rewriter.create<arith::AddIOp>(loc, upperIndex, one);
    }

    Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);
    auto forOp =
        rewriter.create<scf::ForOp>(loc, lowerIndex, upperIndex, stepIndex);

    Block *forBody = forOp.getBody();
    rewriter.setInsertionPointToStart(forBody);

    DenseMap<Type, Value> ivCasts;
    auto getIvAsType = [&](Type type) -> Value {
      auto it = ivCasts.find(type);
      if (it != ivCasts.end())
        return it->second;
      Value casted = CastIndexToType(rewriter, loc, forOp.getInductionVar(),
                                     type);
      ivCasts[type] = casted;
      return casted;
    };

    IRMapping mapping;
    for (Operation &op : *bodyBlock) {
      if (llvm::isa<scf::YieldOp>(op))
        continue;
      if (auto store = llvm::dyn_cast<memref::StoreOp>(&op)) {
        if (store.getMemref() == inductionMemref &&
            store.getIndices().size() == 1 &&
            IsZeroIndex(store.getIndices().front()))
          continue;
      }
      if (auto load = llvm::dyn_cast<memref::LoadOp>(&op)) {
        if (load.getMemref() == inductionMemref &&
            load.getIndices().size() == 1 &&
            IsZeroIndex(load.getIndices().front())) {
          Value casted = getIvAsType(load.getType());
          if (!casted)
            return failure();
          mapping.map(load.getResult(), casted);
          continue;
        }
      }
      rewriter.clone(op, mapping);
    }
    rewriter.eraseOp(loop);
    return success();
  }
};

struct AttachLoopAnnotationsPass
    : public PassWrapper<AttachLoopAnnotationsPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const final {
    return "loom-attach-loop-annotations";
  }
  StringRef getDescription() const final {
    return "Attach loop marker annotations to SCF loop operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ProcessRegion(module.getBodyRegion());

    llvm::SmallVector<Operation *, 8> toErase;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (IsLoopMarkerCallee(func.getName()))
        toErase.push_back(func.getOperation());
    }
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (IsLoopMarkerCallee(func.getName()))
        toErase.push_back(func.getOperation());
    }
    for (Operation *op : toErase)
      op->erase();
  }
};

struct UpliftWhileToForPass
    : public PassWrapper<UpliftWhileToForPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "loom-uplift-while-to-for"; }
  StringRef getDescription() const final {
    return "Uplift scf.while loops to scf.for when possible";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<WhileMemrefToForPattern>(context);
    scf::populateUpliftWhileToForPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace loom {

std::unique_ptr<mlir::Pass> createUpliftWhileToForPass() {
  return std::make_unique<UpliftWhileToForPass>();
}

std::unique_ptr<mlir::Pass> createAttachLoopAnnotationsPass() {
  return std::make_unique<AttachLoopAnnotationsPass>();
}

} // namespace loom
