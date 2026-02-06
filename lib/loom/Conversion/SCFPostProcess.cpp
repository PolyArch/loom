//===-- SCFPostProcess.cpp - SCF post-processing passes ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements post-processing passes for SCF IR. The UpliftWhileToFor
// pass converts scf.while loops with induction patterns to scf.for loops,
// handling both memref-based and iter-arg-based induction variables. The
// AttachLoopAnnotations pass collects Loom pragma marker calls and attaches
// their annotations to corresponding loop operations.
//
//===----------------------------------------------------------------------===//

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
#include "mlir/IR/Verifier.h"
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

bool HasStoreToMemref(Operation *root, Value memref);
Value CloneLoopInvariantValue(Value value, Operation *loop,
                              PatternRewriter &rewriter,
                              DenseMap<Value, Value> &cache);
bool IsPassThroughYield(Block &block);


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

struct StepInfo {
  int64_t constant = 0;
  Value value;
  bool isConst = false;
};

struct StreamStepInfo {
  int64_t constant = 0;
  Value value;
  bool isConst = false;
  llvm::StringRef stepOp;
};

enum class CmpKind {
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  NotEqual,
};

bool IsUnsignedPredicate(arith::CmpIPredicate pred) {
  return pred == arith::CmpIPredicate::ult ||
         pred == arith::CmpIPredicate::ule ||
         pred == arith::CmpIPredicate::ugt ||
         pred == arith::CmpIPredicate::uge;
}

arith::CmpIPredicate SwapPredicate(arith::CmpIPredicate pred) {
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

bool GetCmpKind(arith::CmpIPredicate pred, CmpKind &kind) {
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

bool MatchInductionUpdateValue(Value value, BlockArgument inductionArg,
                               StepInfo &step) {
  if (!value)
    return false;
  value = StripCasts(value);
  if (auto addi = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = StripCasts(addi.getLhs());
    Value rhs = StripCasts(addi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.constant = constant;
        step.isConst = true;
      } else {
        step.value = rhs;
        step.isConst = false;
      }
      return true;
    }
    if (rhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(lhs, constant)) {
        step.constant = constant;
        step.isConst = true;
      } else {
        step.value = lhs;
        step.isConst = false;
      }
      return true;
    }
  }
  if (auto subi = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = StripCasts(subi.getLhs());
    Value rhs = StripCasts(subi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.constant = -constant;
        step.isConst = true;
      } else {
        return false;
      }
      return true;
    }
  }
  return false;
}

bool MatchStreamUpdateValue(Value value, BlockArgument inductionArg,
                            StreamStepInfo &step) {
  if (!value)
    return false;
  value = StripCasts(value);
  if (auto addi = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = StripCasts(addi.getLhs());
    Value rhs = StripCasts(addi.getRhs());
    Value other;
    if (lhs == inductionArg)
      other = rhs;
    else if (rhs == inductionArg)
      other = lhs;
    else
      other = {};
    if (other) {
      int64_t constant = 0;
      if (GetConstantInt(other, constant)) {
        step.isConst = true;
        if (constant < 0) {
          step.constant = -constant;
          step.stepOp = "-=";
        } else {
          step.constant = constant;
          step.stepOp = "+=";
        }
      } else {
        step.isConst = false;
        step.value = other;
        step.stepOp = "+=";
      }
      return true;
    }
  }
  if (auto subi = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = StripCasts(subi.getLhs());
    Value rhs = StripCasts(subi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        if (constant < 0) {
          step.constant = -constant;
          step.stepOp = "+=";
        } else {
          step.constant = constant;
          step.stepOp = "-=";
        }
      } else {
        step.isConst = false;
        step.value = rhs;
        step.stepOp = "-=";
      }
      return true;
    }
  }
  if (auto muli = value.getDefiningOp<arith::MulIOp>()) {
    Value lhs = StripCasts(muli.getLhs());
    Value rhs = StripCasts(muli.getRhs());
    Value other;
    if (lhs == inductionArg)
      other = rhs;
    else if (rhs == inductionArg)
      other = lhs;
    else
      other = {};
    if (other) {
      int64_t constant = 0;
      if (GetConstantInt(other, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = other;
      }
      step.stepOp = "*=";
      return true;
    }
  }
  if (auto divsi = value.getDefiningOp<arith::DivSIOp>()) {
    Value lhs = StripCasts(divsi.getLhs());
    Value rhs = StripCasts(divsi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "/=";
      return true;
    }
  }
  if (auto divui = value.getDefiningOp<arith::DivUIOp>()) {
    Value lhs = StripCasts(divui.getLhs());
    Value rhs = StripCasts(divui.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "/=";
      return true;
    }
  }
  if (auto shl = value.getDefiningOp<arith::ShLIOp>()) {
    Value lhs = StripCasts(shl.getLhs());
    Value rhs = StripCasts(shl.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "<<=";
      return true;
    }
  }
  if (auto shrsi = value.getDefiningOp<arith::ShRSIOp>()) {
    Value lhs = StripCasts(shrsi.getLhs());
    Value rhs = StripCasts(shrsi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = ">>=";
      return true;
    }
  }
  if (auto shrui = value.getDefiningOp<arith::ShRUIOp>()) {
    Value lhs = StripCasts(shrui.getLhs());
    Value rhs = StripCasts(shrui.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (GetConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = ">>=";
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

bool IsIndexLike(Type type) {
  return type && (type.isIndex() || llvm::isa<IntegerType>(type));
}

bool IsSideEffectFree(Operation &op) {
  if (op.getNumRegions() != 0)
    return false;
  if (auto memEffect = llvm::dyn_cast<MemoryEffectOpInterface>(&op))
    return memEffect.hasNoEffect();
  return false;
}

bool IsPassThroughYield(Block &block) {
  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(block.getTerminator());
  if (!yieldOp)
    return false;
  if (yieldOp.getNumOperands() != block.getNumArguments())
    return false;

  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (StripCasts(yieldOp.getOperand(i)) != block.getArgument(i))
      return false;
  }

  for (Operation &op : block) {
    if (llvm::isa<scf::YieldOp>(op))
      continue;
    if (!IsSideEffectFree(op))
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

bool GetStopCond(CmpKind kind, llvm::StringRef &stopCond) {
  switch (kind) {
  case CmpKind::Less:
    stopCond = "<";
    return true;
  case CmpKind::LessEqual:
    stopCond = "<=";
    return true;
  case CmpKind::Greater:
    stopCond = ">";
    return true;
  case CmpKind::GreaterEqual:
    stopCond = ">=";
    return true;
  case CmpKind::NotEqual:
    stopCond = "!=";
    return true;
  }
  return false;
}

struct StreamWhileInfo {
  int ivIndex = -1;
  llvm::StringRef stepOp;
  llvm::StringRef stopCond;
  bool cmpOnUpdate = false;
};

static bool AreSameStep(const StreamStepInfo &lhs, const StreamStepInfo &rhs) {
  if (lhs.stepOp != rhs.stepOp)
    return false;
  if (lhs.isConst != rhs.isConst)
    return false;
  if (lhs.isConst)
    return lhs.constant == rhs.constant;
  if (!lhs.value || !rhs.value)
    return false;
  return StripCasts(lhs.value) == StripCasts(rhs.value);
}

bool AnalyzeStreamableWhile(scf::WhileOp loop, StreamWhileInfo &info,
                            PatternRewriter *rewriter) {
  if (!loop.getBefore().hasOneBlock() || !loop.getAfter().hasOneBlock())
    return false;

  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();

  auto conditionOp = llvm::dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!conditionOp)
    return false;
  if (conditionOp.getArgs().size() != before.getNumArguments())
    return false;

  bool conditionArgsPassThrough = true;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (StripCasts(conditionOp.getArgs()[i]) != before.getArgument(i)) {
      conditionArgsPassThrough = false;
      break;
    }
  }

  bool beforeSideEffectFree = true;
  for (Operation &op : before) {
    if (llvm::isa<scf::ConditionOp>(op))
      continue;
    if (auto load = llvm::dyn_cast<memref::LoadOp>(&op)) {
      if (!HasStoreToMemref(loop, load.getMemref()))
        continue;
    }
    if (!IsSideEffectFree(op)) {
      beforeSideEffectFree = false;
      break;
    }
  }

  bool afterPassThrough = IsPassThroughYield(after);
  bool bodyInBefore = false;
  if (conditionArgsPassThrough && beforeSideEffectFree) {
    bodyInBefore = false;
  } else if (afterPassThrough) {
    bodyInBefore = true;
  } else {
    return false;
  }

  Value condValue = StripCasts(conditionOp.getCondition());
  auto cmpOp = condValue.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp)
    return false;

  auto pred = cmpOp.getPredicate();
  CmpKind cmpKind;
  if (!GetCmpKind(pred, cmpKind))
    return false;

  Value lhs = StripCasts(cmpOp.getLhs());
  Value rhs = StripCasts(cmpOp.getRhs());
  int ivIndex = -1;
  bool ivOnLhs = false;
  bool cmpUsesUpdate = false;
  StreamStepInfo cmpStep;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (lhs == before.getArgument(i)) {
      ivIndex = static_cast<int>(i);
      ivOnLhs = true;
      break;
    }
    if (rhs == before.getArgument(i)) {
      ivIndex = static_cast<int>(i);
      ivOnLhs = false;
      break;
    }
    StreamStepInfo candidate;
    if (MatchStreamUpdateValue(lhs, before.getArgument(i), candidate)) {
      ivIndex = static_cast<int>(i);
      ivOnLhs = true;
      cmpUsesUpdate = true;
      cmpStep = candidate;
      break;
    }
    if (MatchStreamUpdateValue(rhs, before.getArgument(i), candidate)) {
      ivIndex = static_cast<int>(i);
      ivOnLhs = false;
      cmpUsesUpdate = true;
      cmpStep = candidate;
      break;
    }
  }
  if (ivIndex < 0)
    return false;

  if (!ivOnLhs) {
    pred = SwapPredicate(pred);
    if (!GetCmpKind(pred, cmpKind))
      return false;
    std::swap(lhs, rhs);
  }

  if (!IsIndexLike(before.getArgument(ivIndex).getType()))
    return false;

  Value boundValue = rhs;
  if (IsDefinedIn(loop, boundValue)) {
    if (!rewriter)
      return false;
    DenseMap<Value, Value> hoisted;
    OpBuilder::InsertionGuard guard(*rewriter);
    rewriter->setInsertionPoint(loop);
    Value cloned = CloneLoopInvariantValue(boundValue, loop, *rewriter, hoisted);
    if (!cloned)
      return false;
    if (ivOnLhs)
      cmpOp->setOperand(1, cloned);
    else
      cmpOp->setOperand(0, cloned);
    boundValue = cloned;
  }
  if (!IsIndexLike(boundValue.getType()))
    return false;

  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != loop.getInits().size())
    return false;

  if (ivIndex >= static_cast<int>(yieldOp.getNumOperands()))
    return false;

  StreamStepInfo stepInfo;
  Value updateValue = bodyInBefore
                          ? StripCasts(conditionOp.getArgs()[ivIndex])
                          : StripCasts(yieldOp.getOperand(ivIndex));
  BlockArgument updateBase =
      bodyInBefore ? before.getArgument(ivIndex) : after.getArgument(ivIndex);
  if (!MatchStreamUpdateValue(updateValue, updateBase, stepInfo))
    return false;
  if (stepInfo.isConst && stepInfo.constant == 0)
    return false;
  if (!stepInfo.isConst && IsDefinedIn(loop, stepInfo.value))
    return false;
  if (!stepInfo.isConst && !IsIndexLike(stepInfo.value.getType()))
    return false;

  if (!loop.getResult(ivIndex).use_empty())
    return false;

  if (cmpUsesUpdate) {
    if (!AreSameStep(cmpStep, stepInfo))
      return false;
  }

  llvm::StringRef stopCond;
  if (!GetStopCond(cmpKind, stopCond))
    return false;

  info.ivIndex = ivIndex;
  info.stepOp = stepInfo.stepOp;
  info.stopCond = stopCond;
  info.cmpOnUpdate = cmpUsesUpdate;
  return true;
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
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value hoisted = memref::LoadOp::create(rewriter, loc, memref, zeroIndex);
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
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
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
    return arith::IndexCastOp::create(builder, loc, targetType, value);
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
  CmpKind cmpKind;
  if (!GetCmpKind(pred, cmpKind))
    return failure();

  Value lhs = StripCasts(cmpOp.getLhs());
  Value rhs = StripCasts(cmpOp.getRhs());
  int inductionIndex = -1;
  bool compareUsesUpdate = false;
  StepInfo stepInfo;
  Value updateValue;
  bool matched = false;

  auto matchOperand = [&](Value operand, int &idx, bool &usesUpdate,
                          StepInfo &step) -> bool {
    for (unsigned i = 0; i < before.getNumArguments(); ++i) {
      if (operand == before.getArgument(i)) {
        idx = static_cast<int>(i);
        usesUpdate = false;
        return true;
      }
      StepInfo candidateStep;
      if (MatchInductionUpdateValue(operand, before.getArgument(i),
                                    candidateStep)) {
        idx = static_cast<int>(i);
        usesUpdate = true;
        step = candidateStep;
        return true;
      }
    }
    return false;
  };

  int lhsIndex = -1;
  bool lhsUsesUpdate = false;
  StepInfo lhsStep;
  bool lhsMatch = matchOperand(lhs, lhsIndex, lhsUsesUpdate, lhsStep);

  int rhsIndex = -1;
  bool rhsUsesUpdate = false;
  StepInfo rhsStep;
  bool rhsMatch = matchOperand(rhs, rhsIndex, rhsUsesUpdate, rhsStep);

  if (lhsMatch && rhsMatch)
    return failure();
  if (rhsMatch) {
    std::swap(lhs, rhs);
    pred = SwapPredicate(pred);
    GetCmpKind(pred, cmpKind);
    inductionIndex = rhsIndex;
    compareUsesUpdate = rhsUsesUpdate;
    stepInfo = rhsStep;
    matched = true;
  } else if (lhsMatch) {
    inductionIndex = lhsIndex;
    compareUsesUpdate = lhsUsesUpdate;
    stepInfo = lhsStep;
    matched = true;
  }
  if (!matched || inductionIndex < 0)
    return failure();
  updateValue = lhs;

  if (compareUsesUpdate)
    return failure();

  auto yieldOp = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != loop.getInits().size())
    return failure();
  if (conditionOp.getArgs().size() != before.getNumArguments())
    return failure();

  if (compareUsesUpdate) {
    if (inductionIndex >= static_cast<int>(conditionOp.getArgs().size()))
      return failure();
    if (StripCasts(conditionOp.getArgs()[inductionIndex]) !=
        StripCasts(updateValue))
      return failure();
    if (StripCasts(yieldOp.getOperand(inductionIndex)) !=
        after.getArgument(inductionIndex))
      return failure();
  } else {
    if (!MatchInductionUpdateValue(yieldOp.getOperand(inductionIndex),
                                   after.getArgument(inductionIndex),
                                   stepInfo))
      return failure();
  }
  if (stepInfo.isConst && stepInfo.constant == 0)
    return failure();

  if (!loop.getResult(inductionIndex).use_empty())
    return failure();

  Location loc = loop.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  Value initValue = loop.getInits()[inductionIndex];
  Value boundValue = rhs;

  Value boundOutside = boundValue;
  if (IsDefinedIn(loop, boundValue)) {
    DenseMap<Value, Value> hoisted;
    boundOutside = CloneLoopInvariantValue(boundValue, loop, rewriter, hoisted);
    if (!boundOutside)
      return failure();
  }

  Value stepOutside;
  if (!stepInfo.isConst) {
    stepOutside = stepInfo.value;
    if (!stepOutside)
      return failure();
    if (IsDefinedIn(loop, stepOutside)) {
      DenseMap<Value, Value> hoisted;
      stepOutside = CloneLoopInvariantValue(stepOutside, loop, rewriter, hoisted);
      if (!stepOutside)
        return failure();
    }
  }

  Value adjustedBound = boundOutside;

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

  Value lowerIndex;
  Value upperIndex;
  Value initIndex = CastToIndex(rewriter, loc, initValue);
  if (!initIndex)
    return failure();
  Value boundIndex = CastToIndex(rewriter, loc, adjustedBound);
  if (!boundIndex)
    return failure();

  if (directionAscending) {
    lowerIndex = initIndex;
    upperIndex = boundIndex;
    if (cmpKind == CmpKind::LessEqual) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      upperIndex = arith::AddIOp::create(rewriter, loc, upperIndex, one);
    }
  } else {
    Value diff = arith::SubIOp::create(rewriter, loc, initIndex, boundIndex);
    upperIndex = diff;
    if (cmpKind == CmpKind::GreaterEqual) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      upperIndex = arith::AddIOp::create(rewriter, loc, upperIndex, one);
    }
    lowerIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  }

  Value stepIndex;
  if (stepInfo.isConst) {
    int64_t stepConst = stepInfo.constant;
    int64_t stepAbs = stepConst > 0 ? stepConst : -stepConst;
    stepIndex = arith::ConstantIndexOp::create(rewriter, loc, stepAbs);
  } else {
    Value stepValue = stepOutside;
    if (!stepValue)
      return failure();
    stepIndex = CastToIndex(rewriter, loc, stepValue);
    if (!stepIndex)
      return failure();
  }

  SmallVector<Value, 4> initArgs;
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (static_cast<int>(i) == inductionIndex)
      continue;
    initArgs.push_back(loop.getInits()[i]);
  }

  bool useBeforeBody = compareUsesUpdate;
  if (compareUsesUpdate) {
    if (after.getOperations().size() != 1)
      return failure();
    for (unsigned i = 0; i < after.getNumArguments(); ++i) {
      if (StripCasts(yieldOp.getOperand(i)) != after.getArgument(i))
        return failure();
    }
  }

  auto forOp =
      scf::ForOp::create(rewriter, loc, lowerIndex, upperIndex, stepIndex,
                                  initArgs);
  scf::ForOp::ensureTerminator(forOp.getRegion(), rewriter, loc);
  if (auto attr = loop->getAttr("loom.annotations"))
    forOp->setAttr("loom.annotations", attr);
  if (IsUnsignedPredicate(pred))
    forOp.setUnsignedCmp(true);

  Value mappedIvIndex;
  auto getMappedIvIndex = [&](OpBuilder &builder) -> Value {
    if (directionAscending)
      return forOp.getInductionVar();
    if (!mappedIvIndex)
      mappedIvIndex =
          arith::SubIOp::create(builder, loc, initIndex, forOp.getInductionVar());
    return mappedIvIndex;
  };

  Block *forBody = forOp.getBody();
  if (useBeforeBody) {
    Operation *terminator = forBody->getTerminator();
    if (!terminator)
      return failure();
    OpBuilder bodyBuilder(rewriter.getContext());
    bodyBuilder.setInsertionPoint(terminator);
    IRMapping mapping;
    auto newIterArgs = forOp.getRegionIterArgs();
    unsigned iterIndex = 0;
    auto oldYield = llvm::dyn_cast<scf::YieldOp>(terminator);
    if (!oldYield)
      return failure();

    for (unsigned i = 0; i < before.getNumArguments(); ++i) {
      BlockArgument arg = before.getArgument(i);
      if (static_cast<int>(i) == inductionIndex) {
        Value mappedIndex = getMappedIvIndex(bodyBuilder);
        Value casted =
            CastIndexToType(bodyBuilder, loc, mappedIndex, arg.getType());
        if (!casted)
          return failure();
        mapping.map(arg, casted);
      } else {
        mapping.map(arg, newIterArgs[iterIndex++]);
      }
    }

    for (Operation &op : before) {
      if (llvm::isa<scf::ConditionOp>(op))
        continue;
      bodyBuilder.clone(op, mapping);
    }
    SmallVector<Value, 4> newYieldOperands;
    newYieldOperands.reserve(initArgs.size());
    for (unsigned i = 0; i < conditionOp.getArgs().size(); ++i) {
      if (static_cast<int>(i) == inductionIndex)
        continue;
      Value operand = conditionOp.getArgs()[i];
      Value mapped = mapping.lookupOrDefault(operand);
      if (mapped == operand && IsDefinedIn(loop, operand))
        return failure();
      newYieldOperands.push_back(mapped);
    }
    rewriter.modifyOpInPlace(oldYield, [&]() {
      oldYield->setOperands(newYieldOperands);
    });

    if (failed(verify(forOp))) {
      rewriter.eraseOp(forOp);
      return failure();
    }

    unsigned resultIndex = 0;
    for (unsigned i = 0; i < loop.getResults().size(); ++i) {
      if (static_cast<int>(i) == inductionIndex)
        continue;
      loop.getResult(i).replaceAllUsesWith(forOp.getResult(resultIndex++));
    }

    rewriter.eraseOp(loop);
    return success();
  }
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
    Value mappedIndex = getMappedIvIndex(rewriter);
    Value casted = CastIndexToType(rewriter, loc, mappedIndex, type);
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
    scf::YieldOp::create(rewriter, loc, newYieldOperands);

  if (failed(verify(forOp))) {
    rewriter.eraseOp(forOp);
    return failure();
  }

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

void ProcessRegion(Region &region, SmallVectorImpl<StringAttr> &pending) {
  for (Block &block : region) {
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
        ProcessRegion(nested, pending);
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
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      upperIndex = arith::AddIOp::create(rewriter, loc, upperIndex, one);
    }

    Value stepIndex = arith::ConstantIndexOp::create(rewriter, loc, step);
    auto forOp =
        scf::ForOp::create(rewriter, loc, lowerIndex, upperIndex, stepIndex);
    if (auto attr = loop->getAttr("loom.annotations"))
      forOp->setAttr("loom.annotations", attr);

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
    SmallVector<StringAttr, 4> pending;
    ProcessRegion(module.getBodyRegion(), pending);

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

struct MarkWhileStreamablePass
    : public PassWrapper<MarkWhileStreamablePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "loom-mark-while-streamable"; }
  StringRef getDescription() const final {
    return "Annotate streamable scf.while loops for dataflow lowering";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::WhileOp, 8> loops;
    module.walk([&](scf::WhileOp loop) { loops.push_back(loop); });

    for (scf::WhileOp loop : loops) {
      if (loop->hasAttr("loom.stream"))
        continue;
      PatternRewriter rewriter(loop.getContext());
      StreamWhileInfo info;
      if (!AnalyzeStreamableWhile(loop, info, &rewriter))
        continue;
      Builder builder(loop.getContext());
      auto dict = builder.getDictionaryAttr({
          builder.getNamedAttr("iv",
                               builder.getI64IntegerAttr(info.ivIndex)),
          builder.getNamedAttr("step_op",
                               builder.getStringAttr(info.stepOp)),
          builder.getNamedAttr("stop_cond",
                               builder.getStringAttr(info.stopCond)),
          builder.getNamedAttr("cmp_on_update",
                               builder.getBoolAttr(info.cmpOnUpdate)),
      });
      loop->setAttr("loom.stream", dict);
    }
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
    FrozenRewritePatternSet frozen(std::move(patterns));

    for (auto func : getOperation().getOps<func::FuncOp>())
      (void)applyPatternsAndFoldGreedily(func, frozen);
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

std::unique_ptr<mlir::Pass> createMarkWhileStreamablePass() {
  return std::make_unique<MarkWhileStreamablePass>();
}

} // namespace loom
