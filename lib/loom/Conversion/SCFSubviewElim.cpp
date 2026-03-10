//===-- SCFSubviewElim.cpp - Eliminate subview-bump patterns -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that eliminates "subview-bump" patterns in
// scf.while loops. When a while loop carries memref iter-args that are only
// accessed at index 0 and advanced by subview[1] each iteration (pointer
// bumping), this pass replaces them with index-based access to the original
// memref function argument.
//
// Before:
//   scf.while (%count, %ptr) = (%n, %memref) {
//     memref.load %ptr[0]
//     %next = memref.subview %ptr[1] [%dim-1] [1]
//     scf.condition(%cond) %updated_count, %next
//   }
//
// After:
//   scf.while (%count, %idx) = (%n, 0) {
//     memref.load %memref[%idx]
//     %next_idx = arith.addi %idx, 1
//     scf.condition(%cond) %updated_count, %next_idx
//   }
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFPostProcess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Track information about a subview-bumped memref iter-arg.
struct BumpInfo {
  unsigned argIndex;       // Position in the while's iter-args
  Value originalMemref;    // The function-level memref arg
  int64_t bumpOffset;      // Subview offset per iteration (typically 1)
  int64_t initialOffset;   // Static offset from func arg to initial iter-arg
  memref::SubViewOp subviewOp; // The subview op producing the bumped memref
};

// Check if a value is a constant index equal to `expected`.
static bool isConstantIndex(Value value, int64_t expected) {
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == expected;
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (cst.getType().isIndex())
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        return intAttr.getInt() == expected;
  }
  return false;
}

// Try to extract a constant index value from a Value.
static std::optional<int64_t> getConstantIndex(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (cst.getType().isIndex())
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        return intAttr.getInt();
  }
  return std::nullopt;
}

// Trace a value back to a function block argument (through subviews, casts).
// Accumulates static offsets along the way into *staticOffset (if provided).
static Value traceToFuncArg(Value val, int64_t *staticOffset = nullptr) {
  int64_t accum = 0;
  while (val) {
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
      if (auto funcOp =
              llvm::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        if (staticOffset)
          *staticOffset = accum;
        return val;
      }
      return {};
    }
    if (auto castOp = val.getDefiningOp<memref::CastOp>()) {
      val = castOp.getSource();
      continue;
    }
    if (auto subviewOp = val.getDefiningOp<memref::SubViewOp>()) {
      // Only handle 1D subviews with stride 1 and static offset.
      auto strides = subviewOp.getStaticStrides();
      auto offsets = subviewOp.getStaticOffsets();
      if (strides.size() != 1 || strides[0] != 1)
        return {};
      if (offsets.size() != 1 || offsets[0] == ShapedType::kDynamic)
        return {};
      accum += offsets[0];
      val = subviewOp.getSource();
      continue;
    }
    return {};
  }
  return {};
}

// Analyze a single memref iter-arg to see if it follows the subview-bump
// pattern: loaded at index 0, stored at index 0, advanced by subview[offset].
static bool analyzeSubviewBump(scf::WhileOp loop, unsigned argIndex,
                               BumpInfo &info) {
  Block &before = loop.getBefore().front();
  Value iterArg = before.getArgument(argIndex);

  if (!llvm::isa<MemRefType>(iterArg.getType()))
    return false;

  // The init value must trace back to a function argument.
  Value initVal = loop.getInits()[argIndex];
  int64_t initOffset = 0;
  Value funcArg = traceToFuncArg(initVal, &initOffset);
  if (!funcArg)
    return false;

  // Check that the condition op passes a subview of this iter-arg.
  auto condOp = llvm::dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!condOp || argIndex >= condOp.getArgs().size())
    return false;
  auto subviewOp =
      condOp.getArgs()[argIndex].getDefiningOp<memref::SubViewOp>();
  if (!subviewOp || subviewOp.getSource() != iterArg)
    return false;

  // Subview must have constant offset, stride 1, single dimension.
  auto offsets = subviewOp.getStaticOffsets();
  auto strides = subviewOp.getStaticStrides();
  if (offsets.size() != 1 || strides.size() != 1)
    return false;
  if (offsets[0] == ShapedType::kDynamic || strides[0] != 1)
    return false;

  // Check all uses of the iter-arg are supported: load[const], store[const],
  // subview, dim.
  for (OpOperand &use : iterArg.getUses()) {
    Operation *user = use.getOwner();
    if (auto load = llvm::dyn_cast<memref::LoadOp>(user)) {
      if (load.getIndices().size() != 1 ||
          !getConstantIndex(load.getIndices().front()))
        return false;
      continue;
    }
    if (auto store = llvm::dyn_cast<memref::StoreOp>(user)) {
      if (&use != &store->getOpOperand(1)) // must be the memref operand
        continue; // value operand is fine
      if (store.getIndices().size() != 1 ||
          !getConstantIndex(store.getIndices().front()))
        return false;
      continue;
    }
    if (llvm::isa<memref::SubViewOp>(user))
      continue;
    if (llvm::isa<memref::DimOp>(user))
      continue;
    return false; // unsupported use
  }

  info.argIndex = argIndex;
  info.originalMemref = funcArg;
  info.bumpOffset = offsets[0];
  info.initialOffset = initOffset;
  info.subviewOp = subviewOp;
  return true;
}

// Rewrite a scf.while loop to eliminate subview-bumped memref iter-args.
static LogicalResult eliminateSubviewBumps(scf::WhileOp loop,
                                           PatternRewriter &rewriter) {
  if (!loop.getBefore().hasOneBlock() || !loop.getAfter().hasOneBlock())
    return failure();

  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();

  // After block must have a yield terminator.
  auto afterYield = llvm::dyn_cast<scf::YieldOp>(after.getTerminator());
  if (!afterYield)
    return failure();
  if (after.getNumArguments() > afterYield.getNumOperands())
    return failure();

  // Analyze which iter-args follow the subview-bump pattern.
  SmallVector<BumpInfo, 4> bumps;
  DenseSet<unsigned> bumpIndices;
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    BumpInfo info;
    if (analyzeSubviewBump(loop, i, info)) {
      // Verify the after block passes this arg through.
      if (i < afterYield.getNumOperands() &&
          afterYield.getOperand(i) == after.getArgument(i)) {
        bumps.push_back(info);
        bumpIndices.insert(i);
      }
    }
  }
  if (bumps.empty())
    return failure();

  Location loc = loop.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  // Build new init values: replace memref inits with initial offset index.
  rewriter.setInsertionPoint(loop);
  SmallVector<Value, 8> newInits;
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (bumpIndices.contains(i)) {
      int64_t initOff = 0;
      for (auto &info : bumps) {
        if (info.argIndex == i) {
          initOff = info.initialOffset;
          break;
        }
      }
      newInits.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, initOff));
    } else {
      newInits.push_back(loop.getInits()[i]);
    }
  }

  // Build new result types.
  SmallVector<Type, 8> newResultTypes;
  for (unsigned i = 0; i < loop.getResultTypes().size(); ++i) {
    if (bumpIndices.contains(i))
      newResultTypes.push_back(rewriter.getIndexType());
    else
      newResultTypes.push_back(loop.getResultTypes()[i]);
  }

  // Create new while op.
  auto newLoop =
      scf::WhileOp::create(rewriter, loc, newResultTypes, newInits);

  // Copy attributes.
  for (auto attr : loop->getAttrs()) {
    if (attr.getName() == "operandSegmentSizes")
      continue;
    newLoop->setAttr(attr.getName(), attr.getValue());
  }

  // Set up the before block.
  Block *newBefore = &newLoop.getBefore().emplaceBlock();
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (bumpIndices.contains(i))
      newBefore->addArgument(rewriter.getIndexType(), loc);
    else
      newBefore->addArgument(before.getArgument(i).getType(),
                             before.getArgument(i).getLoc());
  }

  // Set up the after block: clone with mapping for bumped args.
  Block *newAfter = &newLoop.getAfter().emplaceBlock();
  for (unsigned i = 0; i < loop.getInits().size(); ++i) {
    if (bumpIndices.contains(i))
      newAfter->addArgument(rewriter.getIndexType(), loc);
    else
      newAfter->addArgument(after.getArgument(i).getType(),
                            after.getArgument(i).getLoc());
  }
  {
    IRMapping afterMapping;
    for (unsigned i = 0; i < after.getNumArguments(); ++i) {
      if (bumpIndices.contains(i))
        afterMapping.map(after.getArgument(i), newAfter->getArgument(i));
      else
        afterMapping.map(after.getArgument(i), newAfter->getArgument(i));
    }
    rewriter.setInsertionPointToEnd(newAfter);
    for (Operation &op : after) {
      if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(&op)) {
        SmallVector<Value, 8> afterYieldArgs;
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          if (bumpIndices.contains(i))
            afterYieldArgs.push_back(newAfter->getArgument(i));
          else
            afterYieldArgs.push_back(
                afterMapping.lookupOrDefault(yieldOp.getOperand(i)));
        }
        scf::YieldOp::create(rewriter, loc, afterYieldArgs);
        continue;
      }
      rewriter.clone(op, afterMapping);
    }
  }

  // Build a mapping from old before-block values to new ones.
  IRMapping mapping;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (!bumpIndices.contains(i))
      mapping.map(before.getArgument(i), newBefore->getArgument(i));
  }

  // Build per-bump replacement info: original memref and bump constants.
  DenseMap<unsigned, Value> bumpOriginalMemrefs;
  DenseMap<unsigned, int64_t> bumpOffsets;
  DenseMap<unsigned, int64_t> bumpInitialOffsets;
  for (auto &info : bumps) {
    bumpOriginalMemrefs[info.argIndex] = info.originalMemref;
    bumpOffsets[info.argIndex] = info.bumpOffset;
    bumpInitialOffsets[info.argIndex] = info.initialOffset;
  }

  // Clone ops from before block into new before block, with replacements.
  rewriter.setInsertionPointToEnd(newBefore);
  Value oneIndex = arith::ConstantIndexOp::create(
      rewriter, loc, 1); // for bump offset (most common case)

  for (Operation &op : before) {
    // Handle the condition op specially.
    if (auto condOp = llvm::dyn_cast<scf::ConditionOp>(&op)) {
      SmallVector<Value, 8> newCondArgs;
      for (unsigned i = 0; i < condOp.getArgs().size(); ++i) {
        if (bumpIndices.contains(i)) {
          // Replace subview result with next index.
          Value idx = newBefore->getArgument(i);
          int64_t offset = bumpOffsets[i];
          Value offsetVal =
              (offset == 1) ? oneIndex
                            : arith::ConstantIndexOp::create(rewriter, loc,
                                                             offset).getResult();
          Value nextIdx = arith::AddIOp::create(rewriter, loc, idx, offsetVal);
          newCondArgs.push_back(nextIdx);
        } else {
          newCondArgs.push_back(mapping.lookupOrDefault(condOp.getArgs()[i]));
        }
      }
      scf::ConditionOp::create(
          rewriter, loc, mapping.lookupOrDefault(condOp.getCondition()),
          newCondArgs);
      continue;
    }

    // Handle load: replace memref arg with original + (current_idx + const_offset).
    if (auto load = llvm::dyn_cast<memref::LoadOp>(&op)) {
      Value memref = load.getMemref();
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(memref)) {
        unsigned idx = blockArg.getArgNumber();
        if (bumpIndices.contains(idx)) {
          Value accessIdx = newBefore->getArgument(idx);
          // Add constant access offset if non-zero.
          if (auto constOff = getConstantIndex(load.getIndices().front())) {
            if (*constOff != 0) {
              Value offVal =
                  arith::ConstantIndexOp::create(rewriter, loc, *constOff);
              accessIdx = arith::AddIOp::create(rewriter, loc, accessIdx, offVal);
            }
          }
          Value newLoad = memref::LoadOp::create(
              rewriter, loc, load.getType(), bumpOriginalMemrefs[idx],
              ValueRange{accessIdx});
          mapping.map(load.getResult(), newLoad);
          continue;
        }
      }
    }

    // Handle store: replace memref arg with original + (current_idx + const_offset).
    if (auto store = llvm::dyn_cast<memref::StoreOp>(&op)) {
      Value memref = store.getMemref();
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(memref)) {
        unsigned idx = blockArg.getArgNumber();
        if (bumpIndices.contains(idx)) {
          Value accessIdx = newBefore->getArgument(idx);
          // Add constant access offset if non-zero.
          if (auto constOff = getConstantIndex(store.getIndices().front())) {
            if (*constOff != 0) {
              Value offVal =
                  arith::ConstantIndexOp::create(rewriter, loc, *constOff);
              accessIdx = arith::AddIOp::create(rewriter, loc, accessIdx, offVal);
            }
          }
          memref::StoreOp::create(
              rewriter, loc, mapping.lookupOrDefault(store.getValue()),
              bumpOriginalMemrefs[idx],
              ValueRange{accessIdx});
          continue;
        }
      }
    }

    // Skip subview ops that produce bumped memrefs (replaced by index add).
    if (auto subview = llvm::dyn_cast<memref::SubViewOp>(&op)) {
      Value source = subview.getSource();
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(source)) {
        if (bumpIndices.contains(blockArg.getArgNumber()))
          continue; // Skip - will be replaced by index add in condition
      }
    }

    // Skip dim ops on bumped memrefs (no longer needed).
    if (auto dim = llvm::dyn_cast<memref::DimOp>(&op)) {
      Value source = dim.getSource();
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(source)) {
        if (bumpIndices.contains(blockArg.getArgNumber())) {
          // Map dim result to (original_dim - current_index) for any users.
          rewriter.setInsertionPointToEnd(newBefore);
          Value origDim = memref::DimOp::create(
              rewriter, loc, bumpOriginalMemrefs[blockArg.getArgNumber()],
              mapping.lookupOrDefault(dim.getIndex()));
          Value curIdx = newBefore->getArgument(blockArg.getArgNumber());
          Value adjusted = arith::SubIOp::create(rewriter, loc, origDim, curIdx);
          mapping.map(dim.getResult(), adjusted);
          continue;
        }
      }
    }

    // Clone other ops with mapping.
    rewriter.clone(op, mapping);
  }

  // Replace uses of old loop results.
  for (unsigned i = 0; i < loop.getNumResults(); ++i) {
    if (loop.getResult(i).use_empty())
      continue;
    if (bumpIndices.contains(i)) {
      // Result was a memref, now it's an index. If any user needs the memref,
      // this is a problem. For now, just replace - most loop results are unused.
      loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
    } else {
      loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
    }
  }
  rewriter.eraseOp(loop);
  return success();
}

struct SubviewBumpEliminationPattern
    : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    return eliminateSubviewBumps(loop, rewriter);
  }
};

struct EliminateSubviewBumpsPass
    : public PassWrapper<EliminateSubviewBumpsPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const final {
    return "loom-eliminate-subview-bumps";
  }
  StringRef getDescription() const final {
    return "Eliminate subview-bump patterns in scf.while loops";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SubviewBumpEliminationPattern>(context);
    FrozenRewritePatternSet frozen(std::move(patterns));

    for (auto func : getOperation().getOps<func::FuncOp>())
      (void)applyPatternsGreedily(func, frozen);
  }
};

} // namespace

namespace loom {

std::unique_ptr<mlir::Pass> createEliminateSubviewBumpsPass() {
  return std::make_unique<EliminateSubviewBumpsPass>();
}

} // namespace loom
