// Lower `scf.if` ops inside `dataflow.graph.func` bodies into the
// dataflow control-routing primitives `dataflow.gate` and
// `dataflow.mux`.
//
// Scope of the rewrite (smoke baseline):
//   For each `scf.if` op found anywhere inside a `dataflow.graph.func`
//   body (post-order traversal so innermost shapes lower first):
//
//     a. scf.if WITH results, both then- and else-regions present, both
//        regions are pure (no `MemoryEffectOpInterface`-effecting op,
//        no call-op-interface op, no nested structured control flow):
//        emit one `dataflow.mux %sel, %then_yield_i, %else_yield_i ->
//        %mux_i : Ti` per result. Lift every body op of both regions
//        into the parent block (in original program order, then-region
//        first). Replace each scf.if result with the matching
//        dataflow.mux output and erase the scf.if.
//
//     b. scf.if WITHOUT results, only the then-region is non-empty
//        (the else-region is implicit / empty), and the then-region is
//        pure (same purity check as above): for every op in the
//        then-region whose result is a graph-friendly scalar
//        (i32 / i64 / f32 / f64 / index / !llvm.ptr), insert
//        `%after_cond, %after_value = dataflow.gate %cond, %op_value`
//        and rewrite the op's downstream uses to consume %after_value.
//        Lift the body ops to the parent block and erase the scf.if.
//
//     c. scf.if WITH results that the mux lift cannot handle (effectful
//        body, missing region, mismatched yield count): keep the scf.if
//        envelope unchanged and instead wrap each gate-friendly result
//        with a body-phase `dataflow.gate %cond, %if.result -> (i1,
//        Ti)`. Downstream consumers see the gate's `after_value`; the
//        unused `after_cond` is discarded. The conditional execution
//        of the body remains an `scf.if`, so memory effects fire on
//        the same lane they did before; only the consumer view changes.
//
//     d. scf.if WITH results whose then-region is a restricted
//        conditional-load reduction body and whose else-region yields the
//        loop-carried values: lift the then body, select a safe in-bounds
//        address for false lanes, then demux both the lifted then result and
//        the else value before muxing the selected lanes. The demuxes consume
//        one token from both sides every iteration, so false-lane safe loads
//        are drained instead of being buffered into a later true lane.
//
//     e. Anything else (no results and effectful body, both no-result
//        regions, scf.if without any gate-friendly result): leave the
//        scf.if alone and emit a remark.
//
// Why post-order: rewriting an outer scf.if first invalidates iterators
// when its body contains another scf.if we still want to lower. Walking
// in post-order means every inner scf.if is observed before its parent,
// so the parent's purity check sees a body that has already been
// flattened where possible.
//
// Why "pure" excludes nested SCF: nested scf.for / scf.while / scf.if
// own their own SSA blocks and possibly carry side effects that the
// surrounding pass cannot reason about. Lifting them out is unsafe;
// muxing across them would lose the iteration semantics.
//
// Why graph-friendly types only: the gate op admits AnyType for the
// value port, but a body op that produces a memref or a !llvm.array
// is plumbing the lowering pipeline still treats specially elsewhere
// (e.g., via unrealized_conversion_cast). Restricting gates to the
// scalar set used by the rest of the pipeline keeps the rewrite from
// surprising downstream consumers.
//
// The pass runs after `loom-lower-reduction-to-stream` so any enclosing
// loop has already been streamed (the `cond` of a scf.if inside a
// streamed body is already a body-phase value; gating it again would
// double-gate). Before `loom-lower-graph-constants` so the constants
// pass still sees the post-mux IR.

#include "Frontend/Lowering/Passes.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace {

constexpr ::llvm::StringLiteral kConditionalStoreSameAddressAttr =
    "loom.conditional_store_same_address";
constexpr ::llvm::StringLiteral kConditionalStoreDoneAttr =
    "loom.conditional_store_done";

// Return true if `t` is one of the scalar types the pass is willing to
// gate. The gate op's IR signature admits AnyType, but the rest of the
// frontend lowering only treats this restricted set as graph-friendly,
// so gating outside of it would surprise downstream consumers
// (e.g., consumers that expect a memref binding rather than a stream).
bool isGateFriendly(::mlir::Type t) {
  if (::llvm::isa<::mlir::IndexType>(t))
    return true;
  if (auto it = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return it.getWidth() == 32 || it.getWidth() == 64;
  if (auto ft = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return ft.getWidth() == 32 || ft.getWidth() == 64;
  if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(t))
    return true;
  return false;
}

::mlir::TypedAttr getIntegerLikeAttr(::mlir::OpBuilder &builder,
                                     ::mlir::Type type,
                                     std::int64_t value) {
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
    return builder.getIntegerAttr(intTy, value);
  if (::llvm::isa<::mlir::IndexType>(type))
    return builder.getIndexAttr(value);
  return {};
}

// A "pure" op for the purpose of scf.if lifting: no
// MemoryEffectOpInterface effects (or the interface reports purity),
// not a call-op-interface op (calls escape the local SSA scope), and
// not a nested structured control-flow op (those carry their own
// regions we cannot flatten safely here).
bool isLiftablePure(::mlir::Operation *op) {
  if (::llvm::isa<::mlir::scf::ForOp, ::mlir::scf::ForallOp,
                  ::mlir::scf::WhileOp, ::mlir::scf::IfOp,
                  ::mlir::scf::ParallelOp, ::mlir::scf::ExecuteRegionOp>(op))
    return false;
  if (::llvm::isa<::mlir::CallOpInterface>(op))
    return false;
  if (::llvm::isa<::mlir::UnrealizedConversionCastOp,
                  ::dataflow::ConstantOp>(op))
    return true;
  // The terminator (scf.yield) is always pure but is special-cased by
  // the caller (it is not lifted; its operands are routed to muxes).
  if (op->hasTrait<::mlir::OpTrait::IsTerminator>())
    return true;
  // MemoryEffectOpInterface: prefer the interface answer when present.
  if (auto eff = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op))
    return eff.hasNoEffect();
  // Fall back to the Pure trait. Ops without an effect interface and
  // without the Pure trait are conservatively treated as non-pure.
  return op->hasTrait<::mlir::OpTrait::IsTerminator>() ? true
                                                       : ::mlir::isPure(op);
}

// True iff every op in `region`'s only block (excluding the terminator)
// is liftable-pure. An empty region trivially passes.
bool regionBodyIsPure(::mlir::Region &region) {
  if (region.empty())
    return true;
  ::mlir::Block &block = region.front();
  for (::mlir::Operation &op : block.without_terminator()) {
    if (!isLiftablePure(&op))
      return false;
  }
  return true;
}

bool isConditionalLoadHelper(::mlir::Operation *op) {
  if (::llvm::isa<::mlir::scf::ForOp, ::mlir::scf::ForallOp,
                  ::mlir::scf::WhileOp, ::mlir::scf::IfOp,
                  ::mlir::scf::ParallelOp, ::mlir::scf::ExecuteRegionOp>(op))
    return false;
  if (::llvm::isa<::mlir::CallOpInterface, ::dataflow::StoreOp,
                  ::mlir::LLVM::StoreOp, ::mlir::LLVM::LoadOp>(op))
    return false;
  if (::llvm::isa<::dataflow::LoadOp, ::mlir::UnrealizedConversionCastOp,
                  ::dataflow::ConstantOp>(op))
    return true;
  return ::mlir::isPure(op);
}

bool valueDependsOnLoad(
    ::mlir::Value value, ::mlir::Region &thenRegion,
    const ::llvm::SmallPtrSetImpl<::mlir::Operation *> &loads,
    ::llvm::SmallPtrSetImpl<::mlir::Value> &seen) {
  if (!value || !seen.insert(value).second)
    return false;
  ::mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (loads.contains(def))
    return true;
  if (!thenRegion.isAncestor(def->getParentRegion()))
    return false;
  for (::mlir::Value operand : def->getOperands()) {
    if (valueDependsOnLoad(operand, thenRegion, loads, seen))
      return true;
  }
  return false;
}

::mlir::Value stripSingleInputCasts(::mlir::Value value) {
  for (;;) {
    auto cast = value.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    if (!cast || cast.getInputs().size() != 1)
      return value;
    value = cast.getInputs().front();
  }
}

bool isLoopCarriedElseValue(::mlir::Value value, ::mlir::scf::IfOp ifOp) {
  value = stripSingleInputCasts(value);
  if (value.getDefiningOp<::dataflow::CarryOp>())
    return true;

  auto arg = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
  if (!arg)
    return false;
  auto parentLoop = ifOp->getParentOfType<::mlir::scf::ForOp>();
  return parentLoop && arg.getOwner() == parentLoop.getBody() &&
         arg.getArgNumber() != 0;
}

struct ConditionalLoadIfMatch {
  ::llvm::SmallVector<::dataflow::LoadOp, 4> loads;
  ::llvm::SmallVector<::mlir::Value, 4> thenValues;
  ::llvm::SmallVector<::mlir::Value, 4> elseValues;
};

bool matchConditionalLoadIf(::mlir::scf::IfOp ifOp,
                            ConditionalLoadIfMatch &match) {
  if (ifOp.getNumResults() == 0 || ifOp.getThenRegion().empty() ||
      ifOp.getElseRegion().empty())
    return false;
  auto *thenBlock = ifOp.thenBlock();
  auto *elseBlock = ifOp.elseBlock();
  if (!thenBlock || !elseBlock)
    return false;
  auto thenYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      thenBlock->getTerminator());
  auto elseYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      elseBlock->getTerminator());
  if (!thenYield || !elseYield)
    return false;
  if (thenYield.getNumOperands() != ifOp.getNumResults() ||
      elseYield.getNumOperands() != ifOp.getNumResults())
    return false;
  if (!elseBlock->without_terminator().empty())
    return false;

  ::llvm::SmallPtrSet<::mlir::Operation *, 4> loadOps;
  for (::mlir::Operation &op : thenBlock->without_terminator()) {
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op)) {
      match.loads.push_back(load);
      loadOps.insert(&op);
      continue;
    }
    if (!isConditionalLoadHelper(&op))
      return false;
  }
  if (match.loads.empty())
    return false;

  for (::mlir::Value value : thenYield.getOperands()) {
    if (!isGateFriendly(value.getType()))
      return false;
    ::llvm::SmallPtrSet<::mlir::Value, 16> seen;
    if (!valueDependsOnLoad(value, ifOp.getThenRegion(), loadOps, seen))
      return false;
    match.thenValues.push_back(value);
  }
  for (::mlir::Value value : elseYield.getOperands()) {
    if (!isGateFriendly(value.getType()))
      return false;
    if (!isLoopCarriedElseValue(value, ifOp))
      return false;
    match.elseValues.push_back(value);
  }
  return true;
}

bool areSameMemoryHandle(::mlir::Value lhs, ::mlir::Value rhs) {
  return lhs == rhs || stripSingleInputCasts(lhs) == stripSingleInputCasts(rhs);
}

::mlir::Attribute getConstantAddressAttr(::mlir::Value value) {
  if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>())
    return constant.getConstValue();
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>())
    return constant.getValue();
  return {};
}

bool areSameStoreAddress(::mlir::Value lhs, ::mlir::Value rhs) {
  if (lhs == rhs)
    return true;
  ::mlir::Attribute lhsAttr = getConstantAddressAttr(lhs);
  ::mlir::Attribute rhsAttr = getConstantAddressAttr(rhs);
  if (!lhsAttr || !rhsAttr)
    return false;
  return lhs.getType() == rhs.getType() && lhsAttr == rhsAttr;
}

bool elseRegionIsEmpty(::mlir::scf::IfOp ifOp) {
  if (ifOp.getElseRegion().empty())
    return true;
  for (::mlir::Operation &op : ifOp.getElseRegion().front()) {
    if (!::llvm::isa<::mlir::scf::YieldOp>(op))
      return false;
  }
  auto yield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      ifOp.getElseRegion().front().getTerminator());
  return !yield || yield.getNumOperands() == 0;
}

struct ConditionalStoreIfMatch {
  ::dataflow::StoreOp store;
  ::dataflow::LoadOp load;
  ::mlir::Value replacement;
  ::mlir::Value preserved;
};

struct ConditionalStoreResultIfMatch {
  ::dataflow::StoreOp store;
  bool storeInThen = false;
  ::llvm::SmallVector<::mlir::Value, 4> thenValues;
  ::llvm::SmallVector<::mlir::Value, 4> elseValues;
};

bool matchConditionalStoreIf(::mlir::scf::IfOp ifOp,
                             ConditionalStoreIfMatch &match) {
  if (ifOp.getNumResults() != 0 || ifOp.getThenRegion().empty() ||
      !elseRegionIsEmpty(ifOp))
    return false;

  auto cmp = ifOp.getCondition().getDefiningOp<::mlir::arith::CmpIOp>();
  if (!cmp)
    return false;

  auto *thenBlock = ifOp.thenBlock();
  if (!thenBlock)
    return false;
  auto yield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      thenBlock->getTerminator());
  if (yield && yield.getNumOperands() != 0)
    return false;

  ::dataflow::StoreOp store;
  for (::mlir::Operation &op : *thenBlock) {
    if (::llvm::isa<::mlir::scf::YieldOp>(op))
      continue;
    if (auto candidate = ::llvm::dyn_cast<::dataflow::StoreOp>(op)) {
      if (store)
        return false;
      store = candidate;
      continue;
    }
    if (!isLiftablePure(&op))
      return false;
  }
  if (!store)
    return false;

  ::mlir::Value replacement = store.getData();
  ::mlir::Value preserved;
  if (replacement == cmp.getRhs())
    preserved = cmp.getLhs();
  else if (replacement == cmp.getLhs())
    preserved = cmp.getRhs();
  else
    return false;
  if (replacement.getType() != preserved.getType())
    return false;

  auto load = preserved.getDefiningOp<::dataflow::LoadOp>();
  if (!load)
    return false;
  bool addressPrechecked =
      ifOp->hasAttr(kConditionalStoreSameAddressAttr);
  bool addressMatches =
      areSameMemoryHandle(load.getMem(), store.getMem()) &&
      areSameStoreAddress(load.getAddr(), store.getAddr());
  if (!addressPrechecked && !addressMatches)
    return false;

  match.store = store;
  match.load = load;
  match.replacement = replacement;
  match.preserved = preserved;
  return true;
}

bool collectConditionalStoreResultBranch(::mlir::Block *block,
                                         ::dataflow::StoreOp &store) {
  if (!block)
    return false;
  store = {};
  for (::mlir::Operation &op : block->without_terminator()) {
    if (auto candidate = ::llvm::dyn_cast<::dataflow::StoreOp>(op)) {
      if (store)
        return false;
      store = candidate;
      continue;
    }
    if (!isLiftablePure(&op))
      return false;
  }
  return true;
}

bool matchConditionalStoreResultIf(::mlir::scf::IfOp ifOp,
                                   ConditionalStoreResultIfMatch &match) {
  if (ifOp.getNumResults() == 0 || ifOp.getThenRegion().empty() ||
      ifOp.getElseRegion().empty())
    return false;

  auto *thenBlock = ifOp.thenBlock();
  auto *elseBlock = ifOp.elseBlock();
  if (!thenBlock || !elseBlock)
    return false;
  auto thenYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      thenBlock->getTerminator());
  auto elseYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      elseBlock->getTerminator());
  if (!thenYield || !elseYield)
    return false;
  if (thenYield.getNumOperands() != ifOp.getNumResults() ||
      elseYield.getNumOperands() != ifOp.getNumResults())
    return false;

  ::dataflow::StoreOp thenStore;
  ::dataflow::StoreOp elseStore;
  if (!collectConditionalStoreResultBranch(thenBlock, thenStore) ||
      !collectConditionalStoreResultBranch(elseBlock, elseStore))
    return false;
  if (static_cast<bool>(thenStore) == static_cast<bool>(elseStore))
    return false;

  for (::mlir::Value value : thenYield.getOperands()) {
    if (!isGateFriendly(value.getType()))
      return false;
    match.thenValues.push_back(value);
  }
  for (::mlir::Value value : elseYield.getOperands()) {
    if (!isGateFriendly(value.getType()))
      return false;
    match.elseValues.push_back(value);
  }

  match.store = thenStore ? thenStore : elseStore;
  match.storeInThen = static_cast<bool>(thenStore);
  return true;
}

// Rewrite one mux-shaped scf.if. Caller guarantees:
//   * `ifOp` has at least one result;
//   * both then- and else-regions are present and have a yield with N
//     operands that match the N if-results;
//   * both region bodies are pure (regionBodyIsPure(both)).
// Lifts every body op of both regions into the parent block (then
// before else, original program order preserved within each region),
// emits one dataflow.mux per result, replaces uses, and erases the
// scf.if.
void rewriteMuxIf(::mlir::scf::IfOp ifOp, ::mlir::OpBuilder &builder) {
  ::mlir::Block *parentBlock = ifOp->getBlock();
  ::mlir::Block::iterator insertPt(ifOp.getOperation());

  // Snapshot yield operands before we move ops out (moving keeps SSA
  // values intact, but we want the operand list pinned in case the
  // yields share an operand defined inside a region that gets moved).
  auto thenYield = ::llvm::cast<::mlir::scf::YieldOp>(
      ifOp.thenBlock()->getTerminator());
  auto elseYield = ::llvm::cast<::mlir::scf::YieldOp>(
      ifOp.elseBlock()->getTerminator());
  ::llvm::SmallVector<::mlir::Value, 4> thenVals(thenYield.getOperands());
  ::llvm::SmallVector<::mlir::Value, 4> elseVals(elseYield.getOperands());

  // Lift body ops out of the regions, in original program order. Then
  // ops first, then else ops; this keeps the printed IR ordering
  // predictable. Graph regions are non-SSA so the relative order of
  // the two sets does not affect program semantics.
  auto liftRegion = [&](::mlir::Region &region) {
    if (region.empty())
      return;
    ::mlir::Block &block = region.front();
    for (auto it = block.begin(); it != block.end();) {
      ::mlir::Operation &op = *it;
      ++it;
      if (::llvm::isa<::mlir::scf::YieldOp>(op))
        continue;
      op.moveBefore(parentBlock, insertPt);
    }
  };
  liftRegion(ifOp.getThenRegion());
  liftRegion(ifOp.getElseRegion());

  // Emit one dataflow.mux per result. The selector type accepted is
  // `AnyTypeOf<[I1, Index]>`; the scf.if cond is i1 by construction.
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(parentBlock, insertPt);
  ::mlir::Value sel = ifOp.getCondition();
  for (size_t i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    ::mlir::Type ty = ifOp.getResult(i).getType();
    // Lane order convention (per spec part-2 section 6): false-lane 0,
    // true-lane 1. The scf.if then-region is the true branch, so
    // inputs[0] must be the else-yield and inputs[1] the then-yield.
    ::llvm::SmallVector<::mlir::Value, 2> inputs{elseVals[i], thenVals[i]};
    auto mux = ::dataflow::MuxOp::create(builder, ifOp.getLoc(), ty, sel,
                                         inputs);
    ifOp.getResult(i).replaceAllUsesWith(mux.getOutput());
  }

  ifOp.erase();
}

bool rewriteConditionalLoadIf(::mlir::scf::IfOp ifOp,
                              ::mlir::OpBuilder &builder) {
  ConditionalLoadIfMatch match;
  if (!matchConditionalLoadIf(ifOp, match))
    return false;

  ::mlir::Block *parentBlock = ifOp->getBlock();
  ::mlir::Block::iterator insertPt(ifOp.getOperation());
  ::mlir::Value cond = ifOp.getCondition();

  ::llvm::SmallVector<::mlir::Operation *, 8> lifted;
  ::mlir::Block &thenBlock = ifOp.getThenRegion().front();
  for (auto it = thenBlock.begin(); it != thenBlock.end();) {
    ::mlir::Operation &op = *it;
    ++it;
    if (::llvm::isa<::mlir::scf::YieldOp>(op))
      continue;
    op.moveBefore(parentBlock, insertPt);
    lifted.push_back(&op);
  }

  for (::mlir::Operation *op : lifted) {
    auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op);
    if (!load)
      continue;
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(load);
    ::mlir::TypedAttr zeroAttr =
        getIntegerLikeAttr(builder, load.getAddr().getType(), 0);
    if (!zeroAttr)
      return false;
    ::mlir::Value fallbackAddr = ::dataflow::ConstantOp::create(
                                     builder, load.getLoc(),
                                     load.getAddr().getType(), load.getCtrl(),
                                     zeroAttr)
                                     .getValue();
    auto safeAddr = ::mlir::arith::SelectOp::create(
        builder, load.getLoc(), cond, load.getAddr(), fallbackAddr);
    load->setOperand(1, safeAddr.getResult());
  }

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(parentBlock, insertPt);
  for (size_t i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    ::mlir::Type ty = ifOp.getResult(i).getType();
    auto elseDemux = ::dataflow::DemuxOp::create(
        builder, ifOp.getLoc(), ::mlir::TypeRange{ty, ty}, cond,
        match.elseValues[i]);
    auto thenDemux = ::dataflow::DemuxOp::create(
        builder, ifOp.getLoc(), ::mlir::TypeRange{ty, ty}, cond,
        match.thenValues[i]);
    ::llvm::SmallVector<::mlir::Value, 2> inputs{elseDemux.getOutputs()[0],
                                                 thenDemux.getOutputs()[1]};
    auto mux =
        ::dataflow::MuxOp::create(builder, ifOp.getLoc(), ty, cond, inputs);
    ifOp.getResult(i).replaceAllUsesWith(mux.getOutput());
  }

  ifOp.erase();
  return true;
}

// Rewrite one gate-shaped scf.if (no results, then-only, then-region
// is pure). Caller guarantees these preconditions.
//
// Strategy: lift every then-region op into the parent block, then for
// each lifted op whose first result is gate-friendly, wrap that result
// with `dataflow.gate %cond, %res` and replace its downstream uses
// (other than the gate op itself) with the gate's `after_value`.
//
// Returns true if at least one gate was emitted (so the corpus
// counters tick up); false if the then-region had no gate-friendly
// results -- in that case the scf.if is still erased but no gate is
// added.
bool rewriteGateIf(::mlir::scf::IfOp ifOp, ::mlir::OpBuilder &builder) {
  ::mlir::Block *parentBlock = ifOp->getBlock();
  ::mlir::Block::iterator insertPt(ifOp.getOperation());

  ::mlir::Value cond = ifOp.getCondition();

  // Collect lifted ops in their original order so we can wrap their
  // results after lifting; iterating the parent block by iterator
  // afterward is more brittle (other ops may also belong to it).
  ::llvm::SmallVector<::mlir::Operation *, 8> lifted;
  if (!ifOp.getThenRegion().empty()) {
    ::mlir::Block &block = ifOp.getThenRegion().front();
    for (auto it = block.begin(); it != block.end();) {
      ::mlir::Operation &op = *it;
      ++it;
      if (::llvm::isa<::mlir::scf::YieldOp>(op))
        continue;
      op.moveBefore(parentBlock, insertPt);
      lifted.push_back(&op);
    }
  }

  bool emittedGate = false;
  for (::mlir::Operation *op : lifted) {
    if (op->getNumResults() == 0)
      continue;
    ::mlir::Value v = op->getResult(0);
    if (!isGateFriendly(v.getType()))
      continue;
    // Wrap the result with a gate. Insertion point is right after the
    // op so the textual ordering reads as "compute then gate".
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(op);
    auto gate = ::dataflow::GateOp::create(
        builder, op->getLoc(), builder.getI1Type(), v.getType(), cond, v);
    ::mlir::Value afterValue = gate.getAfterValue();
    // Replace every other user of `v` with the gate's after_value.
    // The gate op itself must keep reading the raw value `v`.
    v.replaceUsesWithIf(afterValue, [&](::mlir::OpOperand &use) {
      return use.getOwner() != gate.getOperation();
    });
    emittedGate = true;
  }

  ifOp.erase();
  return emittedGate;
}

bool rewriteConditionalStoreIf(::mlir::scf::IfOp ifOp,
                               ::mlir::OpBuilder &builder) {
  ConditionalStoreIfMatch match;
  if (!matchConditionalStoreIf(ifOp, match))
    return false;

  ::mlir::Block *parentBlock = ifOp->getBlock();
  ::mlir::Block::iterator insertPt(ifOp.getOperation());
  ::mlir::Block &thenBlock = ifOp.getThenRegion().front();
  for (auto it = thenBlock.begin(); it != thenBlock.end();) {
    ::mlir::Operation &op = *it;
    ++it;
    if (::llvm::isa<::mlir::scf::YieldOp>(op) ||
        &op == match.store.getOperation())
      continue;
    op.moveBefore(parentBlock, insertPt);
  }

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(ifOp);
  auto selected = ::mlir::arith::SelectOp::create(
      builder, ifOp.getLoc(), ifOp.getCondition(), match.replacement,
      match.preserved);
  auto newStore = ::dataflow::StoreOp::create(
      builder, match.store.getLoc(), builder.getNoneType(),
      match.store.getMem(), match.store.getAddr(), selected.getResult(),
      match.store.getCtrl());
  match.store.getDone().replaceAllUsesWith(newStore.getDone());
  ifOp.erase();
  return true;
}

::mlir::Value routeConditionalLane(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Value cond,
                                   ::mlir::Value value, unsigned lane) {
  auto demux = ::dataflow::DemuxOp::create(
      builder, loc, ::mlir::TypeRange{value.getType(), value.getType()}, cond,
      value);
  return demux.getOutputs()[lane];
}

::dataflow::DemuxOp createConditionalLaneDemux(::mlir::OpBuilder &builder,
                                               ::mlir::Location loc,
                                               ::mlir::Value cond,
                                               ::mlir::Value value) {
  return ::dataflow::DemuxOp::create(
      builder, loc, ::mlir::TypeRange{value.getType(), value.getType()}, cond,
      value);
}

::mlir::Value simplifyStoreAddress(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc,
                                   ::mlir::Value addr) {
  auto cast = addr.getDefiningOp<::mlir::arith::IndexCastOp>();
  if (!cast)
    return addr;
  auto zext = cast.getIn().getDefiningOp<::mlir::LLVM::ZExtOp>();
  if (!zext)
    return addr;
  auto sourceType =
      ::llvm::dyn_cast<::mlir::IntegerType>(zext.getArg().getType());
  if (!sourceType || sourceType.getWidth() != ::loom::getIndexWidth())
    return addr;
  return ::mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                            zext.getArg());
}

bool rewriteConditionalStoreResultIf(::mlir::scf::IfOp ifOp,
                                     ::mlir::OpBuilder &builder) {
  ConditionalStoreResultIfMatch match;
  if (!matchConditionalStoreResultIf(ifOp, match))
    return false;

  ::mlir::Block *parentBlock = ifOp->getBlock();
  ::mlir::Block::iterator insertPt(ifOp.getOperation());
  ::mlir::Value cond = ifOp.getCondition();

  auto liftRegion = [&](::mlir::Region &region) {
    if (region.empty())
      return;
    ::mlir::Block &block = region.front();
    for (auto it = block.begin(); it != block.end();) {
      ::mlir::Operation &op = *it;
      ++it;
      if (::llvm::isa<::mlir::scf::YieldOp>(op))
        continue;
      op.moveBefore(parentBlock, insertPt);
    }
  };
  liftRegion(ifOp.getThenRegion());
  liftRegion(ifOp.getElseRegion());

  unsigned storeLane = match.storeInThen ? 1 : 0;
  ::dataflow::DemuxOp ctrlDemux;
  {
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(match.store);
    ::mlir::Value storeAddr =
        simplifyStoreAddress(builder, match.store.getLoc(),
                             match.store.getAddr());
    match.store->setOperand(
        1, routeConditionalLane(builder, match.store.getLoc(), cond, storeAddr,
                                storeLane));
    match.store->setOperand(
        2, routeConditionalLane(builder, match.store.getLoc(), cond,
                                match.store.getData(), storeLane));
    ctrlDemux = createConditionalLaneDemux(builder, match.store.getLoc(), cond,
                                           match.store.getCtrl());
    match.store->setOperand(3, ctrlDemux.getOutputs()[storeLane]);
  }

  {
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(match.store);
    ::mlir::Value falseDone;
    ::mlir::Value trueDone;
    if (match.storeInThen) {
      falseDone = ctrlDemux.getOutputs()[0];
      trueDone = match.store.getDone();
    } else {
      falseDone = match.store.getDone();
      trueDone = ctrlDemux.getOutputs()[1];
    }
    auto mergedDone = ::dataflow::MuxOp::create(
        builder, match.store.getLoc(), builder.getNoneType(), cond,
        ::llvm::SmallVector<::mlir::Value, 2>{falseDone, trueDone});
    mergedDone->setAttr(kConditionalStoreDoneAttr, builder.getUnitAttr());
  }

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(parentBlock, insertPt);
  for (size_t i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    auto selected = ::mlir::arith::SelectOp::create(
        builder, ifOp.getLoc(), cond, match.thenValues[i],
        match.elseValues[i]);
    ifOp.getResult(i).replaceAllUsesWith(selected.getResult());
  }

  ifOp.erase();
  return true;
}

// Wrap each result of a side-effecting `scf.if %cond -> (Ti...)` with a
// `dataflow.gate %cond, %ifResult` so the scf.if envelope is preserved
// (mandatory for safety: lifting the body would unconditionally execute
// the side effects), but downstream consumers see a body-phase gate
// rather than a raw scf.if result. The gate's `after_cond` is unused
// here -- a future iteration may chain it into nested control routing
// for predicated dataflow consumers; for now we discard it cleanly.
//
// Caller guarantees: `ifOp` has at least one result and at least one
// gate-friendly result type. Returns true if at least one gate was
// emitted, false otherwise (in which case the scf.if is left alone).
bool rewriteSideEffectIf(::mlir::scf::IfOp ifOp,
                         ::mlir::OpBuilder &builder) {
  ::mlir::Value cond = ifOp.getCondition();
  bool emittedAny = false;
  // Insert the gates immediately after the scf.if, in result order, so
  // the rewritten IR reads "compute scf.if; gate result 0; gate result
  // 1; ...; downstream uses".
  for (size_t i = 0, e = ifOp.getNumResults(); i < e; ++i) {
    ::mlir::Value r = ifOp.getResult(i);
    if (!isGateFriendly(r.getType()))
      continue;
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(ifOp.getOperation());
    auto gate = ::dataflow::GateOp::create(
        builder, ifOp.getLoc(), builder.getI1Type(), r.getType(), cond, r);
    ::mlir::Value afterValue = gate.getAfterValue();
    // Replace every consumer of `r` with the gate's `after_value`,
    // except for the gate op itself (which must keep reading the raw
    // scf.if result).
    r.replaceUsesWithIf(afterValue, [&](::mlir::OpOperand &use) {
      return use.getOwner() != gate.getOperation();
    });
    emittedAny = true;
  }
  return emittedAny;
}

// Classify and dispatch one scf.if. Returns:
//   * 0 if the op was left alone (a remark may be emitted by caller).
//   * 1 if the op was rewritten as a mux.
//   * 2 if the op was rewritten as a gate (then-only no-result lift).
//   * 3 if the op was a gate-shape but produced no gates (then-region
//     had only non-gate-friendly results).
//   * 4 if the op was wrapped with side-effect-aware result gates
//     (scf.if envelope preserved, each gate-friendly result wrapped in
//     a `dataflow.gate %cond, %ifResult`).
unsigned rewriteOneIf(::mlir::scf::IfOp ifOp, ::mlir::OpBuilder &builder) {
  bool hasResults = ifOp.getNumResults() > 0;
  bool hasThen = !ifOp.getThenRegion().empty() &&
                 !ifOp.getThenRegion().front().empty();
  bool hasElse = !ifOp.getElseRegion().empty() &&
                 !ifOp.getElseRegion().front().empty();

  if (hasResults) {
    if (rewriteConditionalLoadIf(ifOp, builder))
      return 6;
    if (rewriteConditionalStoreResultIf(ifOp, builder))
      return 7;
    // Mux case requires both regions and pure bodies. Try the lift
    // first; only if the body has memory effects (or the scf.if is
    // missing one of its regions) do we fall back to the
    // side-effect-aware gate wrapping.
    if (hasThen && hasElse &&
        regionBodyIsPure(ifOp.getThenRegion()) &&
        regionBodyIsPure(ifOp.getElseRegion())) {
      // Sanity: yield operand counts must match the result count.
      // The verifier already guarantees this, but a defensive check
      // guards against malformed inputs we may see in negative-bail
      // tests.
      auto thenYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
          ifOp.thenBlock()->getTerminator());
      auto elseYield = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
          ifOp.elseBlock()->getTerminator());
      if (thenYield && elseYield &&
          thenYield.getNumOperands() == ifOp.getNumResults() &&
          elseYield.getNumOperands() == ifOp.getNumResults()) {
        rewriteMuxIf(ifOp, builder);
        return 1;
      }
    }
    // Side-effect-aware gating: keep the scf.if envelope but tag every
    // gate-friendly result with a body-phase `dataflow.gate`. This
    // lets the corpus exercise gate emission on shapes that the mux
    // lift cannot handle (e.g., scf.if whose body issues a
    // dataflow.store or an llvm.store) without changing observable
    // semantics. If no result is gate-friendly the scf.if is left
    // alone.
    if (rewriteSideEffectIf(ifOp, builder))
      return 4;
    return 0;
  }

  // No results.
  if (hasThen && hasElse) {
    // Two-sided no-result scf.if. The dataflow.gate / .mux primitives
    // do not naturally express two-sided side-effecting alternatives
    // here. Bail.
    return 0;
  }
  if (!hasThen) {
    // Empty body or else-only with no results. Nothing to do.
    return 0;
  }
  if (rewriteConditionalStoreIf(ifOp, builder))
    return 5;
  // Then-only, no results: gate case (only fire when cond is true).
  if (!regionBodyIsPure(ifOp.getThenRegion()))
    return 0;
  return rewriteGateIf(ifOp, builder) ? 2 : 3;
}

::mlir::Value materializeIndexDomainValue(
    ::mlir::Value value, ::mlir::OpBuilder &builder,
    ::llvm::DenseMap<::mlir::Value, ::mlir::Value> &cache) {
  if (::llvm::isa<::mlir::IndexType>(value.getType()))
    return value;
  if (!::llvm::isa<::mlir::IntegerType>(value.getType()))
    return {};
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  if (auto cast = value.getDefiningOp<::mlir::arith::IndexCastOp>()) {
    ::mlir::Value input = cast.getIn();
    if (::llvm::isa<::mlir::IndexType>(input.getType())) {
      cache[value] = input;
      return input;
    }
  }

  if (auto add = value.getDefiningOp<::mlir::arith::AddIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(add.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(add.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexAdd =
          ::mlir::arith::AddIOp::create(builder, add.getLoc(), lhs, rhs);
      cache[value] = indexAdd.getResult();
      return indexAdd.getResult();
    }
  }

  if (auto sub = value.getDefiningOp<::mlir::arith::SubIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(sub.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(sub.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexSub =
          ::mlir::arith::SubIOp::create(builder, sub.getLoc(), lhs, rhs);
      cache[value] = indexSub.getResult();
      return indexSub.getResult();
    }
  }

  if (auto mul = value.getDefiningOp<::mlir::arith::MulIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(mul.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(mul.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexMul =
          ::mlir::arith::MulIOp::create(builder, mul.getLoc(), lhs, rhs);
      cache[value] = indexMul.getResult();
      return indexMul.getResult();
    }
  }

  if (auto shl = value.getDefiningOp<::mlir::arith::ShLIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(shl.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(shl.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexShl =
          ::mlir::arith::ShLIOp::create(builder, shl.getLoc(), lhs, rhs);
      cache[value] = indexShl.getResult();
      return indexShl.getResult();
    }
  }

  if (auto shr = value.getDefiningOp<::mlir::arith::ShRUIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(shr.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(shr.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexShr =
          ::mlir::arith::ShRUIOp::create(builder, shr.getLoc(), lhs, rhs);
      cache[value] = indexShr.getResult();
      return indexShr.getResult();
    }
  }

  if (auto andi = value.getDefiningOp<::mlir::arith::AndIOp>()) {
    ::mlir::Value lhs =
        materializeIndexDomainValue(andi.getLhs(), builder, cache);
    ::mlir::Value rhs =
        materializeIndexDomainValue(andi.getRhs(), builder, cache);
    if (lhs && rhs) {
      auto indexAnd =
          ::mlir::arith::AndIOp::create(builder, andi.getLoc(), lhs, rhs);
      cache[value] = indexAnd.getResult();
      return indexAnd.getResult();
    }
  }

  if (auto zext = value.getDefiningOp<::mlir::LLVM::ZExtOp>()) {
    auto sourceType =
        ::llvm::dyn_cast<::mlir::IntegerType>(zext.getArg().getType());
    auto resultType =
        ::llvm::dyn_cast<::mlir::IntegerType>(zext.getResult().getType());
    if (sourceType && resultType &&
        sourceType.getWidth() == ::loom::getIndexWidth() &&
        resultType.getWidth() >= sourceType.getWidth()) {
      ::mlir::Value input =
          materializeIndexDomainValue(zext.getArg(), builder, cache);
      if (input) {
        cache[value] = input;
        return input;
      }
    }
  }

  if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>()) {
    auto typed = ::llvm::dyn_cast<::mlir::TypedAttr>(constant.getConstValue());
    auto integerAttr = typed ? ::llvm::dyn_cast<::mlir::IntegerAttr>(typed)
                             : ::mlir::IntegerAttr{};
    if (integerAttr) {
      ::mlir::TypedAttr indexAttr = getIntegerLikeAttr(
          builder, builder.getIndexType(), integerAttr.getInt());
      auto indexConstant = ::dataflow::ConstantOp::create(
          builder, constant.getLoc(), builder.getIndexType(),
          constant.getCtrl(), ::mlir::cast<::mlir::Attribute>(indexAttr));
      cache[value] = indexConstant.getValue();
      return indexConstant.getValue();
    }
  }

  if (auto invariant = value.getDefiningOp<::dataflow::InvariantOp>()) {
    ::mlir::Value init =
        materializeIndexDomainValue(invariant.getInit(), builder, cache);
    if (init) {
      auto indexInvariant = ::dataflow::InvariantOp::create(
          builder, invariant.getLoc(), builder.getIndexType(),
          invariant.getCond(), init);
      cache[value] = indexInvariant.getOutput();
      return indexInvariant.getOutput();
    }
  }

  auto indexCast = ::mlir::arith::IndexCastOp::create(
      builder, value.getLoc(), builder.getIndexType(), value);
  cache[value] = indexCast.getResult();
  return indexCast.getResult();
}

bool isDataflowMemoryAddressUse(::mlir::OpOperand &use) {
  ::mlir::Operation *owner = use.getOwner();
  return (::llvm::isa<::dataflow::LoadOp, ::dataflow::StoreOp>(owner)) &&
         use.getOperandNumber() == 1;
}

bool valueFeedsOnlyMemoryAddress(::mlir::Value value,
                                 ::llvm::SmallPtrSetImpl<::mlir::Value> &seen) {
  if (value.use_empty())
    return false;
  if (!seen.insert(value).second)
    return false;

  bool sawAddress = false;
  for (::mlir::OpOperand &use : value.getUses()) {
    if (isDataflowMemoryAddressUse(use)) {
      sawAddress = true;
      continue;
    }
    auto select = ::llvm::dyn_cast<::mlir::arith::SelectOp>(use.getOwner());
    if (select && use.getOperandNumber() != 0 &&
        valueFeedsOnlyMemoryAddress(select.getResult(), seen)) {
      sawAddress = true;
      continue;
    }
    return false;
  }
  return sawAddress;
}

bool valueFeedsOnlyMemoryAddress(::mlir::Value value) {
  ::llvm::SmallPtrSet<::mlir::Value, 8> seen;
  return valueFeedsOnlyMemoryAddress(value, seen);
}

bool isMemoryAddressIndexCast(::mlir::arith::IndexCastOp cast) {
  if (!::llvm::isa<::mlir::IndexType>(cast.getType()) ||
      !::llvm::isa<::mlir::IntegerType>(cast.getIn().getType()))
    return false;
  return valueFeedsOnlyMemoryAddress(cast.getResult());
}

bool allUsesAreOperation(::mlir::Value value, ::mlir::Operation *op) {
  for (::mlir::OpOperand &use : value.getUses()) {
    if (use.getOwner() != op)
      return false;
  }
  return true;
}

bool collectIndexDomainUses(
    ::mlir::Value value, ::mlir::Operation *cycleUser,
    ::llvm::SmallVectorImpl<::mlir::arith::IndexCastOp> &addressCasts,
    ::llvm::SmallVectorImpl<std::pair<::dataflow::GraphReturnOp, unsigned>>
        &returnUses) {
  for (::mlir::OpOperand &use : value.getUses()) {
    ::mlir::Operation *owner = use.getOwner();
    if (owner == cycleUser)
      continue;
    if (auto cast = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(owner)) {
      if (!isMemoryAddressIndexCast(cast))
        return false;
      addressCasts.push_back(cast);
      continue;
    }
    if (auto ret = ::llvm::dyn_cast<::dataflow::GraphReturnOp>(owner)) {
      returnUses.push_back({ret, use.getOperandNumber()});
      continue;
    }
    return false;
  }
  return true;
}

void collectDirectMemoryAddressCasts(
    ::mlir::Value value,
    ::llvm::SmallVectorImpl<::mlir::arith::IndexCastOp> &addressCasts) {
  for (::mlir::OpOperand &use : value.getUses()) {
    auto cast = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(use.getOwner());
    if (cast && isMemoryAddressIndexCast(cast))
      addressCasts.push_back(cast);
  }
}

bool isPredicateControlUse(::mlir::OpOperand &use) {
  ::mlir::Operation *owner = use.getOwner();
  ::llvm::StringRef name = owner->getName().getStringRef();
  unsigned operandNumber = use.getOperandNumber();
  if (name == "arith.select" || name == "dataflow.mux" ||
      name == "dataflow.demux" || name == "dataflow.gate")
    return operandNumber == 0;
  if (name == "dataflow.carry" || name == "dataflow.invariant")
    return operandNumber == 0;
  return false;
}

bool valueFeedsOnlyPredicateControls(::mlir::Value value) {
  if (value.use_empty())
    return false;
  for (::mlir::OpOperand &use : value.getUses())
    if (!isPredicateControlUse(use))
      return false;
  return true;
}

bool rewriteOneIndexDomainCmp(::mlir::arith::CmpIOp cmp,
                              ::mlir::OpBuilder &builder) {
  if (!::llvm::isa<::mlir::IntegerType>(cmp.getLhs().getType()) ||
      !::llvm::isa<::mlir::IntegerType>(cmp.getRhs().getType()) ||
      !valueFeedsOnlyPredicateControls(cmp.getResult()))
    return false;

  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 4> lhsAddressCasts;
  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 4> rhsAddressCasts;
  collectDirectMemoryAddressCasts(cmp.getLhs(), lhsAddressCasts);
  collectDirectMemoryAddressCasts(cmp.getRhs(), rhsAddressCasts);
  if (lhsAddressCasts.empty() && rhsAddressCasts.empty())
    return false;

  ::mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(cmp);
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> cache;
  ::mlir::Value indexLhs =
      materializeIndexDomainValue(cmp.getLhs(), builder, cache);
  ::mlir::Value indexRhs =
      materializeIndexDomainValue(cmp.getRhs(), builder, cache);
  if (!indexLhs || !indexRhs)
    return false;

  auto indexCmp = ::mlir::arith::CmpIOp::create(
      builder, cmp.getLoc(), cmp.getPredicate(), indexLhs, indexRhs);
  cmp.getResult().replaceAllUsesWith(indexCmp.getResult());
  auto replaceAddressCasts =
      [](::mlir::Value replacement,
         ::llvm::ArrayRef<::mlir::arith::IndexCastOp> casts) {
        for (::mlir::arith::IndexCastOp cast : casts) {
          if (!cast.getOperation()->getBlock())
            continue;
          if (::mlir::Operation *def = replacement.getDefiningOp()) {
            if (def->getBlock() != cast->getBlock())
              continue;
            if (cast->isBeforeInBlock(def))
              continue;
          }
          cast.replaceAllUsesWith(replacement);
          cast.erase();
        }
      };
  replaceAddressCasts(indexLhs, lhsAddressCasts);
  replaceAddressCasts(indexRhs, rhsAddressCasts);
  cmp.erase();
  return true;
}

bool rewriteIndexDomainCmps(::dataflow::GraphFuncOp graph,
                            ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::arith::CmpIOp, 8> cmps;
  graph.getBody().walk([&](::mlir::arith::CmpIOp cmp) {
    if (::llvm::isa<::mlir::IntegerType>(cmp.getLhs().getType()) &&
        ::llvm::isa<::mlir::IntegerType>(cmp.getRhs().getType()))
      cmps.push_back(cmp);
  });

  bool changed = false;
  for (::mlir::arith::CmpIOp cmp : cmps) {
    if (!cmp.getOperation()->getBlock())
      continue;
    changed |= rewriteOneIndexDomainCmp(cmp, builder);
  }
  return changed;
}

bool eraseInternalIndexDomainCycle(
    ::dataflow::CarryOp carry, ::mlir::arith::AddIOp next,
    ::mlir::Operation *stepOp) {
  ::llvm::SmallPtrSet<::mlir::Operation *, 4> internalOps;
  internalOps.insert(carry.getOperation());
  internalOps.insert(next.getOperation());
  if (stepOp)
    internalOps.insert(stepOp);

  for (::mlir::Operation *op : internalOps) {
    for (::mlir::Value result : op->getResults()) {
      for (::mlir::OpOperand &use : result.getUses()) {
        if (!internalOps.contains(use.getOwner()))
          return false;
      }
    }
  }

  ::llvm::SmallVector<::mlir::Operation *, 4> eraseOps;
  eraseOps.push_back(next.getOperation());
  eraseOps.push_back(carry.getOperation());
  if (stepOp)
    eraseOps.push_back(stepOp);
  for (::mlir::Operation *op : eraseOps)
    op->dropAllDefinedValueUses();
  for (::mlir::Operation *op : eraseOps)
    op->dropAllReferences();
  for (::mlir::Operation *op : eraseOps)
    op->erase();
  return true;
}

bool rewriteOneIndexDomainCarry(::dataflow::CarryOp carry,
                                ::mlir::OpBuilder &builder) {
  if (!::llvm::isa<::mlir::IntegerType>(carry.getOutput().getType()))
    return false;

  auto next = carry.getCarry().getDefiningOp<::mlir::arith::AddIOp>();
  if (!next)
    return false;
  ::mlir::Value carryValue = carry.getOutput();
  ::mlir::Value stepValue;
  if (next.getLhs() == carryValue)
    stepValue = next.getRhs();
  else if (next.getRhs() == carryValue)
    stepValue = next.getLhs();
  else
    return false;

  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 4> addressCasts;
  ::llvm::SmallVector<std::pair<::dataflow::GraphReturnOp, unsigned>, 2>
      carryReturnUses;
  if (!collectIndexDomainUses(carryValue, next.getOperation(), addressCasts,
                              carryReturnUses))
    return false;

  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 2> nextAddressCasts;
  ::llvm::SmallVector<std::pair<::dataflow::GraphReturnOp, unsigned>, 2>
      nextReturnUses;
  if (!collectIndexDomainUses(next.getResult(), carry.getOperation(),
                              nextAddressCasts, nextReturnUses))
    return false;

  if (addressCasts.empty() && nextAddressCasts.empty())
    return false;

  ::mlir::Operation *stepOp = stepValue.getDefiningOp();
  if (stepOp && ::llvm::isa<::dataflow::InvariantOp>(stepOp) &&
      !allUsesAreOperation(stepValue, next.getOperation()))
    return false;

  ::mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(carry);
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> cache;
  ::mlir::Value indexInit =
      materializeIndexDomainValue(carry.getInit(), builder, cache);
  ::mlir::Value indexStep =
      materializeIndexDomainValue(stepValue, builder, cache);
  if (!indexInit || !indexStep)
    return false;

  auto indexCarry = ::dataflow::CarryOp::create(
      builder, carry.getLoc(), builder.getIndexType(), carry.getCond(),
      indexInit, indexInit);
  auto indexNext = ::mlir::arith::AddIOp::create(
      builder, next.getLoc(), indexCarry.getOutput(), indexStep);
  indexCarry->setOperand(2, indexNext.getResult());

  for (::mlir::arith::IndexCastOp cast : addressCasts) {
    cast.replaceAllUsesWith(indexCarry.getOutput());
    cast.erase();
  }
  for (::mlir::arith::IndexCastOp cast : nextAddressCasts) {
    cast.replaceAllUsesWith(indexNext.getResult());
    cast.erase();
  }

  auto replaceReturn = [&](::dataflow::GraphReturnOp ret, unsigned index,
                           ::mlir::Value replacement,
                           ::mlir::Type originalType) {
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(ret);
    auto castBack = ::mlir::arith::IndexCastOp::create(
        builder, ret.getLoc(), originalType, replacement);
    ret->setOperand(index, castBack.getResult());
  };
  for (auto [ret, index] : carryReturnUses)
    replaceReturn(ret, index, indexCarry.getOutput(), carry.getType());
  for (auto [ret, index] : nextReturnUses)
    replaceReturn(ret, index, indexNext.getResult(), next.getType());

  if (!eraseInternalIndexDomainCycle(carry, next, stepOp))
    return false;
  return true;
}

bool rewriteIndexDomainCarryCycles(::dataflow::GraphFuncOp graph,
                                   ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::dataflow::CarryOp, 8> carries;
  graph.getBody().walk([&](::dataflow::CarryOp carry) {
    if (::llvm::isa<::mlir::IntegerType>(carry.getOutput().getType()))
      carries.push_back(carry);
  });

  bool changed = false;
  for (::dataflow::CarryOp carry : carries) {
    if (!carry.getOperation()->getBlock())
      continue;
    changed |= rewriteOneIndexDomainCarry(carry, builder);
  }
  return changed;
}

bool rewriteAddressIndexCasts(::dataflow::GraphFuncOp graph,
                              ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 8> casts;
  graph.getBody().walk([&](::mlir::arith::IndexCastOp cast) {
    if (::llvm::isa<::mlir::IndexType>(cast.getType()) &&
        ::llvm::isa<::mlir::IntegerType>(cast.getIn().getType()))
      casts.push_back(cast);
  });

  bool changed = false;
  for (::mlir::arith::IndexCastOp cast : casts) {
    if (!cast.getOperation()->getBlock())
      continue;
    ::llvm::DenseMap<::mlir::Value, ::mlir::Value> cache;
    builder.setInsertionPoint(cast);
    ::mlir::Value indexValue =
        materializeIndexDomainValue(cast.getIn(), builder, cache);
    if (!indexValue || indexValue == cast.getResult())
      continue;
    cast.replaceAllUsesWith(indexValue);
    cast.erase();
    changed = true;
  }
  return changed;
}

void eraseDeadIndexArithmetic(::dataflow::GraphFuncOp graph) {
  bool changed = true;
  while (changed) {
    changed = false;
    ::llvm::SmallVector<::mlir::Operation *, 8> deadOps;
    graph.getBody().walk([&](::mlir::Operation *op) {
      if (!op->use_empty())
        return;
      if (::llvm::isa<::mlir::arith::IndexCastOp, ::mlir::arith::AddIOp,
                      ::mlir::arith::SubIOp, ::mlir::arith::MulIOp,
                      ::mlir::arith::ShLIOp,
                      ::mlir::arith::ShRUIOp, ::mlir::arith::AndIOp,
                      ::mlir::LLVM::ZExtOp, ::dataflow::ConstantOp,
                      ::dataflow::InvariantOp>(op))
        deadOps.push_back(op);
    });
    for (::mlir::Operation *op : deadOps) {
      op->erase();
      changed = true;
    }
  }
}

// Walk every scf.if directly or transitively inside `graph` in post
// order. We collect first because the rewrite mutates the IR.
void rewriteOneGraph(::dataflow::GraphFuncOp graph,
                     ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::scf::IfOp, 8> ifs;
  graph.getBody().walk<::mlir::WalkOrder::PostOrder>(
      [&](::mlir::scf::IfOp ifOp) { ifs.push_back(ifOp); });
  for (::mlir::scf::IfOp ifOp : ifs) {
    // Skip ops whose parent has already been erased (transitive
    // rewrites can drop a parent that contained `ifOp` if a future
    // iteration extends the lifting to nest deeper). This is a
    // defensive no-op today since rewriteMuxIf / rewriteGateIf only
    // erase the targeted scf.if, not its outer parent.
    if (!ifOp.getOperation()->getBlock())
      continue;
    unsigned rc = rewriteOneIf(ifOp, builder);
    if (rc == 0) {
      ifOp.emitRemark()
          << "loom-lower-graph-control: scf.if shape not lifted "
             "(effectful body, two-sided no-result, or unmodeled result "
             "shape)";
    }
  }
  bool changed = rewriteIndexDomainCarryCycles(graph, builder);
  changed |= rewriteIndexDomainCmps(graph, builder);
  changed |= rewriteAddressIndexCasts(graph, builder);
  if (changed)
    eraseDeadIndexArithmetic(graph);
}

struct LowerGraphControlPass
    : public ::mlir::PassWrapper<LowerGraphControlPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphControlPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-control";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower scf.if ops inside dataflow.graph.func bodies into "
           "dataflow.gate (no-result then-only or effectful with-result "
           "result-wrap) and dataflow.mux (pure with-result both-regions) "
           "primitives. Two-sided no-result and no-result-with-effects "
           "shapes are left in place.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect,
                    ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    ::llvm::SmallVector<::dataflow::GraphFuncOp, 8> graphs;
    for (auto graph : module.getOps<::dataflow::GraphFuncOp>())
      graphs.push_back(graph);

    for (::dataflow::GraphFuncOp graph : graphs) {
      if (graph.isExternal())
        continue;
      rewriteOneGraph(graph, builder);
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerGraphControlPass() {
  return std::make_unique<LowerGraphControlPass>();
}

void registerLowerGraphControlPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphControlPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
