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
//     d. Anything else (no results and effectful body, both no-result
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
#include "llvm/ADT/SmallVector.h"

namespace {

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
  // Then-only, no results: gate case (only fire when cond is true).
  if (!regionBodyIsPure(ifOp.getThenRegion()))
    return 0;
  return rewriteGateIf(ifOp, builder) ? 2 : 3;
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
