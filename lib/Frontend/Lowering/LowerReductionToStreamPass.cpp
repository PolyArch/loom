// Lower the simple-reduction shape of a `scf.for` with iter_args inside
// a `dataflow.graph.func` body into the streaming primitives
// `dataflow.stream` + `dataflow.carry`. The body's compute ops
// (`llvm.getelementptr`, `llvm.load`, `arith.*`, `llvm.store`) are
// hoisted from the `scf.for` region into the surrounding graph.func
// body and rewritten to use the new index / carry SSA values.
//
// Design choice (smoke baseline -- option B in the task brief):
//   The graph.func today receives raw `!llvm.ptr` operands, and
//   `dataflow.load` / `dataflow.store` require `AnyMemRef`. Rather
//   than introduce a memref boundary cast, this pass only lowers the
//   *control* shape (`scf.for + scf.yield + iter_args` -> `dataflow.stream
//   + dataflow.carry`) and leaves `llvm.load`, `llvm.store`, `llvm.getelementptr`
//   in place inside the graph.func body. The streaming primitives carry
//   the iv as an index stream; the body uses that iv to compute the
//   address through gep + llvm.load just like before. This means
//   memory accesses are not yet tokenized -- a future pass needs to
//   bridge memref / ctrl threading.
//
// TODO(loom-frontend): bridge `!llvm.ptr` graph.func operands to
//   `memref<?xT>` so `dataflow.load` / `dataflow.store` can replace
//   the residual `llvm.load` / `llvm.store`.
// TODO(loom-frontend): tokenize memory ops with `dataflow.constant`
//   ctrl tokens + `dataflow.sync` rendezvous (option C).
// TODO(loom-frontend): handle nested `scf.for` / multiple top-level
//   loops / call ops in the body. The current pass bails out on
//   anything other than the single-loop simple-reduction shape and
//   leaves the offending graph.func untouched (with a remark).
//
// Eligibility (per the task brief):
//   * The graph.func body must contain exactly one top-level
//     `scf.for` op (i.e., one in the entry block of the body, plus
//     the `dataflow.graph.return` terminator).
//   * The loop has at least one iter_arg.
//   * Body terminator is `scf.yield` with N operands matching the
//     N iter_args.
//   * `lb` / `ub` / `step` / iter-arg `init` are loop-invariant
//     (defined outside the loop region).
//   * The body has no nested structured-control-flow op
//     (`scf.for`, `scf.forall`, `scf.while`, `scf.if`,
//     `scf.parallel`, `scf.execute_region`).
//   * The body has no `mlir::CallOpInterface` ops.
// If any of these fails, the loop is left in place and a remark is
// emitted on the graph.func.

#include "Frontend/Lowering/Passes.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// Return true if `op` is a structured-control-flow nesting we cannot
// yet handle inside the graph.func reduction body.
bool isNestedStructuredControl(::mlir::Operation *op) {
  return ::llvm::isa<::mlir::scf::ForOp, ::mlir::scf::ForallOp,
                     ::mlir::scf::WhileOp, ::mlir::scf::IfOp,
                     ::mlir::scf::ParallelOp,
                     ::mlir::scf::ExecuteRegionOp>(op);
}

// Eligibility check: walk the body of the loop and report whether
// this loop matches the simple-reduction shape we lower. The first
// disqualifier short-circuits the walk.
bool isEligibleReduction(::mlir::scf::ForOp loop) {
  // Has at least one iter_arg.
  if (loop.getInitArgs().empty())
    return false;

  // lb / ub / step / inits must be loop-invariant (defined outside
  // the loop's region).
  ::mlir::Region &region = loop.getRegion();
  auto definedAbove = [&](::mlir::Value v) {
    if (auto *defOp = v.getDefiningOp())
      return !region.isAncestor(defOp->getParentRegion());
    // Block argument: must belong to a block outside the loop region.
    auto *block = ::llvm::cast<::mlir::BlockArgument>(v).getOwner();
    return !region.isAncestor(block->getParent());
  };
  if (!definedAbove(loop.getLowerBound()))
    return false;
  if (!definedAbove(loop.getUpperBound()))
    return false;
  if (!definedAbove(loop.getStep()))
    return false;
  for (::mlir::Value v : loop.getInitArgs())
    if (!definedAbove(v))
      return false;

  // dataflow.stream requires `SignlessIntegerLike` (signless integer)
  // bounds; `index`-typed loops are outside the smoke shape today.
  // The cmsis-* corpora are lowered from C int/i32/i64 bounds, so this
  // check only excludes hand-written test inputs.
  if (!loop.getLowerBound().getType().isSignlessInteger())
    return false;

  // Body terminator must be scf.yield with N operands matching N iter_args.
  ::mlir::Block &body = loop.getRegion().front();
  auto yieldOp = ::llvm::dyn_cast_or_null<::mlir::scf::YieldOp>(
      body.getTerminator());
  if (!yieldOp)
    return false;
  if (yieldOp.getNumOperands() != loop.getInitArgs().size())
    return false;

  // No nested structured-control-flow op, no call-op-interface op.
  bool eligible = true;
  loop.getBody()->walk([&](::mlir::Operation *op) {
    if (op == loop.getOperation())
      return ::mlir::WalkResult::advance();
    if (isNestedStructuredControl(op)) {
      eligible = false;
      return ::mlir::WalkResult::interrupt();
    }
    if (::llvm::isa<::mlir::CallOpInterface>(op)) {
      eligible = false;
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  return eligible;
}

// Locate the unique top-level scf.for op inside a graph.func body, or
// return null if there is zero or more than one. The graph.func body
// always has a `dataflow.graph.return` terminator; everything else in
// the entry block is candidate compute. Right now this pass only
// runs when there is exactly one top-level scf.for and zero other
// non-terminator non-scf-for ops between the function entry and the
// loop, because that's the only shape `LowerForToGraphPass` produces.
::mlir::scf::ForOp findSoleTopLevelFor(::dataflow::GraphFuncOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  ::mlir::scf::ForOp found;
  for (::mlir::Operation &op : entry.without_terminator()) {
    if (auto loop = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
      if (found)
        return {}; // More than one top-level scf.for: bail out.
      found = loop;
    }
  }
  return found;
}

// Lower one eligible scf.for inside `graph` into dataflow.stream +
// dataflow.carry plus the body ops moved out into the graph entry
// block. Returns success on rewrite, failure if the loop turned out
// to be ineligible (caller already filtered, but this is cheap).
::mlir::LogicalResult lowerOneReduction(::dataflow::GraphFuncOp graph,
                                        ::mlir::scf::ForOp loop,
                                        ::mlir::OpBuilder &builder) {
  ::mlir::Location loc = loop.getLoc();

  // 1. Materialize dataflow.stream right before the scf.for. The
  //    stream consumes the loop's lb / ub / step and emits an iv
  //    stream of the same signless integer type plus an i1 rwc
  //    stream. Keep the signed step operand and use "+=" for the update;
  //    negative static steps therefore need a descending continuation
  //    predicate.
  builder.setInsertionPoint(loop);
  auto stepOpAttr = builder.getStringAttr("+=");
  auto contCondAttr = ::loom::lowering::inferStreamContCond(builder, loop);
  auto streamOp = ::dataflow::StreamOp::create(
      builder, loc, loop.getLowerBound().getType(),
      builder.getI1Type(), loop.getLowerBound(), loop.getUpperBound(),
      loop.getStep(), stepOpAttr, contCondAttr);

  ::mlir::Value idxVal = streamOp.getIndex();
  ::mlir::Value rwcVal = streamOp.getRwc();

  // 2. Materialize one dataflow.carry per iter_arg. The next-token
  //    feed for each carry is the corresponding scf.yield operand.
  //    The graph.func body is graph-region-shaped (no SSA dominance),
  //    so it is legal to refer to the yield operand here even though
  //    we still leave the loop in place until we erase it at the
  //    end. Note: we set the insertion point of the carry ops back
  //    to before the loop too -- after we erase the loop, the body
  //    ops will be moved here and the carry ops sit at the right
  //    position to be observed by them.
  ::mlir::Block &origBody = loop.getRegion().front();
  auto yieldOp = ::llvm::cast<::mlir::scf::YieldOp>(origBody.getTerminator());
  ::llvm::SmallVector<::mlir::Value, 4> carryVals;
  carryVals.reserve(loop.getInitArgs().size());
  for (size_t i = 0, e = loop.getInitArgs().size(); i < e; ++i) {
    ::mlir::Value initVal = loop.getInitArgs()[i];
    ::mlir::Value carryFeed = yieldOp.getOperand(i);
    auto carryOp = ::dataflow::CarryOp::create(
        builder, loc, initVal.getType(), rwcVal, initVal, carryFeed);
    carryVals.push_back(carryOp.getOutput());
  }

  // 3. Rewrite uses of the loop's induction variable + iter_args
  //    inside the body to use the new SSA values.
  loop.getInductionVar().replaceAllUsesWith(idxVal);
  for (size_t i = 0, e = loop.getInitArgs().size(); i < e; ++i)
    loop.getRegionIterArgs()[i].replaceAllUsesWith(carryVals[i]);

  // 4. Replace each loop result with the matching carry's output.
  for (size_t i = 0, e = loop.getResults().size(); i < e; ++i)
    loop.getResult(i).replaceAllUsesWith(carryVals[i]);

  // 5. Move the body's compute ops (everything except scf.yield) out
  //    of the scf.for region into the graph.func entry block, right
  //    after the carry ops. We splice in original program order to
  //    preserve any remaining SSA dependencies between body ops.
  ::mlir::Block &graphEntry = graph.getBody().front();
  // Insert just before the loop op so the scope of the carry ops
  // (already placed before the loop) dominates the body. Since the
  // graph region is non-SSA, this ordering is for readability only.
  ::mlir::Block::iterator insertPt(loop.getOperation());
  for (auto it = origBody.begin(); it != origBody.end();) {
    ::mlir::Operation &op = *it;
    ++it;
    if (::llvm::isa<::mlir::scf::YieldOp>(op))
      continue;
    op.moveBefore(&graphEntry, insertPt);
  }

  // 6. Erase the loop and the scf.yield (still inside the now-empty
  //    body). scf.yield is dropped together with the loop region.
  loop.erase();
  return ::mlir::success();
}

struct LowerReductionToStreamPass
    : public ::mlir::PassWrapper<LowerReductionToStreamPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerReductionToStreamPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-reduction-to-stream";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower the simple-reduction shape of an scf.for inside a "
           "dataflow.graph.func body into dataflow.stream + dataflow.carry "
           "streaming primitives.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::scf::SCFDialect, ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    // Snapshot graph.funcs first so we are not iterating while
    // mutating module-level symbols.
    ::llvm::SmallVector<::dataflow::GraphFuncOp, 8> graphs;
    for (auto graph : module.getOps<::dataflow::GraphFuncOp>())
      graphs.push_back(graph);

    for (::dataflow::GraphFuncOp graph : graphs) {
      if (graph.isExternal())
        continue;
      ::mlir::scf::ForOp loop = findSoleTopLevelFor(graph);
      if (!loop) {
        // Either zero or more-than-one top-level for. Both are
        // outside the smoke shape; leave the body alone.
        graph.emitRemark()
            << "loom-lower-reduction-to-stream: graph body has zero or "
               "multiple top-level scf.for ops; leaving as-is";
        continue;
      }
      if (!isEligibleReduction(loop)) {
        graph.emitRemark()
            << "loom-lower-reduction-to-stream: graph body's scf.for is "
               "not the simple-reduction shape (nested SCF, call op, or "
               "non-invariant bound); leaving as-is";
        continue;
      }
      if (failed(lowerOneReduction(graph, loop, builder))) {
        graph.emitRemark()
            << "loom-lower-reduction-to-stream: lowering aborted "
               "post-eligibility-check; leaving as-is";
        continue;
      }
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerReductionToStreamPass() {
  return std::make_unique<LowerReductionToStreamPass>();
}

void registerLowerReductionToStreamPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerReductionToStreamPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
