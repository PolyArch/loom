// Replace `arith.constant` ops inside `dataflow.graph.func` bodies that
// participate in a streaming-primitive chain with a `dataflow.constant`
// fired by the graph body's `thread_ctrl` block argument.
//
// Heuristic (smoke baseline):
//   For each `dataflow.graph.func` body that already contains at least
//   one `dataflow.stream` op (i.e., the reduction-to-stream pass
//   succeeded earlier in the pipeline, so there is a streaming context
//   to anchor `%ctrl` against):
//     1. Walk every `arith.constant` op directly in the graph entry
//        block.
//     2. If the constant has at least one user that is a streaming
//        primitive op (`dataflow.stream`, `dataflow.carry`,
//        `dataflow.invariant`, `dataflow.load`, `dataflow.store`),
//        materialize a sibling `%dconst = dataflow.constant %ctrl
//        {const_value = <orig.value>} : <T>` op, replace every use of
//        the original constant with `%dconst`, and erase the
//        `arith.constant`.
//     3. The `%ctrl` operand is the graph.func's leading
//        `thread_ctrl : none` block argument (a strict invariant of
//        every graph.func produced by the for-to-graph pass).
//
// Bail conditions (graph left unchanged):
//   * The graph.func is external (no body to walk).
//   * The graph.func's entry block has no `arith.constant` ops with at
//     least one streaming-primitive user (transitive equivalent of the
//     "no streaming context" check; it skips both fully-non-streaming
//     bodies and bodies whose constants are pure scalar plumbing).
//
// Per-constant skip (constant left in place):
//   * The constant has no users at all (DCE will pick it up).
//   * The constant is consumed only by other arith / cast plumbing and
//     never reaches a streaming primitive directly.
//
// Note: the graph.func need not contain a top-level `dataflow.stream`
// op for the rewrite to apply. Several cmsis-* graph bodies bail out
// of the reduction-to-stream pass (because they have a top-level
// pointer-walking scf.for or a nested loop nest the simple-reduction
// shape rejects), yet still contain `dataflow.load` / `dataflow.store`
// streaming primitives the memory pass introduced. In those graphs
// the `%c0 : index` produced by the memory pass legitimately feeds
// the streaming load/store address ports; promoting it is in scope.
//
// Rationale: this surfaces every "scalar literal that the streaming
// loop body folds into" as an explicit dataflow.constant source. The
// existing memory-tokenization pass introduces a `%c0 : index`
// constant that is consumed by every `dataflow.load` / `dataflow.store`
// it emits; lifting that constant to `dataflow.constant` removes the
// last residual `arith.constant` from the lowered streaming bodies in
// the cmsis-* corpora.

#include "Frontend/Lowering/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// Return true if `op` is one of the streaming-primitive ops whose
// presence as a user qualifies a feeding arith.constant for promotion
// to dataflow.constant. The set mirrors the brief's chain definition:
//   stream / carry / invariant / load / store.
bool isStreamingPrimitiveUser(::mlir::Operation *op) {
  return ::llvm::isa<::dataflow::StreamOp, ::dataflow::CarryOp,
                     ::dataflow::InvariantOp, ::dataflow::LoadOp,
                     ::dataflow::StoreOp>(op);
}

// True iff at least one user of `v` is a streaming-primitive op.
bool feedsAnyStreamingPrimitive(::mlir::Value v) {
  for (::mlir::OpOperand &use : v.getUses()) {
    if (isStreamingPrimitiveUser(use.getOwner()))
      return true;
  }
  return false;
}

// The leading `none`-typed signature input is the `thread_ctrl` firing
// token; rely on the lowering pipeline's invariant that graph.func
// signatures always begin with `(none, ...)`.
::mlir::Value getThreadCtrl(::dataflow::GraphFuncOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  ::mlir::Value first = entry.getArgument(0);
  return ::llvm::isa<::mlir::NoneType>(first.getType()) ? first
                                                        : ::mlir::Value{};
}

// Lift every eligible `arith.constant` anywhere inside `graph`'s body
// to a `dataflow.constant` driven by `ctrl`. The walk descends into
// nested scf regions because the memory pass routinely materializes
// the `%c0 : index` constant inside an scf.for / scf.while body when
// the graph's load/store ops are themselves nested. The graph.func's
// `thread_ctrl` block argument is visible from every nested region,
// so the rewrite is SSA-legal regardless of the constant's depth.
// Returns the number of constants converted.
unsigned rewriteOneGraph(::dataflow::GraphFuncOp graph, ::mlir::Value ctrl,
                         ::mlir::OpBuilder &builder) {
  // Collect targets up front so the walk is independent of the
  // mutations performed below.
  ::llvm::SmallVector<::mlir::arith::ConstantOp, 16> targets;
  graph.getBody().walk([&](::mlir::arith::ConstantOp cst) {
    if (cst.use_empty())
      return;
    if (!feedsAnyStreamingPrimitive(cst.getResult()))
      return;
    targets.push_back(cst);
  });

  unsigned converted = 0;
  for (::mlir::arith::ConstantOp cst : targets) {
    ::mlir::TypedAttr valueAttr = cst.getValue();
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(cst.getOperation());
    auto dconst = ::dataflow::ConstantOp::create(
        builder, cst.getLoc(), valueAttr.getType(), ctrl,
        ::mlir::cast<::mlir::Attribute>(valueAttr));
    cst.getResult().replaceAllUsesWith(dconst.getValue());
    cst.erase();
    ++converted;
  }
  return converted;
}

struct LowerGraphConstantsPass
    : public ::mlir::PassWrapper<LowerGraphConstantsPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphConstantsPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-constants";
  }
  ::llvm::StringRef getDescription() const final {
    return "Promote arith.constant ops feeding streaming primitives "
           "inside dataflow.graph.func bodies to dataflow.constant ops "
           "driven by the body's leading thread_ctrl block argument.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
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
      ::mlir::Value ctrl = getThreadCtrl(graph);
      if (!ctrl)
        continue;
      (void)rewriteOneGraph(graph, ctrl, builder);
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerGraphConstantsPass() {
  return std::make_unique<LowerGraphConstantsPass>();
}

void registerLowerGraphConstantsPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphConstantsPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
