// Replace `arith.constant` ops inside `dataflow.graph.func` bodies with a
// `dataflow.constant` fired by the graph body's `thread_ctrl` block argument.
//
// Every constant that remains in a graph is a hardware-visible source. A
// scalar literal may feed a stream primitive, a structured loop bound, or a
// normal arithmetic op such as arith.cmpi. All of those cases must be visible
// to PnR as configurable constant resources rather than residual arith ops.
//
// Bail conditions (graph left unchanged):
//   * The graph.func is external (no body to walk).
//   * The graph.func has no arith.constant ops with users.
//
// Per-constant skip (constant left in place):
//   * The constant has no users at all (DCE will pick it up).
//
// Rationale: this surfaces every graph-local scalar literal as an explicit
// dataflow.constant source. That keeps the Fabric ADG/PnR boundary honest:
// constants consume constant-capable fabric resources and carry their config
// values through the mapping artifact.

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

// Lift every used `arith.constant` anywhere inside `graph`'s body to a
// `dataflow.constant` driven by `ctrl`. The walk descends into
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
    return "Promote arith.constant ops inside dataflow.graph.func bodies "
           "to dataflow.constant ops driven by the body's leading "
           "thread_ctrl block argument.";
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
