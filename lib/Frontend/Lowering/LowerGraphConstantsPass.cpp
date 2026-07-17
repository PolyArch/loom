// Replace graph-local scalar value sources inside `dataflow.graph` bodies
// with a `dataflow.constant` fired by the graph body's `thread_ctrl` block
// argument.
//
// Every scalar seed that remains in a graph is a hardware-visible source. A
// scalar literal or poison seed may feed a stream primitive, a structured loop
// bound, or a normal arithmetic op such as arith.cmpi. All of those cases must
// be visible to PnR as configurable constant resources rather than residual
// non-fabric ops.
//
// Bail conditions (graph left unchanged):
//   * The graph is external (no body to walk).
//   * The graph has no convertible scalar source ops with users.
//
// Per-source skip (source left in place):
//   * The source has no users at all (DCE will pick it up).
//   * A poison source has a type with no zero attribute.
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
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// The distinguished leading `none` block argument is the graph start firing
// token; it is separate from the payload-only FunctionType.
::mlir::Value getThreadCtrl(::dataflow::GraphOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  ::mlir::Value first = entry.getArgument(0);
  return ::llvm::isa<::mlir::NoneType>(first.getType()) ? first
                                                        : ::mlir::Value{};
}

struct ScalarSource {
  ::mlir::Operation *op = nullptr;
  ::mlir::TypedAttr valueAttr;
};

::mlir::TypedAttr zeroAttrForPoison(::mlir::ub::PoisonOp poison,
                                    ::mlir::OpBuilder &builder) {
  ::mlir::Attribute attr = builder.getZeroAttr(poison.getType());
  return ::llvm::dyn_cast_if_present<::mlir::TypedAttr>(attr);
}

// Lift every used scalar source anywhere inside `graph`'s body to a
// `dataflow.constant` driven by `ctrl`. The walk descends into
// nested scf regions because the memory pass routinely materializes
// the `%c0 : index` constant inside an scf.for / scf.while body when
// the graph's load/store ops are themselves nested. The graph's
// `thread_ctrl` block argument is visible from every nested region,
// so the rewrite is SSA-legal regardless of the constant's depth.
// Returns the number of sources converted.
unsigned rewriteOneGraph(::dataflow::GraphOp graph, ::mlir::Value ctrl,
                         ::mlir::OpBuilder &builder) {
  // Collect targets up front so the walk is independent of the
  // mutations performed below.
  ::llvm::SmallVector<ScalarSource, 16> targets;
  graph.getBody().walk([&](::mlir::Operation *op) {
    if (auto cst = ::mlir::dyn_cast<::mlir::arith::ConstantOp>(op)) {
      if (!cst.use_empty())
        targets.push_back(ScalarSource{cst.getOperation(), cst.getValue()});
      return;
    }
    auto poison = ::mlir::dyn_cast<::mlir::ub::PoisonOp>(op);
    if (!poison || poison.use_empty())
      return;
    ::mlir::TypedAttr zero = zeroAttrForPoison(poison, builder);
    if (!zero)
      return;
    targets.push_back(ScalarSource{poison.getOperation(), zero});
  });

  unsigned converted = 0;
  for (ScalarSource target : targets) {
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(target.op);
    auto dconst = ::dataflow::ConstantOp::create(
        builder, target.op->getLoc(), target.valueAttr.getType(), ctrl,
        ::mlir::cast<::mlir::Attribute>(target.valueAttr));
    target.op->getResult(0).replaceAllUsesWith(dconst.getValue());
    target.op->erase();
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
    return "Promote arith.constant ops inside dataflow.graph bodies "
           "to dataflow.constant ops driven by the body's leading "
           "thread_ctrl block argument.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::ub::UBDialect, ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
    for (auto graph : module.getOps<::dataflow::GraphOp>())
      graphs.push_back(graph);

    for (::dataflow::GraphOp graph : graphs) {
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
