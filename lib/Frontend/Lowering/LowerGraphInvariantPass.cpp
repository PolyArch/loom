// Wrap loop-invariant scalar block arguments of a `dataflow.graph.func`
// with `dataflow.invariant` carriers so the streaming-primitive
// contract makes the invariance explicit.
//
// Heuristic (smoke baseline):
//   For each `dataflow.graph.func` body that already contains at
//   least one `dataflow.stream` op:
//     1. Use the first stream's `rwc` as the gating cond for new
//        invariants.
//     2. For each entry-block argument BA that
//          (a) is NOT the leading `thread_ctrl : none` slot,
//          (b) has NOT been bridged to a memref (i.e., is not the
//              source of an `unrealized_conversion_cast` produced by
//              `loom-lower-graph-memory`),
//          (c) may feed a `dataflow.stream`, but also has at least one
//              non-stream use in the graph body,
//          (d) IS used somewhere else inside the graph body,
//          (e) has a type that the dataflow.invariant verifier
//              admits without further bridging (numeric / index /
//              none),
//        materialize `%inv = dataflow.invariant %rwc, %BA : T` once
//        and rewrite all in-body uses of BA (other than the new
//        invariant op itself, each stream operand, and the bridge
//        cast) to read %inv.
//
// Bail conditions (graph left unchanged):
//   * No `dataflow.stream` in the body (the loop is not yet streamed,
//     so there is no rwc to drive the invariant).
//   * Every block arg is already bridged, unused outside stream
//     operands, or non-carriable.
//   * The block arg has a non-numeric type that the invariant op
//     cannot carry.
//
// Rationale: this surfaces the "constant during the loop" semantics
// that the SpatialCore wrapper expects for arguments such as scaling
// factors (`f32`), accumulator initializers, and integer hyperparams.

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// Return true if `t` is one of the types the dataflow.invariant op
// is currently used with in the smoke corpus: numeric scalars (int /
// float), `index`, or `none`. Non-numeric / non-trivial types are
// skipped.
bool isInvariantCarriable(::mlir::Type t) {
  return ::llvm::isa<::mlir::IntegerType, ::mlir::FloatType,
                     ::mlir::IndexType, ::mlir::NoneType>(t);
}

// Locate the first dataflow.stream op directly inside `entry` and
// return it (or null if there is none).
::dataflow::StreamOp findFirstStream(::mlir::Block &entry) {
  for (::mlir::Operation &op : entry.without_terminator()) {
    if (auto s = ::llvm::dyn_cast<::dataflow::StreamOp>(op))
      return s;
  }
  return {};
}

// Mark every block arg consumed by a `unrealized_conversion_cast` as
// already-bridged (the memory pass owns its conversion).
void collectBridgedArgs(::mlir::Block &entry,
                        ::llvm::DenseSet<::mlir::Value> &out) {
  for (::mlir::Operation &op : entry.without_terminator()) {
    if (auto cast = ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(op)) {
      for (::mlir::Value v : cast.getInputs())
        out.insert(v);
    }
  }
}

// Rewrite eligible block arguments of `graph` with dataflow.invariant
// carriers driven by the first stream's rwc. Returns the number of
// invariants emitted.
unsigned rewriteOneGraph(::dataflow::GraphFuncOp graph,
                         ::mlir::OpBuilder &builder) {
  ::mlir::Block &entry = graph.getBody().front();
  ::dataflow::StreamOp stream = findFirstStream(entry);
  if (!stream)
    return 0;
  ::mlir::Value rwc = stream.getRwc();

  ::llvm::DenseSet<::mlir::Value> bridgedPorts;
  collectBridgedArgs(entry, bridgedPorts);

  unsigned added = 0;
  for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
    ::mlir::BlockArgument ba = entry.getArgument(i);
    // (a) Skip the leading thread_ctrl slot; it is the firing token.
    if (i == 0 && ::llvm::isa<::mlir::NoneType>(ba.getType()))
      continue;
    // (b) Skip pointer-typed args that the memory pass already
    //     bridged into a memref.
    if (bridgedPorts.contains(ba))
      continue;
    // (c, d) Skip args with no non-stream in-body uses.
    bool hasNonStreamUse = false;
    for (::mlir::OpOperand &use : ba.getUses()) {
      auto streamUser = ::llvm::dyn_cast<::dataflow::StreamOp>(use.getOwner());
      if (streamUser) {
        ::mlir::Value v = use.get();
        if (v == streamUser.getLb() || v == streamUser.getUb() ||
            v == streamUser.getStep())
          continue;
      }
      hasNonStreamUse = true;
      break;
    }
    if (!hasNonStreamUse)
      continue;
    // (e) Skip non-carriable types (e.g. !llvm.ptr passed through).
    if (!isInvariantCarriable(ba.getType()))
      continue;

    // Materialize the invariant just after the stream so the rwc
    // dominates it textually (graph regions are non-SSA, but keeping
    // the print order tidy makes the lowered IR easier to read).
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(stream);
    auto inv = ::dataflow::InvariantOp::create(builder, ba.getLoc(),
                                               ba.getType(), rwc, ba);
    ::mlir::Value newVal = inv.getOutput();
    // Replace every use of `ba` except (i) the just-created invariant
    // op (which must keep reading the raw block arg), (ii) any
    // unrealized_conversion_cast (the bridge consumes the raw arg),
    // and (iii) any dataflow.stream operand (lb / ub / step), because
    // the stream op owns the raw loop-bound contract.
    ba.replaceUsesWithIf(newVal, [&](::mlir::OpOperand &use) {
      ::mlir::Operation *owner = use.getOwner();
      if (owner == inv.getOperation())
        return false;
      if (::llvm::isa<::mlir::UnrealizedConversionCastOp>(owner))
        return false;
      if (auto s = ::llvm::dyn_cast<::dataflow::StreamOp>(owner)) {
        ::mlir::Value v = use.get();
        if (v == s.getLb() || v == s.getUb() || v == s.getStep())
          return false;
      }
      return true;
    });
    ++added;
  }
  return added;
}

struct LowerGraphInvariantPass
    : public ::mlir::PassWrapper<LowerGraphInvariantPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphInvariantPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-invariant";
  }
  ::llvm::StringRef getDescription() const final {
    return "Wrap loop-invariant block arguments of dataflow.graph.func "
           "bodies with dataflow.invariant carriers driven by the body's "
           "existing dataflow.stream rwc.";
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
      (void)rewriteOneGraph(graph, builder);
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerGraphInvariantPass() {
  return std::make_unique<LowerGraphInvariantPass>();
}

void registerLowerGraphInvariantPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphInvariantPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
