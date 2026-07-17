// Wrap loop-invariant scalar block arguments of a `dataflow.graph.func`
// with `dataflow.invariant` carriers so the streaming-primitive
// contract makes the invariance explicit.
//
// Rewrite scope:
//   For each `dataflow.graph.func` body that contains one directly
//   owned `dataflow.stream` op:
//     1. Use the stream's phase as the gating condition for new
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
//        materialize `%inv = dataflow.invariant %phase, %BA : T` once,
//        project it through `dataflow.gate`,
//        and rewrite all in-body uses of BA (other than the new
//        invariant op itself, each stream operand, each carry init
//        operand, and the bridge cast) to read %inv.
//
// Bail conditions:
//   * No directly owned stream leaves the graph unchanged.
//   * Multiple directly owned streams fail the pass when an eligible
//     invariant needs an unambiguous phase owner.
//   * Every block arg is already bridged, unused outside stream
//     operands, or non-carriable.
//   * The block arg has a non-numeric type that the invariant op
//     cannot carry.
//
// Rationale: this surfaces the "constant during the loop" semantics
// that the SpatialCore wrapper expects for arguments such as scaling
// factors (`f32`) and integer hyperparams. Loop-carried initializers
// remain one-shot carry init tokens.

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

// Return true if `t` is a scalar type supported by this rewrite.
bool isInvariantCarriable(::mlir::Type t) {
  return ::llvm::isa<::mlir::IntegerType, ::mlir::FloatType,
                     ::mlir::IndexType, ::mlir::NoneType>(t);
}

::llvm::SmallVector<::dataflow::StreamOp, 2>
collectDirectStreams(::mlir::Block &entry) {
  ::llvm::SmallVector<::dataflow::StreamOp, 2> streams;
  for (::mlir::Operation &op : entry.without_terminator()) {
    auto stream = ::llvm::dyn_cast<::dataflow::StreamOp>(op);
    if (stream)
      streams.push_back(stream);
  }
  return streams;
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

bool isStreamLoopBoundUse(::mlir::OpOperand &use) {
  auto stream = ::llvm::dyn_cast<::dataflow::StreamOp>(use.getOwner());
  if (!stream)
    return false;
  ::mlir::Value value = use.get();
  return value == stream.getInit() || value == stream.getLimit() ||
         value == stream.getStep();
}

bool isCarryInitUse(::mlir::OpOperand &use) {
  auto carry = ::llvm::dyn_cast<::dataflow::CarryOp>(use.getOwner());
  return carry && use.getOperandNumber() == 1;
}

bool isGraphReturnUse(::mlir::OpOperand &use) {
  return ::llvm::isa<::dataflow::GraphReturnOp>(use.getOwner());
}

// Rewrite eligible block arguments of `graph` with dataflow.invariant
// carriers driven by the unique stream's phase.
::mlir::LogicalResult rewriteOneGraph(::dataflow::GraphFuncOp graph,
                                      ::mlir::OpBuilder &builder) {
  ::mlir::Block &entry = graph.getBody().front();
  ::llvm::DenseSet<::mlir::Value> bridgedPorts;
  collectBridgedArgs(entry, bridgedPorts);

  ::llvm::SmallVector<::mlir::BlockArgument, 4> candidates;
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
      if (isStreamLoopBoundUse(use))
        continue;
      if (isCarryInitUse(use))
        continue;
      if (isGraphReturnUse(use))
        continue;
      hasNonStreamUse = true;
      break;
    }
    if (!hasNonStreamUse)
      continue;
    // (e) Skip non-carriable types (e.g. !llvm.ptr passed through).
    if (!isInvariantCarriable(ba.getType()))
      continue;
    candidates.push_back(ba);
  }

  if (candidates.empty())
    return ::mlir::success();

  auto streams = collectDirectStreams(entry);
  if (streams.empty())
    return ::mlir::success();
  if (streams.size() != 1) {
    graph.emitError("loom-lower-graph-invariant: graph requires invariant "
                    "lowering but has multiple directly owned "
                    "dataflow.stream phase owners");
    return ::mlir::failure();
  }
  ::dataflow::StreamOp stream = streams.front();
  ::mlir::Value phase = stream.getPhase();

  for (::mlir::BlockArgument ba : candidates) {
    // Materialize the invariant just after the stream so the phase
    // dominates it textually (graph regions are non-SSA, but keeping
    // the print order tidy makes the lowered IR easier to read).
    ::mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(stream);
    auto inv = ::dataflow::InvariantOp::create(builder, ba.getLoc(),
                                               ba.getType(), phase, ba);
    builder.setInsertionPointAfter(inv);
    auto gate =
        ::dataflow::GateOp::create(builder, ba.getLoc(), builder.getI1Type(),
                                   ba.getType(), phase, inv.getOutput());
    ::mlir::Value newVal = gate.getAfterValue();
    // Replace every use of `ba` except (i) the just-created invariant
    // op (which must keep reading the raw block arg), (ii) any
    // unrealized_conversion_cast (the bridge consumes the raw arg),
    // and (iii) any dataflow.stream operand (init / limit / step), because
    // the stream op owns the raw loop-bound contract, and (iv) any
    // dataflow.carry init operand, which must remain a one-shot token, and
    // (v) any graph.return operand, which is already in the result domain.
    ba.replaceUsesWithIf(newVal, [&](::mlir::OpOperand &use) {
      ::mlir::Operation *owner = use.getOwner();
      if (owner == inv.getOperation())
        return false;
      if (::llvm::isa<::mlir::UnrealizedConversionCastOp>(owner))
        return false;
      if (isStreamLoopBoundUse(use))
        return false;
      if (isCarryInitUse(use))
        return false;
      if (isGraphReturnUse(use))
        return false;
      return true;
    });
  }
  return ::mlir::success();
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
           "unique directly owned dataflow.stream phase.";
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
      if (::mlir::failed(rewriteOneGraph(graph, builder))) {
        signalPassFailure();
        return;
      }
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
