// Funnel the `%done : none` rendezvous tokens emitted by every
// `dataflow.load` and `dataflow.store` op inside a
// `dataflow.graph.func` body into a single `dataflow.sync` op placed
// right before the terminating `dataflow.graph.return`. The sync's
// outputs would normally feed downstream rendezvous, but at this
// pipeline stage the graph body's only sink is the graph.return, so
// the pass routes `%synced[0]` into the `done_out` slot of the
// graph.return. That keeps the sync alive across the trailing
// `--canonicalize` (sync is `Pure`, so an unused-results form would
// be DCE-ed) and explicitly gates the graph's outgoing token on the
// rendezvous of every memory op inside the body.
//
// Heuristic (smoke baseline):
//   For each `dataflow.graph.func` body:
//     1. Scan the entry block (top-level ops only) and collect every
//        `%done` SSA result produced by a `dataflow.load` or
//        `dataflow.store` op that lives directly there. Memory ops
//        nested inside an scf.for / scf.if region of the graph body
//        are skipped: those regions are SSA-shaped, so their `%done`
//        values cannot be referenced from the outer block.
//     2. Materialize `%synced:N = dataflow.sync %d1, ..., %dN
//        : (none, ..., none) -> (none, ..., none)` immediately before
//        the terminator.
//     3. Replace the leading operand of the `dataflow.graph.return`
//        terminator (the `done_out : none` slot, currently sourced
//        from the body's `thread_ctrl` block argument) with
//        `%synced#0`. Every other terminator operand is left in
//        place. Routing the done port through `%synced#0` is the
//        idiomatic option (b) called out in the task brief.
//
// Bail conditions (graph left unchanged):
//   * The graph.func has zero top-level `dataflow.load` and zero
//     top-level `dataflow.store` ops in its body (no rendezvous is
//     needed at this layer; if every memory op is nested in an
//     scf.for / scf.if, the rewrite would lift values across regions
//     that do not share an SSA scope).
//   * The graph.func is external (no body to walk).
//   * The graph.func's terminator is missing or has zero operands
//     (the graph signature is malformed; refuse to rewrite).
//
// Rationale: this surfaces the "memory-rendezvous" semantics of a
// streaming graph body. Even when a future iteration moves the sync's
// outputs to feed real downstream consumers, the rewrite contract --
// "the graph's done_out token waits for every load/store to retire"
// -- is the right gating shape for a SpatialCore graph.

#include "Frontend/Lowering/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace {

constexpr ::llvm::StringLiteral kConditionalStoreDoneAttr =
    "loom.conditional_store_done";

std::optional<::mlir::Value> findConditionalStoreDone(::mlir::Value storeDone) {
  ::mlir::Value replacement;
  for (::mlir::OpOperand &use : storeDone.getUses()) {
    auto mux = ::llvm::dyn_cast<::dataflow::MuxOp>(use.getOwner());
    if (!mux || !mux->hasAttr(kConditionalStoreDoneAttr))
      continue;
    if (mux.getOutput().getType() != storeDone.getType())
      continue;
    if (replacement)
      return std::nullopt;
    replacement = mux.getOutput();
  }
  if (!replacement)
    return std::nullopt;
  return replacement;
}

std::optional<unsigned> findDemuxOutputIndex(::dataflow::DemuxOp demux,
                                             ::mlir::Value output) {
  for (unsigned i = 0, e = demux.getOutputs().size(); i < e; ++i) {
    if (demux.getOutputs()[i] == output)
      return i;
  }
  return std::nullopt;
}

std::optional<::mlir::Value>
materializeConditionalStoreDone(::dataflow::StoreOp store,
                                ::mlir::OpBuilder &builder,
                                ::dataflow::GraphReturnOp ret) {
  auto ctrlDemux = store.getCtrl().getDefiningOp<::dataflow::DemuxOp>();
  if (!ctrlDemux)
    return std::nullopt;
  std::optional<unsigned> storeLane =
      findDemuxOutputIndex(ctrlDemux, store.getCtrl());
  if (!storeLane)
    return std::nullopt;

  ::llvm::SmallVector<::mlir::Value, 4> inputs;
  inputs.reserve(ctrlDemux.getOutputs().size());
  for (unsigned i = 0, e = ctrlDemux.getOutputs().size(); i < e; ++i) {
    if (i == *storeLane) {
      inputs.push_back(store.getDone());
      continue;
    }
    inputs.push_back(ctrlDemux.getOutputs()[i]);
  }

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(ret);
  auto mergedDone = ::dataflow::MuxOp::create(
      builder, store.getLoc(), builder.getNoneType(), ctrlDemux.getSel(),
      inputs);
  mergedDone->setAttr(kConditionalStoreDoneAttr, builder.getUnitAttr());
  return mergedDone.getOutput();
}

// Collect every `%done : none` token produced by a `dataflow.load` or
// `dataflow.store` op that sits directly in the graph entry block.
// Done tokens defined inside nested scf regions cannot be referenced
// from the outer block (those regions are SSA-shaped, not graph), so
// gathering only top-level memory ops keeps the rendezvous SSA-legal.
// Graphs whose memory ops all live inside an scf.for / scf.if body
// will report an empty token list and the pass will leave them
// unchanged. A future iteration can extend this to emit one sync per
// nested region if downstream consumers need a finer rendezvous.
void collectDoneTokens(::dataflow::GraphFuncOp graph,
                       ::mlir::OpBuilder &builder,
                       ::dataflow::GraphReturnOp ret,
                       ::llvm::SmallVectorImpl<::mlir::Value> &out) {
  ::mlir::Block &entry = graph.getBody().front();
  for (::mlir::Operation &op : entry.without_terminator()) {
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op))
      out.push_back(load.getDone());
    else if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op)) {
      if (std::optional<::mlir::Value> done =
              findConditionalStoreDone(store.getDone())) {
        out.push_back(*done);
        continue;
      }
      if (std::optional<::mlir::Value> done =
              materializeConditionalStoreDone(store, builder, ret)) {
        out.push_back(*done);
        continue;
      }
      out.push_back(store.getDone());
    }
  }
}

// Materialize the sync op for `graph` and rewrite its graph.return so
// the leading done_out slot is sourced from `%synced#0`. Returns true
// on rewrite, false if the graph has no memory ops (or its terminator
// is malformed).
bool rewriteOneGraph(::dataflow::GraphFuncOp graph,
                     ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::Value, 8> dones;
  ::mlir::Block &entry = graph.getBody().front();
  auto ret = ::llvm::dyn_cast_or_null<::dataflow::GraphReturnOp>(
      entry.getTerminator());
  if (!ret)
    return false;
  if (ret.getValues().empty())
    return false;
  collectDoneTokens(graph, builder, ret, dones);
  if (dones.empty())
    return false;

  ::llvm::SmallVector<::mlir::Type, 8> resultTypes(dones.size(),
                                                   builder.getNoneType());

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(ret);
  auto sync = ::dataflow::SyncOp::create(builder, ret.getLoc(), resultTypes,
                                         dones);

  // Route the graph's outgoing `done_out : none` through the sync's
  // first result. Every other terminator operand stays as-is.
  ret.getValuesMutable()[0].assign(sync.getOutputs()[0]);
  return true;
}

struct LowerGraphSyncPass
    : public ::mlir::PassWrapper<LowerGraphSyncPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphSyncPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-sync";
  }
  ::llvm::StringRef getDescription() const final {
    return "Funnel dataflow.load / dataflow.store done tokens inside a "
           "dataflow.graph.func body into a single dataflow.sync op and "
           "route the rendezvous output into the graph's done_out slot.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::func::FuncDialect, ::dataflow::DataflowDialect>();
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

std::unique_ptr<::mlir::Pass> createLowerGraphSyncPass() {
  return std::make_unique<LowerGraphSyncPass>();
}

void registerLowerGraphSyncPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphSyncPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
