#ifndef LOOM_FRONTEND_LOWERING_PASSES_H
#define LOOM_FRONTEND_LOWERING_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PassManager;
} // namespace mlir

namespace loom {
namespace lowering {

// Module-scope pass that walks every func.func and, for each top-level
// scf.forall, emits a sibling dataflow.thread definition at module
// scope plus a dataflow.thread.launch at the original site. The
// forall body is moved into the thread def body and the
// scf.forall.in_parallel terminator is replaced with
// dataflow.thread.yield.
//
// Symbol naming: `t_<funcSym>_<seq>` where <seq> is the source-order
// index of the forall in the function.
//
// Nested foralls are not yet handled; they are left in place with a
// TODO marker.
std::unique_ptr<::mlir::Pass> createLowerForallToThreadPass();

// Module-scope pass that, for each scf.for op with iter_args found
// inside a dataflow.thread body, emits a sibling dataflow.graph.func
// definition at module scope plus a dataflow.graph.launch at the cut
// site inside the thread. scf.for ops without iter_args are left in
// place; only the structured-reduction shape is promoted to a graph.
//
// Symbol naming: `g_<threadSym>_<seq>` where <seq> is the source-order
// index of the scf.for cut inside the thread.
std::unique_ptr<::mlir::Pass> createLowerForToGraphPass();

// Module-scope pass that, for each dataflow.graph.func body whose
// sole top-level scf.for matches the simple-reduction shape, lowers
// that loop into dataflow.stream + dataflow.carry streaming
// primitives plus the original body ops moved out into the graph
// entry block. Graph bodies that do not match (nested SCF, call ops,
// multiple top-level loops) are left in place with a remark.
std::unique_ptr<::mlir::Pass> createLowerReductionToStreamPass();

// Module-scope pass that walks every dataflow.graph.func body and
// rewrites residual `llvm.load` / `llvm.store` ops into
// `dataflow.load` / `dataflow.store` streaming primitives. The
// pointer is bridged to a memref via `unrealized_conversion_cast`
// and the address port is driven by either the indexed gep operand
// or zero (for direct / carry-walking accesses).
std::unique_ptr<::mlir::Pass> createLowerGraphMemoryPass();

// Module-scope pass that wraps loop-invariant scalar block arguments
// of a dataflow.graph.func body with `dataflow.invariant` carriers
// driven by an existing `dataflow.stream`'s rwc. Graphs without a
// stream are left unchanged.
std::unique_ptr<::mlir::Pass> createLowerGraphInvariantPass();

// Module-scope pass that promotes each used `arith.constant` op inside a
// dataflow.graph.func body into a `dataflow.constant` op driven by the body's
// leading `thread_ctrl` block argument. Graph-local scalar literals therefore
// remain visible to PnR as configurable hardware constants, including literals
// feeding scalar arithmetic, structured loop bounds, or streaming primitives.
std::unique_ptr<::mlir::Pass> createLowerGraphConstantsPass();

// Module-scope pass that walks every dataflow.graph.func body,
// rewriting `scf.if` ops in post-order:
//   * scf.if with results, both regions present and pure -> emit one
//     dataflow.mux per result and lift the body ops out;
//   * scf.if with no results, only the then-region populated and pure
//     -> emit dataflow.gate around each gate-friendly body result and
//     lift the body ops out;
//   * effectful or unmodeled shapes are left in place with a remark.
std::unique_ptr<::mlir::Pass> createLowerGraphControlPass();

// Module-scope pass that funnels every `%done : none` token produced
// by a `dataflow.load` / `dataflow.store` inside a dataflow.graph.func
// body into a single `dataflow.sync` op placed before the terminator,
// and routes the sync's first output into the graph.return's
// `done_out` slot. Graphs without any memory ops are left unchanged.
std::unique_ptr<::mlir::Pass> createLowerGraphSyncPass();

// Register the lowering passes with the global pass registry so
// loom-raise-opt can drive them via --loom-lower-forall-to-thread /
// --loom-lower-for-to-graph / --loom-lower-reduction-to-stream /
// --loom-lower-graph-memory / --loom-lower-graph-invariant /
// --loom-lower-graph-control / --loom-lower-graph-constants /
// --loom-lower-graph-sync plus the combined --loom-lower-scf-to-dfg
// pipeline.
void registerLoweringPasses();

// Append the SCF-to-DFG lowering pipeline to the given pass manager:
//   loom-lower-forall-to-thread        (module-level)
//   loom-lower-for-to-graph            (module-level)
//   --canonicalize                     (upstream)
//   loom-lower-reduction-to-stream     (module-level)
//   loom-lower-graph-memory            (module-level)
//   loom-lower-graph-invariant         (module-level)
//   loom-lower-graph-control           (module-level)
//   loom-lower-graph-constants         (module-level)
//   loom-lower-graph-sync              (module-level)
//   --canonicalize                     (upstream)
void buildLoweringPipeline(::mlir::PassManager &pm);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_PASSES_H
