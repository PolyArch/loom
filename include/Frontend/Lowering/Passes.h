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
// inside a dataflow.thread body, emits a sibling dataflow.graph
// definition at module scope plus a dataflow.graph.launch at the cut
// site inside the thread. scf.for ops without iter_args are left in
// place; only the structured-reduction shape is promoted to a graph.
//
// Symbol naming: `g_<threadSym>_<seq>` where <seq> is the source-order
// index of the scf.for cut inside the thread.
std::unique_ptr<::mlir::Pass> createLowerForToGraphPass();

// Module-scope pass that expands known library helper calls inside
// dataflow.graph bodies into primitive operations before PnR.
// Unknown calls are left in place for the existing unsupported-call
// diagnostics.
std::unique_ptr<::mlir::Pass> createLowerKnownLibraryCallsPass();

// Module-scope owner for graph-local memory and structured regions. It
// normalizes supported LLVM and memref accesses to dataflow.load/store,
// computes basic graph-local alias-root partitions, and recursively lowers
// scf.if/scf.for/scf.while while carrying execution, values, and independent
// write/read frontiers. Raw parallel SCF fails before mutation.
std::unique_ptr<::mlir::Pass> createLowerGraphMemoryPass();

// Module-scope pass that promotes each used `arith.constant` op inside a
// dataflow.graph body into a `dataflow.constant` op driven by the body's
// leading `thread_ctrl` block argument. Graph-local scalar literals therefore
// remain visible to PnR as configurable hardware constants, including literals
// feeding scalar arithmetic, structured loop bounds, or streaming primitives.
std::unique_ptr<::mlir::Pass> createLowerGraphConstantsPass();

// Register the lowering passes with the global pass registry so
// loom-raise-opt can drive them via --loom-lower-forall-to-thread /
// --loom-lower-for-to-graph / --loom-lower-graph-memory /
// --loom-lower-graph-constants plus the combined
// --loom-lower-scf-to-dfg pipeline.
void registerLoweringPasses();

// Append the SCF-to-DFG lowering pipeline to the given pass manager:
//   loom-lower-forall-to-thread        (module-level)
//   loom-lower-for-to-graph            (module-level)
//   --canonicalize                     (upstream)
//   loom-lower-known-library-calls     (module-level)
//   loom-lower-graph-memory            (module-level)
//   loom-lower-graph-constants         (module-level)
//   --canonicalize                     (upstream)
void buildLoweringPipeline(::mlir::PassManager &pm);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_PASSES_H
