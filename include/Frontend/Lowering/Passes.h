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

// Register the two lowering passes with the global pass registry so
// loom-raise-opt can drive them via --loom-lower-forall-to-thread /
// --loom-lower-for-to-graph plus the combined
// --loom-lower-scf-to-dfg pipeline.
void registerLoweringPasses();

// Append the SCF-to-DFG lowering pipeline to the given pass manager:
//   loom-lower-forall-to-thread       (module-level)
//   loom-lower-for-to-graph           (module-level)
//   --canonicalize                    (upstream)
void buildLoweringPipeline(::mlir::PassManager &pm);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_PASSES_H
