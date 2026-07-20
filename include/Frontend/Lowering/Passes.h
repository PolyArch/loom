#ifndef LOOM_FRONTEND_LOWERING_PASSES_H
#define LOOM_FRONTEND_LOWERING_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PassManager;
} // namespace mlir

namespace loom {
namespace lowering {

// Module-scope diagnostic pass that rejects implicit host scf.forall thread
// promotion until a recognized Loom thread mapping and a faithfully
// represented domain have been selected by the structured owner.
std::unique_ptr<::mlir::Pass> createLowerForallToThreadPass();

// Module-scope atomic publisher. It consumes explicit loom.spatial_region
// candidates inside dataflow.thread definitions, finalizes a scratch module
// through the graph lowering pipeline and native validator, then publishes
// matching dataflow.graph definitions and launches only when the whole
// transaction succeeds.
//
// Published graph symbols use a construction-local ordinal for readability;
// symbol spelling is not graph or artifact identity.
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
//   loom-lower-for-to-graph            (module-level)
// The for-to-graph publisher internally owns canonicalization, known-library
// expansion, graph memory/control lowering, constant promotion, and native
// validation.
void buildLoweringPipeline(::mlir::PassManager &pm);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_PASSES_H
