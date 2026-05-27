// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-forall-to-thread       (module-level)
//     loom-lower-for-to-graph           (module-level)
//     --canonicalize                    (upstream)
//     loom-lower-reduction-to-stream    (module-level)
//     loom-lower-graph-memory           (module-level)
//     loom-lower-graph-invariant        (module-level)
//     loom-lower-graph-control          (module-level)
//     loom-lower-graph-constants        (module-level)
//     loom-lower-graph-sync             (module-level)
//     --canonicalize                    (upstream)
//
// The forall-to-thread pass runs first so that the for-to-graph pass
// sees scf.for ops already inside dataflow.thread bodies. The
// canonicalizer between for-to-graph and reduction-to-stream cleans
// up trivial dead bridge values before we walk the graph.func body.
// graph-memory runs before graph-invariant so the latter can skip
// pointer args that have already been bridged to memref. graph-control
// runs after graph-invariant so any enclosing loop has already been
// streamed (its `cond` is body-phase) and before graph-constants so
// constant promotion observes the post-mux IR. The graph-constants
// pass runs after graph-memory because the memory pass introduces a
// `%c0 : index` constant whose only consumers are the streaming
// load/store ports; promoting it to dataflow.constant removes the
// last residual arith.constant from streaming bodies. graph-sync runs
// last so it can collect every `%done : none` token produced by
// dataflow.load / dataflow.store ops the prior passes emitted;
// constants before sync gives the lit diffs a stable order. The
// closing canonicalize pass cleans up dead llvm.getelementptr /
// dataflow.carry chains the memory pass leaves behind.

#include "Frontend/Lowering/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace lowering {

void registerLowerForallToThreadPass();
void registerLowerForToGraphPass();
void registerLowerGraphConstantsPass();
void registerLowerGraphControlPass();
void registerLowerGraphInvariantPass();
void registerLowerGraphMemoryPass();
void registerLowerGraphSyncPass();
void registerLowerReductionToStreamPass();

static void buildPipelineOnOpPassManager(::mlir::OpPassManager &pm) {
  pm.addPass(createLowerForallToThreadPass());
  pm.addPass(createLowerForToGraphPass());
  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(createLowerReductionToStreamPass());
  pm.addPass(createLowerGraphMemoryPass());
  pm.addPass(createLowerGraphInvariantPass());
  pm.addPass(createLowerGraphControlPass());
  pm.addPass(createLowerGraphConstantsPass());
  pm.addPass(createLowerGraphSyncPass());
  pm.addPass(::mlir::createCanonicalizerPass());
}

void registerLoweringPasses() {
  registerLowerForallToThreadPass();
  registerLowerForToGraphPass();
  registerLowerGraphConstantsPass();
  registerLowerGraphControlPass();
  registerLowerGraphInvariantPass();
  registerLowerGraphMemoryPass();
  registerLowerGraphSyncPass();
  registerLowerReductionToStreamPass();
  static bool once = []() {
    ::mlir::PassPipelineRegistration<>(
        "loom-lower-scf-to-dfg",
        "Run the standard Loom SCF-to-DFG lowering pipeline.",
        buildPipelineOnOpPassManager);
    return true;
  }();
  (void)once;
}

void buildLoweringPipeline(::mlir::PassManager &pm) {
  buildPipelineOnOpPassManager(pm);
}

} // namespace lowering
} // namespace loom
