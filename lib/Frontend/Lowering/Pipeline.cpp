// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-forall-to-thread       (module-level)
//     loom-lower-for-to-graph           (module-level)
//     --canonicalize                    (upstream)
//     loom-lower-reduction-to-stream    (module-level)
//     loom-lower-graph-memory           (module-level)
//     loom-lower-graph-invariant        (module-level)
//     --canonicalize                    (upstream)
//
// The forall-to-thread pass runs first so that the for-to-graph pass
// sees scf.for ops already inside dataflow.thread bodies. The
// canonicalizer between for-to-graph and reduction-to-stream cleans
// up trivial dead bridge values before we walk the graph.func body.
// graph-memory runs before graph-invariant so the latter can skip
// pointer args that have already been bridged to memref. The closing
// canonicalize pass cleans up dead llvm.getelementptr / dataflow.carry
// chains the memory pass leaves behind.

#include "Frontend/Lowering/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace lowering {

void registerLowerForallToThreadPass();
void registerLowerForToGraphPass();
void registerLowerGraphInvariantPass();
void registerLowerGraphMemoryPass();
void registerLowerReductionToStreamPass();

static void buildPipelineOnOpPassManager(::mlir::OpPassManager &pm) {
  pm.addPass(createLowerForallToThreadPass());
  pm.addPass(createLowerForToGraphPass());
  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(createLowerReductionToStreamPass());
  pm.addPass(createLowerGraphMemoryPass());
  pm.addPass(createLowerGraphInvariantPass());
  pm.addPass(::mlir::createCanonicalizerPass());
}

void registerLoweringPasses() {
  registerLowerForallToThreadPass();
  registerLowerForToGraphPass();
  registerLowerGraphInvariantPass();
  registerLowerGraphMemoryPass();
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
