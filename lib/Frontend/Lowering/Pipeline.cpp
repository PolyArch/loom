// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-forall-to-thread       (module-level)
//     loom-lower-for-to-graph           (module-level)
//     --canonicalize                    (upstream)
//     loom-lower-known-library-calls    (module-level)
//     loom-lower-graph-memory           (module-level)
//     loom-lower-graph-constants        (module-level)
//     --canonicalize                    (upstream)
//
// The forall-to-thread pass runs first so that the for-to-graph pass
// sees scf.for ops already inside dataflow.thread bodies. The
// canonicalizer between graph extraction and graph-region lowering cleans up
// trivial dead bridge values before the recursive owner walks each graph.
// graph-memory owns memory normalization, structured control, values, and
// per-partition frontiers together. graph-constants then promotes remaining
// top-level literals; nested literals were already gated by their recursive
// execution context. The closing canonicalizer removes dead bridge and
// projection values. graph-memory also constructs graph.return's explicit
// retirement frontier from structural execution, value publication, and final
// per-partition read frontiers.

#include "Frontend/Lowering/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace lowering {

void registerLowerForallToThreadPass();
void registerLowerForToGraphPass();
void registerLowerGraphConstantsPass();
void registerLowerKnownLibraryCallsPass();
void registerLowerGraphMemoryPass();

static void buildPipelineOnOpPassManager(::mlir::OpPassManager &pm) {
  pm.addPass(createLowerForallToThreadPass());
  pm.addPass(createLowerForToGraphPass());
  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(createLowerKnownLibraryCallsPass());
  pm.addPass(createLowerGraphMemoryPass());
  pm.addPass(createLowerGraphConstantsPass());
  pm.addPass(::mlir::createCanonicalizerPass());
}

void registerLoweringPasses() {
  registerLowerForallToThreadPass();
  registerLowerForToGraphPass();
  registerLowerGraphConstantsPass();
  registerLowerKnownLibraryCallsPass();
  registerLowerGraphMemoryPass();
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
