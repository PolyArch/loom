// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-forall-to-thread       (module-level)
//     loom-lower-for-to-graph           (module-level)
//
// `loom-lower-for-to-graph` owns the atomic publication transaction. It
// stages structured candidates, runs graph finalization on a scratch module,
// validates the native result, and publishes only the completed module.
//
// The forall-to-thread pass runs first so that the for-to-graph pass
// sees scf.for ops already inside dataflow.thread bodies. The
// The remaining lowering passes stay independently registered for focused
// diagnostics and tests, but the standard pipeline does not rerun them after
// publication.

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
