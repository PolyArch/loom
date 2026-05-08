// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-forall-to-thread       (module-level)
//     loom-lower-for-to-graph           (module-level)
//     --canonicalize                    (upstream)
//
// The forall-to-thread pass runs first so that the for-to-graph pass
// sees scf.for ops already inside dataflow.thread bodies.

#include "Frontend/Lowering/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace lowering {

void registerLowerForallToThreadPass();
void registerLowerForToGraphPass();

static void buildPipelineOnOpPassManager(::mlir::OpPassManager &pm) {
  pm.addPass(createLowerForallToThreadPass());
  pm.addPass(createLowerForToGraphPass());
  pm.addPass(::mlir::createCanonicalizerPass());
}

void registerLoweringPasses() {
  registerLowerForallToThreadPass();
  registerLowerForToGraphPass();
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
