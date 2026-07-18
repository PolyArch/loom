// Pipeline glue and pass-registry hooks for the SCF-to-DFG lowering
// passes. The standard pipeline runs:
//
//     loom-lower-for-to-graph           (module-level)
//
// `loom-lower-for-to-graph` owns the atomic publication transaction. It
// stages structured candidates, runs graph finalization on a scratch module,
// validates the native result, and publishes only the completed module.
//
// Thread ownership must already be present in the Structured Program
// Candidate. The independently registered forall pass only diagnoses raw
// implicit promotion requests.

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
