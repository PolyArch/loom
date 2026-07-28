// Pipeline glue and pass-registry hooks for the LLVM-to-SCF raising
// passes. The standard pipeline is:
//
//     loom-llvm-cf-to-cf
//     loom-lift-cf-to-scf
//     loom-llvm-arith-to-arith
//     loom-normalize-lifted-scf-exit
//     loom-deduplicate-scf-while-state
//     loom-scf-while-to-for
//
// Every pass walks callable regions in place. An imported llvm.func stays
// the sole callable and ABI owner of its LLVM function; nothing is copied
// into another dialect to obtain a pass wrapper.
//
// Pipeline ordering rationale:
//   * cf-to-cf runs first because the CFG-to-SCF transformation recognizes
//     cf branch structure, so LLVM branch terminators must already be in
//     their exact cf form.
//   * arith-to-arith runs after structuring so that CFG recovery decisions
//     are not confused by spurious arith.* ops in the comparison and
//     induction position.
//
// loom-materialize-fmuladd is registered but deliberately absent from this
// pipeline: choosing a fused or split multiply-add is candidate lineage, not
// a mechanical disposition.

#include "Frontend/Raising/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace loom {
namespace raising {

void registerLLVMCfToCfPass();
void registerLiftCFToSCFPass();
void registerLLVMArithToArithPass();
void registerNormalizeLiftedSCFExitPass();
void registerDeduplicateSCFWhileStatePass();
void registerSCFWhileToForPass();
void registerSCFForToForallPass();
void registerMaterializeFMulAddPass();

void registerRaisingPasses() {
  registerLLVMCfToCfPass();
  registerLiftCFToSCFPass();
  registerLLVMArithToArithPass();
  registerNormalizeLiftedSCFExitPass();
  registerDeduplicateSCFWhileStatePass();
  registerSCFWhileToForPass();
  registerSCFForToForallPass();
  registerMaterializeFMulAddPass();
}

void buildRaisingPipeline(::mlir::PassManager &pm) {
  pm.addPass(createLLVMCfToCfPass());
  pm.addPass(createLiftCFToSCFPass());
  pm.addPass(createLLVMArithToArithPass());
  // Canonicalize the lifted exit scaffold to its exact arith.cmpi condition.
  pm.addPass(createNormalizeLiftedSCFExitPass());
  pm.addPass(createDeduplicateSCFWhileStatePass());
  pm.addPass(createSCFWhileToForPass());
}

} // namespace raising
} // namespace loom
