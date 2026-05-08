// Pipeline glue and pass-registry hooks for the LLVM-to-SCF raising
// passes. The standard pipeline is:
//
//     loom-llvm-cf-to-cf
//     loom-llvm-func-to-func
//     --lift-cf-to-scf       (upstream)
//     loom-llvm-arith-to-arith
//     --canonicalize         (upstream)
//
// The lift-cf-to-scf pass walks func.func ops only and matches on cf.*
// branches (see llvm-project/mlir/lib/Conversion/ControlFlowToSCF/
// ControlFlowToSCF.cpp), so we must convert llvm.br/llvm.cond_br first
// AND raise llvm.func to func.func before invoking it. The arith
// rewriting pass runs last so cf-to-scf lifting decisions have not been
// confused by spurious arith.* ops in the comparison/induction position.

#include "Frontend/Raising/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace raising {

void registerLLVMCfToCfPass();
void registerLLVMArithToArithPass();
void registerLLVMFuncToFuncPass();
void registerSCFWhileToForPass();

void registerRaisingPasses() {
  registerLLVMCfToCfPass();
  registerLLVMFuncToFuncPass();
  registerLLVMArithToArithPass();
  registerSCFWhileToForPass();
}

void buildRaisingPipeline(::mlir::PassManager &pm) {
  pm.addPass(createLLVMCfToCfPass());
  pm.addPass(createLLVMFuncToFuncPass());
  pm.addPass(::mlir::createLiftControlFlowToSCFPass());
  pm.addPass(createLLVMArithToArithPass());
  // Fold the scf.if exit-flag pattern lift-cf-to-scf inserts so the
  // counted-loop while-to-for uplift below can see a clean
  // `arith.cmpi <pred>` condition without an interposed scf.if.
  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(createSCFWhileToForPass());
  pm.addPass(::mlir::createCanonicalizerPass());
}

} // namespace raising
} // namespace loom
