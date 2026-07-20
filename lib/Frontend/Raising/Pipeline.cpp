// Pipeline glue and pass-registry hooks for the LLVM-to-SCF raising
// passes. The standard pipeline is:
//
//     loom-llvm-func-to-func              (module-level)
//     loom-llvm-cf-to-cf                  (nested under func.func)
//     --lift-cf-to-scf                    (upstream, nested under func.func)
//     loom-llvm-arith-to-arith            (nested under func.func)
//     --canonicalize                      (upstream)
//     loom-scf-while-to-for
//     --canonicalize
//
// Pipeline ordering rationale:
//   * func-to-func runs FIRST so that llvm.func ops with builtin-only
//     signatures become func.func; aggregate-signature llvm.func ops
//     stay as llvm.func with their bodies in pristine LLVM form.
//   * cf-to-cf and arith-to-arith are then nested under func.func so
//     they ONLY rewrite ops inside lifted functions. Aggregate-signature
//     llvm.func ops stay fully llvm-shaped (their bodies are not
//     half-rewritten), which is the documented contract.
//   * --lift-cf-to-scf walks func.func ops only and matches on cf.*
//     branches (see llvm-project/mlir/lib/Conversion/ControlFlowToSCF/
//     ControlFlowToSCF.cpp), so we must convert llvm.br/llvm.cond_br
//     before invoking it.
//   * The arith rewriting pass runs after --lift-cf-to-scf so cf-to-scf
//     lifting decisions have not been confused by spurious arith.* ops
//     in the comparison/induction position.

#include "Frontend/Raising/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace loom {
namespace raising {

void registerLLVMCfToCfPass();
void registerLLVMArithToArithPass();
void registerLLVMFuncToFuncPass();
void registerSCFWhileToForPass();
void registerSCFForToForallPass();

void registerRaisingPasses() {
  registerLLVMCfToCfPass();
  registerLLVMFuncToFuncPass();
  registerLLVMArithToArithPass();
  registerSCFWhileToForPass();
  registerSCFForToForallPass();
}

void buildRaisingPipeline(::mlir::PassManager &pm) {
  // First raise lift-able llvm.func ops to func.func; aggregate-
  // signature llvm.func ops are skipped and stay in pristine LLVM form.
  pm.addPass(createLLVMFuncToFuncPass());
  // Then run cf-to-cf and (after --lift-cf-to-scf) arith-to-arith
  // strictly inside func.func bodies. This guarantees skipped llvm.func
  // ops keep their bodies untouched, which is the contract callers and
  // downstream layers rely on.
  pm.nest<::mlir::func::FuncOp>().addPass(createLLVMCfToCfPass());
  pm.nest<::mlir::func::FuncOp>().addPass(
      ::mlir::createLiftControlFlowToSCFPass());
  pm.nest<::mlir::func::FuncOp>().addPass(createLLVMArithToArithPass());
  // Fold the scf.if exit-flag pattern lift-cf-to-scf inserts so the
  // counted-loop while-to-for uplift below can see a clean
  // `arith.cmpi <pred>` condition without an interposed scf.if.
  pm.addPass(::mlir::createCanonicalizerPass());
  pm.addPass(createSCFWhileToForPass());
  // Normalize the recovered structured form. Parallelization and other
  // selected optimization decisions belong to the later SCF pipeline.
  pm.addPass(::mlir::createCanonicalizerPass());
}

} // namespace raising
} // namespace loom
