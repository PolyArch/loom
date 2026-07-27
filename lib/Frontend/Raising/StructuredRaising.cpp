#include "Frontend/Raising/StructuredRaising.h"

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_raising_invalid: " + message);
}

} // namespace

llvm::Expected<frontend::StructuredProgramCandidate>
raiseLlvmModuleToStructuredProgram(std::unique_ptr<llvm::Module> module,
                                   StructuredRaisingOptions options) {
  if (!module)
    return invalid("missing LLVM module");
  if (llvm::verifyModule(*module))
    return invalid("LLVM module failed verification");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(options.allowUnregisteredDialects);
  context.loadAllAvailableDialects();
  context.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                      mlir::math::MathDialect, mlir::memref::MemRefDialect,
                      mlir::scf::SCFDialect, mlir::ub::UBDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> raised =
      mlir::translateLLVMIRToModule(std::move(module), &context);
  if (!raised)
    return invalid("LLVM IR import failed");

  static const bool passesRegistered = [] {
    mlir::registerAllPasses();
    registerRaisingPasses();
    return true;
  }();
  (void)passesRegistered;
  mlir::PassManager pipeline(&context);
  pipeline.enableVerifier(options.verifyEach);
  if (options.applyPassManagerCommandLineOptions &&
      failed(mlir::applyPassManagerCLOptions(pipeline)))
    return invalid("cannot apply pass-manager command-line options");
  buildRaisingPipeline(pipeline);
  if (failed(pipeline.run(*raised)))
    return invalid("mechanical LLVM-to-SCF raising failed");

  return frontend::finalizeStructuredProgram(raised.get());
}

} // namespace loom::raising
