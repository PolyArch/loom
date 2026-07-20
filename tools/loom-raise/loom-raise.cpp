// loom-raise: read an LLVM IR (.ll / .bc) file via parseIRFile, translate
// to MLIR using the upstream translateLLVMIRToModule, then run the
// standard Loom raising pipeline (func-to-func -> cf-to-cf ->
// --lift-cf-to-scf -> arith-to-arith -> --canonicalize -> while-to-for
// -> --canonicalize) and emit initial SCF MLIR on stdout or to -o <file>.
// Selected SCF optimization decisions are outside this mechanical pipeline.
//
// CLI shape mirrors mlir-translate / mlir-opt:
//
//     loom-raise [-o output.mlir] [--allow-unregistered-dialects]
//                [--debug] [--print-after-all] input.ll
//
// stdin is read when input is "-" or the positional arg is missing.

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace {

::llvm::cl::opt<std::string> inputFilename(
    ::llvm::cl::Positional,
    ::llvm::cl::desc("<input LLVM IR (.ll/.bc), or - for stdin>"),
    ::llvm::cl::init("-"));

::llvm::cl::opt<std::string> outputFilename(
    "o", ::llvm::cl::desc("Output filename"),
    ::llvm::cl::value_desc("filename"), ::llvm::cl::init("-"));

::llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialects",
    ::llvm::cl::desc("Allow operations from unregistered dialects in output"),
    ::llvm::cl::init(false));

::llvm::cl::opt<bool> verifyEach(
    "loom-verify-each",
    ::llvm::cl::desc("Run the verifier after each transformation pass"),
    ::llvm::cl::init(true));

std::unique_ptr<::llvm::Module>
readLLVMModule(::llvm::LLVMContext &llvmContext, const std::string &filename) {
  ::llvm::SMDiagnostic err;
  if (filename == "-") {
    auto buffer = ::llvm::MemoryBuffer::getSTDIN();
    if (auto bufErr = buffer.getError()) {
      ::llvm::errs() << "loom-raise: cannot read stdin: " << bufErr.message()
                     << "\n";
      return nullptr;
    }
    auto module = ::llvm::parseIR((*buffer)->getMemBufferRef(), err,
                                  llvmContext);
    if (!module) {
      err.print("loom-raise", ::llvm::errs());
      return nullptr;
    }
    return module;
  }
  auto module = ::llvm::parseIRFile(filename, err, llvmContext);
  if (!module) {
    err.print("loom-raise", ::llvm::errs());
    return nullptr;
  }
  return module;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::InitLLVM y(argc, argv);
  ::mlir::registerAsmPrinterCLOptions();
  ::mlir::registerMLIRContextCLOptions();
  ::mlir::registerPassManagerCLOptions();
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Loom LLVM IR -> SCF MLIR raising driver\n"
      "Reads an LLVM IR (.ll / .bc) file, translates it to MLIR via the "
      "upstream LLVMIR-import path, and runs the standard Loom raising "
      "pipeline (loom-llvm-func-to-func, loom-llvm-cf-to-cf, "
      "--lift-cf-to-scf, loom-llvm-arith-to-arith, --canonicalize, "
      "loom-scf-while-to-for, --canonicalize) to produce initial SCF MLIR. "
      "Selected SCF optimization decisions are outside this mechanical "
      "pipeline.\n");

  // Parse LLVM IR.
  ::llvm::LLVMContext llvmContext;
  std::unique_ptr<::llvm::Module> llvmModule =
      readLLVMModule(llvmContext, inputFilename);
  if (!llvmModule) {
    return 1;
  }
  if (::llvm::verifyModule(*llvmModule, &::llvm::errs())) {
    ::llvm::errs() << "loom-raise: input LLVM module failed verifier\n";
    return 1;
  }

  // Set up MLIR.
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::mlir::registerAllFromLLVMIRTranslations(registry);

  ::mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  context.loadAllAvailableDialects();
  // Belt-and-braces: ensure the dialects we know we will produce are
  // loaded even when the cf/scf dialects were not pulled in via a
  // dependent-dialect chain during translation.
  context.loadDialect<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect,
                      ::mlir::func::FuncDialect, ::mlir::LLVM::LLVMDialect,
                      ::mlir::math::MathDialect,
                      ::mlir::memref::MemRefDialect, ::mlir::scf::SCFDialect,
                      ::mlir::ub::UBDialect>();

  // Translate.
  ::mlir::OwningOpRef<::mlir::ModuleOp> module =
      ::mlir::translateLLVMIRToModule(std::move(llvmModule), &context);
  if (!module) {
    ::llvm::errs() << "loom-raise: translateLLVMIRToModule failed\n";
    return 1;
  }

  // Run the raising pipeline.
  ::mlir::registerAllPasses();
  loom::raising::registerRaisingPasses();

  ::mlir::PassManager pm(&context);
  pm.enableVerifier(verifyEach);
  if (failed(::mlir::applyPassManagerCLOptions(pm))) {
    ::llvm::errs() << "loom-raise: failed to apply pass-manager CLI options\n";
    return 1;
  }
  loom::raising::buildRaisingPipeline(pm);

  if (failed(pm.run(*module))) {
    ::llvm::errs() << "loom-raise: pipeline failed\n";
    return 1;
  }

  // Emit.
  std::string errMsg;
  auto outputFile = ::mlir::openOutputFile(outputFilename, &errMsg);
  if (!outputFile) {
    ::llvm::errs() << "loom-raise: cannot open output: " << errMsg << "\n";
    return 1;
  }
  module->print(outputFile->os());
  outputFile->os() << "\n";
  outputFile->keep();

  return 0;
}
