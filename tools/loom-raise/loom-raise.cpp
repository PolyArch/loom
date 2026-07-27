// loom-raise: read an LLVM IR (.ll / .bc) file via parseIRFile, translate
// to an immutable Structured Program Candidate through the shared mechanical
// raising library, then emit its canonical SCF MLIR on stdout or to -o <file>.
// Every imported llvm.func is structured in place and keeps its complete ABI
// envelope.
// Selected SCF optimization decisions are outside this mechanical pipeline.
//
// CLI shape mirrors mlir-translate / mlir-opt:
//
//     loom-raise [-o output.mlir] [--allow-unregistered-dialects]
//                [--debug] [--print-after-all] input.ll
//
// stdin is read when input is "-" or the positional arg is missing.

#include "Frontend/Raising/StructuredRaising.h"

#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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

::llvm::cl::opt<std::string>
    inputFilename(::llvm::cl::Positional,
                  ::llvm::cl::desc("<input LLVM IR (.ll/.bc), or - for stdin>"),
                  ::llvm::cl::init("-"));

::llvm::cl::opt<std::string> outputFilename("o",
                                            ::llvm::cl::desc("Output filename"),
                                            ::llvm::cl::value_desc("filename"),
                                            ::llvm::cl::init("-"));

::llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialects",
    ::llvm::cl::desc("Allow operations from unregistered dialects in output"),
    ::llvm::cl::init(false));

::llvm::cl::opt<bool> verifyEach(
    "loom-verify-each",
    ::llvm::cl::desc("Run the verifier after each transformation pass"),
    ::llvm::cl::init(true));

std::unique_ptr<::llvm::Module> readLLVMModule(::llvm::LLVMContext &llvmContext,
                                               const std::string &filename) {
  ::llvm::SMDiagnostic err;
  if (filename == "-") {
    auto buffer = ::llvm::MemoryBuffer::getSTDIN();
    if (auto bufErr = buffer.getError()) {
      ::llvm::errs() << "loom-raise: cannot read stdin: " << bufErr.message()
                     << "\n";
      return nullptr;
    }
    auto module =
        ::llvm::parseIR((*buffer)->getMemBufferRef(), err, llvmContext);
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
      "pipeline (loom-llvm-cf-to-cf, loom-lift-cf-to-scf, "
      "loom-llvm-arith-to-arith, loom-normalize-lifted-scf-exit, "
      "loom-scf-while-to-for) to produce initial SCF MLIR. "
      "Selected SCF optimization decisions are outside this mechanical "
      "pipeline.\n");

  // Parse LLVM IR.
  ::llvm::LLVMContext llvmContext;
  std::unique_ptr<::llvm::Module> llvmModule =
      readLLVMModule(llvmContext, inputFilename);
  if (!llvmModule) {
    return 1;
  }
  auto candidate = loom::raising::raiseLlvmModuleToStructuredProgram(
      std::move(llvmModule), {allowUnregisteredDialects, verifyEach,
                              /*applyPassManagerCommandLineOptions=*/true});
  if (!candidate) {
    ::llvm::errs() << "loom-raise: " << ::llvm::toString(candidate.takeError())
                   << "\n";
    return 1;
  }

  // Emit.
  std::string errMsg;
  auto outputFile = ::mlir::openOutputFile(outputFilename, &errMsg);
  if (!outputFile) {
    ::llvm::errs() << "loom-raise: cannot open output: " << errMsg << "\n";
    return 1;
  }
  candidate->module()->print(outputFile->os());
  outputFile->os() << "\n";
  outputFile->keep();

  return 0;
}
