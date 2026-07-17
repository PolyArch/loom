// loom-lower: read SCF-shaped MLIR (.mlir) via mlir::parseSourceFile,
// run the standard Loom SCF-to-DFG lowering pipeline in-process via
// PassManager, validate the finalized graphs, and emit the resulting
// DFG-shaped MLIR text on stdout or to -o <file>.
//
// CLI shape mirrors loom-raise / mlir-translate:
//
//     loom-lower [-o output.mlir] [--allow-unregistered-dialects]
//                [--debug] [--print-after-all] input.mlir
//
// stdin is read when input is "-" or the positional arg is missing.
// No subprocess to upstream mlir-opt / mlir-translate is invoked --
// the lowering passes run inside this binary's PassManager.

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Fabric/IR/FabricDialect.h"
#include "Frontend/Lowering/Passes.h"

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
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
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
                  ::llvm::cl::desc("<input SCF MLIR (.mlir), or - for stdin>"),
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

} // namespace

int main(int argc, char **argv) {
  ::llvm::InitLLVM y(argc, argv);
  ::mlir::registerAsmPrinterCLOptions();
  ::mlir::registerMLIRContextCLOptions();
  ::mlir::registerPassManagerCLOptions();
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Loom SCF MLIR -> DFG MLIR lowering driver\n"
      "Reads an SCF-shaped MLIR file (typically the output of loom-raise) "
      "and runs the standard Loom SCF-to-DFG lowering pipeline "
      "(loom-lower-forall-to-thread, loom-lower-for-to-graph, "
      "--canonicalize) in-process via PassManager.\n");

  // Set up MLIR.
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  registry.insert<::fabric::FabricDialect, ::dataflow::DataflowDialect>();

  ::mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  context.loadAllAvailableDialects();
  context
      .loadDialect<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect,
                   ::mlir::func::FuncDialect, ::mlir::LLVM::LLVMDialect,
                   ::mlir::math::MathDialect, ::mlir::memref::MemRefDialect,
                   ::mlir::scf::SCFDialect, ::mlir::ub::UBDialect,
                   ::dataflow::DataflowDialect, ::fabric::FabricDialect>();

  // Parse input.
  std::string errMsg;
  std::unique_ptr<::llvm::MemoryBuffer> buffer;
  if (inputFilename == "-") {
    auto bufOrErr = ::llvm::MemoryBuffer::getSTDIN();
    if (auto err = bufOrErr.getError()) {
      ::llvm::errs() << "loom-lower: cannot read stdin: " << err.message()
                     << "\n";
      return 1;
    }
    buffer = std::move(*bufOrErr);
  } else {
    auto bufOrErr = ::llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (auto err = bufOrErr.getError()) {
      ::llvm::errs() << "loom-lower: cannot read " << inputFilename << ": "
                     << err.message() << "\n";
      return 1;
    }
    buffer = std::move(*bufOrErr);
  }

  ::llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), ::llvm::SMLoc());
  ::mlir::OwningOpRef<::mlir::ModuleOp> module =
      ::mlir::parseSourceFile<::mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    ::llvm::errs() << "loom-lower: failed to parse input MLIR\n";
    return 1;
  }

  // Run the lowering pipeline.
  ::mlir::registerAllPasses();
  loom::lowering::registerLoweringPasses();

  ::mlir::PassManager pm(&context);
  pm.enableVerifier(verifyEach);
  if (failed(::mlir::applyPassManagerCLOptions(pm))) {
    ::llvm::errs() << "loom-lower: failed to apply pass-manager CLI options\n";
    return 1;
  }
  loom::lowering::buildLoweringPipeline(pm);

  if (failed(pm.run(*module))) {
    ::llvm::errs() << "loom-lower: pipeline failed\n";
    return 1;
  }

  if (auto error = ::dataflow::validateFinalizedProgram(*module)) {
    ::llvm::errs() << "loom-lower: final Dataflow validation failed: "
                   << ::llvm::toString(std::move(error)) << "\n";
    return 1;
  }

  // Emit.
  auto outputFile = ::mlir::openOutputFile(outputFilename, &errMsg);
  if (!outputFile) {
    ::llvm::errs() << "loom-lower: cannot open output: " << errMsg << "\n";
    return 1;
  }
  module->print(outputFile->os());
  outputFile->os() << "\n";
  outputFile->keep();
  return 0;
}
