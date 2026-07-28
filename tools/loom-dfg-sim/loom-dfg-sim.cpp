#include "Dataflow/IR/DataflowDialect.h"
#include "Simulator/DFGSimulator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
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
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

static llvm::cl::opt<std::string> inputPath(llvm::cl::Positional,
                                            llvm::cl::desc("<input mlir>"),
                                            llvm::cl::Required);

static llvm::cl::opt<std::string>
    graphName("graph", llvm::cl::desc("dataflow.graph symbol to run"),
              llvm::cl::Required);

static llvm::cl::opt<std::string>
    workloadName("workload",
                 llvm::cl::desc("workload name recorded in the report"));

static llvm::cl::list<std::string>
    runtimeArgs("arg", llvm::cl::desc("runtime argument as index=value"),
                llvm::cl::ZeroOrMore);

static llvm::cl::list<std::string> memrefArgs(
    "memref",
    llvm::cl::desc("memref fixture as index[:byte_offset]=value0,value1,..."),
    llvm::cl::ZeroOrMore);

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("DFG simulation report JSON"),
               llvm::cl::Required);

static llvm::cl::opt<std::uint64_t>
    maxEventSteps("max-event-steps", llvm::cl::desc("maximum event steps"),
                  llvm::cl::init(100000));

static llvm::cl::opt<std::uint64_t> invocations(
    "invocations",
    llvm::cl::desc("number of sequential graph protocol invocations"),
    llvm::cl::init(1));

static llvm::Expected<llvm::SmallVector<loom::sim::DFGRuntimeArg>>
parseRuntimeArgs() {
  llvm::SmallVector<loom::sim::DFGRuntimeArg> parsed;
  for (llvm::StringRef raw : runtimeArgs) {
    std::pair<llvm::StringRef, llvm::StringRef> split = raw.split('=');
    if (split.second.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "--arg expects index=value");
    unsigned index = 0;
    if (split.first.getAsInteger(10, index))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "--arg index must be unsigned");
    parsed.push_back({index, split.second.str()});
  }
  return parsed;
}

static llvm::Expected<llvm::SmallVector<loom::sim::DFGMemoryArg>>
parseMemoryArgs() {
  llvm::SmallVector<loom::sim::DFGMemoryArg> parsed;
  for (llvm::StringRef raw : memrefArgs) {
    std::pair<llvm::StringRef, llvm::StringRef> split = raw.split('=');
    if (split.second.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "--memref expects index=values");
    std::int64_t byteOffset = 0;
    llvm::StringRef indexPart = split.first;
    if (split.first.contains(':')) {
      std::pair<llvm::StringRef, llvm::StringRef> offsetSplit =
          split.first.split(':');
      indexPart = offsetSplit.first;
      if (offsetSplit.second.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "--memref byte offset must be a non-negative integer");
      if (offsetSplit.second.getAsInteger(10, byteOffset) || byteOffset < 0)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "--memref byte offset must be a non-negative integer");
    }
    unsigned index = 0;
    if (indexPart.getAsInteger(10, index))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "--memref index must be unsigned");
    parsed.push_back({index, byteOffset, split.second.str()});
  }
  return parsed;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "loom-dfg-sim: execute a pure dataflow.graph token model\n");

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::ub::UBDialect>();

  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputPath);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "error: could not read " << inputPath << ": "
                 << ec.message() << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "error: could not parse input MLIR\n";
    return 1;
  }

  auto argsOrErr = parseRuntimeArgs();
  if (!argsOrErr) {
    llvm::errs() << "error: " << llvm::toString(argsOrErr.takeError()) << "\n";
    return 1;
  }
  auto memoriesOrErr = parseMemoryArgs();
  if (!memoriesOrErr) {
    llvm::errs() << "error: " << llvm::toString(memoriesOrErr.takeError())
                 << "\n";
    return 1;
  }
  loom::sim::DFGSimulationOptions options;
  options.graphName = graphName;
  options.workloadName = workloadName;
  options.args = std::move(*argsOrErr);
  options.memories = std::move(*memoriesOrErr);
  options.invocations = invocations;
  options.maxEventSteps = maxEventSteps;

  auto reportOrErr = loom::sim::simulateDataflowGraph(*module, options);
  if (!reportOrErr) {
    llvm::errs() << "error: " << llvm::toString(reportOrErr.takeError())
                 << "\n";
    return 1;
  }

  if (llvm::Error err =
          loom::sim::writeDFGSimulationReportJson(outputPath, *reportOrErr)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
