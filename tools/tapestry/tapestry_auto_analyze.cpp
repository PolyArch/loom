//===-- tapestry_auto_analyze.cpp - Auto-analyze CLI tool -------------------===//
//
// Standalone CLI entry point for the auto-analyze pipeline:
//   C/C++ source -> autoAnalyze() -> buildTaskGraphFromAnalysis()
//   -> emitTDG() -> [optional: TDGToSSGBuilder -> SSG summary]
//
//===----------------------------------------------------------------------===//

#include "tapestry/auto_analyze.h"
#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/SystemCompiler/TDGToSSGBuilder.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace tapestry;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    SourcePath("source", cl::desc("Path to C/C++ source file"), cl::Required);

static cl::opt<std::string>
    EntryFunc("entry", cl::desc("Entry function name"), cl::Required);

static cl::opt<std::string>
    OutputPath("output", cl::desc("Output MLIR file path"),
               cl::init("auto_tdg.mlir"));

static cl::opt<bool>
    EmitSSG("emit-ssg", cl::desc("Build SSG and print summary"),
            cl::init(false));

static cl::opt<bool>
    Verbose("verbose", cl::desc("Enable verbose output"), cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "Tapestry auto-analyze pipeline tool\n");

  // Run auto-analysis on the source file.
  AutoAnalyzeOptions opts;
  opts.verbose = Verbose;

  if (Verbose)
    outs() << "Analyzing " << SourcePath << " entry=" << EntryFunc << "\n";

  AutoAnalyzeResult result = autoAnalyze(SourcePath, EntryFunc, opts);

  if (!result.success) {
    errs() << "Auto-analysis failed: " << result.diagnostics << "\n";
    return 1;
  }

  outs() << "Kernels: " << result.numKernels() << "\n";
  outs() << "Edges: " << result.numEdges() << "\n";

  if (Verbose)
    result.dump(outs());

  // Convert AutoAnalyzeResult to TaskGraph.
  TaskGraph tg = buildTaskGraphFromAnalysis(result);

  if (Verbose) {
    outs() << "\nTaskGraph:\n";
    tg.dump();
  }

  // Emit TDG MLIR.
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();

  auto tdgModule = emitTDG(tg, ctx);
  if (!tdgModule) {
    errs() << "Failed to emit TDG MLIR\n";
    return 1;
  }

  if (writeTDGToFile(*tdgModule, OutputPath))
    outs() << "TDG MLIR written to " << OutputPath << "\n";
  else
    errs() << "Warning: could not write TDG MLIR to " << OutputPath << "\n";

  // Optionally build SSG and print summary.
  if (EmitSSG) {
    // For SSG construction, we need DFG modules. In the full pipeline,
    // lowerKernelsToDFG() would produce these. Here we build with an
    // empty DFG map to show the SSG topology (profiles will be empty).
    std::map<std::string, mlir::ModuleOp> dfgModules;

    loom::TDGToSSGBuilder ssgBuilder;
    loom::SSG ssg = ssgBuilder.build(*tdgModule, dfgModules, ctx);

    outs() << "\nSSG Summary:\n";
    outs() << "  Nodes: " << ssg.numNodes() << "\n";
    outs() << "  Edges: " << ssg.numEdges() << "\n";

    for (const auto &node : ssg.nodes()) {
      outs() << "  Kernel: " << node.name << " (type=" << node.kernelType
             << ", variants=" << node.variantSet.size()
             << ", hasDFG=" << (node.hasDFG ? "yes" : "no") << ")\n";
    }
    for (const auto &edge : ssg.edges()) {
      outs() << "  Edge: " << edge.producerName << " -> " << edge.consumerName
             << " (volume=" << edge.dataVolume
             << ", type=" << edge.dataTypeName << ")\n";
    }
  }

  return 0;
}
