#include "loom/SystemCompiler/TapestryPipeline.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace loom;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static cl::opt<std::string> TDGPath(
    "tdg", cl::desc("Path to TDG MLIR input file"), cl::Required);

static cl::opt<std::string> SystemArchPath(
    "system-arch", cl::desc("Path to system architecture JSON file"),
    cl::Required);

static cl::opt<std::string> OutputDir(
    "o", cl::desc("Output directory"), cl::init("tapestry-output"));

static cl::opt<unsigned> MaxIterations(
    "max-benders-iter",
    cl::desc("Maximum Benders decomposition iterations"),
    cl::init(10));

static cl::opt<double> CostThreshold(
    "cost-threshold",
    cl::desc("Minimum relative cost improvement for convergence"),
    cl::init(0.01));

static cl::opt<bool> PerfectNoC(
    "perfect-noc",
    cl::desc("Assume perfect NoC (zero transfer cost)"),
    cl::init(false));

static cl::opt<bool> Verbose(
    "verbose", cl::desc("Enable verbose logging"), cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "Tapestry multi-core compilation\n");

  // Register dialects
  DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect>();
  registry.insert<fabric::FabricDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Configure pipeline
  TapestryPipelineConfig config;
  config.tdgPath = TDGPath;
  config.systemArchPath = SystemArchPath;
  config.outputDir = OutputDir;
  config.verbose = Verbose;
  config.stages = {PipelineStage::COMPILE};

  // Benders options
  config.bendersOpts.maxIterations = MaxIterations;
  config.bendersOpts.costTighteningThreshold = CostThreshold;
  config.bendersOpts.perfectNoC = PerfectNoC;
  config.bendersOpts.verbose = Verbose;

  // Run compilation
  TapestryPipeline pipeline;
  TapestryPipelineResult result = pipeline.run(config, context);

  if (!result.success) {
    errs() << "tapestry-compile: FAILED\n";
    errs() << result.diagnostics << "\n";
    return 1;
  }

  outs() << "tapestry-compile: SUCCESS\n";
  if (result.compilationResult.has_value()) {
    const auto &comp = result.compilationResult.value();
    outs() << "  Cores: " << comp.coreResults.size() << "\n";
    outs() << "  Iterations: " << comp.metrics.numBendersIterations << "\n";
    outs() << "  Compilation time: " << comp.metrics.compilationTimeSec
           << " sec\n";
    outs() << "  Report: " << result.reportPath << "\n";
  }

  return 0;
}
