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

static cl::opt<uint64_t> MaxCycles(
    "max-cycles", cl::desc("Maximum simulation cycles"),
    cl::init(1000000));

static cl::opt<bool> EnableNoCContention(
    "noc-contention",
    cl::desc("Enable NoC contention modeling"),
    cl::init(true));

static cl::opt<bool> EnableTracing(
    "trace", cl::desc("Enable cycle-level tracing"), cl::init(false));

static cl::opt<bool> Verbose(
    "verbose", cl::desc("Enable verbose logging"), cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "Tapestry multi-core simulation\n");

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

  // Configure pipeline: compile then simulate
  TapestryPipelineConfig config;
  config.tdgPath = TDGPath;
  config.systemArchPath = SystemArchPath;
  config.outputDir = OutputDir;
  config.verbose = Verbose;
  config.stages = {PipelineStage::COMPILE, PipelineStage::SIMULATE};

  // Simulation config
  config.simConfig.maxGlobalCycles = MaxCycles;
  config.simConfig.enableNoCContention = EnableNoCContention;
  config.simConfig.enableTracing = EnableTracing;

  // Run pipeline
  TapestryPipeline pipeline;
  TapestryPipelineResult result = pipeline.run(config, context);

  if (!result.success) {
    errs() << "tapestry-simulate: FAILED\n";
    errs() << result.diagnostics << "\n";
    return 1;
  }

  outs() << "tapestry-simulate: SUCCESS\n";
  if (result.simResult.has_value()) {
    const auto &sim = result.simResult.value();
    outs() << "  Total cycles: " << sim.totalGlobalCycles << "\n";
    outs() << "  Cores simulated: " << sim.coreResults.size() << "\n";
    outs() << "  NoC flits: " << sim.nocStats.totalFlitsTransferred << "\n";
  }
  outs() << "  Report: " << result.reportPath << "\n";

  return 0;
}
