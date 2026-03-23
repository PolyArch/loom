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

// Compilation options
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

// Simulation options
static cl::opt<bool> EnableSim(
    "enable-sim",
    cl::desc("Enable multi-core simulation after compilation"),
    cl::init(true));

static cl::opt<uint64_t> MaxCycles(
    "max-cycles", cl::desc("Maximum simulation cycles"),
    cl::init(1000000));

static cl::opt<bool> EnableNoCContention(
    "noc-contention",
    cl::desc("Enable NoC contention modeling"),
    cl::init(true));

static cl::opt<bool> EnableTracing(
    "trace", cl::desc("Enable cycle-level tracing"), cl::init(false));

// RTL generation options
static cl::opt<bool> EnableRTL(
    "enable-rtl",
    cl::desc("Enable multi-core RTL generation"),
    cl::init(true));

static cl::opt<std::string> RtlSourceDir(
    "rtl-source-dir", cl::desc("Path to RTL source directory"),
    cl::init(""));

static cl::opt<std::string> FpIpProfile(
    "fp-ip-profile",
    cl::desc("Floating-point IP profile name"),
    cl::init(""));

static cl::opt<unsigned> MeshRows(
    "mesh-rows", cl::desc("NoC mesh rows"), cl::init(0));

static cl::opt<unsigned> MeshCols(
    "mesh-cols", cl::desc("NoC mesh cols"), cl::init(0));

// General
static cl::opt<bool> Verbose(
    "verbose", cl::desc("Enable verbose logging"), cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "Tapestry full pipeline "
                              "(compile + simulate + RTL)\n");

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

  // Select stages
  config.stages.push_back(PipelineStage::COMPILE);
  if (EnableSim)
    config.stages.push_back(PipelineStage::SIMULATE);
  if (EnableRTL)
    config.stages.push_back(PipelineStage::RTLGEN);

  // Benders options
  config.bendersOpts.maxIterations = MaxIterations;
  config.bendersOpts.costTighteningThreshold = CostThreshold;
  config.bendersOpts.perfectNoC = PerfectNoC;
  config.bendersOpts.verbose = Verbose;

  // Simulation config
  config.simConfig.maxGlobalCycles = MaxCycles;
  config.simConfig.enableNoCContention = EnableNoCContention;
  config.simConfig.enableTracing = EnableTracing;

  // RTL options
  config.rtlSourceDir = RtlSourceDir;
  config.svgenOpts.fpIpProfile = FpIpProfile;
  if (MeshRows > 0)
    config.svgenOpts.meshRows = MeshRows;
  if (MeshCols > 0)
    config.svgenOpts.meshCols = MeshCols;

  // Run full pipeline
  TapestryPipeline pipeline;
  TapestryPipelineResult result = pipeline.run(config, context);

  if (!result.success) {
    errs() << "tapestry-pipeline: FAILED\n";
    errs() << result.diagnostics << "\n";
    return 1;
  }

  // Print summary
  outs() << "tapestry-pipeline: SUCCESS\n";

  if (result.compilationResult.has_value()) {
    const auto &comp = result.compilationResult.value();
    outs() << "  [compile] Cores: " << comp.coreResults.size()
           << ", Iterations: " << comp.metrics.numBendersIterations
           << ", Time: " << comp.metrics.compilationTimeSec << " sec\n";
  }

  if (result.simResult.has_value()) {
    const auto &sim = result.simResult.value();
    outs() << "  [simulate] Total cycles: " << sim.totalGlobalCycles
           << ", NoC flits: " << sim.nocStats.totalFlitsTransferred << "\n";
  }

  if (result.rtlResult.has_value()) {
    const auto &rtl = result.rtlResult.value();
    outs() << "  [rtlgen] Files: " << rtl.allGeneratedFiles.size()
           << ", Top: " << rtl.systemTopFile << "\n";
  }

  outs() << "  Report: " << result.reportPath << "\n";
  return 0;
}
