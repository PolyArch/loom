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

static cl::opt<bool> Verbose(
    "verbose", cl::desc("Enable verbose logging"), cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "Tapestry multi-core RTL generation\n");

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

  // Configure pipeline: compile then generate RTL
  TapestryPipelineConfig config;
  config.tdgPath = TDGPath;
  config.systemArchPath = SystemArchPath;
  config.outputDir = OutputDir;
  config.verbose = Verbose;
  config.stages = {PipelineStage::COMPILE, PipelineStage::RTLGEN};

  // RTL options
  config.rtlSourceDir = RtlSourceDir;
  config.svgenOpts.fpIpProfile = FpIpProfile;
  if (MeshRows > 0)
    config.svgenOpts.meshRows = MeshRows;
  if (MeshCols > 0)
    config.svgenOpts.meshCols = MeshCols;

  // Run pipeline
  TapestryPipeline pipeline;
  TapestryPipelineResult result = pipeline.run(config, context);

  if (!result.success) {
    errs() << "tapestry-rtlgen: FAILED\n";
    errs() << result.diagnostics << "\n";
    return 1;
  }

  outs() << "tapestry-rtlgen: SUCCESS\n";
  if (result.rtlResult.has_value()) {
    const auto &rtl = result.rtlResult.value();
    outs() << "  System top: " << rtl.systemTopFile << "\n";
    outs() << "  Files generated: " << rtl.allGeneratedFiles.size() << "\n";
  }
  outs() << "  Report: " << result.reportPath << "\n";

  return 0;
}
