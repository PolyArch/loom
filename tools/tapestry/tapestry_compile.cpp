#include "loom/SystemCompiler/TapestryPipeline.h"
#include "tapestry/auto_analyze.h"

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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace loom;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

// Input mode: either --tdg (pre-built TDG MLIR) or --auto-tdg (C source)
static cl::opt<std::string> TDGPath(
    "tdg", cl::desc("Path to TDG MLIR input file (existing mode)"),
    cl::init(""));

static cl::opt<std::string> AutoTDGPath(
    "auto-tdg",
    cl::desc("Path to C/C++ source for automatic TDG construction"),
    cl::init(""));

static cl::opt<std::string> EntryFunc(
    "entry",
    cl::desc("Entry function name for auto-tdg mode"),
    cl::init(""));

static cl::opt<std::string> SystemArchPath(
    "system-arch", cl::desc("Path to system architecture JSON file"),
    cl::init(""));

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

static cl::opt<unsigned> MaxKernels(
    "max-kernels",
    cl::desc("Maximum kernels to extract in auto-tdg mode"),
    cl::init(32));

static cl::opt<bool> AnalyzeOnly(
    "analyze-only",
    cl::desc("Only run auto-analysis, do not compile (auto-tdg mode)"),
    cl::init(false));

static cl::opt<bool> Verbose(
    "verbose", cl::desc("Enable verbose logging"), cl::init(false));

//===----------------------------------------------------------------------===//
// Auto-analyze mode: write analysis results to JSON report
//===----------------------------------------------------------------------===//

static int writeAnalysisReport(const tapestry::AutoAnalyzeResult &result,
                               const std::string &outputDir) {
  // Ensure output directory exists
  if (auto ec = sys::fs::create_directories(outputDir)) {
    errs() << "tapestry-compile: cannot create output directory: "
           << ec.message() << "\n";
    return 1;
  }

  // Build JSON report
  json::Object report;
  report["source"] = result.sourcePath;
  report["entry"] = result.entryFunc;
  report["success"] = result.success;
  report["num_kernels"] = result.numKernels();
  report["num_edges"] = result.numEdges();

  if (!result.diagnostics.empty())
    report["diagnostics"] = result.diagnostics;

  // Kernels array
  json::Array kernels;
  for (const auto &binding : result.callBindings) {
    json::Object kernel;
    kernel["name"] = binding.kernelName;
    kernel["call_order"] = binding.callOrder;

    switch (binding.target) {
    case tapestry::KernelTarget::CGRA:
      kernel["target"] = "CGRA";
      break;
    case tapestry::KernelTarget::HOST:
      kernel["target"] = "HOST";
      break;
    case tapestry::KernelTarget::AUTO:
      kernel["target"] = "AUTO";
      break;
    }

    json::Array args;
    for (const auto &arg : binding.argNames)
      args.push_back(arg);
    kernel["args"] = std::move(args);

    kernels.push_back(std::move(kernel));
  }
  report["kernels"] = std::move(kernels);

  // Edges array
  json::Array edges;
  for (const auto &edge : result.edges) {
    json::Object edgeObj;
    edgeObj["producer"] = result.callBindings[edge.producerIndex].kernelName;
    edgeObj["consumer"] = result.callBindings[edge.consumerIndex].kernelName;
    edgeObj["data_type"] = edge.dependency.dataType;
    edgeObj["ordering"] =
        (edge.ordering == loom::Ordering::FIFO ? "FIFO" : "UNORDERED");
    edgeObj["sequential"] = edge.dependency.isSequential;

    if (edge.dependency.elementCount.has_value())
      edgeObj["element_count"] =
          static_cast<int64_t>(edge.dependency.elementCount.value());
    if (!edge.dependency.sharedArgName.empty())
      edgeObj["shared_arg"] = edge.dependency.sharedArgName;

    edges.push_back(std::move(edgeObj));
  }
  report["edges"] = std::move(edges);

  // Write to file
  SmallString<256> reportPath;
  sys::path::append(reportPath, outputDir, "report.json");

  std::error_code ec;
  raw_fd_ostream os(reportPath, ec);
  if (ec) {
    errs() << "tapestry-compile: cannot write report: " << ec.message()
           << "\n";
    return 1;
  }

  os << json::Value(std::move(report));
  os << "\n";
  outs() << "tapestry-compile: report written to " << reportPath << "\n";
  return 0;
}

//===----------------------------------------------------------------------===//
// Auto-TDG mode
//===----------------------------------------------------------------------===//

static int runAutoTDGMode() {
  if (EntryFunc.empty()) {
    errs() << "tapestry-compile: --entry is required with --auto-tdg\n";
    return 1;
  }

  outs() << "tapestry-compile: auto-analyzing " << AutoTDGPath.getValue()
         << " (entry: " << EntryFunc.getValue() << ")\n";

  tapestry::AutoAnalyzeOptions opts;
  opts.maxKernels = MaxKernels;
  opts.verbose = Verbose;

  auto result =
      tapestry::autoAnalyze(AutoTDGPath, EntryFunc, opts);

  if (!result.success) {
    errs() << "tapestry-compile: auto-analysis FAILED\n";
    if (!result.diagnostics.empty())
      errs() << "  " << result.diagnostics << "\n";
    return 1;
  }

  outs() << "tapestry-compile: auto-analysis SUCCESS\n";
  outs() << "  Kernels: " << result.numKernels() << "\n";
  outs() << "  Edges: " << result.numEdges() << "\n";

  // Write analysis report to output directory
  int reportResult = writeAnalysisReport(result, OutputDir);
  if (reportResult != 0)
    return reportResult;

  if (AnalyzeOnly) {
    outs() << "tapestry-compile: analyze-only mode, skipping compilation\n";
    result.dump(outs());
    return 0;
  }

  // If system-arch is provided, run the full compilation pipeline
  if (!SystemArchPath.empty()) {
    // Register dialects for the pipeline
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

    // Configure pipeline using the auto-analyzed source
    TapestryPipelineConfig config;
    config.tdgPath = AutoTDGPath; // Pipeline will re-compile from source
    config.systemArchPath = SystemArchPath;
    config.outputDir = OutputDir;
    config.verbose = Verbose;
    config.stages = {PipelineStage::COMPILE};

    config.bendersOpts.maxIterations = MaxIterations;
    config.bendersOpts.costTighteningThreshold = CostThreshold;
    config.bendersOpts.perfectNoC = PerfectNoC;
    config.bendersOpts.verbose = Verbose;

    TapestryPipeline pipeline;
    TapestryPipelineResult pipelineResult = pipeline.run(config, context);

    if (!pipelineResult.success) {
      errs() << "tapestry-compile: compilation FAILED\n";
      errs() << pipelineResult.diagnostics << "\n";
      return 1;
    }

    outs() << "tapestry-compile: compilation SUCCESS\n";
    if (pipelineResult.compilationResult.has_value()) {
      const auto &comp = pipelineResult.compilationResult.value();
      outs() << "  Cores: " << comp.coreResults.size() << "\n";
      outs() << "  Iterations: " << comp.metrics.numBendersIterations << "\n";
      outs() << "  Compilation time: " << comp.metrics.compilationTimeSec
             << " sec\n";
      outs() << "  Report: " << pipelineResult.reportPath << "\n";
    }
  } else {
    outs() << "tapestry-compile: no --system-arch provided, "
              "skipping hardware compilation\n";
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Legacy TDG mode (pre-built TDG MLIR)
//===----------------------------------------------------------------------===//

static int runTDGMode() {
  if (SystemArchPath.empty()) {
    errs() << "tapestry-compile: --system-arch is required with --tdg\n";
    return 1;
  }

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

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Tapestry multi-core compilation\n\n"
      "Modes:\n"
      "  --tdg <path.mlir>      Load pre-built TDG MLIR\n"
      "  --auto-tdg <source.c>  Automatic TDG construction from C source\n");

  bool hasTDG = !TDGPath.empty();
  bool hasAutoTDG = !AutoTDGPath.empty();

  if (hasTDG && hasAutoTDG) {
    errs() << "tapestry-compile: cannot use both --tdg and --auto-tdg\n";
    return 1;
  }

  if (!hasTDG && !hasAutoTDG) {
    errs() << "tapestry-compile: must specify either --tdg or --auto-tdg\n";
    return 1;
  }

  if (hasAutoTDG)
    return runAutoTDGMode();

  return runTDGMode();
}
