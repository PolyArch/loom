//===-- tapestry_sensitivity.cpp - NoC and memory sensitivity sweeps -*- C++ -*-===//
//
// Runs parameter sensitivity sweeps (SPM size, core count, core type count,
// NoC bandwidth) using the Benders compilation pipeline with synthetic kernels.
// Outputs JSON results to a specified output directory.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/SystemTypes.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/TDGLowering.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>
#include <vector>

using namespace loom::tapestry;

// Command-line options
static llvm::cl::opt<std::string>
    outputDir("output-dir",
              llvm::cl::desc("Directory for sensitivity results"),
              llvm::cl::init("out/experiments/e7_sensitivity"));

static llvm::cl::opt<double>
    mapperBudget("mapper-budget",
                 llvm::cl::desc("Mapper budget in seconds per kernel"),
                 llvm::cl::init(10.0));

static llvm::cl::opt<unsigned>
    maxIter("max-iter",
            llvm::cl::desc("Max Benders iterations"),
            llvm::cl::init(5));

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static bool ensureDir(const std::string &path) {
  std::error_code ec = llvm::sys::fs::create_directories(path);
  if (ec) {
    llvm::errs() << "Failed to create directory '" << path
                 << "': " << ec.message() << "\n";
    return false;
  }
  return true;
}

static bool writeJSON(const std::string &path, const llvm::json::Value &val) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "Failed to open " << path << ": " << ec.message() << "\n";
    return false;
  }
  os << llvm::formatv("{0:2}", val);
  os << "\n";
  return true;
}

/// Create synthetic workload: a mix of add-only, mac (mul+add), and a second
/// add kernel, along with pipeline contracts between them.
struct SyntheticWorkload {
  std::vector<KernelDesc> kernels;
  std::vector<ContractSpec> contracts;
};

static SyntheticWorkload
createSyntheticWorkload(mlir::MLIRContext &ctx,
                        unsigned nocBandwidthBytesPerCycle) {
  SyntheticWorkload w;

  // Kernel 0: add-only
  auto addK = createSyntheticAddKernel("k_add", ctx);
  // Kernel 1: multiply-accumulate (mixed)
  auto macK = createSyntheticMacKernel("k_mac", ctx);
  // Kernel 2: second add-only
  auto addK2 = createSyntheticAddKernel("k_add2", ctx);
  // Kernel 3: another mac
  auto macK2 = createSyntheticMacKernel("k_mac2", ctx);
  // Kernel 4: third add
  auto addK3 = createSyntheticAddKernel("k_add3", ctx);

  if (!addK.dfgModule || !macK.dfgModule || !addK2.dfgModule ||
      !macK2.dfgModule || !addK3.dfgModule)
    return w;

  w.kernels.push_back(std::move(addK));
  w.kernels.push_back(std::move(macK));
  w.kernels.push_back(std::move(addK2));
  w.kernels.push_back(std::move(macK2));
  w.kernels.push_back(std::move(addK3));

  // Pipeline contracts: k_add -> k_mac -> k_add2 -> k_mac2 -> k_add3
  auto makeContract = [&](const std::string &prod, const std::string &cons) {
    ContractSpec c;
    c.producerKernel = prod;
    c.consumerKernel = cons;
    c.dataType = "i32";
    c.elementCount = 1024;
    c.bandwidthBytesPerCycle = nocBandwidthBytesPerCycle;
    return c;
  };

  w.contracts.push_back(makeContract("k_add", "k_mac"));
  w.contracts.push_back(makeContract("k_mac", "k_add2"));
  w.contracts.push_back(makeContract("k_add2", "k_mac2"));
  w.contracts.push_back(makeContract("k_mac2", "k_add3"));

  return w;
}

/// Convert a CompilationResult to a JSON object.
static llvm::json::Object resultToJSON(const CompilationResult &result,
                                       const std::string &configName,
                                       double elapsedSec) {
  llvm::json::Object obj;
  obj["config"] = configName;
  obj["success"] = result.success;
  obj["iterations"] = static_cast<int64_t>(result.iterations);
  obj["total_cost"] = result.totalCost;
  obj["elapsed_seconds"] = elapsedSec;
  if (!result.diagnostics.empty())
    obj["diagnostics"] = result.diagnostics;

  llvm::json::Array assignments;
  for (const auto &a : result.assignments) {
    llvm::json::Object aObj;
    aObj["kernel"] = a.kernelName;
    aObj["core_type_index"] = a.coreTypeIndex;
    aObj["core_instance"] = a.coreInstanceIndex;
    aObj["mapping_success"] = a.mappingSuccess;
    aObj["mapping_cost"] = a.mappingCost;
    aObj["routing_congestion"] = a.routingCongestion;
    aObj["unrouted_edges"] = static_cast<int64_t>(a.unroutedEdges);
    assignments.push_back(std::move(aObj));
  }
  obj["assignments"] = std::move(assignments);
  return obj;
}

/// Run HierarchicalCompiler, time it, return JSON result.
static llvm::json::Object
runAndRecord(const std::string &configName, SystemArchitecture &arch,
             std::vector<KernelDesc> kernels,
             std::vector<ContractSpec> contracts, mlir::MLIRContext &ctx,
             const CompilerConfig &config) {
  llvm::outs() << "  Running config: " << configName << " ("
               << kernels.size() << " kernels, " << contracts.size()
               << " contracts)\n";

  auto startTime = std::chrono::steady_clock::now();
  HierarchicalCompiler driver(arch, std::move(kernels), std::move(contracts), ctx);
  auto result = driver.compile(config);
  auto endTime = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration<double>(endTime - startTime).count();

  llvm::outs() << "    Result: " << (result.success ? "SUCCESS" : "INCOMPLETE")
               << ", iterations=" << result.iterations
               << ", cost=" << result.totalCost
               << ", time=" << llvm::format("%.2f", elapsed) << "s\n";

  unsigned successes = 0;
  for (const auto &a : result.assignments) {
    if (a.mappingSuccess)
      ++successes;
  }
  llvm::outs() << "    Mapped: " << successes << "/"
               << result.assignments.size() << " kernels\n";

  return resultToJSON(result, configName, elapsed);
}

//===----------------------------------------------------------------------===//
// SPM Size Sweep
//===----------------------------------------------------------------------===//

static bool runSPMSweep(mlir::MLIRContext &ctx, const std::string &outDir) {
  llvm::outs() << "\n========================================\n";
  llvm::outs() << "SPM Size Sweep\n";
  llvm::outs() << "========================================\n";

  std::vector<unsigned> spmSizes = {4096, 8192, 16384, 32768, 65536};

  CompilerConfig config;
  config.maxIterations = maxIter;
  config.mapperBudgetSeconds = mapperBudget;
  config.mapperSeed = 42;
  config.verbose = false;

  llvm::json::Array results;

  for (unsigned spmSize : spmSizes) {
    std::string label =
        "spm_" + std::to_string(spmSize / 1024) + "KB";

    // Build architecture: 2 core types (GP + DSP), 2 instances each
    std::vector<CoreTypeSpec> specs;

    CoreTypeSpec gpSpec;
    gpSpec.name = "gp_core";
    gpSpec.meshRows = 2;
    gpSpec.meshCols = 2;
    gpSpec.numInstances = 2;
    gpSpec.includeMultiplier = false;
    gpSpec.includeComparison = true;
    gpSpec.includeMemory = false;
    gpSpec.spmSizeBytes = spmSize;
    specs.push_back(gpSpec);

    CoreTypeSpec dspSpec;
    dspSpec.name = "dsp_core";
    dspSpec.meshRows = 2;
    dspSpec.meshCols = 2;
    dspSpec.numInstances = 2;
    dspSpec.includeMultiplier = true;
    dspSpec.includeComparison = false;
    dspSpec.includeMemory = false;
    dspSpec.spmSizeBytes = spmSize;
    specs.push_back(dspSpec);

    auto arch = buildArchitecture("spm_sweep_" + label, specs, ctx);
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Failed to build arch for " << label << "\n";
      continue;
    }

    // 8 bytes per flit (64-bit data width), 2 flits/cycle baseline
    auto workload = createSyntheticWorkload(ctx, /*nocBW=*/16);
    if (!lowerKernelsToDFG(workload.kernels, ctx)) {
      llvm::errs() << "  Kernel lowering failed for " << label << "\n";
      continue;
    }

    auto obj = runAndRecord(label, arch, std::move(workload.kernels),
                            std::move(workload.contracts), ctx, config);
    obj["spm_size_bytes"] = static_cast<int64_t>(spmSize);
    obj["spm_size_kb"] = static_cast<int64_t>(spmSize / 1024);
    results.push_back(std::move(obj));
  }

  llvm::json::Object summary;
  summary["sweep"] = "spm_size";
  summary["parameter"] = "spm_size_bytes";
  summary["values_kb"] = llvm::json::Array({4, 8, 16, 32, 64});
  summary["data_points"] = static_cast<int64_t>(results.size());
  summary["results"] = std::move(results);

  return writeJSON(outDir + "/spm_sweep.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// Core Count Sweep
//===----------------------------------------------------------------------===//

static bool runCoreCountSweep(mlir::MLIRContext &ctx,
                              const std::string &outDir) {
  llvm::outs() << "\n========================================\n";
  llvm::outs() << "Core Count Sweep\n";
  llvm::outs() << "========================================\n";

  std::vector<unsigned> coreCounts = {2, 4, 6, 8};

  CompilerConfig config;
  config.maxIterations = maxIter;
  config.mapperBudgetSeconds = mapperBudget;
  config.mapperSeed = 42;
  config.verbose = false;

  llvm::json::Array results;

  for (unsigned numCores : coreCounts) {
    std::string label = "cores_" + std::to_string(numCores);

    // Split cores evenly between GP and DSP types
    unsigned gpInstances = numCores / 2;
    unsigned dspInstances = numCores - gpInstances;

    std::vector<CoreTypeSpec> specs;

    CoreTypeSpec gpSpec;
    gpSpec.name = "gp_core";
    gpSpec.meshRows = 2;
    gpSpec.meshCols = 2;
    gpSpec.numInstances = gpInstances;
    gpSpec.includeMultiplier = false;
    gpSpec.includeComparison = true;
    gpSpec.includeMemory = false;
    gpSpec.spmSizeBytes = 8192;
    specs.push_back(gpSpec);

    CoreTypeSpec dspSpec;
    dspSpec.name = "dsp_core";
    dspSpec.meshRows = 2;
    dspSpec.meshCols = 2;
    dspSpec.numInstances = dspInstances;
    dspSpec.includeMultiplier = true;
    dspSpec.includeComparison = false;
    dspSpec.includeMemory = false;
    dspSpec.spmSizeBytes = 8192;
    specs.push_back(dspSpec);

    auto arch = buildArchitecture("core_count_" + label, specs, ctx);
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Failed to build arch for " << label << "\n";
      continue;
    }

    auto workload = createSyntheticWorkload(ctx, /*nocBW=*/16);
    if (!lowerKernelsToDFG(workload.kernels, ctx)) {
      llvm::errs() << "  Kernel lowering failed for " << label << "\n";
      continue;
    }

    auto obj = runAndRecord(label, arch, std::move(workload.kernels),
                            std::move(workload.contracts), ctx, config);
    obj["num_cores"] = static_cast<int64_t>(numCores);
    obj["gp_instances"] = static_cast<int64_t>(gpInstances);
    obj["dsp_instances"] = static_cast<int64_t>(dspInstances);
    results.push_back(std::move(obj));
  }

  llvm::json::Object summary;
  summary["sweep"] = "core_count";
  summary["parameter"] = "num_cores";
  summary["values"] = llvm::json::Array({2, 4, 6, 8});
  summary["data_points"] = static_cast<int64_t>(results.size());
  summary["results"] = std::move(results);

  return writeJSON(outDir + "/core_count_sweep.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// Core Type Count Sweep
//===----------------------------------------------------------------------===//

static bool runCoreTypeSweep(mlir::MLIRContext &ctx,
                             const std::string &outDir) {
  llvm::outs() << "\n========================================\n";
  llvm::outs() << "Core Type Count Sweep\n";
  llvm::outs() << "========================================\n";

  std::vector<unsigned> typeCounts = {1, 2, 3};

  CompilerConfig config;
  config.maxIterations = maxIter;
  config.mapperBudgetSeconds = mapperBudget;
  config.mapperSeed = 42;
  config.verbose = false;

  llvm::json::Array results;

  // Total instances fixed at 4
  constexpr unsigned totalInstances = 4;

  for (unsigned numTypes : typeCounts) {
    std::string label = "types_" + std::to_string(numTypes);

    std::vector<CoreTypeSpec> specs;

    if (numTypes == 1) {
      // Homogeneous: 1 type with all capabilities, 4 instances
      CoreTypeSpec allSpec;
      allSpec.name = "unified_core";
      allSpec.meshRows = 2;
      allSpec.meshCols = 2;
      allSpec.numInstances = totalInstances;
      allSpec.includeMultiplier = true;
      allSpec.includeComparison = true;
      allSpec.includeMemory = false;
      allSpec.spmSizeBytes = 8192;
      specs.push_back(allSpec);
    } else if (numTypes == 2) {
      // 2 types: GP (add+cmp) and DSP (add+mul), 2 instances each
      CoreTypeSpec gpSpec;
      gpSpec.name = "gp_core";
      gpSpec.meshRows = 2;
      gpSpec.meshCols = 2;
      gpSpec.numInstances = 2;
      gpSpec.includeMultiplier = false;
      gpSpec.includeComparison = true;
      gpSpec.includeMemory = false;
      gpSpec.spmSizeBytes = 8192;
      specs.push_back(gpSpec);

      CoreTypeSpec dspSpec;
      dspSpec.name = "dsp_core";
      dspSpec.meshRows = 2;
      dspSpec.meshCols = 2;
      dspSpec.numInstances = 2;
      dspSpec.includeMultiplier = true;
      dspSpec.includeComparison = false;
      dspSpec.includeMemory = false;
      dspSpec.spmSizeBytes = 8192;
      specs.push_back(dspSpec);
    } else {
      // 3 types: GP (add+cmp), DSP (add+mul), FULL (all), distributed
      CoreTypeSpec gpSpec;
      gpSpec.name = "gp_core";
      gpSpec.meshRows = 2;
      gpSpec.meshCols = 2;
      gpSpec.numInstances = 1;
      gpSpec.includeMultiplier = false;
      gpSpec.includeComparison = true;
      gpSpec.includeMemory = false;
      gpSpec.spmSizeBytes = 8192;
      specs.push_back(gpSpec);

      CoreTypeSpec dspSpec;
      dspSpec.name = "dsp_core";
      dspSpec.meshRows = 2;
      dspSpec.meshCols = 2;
      dspSpec.numInstances = 1;
      dspSpec.includeMultiplier = true;
      dspSpec.includeComparison = false;
      dspSpec.includeMemory = false;
      dspSpec.spmSizeBytes = 8192;
      specs.push_back(dspSpec);

      CoreTypeSpec fullSpec;
      fullSpec.name = "full_core";
      fullSpec.meshRows = 2;
      fullSpec.meshCols = 2;
      fullSpec.numInstances = 2;
      fullSpec.includeMultiplier = true;
      fullSpec.includeComparison = true;
      fullSpec.includeMemory = false;
      fullSpec.spmSizeBytes = 8192;
      specs.push_back(fullSpec);
    }

    auto arch = buildArchitecture("type_count_" + label, specs, ctx);
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Failed to build arch for " << label << "\n";
      continue;
    }

    auto workload = createSyntheticWorkload(ctx, /*nocBW=*/16);
    if (!lowerKernelsToDFG(workload.kernels, ctx)) {
      llvm::errs() << "  Kernel lowering failed for " << label << "\n";
      continue;
    }

    auto obj = runAndRecord(label, arch, std::move(workload.kernels),
                            std::move(workload.contracts), ctx, config);
    obj["num_core_types"] = static_cast<int64_t>(numTypes);
    obj["total_instances"] = static_cast<int64_t>(totalInstances);
    results.push_back(std::move(obj));
  }

  llvm::json::Object summary;
  summary["sweep"] = "core_type_count";
  summary["parameter"] = "num_core_types";
  summary["values"] = llvm::json::Array({1, 2, 3});
  summary["total_instances"] = static_cast<int64_t>(totalInstances);
  summary["data_points"] = static_cast<int64_t>(results.size());
  summary["results"] = std::move(results);

  return writeJSON(outDir + "/core_type_sweep.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// NoC Bandwidth Sweep (including perfect-NoC upper bound)
//===----------------------------------------------------------------------===//

static bool runNoCBandwidthSweep(mlir::MLIRContext &ctx,
                                 const std::string &outDir) {
  llvm::outs() << "\n========================================\n";
  llvm::outs() << "NoC Bandwidth Sweep\n";
  llvm::outs() << "========================================\n";

  // NoC bandwidth in flits/cycle. Each flit = 8 bytes (64-bit data width).
  // Sweep: 1, 2, 3, 4 flits/cycle => 8, 16, 24, 32 bytes/cycle
  struct NoCConfig {
    unsigned flitsPerCycle;
    unsigned bytesPerCycle;
    std::string label;
  };
  std::vector<NoCConfig> nocConfigs = {
      {1, 8, "1_flit"},
      {2, 16, "2_flits"},
      {3, 24, "3_flits"},
      {4, 32, "4_flits"},
  };

  CompilerConfig config;
  config.maxIterations = maxIter;
  config.mapperBudgetSeconds = mapperBudget;
  config.mapperSeed = 42;
  config.verbose = false;

  llvm::json::Array results;

  // Standard architecture for all NoC configs
  auto buildBaseArch = [&](const std::string &name) {
    std::vector<CoreTypeSpec> specs;

    CoreTypeSpec gpSpec;
    gpSpec.name = "gp_core";
    gpSpec.meshRows = 2;
    gpSpec.meshCols = 2;
    gpSpec.numInstances = 2;
    gpSpec.includeMultiplier = false;
    gpSpec.includeComparison = true;
    gpSpec.includeMemory = false;
    gpSpec.spmSizeBytes = 8192;
    specs.push_back(gpSpec);

    CoreTypeSpec dspSpec;
    dspSpec.name = "dsp_core";
    dspSpec.meshRows = 2;
    dspSpec.meshCols = 2;
    dspSpec.numInstances = 2;
    dspSpec.includeMultiplier = true;
    dspSpec.includeComparison = false;
    dspSpec.includeMemory = false;
    dspSpec.spmSizeBytes = 8192;
    specs.push_back(dspSpec);

    return buildArchitecture(name, specs, ctx);
  };

  // Sweep finite NoC bandwidths
  for (const auto &noc : nocConfigs) {
    auto arch = buildBaseArch("noc_sweep_" + noc.label);
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Failed to build arch for " << noc.label << "\n";
      continue;
    }

    auto workload = createSyntheticWorkload(ctx, noc.bytesPerCycle);
    if (!lowerKernelsToDFG(workload.kernels, ctx)) {
      llvm::errs() << "  Kernel lowering failed for " << noc.label << "\n";
      continue;
    }

    auto obj = runAndRecord("noc_" + noc.label, arch,
                            std::move(workload.kernels),
                            std::move(workload.contracts), ctx, config);
    obj["flits_per_cycle"] = static_cast<int64_t>(noc.flitsPerCycle);
    obj["bytes_per_cycle"] = static_cast<int64_t>(noc.bytesPerCycle);
    obj["perfect_noc"] = false;
    results.push_back(std::move(obj));
  }

  // Perfect NoC: no inter-core communication constraints (empty contracts)
  {
    llvm::outs() << "\n--- Perfect NoC (upper bound) ---\n";
    auto arch = buildBaseArch("noc_perfect");
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Failed to build arch for perfect NoC\n";
    } else {
      // Create workload with zero bandwidth constraint (no contracts)
      auto workload = createSyntheticWorkload(ctx, 0);
      if (!lowerKernelsToDFG(workload.kernels, ctx)) {
        llvm::errs() << "  Kernel lowering failed for perfect NoC\n";
      } else {
        // Remove all contracts to simulate perfect NoC
        std::vector<ContractSpec> emptyContracts;
        auto obj = runAndRecord("noc_perfect", arch,
                                std::move(workload.kernels),
                                std::move(emptyContracts), ctx, config);
        obj["flits_per_cycle"] = 0;
        obj["bytes_per_cycle"] = 0;
        obj["perfect_noc"] = true;
        results.push_back(std::move(obj));
      }
    }
  }

  llvm::json::Object summary;
  summary["sweep"] = "noc_bandwidth";
  summary["parameter"] = "flits_per_cycle";
  summary["flit_size_bytes"] = 8;
  summary["includes_perfect_noc"] = true;
  summary["data_points"] = static_cast<int64_t>(results.size());
  summary["results"] = std::move(results);

  return writeJSON(outDir + "/noc_bandwidth_sweep.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Tapestry sensitivity sweep experiments\n");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  llvm::outs() << "=== Tapestry Sensitivity Sweeps ===\n";
  llvm::outs() << "Output dir: " << outputDir << "\n";
  llvm::outs() << "Mapper budget: " << mapperBudget << "s\n";
  llvm::outs() << "Max iterations: " << maxIter << "\n\n";

  if (!ensureDir(outputDir))
    return 1;

  bool anyFailed = false;
  unsigned sweepsCompleted = 0;

  auto startTotal = std::chrono::steady_clock::now();

  // SPM size sweep
  if (runSPMSweep(ctx, outputDir))
    ++sweepsCompleted;
  else
    anyFailed = true;

  // Core count sweep
  if (runCoreCountSweep(ctx, outputDir))
    ++sweepsCompleted;
  else
    anyFailed = true;

  // Core type count sweep
  if (runCoreTypeSweep(ctx, outputDir))
    ++sweepsCompleted;
  else
    anyFailed = true;

  // NoC bandwidth sweep (includes perfect NoC upper bound)
  if (runNoCBandwidthSweep(ctx, outputDir))
    ++sweepsCompleted;
  else
    anyFailed = true;

  auto endTotal = std::chrono::steady_clock::now();
  double totalElapsed =
      std::chrono::duration<double>(endTotal - startTotal).count();

  // Write combined summary
  llvm::json::Object combined;
  combined["experiment"] = "E7: NoC and Memory Sensitivity";
  combined["sweeps_completed"] = static_cast<int64_t>(sweepsCompleted);
  combined["sweeps_total"] = 4;
  combined["total_elapsed_seconds"] = totalElapsed;
  combined["status"] = anyFailed ? "PARTIAL" : "COMPLETE";
  combined["output_files"] = llvm::json::Array({
      "spm_sweep.json",
      "core_count_sweep.json",
      "core_type_sweep.json",
      "noc_bandwidth_sweep.json",
  });

  writeJSON(outputDir.getValue() + "/summary.json",
            llvm::json::Value(std::move(combined)));

  llvm::outs() << "\n=== Sensitivity Sweep Summary ===\n";
  llvm::outs() << "Sweeps completed: " << sweepsCompleted << "/4\n";
  llvm::outs() << "Total time: " << llvm::format("%.2f", totalElapsed)
               << "s\n";
  llvm::outs() << "Status: " << (anyFailed ? "PARTIAL" : "COMPLETE") << "\n";
  llvm::outs() << "Results in: " << outputDir << "/\n";

  return anyFailed ? 1 : 0;
}
