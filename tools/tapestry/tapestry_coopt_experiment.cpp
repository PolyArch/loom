//===-- tapestry_coopt_experiment.cpp - Co-optimization experiments -*- C++ -*-===//
//
// Runs co-optimization experiments (E17-E20) that alternate SW and HW
// optimization. Supports multiple modes: convergence analysis, SW-only vs
// HW-only vs co-opt comparison, cross-domain portability, and sensitivity
// analysis across initial architectures.
//
// Output: JSON files with round-by-round history, Pareto points, timing.
//
//===----------------------------------------------------------------------===//

#include "tapestry/co_optimizer.h"

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/BendersDriver.h"
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
namespace tco = tapestry;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    experimentMode("mode",
                   llvm::cl::desc("Experiment mode: convergence | comparison "
                                  "| cross_domain | sensitivity"),
                   llvm::cl::Required);

static llvm::cl::opt<std::string>
    outputDir("output-dir",
              llvm::cl::desc("Directory for experiment results"),
              llvm::cl::init("out/experiments/E17"));

static llvm::cl::opt<unsigned>
    maxRounds("max-rounds",
              llvm::cl::desc("Maximum co-optimization rounds"),
              llvm::cl::init(10));

static llvm::cl::opt<double>
    threshold("threshold",
              llvm::cl::desc("Improvement threshold for convergence"),
              llvm::cl::init(0.01));

static llvm::cl::opt<std::string>
    domain("domain",
           llvm::cl::desc("Domain to run (ai_llm, dsp_ofdm, etc.)"),
           llvm::cl::init("all"));

static llvm::cl::opt<std::string>
    archConfig("arch-config",
               llvm::cl::desc("Architecture config: spectral | homogeneous_gp "
                              "| homogeneous_dsp | random_fu | oversized"),
               llvm::cl::init("spectral"));

static llvm::cl::opt<double>
    mapperBudget("mapper-budget",
                 llvm::cl::desc("Mapper budget in seconds per kernel"),
                 llvm::cl::init(10.0));

static llvm::cl::opt<bool>
    verbose("verbose", llvm::cl::desc("Verbose output"), llvm::cl::init(false));

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

//===----------------------------------------------------------------------===//
// Domain workload construction
//===----------------------------------------------------------------------===//

/// Domain descriptor: kernel count, contract count, complexity.
struct DomainWorkload {
  std::string name;
  std::vector<KernelDesc> kernels;
  std::vector<ContractSpec> contracts;
};

/// Build a multi-kernel workload for a named domain using synthetic kernels
/// that mirror the TDG structure from benchmarks/tapestry/<domain>/.
static DomainWorkload
buildDomainWorkload(const std::string &domainName, mlir::MLIRContext &ctx) {
  DomainWorkload w;
  w.name = domainName;

  // Each domain gets a kernel set mirroring its TDG description,
  // built with synthetic add/mac DFGs that the real pipeline can compile.
  struct KernelSpec {
    std::string name;
    bool isMac; // false => add kernel, true => mac kernel
  };

  struct ContractEdge {
    std::string producer;
    std::string consumer;
    uint64_t elementCount;
    uint64_t bw;
  };

  std::vector<KernelSpec> kSpecs;
  std::vector<ContractEdge> cEdges;

  if (domainName == "ai_llm") {
    kSpecs = {{"qkv_proj", true}, {"attn_score", true}, {"softmax", false},
              {"attn_output", true}, {"ffn1", true}, {"gelu", false},
              {"ffn2", true}, {"layernorm", false}};
    cEdges = {{"qkv_proj", "attn_score", 2048, 8},
              {"attn_score", "softmax", 4096, 8},
              {"softmax", "attn_output", 4096, 8},
              {"attn_output", "ffn1", 16384, 16},
              {"ffn1", "gelu", 65536, 16},
              {"gelu", "ffn2", 65536, 16},
              {"ffn2", "layernorm", 16384, 8}};
  } else if (domainName == "dsp_ofdm") {
    kSpecs = {{"fft_butterfly", true}, {"channel_est", false},
              {"equalizer", false}, {"qam_demod", false},
              {"viterbi", false}, {"crc_check", false}};
    cEdges = {{"fft_butterfly", "channel_est", 4096, 16},
              {"channel_est", "equalizer", 1200, 8},
              {"equalizer", "qam_demod", 1200, 8},
              {"qam_demod", "viterbi", 7200, 16},
              {"viterbi", "crc_check", 1800, 8}};
  } else if (domainName == "arvr_stereo") {
    kSpecs = {{"harris_corner", false}, {"sad_matching", true},
              {"stereo_disparity", true}, {"image_warp", false},
              {"post_filter", false}};
    cEdges = {{"harris_corner", "sad_matching", 4096, 16},
              {"sad_matching", "stereo_disparity", 262144, 32},
              {"stereo_disparity", "image_warp", 4096, 8},
              {"image_warp", "post_filter", 4096, 8}};
  } else if (domainName == "robotics_vio") {
    kSpecs = {{"imu_integration", false}, {"fast_detect", false},
              {"orb_descriptor", false}, {"feature_match", true},
              {"pose_estimate", true}};
    cEdges = {{"imu_integration", "pose_estimate", 600, 4},
              {"fast_detect", "orb_descriptor", 1000, 8},
              {"orb_descriptor", "feature_match", 4000, 16},
              {"feature_match", "pose_estimate", 400, 4}};
  } else if (domainName == "graph_analytics") {
    kSpecs = {{"bfs_traversal", false}, {"pagerank_spmv", true},
              {"triangle_count", false}, {"label_prop", false}};
    cEdges = {{"bfs_traversal", "pagerank_spmv", 1024, 4},
              {"pagerank_spmv", "label_prop", 1024, 4},
              {"bfs_traversal", "triangle_count", 1024, 4}};
  } else if (domainName == "zk_stark") {
    kSpecs = {{"ntt", true}, {"msm", true},
              {"poseidon_hash", false}, {"poly_eval", true},
              {"proof_compose", false}};
    cEdges = {{"ntt", "poly_eval", 1024, 8},
              {"poly_eval", "proof_compose", 256, 4},
              {"poseidon_hash", "proof_compose", 4, 4},
              {"msm", "proof_compose", 3, 4},
              {"ntt", "poseidon_hash", 8, 4}};
  } else {
    llvm::errs() << "Unknown domain: " << domainName << "\n";
    return w;
  }

  // Build synthetic kernels
  for (const auto &ks : kSpecs) {
    KernelDesc kd = ks.isMac ? createSyntheticMacKernel(ks.name, ctx)
                             : createSyntheticAddKernel(ks.name, ctx);
    if (!kd.dfgModule) {
      llvm::errs() << "  Failed to create kernel: " << ks.name << "\n";
      w.kernels.clear();
      return w;
    }
    w.kernels.push_back(std::move(kd));
  }

  // Lower to DFG
  if (!lowerKernelsToDFG(w.kernels, ctx)) {
    llvm::errs() << "  DFG lowering failed for domain " << domainName << "\n";
    w.kernels.clear();
    return w;
  }

  // Build contracts
  for (const auto &ce : cEdges) {
    ContractSpec c;
    c.producerKernel = ce.producer;
    c.consumerKernel = ce.consumer;
    c.dataType = "i32";
    c.elementCount = ce.elementCount;
    c.bandwidthBytesPerCycle = ce.bw;
    w.contracts.push_back(std::move(c));
  }

  return w;
}

//===----------------------------------------------------------------------===//
// Architecture construction variants (for E20 sensitivity)
//===----------------------------------------------------------------------===//

static SystemArchitecture buildInitialArch(const std::string &config,
                                           unsigned numKernels,
                                           mlir::MLIRContext &ctx) {
  if (config == "spectral") {
    // Default: 2 heterogeneous core types (GP + DSP)
    std::vector<CoreTypeSpec> specs;
    CoreTypeSpec gpSpec;
    gpSpec.name = "gp_core";
    gpSpec.meshRows = 2;
    gpSpec.meshCols = 2;
    gpSpec.numInstances = std::max(1u, numKernels / 2);
    gpSpec.includeMultiplier = false;
    gpSpec.includeComparison = true;
    gpSpec.includeMemory = false;
    gpSpec.spmSizeBytes = 4096;
    specs.push_back(gpSpec);

    CoreTypeSpec dspSpec;
    dspSpec.name = "dsp_core";
    dspSpec.meshRows = 2;
    dspSpec.meshCols = 2;
    dspSpec.numInstances = std::max(1u, (numKernels + 1) / 2);
    dspSpec.includeMultiplier = true;
    dspSpec.includeComparison = false;
    dspSpec.includeMemory = false;
    dspSpec.spmSizeBytes = 8192;
    specs.push_back(dspSpec);

    return buildArchitecture("spectral_default", specs, ctx);
  }

  if (config == "homogeneous_gp") {
    std::vector<CoreTypeSpec> specs;
    CoreTypeSpec gpSpec;
    gpSpec.name = "gp_core";
    gpSpec.meshRows = 2;
    gpSpec.meshCols = 2;
    gpSpec.numInstances = std::max(2u, numKernels);
    gpSpec.includeMultiplier = false;
    gpSpec.includeComparison = true;
    gpSpec.includeMemory = false;
    gpSpec.spmSizeBytes = 4096;
    specs.push_back(gpSpec);
    return buildArchitecture("homogeneous_gp", specs, ctx);
  }

  if (config == "homogeneous_dsp") {
    std::vector<CoreTypeSpec> specs;
    CoreTypeSpec dspSpec;
    dspSpec.name = "dsp_core";
    dspSpec.meshRows = 2;
    dspSpec.meshCols = 2;
    dspSpec.numInstances = std::max(2u, numKernels);
    dspSpec.includeMultiplier = true;
    dspSpec.includeComparison = false;
    dspSpec.includeMemory = false;
    dspSpec.spmSizeBytes = 8192;
    specs.push_back(dspSpec);
    return buildArchitecture("homogeneous_dsp", specs, ctx);
  }

  if (config == "random_fu") {
    // 3 core types with different FU mixes
    std::vector<CoreTypeSpec> specs;
    CoreTypeSpec t0;
    t0.name = "arith_core";
    t0.meshRows = 2;
    t0.meshCols = 2;
    t0.numInstances = std::max(1u, numKernels / 3);
    t0.includeMultiplier = true;
    t0.includeComparison = true;
    t0.includeMemory = false;
    t0.spmSizeBytes = 4096;
    specs.push_back(t0);

    CoreTypeSpec t1;
    t1.name = "logic_core";
    t1.meshRows = 2;
    t1.meshCols = 2;
    t1.numInstances = std::max(1u, numKernels / 3);
    t1.includeMultiplier = false;
    t1.includeComparison = true;
    t1.includeMemory = false;
    t1.spmSizeBytes = 4096;
    specs.push_back(t1);

    CoreTypeSpec t2;
    t2.name = "mem_core";
    t2.meshRows = 2;
    t2.meshCols = 2;
    t2.numInstances = std::max(1u, numKernels / 3 + numKernels % 3);
    t2.includeMultiplier = false;
    t2.includeComparison = false;
    t2.includeMemory = true;
    t2.spmSizeBytes = 16384;
    specs.push_back(t2);

    return buildArchitecture("random_fu", specs, ctx);
  }

  if (config == "oversized") {
    // Single large core type with 8x8 mesh
    std::vector<CoreTypeSpec> specs;
    CoreTypeSpec bigSpec;
    bigSpec.name = "big_core";
    bigSpec.meshRows = 4;
    bigSpec.meshCols = 4;
    bigSpec.numInstances = std::max(2u, numKernels);
    bigSpec.includeMultiplier = true;
    bigSpec.includeComparison = true;
    bigSpec.includeMemory = true;
    bigSpec.spmSizeBytes = 32768;
    specs.push_back(bigSpec);
    return buildArchitecture("oversized", specs, ctx);
  }

  llvm::errs() << "Unknown architecture config: " << config << "\n";
  return SystemArchitecture{};
}

//===----------------------------------------------------------------------===//
// Co-optimization result to JSON
//===----------------------------------------------------------------------===//

static llvm::json::Object
coOptResultToJSON(const tco::CoOptResult &result, const std::string &label,
                  double elapsedSec) {
  llvm::json::Object obj;
  obj["label"] = label;
  obj["success"] = result.success;
  obj["rounds"] = static_cast<int64_t>(result.rounds);
  obj["best_throughput"] = result.bestThroughput;
  obj["best_area"] = result.bestArea;
  obj["elapsed_seconds"] = elapsedSec;

  // Round-by-round history
  llvm::json::Array histArr;
  for (const auto &r : result.history) {
    llvm::json::Object rObj;
    rObj["round"] = static_cast<int64_t>(r.round);
    rObj["sw_throughput"] = r.swThroughput;
    rObj["hw_area"] = r.hwArea;
    rObj["sw_transforms"] = static_cast<int64_t>(r.swTransforms);
    rObj["hw_core_types"] = static_cast<int64_t>(r.hwCoreTypes);
    rObj["improved"] = r.improved;
    histArr.push_back(std::move(rObj));
  }
  obj["history"] = std::move(histArr);

  // Pareto frontier
  llvm::json::Array paretoArr;
  for (const auto &p : result.paretoFrontier) {
    llvm::json::Object pObj;
    pObj["throughput"] = p.throughput;
    pObj["area"] = p.area;
    pObj["round"] = static_cast<int64_t>(p.round);
    paretoArr.push_back(std::move(pObj));
  }
  obj["pareto_frontier"] = std::move(paretoArr);

  if (!result.diagnostics.empty())
    obj["diagnostics"] = result.diagnostics;

  return obj;
}

//===----------------------------------------------------------------------===//
// Run co-optimization for a single domain
//===----------------------------------------------------------------------===//

static tco::CoOptResult
runCoOpt(DomainWorkload &workload, const SystemArchitecture &initialArch,
         mlir::MLIRContext &ctx, unsigned maxR, double thresh, bool verb) {
  tco::CoOptOptions coOpts;
  coOpts.maxRounds = maxR;
  coOpts.improvementThreshold = thresh;
  coOpts.verbose = verb;
  coOpts.swOpts.maxIterations = 5;
  coOpts.swOpts.improvementThreshold = thresh;
  coOpts.swOpts.bendersConfig.maxIterations = 5;
  coOpts.swOpts.bendersConfig.mapperBudgetSeconds = mapperBudget;
  coOpts.swOpts.bendersConfig.mapperSeed = 42;
  coOpts.swOpts.verbose = verb;
  coOpts.hwOuterOpts.maxIterations = 20;
  coOpts.hwOuterOpts.seed = 42;
  coOpts.hwOuterOpts.verbose = verb;
  // Tier-B (BO + mapper) disabled by default because the fallback topology
  // from OUTER-HW produces minimal constraints that can trigger ADGBuilder
  // assertion failures when generating spatial switches with unconnected ports.
  // Tier-A analytical derivation is sufficient for experiment metrics.
  coOpts.hwInnerOpts.tier2Enabled = false;
  coOpts.hwInnerOpts.maxInnerIter = 10;
  coOpts.hwInnerOpts.seed = 42;
  coOpts.hwInnerOpts.verbose = verb;

  return tco::co_optimize(workload.kernels, workload.contracts,
                          initialArch, coOpts, &ctx);
}

/// Run SW-only optimization: fix hardware, only optimize software.
static tco::CoOptResult
runSWOnly(DomainWorkload &workload, const SystemArchitecture &initialArch,
          mlir::MLIRContext &ctx, bool verb) {
  tco::CoOptOptions coOpts;
  coOpts.maxRounds = 1; // Single round: SW only
  coOpts.improvementThreshold = 0.0;
  coOpts.verbose = verb;
  coOpts.swOpts.maxIterations = 10; // More SW iterations to compensate
  coOpts.swOpts.improvementThreshold = 0.005;
  coOpts.swOpts.bendersConfig.maxIterations = 5;
  coOpts.swOpts.bendersConfig.mapperBudgetSeconds = mapperBudget;
  coOpts.swOpts.bendersConfig.mapperSeed = 42;
  coOpts.swOpts.verbose = verb;
  // HW parameters set to minimal since HW step still runs once
  coOpts.hwOuterOpts.maxIterations = 1;
  coOpts.hwOuterOpts.seed = 42;
  coOpts.hwInnerOpts.tier2Enabled = false;
  coOpts.hwInnerOpts.maxInnerIter = 1;

  return tco::co_optimize(workload.kernels, workload.contracts,
                          initialArch, coOpts, &ctx);
}

/// Run HW-only optimization: fix software, only optimize hardware.
static tco::CoOptResult
runHWOnly(DomainWorkload &workload, const SystemArchitecture &initialArch,
          mlir::MLIRContext &ctx, bool verb) {
  tco::CoOptOptions coOpts;
  coOpts.maxRounds = 1;
  coOpts.improvementThreshold = 0.0;
  coOpts.verbose = verb;
  // SW set to minimal
  coOpts.swOpts.maxIterations = 1;
  coOpts.swOpts.improvementThreshold = 0.0;
  coOpts.swOpts.bendersConfig.maxIterations = 3;
  coOpts.swOpts.bendersConfig.mapperBudgetSeconds = mapperBudget;
  coOpts.swOpts.bendersConfig.mapperSeed = 42;
  // Full HW optimization (Tier-B disabled for same reason as co-opt)
  coOpts.hwOuterOpts.maxIterations = 50;
  coOpts.hwOuterOpts.seed = 42;
  coOpts.hwOuterOpts.verbose = verb;
  coOpts.hwInnerOpts.tier2Enabled = false;
  coOpts.hwInnerOpts.maxInnerIter = 20;
  coOpts.hwInnerOpts.seed = 42;
  coOpts.hwInnerOpts.verbose = verb;

  return tco::co_optimize(workload.kernels, workload.contracts,
                          initialArch, coOpts, &ctx);
}

//===----------------------------------------------------------------------===//
// Domain list
//===----------------------------------------------------------------------===//

static const std::vector<std::string> ALL_DOMAINS = {
    "ai_llm", "dsp_ofdm", "arvr_stereo",
    "robotics_vio", "graph_analytics", "zk_stark"};

static std::vector<std::string> getDomains() {
  if (domain == "all")
    return ALL_DOMAINS;
  return {domain.getValue()};
}

//===----------------------------------------------------------------------===//
// E17: Co-optimization convergence
//===----------------------------------------------------------------------===//

static bool runE17Convergence(mlir::MLIRContext &ctx) {
  llvm::outs() << "=== E17: Co-Optimization Convergence ===\n\n";

  auto domains = getDomains();
  llvm::json::Array allResults;

  for (const auto &dom : domains) {
    llvm::outs() << "--- Domain: " << dom << " ---\n";

    auto workload = buildDomainWorkload(dom, ctx);
    if (workload.kernels.empty()) {
      llvm::errs() << "  Skipping domain " << dom
                   << ": workload construction failed\n";
      continue;
    }

    auto arch = buildInitialArch("spectral", workload.kernels.size(), ctx);
    if (arch.coreTypes.empty()) {
      llvm::errs() << "  Skipping domain " << dom
                   << ": architecture build failed\n";
      continue;
    }

    auto startTime = std::chrono::steady_clock::now();
    auto result = runCoOpt(workload, arch, ctx, maxRounds, threshold, verbose);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(endTime - startTime).count();

    llvm::outs() << "  Completed: " << result.rounds << " rounds, "
                 << "throughput=" << result.bestThroughput
                 << ", area=" << result.bestArea
                 << ", time=" << llvm::format("%.2f", elapsed) << "s\n";

    auto obj = coOptResultToJSON(result, dom, elapsed);
    obj["domain"] = dom;
    obj["max_rounds"] = static_cast<int64_t>(maxRounds.getValue());
    obj["threshold"] = threshold.getValue();
    obj["num_kernels"] = static_cast<int64_t>(workload.kernels.size());
    obj["num_contracts"] = static_cast<int64_t>(workload.contracts.size());
    allResults.push_back(std::move(obj));
  }

  llvm::json::Object summary;
  summary["experiment"] = "E17: Co-Optimization Convergence";
  summary["mode"] = "convergence";
  summary["domains_run"] = static_cast<int64_t>(allResults.size());
  summary["max_rounds"] = static_cast<int64_t>(maxRounds.getValue());
  summary["threshold"] = threshold.getValue();
  summary["results"] = std::move(allResults);

  return writeJSON(outputDir.getValue() + "/convergence_results.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// E18: SW-only vs HW-only vs Co-opt comparison
//===----------------------------------------------------------------------===//

static bool runE18Comparison(mlir::MLIRContext &ctx) {
  llvm::outs() << "=== E18: SW-Only vs HW-Only vs Co-Opt ===\n\n";

  auto domains = getDomains();
  llvm::json::Array allResults;

  for (const auto &dom : domains) {
    llvm::outs() << "--- Domain: " << dom << " ---\n";

    // Build workload 3 times (one for each mode, since co_optimize consumes)
    struct ModeRun {
      std::string mode;
      std::function<tco::CoOptResult(DomainWorkload &, const SystemArchitecture &,
                                     mlir::MLIRContext &, bool)> runFn;
    };

    std::vector<ModeRun> modes = {
        {"sw_only",
         [](DomainWorkload &w, const SystemArchitecture &a,
            mlir::MLIRContext &c, bool v) { return runSWOnly(w, a, c, v); }},
        {"hw_only",
         [](DomainWorkload &w, const SystemArchitecture &a,
            mlir::MLIRContext &c, bool v) { return runHWOnly(w, a, c, v); }},
        {"co_opt",
         [&](DomainWorkload &w, const SystemArchitecture &a,
             mlir::MLIRContext &c, bool v) {
           return runCoOpt(w, a, c, maxRounds, threshold, v);
         }},
    };

    for (const auto &m : modes) {
      llvm::outs() << "  Mode: " << m.mode << "\n";

      auto workload = buildDomainWorkload(dom, ctx);
      if (workload.kernels.empty()) {
        llvm::errs() << "    Skipping: workload construction failed\n";
        continue;
      }

      auto arch = buildInitialArch("spectral", workload.kernels.size(), ctx);
      if (arch.coreTypes.empty()) {
        llvm::errs() << "    Skipping: architecture build failed\n";
        continue;
      }

      auto startTime = std::chrono::steady_clock::now();
      auto result = m.runFn(workload, arch, ctx, verbose);
      auto endTime = std::chrono::steady_clock::now();
      double elapsed =
          std::chrono::duration<double>(endTime - startTime).count();

      llvm::outs() << "    Result: throughput=" << result.bestThroughput
                   << ", area=" << result.bestArea
                   << ", rounds=" << result.rounds
                   << ", pareto_size=" << result.paretoFrontier.size()
                   << ", time=" << llvm::format("%.2f", elapsed) << "s\n";

      auto obj = coOptResultToJSON(result, dom + "_" + m.mode, elapsed);
      obj["domain"] = dom;
      obj["mode"] = m.mode;
      obj["num_kernels"] = static_cast<int64_t>(workload.kernels.size());

      // Compute throughput/area efficiency
      double efficiency = (result.bestArea > 0.0)
                              ? result.bestThroughput / result.bestArea
                              : 0.0;
      obj["throughput_per_area"] = efficiency;

      allResults.push_back(std::move(obj));
    }
  }

  llvm::json::Object summary;
  summary["experiment"] = "E18: SW-Only vs HW-Only vs Co-Opt";
  summary["mode"] = "comparison";
  summary["data_points"] = static_cast<int64_t>(allResults.size());
  summary["results"] = std::move(allResults);

  return writeJSON(outputDir.getValue() + "/comparison_results.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// E19: Cross-domain hardware portability
//===----------------------------------------------------------------------===//

static bool runE19CrossDomain(mlir::MLIRContext &ctx) {
  llvm::outs() << "=== E19: Cross-Domain Hardware Portability ===\n\n";

  auto domains = getDomains();
  if (domains.size() == 1) {
    // For cross-domain, override to all domains
    domains = ALL_DOMAINS;
    llvm::outs() << "  Using all 6 domains for cross-domain matrix\n";
  }

  // First pass: run co-opt on each domain to get domain-specialized archs
  struct DomainResult {
    std::string domain;
    tco::CoOptResult result;
    SystemArchitecture arch;
    double nativeThroughput;
  };

  std::vector<DomainResult> domResults;

  llvm::outs() << "--- Building domain-specialized architectures ---\n";
  for (const auto &dom : domains) {
    llvm::outs() << "  Optimizing for " << dom << "...\n";

    auto workload = buildDomainWorkload(dom, ctx);
    if (workload.kernels.empty()) {
      llvm::errs() << "    Skipping: workload construction failed\n";
      continue;
    }

    auto initialArch =
        buildInitialArch("spectral", workload.kernels.size(), ctx);
    if (initialArch.coreTypes.empty()) {
      llvm::errs() << "    Skipping: architecture build failed\n";
      continue;
    }

    auto result = runCoOpt(workload, initialArch, ctx, maxRounds, threshold,
                           verbose);

    DomainResult dr;
    dr.domain = dom;
    dr.result = result;
    dr.arch = result.bestArch;
    dr.nativeThroughput = result.bestThroughput;
    domResults.push_back(std::move(dr));

    llvm::outs() << "    Native throughput: " << result.bestThroughput
                 << ", area: " << result.bestArea << "\n";
  }

  // Second pass: cross-compile each SW domain TDG on each HW domain's arch
  llvm::outs() << "\n--- Cross-domain compilation matrix ---\n";

  llvm::json::Array allResults;

  for (const auto &swDom : domResults) {
    for (const auto &hwDom : domResults) {
      llvm::outs() << "  SW=" << swDom.domain << " HW=" << hwDom.domain;

      // Build workload for SW domain
      auto workload = buildDomainWorkload(swDom.domain, ctx);
      if (workload.kernels.empty()) {
        llvm::outs() << " [SKIP: workload failed]\n";
        continue;
      }

      // Compile SW domain's TDG on HW domain's architecture
      BendersConfig bConfig;
      bConfig.maxIterations = 5;
      bConfig.mapperBudgetSeconds = mapperBudget;
      bConfig.mapperSeed = 42;

      BendersDriver driver(hwDom.arch, std::move(workload.kernels),
                           std::move(workload.contracts), ctx);
      auto compResult = driver.compile(bConfig);

      double crossThroughput = compResult.success
                                   ? (compResult.totalCost > 0
                                          ? 1.0 / compResult.totalCost
                                          : 0.0)
                                   : 0.0;

      // Count successful mappings
      unsigned mappedCount = 0;
      for (const auto &a : compResult.assignments) {
        if (a.mappingSuccess)
          ++mappedCount;
      }
      double successRate = compResult.assignments.empty()
                               ? 0.0
                               : static_cast<double>(mappedCount) /
                                     static_cast<double>(
                                         compResult.assignments.size()) *
                                     100.0;

      double vsNativePct =
          (swDom.nativeThroughput > 0.0)
              ? crossThroughput / swDom.nativeThroughput * 100.0
              : 0.0;

      llvm::outs() << " -> " << llvm::format("%.1f%%", vsNativePct)
                   << " native, map=" << llvm::format("%.0f%%", successRate)
                   << "\n";

      llvm::json::Object row;
      row["sw_domain"] = swDom.domain;
      row["hw_domain"] = hwDom.domain;
      row["is_native"] = (swDom.domain == hwDom.domain);
      row["mapping_success_rate"] = successRate;
      row["throughput"] = crossThroughput;
      row["native_throughput"] = swDom.nativeThroughput;
      row["throughput_vs_native_pct"] = vsNativePct;
      row["mapped_kernels"] = static_cast<int64_t>(mappedCount);
      row["total_kernels"] =
          static_cast<int64_t>(compResult.assignments.size());
      allResults.push_back(std::move(row));
    }
  }

  llvm::json::Object summary;
  summary["experiment"] = "E19: Cross-Domain Hardware Portability";
  summary["mode"] = "cross_domain";
  summary["domains"] = static_cast<int64_t>(domResults.size());
  summary["matrix_size"] =
      static_cast<int64_t>(domResults.size() * domResults.size());
  summary["results"] = std::move(allResults);

  return writeJSON(outputDir.getValue() + "/cross_domain_results.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// E20: Initial architecture sensitivity
//===----------------------------------------------------------------------===//

static bool runE20Sensitivity(mlir::MLIRContext &ctx) {
  llvm::outs() << "=== E20: Initial Architecture Sensitivity ===\n\n";

  std::string targetDomain =
      (domain == "all") ? "ai_llm" : domain.getValue();
  llvm::outs() << "Target domain: " << targetDomain << "\n";

  std::vector<std::string> archConfigs = {
      "spectral", "homogeneous_gp", "homogeneous_dsp", "random_fu",
      "oversized"};

  llvm::json::Array allResults;

  for (const auto &ac : archConfigs) {
    llvm::outs() << "\n--- Initial arch: " << ac << " ---\n";

    auto workload = buildDomainWorkload(targetDomain, ctx);
    if (workload.kernels.empty()) {
      llvm::errs() << "  Skipping: workload construction failed\n";
      continue;
    }

    auto initialArch = buildInitialArch(ac, workload.kernels.size(), ctx);
    if (initialArch.coreTypes.empty()) {
      llvm::errs() << "  Skipping: architecture build failed\n";
      continue;
    }

    llvm::outs() << "  Core types: " << initialArch.coreTypes.size() << "\n";
    for (const auto &ct : initialArch.coreTypes) {
      llvm::outs() << "    " << ct.name << ": " << ct.numInstances
                   << " instances, " << ct.totalPEs << " PEs\n";
    }

    auto startTime = std::chrono::steady_clock::now();
    auto result = runCoOpt(workload, initialArch, ctx, maxRounds, threshold,
                           verbose);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(endTime - startTime).count();

    llvm::outs() << "  Converged in " << result.rounds << " rounds: "
                 << "throughput=" << result.bestThroughput
                 << ", area=" << result.bestArea
                 << ", time=" << llvm::format("%.2f", elapsed) << "s\n";

    auto obj = coOptResultToJSON(result, ac, elapsed);
    obj["domain"] = targetDomain;
    obj["initial_config"] = ac;
    obj["initial_core_types"] =
        static_cast<int64_t>(initialArch.coreTypes.size());
    allResults.push_back(std::move(obj));
  }

  llvm::json::Object summary;
  summary["experiment"] = "E20: Initial Architecture Sensitivity";
  summary["mode"] = "sensitivity";
  summary["domain"] = targetDomain;
  summary["max_rounds"] = static_cast<int64_t>(maxRounds.getValue());
  summary["threshold"] = threshold.getValue();
  summary["configs_run"] = static_cast<int64_t>(allResults.size());
  summary["results"] = std::move(allResults);

  return writeJSON(outputDir.getValue() + "/sensitivity_results.json",
                   llvm::json::Value(std::move(summary)));
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Tapestry co-optimization experiments (E17-E20)\n");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  if (!ensureDir(outputDir))
    return 1;

  bool success = false;

  auto startTotal = std::chrono::steady_clock::now();

  if (experimentMode == "convergence") {
    success = runE17Convergence(ctx);
  } else if (experimentMode == "comparison") {
    success = runE18Comparison(ctx);
  } else if (experimentMode == "cross_domain") {
    success = runE19CrossDomain(ctx);
  } else if (experimentMode == "sensitivity") {
    success = runE20Sensitivity(ctx);
  } else {
    llvm::errs() << "Unknown mode: " << experimentMode << "\n";
    llvm::errs() << "Valid modes: convergence, comparison, cross_domain, "
                    "sensitivity\n";
    return 1;
  }

  auto endTotal = std::chrono::steady_clock::now();
  double totalElapsed =
      std::chrono::duration<double>(endTotal - startTotal).count();

  llvm::outs() << "\nTotal elapsed: " << llvm::format("%.2f", totalElapsed)
               << "s\n";
  llvm::outs() << "Status: " << (success ? "COMPLETE" : "FAILED") << "\n";
  llvm::outs() << "Results in: " << outputDir << "/\n";

  return success ? 0 : 1;
}
