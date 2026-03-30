//===-- ablation_runner.cpp - Ablation experiment infrastructure ---*- C++ -*-===//
//
// Implements the ablation experiment infrastructure that evaluates 6
// configurations (Baseline, SW-only, HW-only, Outer-only, Inner-only,
// Full-coopt) across application domains, with automated comparison
// and result reporting.
//
//===----------------------------------------------------------------------===//

#include "tapestry/co_optimizer.h"

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/TDGLowering.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tapestry {

//===----------------------------------------------------------------------===//
// AblationResult helpers
//===----------------------------------------------------------------------===//

std::vector<std::vector<double>> AblationResult::throughputMatrix() const {
  std::vector<std::vector<double>> matrix;
  matrix.reserve(configs.size());
  for (size_t ci = 0; ci < results.size(); ++ci) {
    std::vector<double> row;
    row.reserve(domains.size());
    for (size_t di = 0; di < results[ci].size(); ++di) {
      row.push_back(results[ci][di].bestThroughput);
    }
    matrix.push_back(std::move(row));
  }
  return matrix;
}

std::vector<std::vector<double>> AblationResult::areaMatrix() const {
  std::vector<std::vector<double>> matrix;
  matrix.reserve(configs.size());
  for (size_t ci = 0; ci < results.size(); ++ci) {
    std::vector<double> row;
    row.reserve(domains.size());
    for (size_t di = 0; di < results[ci].size(); ++di) {
      row.push_back(results[ci][di].bestArea);
    }
    matrix.push_back(std::move(row));
  }
  return matrix;
}

//===----------------------------------------------------------------------===//
// buildAblationConfigs
//===----------------------------------------------------------------------===//

std::vector<AblationConfig> buildAblationConfigs() {
  std::vector<AblationConfig> configs;
  configs.reserve(6);

  // Baseline: all layers disabled
  configs.push_back({"Baseline", false, false, false, false});

  // SW-only: SW-Outer + SW-Inner enabled, HW disabled
  configs.push_back({"SW-only", true, false, true, false});

  // HW-only: HW-Outer + HW-Inner enabled, SW disabled
  configs.push_back({"HW-only", false, true, false, true});

  // Outer-only: SW-Outer + HW-Inner (cross: outer-SW with inner-HW)
  configs.push_back({"Outer-only", true, false, false, true});

  // Inner-only: SW-Inner + HW-Inner (inner layers only, outer disabled)
  // SW-Outer disabled, HW-Outer disabled, SW-Inner enabled, HW-Inner enabled
  configs.push_back({"Inner-only", false, false, true, true});

  // Full-coopt: all layers enabled
  configs.push_back({"Full-coopt", true, true, true, true});

  return configs;
}

//===----------------------------------------------------------------------===//
// applyAblationConfig
//===----------------------------------------------------------------------===//

void applyAblationConfig(CoOptOptions &opts, const AblationConfig &config) {
  // SW-Outer (TDG-level transforms):
  // When disabled, set maxIterations to 0 to skip transforms and just do
  // a single BendersDriver evaluation for baseline metric.
  if (!config.enableSW) {
    opts.swOpts.maxIterations = 0;
  }

  // HW-Outer (topology search):
  // When disabled, set maxIterations to 0 to use fallback single-core-type
  // topology.
  if (!config.enableHW) {
    opts.hwOuterOpts.maxIterations = 0;
  }

  // SW-Inner (L2 iterative re-mapping):
  // When disabled, reduce Benders iterations to single-pass (no infeasibility
  // cuts). The bendersConfig.maxIterations controls the L2 re-mapping loop.
  if (!config.enableSWInner) {
    opts.swOpts.compilerConfig.maxIterations = 1;
  }

  // HW-Inner (per-core Bayesian optimization):
  // When disabled, turn off Tier-B (BO + mapper) so only Tier-A analytical
  // derivation runs.
  if (!config.enableHWInner) {
    opts.hwInnerOpts.tier2Enabled = false;
    opts.hwInnerOpts.maxInnerIter = 0;
  }
}

//===----------------------------------------------------------------------===//
// runAblation
//===----------------------------------------------------------------------===//

/// Workload for a single domain: kernels + contracts for co-optimization.
struct DomainWorkload {
  std::vector<loom::tapestry::KernelDesc> kernels;
  std::vector<loom::tapestry::ContractSpec> contracts;
};

/// Build a synthetic workload for the given domain name.
/// Uses the same domain definitions as the experiment tool.
static DomainWorkload
buildDomainWorkload(const std::string &domainName, mlir::MLIRContext &ctx) {
  namespace lt = loom::tapestry;
  DomainWorkload w;

  struct KernelSpec {
    std::string name;
    bool isMac;
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
    return w;
  }

  for (const auto &ks : kSpecs) {
    lt::KernelDesc kd = ks.isMac
                            ? lt::createSyntheticMacKernel(ks.name, ctx)
                            : lt::createSyntheticAddKernel(ks.name, ctx);
    if (!kd.dfgModule)
      return DomainWorkload{};
    w.kernels.push_back(std::move(kd));
  }

  if (!lt::lowerKernelsToDFG(w.kernels, ctx))
    return DomainWorkload{};

  for (const auto &ce : cEdges) {
    lt::ContractSpec c;
    c.producerKernel = ce.producer;
    c.consumerKernel = ce.consumer;
    c.dataType = "i32";
    c.elementCount = ce.elementCount;
    c.bandwidthBytesPerCycle = ce.bw;
    w.contracts.push_back(std::move(c));
  }

  return w;
}

/// Build an initial architecture using the spectral configuration.
static loom::tapestry::SystemArchitecture
buildAblationArch(const std::string &config, unsigned numKernels,
                  mlir::MLIRContext &ctx) {
  namespace lt = loom::tapestry;

  std::vector<lt::CoreTypeSpec> specs;

  lt::CoreTypeSpec gpSpec;
  gpSpec.name = "gp_core";
  gpSpec.meshRows = 2;
  gpSpec.meshCols = 2;
  gpSpec.numInstances = std::max(1u, numKernels / 2);
  gpSpec.includeMultiplier = false;
  gpSpec.includeComparison = true;
  gpSpec.includeMemory = false;
  gpSpec.spmSizeBytes = 4096;
  specs.push_back(gpSpec);

  lt::CoreTypeSpec dspSpec;
  dspSpec.name = "dsp_core";
  dspSpec.meshRows = 2;
  dspSpec.meshCols = 2;
  dspSpec.numInstances = std::max(1u, (numKernels + 1) / 2);
  dspSpec.includeMultiplier = true;
  dspSpec.includeComparison = false;
  dspSpec.includeMemory = false;
  dspSpec.spmSizeBytes = 8192;
  specs.push_back(dspSpec);

  return lt::buildArchitecture(config + "_ablation", specs, ctx);
}

AblationResult runAblation(const CoOptOptions &templateOpts,
                           mlir::MLIRContext *ctx,
                           const std::vector<std::string> &domainNames,
                           const std::string &archConfig,
                           bool verbose) {
  AblationResult ablResult;
  ablResult.configs = buildAblationConfigs();
  ablResult.domains = domainNames;
  ablResult.results.resize(ablResult.configs.size());
  for (auto &row : ablResult.results)
    row.resize(domainNames.size());

  if (!ctx) {
    ablResult.success = false;
    return ablResult;
  }

  bool allSucceeded = true;

  for (size_t di = 0; di < domainNames.size(); ++di) {
    if (verbose)
      llvm::errs() << "runAblation: domain '" << domainNames[di] << "'\n";

    for (size_t ci = 0; ci < ablResult.configs.size(); ++ci) {
      const auto &config = ablResult.configs[ci];
      if (verbose)
        llvm::errs() << "  config '" << config.name << "'\n";

      // Build a fresh workload per config (co_optimize may consume/modify).
      auto workload = buildDomainWorkload(domainNames[di], *ctx);
      if (workload.kernels.empty()) {
        if (verbose)
          llvm::errs() << "    skipping: workload construction failed\n";
        allSucceeded = false;
        continue;
      }

      auto arch = buildAblationArch(
          archConfig, static_cast<unsigned>(workload.kernels.size()), *ctx);
      if (arch.coreTypes.empty()) {
        if (verbose)
          llvm::errs() << "    skipping: architecture build failed\n";
        allSucceeded = false;
        continue;
      }

      // Build co-opt options from the template and apply the ablation config.
      CoOptOptions coOpts = templateOpts;
      coOpts.verbose = verbose;
      applyAblationConfig(coOpts, config);

      CoOptResult result = co_optimize(workload.kernels, workload.contracts,
                                       arch, coOpts, ctx);

      if (verbose)
        llvm::errs() << "    throughput=" << result.bestThroughput
                     << " area=" << result.bestArea
                     << " rounds=" << result.rounds << "\n";

      ablResult.results[ci][di] = std::move(result);

      if (!ablResult.results[ci][di].success)
        allSucceeded = false;
    }
  }

  ablResult.success = allSucceeded;
  return ablResult;
}

//===----------------------------------------------------------------------===//
// ablationResultToJSON
//===----------------------------------------------------------------------===//

/// Compute the geometric mean of a vector of positive values.
/// Returns 0.0 if any value is non-positive or the vector is empty.
static double geometricMean(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  double logSum = 0.0;
  for (double v : values) {
    if (v <= 0.0)
      return 0.0;
    logSum += std::log(v);
  }
  return std::exp(logSum / static_cast<double>(values.size()));
}

llvm::json::Value ablationResultToJSON(const AblationResult &result) {
  llvm::json::Object root;
  root["experiment"] = "ablation";

  // Config names
  llvm::json::Array configNames;
  for (const auto &c : result.configs)
    configNames.push_back(c.name);
  root["configs"] = std::move(configNames);

  // Domain names
  llvm::json::Array domNames;
  for (const auto &d : result.domains)
    domNames.push_back(d);
  root["domains"] = std::move(domNames);

  // Matrix: flat array of per-cell objects
  llvm::json::Array matrix;
  for (size_t ci = 0; ci < result.configs.size(); ++ci) {
    for (size_t di = 0; di < result.domains.size(); ++di) {
      const auto &cell = result.results[ci][di];
      llvm::json::Object entry;
      entry["config"] = result.configs[ci].name;
      entry["domain"] = result.domains[di];
      entry["throughput"] = cell.bestThroughput;
      entry["area"] = cell.bestArea;
      entry["rounds"] = static_cast<int64_t>(cell.rounds);
      entry["success"] = cell.success;
      matrix.push_back(std::move(entry));
    }
  }
  root["matrix"] = std::move(matrix);

  // Summary section
  llvm::json::Object summary;

  // Per-config summary
  llvm::json::Array perConfig;
  for (size_t ci = 0; ci < result.configs.size(); ++ci) {
    llvm::json::Object configSummary;
    configSummary["config"] = result.configs[ci].name;

    std::vector<double> throughputs;
    std::vector<double> areas;
    double totalRounds = 0.0;
    unsigned count = 0;

    for (size_t di = 0; di < result.domains.size(); ++di) {
      const auto &cell = result.results[ci][di];
      if (cell.bestThroughput > 0.0)
        throughputs.push_back(cell.bestThroughput);
      if (cell.bestArea > 0.0 &&
          cell.bestArea < std::numeric_limits<double>::infinity())
        areas.push_back(cell.bestArea);
      totalRounds += static_cast<double>(cell.rounds);
      ++count;
    }

    configSummary["geo_mean_throughput"] = geometricMean(throughputs);
    configSummary["geo_mean_area"] = geometricMean(areas);
    configSummary["avg_rounds"] =
        count > 0 ? totalRounds / static_cast<double>(count) : 0.0;
    perConfig.push_back(std::move(configSummary));
  }
  summary["per_config"] = std::move(perConfig);

  // Per-domain summary
  llvm::json::Array perDomain;
  for (size_t di = 0; di < result.domains.size(); ++di) {
    llvm::json::Object domSummary;
    domSummary["domain"] = result.domains[di];

    double bestThroughput = 0.0;
    double bestArea = std::numeric_limits<double>::infinity();
    std::string bestThroughputConfig;
    std::string bestAreaConfig;

    for (size_t ci = 0; ci < result.configs.size(); ++ci) {
      const auto &cell = result.results[ci][di];
      if (cell.bestThroughput > bestThroughput) {
        bestThroughput = cell.bestThroughput;
        bestThroughputConfig = result.configs[ci].name;
      }
      if (cell.bestArea < bestArea) {
        bestArea = cell.bestArea;
        bestAreaConfig = result.configs[ci].name;
      }
    }

    domSummary["best_config"] = bestThroughputConfig;
    domSummary["best_throughput"] = bestThroughput;
    domSummary["best_area"] = bestArea;
    perDomain.push_back(std::move(domSummary));
  }
  summary["per_domain"] = std::move(perDomain);

  root["summary"] = std::move(summary);
  return llvm::json::Value(std::move(root));
}

//===----------------------------------------------------------------------===//
// generateComparisonReport
//===----------------------------------------------------------------------===//

std::string generateComparisonReport(const AblationResult &result) {
  std::ostringstream report;

  report << "=== Ablation Experiment Comparison Report ===\n\n";

  // Per-config table
  report << "--- Per-Configuration Summary ---\n";
  report << std::left << std::setw(14) << "Config"
         << std::right << std::setw(18) << "GeoMean Throughput"
         << std::setw(14) << "GeoMean Area"
         << std::setw(12) << "Avg Rounds" << "\n";
  report << std::string(58, '-') << "\n";

  // Compute per-config aggregates
  struct ConfigAggregate {
    std::string name;
    double geoMeanThroughput = 0.0;
    double geoMeanArea = 0.0;
    double avgRounds = 0.0;
  };
  std::vector<ConfigAggregate> configAggs;

  for (size_t ci = 0; ci < result.configs.size(); ++ci) {
    ConfigAggregate agg;
    agg.name = result.configs[ci].name;

    std::vector<double> throughputs;
    std::vector<double> areas;
    double totalRounds = 0.0;
    unsigned count = 0;

    for (size_t di = 0; di < result.domains.size(); ++di) {
      const auto &cell = result.results[ci][di];
      if (cell.bestThroughput > 0.0)
        throughputs.push_back(cell.bestThroughput);
      if (cell.bestArea > 0.0 &&
          cell.bestArea < std::numeric_limits<double>::infinity())
        areas.push_back(cell.bestArea);
      totalRounds += static_cast<double>(cell.rounds);
      ++count;
    }

    agg.geoMeanThroughput = geometricMean(throughputs);
    agg.geoMeanArea = geometricMean(areas);
    agg.avgRounds = count > 0 ? totalRounds / static_cast<double>(count) : 0.0;
    configAggs.push_back(agg);

    report << std::left << std::setw(14) << agg.name
           << std::right << std::setw(18) << std::fixed << std::setprecision(6)
           << agg.geoMeanThroughput
           << std::setw(14) << std::setprecision(2) << agg.geoMeanArea
           << std::setw(12) << std::setprecision(1) << agg.avgRounds << "\n";
  }
  report << "\n";

  // Per-domain table
  report << "--- Per-Domain Summary ---\n";
  report << std::left << std::setw(18) << "Domain"
         << std::setw(16) << "Best(throughput)"
         << std::setw(14) << "Best(area)" << "\n";
  report << std::string(48, '-') << "\n";

  for (size_t di = 0; di < result.domains.size(); ++di) {
    double bestThroughput = 0.0;
    double bestArea = std::numeric_limits<double>::infinity();
    std::string bestThroughputConfig;
    std::string bestAreaConfig;

    for (size_t ci = 0; ci < result.configs.size(); ++ci) {
      const auto &cell = result.results[ci][di];
      if (cell.bestThroughput > bestThroughput) {
        bestThroughput = cell.bestThroughput;
        bestThroughputConfig = result.configs[ci].name;
      }
      if (cell.bestArea < bestArea) {
        bestArea = cell.bestArea;
        bestAreaConfig = result.configs[ci].name;
      }
    }

    report << std::left << std::setw(18) << result.domains[di]
           << std::setw(16) << bestThroughputConfig
           << std::setw(14) << bestAreaConfig << "\n";
  }
  report << "\n";

  // Relative improvement section
  report << "--- Relative Improvement ---\n";

  // Find Baseline and Full-coopt indices
  int baselineIdx = -1;
  int fullCooptIdx = -1;
  int swOnlyIdx = -1;
  int hwOnlyIdx = -1;

  for (size_t ci = 0; ci < result.configs.size(); ++ci) {
    if (result.configs[ci].name == "Baseline")
      baselineIdx = static_cast<int>(ci);
    else if (result.configs[ci].name == "Full-coopt")
      fullCooptIdx = static_cast<int>(ci);
    else if (result.configs[ci].name == "SW-only")
      swOnlyIdx = static_cast<int>(ci);
    else if (result.configs[ci].name == "HW-only")
      hwOnlyIdx = static_cast<int>(ci);
  }

  auto pctStr = [](double val) -> std::string {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << val << "%";
    return ss.str();
  };

  if (baselineIdx >= 0 && fullCooptIdx >= 0) {
    double bTp = configAggs[baselineIdx].geoMeanThroughput;
    double fTp = configAggs[fullCooptIdx].geoMeanThroughput;
    double bArea = configAggs[baselineIdx].geoMeanArea;
    double fArea = configAggs[fullCooptIdx].geoMeanArea;

    double throughputGain = (bTp > 0.0) ? (fTp - bTp) / bTp * 100.0 : 0.0;
    double areaReduction = (bArea > 0.0) ? (bArea - fArea) / bArea * 100.0 : 0.0;

    report << "Full-coopt vs Baseline:\n";
    report << "  throughput gain: " << pctStr(throughputGain) << "\n";
    report << "  area reduction: " << pctStr(areaReduction) << "\n";
  }

  if (swOnlyIdx >= 0 && fullCooptIdx >= 0) {
    double swTp = configAggs[swOnlyIdx].geoMeanThroughput;
    double fTp = configAggs[fullCooptIdx].geoMeanThroughput;
    double swArea = configAggs[swOnlyIdx].geoMeanArea;
    double fArea = configAggs[fullCooptIdx].geoMeanArea;

    double throughputGain = (swTp > 0.0) ? (fTp - swTp) / swTp * 100.0 : 0.0;
    double areaReduction = (swArea > 0.0) ? (swArea - fArea) / swArea * 100.0 : 0.0;

    report << "Full-coopt vs SW-only:\n";
    report << "  throughput gain: " << pctStr(throughputGain) << "\n";
    report << "  area reduction: " << pctStr(areaReduction) << "\n";
  }

  if (hwOnlyIdx >= 0 && fullCooptIdx >= 0) {
    double hwTp = configAggs[hwOnlyIdx].geoMeanThroughput;
    double fTp = configAggs[fullCooptIdx].geoMeanThroughput;
    double hwArea = configAggs[hwOnlyIdx].geoMeanArea;
    double fArea = configAggs[fullCooptIdx].geoMeanArea;

    double throughputGain = (hwTp > 0.0) ? (fTp - hwTp) / hwTp * 100.0 : 0.0;
    double areaReduction = (hwArea > 0.0) ? (hwArea - fArea) / hwArea * 100.0 : 0.0;

    report << "Full-coopt vs HW-only:\n";
    report << "  throughput gain: " << pctStr(throughputGain) << "\n";
    report << "  area reduction: " << pctStr(areaReduction) << "\n";
  }

  return report.str();
}

} // namespace tapestry
