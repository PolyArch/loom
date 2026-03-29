//===-- co_optimizer.cpp - SW-HW co-optimization loop -------------*- C++ -*-===//
//
// Implements the top-level co-optimization that alternates software
// optimization (C10 TDGOptimizer) and hardware optimization (C11 OUTER-HW +
// C12 INNER-HW). Each round fixes one side and optimizes the other, using
// TDC contracts as the shared communication channel.
//
//===----------------------------------------------------------------------===//

#include "tapestry/co_optimizer.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>

// Namespace aliases to avoid ambiguity between ::tapestry and loom::tapestry.
namespace lt = loom::tapestry;

namespace tapestry {

//===----------------------------------------------------------------------===//
// Pareto frontier helpers
//===----------------------------------------------------------------------===//

/// Returns true if point A dominates point B (higher throughput AND lower area).
static bool dominates(const ParetoPoint &a, const ParetoPoint &b) {
  return (a.throughput >= b.throughput && a.area <= b.area) &&
         (a.throughput > b.throughput || a.area < b.area);
}

void addParetoPoint(std::vector<ParetoPoint> &frontier,
                    const ParetoPoint &candidate) {
  // Check if any existing point dominates the candidate.
  for (const auto &existing : frontier) {
    if (dominates(existing, candidate))
      return; // Candidate is dominated; skip.
  }

  // Remove existing points dominated by the candidate.
  frontier.erase(
      std::remove_if(frontier.begin(), frontier.end(),
                     [&candidate](const ParetoPoint &p) {
                       return dominates(candidate, p);
                     }),
      frontier.end());

  frontier.push_back(candidate);
}

//===----------------------------------------------------------------------===//
// extractKernelProfiles
//===----------------------------------------------------------------------===//

std::vector<loom::KernelProfile>
extractKernelProfiles(const std::vector<lt::KernelDesc> &kernels) {
  std::vector<loom::KernelProfile> profiles;
  profiles.reserve(kernels.size());

  for (const auto &kd : kernels) {
    loom::KernelProfile profile;
    profile.name = kd.name;
    profile.estimatedSPMBytes = kd.requiredMemoryBytes;

    // Derive a coarse op histogram from required FU/PE counts.
    // The actual DFG analysis populates this in a real pipeline; here we
    // generate a reasonable approximation.
    if (kd.requiredFUs > 0) {
      profile.requiredOps["arith.addi"] = kd.requiredFUs;
    }
    if (kd.requiredPEs > 0) {
      profile.estimatedMinII = 1;
      profile.estimatedComputeCycles =
          static_cast<double>(kd.requiredPEs) * 2.0;
    }

    profiles.push_back(std::move(profile));
  }

  return profiles;
}

//===----------------------------------------------------------------------===//
// buildDefaultArchitecture
//===----------------------------------------------------------------------===//

lt::SystemArchitecture
buildDefaultArchitecture(const std::vector<loom::KernelProfile> &profiles) {
  lt::SystemArchitecture arch;
  arch.name = "default_auto";

  // Create a single balanced core type with enough capacity for all kernels.
  lt::CoreTypeDesc defaultCore;
  defaultCore.name = "balanced_core";
  defaultCore.numInstances =
      std::max(1u, static_cast<unsigned>(profiles.size()));

  // Compute resource lower bounds from kernel profiles.
  unsigned maxPEs = 4;
  unsigned totalFUs = 0;
  unsigned maxSPM = 4096;

  for (const auto &p : profiles) {
    unsigned opCount = p.totalOpCount();
    unsigned neededPEs =
        static_cast<unsigned>(std::ceil(std::sqrt(
            static_cast<double>(std::max(opCount, 1u)))));
    maxPEs = std::max(maxPEs, neededPEs);
    totalFUs += opCount;
    maxSPM = std::max(maxSPM,
                      static_cast<unsigned>(p.estimatedSPMBytes));
  }

  defaultCore.totalPEs = maxPEs;
  defaultCore.totalFUs = totalFUs;
  defaultCore.spmSizeBytes = maxSPM;

  arch.coreTypes.push_back(std::move(defaultCore));
  return arch;
}

//===----------------------------------------------------------------------===//
// Contract conversions
//===----------------------------------------------------------------------===//

std::vector<lt::ContractSpec>
toLoomTapestryContracts(
    const std::vector<loom::ContractSpec> &loomContracts) {
  std::vector<lt::ContractSpec> result;
  result.reserve(loomContracts.size());

  for (const auto &lc : loomContracts) {
    lt::ContractSpec tc;
    tc.producerKernel = lc.producerKernel;
    tc.consumerKernel = lc.consumerKernel;
    tc.dataType = lc.dataTypeName;
    tc.elementCount = 0;
    if (lc.productionRate.has_value())
      tc.elementCount = static_cast<uint64_t>(*lc.productionRate);
    tc.bandwidthBytesPerCycle = 0;
    tc.producerCoreType = -1;
    tc.consumerCoreType = -1;
    result.push_back(std::move(tc));
  }

  return result;
}

std::vector<loom::ContractSpec>
fromLoomTapestryContracts(
    const std::vector<lt::ContractSpec> &tapContracts) {
  std::vector<loom::ContractSpec> result;
  result.reserve(tapContracts.size());

  for (const auto &tc : tapContracts) {
    loom::ContractSpec lc;
    lc.producerKernel = tc.producerKernel;
    lc.consumerKernel = tc.consumerKernel;
    lc.dataTypeName = tc.dataType;
    if (tc.elementCount > 0)
      lc.productionRate = static_cast<int64_t>(tc.elementCount);
    result.push_back(std::move(lc));
  }

  return result;
}

//===----------------------------------------------------------------------===//
// updateContractsFromSW
//===----------------------------------------------------------------------===//

void updateContractsFromSW(
    std::vector<lt::ContractSpec> &contracts,
    const TDGOptimizeResult &swResult) {
  // Propagate achieved rates from the SW optimization result back into the
  // contracts. The BendersResult contains L2Assignment with core assignments
  // that indicate which core each kernel ended up on.
  const auto &assignments = swResult.compilationResult.assignments;

  // Build kernel -> core type mapping from assignments.
  std::unordered_map<std::string, int> kernelCoreType;
  for (const auto &a : assignments) {
    if (a.mappingSuccess) {
      kernelCoreType[a.kernelName] = a.coreTypeIndex;
    }
  }

  // Update contract core type assignments based on actual mapping.
  for (auto &c : contracts) {
    auto producerIt = kernelCoreType.find(c.producerKernel);
    if (producerIt != kernelCoreType.end())
      c.producerCoreType = producerIt->second;

    auto consumerIt = kernelCoreType.find(c.consumerKernel);
    if (consumerIt != kernelCoreType.end())
      c.consumerCoreType = consumerIt->second;

    // Compute communication cost: if producer and consumer are on
    // different core types, there is inter-core communication overhead.
    if (c.producerCoreType >= 0 && c.consumerCoreType >= 0 &&
        c.producerCoreType != c.consumerCoreType) {
      c.communicationCost =
          static_cast<double>(c.elementCount) *
          static_cast<double>(c.bandwidthBytesPerCycle + 1);
    } else {
      c.communicationCost = 0.0;
    }
  }

  // Also update the optimized contracts from the SW result if available.
  if (!swResult.optimizedContracts.empty() &&
      swResult.optimizedContracts.size() == contracts.size()) {
    for (size_t idx = 0; idx < contracts.size(); ++idx) {
      contracts[idx].elementCount =
          swResult.optimizedContracts[idx].elementCount;
      contracts[idx].bandwidthBytesPerCycle =
          swResult.optimizedContracts[idx].bandwidthBytesPerCycle;
    }
  }
}

//===----------------------------------------------------------------------===//
// computeSystemArea
//===----------------------------------------------------------------------===//

double computeSystemArea(
    const loom::HWOuterOptimizerResult &outerResult,
    const std::vector<loom::ADGOptResult> &innerResults) {
  double totalArea = 0.0;

  // Sum per-core-type area (area * instance count).
  const auto &entries = outerResult.topology.coreLibrary.entries;
  for (size_t typeIdx = 0; typeIdx < entries.size(); ++typeIdx) {
    double coreArea = 0.0;
    if (typeIdx < innerResults.size() && innerResults[typeIdx].success) {
      coreArea = innerResults[typeIdx].areaEstimate;
    } else {
      // Fallback: estimate from PE count.
      coreArea = static_cast<double>(entries[typeIdx].minPEs) * 10.0;
    }
    totalArea += coreArea * static_cast<double>(entries[typeIdx].instanceCount);
  }

  // Add NoC and shared memory area overhead.
  // NoC area scales roughly as mesh_size^2 * bandwidth.
  const auto &topo = outerResult.topology;
  double nocArea = static_cast<double>(topo.meshRows * topo.meshCols) *
                   static_cast<double>(topo.nocBandwidth) * 5.0;
  totalArea += nocArea;

  // L2 memory area (proportional to total size).
  double l2Area = static_cast<double>(topo.l2TotalSizeKB) * 0.1;
  totalArea += l2Area;

  return totalArea;
}

//===----------------------------------------------------------------------===//
// buildArchFromHWResults
//===----------------------------------------------------------------------===//

lt::SystemArchitecture buildArchFromHWResults(
    const loom::HWOuterOptimizerResult &outerResult,
    const std::vector<loom::ADGOptResult> &innerResults,
    mlir::MLIRContext *ctx) {
  lt::SystemArchitecture arch;
  arch.name = "coopt_arch";

  const auto &entries = outerResult.topology.coreLibrary.entries;
  for (size_t typeIdx = 0; typeIdx < entries.size(); ++typeIdx) {
    const auto &entry = entries[typeIdx];

    lt::CoreTypeDesc coreType;
    coreType.name = "core_type_" + std::to_string(entry.typeIndex);
    coreType.numInstances = entry.instanceCount;

    // Populate resource estimates from INNER-HW results.
    if (typeIdx < innerResults.size() && innerResults[typeIdx].success) {
      const auto &params = innerResults[typeIdx].params;
      coreType.totalPEs = params.totalPEs();
      coreType.totalFUs =
          static_cast<unsigned>(params.fuRepertoire.size()) *
          params.totalPEs();
      coreType.spmSizeBytes = params.spmSizeKB * 1024;
    } else {
      coreType.totalPEs = entry.minPEs;
      coreType.totalFUs = entry.minPEs;
      coreType.spmSizeBytes = entry.minSPMKB * 1024;
    }

    // Note: adgModule is left as nullptr; the MLIR text is in
    // innerResults[typeIdx].adgMLIR and can be parsed on demand.
    arch.coreTypes.push_back(std::move(coreType));
  }

  return arch;
}

//===----------------------------------------------------------------------===//
// buildLoomContractsForHW (file-local helper)
//===----------------------------------------------------------------------===//

/// Convert lt::ContractSpec list to loom::ContractSpec list
/// for the HWOuterOptimizer (which uses the loom:: namespace).
static std::vector<loom::ContractSpec>
buildLoomContractsForHW(
    const std::vector<lt::ContractSpec> &tapContracts) {
  std::vector<loom::ContractSpec> result;
  result.reserve(tapContracts.size());

  for (const auto &tc : tapContracts) {
    loom::ContractSpec lc;
    lc.producerKernel = tc.producerKernel;
    lc.consumerKernel = tc.consumerKernel;
    lc.dataTypeName = tc.dataType;
    if (tc.elementCount > 0)
      lc.productionRate = static_cast<int64_t>(tc.elementCount);
    if (tc.bandwidthBytesPerCycle > 0)
      lc.consumptionRate =
          static_cast<int64_t>(tc.bandwidthBytesPerCycle);
    result.push_back(std::move(lc));
  }

  return result;
}

//===----------------------------------------------------------------------===//
// runSWStep -- fixes hardware, optimizes software
//===----------------------------------------------------------------------===//

/// Run the SW optimization step (TDGOptimizer) with the current architecture.
static TDGOptimizeResult
runSWStep(std::vector<lt::KernelDesc> &kernels,
          std::vector<lt::ContractSpec> &contracts,
          const lt::SystemArchitecture &arch,
          const CoOptOptions &coOpts,
          mlir::MLIRContext *ctx) {
  TDGOptimizer swOptimizer(coOpts.swOpts, *ctx);
  return swOptimizer.optimize(kernels, contracts, arch);
}

//===----------------------------------------------------------------------===//
// HWStepResult
//===----------------------------------------------------------------------===//

/// Result struct for the HW optimization step.
struct HWStepResult {
  loom::HWOuterOptimizerResult outerResult;
  std::vector<loom::ADGOptResult> innerResults;
  lt::SystemArchitecture newArch;
  double totalArea = 0.0;
  bool success = false;
};

//===----------------------------------------------------------------------===//
// runHWStep -- fixes software, optimizes hardware
//===----------------------------------------------------------------------===//

/// Run the HW optimization step (OUTER-HW + INNER-HW).
static HWStepResult
runHWStep(const std::vector<lt::KernelDesc> &kernels,
          const std::vector<lt::ContractSpec> &contracts,
          const std::vector<loom::KernelProfile> &profiles,
          const CoOptOptions &coOpts,
          mlir::MLIRContext *ctx) {
  HWStepResult hwStep;

  // Convert contracts for HW optimizer.
  auto loomContracts = buildLoomContractsForHW(contracts);

  // OUTER-HW: determine core type library and system topology.
  loom::HWOuterOptimizer outerOpt(coOpts.hwOuterOpts);
  hwStep.outerResult = outerOpt.optimize(loomContracts, profiles);

  if (!hwStep.outerResult.success) {
    // If OUTER-HW fails (e.g., Python script unavailable), build a
    // fallback single-core-type topology from kernel profiles.
    if (coOpts.verbose) {
      llvm::errs() << "  OUTER-HW failed ("
                   << hwStep.outerResult.diagnostics
                   << "); using fallback topology.\n";
    }

    // Build a minimal fallback topology.
    hwStep.outerResult.success = true;
    auto &topo = hwStep.outerResult.topology;
    topo.nocTopology = "mesh";
    topo.meshRows = 2;
    topo.meshCols = 2;
    topo.nocBandwidth = 1;
    topo.l2TotalSizeKB = 256;
    topo.l2BankCount = 4;

    loom::CoreTypeLibraryEntry fallbackEntry;
    fallbackEntry.typeIndex = 0;
    fallbackEntry.role = loom::CoreRole::BALANCED;
    fallbackEntry.instanceCount =
        std::max(1u, static_cast<unsigned>(kernels.size()));
    fallbackEntry.minPEs = 4;
    fallbackEntry.minSPMKB = 4;
    for (const auto &kd : kernels)
      fallbackEntry.assignedKernels.push_back(kd.name);
    topo.coreLibrary.entries.push_back(fallbackEntry);
  }

  // INNER-HW: optimize each core type's ADG.
  hwStep.innerResults = loom::optimizeAllCoreTypes(
      hwStep.outerResult.topology.coreLibrary,
      profiles, ctx, coOpts.hwInnerOpts);

  // Compute total system area.
  hwStep.totalArea = computeSystemArea(hwStep.outerResult, hwStep.innerResults);

  // Build the new SystemArchitecture.
  hwStep.newArch =
      buildArchFromHWResults(hwStep.outerResult, hwStep.innerResults, ctx);

  hwStep.success = true;
  return hwStep;
}

//===----------------------------------------------------------------------===//
// Hard upper bound for maxRounds
//===----------------------------------------------------------------------===//

static constexpr unsigned kMaxRoundsHardCap = 20;

//===----------------------------------------------------------------------===//
// co_optimize -- main entry point
//===----------------------------------------------------------------------===//

CoOptResult co_optimize(
    std::vector<lt::KernelDesc> kernels,
    std::vector<lt::ContractSpec> contracts,
    const lt::SystemArchitecture &initialArch,
    const CoOptOptions &coOpts,
    mlir::MLIRContext *ctx) {
  CoOptResult result;
  std::ostringstream diag;

  if (kernels.empty()) {
    result.success = false;
    result.diagnostics = "No kernels provided for co-optimization";
    return result;
  }

  if (!ctx) {
    result.success = false;
    result.diagnostics = "MLIRContext is null";
    return result;
  }

  // Clamp maxRounds to the hard upper bound.
  unsigned effectiveMaxRounds = coOpts.maxRounds;
  if (effectiveMaxRounds > kMaxRoundsHardCap) {
    if (coOpts.verbose)
      llvm::errs() << "WARNING: maxRounds=" << effectiveMaxRounds
                   << " exceeds hard cap " << kMaxRoundsHardCap
                   << "; clamping to " << kMaxRoundsHardCap << "\n";
    diag << "maxRounds clamped from " << effectiveMaxRounds
         << " to " << kMaxRoundsHardCap << "\n";
    effectiveMaxRounds = kMaxRoundsHardCap;
  }

  if (coOpts.verbose)
    llvm::errs() << "Co-optimization: starting with "
                 << kernels.size() << " kernels, "
                 << contracts.size() << " contracts, "
                 << "max " << effectiveMaxRounds << " rounds"
                 << (coOpts.enableSW ? "" : " [SW disabled]")
                 << (coOpts.enableHW ? "" : " [HW disabled]")
                 << "\n";

  // Build effective sub-options with inner-layer flags applied.
  CoOptOptions effectiveOpts = coOpts;
  effectiveOpts.maxRounds = effectiveMaxRounds;

  if (!coOpts.enableSWInner) {
    effectiveOpts.swOpts.maxIterations = 1;
  }
  if (!coOpts.enableHWInner) {
    effectiveOpts.hwInnerOpts.tier2Enabled = false;
  }

  // Extract kernel profiles for HW optimizer.
  auto profiles = extractKernelProfiles(kernels);

  // Use provided architecture or derive a default one.
  lt::SystemArchitecture currentArch = initialArch;

  // Warm-start detection: if the caller provides a non-empty architecture
  // AND contracts with assigned core types (producerCoreType >= 0), the
  // contracts came from a prior optimization. Skip buildDefaultArchitecture
  // and run an initial SW evaluation to establish baseline throughput.
  bool warmStart = false;
  if (!currentArch.coreTypes.empty()) {
    bool hasAssignedContracts = false;
    for (const auto &c : contracts) {
      if (c.elementCount > 0 && c.producerCoreType >= 0) {
        hasAssignedContracts = true;
        break;
      }
    }
    if (hasAssignedContracts) {
      warmStart = true;
    }
  }

  if (currentArch.coreTypes.empty()) {
    currentArch = buildDefaultArchitecture(profiles);
    if (coOpts.verbose)
      llvm::errs() << "  Derived default architecture: "
                   << currentArch.coreTypes.size() << " core type(s), "
                   << currentArch.coreTypes[0].numInstances
                   << " instance(s)\n";
  }

  // Initialize convergence monitor.
  ConvergenceMonitor monitor;
  monitor.improvementThreshold = coOpts.improvementThreshold;

  std::vector<lt::KernelDesc> bestKernels = kernels;
  std::vector<lt::ContractSpec> bestContracts = contracts;
  lt::SystemArchitecture bestArchSaved = currentArch;
  lt::BendersResult bestBenders;

  // Track last-round metrics for carry-forward when a step is disabled.
  double lastSwThroughput = 0.0;
  double lastHwArea = std::numeric_limits<double>::infinity();
  unsigned lastHwCoreTypes = 0;

  // Warm-start: run one SW evaluation pass to establish baseline (round 0).
  if (warmStart) {
    result.warmStartUsed = true;
    if (coOpts.verbose)
      llvm::errs() << "  Warm-start: evaluating baseline with "
                   << "pre-existing architecture + contracts\n";

    TDGOptimizeResult warmSwResult =
        runSWStep(kernels, contracts, currentArch, effectiveOpts, ctx);

    if (warmSwResult.success) {
      lastSwThroughput = warmSwResult.bestThroughput;
      kernels = warmSwResult.optimizedKernels;
      contracts = warmSwResult.optimizedContracts;
      updateContractsFromSW(contracts, warmSwResult);
      bestKernels = kernels;
      bestContracts = contracts;
      bestBenders = warmSwResult.compilationResult;

      // Record round 0.
      CoOptResult::RoundRecord r0;
      r0.round = 0;
      r0.swThroughput = lastSwThroughput;
      r0.hwArea = std::numeric_limits<double>::infinity();
      r0.swTransforms =
          static_cast<unsigned>(warmSwResult.transformHistory.size());
      r0.hwCoreTypes =
          static_cast<unsigned>(currentArch.coreTypes.size());
      r0.improved = true;
      r0.reason = "warm-start baseline";
      result.history.push_back(r0);

      // Seed the convergence monitor.
      monitor.bestThroughput = lastSwThroughput;

      diag << "round,swThroughput,hwArea,swTransforms,hwCoreTypes,"
           << "improved,reason\n";
      diag << "0," << lastSwThroughput << ",inf,"
           << r0.swTransforms << "," << r0.hwCoreTypes
           << ",true,warm-start baseline\n";

      if (coOpts.verbose)
        llvm::errs() << "  Warm-start baseline: throughput="
                     << lastSwThroughput << "\n";
    } else {
      if (coOpts.verbose)
        llvm::errs() << "  Warm-start SW evaluation failed: "
                     << warmSwResult.diagnostics
                     << "; proceeding with standard loop\n";
      diag << "warm-start SW evaluation failed; standard loop\n";
    }
  } else {
    // CSV header for diagnostics.
    diag << "round,swThroughput,hwArea,swTransforms,hwCoreTypes,"
         << "improved,reason\n";
  }

  // Initialize last HW core types from current architecture.
  lastHwCoreTypes =
      static_cast<unsigned>(currentArch.coreTypes.size());

  for (unsigned round = 1; round <= effectiveMaxRounds; ++round) {
    if (coOpts.verbose)
      llvm::errs() << "=== Round " << round << " / "
                   << effectiveMaxRounds << " ===\n";

    CoOptResult::RoundRecord record;
    record.round = round;

    double swThroughput = lastSwThroughput;
    unsigned swTransforms = 0;
    TDGOptimizeResult swResult;

    // ---- SW Step: fix hardware, optimize software ----
    if (effectiveOpts.enableSW) {
      if (coOpts.verbose)
        llvm::errs() << "  SW step: running TDGOptimizer...\n";

      swResult =
          runSWStep(kernels, contracts, currentArch, effectiveOpts, ctx);

      if (swResult.success) {
        swThroughput = swResult.bestThroughput;
        kernels = swResult.optimizedKernels;
        contracts = swResult.optimizedContracts;
        swTransforms =
            static_cast<unsigned>(swResult.transformHistory.size());

        if (coOpts.verbose)
          llvm::errs() << "  SW step succeeded: throughput = "
                       << swThroughput
                       << ", iterations = " << swResult.iterations
                       << ", transforms = " << swTransforms << "\n";
      } else {
        if (coOpts.verbose)
          llvm::errs() << "  SW step failed: "
                       << swResult.diagnostics << "\n";
      }

      // Propagate achieved rates back into contracts.
      updateContractsFromSW(contracts, swResult);
    } else {
      if (coOpts.verbose)
        llvm::errs() << "  SW step: SKIPPED (enableSW=false), "
                     << "carrying forward throughput="
                     << lastSwThroughput << "\n";
    }

    lastSwThroughput = swThroughput;
    record.swThroughput = swThroughput;
    record.swTransforms = swTransforms;

    // ---- HW Step: fix software, optimize hardware ----
    double hwArea = lastHwArea;
    unsigned hwCoreTypes = lastHwCoreTypes;

    if (effectiveOpts.enableHW) {
      if (coOpts.verbose)
        llvm::errs() << "  HW step: running OUTER-HW + INNER-HW...\n";

      HWStepResult hwStep =
          runHWStep(kernels, contracts, profiles, effectiveOpts, ctx);

      if (hwStep.success) {
        hwArea = hwStep.totalArea;
        currentArch = hwStep.newArch;
        hwCoreTypes =
            hwStep.outerResult.topology.coreLibrary.numTypes();

        if (coOpts.verbose)
          llvm::errs() << "  HW step succeeded: area = " << hwArea
                       << ", core types = " << hwCoreTypes << "\n";
      } else {
        if (coOpts.verbose)
          llvm::errs() << "  HW step failed\n";
        hwArea = std::numeric_limits<double>::infinity();
      }
    } else {
      if (coOpts.verbose)
        llvm::errs() << "  HW step: SKIPPED (enableHW=false), "
                     << "carrying forward area=" << lastHwArea << "\n";
    }

    lastHwArea = hwArea;
    lastHwCoreTypes = hwCoreTypes;
    record.hwArea = hwArea;
    record.hwCoreTypes = hwCoreTypes;

    // ---- Pareto frontier update ----
    if (swThroughput > 0.0 &&
        hwArea < std::numeric_limits<double>::infinity()) {
      ParetoPoint candidate;
      candidate.throughput = swThroughput;
      candidate.area = hwArea;
      candidate.round = round;
      addParetoPoint(result.paretoFrontier, candidate);
    }

    // ---- Convergence check via ConvergenceMonitor ----
    std::string reason;
    bool improved = monitor.checkImproved(swThroughput, hwArea, reason);

    record.improved = improved;
    record.reason = reason;
    result.history.push_back(record);

    // Structured diagnostic line (CSV-ready).
    diag << round << "," << swThroughput << "," << hwArea << ","
         << swTransforms << "," << hwCoreTypes << ","
         << (improved ? "true" : "false") << "," << reason << "\n";

    if (improved) {
      if (swThroughput > 0.0 &&
          (swThroughput >= monitor.bestThroughput ||
           effectiveOpts.enableSW)) {
        bestKernels = kernels;
        bestContracts = contracts;
        if (swResult.success)
          bestBenders = swResult.compilationResult;
      }
      if (hwArea < std::numeric_limits<double>::infinity()) {
        bestArchSaved = currentArch;
      }

      if (coOpts.verbose)
        llvm::errs() << "  Improved: " << reason << "\n";
    } else {
      if (coOpts.verbose)
        llvm::errs() << "  Converged: " << reason << "\n";
      result.rounds = round;
      break;
    }

    result.rounds = round;
  }

  // If the loop ran to completion without breaking (all rounds improved),
  // rounds was set inside the loop at the last iteration.
  if (result.rounds == 0 && !result.history.empty())
    result.rounds = result.history.back().round;

  // ---- Populate final result ----
  result.success = monitor.bestThroughput > 0.0;
  result.bestKernels = std::move(bestKernels);
  result.bestContracts = std::move(bestContracts);
  result.bestArch = std::move(bestArchSaved);
  result.bestThroughput = monitor.bestThroughput;
  result.bestArea = monitor.bestArea;
  result.bestBendersResult = std::move(bestBenders);
  result.diagnostics = diag.str();

  if (coOpts.verbose) {
    llvm::errs() << "Co-optimization complete: "
                 << result.rounds << " round(s), "
                 << "throughput=" << result.bestThroughput
                 << ", area=" << result.bestArea
                 << ", Pareto points=" << result.paretoFrontier.size()
                 << "\n";
  }

  return result;
}

} // namespace tapestry
