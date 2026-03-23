#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/SystemCompiler/KernelProfiler.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>
#include <sstream>

namespace loom {

/// Create a zero-cost NoC schedule for perfect-NoC upper-bound analysis.
static NoCSchedule makePerfectNoCSchedule() {
  NoCSchedule sched;
  sched.maxLinkUtilization = 0.0;
  sched.avgLinkUtilization = 0.0;
  sched.totalTransferCycles = 0;
  sched.hasContention = false;
  return sched;
}

/// Extract contracts from the TDG module.
/// Walks the MLIR module for contract annotations and returns them as a vector.
/// Stub: returns empty contracts since TDG dialect extraction is not yet wired.
static std::vector<ContractSpec>
extractContracts(mlir::ModuleOp /*tdgModule*/) {
  // TODO: Walk tdgModule for tdg.contract ops and convert to ContractSpec.
  return {};
}

/// Run L2 compilations for all cores, optionally in parallel.
static std::vector<L2Result>
runL2Compilations(const std::vector<L2Assignment> &l2Assignments,
                  const MapperOptions &baseMapperOpts, mlir::MLIRContext *ctx,
                  unsigned numParallel) {
  std::vector<L2Result> results;
  results.reserve(l2Assignments.size());

  if (numParallel <= 1 || l2Assignments.size() <= 1) {
    // Sequential execution.
    for (const auto &assignment : l2Assignments) {
      L2CoreCompiler compiler;
      results.push_back(compiler.compile(assignment, baseMapperOpts, ctx));
    }
    return results;
  }

  // Parallel execution using std::async.
  // Note: Each L2 compilation is independent (separate core, separate DFG).
  // MLIR contexts are not thread-safe, so parallel mode requires per-thread
  // contexts in production. For now, fall back to sequential.
  for (const auto &assignment : l2Assignments) {
    L2CoreCompiler compiler;
    results.push_back(compiler.compile(assignment, baseMapperOpts, ctx));
  }
  return results;
}

/// Collect infeasibility cuts from L2 results that had mapping failures.
static std::vector<InfeasibilityCut>
collectCuts(const std::vector<L2Result> &l2Results) {
  std::vector<InfeasibilityCut> cuts;
  for (const auto &result : l2Results) {
    if (result.allKernelsMapped)
      continue;
    for (const auto &kr : result.kernelResults) {
      if (!kr.success && kr.cut.has_value())
        cuts.push_back(kr.cut.value());
    }
  }
  return cuts;
}

/// Collect cost summaries from all L2 results.
static std::vector<CoreCostSummary>
collectCostSummaries(const std::vector<L2Result> &l2Results) {
  std::vector<CoreCostSummary> summaries;
  summaries.reserve(l2Results.size());
  for (const auto &result : l2Results)
    summaries.push_back(result.costSummary);
  return summaries;
}

/// Check if all L2 subproblems succeeded (all kernels mapped on all cores).
static bool allL2Succeeded(const std::vector<L2Result> &l2Results) {
  return std::all_of(l2Results.begin(), l2Results.end(),
                     [](const L2Result &r) { return r.allKernelsMapped; });
}

/// Compute system metrics from the final converged state.
static SystemMetrics
computeSystemMetrics(const std::vector<CoreCostSummary> &costSummaries,
                     const NoCSchedule &nocSchedule, unsigned numIterations,
                     double compilationTimeSec) {
  SystemMetrics m;
  m.numBendersIterations = numIterations;
  m.compilationTimeSec = compilationTimeSec;
  m.totalNoCTransferCycles = nocSchedule.totalTransferCycles;

  if (costSummaries.empty())
    return m;

  // Compute throughput from achieved IIs: system throughput is bounded
  // by the slowest core (bottleneck).
  double maxII = 0.0;
  double totalUtil = 0.0;
  double maxUtil = 0.0;
  unsigned successCount = 0;

  for (const auto &cs : costSummaries) {
    if (!cs.success)
      continue;
    ++successCount;
    totalUtil += cs.totalPEUtilization;
    maxUtil = std::max(maxUtil, cs.totalPEUtilization);
    for (const auto &km : cs.kernelMetrics) {
      maxII = std::max(maxII, static_cast<double>(km.achievedII));
    }
  }

  if (successCount > 0) {
    m.avgCoreUtilization = totalUtil / successCount;
    m.maxCoreUtilization = maxUtil;
  }

  // Throughput = 1/max_II (elements per cycle at the bottleneck).
  if (maxII > 0.0)
    m.throughput = 1.0 / maxII;

  // Critical path latency: sum of all achieved IIs along the longest path.
  // Simplified: use the maximum single-kernel II as an approximation.
  m.criticalPathLatency = maxII;

  return m;
}

TapestryCompilationResult
BendersDriver::compile(const TapestryCompilationInput &input,
                       const Options &opts) {
  auto startTime = std::chrono::steady_clock::now();

  TapestryCompilationResult failResult;
  failResult.success = false;

  // --- Preprocessing ---
  std::vector<ContractSpec> contracts = extractContracts(input.tdgModule);

  KernelProfiler profiler;
  std::vector<KernelProfile> kernelProfiles =
      profiler.profileAll(input.tdgModule, input.ctx);

  std::map<std::string, mlir::ModuleOp> kernelDFGs =
      lowerKernelsToDFG(input.tdgModule, input.ctx);

  if (kernelProfiles.empty()) {
    failResult.diagnostics =
        "No kernels found in TDG module; nothing to compile.";
    return failResult;
  }

  if (opts.verbose)
    llvm::errs() << "BendersDriver: " << kernelProfiles.size()
                 << " kernels, " << input.architecture.totalCoreInstances()
                 << " core instances\n";

  // --- Benders loop state ---
  std::vector<InfeasibilityCut> allCuts;
  std::vector<IterationRecord> iterationHistory;
  double bestObjective = std::numeric_limits<double>::infinity();

  // Best result found so far (valid only when at least one iteration
  // converges).
  TapestryCompilationResult bestResult;
  bestResult.success = false;

  L1CoreAssigner l1Assigner;
  L1AssignerOptions l1Opts;
  l1Opts.verbose = opts.verbose;

  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  nocOpts.verbose = opts.verbose;

  BufferAllocator bufferAllocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.verbose = opts.verbose;

  DMAScheduler dmaScheduler;
  DMASchedulerOptions dmaOpts;
  dmaOpts.verbose = opts.verbose;

  unsigned effectiveParallel =
      opts.numParallelL2 > 0 ? opts.numParallelL2 : 1;

  for (unsigned iter = 1; iter <= opts.maxIterations; ++iter) {
    if (opts.verbose)
      llvm::errs() << "BendersDriver: --- iteration " << iter << " ---\n";

    // ================================================================
    // L1 MASTER PROBLEM
    // ================================================================

    // Solve core assignment with accumulated cuts.
    AssignmentResult assignment = l1Assigner.solve(
        kernelProfiles, contracts, input.architecture, allCuts, l1Opts);

    if (!assignment.feasible) {
      std::ostringstream diag;
      diag << "L1 infeasible at iteration " << iter
           << ": all assignments eliminated by " << allCuts.size() << " cuts.";
      failResult.diagnostics = diag.str();
      failResult.iterationHistory = iterationHistory;
      return failResult;
    }

    // Schedule NoC (or use perfect zero-cost schedule).
    NoCSchedule nocSchedule;
    if (opts.perfectNoC) {
      nocSchedule = makePerfectNoCSchedule();
    } else {
      nocSchedule = nocScheduler.schedule(assignment, contracts,
                                          input.architecture, nocOpts);
    }

    // Allocate buffers.
    BufferAllocationPlan bufferPlan = bufferAllocator.allocate(
        assignment, contracts, nocSchedule, input.architecture, bufOpts);

    // ================================================================
    // L2 SUBPROBLEMS
    // ================================================================

    std::vector<L2Assignment> l2Assignments =
        buildL2Assignments(assignment, kernelDFGs, contracts,
                           input.architecture);

    std::vector<L2Result> l2Results =
        runL2Compilations(l2Assignments, input.baseMapperOpts, input.ctx,
                          effectiveParallel);

    // Collect feedback.
    std::vector<InfeasibilityCut> newCuts = collectCuts(l2Results);
    std::vector<CoreCostSummary> costSummaries =
        collectCostSummaries(l2Results);

    // ================================================================
    // CONVERGENCE CHECK
    // ================================================================

    IterationRecord record;
    record.iteration = iter;
    record.assignment = assignment;
    record.cuts = newCuts;
    record.costSummaries = costSummaries;
    record.converged = false;

    if (newCuts.empty() && allL2Succeeded(l2Results)) {
      // All kernels mapped successfully on all cores.
      record.converged = true;

      double currentObjective =
          computeObjective(assignment, nocSchedule, costSummaries);

      if (currentObjective < bestObjective - opts.costTighteningThreshold) {
        bestObjective = currentObjective;

        // Schedule DMA for the converged solution.
        DMASchedule dmaSchedule = dmaScheduler.schedule(
            bufferPlan, nocSchedule, contracts, assignment,
            input.architecture, dmaOpts);

        bestResult = assembleResult(l2Results, l2Assignments, assignment,
                                    nocSchedule, bufferPlan, dmaSchedule,
                                    costSummaries);
        bestResult.success = true;
        bestResult.iterationHistory = iterationHistory;

        if (opts.verbose)
          llvm::errs() << "BendersDriver: new best objective = "
                       << currentObjective << "\n";

        // If cost tightening is disabled, converge immediately.
        if (!opts.enableCostTightening) {
          iterationHistory.push_back(record);
          break;
        }
        // Otherwise continue to see if further tightening is possible.
      } else {
        // No significant improvement; converged.
        if (opts.verbose)
          llvm::errs() << "BendersDriver: converged (no significant "
                       << "improvement)\n";
        iterationHistory.push_back(record);
        break;
      }
    } else {
      // Some kernels failed to map.
      if (opts.enableInfeasibilityCuts && !newCuts.empty()) {
        allCuts.insert(allCuts.end(), newCuts.begin(), newCuts.end());
        if (opts.verbose)
          llvm::errs() << "BendersDriver: added " << newCuts.size()
                       << " infeasibility cuts (total: " << allCuts.size()
                       << ")\n";
      }

      // Update contract cost estimates from successful L2 results.
      if (opts.enableCostTightening)
        updateContractCosts(contracts, costSummaries);
    }

    iterationHistory.push_back(record);
  }

  // --- Finalize ---
  auto endTime = std::chrono::steady_clock::now();
  double totalSec =
      std::chrono::duration<double>(endTime - startTime).count();

  if (!bestResult.success) {
    std::ostringstream diag;
    diag << "Failed to converge in " << opts.maxIterations
         << " iterations. " << allCuts.size()
         << " infeasibility cuts accumulated.";
    failResult.diagnostics = diag.str();
    failResult.iterationHistory = iterationHistory;
    return failResult;
  }

  bestResult.metrics = computeSystemMetrics(
      bestResult.iterationHistory.empty()
          ? std::vector<CoreCostSummary>()
          : bestResult.iterationHistory.back().costSummaries,
      bestResult.finalNoCSchedule,
      static_cast<unsigned>(iterationHistory.size()), totalSec);

  // Ensure iteration history is complete.
  bestResult.iterationHistory = iterationHistory;

  if (opts.verbose)
    llvm::errs() << "BendersDriver: compilation complete in "
                 << totalSec << "s, " << iterationHistory.size()
                 << " iterations\n";

  return bestResult;
}

} // namespace loom
