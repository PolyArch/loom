#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/SystemCompiler/BendersHelpers.h"
#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/DMAScheduler.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/SystemCompiler/NoCScheduler.h"
#include "loom/SystemCompiler/SystemTypes.h"
#include "loom/SystemCompiler/TypeAdapters.h"
#include "loom/Mapper/MapperOptions.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace loom {
namespace syscomp {

BendersDriver::BendersDriver(const BendersDriverOptions &options)
    : options_(options) {}

void BendersDriver::addTask(const BendersTask &task) {
  tasks_.push_back(task);
}

void BendersDriver::addEdge(const BendersEdge &edge) {
  edges_.push_back(edge);
}

BendersResult BendersDriver::solve() {
  BendersResult result;

  if (tasks_.empty()) {
    result.feasible = true;
    result.statusMessage = "no tasks to partition";
    return result;
  }

  if (options_.numCores == 0) {
    result.feasible = false;
    result.statusMessage = "numCores is zero";
    return result;
  }

  unsigned numTasks = static_cast<unsigned>(tasks_.size());
  unsigned numCores = options_.numCores;
  result.taskAssignment.resize(numTasks, 0);

  // Greedy initial assignment: round-robin by estimated cycles, assigning
  // the longest-remaining task to the least-loaded core.
  std::vector<unsigned> taskOrder(numTasks);
  std::iota(taskOrder.begin(), taskOrder.end(), 0);
  std::sort(taskOrder.begin(), taskOrder.end(),
            [this](unsigned a, unsigned b) {
              return tasks_[a].estimatedCycles > tasks_[b].estimatedCycles;
            });

  // Per-core load (cycles) and SPM usage.
  std::vector<uint64_t> coreLoad(numCores, 0);
  std::vector<uint64_t> coreSpm(numCores, 0);

  for (unsigned ti : taskOrder) {
    // Find the core with least load that has enough SPM budget.
    unsigned bestCore = 0;
    uint64_t bestLoad = UINT64_MAX;
    bool found = false;

    for (unsigned ci = 0; ci < numCores; ++ci) {
      if (coreSpm[ci] + tasks_[ti].spmBytes > options_.spmBudgetBytes)
        continue;
      if (coreLoad[ci] < bestLoad) {
        bestLoad = coreLoad[ci];
        bestCore = ci;
        found = true;
      }
    }

    if (!found) {
      result.feasible = false;
      std::ostringstream oss;
      oss << "task '" << tasks_[ti].name
          << "' cannot fit in any core's SPM budget";
      result.statusMessage = oss.str();
      return result;
    }

    result.taskAssignment[ti] = bestCore;
    coreLoad[bestCore] += tasks_[ti].estimatedCycles;
    coreSpm[bestCore] += tasks_[ti].spmBytes;
  }

  // Compute objective: makespan (max core load) + cross-core communication
  // penalty.
  uint64_t makespan = *std::max_element(coreLoad.begin(), coreLoad.end());

  double commPenalty = 0.0;
  for (const auto &e : edges_) {
    if (result.taskAssignment[e.srcTaskIndex] !=
        result.taskAssignment[e.dstTaskIndex]) {
      // Cross-core edge: add transfer time as penalty.
      if (options_.nocBandwidthBytesPerCycle > 0.0) {
        commPenalty += static_cast<double>(e.dataBytes) /
                       options_.nocBandwidthBytesPerCycle;
      }
    }
  }

  result.feasible = true;
  result.iterations = 1;
  result.objectiveValue =
      static_cast<double>(makespan) + commPenalty;
  result.statusMessage = "greedy partitioning converged";
  return result;
}

} // namespace syscomp

// -----------------------------------------------------------------------
// tapestry::BendersDriver -- heterogeneous multi-core decomposition
// -----------------------------------------------------------------------
namespace tapestry {

BendersDriver::BendersDriver(const SystemArchitecture &arch,
                             std::vector<KernelDesc> kernels,
                             std::vector<ContractSpec> contracts,
                             mlir::MLIRContext &ctx)
    : arch_(arch), kernels_(std::move(kernels)),
      contracts_(std::move(contracts)), ctx_(ctx) {}

BendersResult BendersDriver::compile(const BendersConfig &config) {
  BendersResult result;

  if (kernels_.empty()) {
    result.success = true;
    result.diagnostics = "no kernels to partition";
    return result;
  }

  if (arch_.coreTypes.empty()) {
    result.success = false;
    result.diagnostics = "no core types in architecture";
    return result;
  }

  // Convert tapestry types to loom root types for L1/L2 components.
  loom::SystemArchitecture l1Arch = toL1Architecture(arch_, &ctx_);
  std::vector<loom::KernelProfile> kernelProfiles =
      toKernelProfiles(kernels_, &ctx_);
  std::vector<loom::ContractSpec> l1Contracts = toL1Contracts(contracts_);
  std::map<std::string, mlir::ModuleOp> kernelDFGs =
      buildKernelDFGMap(kernels_);

  // Mapper options from config.
  MapperOptions mapperOpts;
  mapperOpts.budgetSeconds = config.mapperBudgetSeconds;
  mapperOpts.seed = static_cast<int>(config.mapperSeed);

  // Benders iteration state.
  std::vector<loom::InfeasibilityCut> accumulatedCuts;
  std::optional<loom::TapestryCompilationResult> bestResult;
  double bestObjective = std::numeric_limits<double>::max();

  L1CoreAssigner l1Assigner;
  L1AssignerOptions l1Opts;
  l1Opts.verbose = config.verbose;

  if (config.verbose) {
    llvm::outs() << "BendersDriver: starting bilevel compilation\n"
                 << "  kernels=" << kernels_.size()
                 << "  coreTypes=" << arch_.coreTypes.size()
                 << "  maxIter=" << config.maxIterations << "\n";
  }

  unsigned lastIteration = 0;
  for (unsigned iter = 1; iter <= config.maxIterations; ++iter) {
    lastIteration = iter;

    if (config.verbose) {
      llvm::outs() << "\n--- Benders iteration " << iter << " ---\n"
                   << "  accumulated cuts: " << accumulatedCuts.size() << "\n";
    }

    // --- L1 MASTER PROBLEM ---
    loom::AssignmentResult assignment =
        l1Assigner.solve(kernelProfiles, l1Contracts, l1Arch,
                         accumulatedCuts, l1Opts);

    if (!assignment.feasible) {
      if (config.verbose)
        llvm::outs() << "  L1 solver: INFEASIBLE\n";
      result.success = false;
      result.iterations = iter;
      result.diagnostics =
          "L1 infeasible after " +
          std::to_string(accumulatedCuts.size()) + " cuts";
      return result;
    }

    if (config.verbose) {
      llvm::outs() << "  L1 solver: feasible, objective="
                   << assignment.objectiveValue << "\n";
      for (const auto &ca : assignment.coreAssignments) {
        if (ca.assignedKernels.empty())
          continue;
        llvm::outs() << "    core " << ca.coreInstanceIdx
                     << " (" << ca.coreTypeName << "): ";
        for (const auto &kn : ca.assignedKernels)
          llvm::outs() << kn << " ";
        llvm::outs() << "\n";
      }
    }

    // --- NoC Scheduling ---
    NoCScheduler nocScheduler;
    NoCSchedulerOptions nocOpts;
    nocOpts.verbose = config.verbose;
    loom::NoCSchedule nocSchedule =
        nocScheduler.schedule(assignment, l1Contracts, l1Arch, nocOpts);

    // --- Buffer Allocation ---
    BufferAllocator bufferAllocator;
    BufferAllocatorOptions bufOpts;
    bufOpts.verbose = config.verbose;
    loom::BufferAllocationPlan bufferPlan =
        bufferAllocator.allocate(assignment, l1Contracts, nocSchedule,
                                l1Arch, bufOpts);

    // --- L2 SUBPROBLEMS ---
    std::vector<loom::L2Assignment> l2Assignments =
        buildL2Assignments(assignment, kernelDFGs, l1Contracts, l1Arch);

    // Populate ADG modules from tapestry architecture.
    populateL2ADGs(l2Assignments, arch_);

    std::vector<loom::InfeasibilityCut> newCuts;
    bool allMapped = true;
    std::vector<loom::L2Result> l2Results;
    std::vector<loom::CoreCostSummary> costSummaries;

    L2CoreCompiler l2Compiler;

    for (const auto &l2Assign : l2Assignments) {
      if (config.verbose) {
        llvm::outs() << "  L2 compiling core '"
                     << l2Assign.coreInstanceName << "' ("
                     << l2Assign.coreType << "): "
                     << l2Assign.kernels.size() << " kernels\n";
      }

      loom::L2Result l2Result =
          l2Compiler.compile(l2Assign, mapperOpts, &ctx_);
      l2Results.push_back(l2Result);
      costSummaries.push_back(l2Result.costSummary);

      if (!l2Result.allKernelsMapped) {
        allMapped = false;
        // Collect infeasibility cuts from failed kernels.
        for (const auto &kr : l2Result.kernelResults) {
          if (!kr.success && kr.cut.has_value()) {
            newCuts.push_back(kr.cut.value());
            if (config.verbose) {
              llvm::outs() << "    kernel '" << kr.kernelName
                           << "' FAILED, cut: "
                           << cutReasonToString(kr.cut->reason) << "\n";
            }
          }
        }
      } else {
        if (config.verbose)
          llvm::outs() << "    all kernels mapped successfully\n";
      }
    }

    // --- CONVERGENCE CHECK ---
    if (allMapped) {
      double objective =
          computeObjective(assignment, nocSchedule, costSummaries);

      if (config.verbose) {
        llvm::outs() << "  all cores mapped, objective=" << objective
                     << " (best=" << bestObjective << ")\n";
      }

      // --- DMA Scheduling ---
      DMAScheduler dmaScheduler;
      DMASchedulerOptions dmaOpts;
      dmaOpts.verbose = config.verbose;
      loom::DMASchedule dmaSchedule = dmaScheduler.schedule(
          bufferPlan, nocSchedule, l1Contracts, assignment, l1Arch, dmaOpts);

      if (objective < bestObjective) {
        bestObjective = objective;
        bestResult = assembleResult(l2Results, l2Assignments, assignment,
                                    nocSchedule, bufferPlan, dmaSchedule,
                                    costSummaries);
      }

      if (newCuts.empty()) {
        if (config.verbose)
          llvm::outs() << "  converged: all mapped, no new cuts\n";
        break;
      }
    }

    // Feed cuts back to L1 for next iteration.
    if (!newCuts.empty()) {
      accumulatedCuts.insert(accumulatedCuts.end(),
                             newCuts.begin(), newCuts.end());
      updateContractCosts(l1Contracts, costSummaries);

      if (config.verbose) {
        llvm::outs() << "  added " << newCuts.size()
                     << " new cuts, total=" << accumulatedCuts.size()
                     << "\n";
      }
    }
  }

  // Finalize result.
  if (bestResult.has_value()) {
    result = toBendersResult(bestResult.value(), kernels_, arch_,
                             lastIteration);
    result.success = true;
    if (config.verbose)
      llvm::outs() << "\nBendersDriver: converged in "
                   << lastIteration << " iterations\n";
  } else {
    result.success = false;
    result.iterations = lastIteration;
    result.diagnostics =
        "no feasible mapping found in " +
        std::to_string(config.maxIterations) + " iterations";
    if (config.verbose)
      llvm::outs() << "\nBendersDriver: FAILED - " << result.diagnostics
                   << "\n";
  }

  return result;
}

} // namespace tapestry
} // namespace loom
