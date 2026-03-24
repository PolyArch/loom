#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/SystemCompiler/ExecutionModel.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
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

  // Validate execution mode.
  if (config.executionModel.mode != ExecutionMode::BATCH_SEQUENTIAL) {
    result.success = false;
    result.diagnostics =
        std::string(executionModeToString(config.executionModel.mode)) +
        " execution mode is not supported in current version; "
        "only BATCH_SEQUENTIAL is implemented";
    return result;
  }

  // Greedy round-robin assignment: assign each kernel to the core type
  // whose total instances can accommodate it.
  unsigned numTypes = static_cast<unsigned>(arch_.coreTypes.size());

  for (unsigned ki = 0; ki < kernels_.size(); ++ki) {
    unsigned bestType = ki % numTypes;

    L2Assignment assign;
    assign.kernelName = kernels_[ki].name;
    assign.coreTypeIndex = static_cast<int>(bestType);
    assign.coreInstanceIndex = 0;

    if (bestType < numTypes)
      assign.coreADG = arch_.coreTypes[bestType].adgModule;

    assign.mappingSuccess = true;
    assign.mappingCost = 1.0;
    result.assignments.push_back(std::move(assign));
  }

  // Compute temporal schedule for the assignment.
  // Build a synthetic AssignmentResult and CostSummary from the greedy
  // assignment to feed into the temporal scheduler.
  loom::AssignmentResult syntheticAssignment;
  syntheticAssignment.feasible = true;

  // Build per-core assignment structures.
  // Map from (coreTypeIndex, coreInstanceIndex) to a flat core index.
  unsigned totalInstances = 0;
  for (const auto &ct : arch_.coreTypes)
    totalInstances += ct.numInstances;
  if (totalInstances == 0)
    totalInstances = numTypes;

  syntheticAssignment.coreAssignments.resize(totalInstances);
  for (unsigned ci = 0; ci < totalInstances; ++ci) {
    syntheticAssignment.coreAssignments[ci].coreInstanceIdx = ci;
    // Determine core type name.
    unsigned offset = 0;
    for (const auto &ct : arch_.coreTypes) {
      unsigned count = ct.numInstances > 0 ? ct.numInstances : 1;
      if (ci < offset + count) {
        syntheticAssignment.coreAssignments[ci].coreTypeName = ct.name;
        break;
      }
      offset += count;
    }
  }

  // Assign kernels to cores.
  for (const auto &assign : result.assignments) {
    unsigned typeIdx = static_cast<unsigned>(assign.coreTypeIndex);
    unsigned instIdx = static_cast<unsigned>(assign.coreInstanceIndex);
    // Compute flat core index.
    unsigned flatIdx = 0;
    for (unsigned ti = 0; ti < typeIdx && ti < arch_.coreTypes.size(); ++ti) {
      unsigned count = arch_.coreTypes[ti].numInstances;
      flatIdx += (count > 0 ? count : 1);
    }
    flatIdx += instIdx;

    if (flatIdx < syntheticAssignment.coreAssignments.size()) {
      syntheticAssignment.kernelToCore[assign.kernelName] = flatIdx;
      syntheticAssignment.coreAssignments[flatIdx].assignedKernels.push_back(
          assign.kernelName);
    }
  }

  // Build synthetic cost summaries with default achieved II.
  std::vector<loom::CoreCostSummary> syntheticCosts;
  for (const auto &ca : syntheticAssignment.coreAssignments) {
    if (ca.assignedKernels.empty())
      continue;
    loom::CoreCostSummary cs;
    cs.coreInstanceName = ca.coreTypeName + "_" +
                          std::to_string(ca.coreInstanceIdx);
    cs.coreType = ca.coreTypeName;
    cs.success = true;
    for (const auto &kn : ca.assignedKernels) {
      loom::KernelMetrics km;
      km.kernelName = kn;
      km.achievedII = 1; // Default: 1 cycle per iteration
      cs.kernelMetrics.push_back(km);
    }
    syntheticCosts.push_back(std::move(cs));
  }

  // Build synthetic contracts in loom::ContractSpec format.
  std::vector<loom::ContractSpec> l1Contracts;
  for (const auto &tc : contracts_) {
    loom::ContractSpec lc;
    lc.producerKernel = tc.producerKernel;
    lc.consumerKernel = tc.consumerKernel;
    lc.dataTypeName = tc.dataType;
    if (tc.elementCount > 0)
      lc.productionRate = static_cast<int64_t>(tc.elementCount);
    l1Contracts.push_back(std::move(lc));
  }

  // Run the temporal scheduler.
  loom::TemporalSchedule schedule;
  std::string schedErr = computeTemporalSchedule(
      syntheticAssignment, syntheticCosts, l1Contracts,
      config.executionModel, schedule);

  if (schedErr.empty()) {
    result.temporalSchedule = schedule;

    if (config.verbose) {
      llvm::outs() << "Temporal schedule (BATCH_SEQUENTIAL):\n";
      for (const auto &cs : schedule.coreSchedules) {
        llvm::outs() << "  Core " << cs.coreInstanceName << ": "
                     << cs.kernelOrder.size() << " kernels, "
                     << cs.totalCycles << " cycles"
                     << " (reconfig=" << cs.reconfigCount << ")\n";
        for (const auto &kt : cs.kernelTimings) {
          llvm::outs() << "    " << kt.kernelName
                       << ": tripCount=" << kt.tripCount
                       << " II=" << kt.achievedII
                       << " exec=" << kt.executionCycles << "\n";
        }
      }
      llvm::outs() << "  System latency: " << schedule.systemLatencyCycles
                   << " (max_core=" << schedule.maxCoreCycles
                   << " + noc=" << schedule.nocOverheadCycles << ")\n";
    }
  } else if (config.verbose) {
    llvm::outs() << "Temporal scheduling skipped: " << schedErr << "\n";
  }

  result.success = true;
  result.iterations = 1;
  result.totalCost = static_cast<double>(kernels_.size());
  return result;
}

} // namespace tapestry
} // namespace loom
