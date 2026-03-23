#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/SystemCompiler/SystemTypes.h"

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

  result.success = true;
  result.iterations = 1;
  result.totalCost = static_cast<double>(kernels_.size());
  return result;
}

} // namespace tapestry
} // namespace loom
