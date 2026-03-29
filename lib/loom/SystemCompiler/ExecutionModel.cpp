//===-- ExecutionModel.cpp - Temporal execution model implementation -------===//
//
// Implements the temporal scheduling for the Tapestry multi-core compiler.
// Currently supports BATCH_SEQUENTIAL mode where kernels execute sequentially
// on each core with reconfiguration gaps between them.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/ExecutionModel.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <sstream>

namespace loom {

//===----------------------------------------------------------------------===//
// ExecutionMode string conversion
//===----------------------------------------------------------------------===//

const char *executionModeToString(ExecutionMode mode) {
  switch (mode) {
  case ExecutionMode::BATCH_SEQUENTIAL:
    return "BATCH_SEQUENTIAL";
  case ExecutionMode::PIPELINE_PARALLEL:
    return "PIPELINE_PARALLEL";
  case ExecutionMode::SPATIAL_PARALLEL:
    return "SPATIAL_PARALLEL";
  case ExecutionMode::SPATIAL_SHARING:
    return "SPATIAL_SHARING";
  }
  return "BATCH_SEQUENTIAL";
}

ExecutionMode executionModeFromString(const std::string &s) {
  if (s == "PIPELINE_PARALLEL")
    return ExecutionMode::PIPELINE_PARALLEL;
  if (s == "SPATIAL_PARALLEL")
    return ExecutionMode::SPATIAL_PARALLEL;
  if (s == "SPATIAL_SHARING")
    return ExecutionMode::SPATIAL_SHARING;
  return ExecutionMode::BATCH_SEQUENTIAL;
}

//===----------------------------------------------------------------------===//
// Topological Sort
//===----------------------------------------------------------------------===//

std::vector<std::string>
topologicalSortKernels(const std::vector<std::string> &kernels,
                       const std::vector<ContractSpec> &contracts) {
  if (kernels.size() <= 1)
    return kernels;

  // Build a set for quick membership check.
  std::set<std::string> kernelSet(kernels.begin(), kernels.end());

  // Build adjacency list and in-degree map for kernels on this core.
  std::map<std::string, std::vector<std::string>> adj;
  std::map<std::string, unsigned> inDegree;

  for (const auto &k : kernels) {
    adj[k] = {};
    inDegree[k] = 0;
  }

  for (const auto &contract : contracts) {
    bool prodLocal = kernelSet.count(contract.producerKernel) > 0;
    bool consLocal = kernelSet.count(contract.consumerKernel) > 0;
    if (prodLocal && consLocal) {
      adj[contract.producerKernel].push_back(contract.consumerKernel);
      inDegree[contract.consumerKernel]++;
    }
  }

  // Kahn's algorithm for topological sort.
  std::queue<std::string> ready;
  for (const auto &k : kernels) {
    if (inDegree[k] == 0)
      ready.push(k);
  }

  std::vector<std::string> sorted;
  sorted.reserve(kernels.size());

  while (!ready.empty()) {
    std::string current = ready.front();
    ready.pop();
    sorted.push_back(current);

    for (const auto &neighbor : adj[current]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] == 0)
        ready.push(neighbor);
    }
  }

  // If topological sort didn't include all kernels (cycle detected),
  // fall back to original order.
  if (sorted.size() != kernels.size())
    return kernels;

  return sorted;
}

//===----------------------------------------------------------------------===//
// Trip Count Lookup
//===----------------------------------------------------------------------===//

unsigned lookupTripCount(const std::string &kernelName,
                         const std::vector<CoreCostSummary> &costSummaries) {
  // In the current implementation, trip counts are not yet annotated in
  // KernelMetrics. Use a default value of 1000 iterations.
  // Future versions will extract this from RateAnalyzer or TDG annotations.
  (void)kernelName;
  (void)costSummaries;
  return 1000;
}

//===----------------------------------------------------------------------===//
// NoC Overhead Computation
//===----------------------------------------------------------------------===//

uint64_t computeNoCOverhead(const AssignmentResult &assignment,
                            const std::vector<ContractSpec> &contracts) {
  uint64_t totalOverhead = 0;

  for (const auto &contract : contracts) {
    auto prodIt = assignment.kernelToCore.find(contract.producerKernel);
    auto consIt = assignment.kernelToCore.find(contract.consumerKernel);

    if (prodIt == assignment.kernelToCore.end() ||
        consIt == assignment.kernelToCore.end())
      continue;

    // Only inter-core edges contribute NoC overhead.
    if (prodIt->second == consIt->second)
      continue;

    // Estimate transfer cycles based on data volume and element size.
    unsigned elemSize = estimateElementSize(contract.dataTypeName);
    int64_t volume = 0;
    if (contract.productionRate.has_value())
      volume = contract.productionRate.value();
    else
      volume = 256; // Default element count

    uint64_t dataBytes = static_cast<uint64_t>(volume) * elemSize;
    // Assume a baseline transfer rate of 8 bytes/cycle.
    constexpr uint64_t kDefaultBandwidthBytesPerCycle = 8;
    uint64_t transferCycles = dataBytes / kDefaultBandwidthBytesPerCycle;
    if (transferCycles == 0)
      transferCycles = 1;

    totalOverhead += transferCycles;
  }

  return totalOverhead;
}

//===----------------------------------------------------------------------===//
// Temporal Schedule Computation
//===----------------------------------------------------------------------===//

std::string computeTemporalSchedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule) {

  // Only BATCH_SEQUENTIAL is currently supported.
  if (config.mode != ExecutionMode::BATCH_SEQUENTIAL) {
    std::string modeName = executionModeToString(config.mode);
    return modeName +
           std::string(" execution mode is not supported in current version; "
                       "only BATCH_SEQUENTIAL is implemented");
  }

  outSchedule = TemporalSchedule();
  outSchedule.mode = ExecutionMode::BATCH_SEQUENTIAL;

  // Build a lookup from kernel name to its achieved II.
  std::map<std::string, unsigned> kernelAchievedII;
  for (const auto &cs : costSummaries) {
    if (!cs.success)
      continue;
    for (const auto &km : cs.kernelMetrics) {
      kernelAchievedII[km.kernelName] = km.achievedII;
    }
  }

  // Process each core that has assigned kernels.
  for (const auto &coreAssign : assignment.coreAssignments) {
    if (coreAssign.assignedKernels.empty())
      continue;

    CoreSchedule coreSchedule;
    coreSchedule.coreInstanceName =
        coreAssign.coreTypeName + "_" +
        std::to_string(coreAssign.coreInstanceIdx);

    // Topological sort kernels on this core.
    coreSchedule.kernelOrder =
        topologicalSortKernels(coreAssign.assignedKernels, contracts);

    // Compute timing for each kernel.
    uint64_t totalCycles = 0;
    unsigned kernelIndex = 0;

    for (const auto &kernelName : coreSchedule.kernelOrder) {
      KernelTiming timing;
      timing.kernelName = kernelName;

      // Look up achieved II from mapper results.
      auto iiIt = kernelAchievedII.find(kernelName);
      if (iiIt != kernelAchievedII.end()) {
        timing.achievedII = iiIt->second;
      } else {
        // Fallback: use a default II of 1 if no mapper data available.
        timing.achievedII = 1;
      }

      // Look up trip count.
      timing.tripCount = lookupTripCount(kernelName, costSummaries);

      // Total kernel execution: tripCount * achievedII.
      timing.executionCycles =
          static_cast<uint64_t>(timing.tripCount) * timing.achievedII;

      totalCycles += timing.executionCycles;

      // Add reconfiguration cost between sequential kernels.
      if (kernelIndex > 0) {
        totalCycles += config.reconfigCycles;
        coreSchedule.reconfigCount++;
      }

      coreSchedule.kernelTimings.push_back(timing);
      kernelIndex++;
    }

    coreSchedule.totalCycles = totalCycles;
    outSchedule.coreSchedules.push_back(std::move(coreSchedule));
  }

  // System latency = max core latency + inter-core NoC overhead.
  uint64_t maxCoreCycles = 0;
  for (const auto &cs : outSchedule.coreSchedules) {
    if (cs.totalCycles > maxCoreCycles)
      maxCoreCycles = cs.totalCycles;
  }
  outSchedule.maxCoreCycles = maxCoreCycles;

  outSchedule.nocOverheadCycles =
      computeNoCOverhead(assignment, contracts);

  outSchedule.systemLatencyCycles =
      outSchedule.maxCoreCycles + outSchedule.nocOverheadCycles;

  return {};
}

} // namespace loom
