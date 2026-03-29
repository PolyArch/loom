//===-- TemporalScheduler.cpp - Multi-mode temporal scheduling impl --------===//
//
// Implements four temporal scheduling modes for the Tapestry system compiler:
// BATCH_SEQUENTIAL, PIPELINE_PARALLEL, SPATIAL_PARALLEL, and SPATIAL_SHARING.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/TemporalScheduler.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <sstream>

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: build kernel-to-achievedII lookup from cost summaries
//===----------------------------------------------------------------------===//

namespace {

std::map<std::string, unsigned>
buildAchievedIIMap(const std::vector<CoreCostSummary> &costSummaries) {
  std::map<std::string, unsigned> result;
  for (const auto &cs : costSummaries) {
    if (!cs.success)
      continue;
    for (const auto &km : cs.kernelMetrics) {
      result[km.kernelName] = km.achievedII;
    }
  }
  return result;
}

/// Build a kernel-to-core lookup from assignment.
std::map<std::string, unsigned>
buildKernelToCoreMap(const AssignmentResult &assignment) {
  return assignment.kernelToCore;
}

/// Compute execution cycles for a kernel given II and trip count.
uint64_t computeKernelDuration(unsigned tripCount, unsigned achievedII) {
  return static_cast<uint64_t>(tripCount) * achievedII;
}

/// Global topological sort of all kernels in the system.
std::vector<std::string>
globalTopologicalSort(const std::vector<std::string> &allKernels,
                      const std::vector<ContractSpec> &contracts) {
  if (allKernels.size() <= 1)
    return allKernels;

  std::set<std::string> kernelSet(allKernels.begin(), allKernels.end());

  std::map<std::string, std::vector<std::string>> adj;
  std::map<std::string, unsigned> inDegree;

  for (const auto &k : allKernels) {
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

  std::queue<std::string> ready;
  for (const auto &k : allKernels) {
    if (inDegree[k] == 0)
      ready.push(k);
  }

  std::vector<std::string> sorted;
  sorted.reserve(allKernels.size());

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

  if (sorted.size() != allKernels.size())
    return allKernels;

  return sorted;
}

/// Collect all kernel names from an assignment result.
std::vector<std::string>
collectAllKernels(const AssignmentResult &assignment) {
  std::vector<std::string> result;
  for (const auto &ca : assignment.coreAssignments) {
    for (const auto &k : ca.assignedKernels) {
      result.push_back(k);
    }
  }
  return result;
}

/// Compute pipeline initiation delay for a producer kernel.
/// This is the time for the producer to emit its first tile of output,
/// estimated as duration / tileCount.
uint64_t computePipelineInitiationDelay(uint64_t producerDuration,
                                        unsigned defaultTileCount) {
  unsigned tileCount = defaultTileCount > 0 ? defaultTileCount : 4;
  return producerDuration / tileCount;
}

/// Build a map from consumer kernel to its producer dependency edges.
std::map<std::string, std::vector<const ContractSpec *>>
buildConsumerDependencies(const std::vector<ContractSpec> &contracts) {
  std::map<std::string, std::vector<const ContractSpec *>> result;
  for (const auto &c : contracts) {
    result[c.consumerKernel].push_back(&c);
  }
  return result;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public dispatch
//===----------------------------------------------------------------------===//

std::string TemporalScheduler::schedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule) {

  switch (config.mode) {
  case ExecutionMode::BATCH_SEQUENTIAL:
    return computeTemporalSchedule(assignment, costSummaries, contracts,
                                   config, outSchedule);

  case ExecutionMode::PIPELINE_PARALLEL:
    return computePipelineSchedule(assignment, costSummaries, contracts,
                                   config, outSchedule);

  case ExecutionMode::SPATIAL_PARALLEL:
    return computeSpatialParallelSchedule(assignment, costSummaries, contracts,
                                          config, outSchedule);

  case ExecutionMode::SPATIAL_SHARING:
    return computeSpatialSharingSchedule(assignment, costSummaries, contracts,
                                         config, outSchedule);
  }

  return "Unknown execution mode";
}

//===----------------------------------------------------------------------===//
// PIPELINE_PARALLEL: ASAP greedy scheduling with pipeline overlap
//===----------------------------------------------------------------------===//

std::string TemporalScheduler::computePipelineSchedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule) {

  outSchedule = TemporalSchedule();
  outSchedule.mode = ExecutionMode::PIPELINE_PARALLEL;

  auto iiMap = buildAchievedIIMap(costSummaries);
  auto kernelToCoreMap = buildKernelToCoreMap(assignment);

  // If the L1 solver already produced start times, use them directly.
  bool useSolverTimes = !assignment.kernelStartTimes.empty();

  // Collect all kernels and perform global topological sort.
  auto allKernels = collectAllKernels(assignment);
  auto sortedKernels = globalTopologicalSort(allKernels, contracts);

  // Compute duration for each kernel.
  std::map<std::string, uint64_t> kernelDuration;
  for (const auto &kName : sortedKernels) {
    unsigned ii = 1;
    auto iiIt = iiMap.find(kName);
    if (iiIt != iiMap.end())
      ii = iiIt->second;
    unsigned tripCount = lookupTripCount(kName, costSummaries);
    kernelDuration[kName] = computeKernelDuration(tripCount, ii);
  }

  // Build consumer dependency map.
  auto consumerDeps = buildConsumerDependencies(contracts);

  // ASAP scheduling: assign start times respecting dependencies and
  // same-core serialization.
  std::map<std::string, uint64_t> startTimes;

  if (useSolverTimes) {
    startTimes = assignment.kernelStartTimes;
  } else {
    // Track per-core availability (earliest time the core is free).
    std::map<unsigned, uint64_t> coreAvailableAt;

    for (const auto &kName : sortedKernels) {
      uint64_t earliestStart = 0;

      // Dependency constraint: producer must have started + pipeline delay.
      auto depIt = consumerDeps.find(kName);
      if (depIt != consumerDeps.end()) {
        for (const auto *contract : depIt->second) {
          auto prodStartIt = startTimes.find(contract->producerKernel);
          if (prodStartIt == startTimes.end())
            continue;

          uint64_t prodStart = prodStartIt->second;
          uint64_t prodDuration = kernelDuration[contract->producerKernel];

          // Check if producer and consumer are on different cores.
          auto prodCoreIt = kernelToCoreMap.find(contract->producerKernel);
          auto consCoreIt = kernelToCoreMap.find(kName);
          bool sameCoreEdge =
              (prodCoreIt != kernelToCoreMap.end() &&
               consCoreIt != kernelToCoreMap.end() &&
               prodCoreIt->second == consCoreIt->second);

          if (sameCoreEdge) {
            // Same core: must wait for producer to finish + reconfig.
            uint64_t constraint =
                prodStart + prodDuration + config.reconfigCycles;
            if (constraint > earliestStart)
              earliestStart = constraint;
          } else {
            // Different cores: pipeline overlap allowed.
            uint64_t pipelineDelay = computePipelineInitiationDelay(
                prodDuration, config.defaultTileCount);
            uint64_t constraint = prodStart + pipelineDelay;
            if (constraint > earliestStart)
              earliestStart = constraint;
          }
        }
      }

      // Same-core serialization: must not overlap with other kernels
      // already scheduled on the same core.
      auto coreIt = kernelToCoreMap.find(kName);
      if (coreIt != kernelToCoreMap.end()) {
        unsigned coreIdx = coreIt->second;
        auto availIt = coreAvailableAt.find(coreIdx);
        if (availIt != coreAvailableAt.end()) {
          if (availIt->second > earliestStart)
            earliestStart = availIt->second;
        }
        // Update core availability: after this kernel finishes + reconfig.
        coreAvailableAt[coreIdx] =
            earliestStart + kernelDuration[kName] + config.reconfigCycles;
      }

      startTimes[kName] = earliestStart;
    }
  }

  // Build per-core schedules.
  std::map<unsigned, std::vector<std::string>> coreKernels;
  for (const auto &ca : assignment.coreAssignments) {
    for (const auto &k : ca.assignedKernels) {
      coreKernels[ca.coreInstanceIdx].push_back(k);
    }
  }

  // Sort kernels on each core by start time.
  for (auto &entry : coreKernels) {
    std::sort(entry.second.begin(), entry.second.end(),
              [&](const std::string &a, const std::string &b) {
                return startTimes[a] < startTimes[b];
              });
  }

  uint64_t maxEndTime = 0;

  for (const auto &coreAssign : assignment.coreAssignments) {
    if (coreAssign.assignedKernels.empty())
      continue;

    CoreSchedule coreSchedule;
    coreSchedule.coreInstanceName =
        coreAssign.coreTypeName + "_" +
        std::to_string(coreAssign.coreInstanceIdx);

    auto &kernels = coreKernels[coreAssign.coreInstanceIdx];
    coreSchedule.kernelOrder = kernels;

    uint64_t coreEndTime = 0;
    unsigned prevIdx = 0;

    for (const auto &kName : kernels) {
      KernelTiming timing;
      timing.kernelName = kName;

      auto iiIt = iiMap.find(kName);
      timing.achievedII = (iiIt != iiMap.end()) ? iiIt->second : 1;
      timing.tripCount = lookupTripCount(kName, costSummaries);
      timing.executionCycles = computeKernelDuration(timing.tripCount,
                                                     timing.achievedII);
      timing.startTime = startTimes[kName];

      uint64_t endTime = timing.startTime + timing.executionCycles;
      if (endTime > coreEndTime)
        coreEndTime = endTime;

      if (prevIdx > 0)
        coreSchedule.reconfigCount++;

      coreSchedule.kernelTimings.push_back(timing);
      prevIdx++;
    }

    coreSchedule.totalCycles = coreEndTime;
    if (coreEndTime > maxEndTime)
      maxEndTime = coreEndTime;

    outSchedule.coreSchedules.push_back(std::move(coreSchedule));
  }

  outSchedule.maxCoreCycles = maxEndTime;
  outSchedule.nocOverheadCycles =
      computeNoCOverhead(assignment, contracts);
  outSchedule.systemLatencyCycles =
      outSchedule.maxCoreCycles + outSchedule.nocOverheadCycles;

  return {};
}

//===----------------------------------------------------------------------===//
// SPATIAL_PARALLEL: all cores start at time 0, sequential within cores
//===----------------------------------------------------------------------===//

std::string TemporalScheduler::computeSpatialParallelSchedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule) {

  outSchedule = TemporalSchedule();
  outSchedule.mode = ExecutionMode::SPATIAL_PARALLEL;

  auto iiMap = buildAchievedIIMap(costSummaries);

  for (const auto &coreAssign : assignment.coreAssignments) {
    if (coreAssign.assignedKernels.empty())
      continue;

    CoreSchedule coreSchedule;
    coreSchedule.coreInstanceName =
        coreAssign.coreTypeName + "_" +
        std::to_string(coreAssign.coreInstanceIdx);

    // Topological sort within this core.
    coreSchedule.kernelOrder =
        topologicalSortKernels(coreAssign.assignedKernels, contracts);

    // All cores start at time 0. Within each core, kernels are sequential.
    uint64_t coreOffset = 0;
    unsigned kernelIndex = 0;

    for (const auto &kName : coreSchedule.kernelOrder) {
      KernelTiming timing;
      timing.kernelName = kName;

      auto iiIt = iiMap.find(kName);
      timing.achievedII = (iiIt != iiMap.end()) ? iiIt->second : 1;
      timing.tripCount = lookupTripCount(kName, costSummaries);
      timing.executionCycles = computeKernelDuration(timing.tripCount,
                                                     timing.achievedII);

      // Add reconfig gap before non-first kernels.
      if (kernelIndex > 0) {
        coreOffset += config.reconfigCycles;
        coreSchedule.reconfigCount++;
      }

      // All cores start at 0; within a core, kernels are sequential.
      timing.startTime = coreOffset;
      coreOffset += timing.executionCycles;

      coreSchedule.kernelTimings.push_back(timing);
      kernelIndex++;
    }

    coreSchedule.totalCycles = coreOffset;
    outSchedule.coreSchedules.push_back(std::move(coreSchedule));
  }

  // System latency = max core latency + NoC overhead.
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

//===----------------------------------------------------------------------===//
// SPATIAL_SHARING: concurrent kernels via partitioning, no reconfig
//===----------------------------------------------------------------------===//

std::string TemporalScheduler::computeSpatialSharingSchedule(
    const AssignmentResult &assignment,
    const std::vector<CoreCostSummary> &costSummaries,
    const std::vector<ContractSpec> &contracts,
    const ExecutionModelConfig &config,
    TemporalSchedule &outSchedule) {

  outSchedule = TemporalSchedule();
  outSchedule.mode = ExecutionMode::SPATIAL_SHARING;

  auto iiMap = buildAchievedIIMap(costSummaries);

  for (const auto &coreAssign : assignment.coreAssignments) {
    if (coreAssign.assignedKernels.empty())
      continue;

    CoreSchedule coreSchedule;
    coreSchedule.coreInstanceName =
        coreAssign.coreTypeName + "_" +
        std::to_string(coreAssign.coreInstanceIdx);

    coreSchedule.kernelOrder = coreAssign.assignedKernels;

    // In SPATIAL_SHARING, all kernels on a core execute concurrently.
    // Per-core latency = max(kernel execution cycles).
    // No reconfiguration penalty.
    uint64_t maxKernelCycles = 0;

    for (const auto &kName : coreSchedule.kernelOrder) {
      KernelTiming timing;
      timing.kernelName = kName;

      auto iiIt = iiMap.find(kName);
      timing.achievedII = (iiIt != iiMap.end()) ? iiIt->second : 1;
      timing.tripCount = lookupTripCount(kName, costSummaries);
      timing.executionCycles = computeKernelDuration(timing.tripCount,
                                                     timing.achievedII);

      // All kernels start simultaneously at time 0 on this core.
      timing.startTime = 0;

      if (timing.executionCycles > maxKernelCycles)
        maxKernelCycles = timing.executionCycles;

      coreSchedule.kernelTimings.push_back(timing);
    }

    // No reconfiguration in spatial sharing mode.
    coreSchedule.reconfigCount = 0;
    coreSchedule.totalCycles = maxKernelCycles;

    outSchedule.coreSchedules.push_back(std::move(coreSchedule));
  }

  // System latency = max(per-core latency) + NoC overhead.
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
