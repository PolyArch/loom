#include "fcc/Simulator/CycleKernel.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>

namespace fcc {
namespace sim {

namespace {

double safeRatio(uint64_t num, uint64_t den) {
  if (den == 0)
    return 0.0;
  return static_cast<double>(num) / static_cast<double>(den);
}

const char *moduleKindLabel(StaticModuleKind kind) {
  switch (kind) {
  case StaticModuleKind::BoundaryInput:
    return "boundary_input";
  case StaticModuleKind::BoundaryOutput:
    return "boundary_output";
  case StaticModuleKind::FunctionUnit:
    return "function_unit";
  case StaticModuleKind::SpatialSwitch:
    return "spatial_switch";
  case StaticModuleKind::TemporalSwitch:
    return "temporal_switch";
  case StaticModuleKind::AddTag:
    return "add_tag";
  case StaticModuleKind::MapTag:
    return "map_tag";
  case StaticModuleKind::DelTag:
    return "del_tag";
  case StaticModuleKind::Fifo:
    return "fifo";
  case StaticModuleKind::Memory:
    return "memory";
  case StaticModuleKind::ExtMemory:
    return "extmemory";
  case StaticModuleKind::TemporalPE:
    return "temporal_pe";
  case StaticModuleKind::Unknown:
  default:
    return "unknown";
  }
}

uint64_t counterValue(const std::vector<NamedCounter> &counters,
                      const char *name) {
  for (const NamedCounter &counter : counters) {
    if (counter.name == name)
      return counter.value;
  }
  return 0;
}

} // namespace

AcceleratorStats CycleKernel::buildAcceleratorStats(
    uint64_t configLoadStartCycle, uint64_t configLoadCycles,
    uint64_t kernelLaunchCycle, uint64_t configDmaRequestCount,
    uint64_t configDmaReadBytes) const {
  AcceleratorStats stats;
  const uint64_t configLoadEndCycle = configLoadStartCycle + configLoadCycles;
  const uint64_t kernelEndCycle = kernelLaunchCycle + currentCycle_;

  stats.totalCycles = std::max(configLoadEndCycle, kernelEndCycle);
  stats.kernelCycles = currentCycle_;
  stats.loadRequestCount = loadRequestCount_;
  stats.storeRequestCount = storeRequestCount_;
  stats.loadBytes = loadBytes_;
  stats.storeBytes = storeBytes_;

  stats.configLoad.wordCount = configImage_.words.size();
  stats.configLoad.byteCount = configImage_.words.size() * sizeof(uint32_t);
  stats.configLoad.wordsPerCycle = config_.configWordsPerCycle;
  stats.configLoad.startCycle = configLoadStartCycle;
  stats.configLoad.endCycle = configLoadEndCycle;
  stats.configLoad.cycles = configLoadCycles;
  stats.configLoad.dmaRequestCount = configDmaRequestCount;
  stats.configLoad.dmaReadBytes = configDmaReadBytes;
  stats.configLoad.kernelLaunchCycle = kernelLaunchCycle;

  uint64_t firstKernelActivityCycle = kernelLaunchCycle;
  bool foundKernelActivity = false;
  std::unordered_map<uint32_t, uint64_t> firstUseCycleByNode;
  firstUseCycleByNode.reserve(traceDocument_.modules.size());
  for (const TraceEvent &event : traceDocument_.events) {
    if (event.eventKind == EventKind::InvocationStart ||
        event.eventKind == EventKind::InvocationDone ||
        event.eventKind == EventKind::DeviceError)
      continue;
    uint64_t absoluteCycle = kernelLaunchCycle + event.cycle;
    if (!foundKernelActivity) {
      firstKernelActivityCycle = absoluteCycle;
      foundKernelActivity = true;
    }
    auto it = firstUseCycleByNode.find(event.hwNodeId);
    if (it == firstUseCycleByNode.end())
      firstUseCycleByNode.emplace(event.hwNodeId, absoluteCycle);
    else
      it->second = std::min(it->second, absoluteCycle);
  }
  stats.configLoad.kernelFirstActiveCycle = firstKernelActivityCycle;
  const uint64_t overlapStart = std::max(kernelLaunchCycle, configLoadStartCycle);
  const uint64_t overlapEnd = std::min(kernelEndCycle, configLoadEndCycle);
  stats.configLoad.configExecOverlapCycles =
      (overlapEnd > overlapStart) ? (overlapEnd - overlapStart) : 0;
  stats.configLoad.configExecExposedCycles =
      configLoadCycles - stats.configLoad.configExecOverlapCycles;
  stats.configLoad.configOverlapEfficiency =
      safeRatio(stats.configLoad.configExecOverlapCycles, configLoadCycles);

  stats.configSlices = configSliceTimings_;
  for (ConfigSliceTiming &slice : stats.configSlices) {
    slice.startCycle += configLoadStartCycle;
    slice.endCycle += configLoadStartCycle;
  }

  std::unordered_map<uint32_t, PerfSnapshot> perfByNode;
  perfByNode.reserve(modules_.size());
  for (const auto &module : modules_) {
    PerfSnapshot perf = module->getPerfSnapshot();
    perf.nodeIndex = module->hwNodeId;
    perfByNode[module->hwNodeId] = perf;
  }

  stats.modules.reserve(modules_.size());
  std::set<std::string> usedSpatialPEs;
  std::set<std::string> usedTemporalPEs;
  for (const auto &module : modules_) {
    ModulePerfDetail detail;
    detail.hwNodeId = module->hwNodeId;
    detail.name = module->name;
    detail.kind = moduleKindLabel(module->kind);
    auto componentIt = moduleComponentName_.find(module->hwNodeId);
    detail.componentName =
        (componentIt != moduleComponentName_.end()) ? componentIt->second : module->name;
    auto fuNameIt = moduleFunctionUnitName_.find(module->hwNodeId);
    detail.functionUnitName =
        (fuNameIt != moduleFunctionUnitName_.end()) ? fuNameIt->second : module->name;
    detail.configured = configuredModuleNodes_.count(module->hwNodeId) != 0;
    detail.staticallyUsed =
        detail.configured || configuredFunctionUnits_.count(module->hwNodeId) != 0;

    auto readyIt = moduleConfigReadyCycle_.find(module->hwNodeId);
    if (readyIt != moduleConfigReadyCycle_.end())
      detail.configReadyCycle = configLoadStartCycle + readyIt->second;

    auto firstUseIt = firstUseCycleByNode.find(module->hwNodeId);
    if (firstUseIt != firstUseCycleByNode.end()) {
      detail.hasFirstUseCycle = true;
      detail.firstUseCycle = firstUseIt->second;
      detail.configSlackCycles =
          static_cast<int64_t>(detail.firstUseCycle) -
          static_cast<int64_t>(detail.configReadyCycle);
    }

    auto perfIt = perfByNode.find(module->hwNodeId);
    if (perfIt != perfByNode.end()) {
      const PerfSnapshot &perf = perfIt->second;
      detail.activeCycles = perf.activeCycles;
      detail.stallCyclesIn = perf.stallCyclesIn;
      detail.stallCyclesOut = perf.stallCyclesOut;
      detail.tokensIn = perf.tokensIn;
      detail.tokensOut = perf.tokensOut;
    }

    detail.logicalFireCount = module->getLogicalFireCount();
    detail.inputCaptureCount = module->getInputCaptureCount();
    detail.outputTransferCount = module->getOutputTransferCount();
    detail.counters = module->getDebugCounters();
    detail.outputBusyCycles =
        counterValue(detail.counters, "output_busy_cycles");
    detail.inputLatchedCycles =
        counterValue(detail.counters, "input_latched_cycles");
    detail.dynamicUtilization = safeRatio(detail.activeCycles, currentCycle_);
    detail.dynamicallyUsed =
        detail.activeCycles != 0 || detail.stallCyclesIn != 0 ||
        detail.stallCyclesOut != 0 || detail.tokensIn != 0 ||
        detail.tokensOut != 0 || detail.logicalFireCount != 0 ||
        detail.inputCaptureCount != 0 || detail.outputTransferCount != 0 ||
        detail.hasFirstUseCycle;

    if (detail.kind == "function_unit" && detail.staticallyUsed &&
        !detail.componentName.empty()) {
      if (detail.componentName.find("temporal") != std::string::npos)
        usedTemporalPEs.insert(detail.componentName);
      else
        usedSpatialPEs.insert(detail.componentName);
    }
    if (detail.kind == "temporal_pe" && detail.staticallyUsed)
      usedTemporalPEs.insert(detail.componentName);

    stats.modules.push_back(std::move(detail));
  }

  stats.staticUtilization.totalModules = staticModel_.getModules().size();
  stats.staticUtilization.configuredModules = configuredModuleNodes_.size();
  for (const StaticModuleDesc &module : staticModel_.getModules()) {
    if (module.kind == StaticModuleKind::FunctionUnit)
      ++stats.staticUtilization.totalFunctionUnits;
  }
  stats.staticUtilization.mappedFunctionUnits = configuredFunctionUnits_.size();
  for (const StaticPEDesc &pe : staticModel_.getPEs()) {
    if (pe.peKind == "spatial_pe")
      ++stats.staticUtilization.totalSpatialPEs;
    else if (pe.peKind == "temporal_pe")
      ++stats.staticUtilization.totalTemporalPEs;
  }
  stats.staticUtilization.usedSpatialPEs = usedSpatialPEs.size();
  stats.staticUtilization.usedTemporalPEs = usedTemporalPEs.size();
  stats.staticUtilization.configuredModuleRatio = safeRatio(
      stats.staticUtilization.configuredModules,
      stats.staticUtilization.totalModules);
  stats.staticUtilization.mappedFunctionUnitRatio = safeRatio(
      stats.staticUtilization.mappedFunctionUnits,
      stats.staticUtilization.totalFunctionUnits);
  stats.staticUtilization.usedSpatialPERatio = safeRatio(
      stats.staticUtilization.usedSpatialPEs,
      stats.staticUtilization.totalSpatialPEs);
  stats.staticUtilization.usedTemporalPERatio = safeRatio(
      stats.staticUtilization.usedTemporalPEs,
      stats.staticUtilization.totalTemporalPEs);

  stats.dynamicUtilization.kernelCycles = currentCycle_;
  stats.dynamicUtilization.activeCycles = activeCycles_;
  stats.dynamicUtilization.idleCycles = idleCycles_;
  stats.dynamicUtilization.fabricActiveCycles = fabricActiveCycles_;
  stats.dynamicUtilization.needMemIssueCycles = needMemIssueCycles_;
  stats.dynamicUtilization.waitMemRespCycles = waitMemRespCycles_;
  stats.dynamicUtilization.budgetBoundaryCount = budgetBoundaryCount_;
  stats.dynamicUtilization.deadlockBoundaryCount = deadlockBoundaryCount_;
  stats.dynamicUtilization.maxInflightMemoryRequests =
      maxInflightMemoryRequests_;
  stats.dynamicUtilization.activeCycleRatio =
      safeRatio(stats.dynamicUtilization.activeCycles, currentCycle_);
  stats.dynamicUtilization.fabricActiveRatio =
      safeRatio(stats.dynamicUtilization.fabricActiveCycles, currentCycle_);
  stats.dynamicUtilization.memIssueRatio =
      safeRatio(stats.dynamicUtilization.needMemIssueCycles, currentCycle_);
  stats.dynamicUtilization.memWaitRatio =
      safeRatio(stats.dynamicUtilization.waitMemRespCycles, currentCycle_);

  stats.memoryRegions.reserve(memoryRegionPerf_.size());
  for (const auto &entry : memoryRegionPerf_) {
    MemoryRegionPerfSummary region;
    region.regionId = entry.first;
    region.loadRequestCount = entry.second.loadRequestCount;
    region.storeRequestCount = entry.second.storeRequestCount;
    region.loadBytes = entry.second.loadBytes;
    region.storeBytes = entry.second.storeBytes;
    region.hasFirstRequestCycle = entry.second.hasFirstRequestCycle;
    region.firstRequestCycle = entry.second.firstRequestCycle + kernelLaunchCycle;
    region.hasLastCompletionCycle = entry.second.hasLastCompletionCycle;
    region.lastCompletionCycle =
        entry.second.lastCompletionCycle + kernelLaunchCycle;
    stats.memoryRegions.push_back(std::move(region));
  }
  std::sort(stats.memoryRegions.begin(), stats.memoryRegions.end(),
            [](const MemoryRegionPerfSummary &lhs,
               const MemoryRegionPerfSummary &rhs) {
              return lhs.regionId < rhs.regionId;
            });
  std::sort(stats.modules.begin(), stats.modules.end(),
            [](const ModulePerfDetail &lhs, const ModulePerfDetail &rhs) {
              return lhs.hwNodeId < rhs.hwNodeId;
            });
  std::sort(stats.configSlices.begin(), stats.configSlices.end(),
            [](const ConfigSliceTiming &lhs, const ConfigSliceTiming &rhs) {
              if (lhs.startCycle != rhs.startCycle)
                return lhs.startCycle < rhs.startCycle;
              return lhs.wordOffset < rhs.wordOffset;
            });

  return stats;
}

} // namespace sim
} // namespace fcc
