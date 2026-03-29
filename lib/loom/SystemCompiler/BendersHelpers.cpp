#include "loom/SystemCompiler/BendersHelpers.h"

#include <algorithm>
#include <map>
#include <numeric>

namespace loom {

/// Find contracts where the given kernel is the consumer.
static std::vector<ContractSpec>
contractsInto(const std::string &kernelName,
              const std::vector<ContractSpec> &contracts) {
  std::vector<ContractSpec> result;
  for (const auto &c : contracts) {
    if (c.consumerKernel == kernelName)
      result.push_back(c);
  }
  return result;
}

/// Find contracts where the given kernel is the producer.
static std::vector<ContractSpec>
contractsOutOf(const std::string &kernelName,
               const std::vector<ContractSpec> &contracts) {
  std::vector<ContractSpec> result;
  for (const auto &c : contracts) {
    if (c.producerKernel == kernelName)
      result.push_back(c);
  }
  return result;
}

/// Generate a core instance name from type name and instance index.
static std::string makeCoreInstanceName(const std::string &coreType,
                                        unsigned instanceIdx) {
  return coreType + "_" + std::to_string(instanceIdx);
}

std::vector<L2Assignment>
buildL2Assignments(const AssignmentResult &assignment,
                   const std::map<std::string, mlir::ModuleOp> &kernelDFGs,
                   const std::vector<ContractSpec> &contracts,
                   const SystemArchitecture &arch) {
  std::vector<L2Assignment> l2Assignments;
  l2Assignments.reserve(assignment.coreAssignments.size());

  for (const auto &coreAssign : assignment.coreAssignments) {
    if (coreAssign.assignedKernels.empty())
      continue;

    L2Assignment l2;
    l2.coreInstanceName =
        makeCoreInstanceName(coreAssign.coreTypeName,
                             coreAssign.coreInstanceIdx);
    l2.coreType = coreAssign.coreTypeName;
    // coreADG will be set by the caller or looked up from the architecture.
    // For now, leave it as a null ModuleOp.

    for (const auto &kernelName : coreAssign.assignedKernels) {
      L2Assignment::KernelAssignment ka;
      ka.kernelName = kernelName;

      auto it = kernelDFGs.find(kernelName);
      if (it != kernelDFGs.end())
        ka.kernelDFG = it->second;

      ka.inputContracts = contractsInto(kernelName, contracts);
      ka.outputContracts = contractsOutOf(kernelName, contracts);

      l2.kernels.push_back(std::move(ka));
    }

    l2Assignments.push_back(std::move(l2));
  }

  return l2Assignments;
}

void updateContractCosts(std::vector<ContractSpec> &contracts,
                         const std::vector<CoreCostSummary> &costSummaries) {
  // Build a lookup from kernel name to achieved stream rate.
  std::map<std::string, double> achievedRates;
  for (const auto &summary : costSummaries) {
    if (!summary.success)
      continue;
    for (const auto &km : summary.kernelMetrics) {
      if (km.achievedStreamRate > 0.0)
        achievedRates[km.kernelName] = km.achievedStreamRate;
    }
  }

  // Update production rate estimates on contracts whose producer has
  // achieved metrics.
  for (auto &contract : contracts) {
    auto it = achievedRates.find(contract.producerKernel);
    if (it != achievedRates.end()) {
      contract.achievedProductionRate =
          static_cast<int64_t>(it->second);
    }
  }
}

double computeObjective(const AssignmentResult &assignment,
                        const NoCSchedule &nocSchedule,
                        const std::vector<CoreCostSummary> &costSummaries) {
  // Objective = latencyWeight * criticalPathLatency + nocWeight * nocCycles.
  //
  // Critical path = max over all cores of:
  //   sum(tripCount * achievedII) + reconfigCycles * (numKernels - 1)
  //
  // Note: tripCount defaults to 1000 when not annotated.
  constexpr double latencyWeight = 1.0;
  constexpr double nocWeight = 0.5;
  constexpr unsigned kDefaultTripCount = 1000;
  constexpr unsigned kDefaultReconfigCycles = 100;

  // Build per-core latency using the assignment structure.
  // Group kernel metrics by core instance name.
  std::map<std::string, std::vector<const KernelMetrics *>> coreKernelMetrics;
  for (const auto &cs : costSummaries) {
    if (!cs.success)
      continue;
    for (const auto &km : cs.kernelMetrics) {
      coreKernelMetrics[cs.coreInstanceName].push_back(&km);
    }
  }

  double criticalPath = 0.0;
  for (const auto &entry : coreKernelMetrics) {
    const auto &kernelList = entry.second;
    double coreLatency = 0.0;
    for (const auto *km : kernelList) {
      // Total kernel execution = tripCount * achievedII.
      unsigned tripCount = kDefaultTripCount;
      coreLatency += static_cast<double>(tripCount) * km->achievedII;
    }
    // Add reconfiguration gaps between sequential kernels on the same core.
    if (kernelList.size() > 1) {
      coreLatency += kDefaultReconfigCycles * (kernelList.size() - 1);
    }
    if (coreLatency > criticalPath)
      criticalPath = coreLatency;
  }

  double nocCost = static_cast<double>(nocSchedule.totalTransferCycles);

  return latencyWeight * criticalPath + nocWeight * nocCost;
}

double computeObjective(const AssignmentResult &assignment,
                        const NoCSchedule &nocSchedule,
                        const BufferAllocationPlan &bufferPlan,
                        const std::vector<CoreCostSummary> &costSummaries) {
  // Start with the base objective (latency + NoC).
  double base = computeObjective(assignment, nocSchedule, costSummaries);

  // Add buffer allocation penalty.
  // SPM allocations incur no penalty; L2 and DRAM allocations add cost
  // proportional to their data volumes.
  constexpr double bufferWeight = 0.1;
  constexpr double l2CostPerByte = 0.01;
  constexpr double dramCostPerByte = 0.1;

  double bufferPenalty = 0.0;
  for (const auto &alloc : bufferPlan.allocations) {
    switch (alloc.location) {
    case BufferAllocation::SPM_PRODUCER:
    case BufferAllocation::SPM_CONSUMER:
      // No penalty for SPM allocations.
      break;
    case BufferAllocation::SHARED_L2:
      bufferPenalty += l2CostPerByte * static_cast<double>(alloc.sizeBytes);
      break;
    case BufferAllocation::EXTERNAL_DRAM:
      bufferPenalty += dramCostPerByte * static_cast<double>(alloc.sizeBytes);
      break;
    }
  }

  return base + bufferWeight * bufferPenalty;
}

TapestryCompilationResult
assembleResult(const std::vector<L2Result> &l2Results,
               const std::vector<L2Assignment> &l2Assignments,
               const AssignmentResult &assignment,
               const NoCSchedule &nocSchedule,
               const BufferAllocationPlan &bufferPlan,
               const DMASchedule &dmaSchedule,
               const std::vector<CoreCostSummary> &costSummaries) {
  TapestryCompilationResult result;
  result.success = true;
  result.finalAssignment = assignment;
  result.finalNoCSchedule = nocSchedule;
  result.finalBufferPlan = bufferPlan;
  result.finalDMASchedule = dmaSchedule;

  // Build per-core results.
  for (size_t i = 0; i < l2Results.size() && i < l2Assignments.size(); ++i) {
    const auto &l2r = l2Results[i];
    const auto &l2a = l2Assignments[i];

    TapestryCoreResult cr;
    cr.coreInstanceName = l2a.coreInstanceName;
    cr.coreType = l2a.coreType;
    cr.adgModule = l2a.coreADG;

    for (const auto &ka : l2a.kernels)
      cr.assignedKernels.push_back(ka.kernelName);

    cr.l2Result = l2r;
    cr.aggregateConfigBlob = l2r.aggregateConfig;

    // Match NoC routes for this core.
    for (const auto &route : nocSchedule.routes) {
      if (route.producerCore == l2a.coreInstanceName ||
          route.consumerCore == l2a.coreInstanceName) {
        cr.nocRoute = route;
        break;
      }
    }

    // Match buffer allocations for this core.
    for (const auto &alloc : bufferPlan.allocations) {
      if (alloc.coreInstanceIdx ==
          assignment.coreAssignments[i].coreInstanceIdx) {
        cr.buffers = alloc;
        break;
      }
    }

    result.coreResults.push_back(std::move(cr));
  }

  return result;
}

} // namespace loom
