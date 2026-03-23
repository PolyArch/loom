#include "loom/SystemCompiler/BendersDriver.h"

#include <algorithm>
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
  // Objective = latencyWeight * totalLatency + nocWeight * nocTransferCycles.
  constexpr double latencyWeight = 1.0;
  constexpr double nocWeight = 0.5;

  double totalLatency = 0.0;
  for (const auto &cs : costSummaries) {
    if (!cs.success)
      continue;
    for (const auto &km : cs.kernelMetrics) {
      totalLatency += static_cast<double>(km.achievedII);
    }
  }

  double nocCost = static_cast<double>(nocSchedule.totalTransferCycles);

  return latencyWeight * totalLatency + nocWeight * nocCost;
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

    CoreResult cr;
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
