#include "loom/SystemCompiler/FlatCompilerBaselines.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#ifdef LOOM_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#endif

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: evaluate an assignment's objective value
//===----------------------------------------------------------------------===//

namespace {

/// Evaluate the objective value for a given assignment.
double evaluateAssignment(const std::map<std::string, unsigned> &kernelToCore,
                          const std::vector<KernelProfile> &kernels,
                          const std::vector<ContractSpec> &contracts,
                          const SystemArchitecture &arch) {
  double latencyCost = 0.0;
  for (const auto &kernel : kernels) {
    latencyCost += kernel.estimatedComputeCycles;
  }

  double nocCost = 0.0;
  for (const auto &contract : contracts) {
    auto pIt = kernelToCore.find(contract.producerKernel);
    auto cIt = kernelToCore.find(contract.consumerKernel);
    if (pIt == kernelToCore.end() || cIt == kernelToCore.end())
      continue;
    if (pIt->second != cIt->second) {
      int dist =
          manhattanDistance(pIt->second, cIt->second, arch.nocSpec.meshCols);
      int64_t volume = 1;
      if (contract.productionRate.has_value())
        volume = contract.productionRate.value();
      volume *= estimateElementSize(contract.dataTypeName);
      nocCost += dist * volume;
    }
  }

  return latencyCost + 0.5 * nocCost;
}

/// Check if an assignment is feasible (respects capacity and compatibility).
bool isAssignmentFeasible(const std::map<std::string, unsigned> &kernelToCore,
                          const std::vector<KernelProfile> &kernels,
                          const SystemArchitecture &arch) {
  unsigned numCores = arch.totalCoreInstances();

  // Per-core resource accumulator.
  std::vector<std::map<std::string, unsigned>> coreFuUsage(numCores);
  std::vector<uint64_t> coreSpmUsage(numCores, 0);

  for (const auto &kernel : kernels) {
    auto it = kernelToCore.find(kernel.name);
    if (it == kernelToCore.end())
      return false;
    unsigned c = it->second;
    if (c >= numCores)
      return false;

    // Check type compatibility.
    if (!isKernelCompatible(kernel, arch.typeForInstance(c)))
      return false;

    // Accumulate FU usage.
    for (const auto &opEntry : kernel.requiredOps)
      coreFuUsage[c][opEntry.first] += opEntry.second;

    coreSpmUsage[c] += kernel.estimatedSPMBytes;
  }

  // Check capacity constraints.
  for (unsigned c = 0; c < numCores; ++c) {
    const auto &coreType = arch.typeForInstance(c);
    for (const auto &fuEntry : coreFuUsage[c]) {
      auto it = coreType.fuTypeCounts.find(fuEntry.first);
      if (it == coreType.fuTypeCounts.end() || fuEntry.second > it->second)
        return false;
    }
    if (coreSpmUsage[c] > coreType.spmBytes)
      return false;
  }

  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// MonolithicILPBaseline
//===----------------------------------------------------------------------===//

#ifdef LOOM_HAVE_ORTOOLS

using namespace operations_research::sat;

FlatBaselineResult MonolithicILPBaseline::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch, const Options &opts) {

  FlatBaselineResult result;
  result.methodName = "MonolithicILP";

  unsigned numKernels = static_cast<unsigned>(kernels.size());
  unsigned numCores = arch.totalCoreInstances();

  if (numKernels == 0 || numCores == 0) {
    result.feasible = (numKernels == 0);
    return result;
  }

  // Build kernel name -> index map.
  std::map<std::string, unsigned> kernelIdx;
  for (unsigned k = 0; k < numKernels; ++k)
    kernelIdx[kernels[k].name] = k;

  CpModelBuilder model;

  // Decision variables: x[k][c].
  std::vector<std::vector<BoolVar>> x(numKernels);
  for (unsigned k = 0; k < numKernels; ++k) {
    x[k].resize(numCores);
    for (unsigned c = 0; c < numCores; ++c) {
      x[k][c] = model.NewBoolVar().WithName(
          "m_x_k" + std::to_string(k) + "_c" + std::to_string(c));
    }
  }

  // Each kernel to exactly one core.
  for (unsigned k = 0; k < numKernels; ++k)
    model.AddExactlyOne(x[k]);

  // Type compatibility.
  for (unsigned k = 0; k < numKernels; ++k) {
    for (unsigned c = 0; c < numCores; ++c) {
      if (!isKernelCompatible(kernels[k], arch.typeForInstance(c)))
        model.FixVariable(x[k][c], false);
    }
  }

  // FU capacity.
  for (unsigned c = 0; c < numCores; ++c) {
    const auto &coreType = arch.typeForInstance(c);
    for (const auto &fuEntry : coreType.fuTypeCounts) {
      LinearExpr demand;
      bool hasDemand = false;
      for (unsigned k = 0; k < numKernels; ++k) {
        auto opIt = kernels[k].requiredOps.find(fuEntry.first);
        if (opIt != kernels[k].requiredOps.end() && opIt->second > 0) {
          demand += x[k][c] * static_cast<int64_t>(opIt->second);
          hasDemand = true;
        }
      }
      if (hasDemand)
        model.AddLessOrEqual(demand, static_cast<int64_t>(fuEntry.second));
    }

    // SPM capacity.
    if (coreType.spmBytes > 0) {
      LinearExpr spmDemand;
      bool hasSpmDemand = false;
      for (unsigned k = 0; k < numKernels; ++k) {
        if (kernels[k].estimatedSPMBytes > 0) {
          spmDemand +=
              x[k][c] * static_cast<int64_t>(kernels[k].estimatedSPMBytes);
          hasSpmDemand = true;
        }
      }
      if (hasSpmDemand)
        model.AddLessOrEqual(spmDemand,
                             static_cast<int64_t>(coreType.spmBytes));
    }
  }

  // Objective: latency + NoC cost (monolithic formulation).
  constexpr int64_t kScale = 1000;
  LinearExpr objective;

  for (unsigned k = 0; k < numKernels; ++k) {
    int64_t cost = static_cast<int64_t>(
        std::llround(kernels[k].estimatedComputeCycles));
    if (cost <= 0)
      cost = static_cast<int64_t>(kernels[k].totalOpCount());
    for (unsigned c = 0; c < numCores; ++c)
      objective += x[k][c] * cost;
  }

  // NoC cost with linearized product.
  for (const auto &contract : contracts) {
    auto pkIt = kernelIdx.find(contract.producerKernel);
    auto ckIt = kernelIdx.find(contract.consumerKernel);
    if (pkIt == kernelIdx.end() || ckIt == kernelIdx.end())
      continue;
    unsigned pk = pkIt->second;
    unsigned ck = ckIt->second;
    int64_t volume = 1;
    if (contract.productionRate.has_value())
      volume = contract.productionRate.value();
    volume *= estimateElementSize(contract.dataTypeName);

    for (unsigned cp = 0; cp < numCores; ++cp) {
      for (unsigned cc = 0; cc < numCores; ++cc) {
        if (cp == cc)
          continue;
        int dist = manhattanDistance(cp, cc, arch.nocSpec.meshCols);
        if (dist == 0)
          continue;
        int64_t pairCost = dist * volume / 2; // 0.5 weight
        if (pairCost <= 0)
          continue;
        BoolVar both = model.NewBoolVar();
        model.AddImplication(both, x[pk][cp]);
        model.AddImplication(both, x[ck][cc]);
        model.AddBoolOr({both, x[pk][cp].Not(), x[ck][cc].Not()});
        objective += both * pairCost;
      }
    }
  }

  model.Minimize(objective);

  // Solve with timeout.
  Model satModel;
  SatParameters params;
  params.set_max_time_in_seconds(static_cast<double>(opts.timeoutSec));
  unsigned numWorkers = opts.numWorkers;
  if (numWorkers == 0) {
    numWorkers = std::thread::hardware_concurrency();
    if (numWorkers == 0)
      numWorkers = 4;
    numWorkers = std::min(numWorkers, 8u);
  }
  params.set_num_search_workers(static_cast<int>(numWorkers));
  satModel.Add(NewSatParameters(params));

  auto startTime = std::chrono::steady_clock::now();
  const CpSolverResponse response = SolveCpModel(model.Build(), &satModel);
  auto endTime = std::chrono::steady_clock::now();

  result.solveTimeSec =
      std::chrono::duration<double>(endTime - startTime).count();

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    result.feasible = true;
    result.assignment.feasible = true;
    for (unsigned k = 0; k < numKernels; ++k) {
      for (unsigned c = 0; c < numCores; ++c) {
        if (SolutionBooleanValue(response, x[k][c])) {
          result.assignment.kernelToCore[kernels[k].name] = c;
          break;
        }
      }
    }
    result.assignment.objectiveValue =
        static_cast<double>(response.objective_value()) / kScale;
  } else {
    result.timedOut =
        (response.status() == CpSolverStatus::UNKNOWN);
  }

  if (opts.verbose) {
    llvm::outs() << "Monolithic ILP: "
                 << (result.feasible ? "feasible" : "infeasible/timeout")
                 << " in " << result.solveTimeSec << "s\n";
  }

  return result;
}

#else // !LOOM_HAVE_ORTOOLS

FlatBaselineResult MonolithicILPBaseline::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch, const Options &opts) {
  (void)kernels;
  (void)contracts;
  (void)arch;
  (void)opts;
  FlatBaselineResult result;
  result.methodName = "MonolithicILP";
  result.feasible = false;
  return result;
}

#endif // LOOM_HAVE_ORTOOLS

//===----------------------------------------------------------------------===//
// HeuristicFlatBaseline
//===----------------------------------------------------------------------===//

FlatBaselineResult HeuristicFlatBaseline::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch, const Options &opts) {

  FlatBaselineResult result;
  result.methodName = "HeuristicFlat";

  auto startTime = std::chrono::steady_clock::now();

  unsigned numCores = arch.totalCoreInstances();
  if (kernels.empty()) {
    result.feasible = true;
    result.assignment.feasible = true;
    return result;
  }
  if (numCores == 0) {
    result.feasible = false;
    return result;
  }

  // Per-core remaining FU capacity.
  std::vector<std::map<std::string, unsigned>> remainingFU(numCores);
  std::vector<uint64_t> remainingSPM(numCores);
  for (unsigned c = 0; c < numCores; ++c) {
    const auto &ct = arch.typeForInstance(c);
    remainingFU[c] = ct.fuTypeCounts;
    remainingSPM[c] = ct.spmBytes;
  }

  // Greedy: for each kernel, pick the core with the most matching FUs
  // that still has capacity.
  for (const auto &kernel : kernels) {
    int bestCore = -1;
    int bestScore = -1;

    for (unsigned c = 0; c < numCores; ++c) {
      if (!isKernelCompatible(kernel, arch.typeForInstance(c)))
        continue;

      // Check remaining capacity.
      bool hasCapacity = true;
      for (const auto &opEntry : kernel.requiredOps) {
        auto it = remainingFU[c].find(opEntry.first);
        if (it == remainingFU[c].end() || it->second < opEntry.second) {
          hasCapacity = false;
          break;
        }
      }
      if (!hasCapacity)
        continue;
      if (kernel.estimatedSPMBytes > remainingSPM[c])
        continue;

      // Score: total matching FU count.
      int score = 0;
      for (const auto &opEntry : kernel.requiredOps) {
        auto it = remainingFU[c].find(opEntry.first);
        if (it != remainingFU[c].end())
          score += static_cast<int>(it->second);
      }

      if (score > bestScore) {
        bestScore = score;
        bestCore = static_cast<int>(c);
      }
    }

    if (bestCore < 0) {
      result.feasible = false;
      auto endTime = std::chrono::steady_clock::now();
      result.solveTimeSec =
          std::chrono::duration<double>(endTime - startTime).count();
      return result;
    }

    unsigned c = static_cast<unsigned>(bestCore);
    result.assignment.kernelToCore[kernel.name] = c;

    // Consume capacity.
    for (const auto &opEntry : kernel.requiredOps) {
      auto it = remainingFU[c].find(opEntry.first);
      if (it != remainingFU[c].end())
        it->second -= std::min(it->second, opEntry.second);
    }
    remainingSPM[c] -= std::min(remainingSPM[c], kernel.estimatedSPMBytes);
  }

  result.feasible = true;
  result.assignment.feasible = true;
  result.assignment.objectiveValue =
      evaluateAssignment(result.assignment.kernelToCore, kernels, contracts,
                         arch);

  auto endTime = std::chrono::steady_clock::now();
  result.solveTimeSec =
      std::chrono::duration<double>(endTime - startTime).count();

  if (opts.verbose) {
    llvm::outs() << "Heuristic flat baseline: feasible in "
                 << result.solveTimeSec << "s, objective="
                 << result.assignment.objectiveValue << "\n";
  }

  return result;
}

//===----------------------------------------------------------------------===//
// ExhaustiveSmallInstance
//===----------------------------------------------------------------------===//

FlatBaselineResult ExhaustiveSmallInstance::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch, const Options &opts) {

  FlatBaselineResult result;
  result.methodName = "ExhaustiveSmall";

  unsigned numKernels = static_cast<unsigned>(kernels.size());
  unsigned numCores = arch.totalCoreInstances();

  // Check size limits.
  if (numKernels > opts.maxKernels ||
      arch.coreTypes.size() > opts.maxCoreTypes) {
    result.feasible = false;
    return result;
  }

  if (numKernels == 0) {
    result.feasible = true;
    result.assignment.feasible = true;
    return result;
  }

  if (numCores == 0) {
    result.feasible = false;
    return result;
  }

  auto startTime = std::chrono::steady_clock::now();

  // Enumerate all possible assignments: numCores^numKernels combinations.
  // Each kernel can go to any of numCores instances.
  std::vector<unsigned> assignment(numKernels, 0);
  double bestObjective = std::numeric_limits<double>::max();
  std::map<std::string, unsigned> bestMapping;
  bool foundFeasible = false;

  // Iterate through all assignments.
  bool done = false;
  while (!done) {
    // Build the mapping.
    std::map<std::string, unsigned> mapping;
    for (unsigned k = 0; k < numKernels; ++k)
      mapping[kernels[k].name] = assignment[k];

    // Check feasibility.
    if (isAssignmentFeasible(mapping, kernels, arch)) {
      double obj = evaluateAssignment(mapping, kernels, contracts, arch);
      if (obj < bestObjective) {
        bestObjective = obj;
        bestMapping = mapping;
        foundFeasible = true;
      }
    }

    // Increment assignment (odometer-style).
    unsigned pos = 0;
    while (pos < numKernels) {
      assignment[pos]++;
      if (assignment[pos] < numCores)
        break;
      assignment[pos] = 0;
      pos++;
    }
    if (pos >= numKernels)
      done = true;
  }

  auto endTime = std::chrono::steady_clock::now();
  result.solveTimeSec =
      std::chrono::duration<double>(endTime - startTime).count();

  if (foundFeasible) {
    result.feasible = true;
    result.assignment.feasible = true;
    result.assignment.kernelToCore = bestMapping;
    result.assignment.objectiveValue = bestObjective;
  }

  if (opts.verbose) {
    llvm::outs() << "Exhaustive enumeration: "
                 << (foundFeasible ? "feasible" : "infeasible") << " in "
                 << result.solveTimeSec << "s\n";
  }

  return result;
}

} // namespace loom
