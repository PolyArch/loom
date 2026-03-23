#include "loom/SystemCompiler/L1CoreAssignment.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <thread>

#ifdef LOOM_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#endif

namespace loom {

//===----------------------------------------------------------------------===//
// SystemArchitecture
//===----------------------------------------------------------------------===//

unsigned SystemArchitecture::totalCoreInstances() const {
  unsigned total = 0;
  for (const auto &ct : coreTypes)
    total += ct.instanceCount;
  return total;
}

const CoreTypeSpec &
SystemArchitecture::typeForInstance(unsigned instanceIdx) const {
  unsigned offset = 0;
  for (const auto &ct : coreTypes) {
    if (instanceIdx < offset + ct.instanceCount)
      return ct;
    offset += ct.instanceCount;
  }
  assert(false && "instanceIdx out of range");
  return coreTypes.back();
}

unsigned SystemArchitecture::typeIndexForInstance(unsigned instanceIdx) const {
  unsigned offset = 0;
  for (unsigned typeIdx = 0; typeIdx < coreTypes.size(); ++typeIdx) {
    if (instanceIdx < offset + coreTypes[typeIdx].instanceCount)
      return typeIdx;
    offset += coreTypes[typeIdx].instanceCount;
  }
  assert(false && "instanceIdx out of range");
  return static_cast<unsigned>(coreTypes.size()) - 1;
}

const std::string &
SystemArchitecture::typeNameForInstance(unsigned instanceIdx) const {
  return typeForInstance(instanceIdx).typeName;
}

unsigned SystemArchitecture::firstInstanceOfType(unsigned typeIdx) const {
  unsigned offset = 0;
  for (unsigned idx = 0; idx < typeIdx && idx < coreTypes.size(); ++idx)
    offset += coreTypes[idx].instanceCount;
  return offset;
}

//===----------------------------------------------------------------------===//
// KernelProfile
//===----------------------------------------------------------------------===//

unsigned KernelProfile::totalOpCount() const {
  unsigned total = 0;
  for (const auto &entry : requiredOps)
    total += entry.second;
  return total;
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

int manhattanDistance(unsigned coreA, unsigned coreB, unsigned meshCols) {
  if (meshCols == 0)
    return 0;
  int rowA = static_cast<int>(coreA / meshCols);
  int colA = static_cast<int>(coreA % meshCols);
  int rowB = static_cast<int>(coreB / meshCols);
  int colB = static_cast<int>(coreB % meshCols);
  return std::abs(rowA - rowB) + std::abs(colA - colB);
}

unsigned estimateElementSize(const std::string &dataTypeName) {
  if (dataTypeName == "f64" || dataTypeName == "i64")
    return 8;
  if (dataTypeName == "f32" || dataTypeName == "i32")
    return 4;
  if (dataTypeName == "f16" || dataTypeName == "bf16" || dataTypeName == "i16")
    return 2;
  if (dataTypeName == "i8")
    return 1;
  // Default to 4 bytes for unknown types.
  return 4;
}

bool isKernelCompatible(const KernelProfile &kernel,
                        const CoreTypeSpec &coreType) {
  for (const auto &entry : kernel.requiredOps) {
    auto it = coreType.fuTypeCounts.find(entry.first);
    if (it == coreType.fuTypeCounts.end() || it->second == 0)
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// L1 Core Assigner -- CP-SAT Formulation
//===----------------------------------------------------------------------===//

#ifdef LOOM_HAVE_ORTOOLS

using namespace operations_research::sat;

namespace {

/// Build a kernel name -> index lookup table.
std::map<std::string, unsigned>
buildKernelIndex(const std::vector<KernelProfile> &kernels) {
  std::map<std::string, unsigned> idx;
  for (unsigned k = 0; k < kernels.size(); ++k)
    idx[kernels[k].name] = k;
  return idx;
}

/// Extract the assignment result from a solved CP-SAT model.
AssignmentResult
extractAssignment(const CpSolverResponse &response,
                  const std::vector<std::vector<BoolVar>> &x,
                  const std::vector<KernelProfile> &kernels,
                  const SystemArchitecture &arch) {
  AssignmentResult result;
  result.feasible = true;

  unsigned numCores = arch.totalCoreInstances();

  // Extract kernel -> core mapping.
  for (unsigned k = 0; k < kernels.size(); ++k) {
    for (unsigned c = 0; c < numCores; ++c) {
      if (SolutionBooleanValue(response, x[k][c])) {
        result.kernelToCore[kernels[k].name] = c;
        break;
      }
    }
  }

  // Build per-core assignment details.
  result.coreAssignments.resize(numCores);
  for (unsigned c = 0; c < numCores; ++c) {
    result.coreAssignments[c].coreInstanceIdx = c;
    result.coreAssignments[c].coreTypeName =
        arch.typeNameForInstance(c);
  }

  for (const auto &entry : result.kernelToCore) {
    result.coreAssignments[entry.second].assignedKernels.push_back(
        entry.first);
  }

  // Estimate per-core utilization.
  for (unsigned c = 0; c < numCores; ++c) {
    const auto &coreType = arch.typeForInstance(c);
    if (coreType.numFUs == 0)
      continue;
    unsigned totalOps = 0;
    for (const auto &kName : result.coreAssignments[c].assignedKernels) {
      auto it = std::find_if(kernels.begin(), kernels.end(),
                             [&](const KernelProfile &kp) {
                               return kp.name == kName;
                             });
      if (it != kernels.end())
        totalOps += it->totalOpCount();
    }
    result.coreAssignments[c].estimatedUtilization =
        static_cast<double>(totalOps) / coreType.numFUs;
  }

  result.objectiveValue =
      static_cast<double>(response.objective_value()) / 1000.0;
  return result;
}

} // namespace

AssignmentResult L1CoreAssigner::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch,
    const std::vector<InfeasibilityCut> &cuts,
    const L1AssignerOptions &opts) {

  if (kernels.empty()) {
    AssignmentResult result;
    result.feasible = true;
    return result;
  }

  unsigned numKernels = static_cast<unsigned>(kernels.size());
  unsigned numCores = arch.totalCoreInstances();

  if (numCores == 0) {
    AssignmentResult result;
    result.feasible = false;
    return result;
  }

  auto kernelIdx = buildKernelIndex(kernels);

  CpModelBuilder model;

  // --- Decision variables ---
  // x[k][c] = 1 if kernel k is assigned to core instance c.
  std::vector<std::vector<BoolVar>> x(numKernels);
  for (unsigned k = 0; k < numKernels; ++k) {
    x[k].resize(numCores);
    for (unsigned c = 0; c < numCores; ++c) {
      x[k][c] = model.NewBoolVar().WithName(
          "x_k" + std::to_string(k) + "_c" + std::to_string(c));
    }
  }

  // --- Constraint 1: Each kernel assigned to exactly one core ---
  for (unsigned k = 0; k < numKernels; ++k) {
    model.AddExactlyOne(x[k]);
  }

  // --- Constraint 5: Type compatibility ---
  // Must come before capacity constraints to prune incompatible assignments.
  for (unsigned k = 0; k < numKernels; ++k) {
    for (unsigned c = 0; c < numCores; ++c) {
      if (!isKernelCompatible(kernels[k], arch.typeForInstance(c))) {
        model.FixVariable(x[k][c], false);
      }
    }
  }

  // --- Constraint 3: Benders infeasibility cuts ---
  // "Kernel K cannot be assigned to any core of type T".
  for (const auto &cut : cuts) {
    auto kIt = kernelIdx.find(cut.kernelName);
    if (kIt == kernelIdx.end())
      continue;
    unsigned k = kIt->second;
    for (unsigned typeIdx = 0; typeIdx < arch.coreTypes.size(); ++typeIdx) {
      if (arch.coreTypes[typeIdx].typeName != cut.coreType)
        continue;
      unsigned base = arch.firstInstanceOfType(typeIdx);
      for (unsigned inst = 0; inst < arch.coreTypes[typeIdx].instanceCount;
           ++inst) {
        model.FixVariable(x[k][base + inst], false);
      }
    }
  }

  // --- Constraint 2: Core capacity ---
  // For each core instance, sum of kernel resource demands <= capacity.

  // FU type capacity constraints.
  for (unsigned c = 0; c < numCores; ++c) {
    const auto &coreType = arch.typeForInstance(c);

    for (const auto &fuEntry : coreType.fuTypeCounts) {
      const std::string &opType = fuEntry.first;
      unsigned maxCount = fuEntry.second;

      LinearExpr opDemand;
      bool hasDemand = false;
      for (unsigned k = 0; k < numKernels; ++k) {
        auto opIt = kernels[k].requiredOps.find(opType);
        if (opIt != kernels[k].requiredOps.end() && opIt->second > 0) {
          opDemand += x[k][c] * static_cast<int64_t>(opIt->second);
          hasDemand = true;
        }
      }
      if (hasDemand) {
        model.AddLessOrEqual(opDemand, static_cast<int64_t>(maxCount));
      }
    }

    // SPM capacity constraint.
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
      if (hasSpmDemand) {
        model.AddLessOrEqual(spmDemand,
                             static_cast<int64_t>(coreType.spmBytes));
      }
    }
  }

  // --- Constraint 4: Load balancing ---
  // Use integer-scaled utilization (0..1000) for load balance tracking.
  // Only apply to cores that can have kernels assigned.
  if (opts.loadBalanceWeight > 0.0 && numCores > 1) {
    // Compute per-core capacity (total FU count as proxy).
    std::vector<int64_t> coreCapacity(numCores);
    for (unsigned c = 0; c < numCores; ++c) {
      const auto &ct = arch.typeForInstance(c);
      coreCapacity[c] = std::max<int64_t>(1, ct.numFUs);
    }

    // Per-core total ops (integer).
    std::vector<IntVar> coreOps(numCores);
    for (unsigned c = 0; c < numCores; ++c) {
      LinearExpr totalOps;
      for (unsigned k = 0; k < numKernels; ++k) {
        totalOps +=
            x[k][c] * static_cast<int64_t>(kernels[k].totalOpCount());
      }
      int64_t maxOpsOnCore = 0;
      for (unsigned k = 0; k < numKernels; ++k)
        maxOpsOnCore += kernels[k].totalOpCount();
      coreOps[c] = model.NewIntVar(operations_research::Domain(0, maxOpsOnCore))
                       .WithName("ops_c" + std::to_string(c));
      model.AddEquality(coreOps[c], totalOps);
    }

    // Max ops across all cores.
    IntVar maxOps =
        model.NewIntVar(operations_research::Domain(0, static_cast<int64_t>(numKernels) * 10000))
            .WithName("max_ops");
    model.AddMaxEquality(maxOps, std::vector<IntVar>(coreOps.begin(), coreOps.end()));

    // Min ops across all cores.
    IntVar minOps =
        model.NewIntVar(operations_research::Domain(0, static_cast<int64_t>(numKernels) * 10000))
            .WithName("min_ops");
    model.AddMinEquality(minOps, std::vector<IntVar>(coreOps.begin(), coreOps.end()));

    // Compute the threshold as an absolute ops difference.
    // threshold * max_capacity approximates the allowed gap.
    int64_t maxCap = 0;
    for (unsigned c = 0; c < numCores; ++c)
      maxCap = std::max(maxCap, coreCapacity[c]);
    int64_t absThreshold = static_cast<int64_t>(
        std::ceil(opts.loadBalanceThreshold * static_cast<double>(maxCap)));
    absThreshold = std::max<int64_t>(absThreshold, 1);

    model.AddLessOrEqual(LinearExpr(maxOps) - LinearExpr(minOps),
                         absThreshold);
  }

  // --- Objective function ---
  // Scale everything to integer (x1000) for CP-SAT.
  constexpr int64_t kScale = 1000;
  LinearExpr objective;

  // Component 1: Critical path latency approximation.
  // Sum of estimated kernel compute cycles weighted by assignment.
  int64_t latencyScaled =
      static_cast<int64_t>(std::llround(opts.latencyWeight * kScale));
  if (latencyScaled > 0) {
    for (unsigned k = 0; k < numKernels; ++k) {
      int64_t kernelCost =
          static_cast<int64_t>(std::llround(kernels[k].estimatedComputeCycles));
      if (kernelCost <= 0)
        kernelCost = static_cast<int64_t>(kernels[k].totalOpCount());
      // The latency is incurred regardless of which core, but weight by
      // core type speed ratio (simplified: all cores run at same speed).
      for (unsigned c = 0; c < numCores; ++c) {
        objective += x[k][c] * (latencyScaled * kernelCost / kScale);
      }
    }
  }

  // Component 2: NoC transfer cost.
  // For each contract edge, if producer and consumer are on different cores,
  // add hop_distance * data_volume cost.
  int64_t nocScaled =
      static_cast<int64_t>(std::llround(opts.nocCostWeight * kScale));
  if (nocScaled > 0 && !contracts.empty()) {
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

      // Linearize: for each pair of cores (cp, cc) where cp != cc,
      // create auxiliary bool var = x[pk][cp] AND x[ck][cc].
      for (unsigned cp = 0; cp < numCores; ++cp) {
        for (unsigned cc = 0; cc < numCores; ++cc) {
          if (cp == cc)
            continue;

          int dist = manhattanDistance(cp, cc, arch.nocSpec.meshCols);
          if (dist == 0)
            continue;

          int64_t pairCost = nocScaled * dist * volume / kScale;
          if (pairCost <= 0)
            continue;

          // Reify: both = (x[pk][cp] AND x[ck][cc]).
          BoolVar both = model.NewBoolVar().WithName(
              "noc_p" + std::to_string(pk) + "_c" + std::to_string(ck) +
              "_cp" + std::to_string(cp) + "_cc" + std::to_string(cc));

          // both => x[pk][cp] and both => x[ck][cc]
          model.AddImplication(both, x[pk][cp]);
          model.AddImplication(both, x[ck][cc]);
          // x[pk][cp] AND x[ck][cc] => both
          model.AddBoolOr({both, x[pk][cp].Not(), x[ck][cc].Not()});

          objective += both * pairCost;
        }
      }
    }
  }

  // Component 3: Data locality bonus (subtracted from objective).
  if (opts.enableDataLocality && !contracts.empty()) {
    int64_t localityScaled =
        static_cast<int64_t>(std::llround(opts.loadBalanceWeight * kScale));
    if (localityScaled > 0) {
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

        int64_t bonus = localityScaled * volume / kScale;
        if (bonus <= 0)
          bonus = 1;

        // Reward co-location: subtract bonus when both on same core.
        for (unsigned c = 0; c < numCores; ++c) {
          BoolVar colocated = model.NewBoolVar().WithName(
              "coloc_" + std::to_string(pk) + "_" + std::to_string(ck) +
              "_c" + std::to_string(c));
          model.AddImplication(colocated, x[pk][c]);
          model.AddImplication(colocated, x[ck][c]);
          model.AddBoolOr({colocated, x[pk][c].Not(), x[ck][c].Not()});
          objective -= colocated * bonus;
        }
      }
    }
  }

  model.Minimize(objective);

  // --- Solve ---
  Model satModel;
  SatParameters params;
  params.set_max_time_in_seconds(
      static_cast<double>(opts.maxSolverTimeSec));

  unsigned numWorkers = opts.numWorkers;
  if (numWorkers == 0) {
    numWorkers = std::thread::hardware_concurrency();
    if (numWorkers == 0)
      numWorkers = 4;
    numWorkers = std::min(numWorkers, 8u);
  }
  params.set_num_search_workers(static_cast<int>(numWorkers));
  satModel.Add(NewSatParameters(params));

  if (opts.verbose) {
    llvm::outs() << "L1 core assignment: " << numKernels << " kernels, "
                 << numCores << " core instances, "
                 << cuts.size() << " Benders cuts\n";
  }

  auto startTime = std::chrono::steady_clock::now();
  const CpSolverResponse response = SolveCpModel(model.Build(), &satModel);
  auto endTime = std::chrono::steady_clock::now();

  double solveTimeSec =
      std::chrono::duration<double>(endTime - startTime).count();

  if (opts.verbose) {
    llvm::outs() << "L1 solver finished in " << solveTimeSec
                 << "s, status=" << static_cast<int>(response.status())
                 << "\n";
  }

  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    AssignmentResult result;
    result.feasible = false;
    return result;
  }

  return extractAssignment(response, x, kernels, arch);
}

#else // !LOOM_HAVE_ORTOOLS

AssignmentResult L1CoreAssigner::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch,
    const std::vector<InfeasibilityCut> &cuts,
    const L1AssignerOptions &opts) {
  (void)contracts;
  (void)cuts;

  // Fallback: round-robin assignment when OR-Tools is not available.
  AssignmentResult result;
  unsigned numCores = arch.totalCoreInstances();
  if (numCores == 0 || kernels.empty()) {
    result.feasible = kernels.empty();
    return result;
  }

  result.feasible = true;
  result.coreAssignments.resize(numCores);
  for (unsigned c = 0; c < numCores; ++c) {
    result.coreAssignments[c].coreInstanceIdx = c;
    result.coreAssignments[c].coreTypeName = arch.typeNameForInstance(c);
  }

  for (unsigned k = 0; k < kernels.size(); ++k) {
    // Find a compatible core, round-robin among compatible ones.
    bool assigned = false;
    for (unsigned c = 0; c < numCores; ++c) {
      unsigned coreIdx = (k + c) % numCores;
      if (isKernelCompatible(kernels[k], arch.typeForInstance(coreIdx))) {
        result.kernelToCore[kernels[k].name] = coreIdx;
        result.coreAssignments[coreIdx].assignedKernels.push_back(
            kernels[k].name);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      result.feasible = false;
      return result;
    }
  }

  if (opts.verbose) {
    llvm::outs() << "L1 core assignment (fallback round-robin): "
                 << kernels.size() << " kernels assigned to " << numCores
                 << " cores\n";
  }

  return result;
}

#endif // LOOM_HAVE_ORTOOLS

} // namespace loom
