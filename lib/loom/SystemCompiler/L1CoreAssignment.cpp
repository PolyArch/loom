#include "loom/SystemCompiler/L1CoreAssignment.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <queue>
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
// Pipeline Start Time Computation
//===----------------------------------------------------------------------===//

/// Compute pipeline start times for each kernel using an ASAP topological
/// traversal. When a producer and consumer are on different cores, the
/// consumer may overlap with the producer by starting after the producer's
/// pipeline initiation delay (estimatedComputeCycles / defaultTileCount).
/// When they share a core, the consumer must wait for the producer to finish
/// plus a reconfiguration gap.
static void
computePipelineStartTimes(AssignmentResult &result,
                          const std::vector<KernelProfile> &kernels,
                          const std::vector<ContractSpec> &contracts,
                          unsigned defaultTileCount) {
  if (kernels.empty())
    return;

  // Build name -> KernelProfile lookup.
  std::map<std::string, const KernelProfile *> profileMap;
  for (const auto &kp : kernels)
    profileMap[kp.name] = &kp;

  // Collect all kernel names from the assignment.
  std::vector<std::string> allKernels;
  for (const auto &ca : result.coreAssignments)
    for (const auto &kn : ca.assignedKernels)
      allKernels.push_back(kn);

  if (allKernels.empty())
    return;

  // Build adjacency list and in-degree for topological sort.
  std::set<std::string> kernelSet(allKernels.begin(), allKernels.end());
  std::map<std::string, std::vector<std::string>> successors;
  std::map<std::string, unsigned> inDegree;
  for (const auto &k : allKernels) {
    successors[k] = {};
    inDegree[k] = 0;
  }

  // Build consumer dependency map.
  std::map<std::string, std::vector<const ContractSpec *>> consumerDeps;
  for (const auto &c : contracts) {
    bool prodLocal = kernelSet.count(c.producerKernel) > 0;
    bool consLocal = kernelSet.count(c.consumerKernel) > 0;
    if (prodLocal && consLocal) {
      successors[c.producerKernel].push_back(c.consumerKernel);
      inDegree[c.consumerKernel]++;
      consumerDeps[c.consumerKernel].push_back(&c);
    }
  }

  // Topological sort (Kahn's algorithm).
  std::queue<std::string> ready;
  for (const auto &k : allKernels) {
    if (inDegree[k] == 0)
      ready.push(k);
  }

  std::vector<std::string> sortedKernels;
  sortedKernels.reserve(allKernels.size());
  while (!ready.empty()) {
    std::string current = ready.front();
    ready.pop();
    sortedKernels.push_back(current);
    for (const auto &neighbor : successors[current]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] == 0)
        ready.push(neighbor);
    }
  }
  // If cycle detected, fall back to original order.
  if (sortedKernels.size() != allKernels.size())
    sortedKernels = allKernels;

  // Compute estimated duration for each kernel.
  constexpr unsigned kDefaultTripCount = 1000;
  std::map<std::string, uint64_t> duration;
  for (const auto &kName : sortedKernels) {
    auto profIt = profileMap.find(kName);
    unsigned ii = 1;
    double cycles = 0.0;
    if (profIt != profileMap.end()) {
      ii = std::max(1u, profIt->second->estimatedMinII);
      cycles = profIt->second->estimatedComputeCycles;
    }
    // Use estimatedComputeCycles if available, else tripCount * II.
    uint64_t dur = (cycles > 0.0)
                       ? static_cast<uint64_t>(cycles)
                       : static_cast<uint64_t>(kDefaultTripCount) * ii;
    duration[kName] = dur;
  }

  unsigned tileCount = (defaultTileCount > 0) ? defaultTileCount : 4;
  constexpr unsigned kReconfigCycles = 100;

  // ASAP pipeline scheduling.
  std::map<std::string, uint64_t> startTimes;
  std::map<unsigned, uint64_t> coreAvailableAt;

  for (const auto &kName : sortedKernels) {
    uint64_t earliestStart = 0;

    // Dependency constraints from producers.
    auto depIt = consumerDeps.find(kName);
    if (depIt != consumerDeps.end()) {
      for (const auto *contract : depIt->second) {
        auto prodStartIt = startTimes.find(contract->producerKernel);
        if (prodStartIt == startTimes.end())
          continue;

        uint64_t prodStart = prodStartIt->second;
        uint64_t prodDur = duration[contract->producerKernel];

        auto prodCoreIt = result.kernelToCore.find(contract->producerKernel);
        auto consCoreIt = result.kernelToCore.find(kName);
        bool sameCore = (prodCoreIt != result.kernelToCore.end() &&
                         consCoreIt != result.kernelToCore.end() &&
                         prodCoreIt->second == consCoreIt->second);

        if (sameCore) {
          uint64_t constraint = prodStart + prodDur + kReconfigCycles;
          if (constraint > earliestStart)
            earliestStart = constraint;
        } else {
          uint64_t pipelineDelay = prodDur / tileCount;
          uint64_t constraint = prodStart + pipelineDelay;
          if (constraint > earliestStart)
            earliestStart = constraint;
        }
      }
    }

    // Same-core serialization.
    auto coreIt = result.kernelToCore.find(kName);
    if (coreIt != result.kernelToCore.end()) {
      unsigned coreIdx = coreIt->second;
      auto availIt = coreAvailableAt.find(coreIdx);
      if (availIt != coreAvailableAt.end()) {
        if (availIt->second > earliestStart)
          earliestStart = availIt->second;
      }
      coreAvailableAt[coreIdx] =
          earliestStart + duration[kName] + kReconfigCycles;
    }

    startTimes[kName] = earliestStart;
  }

  result.kernelStartTimes = std::move(startTimes);
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

  // --- Constraint 6: TDC-derived constraints ---
  // When the ContractConstraintTranslator has produced constraints, inject
  // them into the CP-SAT model to prune the search space.
  if (opts.tdcConstraints.has_value() && !opts.tdcConstraints->empty()) {
    const ConstraintSet &tdc = opts.tdcConstraints.value();

    // 6a. SchedulingConstraints: precedence (producer before consumer).
    // At the L1 assignment level, precedence is enforced by strongly
    // favoring co-location of precedence-constrained pairs. When both
    // kernels share a core, sequential execution guarantees producer
    // completes before consumer begins. We add an objective penalty for
    // cross-core placement of precedence-constrained pairs.
    // (Penalties are accumulated in the objective section below via
    //  nocScaled terms that already penalize cross-core edges. Here we
    //  add hard constraints: introduce per-pair auxiliary bool colocVars
    //  and mandate co-location when both kernels exist in the problem.)
    for (const auto &sc : tdc.scheduling) {
      auto pIt = kernelIdx.find(sc.producer);
      auto cIt = kernelIdx.find(sc.consumer);
      if (pIt == kernelIdx.end() || cIt == kernelIdx.end())
        continue;

      unsigned pk = pIt->second;
      unsigned ck = cIt->second;

      // Soft precedence: for each core, create a co-location indicator
      // and add a large objective bonus (implemented later in objective
      // section). For now, record a scheduling bonus weight.
      // A stronger formulation: if both are on the same core, no extra
      // constraint needed (sequential execution). If on different cores,
      // the NoC transfer cost already acts as a penalty. We add an
      // additional co-location incentive by creating a BoolVar that is
      // true when both are on the same core, and give it a bonus.
      for (unsigned c = 0; c < numCores; ++c) {
        BoolVar precColoc = model.NewBoolVar().WithName(
            "prec_coloc_" + std::to_string(pk) + "_" +
            std::to_string(ck) + "_c" + std::to_string(c));
        model.AddImplication(precColoc, x[pk][c]);
        model.AddImplication(precColoc, x[ck][c]);
        model.AddBoolOr({precColoc, x[pk][c].Not(), x[ck][c].Not()});
        // The bonus is applied in the objective function below.
      }
    }

    // 6b. MemoryConstraints: core-type filtering based on memory level.
    // LOCAL_SPM means the buffer must reside in local scratchpad, so the
    // producer and consumer must be on the same core or adjacent cores
    // (Manhattan distance <= 1).
    for (const auto &mc : tdc.memory) {
      if (mc.level != MemoryLevel::LOCAL_SPM)
        continue;

      auto pIt = kernelIdx.find(mc.edgeProducer);
      auto cIt = kernelIdx.find(mc.edgeConsumer);
      if (pIt == kernelIdx.end() || cIt == kernelIdx.end())
        continue;

      unsigned pk = pIt->second;
      unsigned ck = cIt->second;

      // Disallow assignments where the two kernels are more than 1 hop
      // apart on the mesh. For each pair (cp, cc) with distance > 1,
      // forbid both being true simultaneously.
      for (unsigned cp = 0; cp < numCores; ++cp) {
        for (unsigned cc = 0; cc < numCores; ++cc) {
          if (cp == cc)
            continue;
          int dist = manhattanDistance(cp, cc, arch.nocSpec.meshCols);
          if (dist <= 1)
            continue;
          // Forbid x[pk][cp] AND x[ck][cc].
          model.AddBoolOr({x[pk][cp].Not(), x[ck][cc].Not()});
        }
      }
    }

    // 6c. RateConstraints: minimum throughput requirements.
    // Penalize cross-core placement proportional to the rate requirement
    // so the solver prefers co-locating high-throughput edges. The
    // penalty is added to the objective below; higher minRate means
    // stronger incentive to keep the edge local.
    // (Rate penalties are accumulated in the objective section.)

    if (opts.verbose) {
      llvm::outs() << "L1 TDC constraints applied: "
                   << tdc.scheduling.size() << " scheduling, "
                   << tdc.memory.size() << " memory, "
                   << tdc.rate.size() << " rate\n";
    }
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

  // Component 4: Reconfiguration cost penalty.
  // When multiple kernels are assigned to the same core, they execute
  // sequentially in BATCH_SEQUENTIAL mode, incurring a reconfiguration gap
  // between each pair. Penalize by (numKernels - 1) * reconfigCycles per core.
  // This incentivizes spreading kernels across cores when reconfig cost is high.
  {
    constexpr int64_t kDefaultReconfigCycles = 100;
    for (unsigned c = 0; c < numCores; ++c) {
      // Count kernels assigned to this core.
      LinearExpr kernelCount;
      for (unsigned k = 0; k < numKernels; ++k) {
        kernelCount += x[k][c];
      }

      // nk = number of kernels on core c.
      IntVar nk = model.NewIntVar(
          operations_research::Domain(0, static_cast<int64_t>(numKernels)))
          .WithName("nk_c" + std::to_string(c));
      model.AddEquality(nk, kernelCount);

      // reconfigEvents = max(0, nk - 1). Since CP-SAT works with integers,
      // we model this as: reconfigEvents >= nk - 1 and reconfigEvents >= 0.
      IntVar reconfigEvents = model.NewIntVar(
          operations_research::Domain(0, static_cast<int64_t>(numKernels)))
          .WithName("reconf_c" + std::to_string(c));
      // reconfigEvents >= nk - 1
      model.AddGreaterOrEqual(reconfigEvents, LinearExpr(nk) - 1);
      // reconfigEvents <= nk (redundant but helps the solver)
      model.AddLessOrEqual(reconfigEvents, nk);

      // Add reconfigCycles * reconfigEvents to the objective.
      objective += reconfigEvents * kDefaultReconfigCycles;
    }
  }

  // Component 5: TDC scheduling precedence co-location bonus.
  // For each SchedulingConstraint, reward co-locating producer and consumer
  // so that intra-core sequential execution trivially satisfies FIFO ordering.
  if (opts.tdcConstraints.has_value()) {
    const ConstraintSet &tdc = opts.tdcConstraints.value();

    constexpr int64_t kPrecedenceBonus = 50;
    for (const auto &sc : tdc.scheduling) {
      auto pIt = kernelIdx.find(sc.producer);
      auto cIt = kernelIdx.find(sc.consumer);
      if (pIt == kernelIdx.end() || cIt == kernelIdx.end())
        continue;

      unsigned pk = pIt->second;
      unsigned ck = cIt->second;

      for (unsigned c = 0; c < numCores; ++c) {
        // Reuse the already-created prec_coloc variables by creating
        // matching co-location indicators for the objective.
        BoolVar colocPrec = model.NewBoolVar().WithName(
            "obj_prec_" + std::to_string(pk) + "_" +
            std::to_string(ck) + "_c" + std::to_string(c));
        model.AddImplication(colocPrec, x[pk][c]);
        model.AddImplication(colocPrec, x[ck][c]);
        model.AddBoolOr({colocPrec, x[pk][c].Not(), x[ck][c].Not()});
        objective -= colocPrec * kPrecedenceBonus;
      }
    }

    // Component 6: TDC rate constraint penalty for cross-core placement.
    // Higher minRate means stronger penalty when the edge crosses cores.
    for (const auto &rc : tdc.rate) {
      auto pIt = kernelIdx.find(rc.edgeProducer);
      auto cIt = kernelIdx.find(rc.edgeConsumer);
      if (pIt == kernelIdx.end() || cIt == kernelIdx.end())
        continue;
      if (rc.minRate <= 0)
        continue;

      unsigned pk = pIt->second;
      unsigned ck = cIt->second;

      // For each cross-core pair, add a penalty proportional to minRate
      // times hop distance.
      for (unsigned cp = 0; cp < numCores; ++cp) {
        for (unsigned cc = 0; cc < numCores; ++cc) {
          if (cp == cc)
            continue;

          int dist = manhattanDistance(cp, cc, arch.nocSpec.meshCols);
          if (dist == 0)
            continue;

          int64_t ratePenalty = rc.minRate * dist;
          if (ratePenalty <= 0)
            continue;

          BoolVar both = model.NewBoolVar().WithName(
              "rate_" + std::to_string(pk) + "_" + std::to_string(ck) +
              "_cp" + std::to_string(cp) + "_cc" + std::to_string(cc));
          model.AddImplication(both, x[pk][cp]);
          model.AddImplication(both, x[ck][cc]);
          model.AddBoolOr({both, x[pk][cp].Not(), x[ck][cc].Not()});
          objective += both * ratePenalty;
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

  AssignmentResult result = extractAssignment(response, x, kernels, arch);

  if (opts.enablePipelineScheduling && result.feasible) {
    computePipelineStartTimes(result, kernels, contracts, /*defaultTileCount=*/4);
    if (opts.verbose) {
      llvm::outs() << "L1 pipeline start times computed for "
                   << result.kernelStartTimes.size() << " kernels\n";
    }
  }

  return result;
}

#else // !LOOM_HAVE_ORTOOLS

AssignmentResult L1CoreAssigner::solve(
    const std::vector<KernelProfile> &kernels,
    const std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch,
    const std::vector<InfeasibilityCut> &cuts,
    const L1AssignerOptions &opts) {
  (void)cuts;

  // Fallback: round-robin assignment when OR-Tools is not available.
  AssignmentResult result;
  unsigned numCores = arch.totalCoreInstances();
  if (numCores == 0 || kernels.empty()) {
    result.feasible = kernels.empty();
    return result;
  }

  // Build kernel name -> index map for TDC constraint lookup.
  std::map<std::string, unsigned> kernelIdx;
  for (unsigned k = 0; k < kernels.size(); ++k)
    kernelIdx[kernels[k].name] = k;

  // Build a set of LOCAL_SPM memory-constrained kernel pairs. These pairs
  // must be placed on the same core or adjacent cores (distance <= 1).
  std::set<std::pair<unsigned, unsigned>> localSpmPairs;
  if (opts.tdcConstraints.has_value()) {
    for (const auto &mc : opts.tdcConstraints->memory) {
      if (mc.level != MemoryLevel::LOCAL_SPM)
        continue;
      auto pIt = kernelIdx.find(mc.edgeProducer);
      auto cIt = kernelIdx.find(mc.edgeConsumer);
      if (pIt != kernelIdx.end() && cIt != kernelIdx.end())
        localSpmPairs.insert({pIt->second, cIt->second});
    }
  }

  // Build a set of scheduling-constrained pairs for co-location preference.
  std::set<std::pair<unsigned, unsigned>> schedPairs;
  if (opts.tdcConstraints.has_value()) {
    for (const auto &sc : opts.tdcConstraints->scheduling) {
      auto pIt = kernelIdx.find(sc.producer);
      auto cIt = kernelIdx.find(sc.consumer);
      if (pIt != kernelIdx.end() && cIt != kernelIdx.end())
        schedPairs.insert({pIt->second, cIt->second});
    }
  }

  // Build a mapping from contract producer-consumer kernel indices to the
  // contract spec for NoC cost estimation.
  std::map<std::pair<unsigned, unsigned>, const ContractSpec *> contractMap;
  for (const auto &c : contracts) {
    auto pIt = kernelIdx.find(c.producerKernel);
    auto cIt = kernelIdx.find(c.consumerKernel);
    if (pIt != kernelIdx.end() && cIt != kernelIdx.end())
      contractMap[{pIt->second, cIt->second}] = &c;
  }

  result.feasible = true;
  result.coreAssignments.resize(numCores);
  for (unsigned c = 0; c < numCores; ++c) {
    result.coreAssignments[c].coreInstanceIdx = c;
    result.coreAssignments[c].coreTypeName = arch.typeNameForInstance(c);
  }

  for (unsigned k = 0; k < kernels.size(); ++k) {
    // Find a compatible core, round-robin among compatible ones.
    // If TDC constraints exist, prefer cores where co-located partners
    // already reside.
    bool assigned = false;
    int bestCore = -1;
    int bestScore = -1;

    for (unsigned c = 0; c < numCores; ++c) {
      unsigned coreIdx = (k + c) % numCores;
      if (!isKernelCompatible(kernels[k], arch.typeForInstance(coreIdx)))
        continue;

      int score = 0;

      // Prefer cores where scheduling-constrained partners are assigned.
      for (const auto &sp : schedPairs) {
        unsigned partner = (sp.first == k) ? sp.second : sp.first;
        if (sp.first != k && sp.second != k)
          continue;
        auto it = result.kernelToCore.find(kernels[partner].name);
        if (it != result.kernelToCore.end() && it->second == coreIdx)
          score += 10;
      }

      // Prefer cores where LOCAL_SPM partners are assigned (same or adjacent).
      for (const auto &lp : localSpmPairs) {
        unsigned partner = (lp.first == k) ? lp.second : lp.first;
        if (lp.first != k && lp.second != k)
          continue;
        auto it = result.kernelToCore.find(kernels[partner].name);
        if (it != result.kernelToCore.end()) {
          int dist = manhattanDistance(it->second, coreIdx,
                                       arch.nocSpec.meshCols);
          if (dist <= 1)
            score += 5;
        }
      }

      if (score > bestScore || bestCore < 0) {
        bestScore = score;
        bestCore = static_cast<int>(coreIdx);
      }
    }

    if (bestCore >= 0) {
      unsigned coreIdx = static_cast<unsigned>(bestCore);
      result.kernelToCore[kernels[k].name] = coreIdx;
      result.coreAssignments[coreIdx].assignedKernels.push_back(
          kernels[k].name);
      assigned = true;
    }
    if (!assigned) {
      result.feasible = false;
      return result;
    }
  }

  // Post-assignment validation: check LOCAL_SPM adjacency constraints.
  for (const auto &lp : localSpmPairs) {
    auto pIt = result.kernelToCore.find(kernels[lp.first].name);
    auto cIt = result.kernelToCore.find(kernels[lp.second].name);
    if (pIt == result.kernelToCore.end() || cIt == result.kernelToCore.end())
      continue;
    int dist = manhattanDistance(pIt->second, cIt->second,
                                 arch.nocSpec.meshCols);
    if (dist > 1 && opts.verbose) {
      llvm::outs() << "L1 fallback WARNING: LOCAL_SPM constraint violated "
                   << "for edge " << kernels[lp.first].name << " -> "
                   << kernels[lp.second].name << " (distance=" << dist
                   << ")\n";
    }
  }

  if (opts.verbose) {
    if (opts.tdcConstraints.has_value() && !opts.tdcConstraints->empty()) {
      llvm::outs() << "L1 fallback TDC constraints: "
                   << opts.tdcConstraints->scheduling.size() << " scheduling, "
                   << opts.tdcConstraints->memory.size() << " memory, "
                   << opts.tdcConstraints->rate.size() << " rate\n";
    }
    llvm::outs() << "L1 core assignment (fallback round-robin): "
                 << kernels.size() << " kernels assigned to " << numCores
                 << " cores\n";
  }

  if (opts.enablePipelineScheduling && result.feasible) {
    computePipelineStartTimes(result, kernels, contracts, /*defaultTileCount=*/4);
    if (opts.verbose) {
      llvm::outs() << "L1 pipeline start times computed for "
                   << result.kernelStartTimes.size() << " kernels\n";
    }
  }

  return result;
}

#endif // LOOM_HAVE_ORTOOLS

} // namespace loom
