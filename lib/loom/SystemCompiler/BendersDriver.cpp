//===-- BendersDriver.cpp - Multi-core Benders decomposition driver *- C++ -*-===//
//
// Implements the Benders decomposition loop for multi-core CGRA compilation.
// Master problem assigns kernels to core types; sub-problems map each kernel
// using the Loom mapper.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/ADG/ADGVerifier.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/DFGBuilder.h"
#include "loom/Mapper/Mapper.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace loom::tapestry;

BendersDriver::BendersDriver(const SystemArchitecture &arch,
                             std::vector<KernelDesc> kernels,
                             std::vector<ContractSpec> contracts,
                             mlir::MLIRContext &ctx)
    : arch_(arch), kernels_(std::move(kernels)),
      contracts_(std::move(contracts)), ctx_(ctx) {}

//===----------------------------------------------------------------------===//
// Master problem: assign kernels to core types
//===----------------------------------------------------------------------===//

std::vector<int> BendersDriver::solveMasterProblem(unsigned iteration) {
  const unsigned numKernels = kernels_.size();
  const unsigned numCoreTypes = arch_.coreTypes.size();

  if (numCoreTypes == 0 || numKernels == 0)
    return {};

  // Assignment vector: kernelIndex -> coreTypeIndex
  std::vector<int> assignment(numKernels, 0);

  if (iteration == 0) {
    // Initial assignment: distribute kernels round-robin across core types,
    // weighted by resource compatibility.
    for (unsigned k = 0; k < numKernels; ++k) {
      int bestType = 0;
      double bestScore = -1e9;

      for (unsigned t = 0; t < numCoreTypes; ++t) {
        const auto &coreType = arch_.coreTypes[t];
        double score = 0.0;

        // Capacity score: prefer cores with enough PEs
        if (coreType.totalPEs >= kernels_[k].requiredPEs)
          score += 10.0;
        else
          score -= 5.0 * (kernels_[k].requiredPEs - coreType.totalPEs);

        // FU count score
        if (coreType.totalFUs >= kernels_[k].requiredFUs)
          score += 5.0;

        // Avoid overloading any single core type (load balancing)
        unsigned currentLoad = 0;
        for (unsigned j = 0; j < k; ++j) {
          if (assignment[j] == static_cast<int>(t))
            ++currentLoad;
        }
        if (currentLoad >= coreType.numInstances)
          score -= 20.0;

        if (score > bestScore) {
          bestScore = score;
          bestType = static_cast<int>(t);
        }
      }
      assignment[k] = bestType;
    }
  } else {
    // Use cuts from previous iterations to guide assignment.
    // Start from the previous assignment and apply penalty-based reassignment.
    // Re-derive from the last iteration's cuts.
    for (unsigned k = 0; k < numKernels; ++k) {
      int bestType = 0;
      double bestScore = -1e9;

      for (unsigned t = 0; t < numCoreTypes; ++t) {
        double score = 0.0;

        // Base capacity score
        const auto &coreType = arch_.coreTypes[t];
        if (coreType.totalPEs >= kernels_[k].requiredPEs)
          score += 10.0;

        // Apply Benders cuts: penalize (kernel, coreType) pairs that failed
        for (const auto &cut : cuts_) {
          if (cut.kernelName == kernels_[k].name &&
              cut.coreTypeIndex == static_cast<int>(t)) {
            score -= cut.penalty;
          }
        }

        // Communication affinity: prefer co-locating connected kernels
        for (const auto &contract : contracts_) {
          if (contract.producerKernel == kernels_[k].name ||
              contract.consumerKernel == kernels_[k].name) {
            // Find the other kernel's assignment
            for (unsigned j = 0; j < k; ++j) {
              bool isPartner =
                  (contract.producerKernel == kernels_[j].name ||
                   contract.consumerKernel == kernels_[j].name);
              if (isPartner && assignment[j] == static_cast<int>(t))
                score += 3.0; // Bonus for co-location
            }
          }
        }

        if (score > bestScore) {
          bestScore = score;
          bestType = static_cast<int>(t);
        }
      }
      assignment[k] = bestType;
    }
  }

  return assignment;
}

//===----------------------------------------------------------------------===//
// Sub-problem: map a kernel to its assigned core
//===----------------------------------------------------------------------===//

L2Assignment BendersDriver::solveSubProblem(const KernelDesc &kernel,
                                             int coreTypeIndex,
                                             const BendersConfig &config) {
  L2Assignment result;
  result.kernelName = kernel.name;
  result.coreTypeIndex = coreTypeIndex;

  if (coreTypeIndex < 0 ||
      coreTypeIndex >= static_cast<int>(arch_.coreTypes.size())) {
    llvm::errs() << "BendersDriver: invalid core type index " << coreTypeIndex
                 << " for kernel '" << kernel.name << "'\n";
    return result;
  }

  const auto &coreType = arch_.coreTypes[coreTypeIndex];
  result.coreADG = coreType.adgModule;

  if (!result.coreADG) {
    llvm::errs() << "BendersDriver: core type '" << coreType.name
                 << "' has no ADG module\n";
    return result;
  }

  if (!kernel.dfgModule) {
    llvm::errs() << "BendersDriver: kernel '" << kernel.name
                 << "' has no DFG module\n";
    return result;
  }

  // Flatten the ADG for the mapper
  loom::ADGFlattener flattener;
  if (!flattener.flatten(result.coreADG, &ctx_)) {
    llvm::errs() << "BendersDriver: ADG flattening failed for core type '"
                 << coreType.name << "'\n";
    return result;
  }

  // Build the DFG from the kernel module
  loom::DFGBuilder dfgBuilder;
  if (!dfgBuilder.build(kernel.dfgModule, &ctx_)) {
    llvm::errs() << "BendersDriver: DFG building failed for kernel '"
                 << kernel.name << "'\n";
    return result;
  }

  // Configure and run the mapper
  loom::Mapper::Options mapOpts;
  mapOpts.budgetSeconds = config.mapperBudgetSeconds;
  mapOpts.seed = config.mapperSeed;
  mapOpts.verbose = config.verbose;

  loom::Mapper mapper;
  auto mapResult =
      mapper.run(dfgBuilder.getDFG(), flattener.getADG(), flattener,
                 result.coreADG, mapOpts);

  result.mappingSuccess = mapResult.success;
  result.mappingCost = mapResult.success ? mapResult.timingSummary.estimatedClockPeriod : 1e6;
  result.unroutedEdges = 0;

  if (!mapResult.success) {
    // Count unrouted edges from diagnostics
    result.routingCongestion = 1.0;
    if (config.verbose) {
      llvm::outs() << "BendersDriver: mapper failed for kernel '"
                   << kernel.name << "' on core type '" << coreType.name
                   << "'\n";
      if (!mapResult.diagnostics.empty())
        llvm::outs() << "  diagnostics: " << mapResult.diagnostics << "\n";
    }
  } else {
    result.routingCongestion = 0.0;
    if (config.verbose) {
      llvm::outs() << "BendersDriver: mapper succeeded for kernel '"
                   << kernel.name << "' on core type '" << coreType.name
                   << "' (cost=" << result.mappingCost << ")\n";
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Benders cuts
//===----------------------------------------------------------------------===//

void BendersDriver::addBendersCut(const L2Assignment &assignment,
                                   unsigned iteration) {
  if (assignment.mappingSuccess)
    return; // No cut needed for successful mappings

  BendersCut cut;
  cut.iteration = iteration;
  cut.kernelName = assignment.kernelName;
  cut.coreTypeIndex = assignment.coreTypeIndex;
  // Penalty proportional to how badly it failed
  cut.penalty = 10.0 + 5.0 * assignment.routingCongestion;
  cuts_.push_back(cut);
}

//===----------------------------------------------------------------------===//
// Convergence check
//===----------------------------------------------------------------------===//

bool BendersDriver::checkConvergence(
    const std::vector<L2Assignment> &assignments) const {
  return std::all_of(assignments.begin(), assignments.end(),
                     [](const L2Assignment &a) { return a.mappingSuccess; });
}

//===----------------------------------------------------------------------===//
// Main compile loop
//===----------------------------------------------------------------------===//

BendersResult BendersDriver::compile(const BendersConfig &config) {
  BendersResult result;

  if (arch_.coreTypes.empty()) {
    result.diagnostics = "No core types in SystemArchitecture";
    return result;
  }

  if (kernels_.empty()) {
    result.diagnostics = "No kernels to compile";
    return result;
  }

  if (config.verbose) {
    llvm::outs() << "BendersDriver: starting compilation with "
                 << kernels_.size() << " kernels, " << arch_.coreTypes.size()
                 << " core types, max " << config.maxIterations
                 << " iterations\n";
  }

  for (unsigned iter = 0; iter < config.maxIterations; ++iter) {
    result.iterations = iter + 1;

    if (config.verbose) {
      llvm::outs() << "\n=== Benders iteration " << (iter + 1) << " ===\n";
    }

    // Master problem: assign kernels to core types
    auto assignment = solveMasterProblem(iter);
    if (assignment.size() != kernels_.size()) {
      result.diagnostics = "Master problem returned invalid assignment";
      return result;
    }

    if (config.verbose) {
      llvm::outs() << "Master assignment:\n";
      for (unsigned k = 0; k < kernels_.size(); ++k) {
        llvm::outs() << "  " << kernels_[k].name << " -> "
                     << arch_.coreTypes[assignment[k]].name << "\n";
      }
    }

    // Sub-problems: map each kernel
    std::vector<L2Assignment> l2Results;
    double totalCost = 0.0;

    for (unsigned k = 0; k < kernels_.size(); ++k) {
      auto l2 = solveSubProblem(kernels_[k], assignment[k], config);
      totalCost += l2.mappingCost;
      l2Results.push_back(std::move(l2));
    }

    result.totalCost = totalCost;
    result.assignments = l2Results;

    // Check convergence
    if (checkConvergence(l2Results)) {
      result.success = true;
      if (config.verbose) {
        llvm::outs() << "\nBendersDriver: converged after " << (iter + 1)
                     << " iterations (total cost=" << totalCost << ")\n";
      }
      return result;
    }

    // Add Benders cuts from failed sub-problems
    for (const auto &l2 : l2Results) {
      addBendersCut(l2, iter);
    }

    if (config.verbose) {
      unsigned successes = 0;
      for (const auto &l2 : l2Results) {
        if (l2.mappingSuccess)
          ++successes;
      }
      llvm::outs() << "Iteration " << (iter + 1) << ": " << successes << "/"
                   << kernels_.size() << " kernels mapped successfully\n";
    }
  }

  result.diagnostics = "Did not converge within " +
                       std::to_string(config.maxIterations) + " iterations";
  if (config.verbose) {
    llvm::outs() << "\nBendersDriver: " << result.diagnostics << "\n";
  }

  return result;
}
