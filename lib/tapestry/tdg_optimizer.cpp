//===-- tdg_optimizer.cpp - TDG iterative optimization loop --------*- C++ -*-===//
//
// Implements the iterative TDG optimization loop. Each iteration:
//   1. Evaluates the current TDG via HierarchicalCompiler mapping.
//   2. Computes system throughput from the result.
//   3. Tries contract-gated transforms (retile, replicate).
//   4. Accepts transforms that improve throughput.
//   5. Converges when no transform improves or max iterations reached.
//
//===----------------------------------------------------------------------===//

#include "tapestry/tdg_optimizer.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

using namespace tapestry;
using namespace loom::tapestry;

//===----------------------------------------------------------------------===//
// Throughput computation
//===----------------------------------------------------------------------===//

/// Compute throughput from a CompilationResult. Throughput is the inverse of the
/// total cost (lower cost = higher throughput). Returns 0 on failure.
static double computeThroughput(const CompilationResult &result) {
  if (!result.success || result.totalCost <= 0.0)
    return 0.0;
  return 1.0 / result.totalCost;
}

/// Compute the maximum mapping cost across all assigned kernels.
/// The kernel with the highest cost is the bottleneck.
static double findMaxKernelCost(const CompilationResult &result) {
  double maxCost = 0.0;
  for (const auto &a : result.assignments) {
    if (a.mappingSuccess && a.mappingCost > maxCost)
      maxCost = a.mappingCost;
  }
  return maxCost;
}

//===----------------------------------------------------------------------===//
// TDGOptimizer
//===----------------------------------------------------------------------===//

TDGOptimizer::TDGOptimizer(const TDGOptimizeOptions &options,
                           mlir::MLIRContext &ctx)
    : options_(options), ctx_(ctx) {}

TDGOptimizeResult
TDGOptimizer::optimize(std::vector<KernelDesc> kernels,
                       std::vector<ContractSpec> contracts,
                       const SystemArchitecture &arch,
                       mlir::ModuleOp systemADG) {
  TDGOptimizeResult result;
  result.optimizedKernels = kernels;
  result.optimizedContracts = contracts;

  if (kernels.empty()) {
    result.success = false;
    result.diagnostics = "No kernels to optimize";
    return result;
  }

  double bestThroughput = 0.0;
  CompilationResult bestCompilationResult;

  for (unsigned iter = 1; iter <= options_.maxIterations; ++iter) {
    if (options_.verbose)
      llvm::outs() << "TDGOptimizer: iteration " << iter << "\n";

    // Evaluate current TDG configuration.
    CompilationResult currentResult;
    double currentThroughput =
        evaluate(result.optimizedKernels, result.optimizedContracts,
                 arch, currentResult);

    if (options_.verbose)
      llvm::outs() << "  throughput = " << currentThroughput
                    << " (cost = " << currentResult.totalCost << ")\n";

    // Update best if improved.
    if (currentResult.success &&
        currentThroughput > bestThroughput + options_.improvementThreshold) {
      bestThroughput = currentThroughput;
      bestCompilationResult = currentResult;
    }

    if (!currentResult.success) {
      // Mapping failed entirely. For MVP, HierarchicalCompiler handles reassignment
      // via its own iteration loop. If it fails, report diagnostics.
      if (bestThroughput > 0.0) {
        // We had a previous successful configuration; keep it.
        if (options_.verbose)
          llvm::outs() << "  Mapping failed; reverting to previous best.\n";
        result.iterations = iter;
        break;
      }
      // No previous success; continue trying transforms.
      result.diagnostics = "HierarchicalCompiler mapping failed: " +
                           currentResult.diagnostics;
    }

    // Try transforms in priority order.
    bool improved = false;

    // Priority 1: Retile to fix rate imbalances.
    improved = tryRetileTransforms(
        result.optimizedKernels, result.optimizedContracts, arch,
        bestThroughput, result.transformHistory, iter);

    // Priority 2: Replicate bottleneck kernels.
    if (!improved) {
      improved = tryReplicateTransforms(
          result.optimizedKernels, result.optimizedContracts, arch,
          bestThroughput, currentResult, result.transformHistory, iter);
    }

    result.iterations = iter;

    if (!improved) {
      if (options_.verbose)
        llvm::outs() << "  No beneficial transform found; converged.\n";
      break;
    }
  }

  // Populate final result.
  result.success = bestThroughput > 0.0;
  result.compilationResult = bestCompilationResult;
  result.bestThroughput = bestThroughput;

  if (!result.success && result.diagnostics.empty())
    result.diagnostics = "Optimization loop exhausted without successful mapping";

  return result;
}

double TDGOptimizer::evaluate(const std::vector<KernelDesc> &kernels,
                              const std::vector<ContractSpec> &contracts,
                              const SystemArchitecture &arch,
                              CompilationResult &outResult) {
  // Create a HierarchicalCompiler with the current TDG configuration and run it.
  HierarchicalCompiler driver(arch, kernels, contracts, ctx_);
  outResult = driver.compile(options_.compilerConfig);
  return computeThroughput(outResult);
}

//===----------------------------------------------------------------------===//
// Retile Transform
//===----------------------------------------------------------------------===//

bool TDGOptimizer::isRateImbalanced(const ContractSpec &contract,
                                    const CompilationResult &result) {
  // A contract is rate-imbalanced if its bandwidth is either 0 or if the
  // producer and consumer are on different core types with different capacities.
  // For MVP, use a simple heuristic: if elementCount > 0 and
  // bandwidthBytesPerCycle is very low relative to element count, the rate
  // is imbalanced.
  if (contract.elementCount == 0)
    return false;

  // Check if producer and consumer ended up on different core types.
  if (contract.producerCoreType >= 0 && contract.consumerCoreType >= 0 &&
      contract.producerCoreType != contract.consumerCoreType)
    return true;

  // Check if communication cost is disproportionately high.
  if (contract.communicationCost > 0.0 && result.totalCost > 0.0) {
    double ratio = contract.communicationCost / result.totalCost;
    if (ratio > 0.3)
      return true;
  }

  return false;
}

bool TDGOptimizer::tryRetileTransforms(
    std::vector<KernelDesc> &kernels,
    std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch,
    double currentThroughput,
    std::vector<TransformRecord> &history,
    unsigned iteration) {

  // Scan contracts for retile candidates.
  for (size_t ci = 0; ci < contracts.size(); ++ci) {
    auto &contract = contracts[ci];

    // Gate: check permission. ContractSpec in tapestry namespace does not
    // have mayRetile, so we always allow retile for MVP. When loom::ContractSpec
    // permissions are integrated, this should check contract.mayRetile.

    // Save original state for rollback.
    ContractSpec savedContract = contract;

    // Attempt retile.
    if (!applyRetile(contract, {})) {
      contract = savedContract;
      continue;
    }

    // Evaluate the modified TDG.
    CompilationResult candidateResult;
    double candidateThroughput =
        evaluate(kernels, contracts, arch, candidateResult);

    TransformRecord record;
    record.iteration = iteration;
    record.transformType = "retile";
    record.targetKernel = contract.producerKernel + "->" +
                          contract.consumerKernel;
    record.throughputBefore = currentThroughput;
    record.throughputAfter = candidateThroughput;

    if (candidateThroughput >
        currentThroughput + options_.improvementThreshold) {
      record.accepted = true;
      history.push_back(record);

      if (options_.verbose)
        llvm::outs() << "  Retile accepted: "
                      << record.targetKernel
                      << " (throughput " << currentThroughput
                      << " -> " << candidateThroughput << ")\n";
      return true;
    }

    // Reject: rollback.
    record.accepted = false;
    history.push_back(record);
    contract = savedContract;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Replicate Transform
//===----------------------------------------------------------------------===//

std::string TDGOptimizer::findBottleneckKernel(
    const CompilationResult &result,
    const std::vector<KernelDesc> &kernels) {
  if (!result.success || result.assignments.empty())
    return "";

  // Find the assignment with the highest mapping cost.
  double maxCost = 0.0;
  std::string bottleneck;

  for (const auto &assignment : result.assignments) {
    if (assignment.mappingSuccess && assignment.mappingCost > maxCost) {
      maxCost = assignment.mappingCost;
      bottleneck = assignment.kernelName;
    }
  }

  return bottleneck;
}

bool TDGOptimizer::tryReplicateTransforms(
    std::vector<KernelDesc> &kernels,
    std::vector<ContractSpec> &contracts,
    const SystemArchitecture &arch,
    double currentThroughput,
    const CompilationResult &lastResult,
    std::vector<TransformRecord> &history,
    unsigned iteration) {

  std::string bottleneck = findBottleneckKernel(lastResult, kernels);
  if (bottleneck.empty())
    return false;

  // Gate: check permission. For MVP, we always allow replication.
  // When loom::ContractSpec permissions are integrated, check mayReplicate
  // on all contracts touching this kernel.

  // Check that we have enough core instances to accommodate a replica.
  unsigned totalInstances = 0;
  for (const auto &ct : arch.coreTypes)
    totalInstances += ct.numInstances;

  if (totalInstances <= kernels.size()) {
    if (options_.verbose)
      llvm::outs() << "  Cannot replicate: no spare core instances.\n";
    return false;
  }

  // Save state for rollback.
  auto savedKernels = kernels;
  auto savedContracts = contracts;

  if (!applyReplicate(bottleneck, kernels, contracts, ctx_)) {
    kernels = savedKernels;
    contracts = savedContracts;
    return false;
  }

  // Evaluate.
  CompilationResult candidateResult;
  double candidateThroughput =
      evaluate(kernels, contracts, arch, candidateResult);

  TransformRecord record;
  record.iteration = iteration;
  record.transformType = "replicate";
  record.targetKernel = bottleneck;
  record.throughputBefore = currentThroughput;
  record.throughputAfter = candidateThroughput;

  if (candidateThroughput >
      currentThroughput + options_.improvementThreshold) {
    record.accepted = true;
    history.push_back(record);

    if (options_.verbose)
      llvm::outs() << "  Replicate accepted: " << bottleneck
                    << " (throughput " << currentThroughput
                    << " -> " << candidateThroughput << ")\n";
    return true;
  }

  // Reject: rollback.
  record.accepted = false;
  history.push_back(record);
  kernels = savedKernels;
  contracts = savedContracts;
  return false;
}

//===----------------------------------------------------------------------===//
// Transform Implementations (free functions)
//===----------------------------------------------------------------------===//

bool tapestry::applyRetile(ContractSpec &contract,
                           const CompilationResult &result) {
  // Retile heuristic: if elementCount is large, try halving it to reduce
  // per-tile data volume (improves SPM fit). If elementCount is small,
  // try doubling it (amortizes transfer overhead).
  //
  // The threshold for "large" is 1024 elements.
  if (contract.elementCount == 0)
    return false;

  uint64_t originalCount = contract.elementCount;

  if (contract.elementCount > 1024) {
    // Large tile: halve to improve SPM fit.
    contract.elementCount = contract.elementCount / 2;
  } else if (contract.elementCount < 64) {
    // Very small tile: double to amortize overhead.
    contract.elementCount = contract.elementCount * 2;
  } else {
    // Mid-range: try reducing by 25%.
    contract.elementCount = (contract.elementCount * 3) / 4;
  }

  // Ensure minimum element count.
  if (contract.elementCount < 1)
    contract.elementCount = 1;

  // Return true if we actually changed something.
  return contract.elementCount != originalCount;
}

bool tapestry::applyReplicate(const std::string &kernelName,
                              std::vector<KernelDesc> &kernels,
                              std::vector<ContractSpec> &contracts,
                              mlir::MLIRContext &ctx) {
  // Find the kernel to replicate.
  int kernelIdx = -1;
  for (size_t i = 0; i < kernels.size(); ++i) {
    if (kernels[i].name == kernelName) {
      kernelIdx = static_cast<int>(i);
      break;
    }
  }

  if (kernelIdx < 0)
    return false;

  const auto &original = kernels[static_cast<size_t>(kernelIdx)];

  // Create a replica kernel with a suffixed name.
  KernelDesc replica;
  replica.name = kernelName + "_replica";
  replica.dfgModule = original.dfgModule; // Share the DFG module
  replica.requiredPEs = original.requiredPEs;
  replica.requiredFUs = original.requiredFUs;
  replica.requiredMemoryBytes = original.requiredMemoryBytes;

  kernels.push_back(replica);

  // Split contracts: for each contract where the original kernel is a
  // producer, add a parallel contract from the replica. For each contract
  // where it is a consumer, add a parallel contract to the replica.
  //
  // The element counts are halved (round-robin split).
  std::vector<ContractSpec> newContracts;
  for (auto &c : contracts) {
    if (c.producerKernel == kernelName) {
      ContractSpec replicaContract = c;
      replicaContract.producerKernel = replica.name;
      // Split output: each produces half.
      uint64_t halfCount = c.elementCount / 2;
      c.elementCount = halfCount + (c.elementCount % 2); // Original gets remainder
      replicaContract.elementCount = halfCount;
      newContracts.push_back(replicaContract);
    }

    if (c.consumerKernel == kernelName) {
      ContractSpec replicaContract = c;
      replicaContract.consumerKernel = replica.name;
      // Split input: each consumes half.
      uint64_t halfCount = c.elementCount / 2;
      c.elementCount = halfCount + (c.elementCount % 2);
      replicaContract.elementCount = halfCount;
      newContracts.push_back(replicaContract);
    }
  }

  // Append new contracts.
  contracts.insert(contracts.end(), newContracts.begin(), newContracts.end());

  return true;
}
