//===-- tdg_optimizer.h - TDG iterative optimization loop ---------*- C++ -*-===//
//
// Iterative optimization of Task Dataflow Graphs using contract-driven
// transforms and BendersDriver feedback. The optimizer applies greedy
// transforms (retile, replicate) gated by contract permissions, evaluates
// each variant via BendersDriver mapping, and converges on the best TDG.
//
//===----------------------------------------------------------------------===//

#ifndef TAPESTRY_TDG_OPTIMIZER_H
#define TAPESTRY_TDG_OPTIMIZER_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// Transform Records
//===----------------------------------------------------------------------===//

/// Records one attempted transform during optimization.
struct TransformRecord {
  unsigned iteration = 0;
  std::string transformType; // "retile", "replicate"
  std::string targetKernel;
  double throughputBefore = 0.0;
  double throughputAfter = 0.0;
  bool accepted = false;
};

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

/// Configuration for the TDG optimization loop.
struct TDGOptimizeOptions {
  /// Maximum number of optimization iterations.
  unsigned maxIterations = 10;

  /// Minimum relative throughput improvement to accept a transform.
  double improvementThreshold = 0.01;

  /// BendersDriver configuration for each evaluation.
  loom::tapestry::BendersConfig bendersConfig;

  /// Enable verbose logging.
  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Result
//===----------------------------------------------------------------------===//

/// Result of the TDG optimization loop.
struct TDGOptimizeResult {
  bool success = false;

  /// The (possibly transformed) kernel descriptors after optimization.
  std::vector<loom::tapestry::KernelDesc> optimizedKernels;

  /// The (possibly transformed) contracts after optimization.
  std::vector<loom::tapestry::ContractSpec> optimizedContracts;

  /// The final BendersDriver compilation result.
  loom::tapestry::BendersResult compilationResult;

  /// Total number of iterations executed.
  unsigned iterations = 0;

  /// Best achieved system throughput (inverse of total cost).
  double bestThroughput = 0.0;

  /// History of all attempted transforms.
  std::vector<TransformRecord> transformHistory;

  /// Diagnostic message on failure.
  std::string diagnostics;
};

//===----------------------------------------------------------------------===//
// TDG Optimizer
//===----------------------------------------------------------------------===//

/// Iterative TDG optimizer using contract-driven transforms and
/// BendersDriver feedback.
///
/// The optimization loop:
///   1. Evaluate current TDG via BendersDriver.
///   2. Compute system throughput from the result.
///   3. Identify candidate transforms gated by contract permissions.
///   4. Apply best candidate, re-evaluate, accept if improved.
///   5. Repeat until convergence or maxIterations.
class TDGOptimizer {
public:
  TDGOptimizer(const TDGOptimizeOptions &options, mlir::MLIRContext &ctx);

  /// Run the optimization loop on the given kernels/contracts/architecture.
  ///
  /// \param kernels     Kernel descriptors (DFG modules).
  /// \param contracts   Inter-kernel communication contracts.
  /// \param arch        System architecture (core types + ADG modules).
  /// \param systemADG   Optional MLIR module for the system (for C13 interface).
  /// \returns           Optimization result with transformed TDG.
  TDGOptimizeResult optimize(std::vector<loom::tapestry::KernelDesc> kernels,
                             std::vector<loom::tapestry::ContractSpec> contracts,
                             const loom::tapestry::SystemArchitecture &arch,
                             mlir::ModuleOp systemADG = nullptr);

private:
  /// Evaluate a TDG configuration via BendersDriver.
  /// Returns throughput (higher is better), or 0.0 on failure.
  double evaluate(const std::vector<loom::tapestry::KernelDesc> &kernels,
                  const std::vector<loom::tapestry::ContractSpec> &contracts,
                  const loom::tapestry::SystemArchitecture &arch,
                  loom::tapestry::BendersResult &outResult);

  /// Try applying retile transforms to improve rate balance.
  /// Returns true if a beneficial transform was found and applied.
  bool tryRetileTransforms(
      std::vector<loom::tapestry::KernelDesc> &kernels,
      std::vector<loom::tapestry::ContractSpec> &contracts,
      const loom::tapestry::SystemArchitecture &arch,
      double currentThroughput,
      std::vector<TransformRecord> &history,
      unsigned iteration);

  /// Try applying replicate transforms to relieve bottlenecks.
  /// Returns true if a beneficial transform was found and applied.
  bool tryReplicateTransforms(
      std::vector<loom::tapestry::KernelDesc> &kernels,
      std::vector<loom::tapestry::ContractSpec> &contracts,
      const loom::tapestry::SystemArchitecture &arch,
      double currentThroughput,
      const loom::tapestry::BendersResult &lastResult,
      std::vector<TransformRecord> &history,
      unsigned iteration);

  /// Find the bottleneck kernel from a BendersDriver result.
  /// Returns the kernel name, or empty string if none found.
  std::string findBottleneckKernel(
      const loom::tapestry::BendersResult &result,
      const std::vector<loom::tapestry::KernelDesc> &kernels);

  /// Check if a contract has a rate imbalance worth retiling.
  bool isRateImbalanced(const loom::tapestry::ContractSpec &contract,
                        const loom::tapestry::BendersResult &result);

  TDGOptimizeOptions options_;
  mlir::MLIRContext &ctx_;
};

//===----------------------------------------------------------------------===//
// Transform Implementations
//===----------------------------------------------------------------------===//

/// Apply a retile transform to a contract edge.
/// Adjusts the tile shape to better balance producer/consumer rates.
/// Returns true if the tile shape was changed.
bool applyRetile(loom::tapestry::ContractSpec &contract,
                 const loom::tapestry::BendersResult &result);

/// Apply a replicate transform to a bottleneck kernel.
/// Duplicates the kernel and splits input/output contracts accordingly.
/// Returns true if replication was applied.
bool applyReplicate(const std::string &kernelName,
                    std::vector<loom::tapestry::KernelDesc> &kernels,
                    std::vector<loom::tapestry::ContractSpec> &contracts,
                    mlir::MLIRContext &ctx);

} // namespace tapestry

#endif // TAPESTRY_TDG_OPTIMIZER_H
