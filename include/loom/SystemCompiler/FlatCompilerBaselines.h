#ifndef LOOM_SYSTEMCOMPILER_FLATCOMPILERBASELINES_H
#define LOOM_SYSTEMCOMPILER_FLATCOMPILERBASELINES_H

#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Baseline Result
//===----------------------------------------------------------------------===//

/// Result from a flat (non-decomposed) compiler baseline.
struct FlatBaselineResult {
  bool feasible = false;
  bool timedOut = false;
  double solveTimeSec = 0.0;

  /// Assignment (same format as L1 result).
  AssignmentResult assignment;

  /// Description of the baseline method used.
  std::string methodName;
};

//===----------------------------------------------------------------------===//
// Monolithic ILP Baseline
//===----------------------------------------------------------------------===//

/// Monolithic ILP baseline: formulates the entire multi-core mapping as a
/// single ILP problem. Combines L1 assignment and L2 per-node placement
/// constraints into one model. Expected to time out on large instances,
/// which is valid experimental data proving the necessity of decomposition.
class MonolithicILPBaseline {
public:
  struct Options {
    unsigned timeoutSec = 300;
    unsigned numWorkers = 0; // 0 = auto-detect
    bool verbose = false;
  };

  /// Solve the monolithic ILP.
  FlatBaselineResult solve(const std::vector<KernelProfile> &kernels,
                           const std::vector<ContractSpec> &contracts,
                           const SystemArchitecture &arch,
                           const Options &opts);
};

//===----------------------------------------------------------------------===//
// Heuristic Flat Baseline
//===----------------------------------------------------------------------===//

/// Greedy heuristic baseline: assigns each kernel to the core type with the
/// most matching FUs, then runs L2 mapping per core without Benders feedback.
/// Single pass, no iteration.
class HeuristicFlatBaseline {
public:
  struct Options {
    bool verbose = false;
  };

  /// Run the greedy assignment heuristic.
  FlatBaselineResult solve(const std::vector<KernelProfile> &kernels,
                           const std::vector<ContractSpec> &contracts,
                           const SystemArchitecture &arch,
                           const Options &opts);
};

//===----------------------------------------------------------------------===//
// Exhaustive Small Instance Solver
//===----------------------------------------------------------------------===//

/// Exhaustive enumeration for small instances (|K| <= maxKernels,
/// |T| <= maxTypes). Enumerates all possible kernel-to-core assignments,
/// evaluating each to find the true optimal. Used for optimality gap
/// measurement against the bilevel Benders result.
class ExhaustiveSmallInstance {
public:
  struct Options {
    unsigned maxKernels = 3;
    unsigned maxCoreTypes = 2;
    bool verbose = false;
  };

  /// Solve by exhaustive enumeration.
  /// Returns infeasible result if the instance exceeds size limits.
  FlatBaselineResult solve(const std::vector<KernelProfile> &kernels,
                           const std::vector<ContractSpec> &contracts,
                           const SystemArchitecture &arch,
                           const Options &opts);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_FLATCOMPILERBASELINES_H
