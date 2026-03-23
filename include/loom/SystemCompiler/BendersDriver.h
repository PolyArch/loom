//===-- BendersDriver.h - Multi-core Benders decomposition driver --*- C++ -*-===//
//
// Drives the Benders decomposition for multi-core CGRA compilation.
// The master problem assigns kernels to core types, and the sub-problems
// map each kernel onto its assigned core's ADG using the Loom mapper.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H
#define LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H

#include "loom/SystemCompiler/SystemTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

namespace loom {
namespace tapestry {

/// Configuration for the Benders driver.
struct BendersConfig {
  unsigned maxIterations = 10;
  double mapperBudgetSeconds = 30.0;
  int mapperSeed = 42;
  bool verbose = true;
};

/// The Benders decomposition driver for multi-core compilation.
///
/// Usage:
///   1. Construct with a SystemArchitecture, kernels, and contracts.
///   2. Call compile() to run the decomposition loop.
///   3. Inspect the BendersResult for assignments and metrics.
class BendersDriver {
public:
  BendersDriver(const SystemArchitecture &arch,
                std::vector<KernelDesc> kernels,
                std::vector<ContractSpec> contracts,
                mlir::MLIRContext &ctx);

  /// Run the Benders decomposition. Returns the result.
  BendersResult compile(const BendersConfig &config = {});

private:
  /// Master problem: assign kernels to core types.
  /// Returns a kernel->coreTypeIndex mapping.
  std::vector<int> solveMasterProblem(unsigned iteration);

  /// Sub-problem: map one kernel to its assigned core's ADG.
  /// Returns the L2Assignment with mapping results.
  L2Assignment solveSubProblem(const KernelDesc &kernel,
                               int coreTypeIndex,
                               const BendersConfig &config);

  /// Generate a Benders cut from sub-problem feedback.
  void addBendersCut(const L2Assignment &assignment, unsigned iteration);

  /// Check convergence: all sub-problems succeeded.
  bool checkConvergence(const std::vector<L2Assignment> &assignments) const;

  const SystemArchitecture &arch_;
  std::vector<KernelDesc> kernels_;
  std::vector<ContractSpec> contracts_;
  mlir::MLIRContext &ctx_;

  /// Accumulated cuts: (iteration, kernelName, coreTypeIndex) -> penalty.
  struct BendersCut {
    unsigned iteration;
    std::string kernelName;
    int coreTypeIndex;
    double penalty;
  };
  std::vector<BendersCut> cuts_;
};

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_BENDERSDRIVER_H
