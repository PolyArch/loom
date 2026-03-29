//===-- co_optimizer.h - SW-HW co-optimization loop ---------------*- C++ -*-===//
//
// Top-level co-optimization that alternates software optimization (C10
// TDGOptimizer) and hardware optimization (C11 OUTER-HW + C12 INNER-HW).
// Each round:
//   1. Fixes hardware, runs SW optimizer to improve TDG throughput.
//   2. Fixes software, runs HW optimizer to minimize architecture area.
//   3. Checks convergence: exits when neither throughput nor area improves
//      by more than the threshold.
//
// TDC contracts carry real achieved rates between steps.
// The output is a Pareto frontier of (throughput, area) design points.
//
//===----------------------------------------------------------------------===//

#ifndef TAPESTRY_CO_OPTIMIZER_H
#define TAPESTRY_CO_OPTIMIZER_H

#include "tapestry/tdg_optimizer.h"

#include "loom/SystemCompiler/HWInnerOptimizer.h"
#include "loom/SystemCompiler/HWOuterOptimizer.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// Pareto point
//===----------------------------------------------------------------------===//

/// A single (throughput, area) design point on the Pareto frontier.
struct ParetoPoint {
  double throughput = 0.0;
  double area = 0.0;
  unsigned round = 0;
};

//===----------------------------------------------------------------------===//
// ConvergenceMonitor
//===----------------------------------------------------------------------===//

/// Encapsulates convergence state for the co-optimization loop.
/// Tracks best throughput and area, and determines whether a given round
/// showed sufficient improvement to continue iterating.
struct ConvergenceMonitor {
  double bestThroughput = 0.0;
  double bestArea = std::numeric_limits<double>::infinity();
  double improvementThreshold = 0.01;

  /// Check whether the given (throughput, area) pair represents a meaningful
  /// improvement over the current best. Updates internal state and returns
  /// true if improvement exceeds the threshold.
  /// \param throughput   SW throughput from this round.
  /// \param area         HW area from this round.
  /// \param outReason    Human-readable string describing the decision.
  /// \returns true if the round improved, false if converged.
  bool checkImproved(double throughput, double area, std::string &outReason) {
    bool throughputImproved =
        throughput > bestThroughput * (1.0 + improvementThreshold);
    bool areaImproved =
        area < bestArea * (1.0 - improvementThreshold);
    bool improved = throughputImproved || areaImproved;

    std::ostringstream oss;
    if (improved) {
      if (throughputImproved && areaImproved)
        oss << "both throughput and area improved";
      else if (throughputImproved)
        oss << "throughput improved";
      else
        oss << "area improved";
    } else {
      oss << "no significant improvement "
          << "(throughput=" << throughput
          << " vs best=" << bestThroughput
          << ", area=" << area
          << " vs best=" << bestArea << ")";
    }
    outReason = oss.str();

    // Update tracked bests unconditionally so that downstream callers
    // can compare against the latest observed values.
    if (throughput > bestThroughput)
      bestThroughput = throughput;
    if (area < bestArea)
      bestArea = area;

    return improved;
  }
};

//===----------------------------------------------------------------------===//
// CoOptOptions
//===----------------------------------------------------------------------===//

/// Configuration for the SW-HW co-optimization loop.
struct CoOptOptions {
  /// Maximum number of alternating SW-HW rounds.
  unsigned maxRounds = 5;

  /// Minimum relative improvement threshold to continue iterating.
  /// Loop terminates when neither throughput nor area improves by more
  /// than this fraction.
  double improvementThreshold = 0.01;

  /// Software (C10) optimizer configuration.
  TDGOptimizeOptions swOpts;

  /// Hardware outer (C11) optimizer configuration.
  loom::HWOuterOptimizerOptions hwOuterOpts;

  /// Hardware inner (C12) per-core optimizer configuration.
  loom::HWInnerOptimizerOptions hwInnerOpts;

  /// Enable verbose diagnostic logging.
  bool verbose = false;

  /// Enable the SW optimization step. When false, the SW step is skipped
  /// and throughput is carried forward from the previous round.
  bool enableSW = true;

  /// Enable the HW optimization step. When false, the HW step is skipped
  /// and the architecture remains unchanged from the previous round.
  bool enableHW = true;

  /// Enable inner-layer SW iteration (L2 iterative re-mapping in
  /// TDGOptimizer). When false, TDGOptimizeOptions::maxIterations is
  /// forced to 1 (single-pass).
  bool enableSWInner = true;

  /// Enable inner-layer HW iteration (Tier-B BO + mapper). When false,
  /// HWInnerOptimizerOptions::tier2Enabled is forced to false.
  bool enableHWInner = true;
};

//===----------------------------------------------------------------------===//
// CoOptResult
//===----------------------------------------------------------------------===//

/// Result of the co-optimization loop.
struct CoOptResult {
  bool success = false;

  /// Best kernel descriptors after optimization.
  std::vector<loom::tapestry::KernelDesc> bestKernels;

  /// Best system architecture (from loom::tapestry).
  loom::tapestry::SystemArchitecture bestArch;

  /// Best contracts after optimization.
  std::vector<loom::tapestry::ContractSpec> bestContracts;

  /// Best achieved throughput (ops/cycle).
  double bestThroughput = 0.0;

  /// Best achieved area (abstract area units).
  double bestArea = std::numeric_limits<double>::infinity();

  /// Non-dominated Pareto frontier across all rounds.
  std::vector<ParetoPoint> paretoFrontier;

  /// Total number of rounds executed.
  unsigned rounds = 0;

  /// Per-round history.
  struct RoundRecord {
    unsigned round = 0;
    double swThroughput = 0.0;
    double hwArea = 0.0;
    unsigned swTransforms = 0;
    unsigned hwCoreTypes = 0;
    bool improved = false;
    std::string reason;
  };
  std::vector<RoundRecord> history;

  /// Whether a warm-start initialization path was taken (baseline from
  /// pre-existing architecture + contracts rather than default).
  bool warmStartUsed = false;

  /// Final BendersResult from the best SW step.
  loom::tapestry::BendersResult bestBendersResult;

  /// Diagnostic messages.
  std::string diagnostics;
};

//===----------------------------------------------------------------------===//
// co_optimize()
//===----------------------------------------------------------------------===//

/// Top-level co-optimization entry point.
///
/// Alternates SW optimization (TDGOptimizer) and HW optimization
/// (HWOuterOptimizer + HWInnerOptimizer) in a bounded loop.
///
/// \param kernels      Kernel descriptors (DFG modules from C08).
/// \param contracts    Inter-kernel communication contracts.
/// \param initialArch  Initial system architecture. If coreTypes is empty,
///                     an initial architecture is derived from kernel profiles.
/// \param coOpts       Co-optimization configuration.
/// \param ctx          MLIR context with required dialects registered.
/// \returns            CoOptResult with the best TDG+architecture pair and
///                     the Pareto frontier.
CoOptResult co_optimize(std::vector<loom::tapestry::KernelDesc> kernels,
                        std::vector<loom::tapestry::ContractSpec> contracts,
                        const loom::tapestry::SystemArchitecture &initialArch,
                        const CoOptOptions &coOpts,
                        mlir::MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Helper: extract kernel profiles from KernelDesc
//===----------------------------------------------------------------------===//

/// Build KernelProfile list from KernelDesc list for HW optimizer input.
std::vector<loom::KernelProfile>
extractKernelProfiles(const std::vector<loom::tapestry::KernelDesc> &kernels);

//===----------------------------------------------------------------------===//
// Helper: build initial architecture from kernel profiles
//===----------------------------------------------------------------------===//

/// Derive a default SystemArchitecture from kernel profiles when none is
/// provided. Assigns all kernels to a single balanced core type.
loom::tapestry::SystemArchitecture
buildDefaultArchitecture(const std::vector<loom::KernelProfile> &profiles);

//===----------------------------------------------------------------------===//
// Helper: convert loom::ContractSpec <-> loom::tapestry::ContractSpec
//===----------------------------------------------------------------------===//

/// Convert loom::ContractSpec (from Contract.h) to loom::tapestry::ContractSpec
/// (from SystemTypes.h) for BendersDriver compatibility.
std::vector<loom::tapestry::ContractSpec>
toLoomTapestryContracts(const std::vector<loom::ContractSpec> &loomContracts);

/// Convert loom::tapestry::ContractSpec back to loom::ContractSpec.
std::vector<loom::ContractSpec>
fromLoomTapestryContracts(
    const std::vector<loom::tapestry::ContractSpec> &tapContracts);

//===----------------------------------------------------------------------===//
// Helper: update contracts from SW result
//===----------------------------------------------------------------------===//

/// Propagate achieved rates from the SW optimization BendersResult back
/// into loom::ContractSpec achieved fields.
void updateContractsFromSW(
    std::vector<loom::tapestry::ContractSpec> &contracts,
    const TDGOptimizeResult &swResult);

//===----------------------------------------------------------------------===//
// Helper: compute total system area
//===----------------------------------------------------------------------===//

/// Compute total system area from HW outer topology and inner per-core areas.
double computeSystemArea(
    const loom::HWOuterOptimizerResult &outerResult,
    const std::vector<loom::ADGOptResult> &innerResults);

//===----------------------------------------------------------------------===//
// Helper: Pareto frontier management
//===----------------------------------------------------------------------===//

/// Add a point to the Pareto frontier if it is non-dominated.
/// Removes any existing points dominated by the new one.
void addParetoPoint(std::vector<ParetoPoint> &frontier,
                    const ParetoPoint &candidate);

//===----------------------------------------------------------------------===//
// Helper: build SystemArchitecture from HW results
//===----------------------------------------------------------------------===//

/// Construct a loom::tapestry::SystemArchitecture from OUTER-HW topology
/// and INNER-HW ADG results.
loom::tapestry::SystemArchitecture
buildArchFromHWResults(const loom::HWOuterOptimizerResult &outerResult,
                       const std::vector<loom::ADGOptResult> &innerResults,
                       mlir::MLIRContext *ctx);

} // namespace tapestry

#endif // TAPESTRY_CO_OPTIMIZER_H
