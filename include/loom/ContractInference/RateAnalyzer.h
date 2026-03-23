#ifndef LOOM_CONTRACTINFERENCE_RATEANALYZER_H
#define LOOM_CONTRACTINFERENCE_RATEANALYZER_H

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>

namespace loom {

/// Result of production or consumption rate analysis for a kernel operand.
struct RateResult {
  /// Statically known element count per invocation (if determinable).
  std::optional<int64_t> elementsPerInvocation;

  /// True if the rate is statically exact; false if estimated conservatively.
  bool isExact = false;
};

/// Analyzes kernel bodies (SCF/affine loops) to derive production and
/// consumption rates for inter-kernel communication edges.
class RateAnalyzer {
public:
  /// Analyze a kernel body to determine the number of output elements
  /// produced per invocation for a given output operand.
  /// Falls back to conservative estimate (1 element) for dynamic loops.
  RateResult analyzeProductionRate(mlir::Region &kernelBody,
                                   mlir::Value outputOperand);

  /// Analyze a kernel body to determine the number of input elements
  /// consumed per invocation for a given input operand.
  /// Falls back to conservative estimate (1 element) for dynamic loops.
  RateResult analyzeConsumptionRate(mlir::Region &kernelBody,
                                    mlir::Value inputOperand);

private:
  /// Extract the trip count from an scf.for op. Returns nullopt for
  /// dynamic bounds.
  std::optional<int64_t> extractTripCount(mlir::Operation *forOp);

  /// Compute the aggregate trip count across nested loops in a region.
  /// Multiplies trip counts of all nested scf.for ops.
  int64_t computeNestedTripCount(mlir::Region &region, bool &allStatic);

  /// Check whether the kernel body is a reduction pattern (single output).
  bool isReductionPattern(mlir::Region &kernelBody,
                          mlir::Value outputOperand);
};

} // namespace loom

#endif
