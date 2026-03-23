#include "loom/ContractInference/RateAnalyzer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <optional>

using namespace loom;
using namespace mlir;

std::optional<int64_t> RateAnalyzer::extractTripCount(Operation *op) {
  auto forOp = dyn_cast<scf::ForOp>(op);
  if (!forOp)
    return std::nullopt;

  // Try to extract constant bounds from the scf.for operands.
  auto getConstant = [](Value v) -> std::optional<int64_t> {
    if (auto defOp = v.getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
          return intAttr.getInt();
      }
    }
    // Check for block arguments that are constants (from parent ops).
    return std::nullopt;
  };

  auto lowerBound = getConstant(forOp.getLowerBound());
  auto upperBound = getConstant(forOp.getUpperBound());
  auto step = getConstant(forOp.getStep());

  if (!lowerBound || !upperBound || !step)
    return std::nullopt;

  int64_t lb = *lowerBound;
  int64_t ub = *upperBound;
  int64_t st = *step;

  if (st <= 0)
    return std::nullopt;

  int64_t tripCount = (ub - lb + st - 1) / st;
  return (tripCount > 0) ? std::optional<int64_t>(tripCount) : std::nullopt;
}

int64_t RateAnalyzer::computeNestedTripCount(Region &region, bool &allStatic) {
  int64_t totalTripCount = 1;
  allStatic = true;

  for (Block &block : region) {
    for (Operation &op : block) {
      if (isa<scf::ForOp>(op)) {
        auto tc = extractTripCount(&op);
        if (tc) {
          totalTripCount *= *tc;
        } else {
          allStatic = false;
          // Conservative: assume 1 iteration for dynamic loops.
          totalTripCount *= 1;
        }

        // Recurse into nested loops.
        auto forOp = cast<scf::ForOp>(op);
        Region &bodyRegion = forOp.getRegion();
        bool innerStatic = true;
        int64_t innerCount =
            computeNestedTripCount(bodyRegion, innerStatic);
        if (!innerStatic)
          allStatic = false;
        // The innerCount is the trip count of deeper nesting levels.
        // Multiply into the total (already includes current level).
        if (innerCount > 1)
          totalTripCount *= innerCount;
      }
    }
  }

  return totalTripCount;
}

bool RateAnalyzer::isReductionPattern(Region &kernelBody,
                                      Value outputOperand) {
  // A reduction pattern is characterized by:
  //   - The output is accumulated across loop iterations
  //   - Only a single final value is produced
  //
  // Heuristic: if there are loop-carried values (iter_args in scf.for)
  // that feed into the output, this is a reduction.
  for (Block &block : kernelBody) {
    for (Operation &op : block) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        // If the for loop has iter_args (loop-carried dependencies),
        // and the output depends on a result of the for loop,
        // treat this as a reduction pattern.
        if (!forOp.getInitArgs().empty()) {
          for (auto result : forOp.getResults()) {
            if (result == outputOperand)
              return true;
          }
        }
      }
    }
  }
  return false;
}

RateResult RateAnalyzer::analyzeProductionRate(Region &kernelBody,
                                               Value outputOperand) {
  RateResult result;

  // Check for reduction pattern first -- single output element.
  if (isReductionPattern(kernelBody, outputOperand)) {
    result.elementsPerInvocation = 1;
    result.isExact = true;
    return result;
  }

  // Walk the kernel body looking for scf.for loops and compute
  // the aggregate trip count across all nesting levels.
  bool allStatic = true;
  int64_t tripCount = computeNestedTripCount(kernelBody, allStatic);

  if (tripCount > 0) {
    result.elementsPerInvocation = tripCount;
    result.isExact = allStatic;
  } else {
    // Conservative fallback: 1 element per invocation.
    result.elementsPerInvocation = 1;
    result.isExact = false;
  }

  return result;
}

RateResult RateAnalyzer::analyzeConsumptionRate(Region &kernelBody,
                                                Value inputOperand) {
  RateResult result;

  // Consumption rate analysis mirrors production rate.
  // Walk loops to determine how many input elements are accessed.
  bool allStatic = true;
  int64_t tripCount = computeNestedTripCount(kernelBody, allStatic);

  if (tripCount > 0) {
    result.elementsPerInvocation = tripCount;
    result.isExact = allStatic;
  } else {
    // Conservative fallback.
    result.elementsPerInvocation = 1;
    result.isExact = false;
  }

  return result;
}
