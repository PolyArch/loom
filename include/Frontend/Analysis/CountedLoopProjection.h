#ifndef LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H
#define LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <optional>

namespace loom::frontend::analysis {

/// Exact finite domain of a post-tested counted loop. The loop executes
/// lowerBound, lowerBound + step, ..., upperBound - step and its failed
/// condition publishes upperBound. Every non-induction state lane is fed back
/// through the identity after-region in the same ordinal.
struct ExactPostTestedCountedLoopProjection final {
  mlir::scf::WhileOp loop;
  unsigned inductionLane = 0;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  std::optional<llvm::APInt> lowerBoundValue;
  std::optional<llvm::APInt> upperBoundValue;
  std::optional<llvm::APInt> stepValue;
  /// The domain is exact only under unsigned comparison: the induction
  /// update carries a no-unsigned-wrap contract without a signed one or an
  /// enclosing positive proof, so the counted form must compare unsigned.
  bool unsignedComparison = false;
};

/// Projects the closed post-tested shape emitted for a finite latch-tested
/// counted loop. A dynamic upper bound is accepted only for the zero-based,
/// unit-step shape when an enclosing true branch proves that bound strictly
/// positive, or when the induction update's no-wrap contract proves the
/// landing sequence: a wrapping update would be poison at the latch, so a
/// defined execution reaches the bound exactly, signed under `nsw` and
/// unsigned under `nuw`. Unknown, wrapping, non-landing, side-effecting
/// after-region, or non-ordinal feedback shapes return no projection.
std::optional<ExactPostTestedCountedLoopProjection>
projectExactPostTestedCountedLoop(mlir::scf::WhileOp loop);

/// Folds one integer value to a constant. A caller that owns a stronger
/// folding than literal matching supplies it here.
using ConstantIntegerValue =
    llvm::function_ref<std::optional<llvm::APInt>(mlir::Value)>;

/// One zero-based unit-step counted loop whose body runs exactly
/// `iterationCount` times with `induction` taking every value in
/// `[0, iterationCount)`. The structured `scf.for` form and the post-tested
/// `scf.while` form project to this same shape.
struct UnitStrideCountedLoop final {
  mlir::BlockArgument induction;
  std::uint64_t iterationCount = 0;
};

std::optional<UnitStrideCountedLoop>
projectUnitStrideCountedLoop(mlir::Operation *loop,
                             ConstantIntegerValue constantValue);

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H
