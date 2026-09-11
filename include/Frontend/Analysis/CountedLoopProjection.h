#ifndef LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H
#define LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APInt.h"

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

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_COUNTEDLOOPPROJECTION_H
