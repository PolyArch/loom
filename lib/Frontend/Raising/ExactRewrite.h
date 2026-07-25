#ifndef LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H
#define LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H

#include "CallableRegions.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace raising {

// Offer `patterns` once to each operation already present in `region`.
//
// The greedy driver is deliberately not used here. It unconditionally erases
// unreachable blocks, folds, CSEs constants and iterates to a fixed point, and
// none of those decisions belong to a pass whose contract is a single exact
// respelling of the operations it declares. Only operations that exist before
// the walk are offered a pattern, so a replacement is never revisited, and an
// unreachable block survives until the structuring pass removes it explicitly.
//
// The walk is the callable-ownership walk, so a nested callable's body is
// never offered patterns here: it is owned by that callable and processed by
// its own region-level application.
inline void
applyExactPatternsOnce(::mlir::Region &region,
                       const ::mlir::FrozenRewritePatternSet &patterns) {
  ::llvm::SmallVector<::mlir::Operation *> candidates;
  forEachOwnedOperation(region, [&](::mlir::Operation *op) {
    candidates.push_back(op);
    return ::mlir::WalkResult::advance();
  });

  ::mlir::PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();
  ::mlir::PatternRewriter rewriter(region.getContext());
  for (::mlir::Operation *op : candidates)
    (void)applicator.matchAndRewrite(op, rewriter);
}

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H
