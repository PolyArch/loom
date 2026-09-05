#ifndef LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H
#define LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H

#include "Frontend/Analysis/CallableRegions.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace raising {
namespace detail {

inline void collectOwnedOperationsPostOrder(
    ::mlir::Region &region,
    ::llvm::SmallVectorImpl<::mlir::Operation *> &candidates) {
  for (::mlir::Block &block : region) {
    for (::mlir::Operation &op : block) {
      if (::loom::frontend::analysis::isCallableOp(&op))
        continue;
      for (::mlir::Region &nested : op.getRegions())
        collectOwnedOperationsPostOrder(nested, candidates);
      candidates.push_back(&op);
    }
  }
}

} // namespace detail

// Offer `patterns` once to each operation already present in `region`.
//
// The greedy driver is deliberately not used here. It unconditionally erases
// unreachable blocks, folds, CSEs constants and iterates to a fixed point, and
// none of those decisions belong to a pass whose contract is a single exact
// respelling of the operations it declares. Only operations that exist before
// the walk are offered a pattern, so a replacement is never revisited, and an
// unreachable block survives until the structuring pass removes it explicitly.
//
// The snapshot is descendant-first. A pattern may replace an operation that
// owns regions, so visiting an ancestor first could erase descendants whose
// pointers were already saved. Processing every descendant before its owning
// operation keeps the remaining snapshot valid. Nested callable bodies are
// pruned because they are processed by their own region-level application.
inline void
applyExactPatternsOnce(::mlir::Region &region,
                       const ::mlir::FrozenRewritePatternSet &patterns) {
  ::llvm::SmallVector<::mlir::Operation *> candidates;
  detail::collectOwnedOperationsPostOrder(region, candidates);

  ::mlir::PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();
  ::mlir::PatternRewriter rewriter(region.getContext());
  for (::mlir::Operation *op : candidates)
    (void)applicator.matchAndRewrite(op, rewriter);
}

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_EXACTREWRITE_H
