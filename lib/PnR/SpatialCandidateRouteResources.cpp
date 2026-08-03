#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <system_error>

using namespace loom::pnr;

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error increment(PnrIndex &value, PnrIndex amount,
                      llvm::StringRef subject) {
  if (amount > std::numeric_limits<PnrIndex>::max() - value)
    return candidateError(subject + " count overflows PnrIndex");
  value += amount;
  return llvm::Error::success();
}

} // namespace

llvm::Error SpatialMoveTransaction::collectRouteTraversalDeltas() {
  if (routeDeltasCollected_)
    return llvm::Error::success();
  std::uint64_t proposedUnroutedObligationCount =
      state_->unroutedObligationCount_;
  for (PnrIndex logicalNet : scratch_->touchedRoutes_) {
    RouteTreeTransaction &route = *scratch_->routeTransactions_[logicalNet];
    auto deltas = route.prepare();
    if (!deltas) {
      rollbackAppliedRouteResources();
      return deltas.takeError();
    }
    const std::uint64_t sinkCount =
        state_->problem_->transfers().logicalNets()[logicalNet].sinkCount;
    if (!route.initiallyRouted() && route.proposedRouted()) {
      if (proposedUnroutedObligationCount < sinkCount) {
        rollbackAppliedRouteResources();
        return candidateError("unrouted obligation count underflows u64");
      }
      proposedUnroutedObligationCount -= sinkCount;
    } else if (route.initiallyRouted() && !route.proposedRouted()) {
      if (sinkCount > std::numeric_limits<std::uint64_t>::max() -
                          proposedUnroutedObligationCount) {
        rollbackAppliedRouteResources();
        return candidateError("unrouted obligation count overflows u64");
      }
      proposedUnroutedObligationCount += sinkCount;
    }
    for (const RouteTreeTraversalDelta &delta : *deltas) {
      if (llvm::Error error = state_->routeResources_.applyTraversalDelta(
              logicalNet, delta.traversal, delta.removed, delta.added)) {
        rollbackAppliedRouteResources();
        return error;
      }
      ++scratch_->resourcePartiallyAppliedDeltaCount_;
      const PnrIndex traversal = delta.traversal;
      if (traversal >= scratch_->traversalDeltaMarks_.size()) {
        rollbackAppliedRouteResources();
        return candidateError("route selected an out-of-range traversal");
      }
      if (scratch_->traversalDeltaMarks_[traversal] !=
          scratch_->traversalEpoch_) {
        scratch_->traversalDeltaMarks_[traversal] = scratch_->traversalEpoch_;
        scratch_->traversalRemoved_[traversal] = 0;
        scratch_->traversalAdded_[traversal] = 0;
        scratch_->touchedTraversals_.push_back(traversal);
      }
      if (llvm::Error error =
              increment(scratch_->traversalRemoved_[traversal], delta.removed,
                        "route traversal removal")) {
        rollbackAppliedRouteResources();
        return error;
      }
      if (llvm::Error error =
              increment(scratch_->traversalAdded_[traversal], delta.added,
                        "route traversal addition")) {
        rollbackAppliedRouteResources();
        return error;
      }
    }
    ++scratch_->resourceFullyAppliedRouteCount_;
    scratch_->resourcePartiallyAppliedDeltaCount_ = 0;
  }

  llvm::sort(scratch_->touchedTraversals_);
  for (PnrIndex traversal : scratch_->touchedTraversals_) {
    PnrIndex &removed = scratch_->traversalRemoved_[traversal];
    PnrIndex &added = scratch_->traversalAdded_[traversal];
    const PnrIndex cancelled = std::min(removed, added);
    removed -= cancelled;
    added -= cancelled;
    if (removed != 0)
      if (llvm::Error error =
              scratch_->handshakeTransaction_->removeTraversalUses(traversal,
                                                                   removed)) {
        rollbackAppliedRouteResources();
        return error;
      }
  }
  for (PnrIndex traversal : scratch_->touchedTraversals_) {
    const PnrIndex added = scratch_->traversalAdded_[traversal];
    if (added != 0)
      if (llvm::Error error = scratch_->handshakeTransaction_->addTraversalUses(
              traversal, added)) {
        rollbackAppliedRouteResources();
        return error;
      }
  }
  state_->unroutedObligationCount_ = proposedUnroutedObligationCount;
  routeViolationApplied_ = true;
  routeDeltasCollected_ = true;
  return llvm::Error::success();
}

void SpatialMoveTransaction::rollbackAppliedRouteResources() noexcept {
  if (!scratch_)
    return;
  if (routeViolationApplied_) {
    state_->unroutedObligationCount_ = initialUnroutedObligationCount_;
    routeViolationApplied_ = false;
  }
  const std::size_t full = scratch_->resourceFullyAppliedRouteCount_;
  if (scratch_->resourcePartiallyAppliedDeltaCount_ != 0) {
    assert(full < scratch_->touchedRoutes_.size());
    const PnrIndex logicalNet = scratch_->touchedRoutes_[full];
    const auto deltas =
        llvm::cantFail(scratch_->routeTransactions_[logicalNet]->prepare());
    for (std::size_t index = scratch_->resourcePartiallyAppliedDeltaCount_;
         index != 0; --index) {
      const RouteTreeTraversalDelta &delta = deltas[index - 1];
      state_->routeResources_.revertTraversalDelta(logicalNet, delta.traversal,
                                                   delta.removed, delta.added);
    }
  }
  for (std::size_t route = full; route != 0; --route) {
    const PnrIndex logicalNet = scratch_->touchedRoutes_[route - 1];
    const auto deltas =
        llvm::cantFail(scratch_->routeTransactions_[logicalNet]->prepare());
    for (std::size_t index = deltas.size(); index != 0; --index) {
      const RouteTreeTraversalDelta &delta = deltas[index - 1];
      state_->routeResources_.revertTraversalDelta(logicalNet, delta.traversal,
                                                   delta.removed, delta.added);
    }
  }
  scratch_->resourceFullyAppliedRouteCount_ = 0;
  scratch_->resourcePartiallyAppliedDeltaCount_ = 0;
  routeDeltasCollected_ = false;
}

void SpatialMoveTransaction::acceptAppliedRouteResources() noexcept {
  scratch_->resourceFullyAppliedRouteCount_ = 0;
  scratch_->resourcePartiallyAppliedDeltaCount_ = 0;
  routeDeltasCollected_ = false;
  routeViolationApplied_ = false;
}
