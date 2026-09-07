#include "PnR/SpatialCandidateState.h"

#include "SpatialProgressIndex.h"

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
  if (llvm::Error error = synchronizeProgressTraversalDeltas()) {
    rollbackAppliedRouteResources();
    return error;
  }
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
    if (state_->usesRegisterFifo(logicalNet)) {
      if (route.proposedRouted()) {
        rollbackAppliedRouteResources();
        return candidateError(
            "register-FIFO transfer also has a proposed external route");
      }
    } else if (!route.initiallyRouted() && route.proposedRouted()) {
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
  rollbackProgressProjection();
}

void SpatialMoveTransaction::acceptAppliedRouteResources() noexcept {
  scratch_->resourceFullyAppliedRouteCount_ = 0;
  scratch_->resourcePartiallyAppliedDeltaCount_ = 0;
  routeDeltasCollected_ = false;
  routeViolationApplied_ = false;
  acceptProgressProjection();
}

llvm::Error SpatialMoveTransaction::applyProgressTraversalDelta(
    PnrIndex logicalNet, PnrIndex traversal, PnrIndex removed,
    PnrIndex added) {
  if (traversal >= state_->problem_->routing().traversals().size())
    return candidateError("progress traversal is out of range");
  if (state_->problem_->progressIndex().traversalOwner(traversal) ==
      getInvalidPnrIndex())
    return llvm::Error::success();
  if (llvm::Error error = state_->progressState_.applyTraversalDelta(
          logicalNet, traversal, removed, added))
    return error;
  scratch_->progressTraversalDeltas_.push_back(
      {logicalNet, traversal, removed, added});
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::synchronizeProgressTraversalDeltas() {
  for (PnrIndex logicalNet : scratch_->touchedRoutes_) {
    RouteTreeTransaction &route = *scratch_->routeTransactions_[logicalNet];
    const auto deltas = route.recordedTraversalDeltas();
    std::size_t &applied =
        scratch_->progressRecordedRouteDeltaCounts_[logicalNet];
    if (scratch_->progressRecordedRouteDeltaEpochs_[logicalNet] !=
        scratch_->progressRecordedRouteDeltaEpoch_) {
      applied = 0;
      scratch_->progressRecordedRouteDeltaEpochs_[logicalNet] =
          scratch_->progressRecordedRouteDeltaEpoch_;
    }
    if (applied > deltas.size())
      return candidateError(
          "RouteTree progress journal lost recorded traversal deltas");
    for (const RouteTreeTraversalDelta &delta : deltas.drop_front(applied)) {
      if (llvm::Error error = applyProgressTraversalDelta(
              logicalNet, delta.traversal, delta.removed, delta.added))
        return error;
      ++applied;
    }
  }

  for (PnrIndex logicalNet : scratch_->progressDirtyNets_) {
    const bool desiredTerminalActive =
        !state_->usesRegisterFifo(logicalNet) &&
        state_->routeTrees_[logicalNet]->isRouted();
    const bool currentTerminalActive =
        scratch_->progressTerminalActive_[logicalNet] != 0;
    if (llvm::Error error = changeProgressTerminalSelections(
            logicalNet, currentTerminalActive, desiredTerminalActive))
      return error;
    scratch_->progressTerminalActive_[logicalNet] = desiredTerminalActive;
  }
  return llvm::Error::success();
}

llvm::Expected<SpatialMoveRouteSavepoint>
SpatialMoveTransaction::saveRoutes() const {
  if (llvm::Error error = ensureCollecting())
    return std::move(error);
  if (routeDeltasCollected_)
    return candidateError("cannot savepoint collected route deltas");
  SpatialMoveRouteSavepoint savepoint;
  savepoint.touchedRouteCount = scratch_->touchedRoutes_.size();
  savepoint.progressTraversalDeltaCount =
      scratch_->progressTraversalDeltas_.size();
  savepoint.progressDirtyNetCount = scratch_->progressDirtyNets_.size();
  savepoint.decisionDeltaCount = scratch_->decisionDeltas_.size();
  savepoint.routes.reserve(savepoint.touchedRouteCount);
  savepoint.progressRecordedRouteDeltaCounts.reserve(
      savepoint.touchedRouteCount);
  for (PnrIndex logicalNet : scratch_->touchedRoutes_) {
    auto route = scratch_->routeTransactions_[logicalNet]->savepoint();
    if (!route)
      return route.takeError();
    savepoint.routes.push_back(std::move(*route));
    savepoint.progressRecordedRouteDeltaCounts.push_back(
        scratch_->progressRecordedRouteDeltaEpochs_[logicalNet] ==
                scratch_->progressRecordedRouteDeltaEpoch_
            ? scratch_->progressRecordedRouteDeltaCounts_[logicalNet]
            : 0);
  }
  savepoint.progressTerminalActive.reserve(savepoint.progressDirtyNetCount);
  for (PnrIndex logicalNet : scratch_->progressDirtyNets_)
    savepoint.progressTerminalActive.push_back(
        scratch_->progressTerminalActive_[logicalNet]);
  return savepoint;
}

llvm::Error
SpatialMoveTransaction::restoreRoutes(SpatialMoveRouteSavepoint &&savepoint) {
  if (llvm::Error error = ensureCollecting())
    return error;
  if (routeDeltasCollected_)
    return candidateError("cannot restore collected route deltas");
  if (savepoint.touchedRouteCount > scratch_->touchedRoutes_.size() ||
      savepoint.routes.size() != savepoint.touchedRouteCount ||
      savepoint.progressRecordedRouteDeltaCounts.size() !=
          savepoint.touchedRouteCount ||
      savepoint.progressTraversalDeltaCount >
          scratch_->progressTraversalDeltas_.size() ||
      savepoint.progressDirtyNetCount > scratch_->progressDirtyNets_.size() ||
      savepoint.progressTerminalActive.size() !=
          savepoint.progressDirtyNetCount)
    return candidateError("route savepoint lies after the current move");
  if (savepoint.decisionDeltaCount != scratch_->decisionDeltas_.size())
    return candidateError("route savepoint cannot span decision changes");
  if (!scratch_->progressDependencyDeltas_.empty())
    return candidateError(
        "route savepoint cannot span a progress dependency projection");

  for (std::size_t index = scratch_->progressTraversalDeltas_.size();
       index != savepoint.progressTraversalDeltaCount; --index) {
    const auto &delta = scratch_->progressTraversalDeltas_[index - 1];
    state_->progressState_.revertTraversalDelta(
        delta.logicalNet, delta.traversal, delta.removed, delta.added);
  }
  scratch_->progressTraversalDeltas_.resize(
      savepoint.progressTraversalDeltaCount);
  for (std::size_t index = scratch_->progressDirtyNets_.size();
       index != savepoint.progressDirtyNetCount; --index)
    scratch_->progressDirtyNetMarks_[scratch_->progressDirtyNets_[index - 1]] =
        0;
  scratch_->progressDirtyNets_.resize(savepoint.progressDirtyNetCount);
  for (auto [ordinal, logicalNet] :
       llvm::enumerate(scratch_->progressDirtyNets_))
    scratch_->progressTerminalActive_[logicalNet] =
        savepoint.progressTerminalActive[ordinal];

  for (std::size_t index = scratch_->touchedRoutes_.size();
       index != savepoint.touchedRouteCount; --index) {
    const PnrIndex logicalNet = scratch_->touchedRoutes_[index - 1];
    scratch_->routeTransactions_[logicalNet]->rollback();
    scratch_->routeTransactions_[logicalNet].reset();
    scratch_->progressRecordedRouteDeltaCounts_[logicalNet] = 0;
  }
  scratch_->touchedRoutes_.resize(savepoint.touchedRouteCount);
  for (std::size_t index = savepoint.touchedRouteCount; index != 0; --index) {
    const PnrIndex logicalNet = scratch_->touchedRoutes_[index - 1];
    if (llvm::Error error = scratch_->routeTransactions_[logicalNet]->rollbackTo(
            std::move(savepoint.routes[index - 1])))
      return error;
    scratch_->progressRecordedRouteDeltaCounts_[logicalNet] =
        savepoint.progressRecordedRouteDeltaCounts[index - 1];
    scratch_->progressRecordedRouteDeltaEpochs_[logicalNet] =
        scratch_->progressRecordedRouteDeltaEpoch_;
  }
  return llvm::Error::success();
}

llvm::Error SpatialMoveTransaction::synchronizeProgressProjection() {
  for (PnrIndex logicalNet : scratch_->progressDirtyNets_) {
    const bool firstProjection =
        scratch_->progressDependencyJournalMarks_[logicalNet] !=
        scratch_->decisionEpoch_;
    const RouteTreeState *progressRoute =
        state_->routeTrees_[logicalNet].get();
    if (scratch_->routeTransactions_[logicalNet]) {
      auto prepared =
          scratch_->routeTransactions_[logicalNet]->preparedState();
      if (!prepared)
        return prepared.takeError();
      progressRoute = *prepared;
    }
    auto oldCapacity =
        state_->progressState_.replaceLogicalNetCapacityProjection(
            *state_, logicalNet, progressRoute);
    if (!oldCapacity)
      return oldCapacity.takeError();
    if (firstProjection) {
      scratch_->progressDependencyJournalMarks_[logicalNet] =
          scratch_->decisionEpoch_;
      scratch_->progressDependencyDeltas_.push_back(
          {logicalNet,
           state_->progressState_
               .logicalNetRouteDependencyViolationCount(logicalNet),
           std::move(*oldCapacity)});
    }
    if (llvm::Error error =
            state_->progressState_.refreshLogicalNetRouteDependencies(
                *state_, logicalNet))
      return error;
  }
  for (PnrIndex logicalNet : scratch_->progressDirtyNets_)
    scratch_->progressDirtyNetMarks_[logicalNet] = 0;
  scratch_->progressDirtyNets_.clear();
  return llvm::Error::success();
}

void SpatialMoveTransaction::rollbackProgressProjection() noexcept {
  if (!scratch_)
    return;
  for (const SpatialCandidateScratch::ProgressDependencyDelta &delta :
       llvm::reverse(scratch_->progressDependencyDeltas_)) {
    state_->progressState_.restoreLogicalNetRouteDependencyCount(
        delta.logicalNet, delta.oldCount);
    state_->progressState_.restoreLogicalNetCapacityProjection(
        delta.logicalNet, delta.oldCapacityProjection);
  }
  for (const SpatialCandidateScratch::ProgressTraversalDelta &delta :
       llvm::reverse(scratch_->progressTraversalDeltas_))
    state_->progressState_.revertTraversalDelta(
        delta.logicalNet, delta.traversal, delta.removed, delta.added);
  for (PnrIndex logicalNet : scratch_->progressDirtyNets_)
    scratch_->progressDirtyNetMarks_[logicalNet] = 0;
  scratch_->progressDirtyNets_.clear();
  scratch_->advanceProgressRouteDeltaEpoch();
  scratch_->progressTraversalDeltas_.clear();
  scratch_->progressDependencyDeltas_.clear();
}

void SpatialMoveTransaction::acceptProgressProjection() noexcept {
  for (PnrIndex logicalNet : scratch_->progressDirtyNets_)
    scratch_->progressDirtyNetMarks_[logicalNet] = 0;
  scratch_->progressDirtyNets_.clear();
  scratch_->advanceProgressRouteDeltaEpoch();
  scratch_->progressTraversalDeltas_.clear();
  scratch_->progressDependencyDeltas_.clear();
}
