#include "CGRAPhysicalActionRuntime.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Expected<SpatialEventCoordinate>
addCycles(const SpatialEventCoordinate &coordinate, std::uint32_t cycles) {
  const auto ratio = coordinate.referenceCycle;
  const __uint128_t numerator =
      static_cast<__uint128_t>(ratio.numerator()) +
      static_cast<__uint128_t>(cycles) * ratio.denominator();
  if (numerator > std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "CGRA event coordinate overflows u64");
  auto advanced = ::loom::evaluation::ExactRatio::get(
      static_cast<std::uint64_t>(numerator), ratio.denominator());
  if (!advanced)
    return advanced.takeError();
  return SpatialEventCoordinate{*advanced, coordinate.delta};
}

std::uint64_t encodePayload(std::uint64_t slot,
                            CgraPhysicalActionRuntime::InternalKind kind) {
  return (slot << 2) | static_cast<std::uint64_t>(kind);
}

std::pair<std::uint64_t, CgraPhysicalActionRuntime::InternalKind>
decodePayload(std::uint64_t payload) {
  return {payload >> 2,
          static_cast<CgraPhysicalActionRuntime::InternalKind>(payload & 3)};
}

} // namespace

llvm::Expected<CgraPhysicalActionRuntime>
CgraPhysicalActionRuntime::create(const CgraResourceRuntimePlan &resources,
                                  llvm::ArrayRef<CgraPhysicalUseTiming> uses) {
  if (uses.size() != resources.selectedUses.size())
    return invalid(
        "CGRA physical timing must cover every selected resource use");
  std::vector<CgraPhysicalUseTiming> frozen(uses.begin(), uses.end());
  for (auto [ordinal, use] : llvm::enumerate(frozen)) {
    if (use.selectedUseOrdinal != ordinal)
      return invalid("CGRA physical timing is not in selected-use order");
    if (use.commitRank && (use.acquireRank > *use.commitRank ||
                           *use.commitRank > use.releaseRank))
      return invalid("CGRA physical action commit rank is out of order");
    if (use.acquireRank > use.releaseRank)
      return invalid("CGRA physical action release precedes acquisition");
    if (use.commitRank.has_value() != use.commitEventOrdinal.has_value())
      return invalid("CGRA physical action commit timing is incomplete");
  }
  auto runtime = CgraResourceRuntime::create(resources);
  if (!runtime)
    return runtime.takeError();
  return CgraPhysicalActionRuntime(std::move(frozen), std::move(*runtime));
}

llvm::Error
CgraPhysicalActionRuntime::schedule(std::uint64_t actionSlot, InternalKind kind,
                                    SpatialEventCoordinate coordinate,
                                    std::uint32_t ownerEventOrdinal) {
  if (actionSlot > std::numeric_limits<std::uint64_t>::max() >> 2)
    return invalid("CGRA physical action slot exceeds event payload domain");
  const Action &action = actions_[actionSlot];
  events_.schedule(
      CgraScheduledEvent{{std::move(coordinate), action.actionOrdinal,
                          action.occurrenceOrdinal, ownerEventOrdinal},
                         encodePayload(actionSlot, kind)});
  return llvm::Error::success();
}

llvm::Expected<CgraPhysicalLifecycleEvent>
CgraPhysicalActionRuntime::request(std::uint64_t actionOrdinal,
                                   std::uint64_t occurrenceOrdinal,
                                   SpatialEventCoordinate coordinate) {
  const CgraPhysicalActionRequest request{actionOrdinal, occurrenceOrdinal};
  auto requested = requestBatch(llvm::ArrayRef(request), std::move(coordinate));
  if (!requested)
    return requested.takeError();
  return std::move(requested->front());
}

llvm::Expected<std::vector<CgraPhysicalLifecycleEvent>>
CgraPhysicalActionRuntime::requestBatch(
    llvm::ArrayRef<CgraPhysicalActionRequest> requests,
    SpatialEventCoordinate coordinate) {
  if (lastCoordinate_ &&
      compareSpatialEventCoordinates(coordinate, *lastCoordinate_) < 0)
    return invalid("CGRA physical request precedes the execution calendar");
  if (requests.empty())
    return std::vector<CgraPhysicalLifecycleEvent>{};
  if (requests.size() >
      std::numeric_limits<std::uint64_t>::max() - activeActionCount_)
    return invalid("CGRA active physical action count exceeds u64");
  const std::size_t newSlots = requests.size() > freeActionSlots_.size()
                                   ? requests.size() - freeActionSlots_.size()
                                   : 0;
  if (newSlots != 0 &&
      (actions_.size() > (std::numeric_limits<std::uint64_t>::max() >> 2) ||
       newSlots - 1 >
           (std::numeric_limits<std::uint64_t>::max() >> 2) - actions_.size()))
    return invalid("CGRA physical action slot exceeds event payload domain");

  llvm::DenseSet<std::pair<std::uint64_t, std::uint64_t>> unique;
  unique.reserve(requests.size());
  std::vector<SpatialEventCoordinate> acquisitions;
  acquisitions.reserve(requests.size());
  for (const CgraPhysicalActionRequest &request : requests) {
    if (request.actionOrdinal >= uses_.size())
      return invalid("CGRA physical request names an unknown action");
    const auto key =
        std::make_pair(request.actionOrdinal, request.occurrenceOrdinal);
    if (activeActions_.contains(key) || !unique.insert(key).second)
      return invalid("CGRA physical action occurrence is already active");
    auto acquire =
        addCycles(coordinate, uses_[request.actionOrdinal].acquireRank);
    if (!acquire)
      return acquire.takeError();
    acquisitions.push_back(std::move(*acquire));
  }

  std::vector<CgraPhysicalLifecycleEvent> result;
  result.reserve(requests.size());
  for (auto [request, acquire] : llvm::zip(requests, acquisitions)) {
    std::uint64_t slot = 0;
    if (freeActionSlots_.empty()) {
      slot = actions_.size();
      actions_.push_back(
          Action{request.actionOrdinal, request.occurrenceOrdinal,
                 ActionState::Requested, std::nullopt, false, false});
    } else {
      slot = freeActionSlots_.back();
      freeActionSlots_.pop_back();
      actions_[slot] = Action{request.actionOrdinal,
                              request.occurrenceOrdinal,
                              ActionState::Requested,
                              std::nullopt,
                              false,
                              false};
    }
    const auto key =
        std::make_pair(request.actionOrdinal, request.occurrenceOrdinal);
    const bool inserted = activeActions_.try_emplace(key, slot).second;
    assert(inserted && "preflighted physical action must be unique");
    ++activeActionCount_;
    const CgraPhysicalUseTiming &use = uses_[request.actionOrdinal];
    events_.schedule(
        CgraScheduledEvent{{acquire, request.actionOrdinal,
                            request.occurrenceOrdinal, use.acquireEventOrdinal},
                           encodePayload(slot, InternalKind::Acquire)});
    result.push_back({CgraPhysicalLifecycleKind::Requested,
                      request.actionOrdinal, request.occurrenceOrdinal,
                      use.acquireEventOrdinal, coordinate});
  }
  llvm::sort(result, [](const CgraPhysicalLifecycleEvent &lhs,
                        const CgraPhysicalLifecycleEvent &rhs) {
    return std::tie(lhs.actionOrdinal, lhs.occurrenceOrdinal,
                    lhs.ownerEventOrdinal) < std::tie(rhs.actionOrdinal,
                                                      rhs.occurrenceOrdinal,
                                                      rhs.ownerEventOrdinal);
  });
  return result;
}

llvm::Error CgraPhysicalActionRuntime::satisfyCausalRelease(
    std::uint64_t actionOrdinal, std::uint64_t occurrenceOrdinal,
    SpatialEventCoordinate coordinate) {
  if (actionOrdinal >= uses_.size())
    return invalid("CGRA causal release names an unknown action");
  if (!uses_[actionOrdinal].requiresCausalRelease)
    return invalid("CGRA physical action has no causal release condition");
  if (lastCoordinate_ &&
      compareSpatialEventCoordinates(coordinate, *lastCoordinate_) < 0)
    return invalid("CGRA causal release precedes the execution calendar");
  auto found = activeActions_.find({actionOrdinal, occurrenceOrdinal});
  if (found == activeActions_.end())
    return invalid("CGRA causal release names an inactive action");
  Action &action = actions_[found->second];
  if (action.state != ActionState::Granted || !action.envelope)
    return invalid("CGRA causal release precedes resource grant");
  if (action.causalReleaseReached)
    return invalid("CGRA physical action received causal release twice");
  if (action.intrinsicReleaseReached) {
    auto release = nextSpatialDelta(coordinate);
    if (!release)
      return release.takeError();
    if (llvm::Error error =
            schedule(found->second, InternalKind::Release, *release,
                     uses_[actionOrdinal].releaseEventOrdinal))
      return error;
  }
  action.causalReleaseReached = true;
  return llvm::Error::success();
}

llvm::Expected<std::optional<CgraPhysicalLifecycleFrame>>
CgraPhysicalActionRuntime::advance() {
  auto next = events_.popNextFrameView();
  if (!next)
    return next.takeError();
  if (!*next)
    return std::optional<CgraPhysicalLifecycleFrame>{};

  const CgraEventFrameView internal = **next;
  lastCoordinate_ = internal.coordinate;
  CgraPhysicalLifecycleFrame result{internal.coordinate, {}};

  struct Due final {
    std::uint64_t slot = 0;
    InternalKind kind = InternalKind::Acquire;
    std::uint32_t ownerEventOrdinal = 0;
  };
  std::vector<Due> due;
  due.reserve(internal.events.size());
  for (const CgraScheduledEvent &event : internal.events) {
    auto [slot, kind] = decodePayload(event.payload);
    if (slot >= actions_.size())
      return invalid("CGRA physical event names an unknown action slot");
    due.push_back({slot, kind, event.order.ownerEventOrdinal});
  }

  llvm::stable_sort(due, [](const Due &lhs, const Due &rhs) {
    return std::make_tuple(lhs.kind, lhs.slot, lhs.ownerEventOrdinal) <
           std::make_tuple(rhs.kind, rhs.slot, rhs.ownerEventOrdinal);
  });

  std::vector<CgraResourceRequest> requests;
  std::vector<std::uint64_t> requestSlots;
  for (const Due &event : due) {
    Action &action = actions_[event.slot];
    const CgraPhysicalUseTiming &use = uses_[action.actionOrdinal];
    switch (event.kind) {
    case InternalKind::Commit:
      if (action.state != ActionState::Granted)
        return invalid("CGRA physical commit precedes resource grant");
      result.events.push_back({CgraPhysicalLifecycleKind::Committed,
                               action.actionOrdinal, action.occurrenceOrdinal,
                               event.ownerEventOrdinal, result.coordinate});
      break;
    case InternalKind::Release:
      if (action.state != ActionState::Granted || !action.envelope)
        return invalid("CGRA physical release has no active claim envelope");
      action.intrinsicReleaseReached = true;
      if (use.requiresCausalRelease && !action.causalReleaseReached)
        break;
      if (llvm::Error error = resources_.release(*action.envelope))
        return std::move(error);
      action.state = ActionState::Retired;
      action.envelope.reset();
      activeActions_.erase(
          std::make_pair(action.actionOrdinal, action.occurrenceOrdinal));
      --activeActionCount_;
      freeActionSlots_.push_back(event.slot);
      result.events.push_back({CgraPhysicalLifecycleKind::Retired,
                               action.actionOrdinal, action.occurrenceOrdinal,
                               event.ownerEventOrdinal, result.coordinate});
      break;
    case InternalKind::CommitRelease:
      if (action.state != ActionState::Granted || !action.envelope)
        return invalid(
            "CGRA atomic commit/release has no active claim envelope");
      result.events.push_back({CgraPhysicalLifecycleKind::Committed,
                               action.actionOrdinal, action.occurrenceOrdinal,
                               event.ownerEventOrdinal, result.coordinate});
      action.intrinsicReleaseReached = true;
      if (use.requiresCausalRelease && !action.causalReleaseReached)
        break;
      if (llvm::Error error = resources_.release(*action.envelope))
        return std::move(error);
      action.state = ActionState::Retired;
      action.envelope.reset();
      activeActions_.erase(
          std::make_pair(action.actionOrdinal, action.occurrenceOrdinal));
      --activeActionCount_;
      freeActionSlots_.push_back(event.slot);
      result.events.push_back({CgraPhysicalLifecycleKind::Retired,
                               action.actionOrdinal, action.occurrenceOrdinal,
                               event.ownerEventOrdinal, result.coordinate});
      break;
    case InternalKind::Acquire:
      if (action.state != ActionState::Requested)
        return invalid("CGRA physical acquisition has invalid action state");
      requests.push_back({use.selectedUseOrdinal, action.occurrenceOrdinal});
      requestSlots.push_back(event.slot);
      break;
    }
  }

  if (!requests.empty()) {
    llvm::SmallVector<CgraResourceGrant, 8> grants;
    if (llvm::Error error = resources_.grant(requests, grants))
      return std::move(error);
    llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, CgraClaimEnvelope>
        granted;
    granted.reserve(grants.size());
    for (const CgraResourceGrant &grant : grants)
      granted.try_emplace(
          std::make_pair(grant.selectedUseOrdinal, grant.occurrenceOrdinal),
          grant.claimEnvelope);

    for (std::uint64_t slot : requestSlots) {
      Action &action = actions_[slot];
      const CgraPhysicalUseTiming &use = uses_[action.actionOrdinal];
      auto accepted = granted.find(
          std::make_pair(use.selectedUseOrdinal, action.occurrenceOrdinal));
      if (accepted == granted.end()) {
        auto retry = addCycles(result.coordinate, 1);
        if (!retry)
          return retry.takeError();
        if (llvm::Error error = schedule(slot, InternalKind::Acquire, *retry,
                                         use.acquireEventOrdinal))
          return std::move(error);
        continue;
      }

      action.state = ActionState::Granted;
      action.envelope = accepted->second;
      result.events.push_back({CgraPhysicalLifecycleKind::Granted,
                               action.actionOrdinal, action.occurrenceOrdinal,
                               use.acquireEventOrdinal, result.coordinate});
      const bool combinedCommitRelease =
          use.commitRank && *use.commitRank == use.releaseRank &&
          *use.commitEventOrdinal == use.releaseEventOrdinal;
      if (combinedCommitRelease) {
        auto commitRelease =
            addCycles(result.coordinate, *use.commitRank - use.acquireRank);
        if (!commitRelease)
          return commitRelease.takeError();
        if (llvm::Error error =
                schedule(slot, InternalKind::CommitRelease, *commitRelease,
                         *use.commitEventOrdinal))
          return std::move(error);
      } else if (use.commitRank) {
        auto commit =
            addCycles(result.coordinate, *use.commitRank - use.acquireRank);
        if (!commit)
          return commit.takeError();
        if (llvm::Error error = schedule(slot, InternalKind::Commit, *commit,
                                         *use.commitEventOrdinal))
          return std::move(error);
      }
      if (!combinedCommitRelease) {
        auto release =
            addCycles(result.coordinate, use.releaseRank - use.acquireRank);
        if (!release)
          return release.takeError();
        if (llvm::Error error = schedule(slot, InternalKind::Release, *release,
                                         use.releaseEventOrdinal))
          return std::move(error);
      }
    }
  }

  llvm::stable_sort(result.events, [](const CgraPhysicalLifecycleEvent &lhs,
                                      const CgraPhysicalLifecycleEvent &rhs) {
    return std::tie(lhs.actionOrdinal, lhs.occurrenceOrdinal,
                    lhs.ownerEventOrdinal, lhs.kind) <
           std::tie(rhs.actionOrdinal, rhs.occurrenceOrdinal,
                    rhs.ownerEventOrdinal, rhs.kind);
  });
  return std::optional<CgraPhysicalLifecycleFrame>(std::move(result));
}

} // namespace loom::sim::detail
