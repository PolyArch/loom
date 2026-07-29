//===- DFGSimulatorPlainMemory.cpp - Plain-memory conflict authority ------===//
//
// One owner for the runtime conflict question a plain access asks: which byte
// ranges an issued access covered, whether a ready set overlaps itself, and
// whether an already issued hazard is covered by the ctrl order a candidate
// carries. Admission decides only that question. Legality a finalized program
// already proved, such as the distinctness of a plain scatter's active
// destinations, is not re-derived here.
//
// The index stores effect handles and no relation between them;
// MemorySynchronization remains the only authority that orders two effects.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace loom::sim;
using namespace loom::sim::detail;

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

void canonicalizeMemoryActionRanges(
    llvm::SmallVectorImpl<std::pair<std::int64_t, std::int64_t>> &ranges) {
  llvm::sort(ranges);
  std::size_t output = 0;
  for (std::size_t input = 0; input < ranges.size(); ++input) {
    const auto [begin, end] = ranges[input];
    if (begin >= end)
      continue;
    if (output == 0 || ranges[output - 1].second < begin) {
      ranges[output++] = {begin, end};
      continue;
    }
    ranges[output - 1].second = std::max(ranges[output - 1].second, end);
  }
  ranges.resize(output);
}

// True when one ready set overlaps itself on a hazard, which is exactly the
// conflict no explicit order can resolve within one scheduler decision. The
// ranges of a single action are canonical, so an action never conflicts with
// itself here.
static bool readyPlainMemoryActionsConflict(
    llvm::ArrayRef<ReadyPlainMemoryAction> actions) {
  struct Range {
    std::uint64_t rootId;
    std::int64_t begin;
    std::int64_t end;
    bool isWrite;
  };

  llvm::SmallVector<Range, 8> ranges;
  for (const ReadyPlainMemoryAction &ready : actions) {
    for (const auto &[begin, end] : ready.action.byteRanges)
      ranges.push_back(
          Range{ready.action.rootId, begin, end, ready.action.isWrite});
  }
  llvm::sort(ranges, [](const Range &lhs, const Range &rhs) {
    if (lhs.rootId != rhs.rootId)
      return lhs.rootId < rhs.rootId;
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    if (lhs.end != rhs.end)
      return lhs.end < rhs.end;
    return lhs.isWrite < rhs.isWrite;
  });

  std::optional<std::uint64_t> rootId;
  std::int64_t maximalEnd = 0;
  std::int64_t maximalWriteEnd = 0;
  bool hasRange = false;
  bool hasWrite = false;
  for (const Range &range : ranges) {
    if (!rootId || *rootId != range.rootId) {
      rootId = range.rootId;
      maximalEnd = range.end;
      maximalWriteEnd = range.end;
      hasRange = true;
      hasWrite = range.isWrite;
      continue;
    }
    if ((range.isWrite && hasRange && maximalEnd > range.begin) ||
        (!range.isWrite && hasWrite && maximalWriteEnd > range.begin))
      return true;
    maximalEnd = std::max(maximalEnd, range.end);
    if (range.isWrite) {
      maximalWriteEnd =
          hasWrite ? std::max(maximalWriteEnd, range.end) : range.end;
      hasWrite = true;
    }
  }
  return false;
}

llvm::SmallVector<SyncEffectId>
PlainMemoryConflictIndex::query(const MemoryActionRecord &action) const {
  llvm::SmallVector<SyncEffectId> effects;
  auto root = intervals_.find(action.rootId);
  if (root == intervals_.end())
    return effects;

  const IntervalMap &intervals = root->second->intervals;
  for (const auto &[begin, end] : action.byteRanges) {
    auto interval = intervals.find(begin);
    while (interval.valid() && interval.start() < end) {
      const Hazards &hazards = interval.value();
      if (hazards.write)
        effects.push_back(*hazards.write);
      if (action.isWrite)
        effects.append(hazards.reads.begin(), hazards.reads.end());
      ++interval;
    }
  }
  llvm::sort(effects);
  effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
  return effects;
}

void PlainMemoryConflictIndex::retain(const MemoryActionRecord &action,
                                      SyncEffectId effect,
                                      MemorySynchronization &synchronization) {
  if (action.byteRanges.empty())
    return;
  std::unique_ptr<RootIntervals> &root = intervals_[action.rootId];
  if (!root)
    root = std::make_unique<RootIntervals>();
  for (const auto &[begin, end] : action.byteRanges)
    updateRange(*root, begin, end, action.isWrite, effect, synchronization);
}

void PlainMemoryConflictIndex::applyAccess(
    Hazards &hazards, bool isWrite, SyncEffectId effect,
    MemorySynchronization &synchronization) {
  if (isWrite) {
    hazards.write = effect;
    hazards.reads.clear();
    return;
  }
  hazards.reads.push_back(effect);
  if (hazards.reads.size() > 1)
    hazards.reads = llvm::cantFail(
        synchronization.maximalHappensBeforeFrontier(hazards.reads));
}

PlainMemoryConflictIndex::Hazards
PlainMemoryConflictIndex::makeHazards(bool isWrite, SyncEffectId effect,
                                      MemorySynchronization &synchronization) {
  Hazards hazards;
  applyAccess(hazards, isWrite, effect, synchronization);
  return hazards;
}

void PlainMemoryConflictIndex::updateRange(
    RootIntervals &root, std::int64_t begin, std::int64_t end, bool isWrite,
    SyncEffectId effect, MemorySynchronization &synchronization) {
  llvm::SmallVector<IntervalReplacement, 4> existing;
  auto interval = root.intervals.find(begin);
  while (interval.valid() && interval.start() < end) {
    existing.push_back(IntervalReplacement{interval.start(), interval.stop(),
                                           interval.value()});
    interval.erase();
  }

  llvm::SmallVector<IntervalReplacement, 6> replacements;
  std::int64_t cursor = begin;
  for (const IntervalReplacement &prior : existing) {
    if (prior.begin < begin)
      replacements.push_back(
          IntervalReplacement{prior.begin, begin, prior.hazards});

    const std::int64_t overlapBegin = std::max(begin, prior.begin);
    if (cursor < overlapBegin)
      replacements.push_back(IntervalReplacement{
          cursor, overlapBegin, makeHazards(isWrite, effect, synchronization)});

    const std::int64_t overlapEnd = std::min(end, prior.end);
    Hazards updated = prior.hazards;
    applyAccess(updated, isWrite, effect, synchronization);
    replacements.push_back(
        IntervalReplacement{overlapBegin, overlapEnd, std::move(updated)});
    cursor = std::max(cursor, overlapEnd);

    if (prior.end > end) {
      replacements.push_back(
          IntervalReplacement{end, prior.end, prior.hazards});
      cursor = end;
    }
  }
  if (cursor < end)
    replacements.push_back(IntervalReplacement{
        cursor, end, makeHazards(isWrite, effect, synchronization)});

  for (IntervalReplacement &replacement : replacements)
    root.intervals.insert(replacement.begin, replacement.end,
                          std::move(replacement.hazards));
}

// An unordered conflict is knowable only from the resolved runtime addresses,
// so the exact model reports the capability it lacks instead of choosing an
// arbitrary order. The whole ready set is dropped, so no admitted action of
// this rejected decision can still fire.
static bool rejectPlainMemoryConflict(SimulatorState &state) {
  state.admittedPlainMemoryActions.clear();
  state.diagnostics.push_back(
      "unordered plain accesses conflict on the same memory");
  state.failure = RunFailure::UnsupportedCapability;
  return false;
}

bool admitReadyPlainMemoryActions(SimulatorState &state) {
  state.admittedPlainMemoryActions.clear();
  llvm::SmallVector<mlir::Operation *> readyOperations;
  llvm::SmallVector<ReadyPlainMemoryAction> ready;
  llvm::SmallVector<unsigned, 8> inactiveCandidates;
  llvm::SmallVector<std::string> projectionDiagnostics;
  for (int ordinal = state.plainMemoryCandidates.find_first(); ordinal >= 0;
       ordinal = state.plainMemoryCandidates.find_next(ordinal)) {
    const ActorExecutionPlan &plan = state.execution->actorPlans[ordinal];
    mlir::Operation *operation = plan.operation;
    state.currentActorPlan = &plan;
    PlainMemoryActionProjection projection =
        projectReadyPlainMemoryAction(operation, state);
    state.currentActorPlan = nullptr;
    for (std::string &diagnostic : projection.diagnostics)
      projectionDiagnostics.push_back(std::move(diagnostic));
    if (!projection.ready) {
      inactiveCandidates.push_back(static_cast<unsigned>(ordinal));
      continue;
    }
    readyOperations.push_back(operation);
    ready.push_back(std::move(*projection.ready));
  }
  for (unsigned ordinal : inactiveCandidates)
    state.plainMemoryCandidates.reset(ordinal);

  if (readyPlainMemoryActionsConflict(ready))
    return rejectPlainMemoryConflict(state);

  for (const ReadyPlainMemoryAction &candidate : ready) {
    llvm::SmallVector<SyncEffectId> conflict =
        state.memoryActions.query(candidate.action);
    if (!conflict.empty() && !state.memorySync->areCoveredByHappensBefore(
                                 conflict, candidate.ctrlFrontier))
      return rejectPlainMemoryConflict(state);
  }

  if (!projectionDiagnostics.empty()) {
    for (std::string &diagnostic : projectionDiagnostics)
      state.diagnostics.push_back(std::move(diagnostic));
    return false;
  }

  for (auto [index, operation] : llvm::enumerate(readyOperations))
    state.admittedPlainMemoryActions.try_emplace(operation,
                                                 std::move(ready[index]));
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
