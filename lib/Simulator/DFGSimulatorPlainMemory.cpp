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
  llvm::SmallVector<std::pair<std::int64_t, std::int64_t>> merged;
  for (const auto &range : ranges) {
    if (range.first >= range.second)
      continue;
    if (merged.empty() || merged.back().second < range.first) {
      merged.push_back(range);
      continue;
    }
    merged.back().second = std::max(merged.back().second, range.second);
  }
  ranges.assign(merged.begin(), merged.end());
}

ReadyPlainMemoryConflictScan
scanReadyPlainMemoryConflicts(llvm::ArrayRef<ReadyPlainMemoryAction> actions) {
  struct Range {
    std::uint64_t rootId;
    std::int64_t begin;
    std::int64_t end;
    bool isWrite;
  };

  llvm::SmallVector<Range> ranges;
  for (const ReadyPlainMemoryAction &ready : actions) {
    auto actionRanges = ready.action.byteRanges;
    canonicalizeMemoryActionRanges(actionRanges);
    for (const auto &[begin, end] : actionRanges)
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

  ReadyPlainMemoryConflictScan result;
  std::optional<std::uint64_t> rootId;
  std::int64_t maximalEnd = 0;
  std::int64_t maximalWriteEnd = 0;
  bool hasRange = false;
  bool hasWrite = false;
  for (const Range &range : ranges) {
    ++result.inspectedRanges;
    if (!rootId || *rootId != range.rootId) {
      rootId = range.rootId;
      maximalEnd = range.end;
      maximalWriteEnd = range.end;
      hasRange = true;
      hasWrite = range.isWrite;
      continue;
    }
    if ((range.isWrite && hasRange && maximalEnd > range.begin) ||
        (!range.isWrite && hasWrite && maximalWriteEnd > range.begin)) {
      result.hasConflict = true;
      return result;
    }
    maximalEnd = std::max(maximalEnd, range.end);
    if (range.isWrite) {
      maximalWriteEnd =
          hasWrite ? std::max(maximalWriteEnd, range.end) : range.end;
      hasWrite = true;
    }
  }
  return result;
}

PlainMemoryConflictQuery
PlainMemoryConflictIndex::query(MemoryActionRecord action) const {
  PlainMemoryConflictQuery result;
  canonicalizeMemoryActionRanges(action.byteRanges);
  auto root = intervals_.find(action.rootId);
  if (root == intervals_.end())
    return result;

  const IntervalMap &intervals = root->second->intervals;
  for (const auto &[begin, end] : action.byteRanges) {
    auto interval = intervals.find(begin);
    while (interval.valid() && interval.start() < end) {
      ++result.inspectedIntervals;
      const Hazards &hazards = interval.value();
      if (hazards.write)
        result.effects.push_back(*hazards.write);
      if (action.isWrite)
        result.effects.append(hazards.reads.begin(), hazards.reads.end());
      ++interval;
    }
  }
  llvm::sort(result.effects);
  result.effects.erase(
      std::unique(result.effects.begin(), result.effects.end()),
      result.effects.end());
  return result;
}

void PlainMemoryConflictIndex::retain(MemoryActionRecord action,
                                      SyncEffectId effect,
                                      MemorySynchronization &synchronization) {
  canonicalizeMemoryActionRanges(action.byteRanges);
  if (action.byteRanges.empty())
    return;
  std::unique_ptr<RootIntervals> &root = intervals_[action.rootId];
  if (!root)
    root = std::make_unique<RootIntervals>();
  for (const auto &[begin, end] : action.byteRanges)
    updateRange(*root, begin, end, action.isWrite, effect, synchronization);
}

std::size_t
PlainMemoryConflictIndex::intervalCount(std::uint64_t rootId) const {
  auto root = intervals_.find(rootId);
  if (root == intervals_.end())
    return 0;
  return std::distance(root->second->intervals.begin(),
                       root->second->intervals.end());
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
  llvm::SmallVector<std::uint64_t> inactiveCandidates;
  llvm::SmallVector<std::string> projectionDiagnostics;
  for (const auto &[ordinal, operation] : state.plainMemoryCandidates) {
    PlainMemoryActionProjection projection =
        projectReadyPlainMemoryAction(operation, state);
    for (std::string &diagnostic : projection.diagnostics)
      projectionDiagnostics.push_back(std::move(diagnostic));
    if (!projection.ready) {
      inactiveCandidates.push_back(ordinal);
      continue;
    }
    readyOperations.push_back(operation);
    ready.push_back(std::move(*projection.ready));
  }
  for (std::uint64_t ordinal : inactiveCandidates)
    state.plainMemoryCandidates.erase(ordinal);

  if (scanReadyPlainMemoryConflicts(ready).hasConflict)
    return rejectPlainMemoryConflict(state);

  for (const ReadyPlainMemoryAction &candidate : ready) {
    PlainMemoryConflictQuery conflict =
        state.memoryActions.query(candidate.action);
    if (!conflict.effects.empty() &&
        !state.memorySync->areCoveredByHappensBefore(conflict.effects,
                                                     candidate.ctrlFrontier))
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
