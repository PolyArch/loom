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
static bool
readyPlainMemoryActionsConflict(llvm::ArrayRef<ReadyMemoryAction> actions) {
  struct Range {
    std::uint64_t rootId;
    std::int64_t begin;
    std::int64_t end;
    bool isWrite;
    bool isAtomic;
  };

  llvm::SmallVector<Range, 8> ranges;
  for (const ReadyMemoryAction &ready : actions) {
    for (const auto &[begin, end] : ready.action.byteRanges)
      ranges.push_back(Range{ready.action.rootId, begin, end,
                             ready.action.isWrite, ready.action.isAtomic});
  }
  llvm::sort(ranges, [](const Range &lhs, const Range &rhs) {
    if (lhs.rootId != rhs.rootId)
      return lhs.rootId < rhs.rootId;
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    if (lhs.end != rhs.end)
      return lhs.end < rhs.end;
    if (lhs.isWrite != rhs.isWrite)
      return lhs.isWrite < rhs.isWrite;
    return lhs.isAtomic < rhs.isAtomic;
  });

  std::optional<std::uint64_t> rootId;
  std::int64_t plainEnd = 0;
  std::int64_t plainWriteEnd = 0;
  std::int64_t atomicEnd = 0;
  std::int64_t atomicWriteEnd = 0;
  bool hasPlain = false;
  bool hasPlainWrite = false;
  bool hasAtomic = false;
  bool hasAtomicWrite = false;
  for (const Range &range : ranges) {
    if (!rootId || *rootId != range.rootId) {
      rootId = range.rootId;
      hasPlain = false;
      hasPlainWrite = false;
      hasAtomic = false;
      hasAtomicWrite = false;
    }
    const bool conflicts =
        range.isAtomic
            ? (range.isWrite ? hasPlain && plainEnd > range.begin
                             : hasPlainWrite && plainWriteEnd > range.begin)
            : (range.isWrite
                   ? (hasPlain && plainEnd > range.begin) ||
                         (hasAtomic && atomicEnd > range.begin)
                   : (hasPlainWrite && plainWriteEnd > range.begin) ||
                         (hasAtomicWrite && atomicWriteEnd > range.begin));
    if (conflicts)
      return true;
    if (range.isAtomic) {
      atomicEnd = hasAtomic ? std::max(atomicEnd, range.end) : range.end;
      hasAtomic = true;
      if (range.isWrite) {
        atomicWriteEnd =
            hasAtomicWrite ? std::max(atomicWriteEnd, range.end) : range.end;
        hasAtomicWrite = true;
      }
    } else {
      plainEnd = hasPlain ? std::max(plainEnd, range.end) : range.end;
      hasPlain = true;
      if (range.isWrite) {
        plainWriteEnd =
            hasPlainWrite ? std::max(plainWriteEnd, range.end) : range.end;
        hasPlainWrite = true;
      }
    }
  }
  return false;
}

static bool readyAtomicActionsHaveInexactOverlap(
    llvm::ArrayRef<ReadyMemoryAction> actions) {
  for (auto [leftIndex, left] : llvm::enumerate(actions)) {
    if (!left.action.isAtomic)
      continue;
    for (const ReadyMemoryAction &right : actions.drop_front(leftIndex + 1)) {
      if (!right.action.isAtomic || left.action.rootId != right.action.rootId ||
          left.action.byteRanges == right.action.byteRanges ||
          (!left.action.isWrite && !right.action.isWrite))
        continue;
      for (const auto &[leftBegin, leftEnd] : left.action.byteRanges)
        for (const auto &[rightBegin, rightEnd] : right.action.byteRanges)
          if (leftBegin < rightEnd && rightBegin < leftEnd)
            return true;
    }
  }
  return false;
}

llvm::SmallVector<SyncEffectId>
PlainMemoryConflictIndex::queryKind(const MemoryActionRecord &action,
                                    bool isAtomic) const {
  llvm::SmallVector<SyncEffectId> effects;
  auto root = intervals_.find(action.rootId);
  if (root == intervals_.end())
    return effects;

  const IntervalMap &intervals = root->second->intervals;
  for (const auto &[begin, end] : action.byteRanges) {
    auto interval = intervals.find(begin);
    while (interval.valid() && interval.start() < end) {
      const Hazards &hazards = interval.value();
      auto collect = [&](const AccessHazards &accesses) {
        effects.append(accesses.writes.begin(), accesses.writes.end());
        if (action.isWrite)
          effects.append(accesses.reads.begin(), accesses.reads.end());
      };
      collect(isAtomic ? hazards.atomic : hazards.plain);
      ++interval;
    }
  }
  llvm::sort(effects);
  effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
  return effects;
}

llvm::SmallVector<SyncEffectId> PlainMemoryConflictIndex::querySameKind(
    const MemoryActionRecord &action) const {
  return queryKind(action, action.isAtomic);
}

llvm::SmallVector<SyncEffectId> PlainMemoryConflictIndex::queryCrossKind(
    const MemoryActionRecord &action) const {
  return queryKind(action, !action.isAtomic);
}

static std::optional<AtomicObjectKey>
atomicObjectKey(const MemoryActionRecord &action) {
  if (!action.isAtomic || action.byteRanges.size() != 1)
    return std::nullopt;
  const auto [begin, end] = action.byteRanges.front();
  if (begin < 0 || end <= begin)
    return std::nullopt;
  return AtomicObjectKey{action.rootId, static_cast<std::uint64_t>(begin),
                         static_cast<std::uint64_t>(end - begin)};
}

bool PlainMemoryConflictIndex::hasInexactAtomicHazard(
    const MemoryActionRecord &action) const {
  if (!action.isAtomic)
    return false;
  const std::optional<AtomicObjectKey> candidate = atomicObjectKey(action);
  if (!candidate)
    return true;
  const std::uint64_t candidateEnd =
      candidate->canonicalByteOffset + candidate->accessByteSize;
  for (const auto &[prior, priorWrites] : atomicObjectWrites_) {
    if (prior.logicalRootId != candidate->logicalRootId || prior == *candidate)
      continue;
    const std::uint64_t priorEnd =
        prior.canonicalByteOffset + prior.accessByteSize;
    if (candidate->canonicalByteOffset < priorEnd &&
        prior.canonicalByteOffset < candidateEnd &&
        (action.isWrite || priorWrites))
      return true;
  }
  return false;
}

void PlainMemoryConflictIndex::retain(const MemoryActionRecord &action,
                                      SyncEffectId effect,
                                      MemorySynchronization &synchronization) {
  if (action.byteRanges.empty())
    return;
  if (action.isAtomic) {
    const std::optional<AtomicObjectKey> key = atomicObjectKey(action);
    assert(key && "retained atomic action has no exact object key");
    auto [object, inserted] = atomicObjectWrites_.try_emplace(*key, false);
    (void)inserted;
    object->second |= action.isWrite;
  }
  std::unique_ptr<RootIntervals> &root = intervals_[action.rootId];
  if (!root)
    root = std::make_unique<RootIntervals>();
  for (const auto &[begin, end] : action.byteRanges)
    updateRange(*root, begin, end, action.isWrite, action.isAtomic, effect,
                synchronization);
}

void PlainMemoryConflictIndex::applyAccess(
    AccessHazards &hazards, bool isWrite, SyncEffectId effect,
    MemorySynchronization &synchronization) {
  if (isWrite) {
    hazards.reads.erase(
        std::remove_if(hazards.reads.begin(), hazards.reads.end(),
                       [&](SyncEffectId read) {
                         return synchronization.happensBefore(read, effect);
                       }),
        hazards.reads.end());
    hazards.writes.push_back(effect);
    if (hazards.writes.size() > 1)
      hazards.writes = llvm::cantFail(
          synchronization.maximalHappensBeforeFrontier(hazards.writes));
    return;
  }
  hazards.reads.push_back(effect);
  if (hazards.reads.size() > 1)
    hazards.reads = llvm::cantFail(
        synchronization.maximalHappensBeforeFrontier(hazards.reads));
}

PlainMemoryConflictIndex::Hazards
PlainMemoryConflictIndex::makeHazards(bool isWrite, bool isAtomic,
                                      SyncEffectId effect,
                                      MemorySynchronization &synchronization) {
  Hazards hazards;
  applyAccess(isAtomic ? hazards.atomic : hazards.plain, isWrite, effect,
              synchronization);
  return hazards;
}

void PlainMemoryConflictIndex::updateRange(
    RootIntervals &root, std::int64_t begin, std::int64_t end, bool isWrite,
    bool isAtomic, SyncEffectId effect,
    MemorySynchronization &synchronization) {
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
          cursor, overlapBegin,
          makeHazards(isWrite, isAtomic, effect, synchronization)});

    const std::int64_t overlapEnd = std::min(end, prior.end);
    Hazards updated = prior.hazards;
    applyAccess(isAtomic ? updated.atomic : updated.plain, isWrite, effect,
                synchronization);
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
        cursor, end, makeHazards(isWrite, isAtomic, effect, synchronization)});

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

static bool rejectMixedMemoryConflict(SimulatorState &state) {
  state.admittedPlainMemoryActions.clear();
  state.diagnostics.push_back(
      "mixed atomic/plain write hazard has no exact DFG value/version "
      "correspondence");
  state.failure = RunFailure::UnsupportedCapability;
  return false;
}

static bool rejectInexactAtomicOverlap(SimulatorState &state) {
  state.admittedPlainMemoryActions.clear();
  state.diagnostics.push_back(
      "overlapping atomic actions do not share one AtomicObjectKey");
  state.failure = RunFailure::UnsupportedCapability;
  return false;
}

bool admitReadyPlainMemoryActions(SimulatorState &state) {
  state.admittedPlainMemoryActions.clear();
  llvm::SmallVector<mlir::Operation *> readyOperations;
  llvm::SmallVector<ReadyMemoryAction> ready;
  llvm::SmallVector<unsigned, 8> inactiveCandidates;
  llvm::SmallVector<std::string> projectionDiagnostics;
  for (int ordinal = state.plainMemoryCandidates.find_first(); ordinal >= 0;
       ordinal = state.plainMemoryCandidates.find_next(ordinal)) {
    const ActorExecutionPlan &plan = state.execution->actorPlans[ordinal];
    mlir::Operation *operation = plan.operation;
    state.currentActorPlan = &plan;
    MemoryActionProjection projection =
        projectReadyMemoryAction(operation, state);
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

  // Atomic actions use their own provider, but same-wave plain/atomic overlap
  // must be rejected before either provider commits. Atomic actions on the
  // same exact object are ordered by MemoryAtomicOrder; inexact byte overlap
  // involving a write has no shared modification order and fails closed here.
  for (int ordinal = state.nextActorCandidates.find_first(); ordinal >= 0;
       ordinal = state.nextActorCandidates.find_next(ordinal)) {
    const ActorExecutionPlan &plan = state.execution->actorPlans[ordinal];
    if (!plan.memory || plan.isPlainMemory())
      continue;
    state.currentActorPlan = &plan;
    MemoryActionProjection projection =
        projectReadyMemoryAction(plan.operation, state);
    state.currentActorPlan = nullptr;
    for (std::string &diagnostic : projection.diagnostics)
      projectionDiagnostics.push_back(std::move(diagnostic));
    if (projection.ready)
      ready.push_back(std::move(*projection.ready));
  }

  if (readyAtomicActionsHaveInexactOverlap(ready))
    return rejectInexactAtomicOverlap(state);
  if (readyPlainMemoryActionsConflict(ready))
    return rejectPlainMemoryConflict(state);

  for (const ReadyMemoryAction &candidate : ready) {
    if (candidate.action.isAtomic)
      continue;
    if (!state.memoryActions.queryCrossKind(candidate.action).empty())
      return rejectMixedMemoryConflict(state);
    llvm::SmallVector<SyncEffectId> conflict =
        state.memoryActions.querySameKind(candidate.action);
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
