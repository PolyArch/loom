//===- DFGSimulatorMemoryOrder.cpp - Token memory-order frontiers ---------===//
//
// Resolution of the memory-order frontiers that simulator tokens carry.
//
// A frontier is accumulated in transient mutable state and interned only when
// a token actually carries it, so the run retains one immutable copy per
// distinct published frontier and every carrier holds a dense handle.
//
// A stateful actor's activation retains one union across its firings. The
// union trades between the activation's slot and the firing slot with its
// reduction and publication memos intact, so an unchanged union is never
// reduced or interned twice and order that no token carries is never
// interned at all.
//
// MemorySynchronization stays the sole authority for happens-before and for
// reducing a frontier to its maximal members; nothing here relates two
// effects.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace loom::sim::detail {

void reduceMemoryOrderFrontier(SimulatorState &state,
                               llvm::SmallVectorImpl<SyncEffectId> &frontier) {
  if (frontier.size() < 2)
    return;
  assert(state.memorySync && "a memory-order witness requires its authority");
  // The authority canonicalizes and reduces in one pass and owns that shape;
  // canonicalizing here first would only repeat its work.
  llvm::SmallVector<SyncEffectId> reduced =
      llvm::cantFail(state.memorySync->maximalHappensBeforeFrontier(frontier));
  frontier.assign(reduced.begin(), reduced.end());
}

void reduceMemoryOrder(SimulatorState &state,
                       MemoryOrderAccumulator &accumulator) {
  if (accumulator.isReduced())
    return;
  llvm::SmallVector<SyncEffectId, 4> frontier(accumulator.elements().begin(),
                                              accumulator.elements().end());
  reduceMemoryOrderFrontier(state, frontier);
  accumulator.adoptReduced(frontier);
}

MemoryOrderFrontierId publishMemoryOrder(SimulatorState &state,
                                         MemoryOrderAccumulator &accumulator) {
  // The arena reserves handle zero for the empty frontier. Leaving an empty
  // accumulator pristine makes the overwhelmingly common no-memory firing
  // and every subsequent not-ready scheduler probe allocation-free.
  if (accumulator.empty())
    return MemoryOrderFrontierId{};
  if (std::optional<MemoryOrderFrontierId> published = accumulator.published())
    return *published;
  // Reduction leaves the elements canonical, so interning never sorts again.
  reduceMemoryOrder(state, accumulator);
  const MemoryOrderFrontierId id =
      state.memoryOrderFrontiers.internCanonical(accumulator.elements());
  accumulator.markPublished(id);
  return id;
}

MemoryOrderFrontierId publishFiredMemoryOrder(SimulatorState &state,
                                              MemoryOrderFrontierId carried) {
  if (state.firingMemoryOrderFrontier.empty())
    return carried;
  // Every result of one firing observes the firing's order, so the firing
  // reduces and interns once and each further result copies only the handle.
  const MemoryOrderFrontierId fired =
      publishMemoryOrder(state, state.firingMemoryOrderFrontier);
  if (carried.empty())
    return fired;
  // A token the firing consumed already contributed its order through
  // popToken, so the firing frontier covers it. Order the firing never
  // consumed does not: a value read out of memory carries a witness of its
  // own and keeps it. Merging only that case leaves every ordinary result on
  // the shared handle.
  if (state.firingMemoryOrderFrontier.hasAbsorbed(carried))
    return fired;
  MemoryOrderAccumulator merged;
  merged.append(state.memoryOrderFrontiers.elements(carried));
  merged.append(state.memoryOrderFrontiers.elements(fired));
  return publishMemoryOrder(state, merged);
}

void retainAndPublishActivationMemoryOrder(SimulatorState &state,
                                           mlir::Operation *actor) {
  MemoryOrderAccumulator &activation =
      state.activationMemoryOrderFrontiers[actor];
  // The firing publishes the activation's whole retained union, so the union
  // moves into the firing slot as one accumulator, memos included, rather
  // than being copied into a fresh one. A copy would look unpublished and
  // would have absorbed nothing, so the first emission would reduce and
  // intern the union again and a forwarded token would merge back into the
  // frontier it already contributed to. Folding the firing's consumed order
  // in before the trade keeps a contribution the union already represents
  // from touching the union's memos, so a firing that consumed nothing new
  // republishes the unchanged union as one handle lookup, and a transition
  // that emits nothing never reduces or interns the order it is about to
  // drop.
  activation.absorbAll(state.firingMemoryOrderFrontier);
  std::swap(state.firingMemoryOrderFrontier, activation);
}

void releaseActivationMemoryOrder(SimulatorState &state, mlir::Operation *actor,
                                  bool retire) {
  auto retained = state.activationMemoryOrderFrontiers.find(actor);
  if (retained == state.activationMemoryOrderFrontiers.end())
    return;
  if (retire) {
    state.activationMemoryOrderFrontiers.erase(retained);
    return;
  }
  // The union, with the memos its publications set, rests in the activation
  // slot until the next firing trades for it. The firing slot keeps the
  // firing's own consumed remnant, which the scheduler clears before the
  // next attempt.
  std::swap(state.firingMemoryOrderFrontier, retained->second);
}

} // namespace loom::sim::detail
