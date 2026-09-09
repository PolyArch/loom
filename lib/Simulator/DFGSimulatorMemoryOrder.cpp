//===- DFGSimulatorMemoryOrder.cpp - Token memory-order frontiers ---------===//
//
// Resolution of the memory-order frontiers that simulator tokens carry.
//
// Publication joins incoming effect witnesses through MemorySynchronization,
// the sole happens-before authority. A join follows every input without
// ordering the inputs with respect to one another. Tokens carry that one
// witness; a stateful activation retains its accumulator and publication memo
// across firings so unchanged order is forwarded directly.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include <cassert>
#include <utility>

namespace loom::sim::detail {

MemoryOrderFrontierId publishMemoryOrder(SimulatorState &state,
                                         MemoryOrderAccumulator &accumulator) {
  // Handle zero is the empty frontier. Leaving an empty
  // accumulator pristine makes the overwhelmingly common no-memory firing
  // and every subsequent not-ready scheduler probe allocation-free.
  if (accumulator.empty())
    return MemoryOrderFrontierId{};
  if (std::optional<MemoryOrderFrontierId> published = accumulator.published())
    return *published;
  llvm::SmallVector<SyncEffectId, 4> predecessors;
  for (MemoryOrderFrontierId frontier : accumulator.frontiers())
    predecessors.push_back(frontier.effect());
  llvm::sort(predecessors);
  predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                     predecessors.end());
  // A join records only that every incoming effect precedes this publication;
  // it creates no order between those effects. Keeping that event as the
  // token's witness avoids expanding the same control join at each later
  // memory access. The memory-order authority owns every resulting relation.
  const SyncEffectId effect =
      predecessors.size() == 1
          ? predecessors.front()
          : llvm::cantFail(memorySynchronization(state)
                               .declareEffectSequencedAfter(predecessors));
  const MemoryOrderFrontierId id =
      MemoryOrderFrontierId::fromEffect(effect);
  accumulator.markPublished(id);
  return id;
}

MemoryOrderFrontierId publishFiredMemoryOrder(SimulatorState &state,
                                              MemoryOrderFrontierId carried) {
  if (state.firingMemoryOrderFrontier.empty())
    return carried;
  // Every result of one firing observes the firing's order, so the firing
  // joins once and each further result copies only the handle.
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
  merged.absorb(carried);
  merged.absorb(fired);
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
  // join the inputs again and a forwarded token would merge back into the
  // frontier it already contributed to. Folding the firing's consumed order
  // in before the trade keeps a contribution the union already represents
  // from touching the union's memos, so a firing that consumed nothing new
  // republishes the unchanged union as one handle lookup, and a transition
  // that emits nothing never joins the order it is about to
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
  // The union, with the memo its publication set, rests in the activation
  // slot until the next firing trades for it. The firing slot keeps the
  // firing's own consumed remnant, which the scheduler clears before the
  // next attempt.
  std::swap(state.firingMemoryOrderFrontier, retained->second);
}

} // namespace loom::sim::detail
