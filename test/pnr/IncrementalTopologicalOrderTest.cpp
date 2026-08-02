#include "IncrementalTopologicalOrder.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::pnr::FrozenSpatialHandshakeArc;
using loom::pnr::PnrIndex;
using loom::pnr::detail::IncrementalTopologicalGraphView;
using loom::pnr::detail::IncrementalTopologicalOrder;
using loom::pnr::detail::IncrementalTopologicalOrderHandle;
using loom::pnr::detail::IncrementalTopologicalScratch;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "incremental topological order test failed: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

struct GraphFixture final {
  std::vector<FrozenSpatialHandshakeArc> arcs{
      {0, 2}, {1, 0}, {2, 4}, {3, 1}, {4, 1}};
  std::vector<PnrIndex> adjacencyOffsets{0, 1, 2, 3, 4, 5};
  std::vector<PnrIndex> reverseAdjacencyOffsets{0, 1, 3, 4, 4, 5};
  std::vector<PnrIndex> reverseArcOrdinals{1, 3, 4, 0, 2};

  IncrementalTopologicalGraphView view() const {
    return {5, arcs, adjacencyOffsets, reverseAdjacencyOffsets,
            reverseArcOrdinals};
  }
};

void requireValidOrder(const IncrementalTopologicalOrder &order) {
  requireSuccess(order.verify());
  for (PnrIndex arc = 0; arc < 5; ++arc) {
    if (!order.isArcActive(arc))
      continue;
    const auto endpoints = order.graph().arcs[arc];
    if (order.rank(endpoints.source) >= order.rank(endpoints.destination))
      fail("active arc violates the maintained topological order");
  }
}

void rankViolatingInsertionReordersOnlyTheAffectedState() {
  GraphFixture fixture;
  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(fixture.view(), {0, 2}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(fixture.view()));
  const std::size_t retainedBytes = scratch.retainedStorageBytes();

  {
    auto transaction = take(order->beginTransaction(scratch));
    if (!take(transaction.insertArc(4)))
      fail("acyclic rank-violating insertion was rejected");
    requireSuccess(transaction.commit());
  }
  requireValidOrder(*order);
  if (!order->isArcActive(4) || order->rank(4) >= order->rank(1))
    fail("rank-violating insertion was not reflected in the maintained order");

  {
    auto transaction = take(order->beginTransaction(scratch));
    if (!take(transaction.insertArc(3)))
      fail("rank-respecting insertion was rejected");
    requireSuccess(transaction.commit());
  }
  requireValidOrder(*order);
  if (scratch.retainedStorageBytes() != retainedBytes)
    fail("warm local insertion expanded topological scratch storage");
}

void cycleWitnessAndRollbackAreDeterministic() {
  GraphFixture fixture;
  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(fixture.view(), {0, 2, 4}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(fixture.view()));
  const std::vector<PnrIndex> originalOrder(order->order().begin(),
                                            order->order().end());
  const std::vector<PnrIndex> originalRank(order->ranks().begin(),
                                           order->ranks().end());

  {
    auto transaction = take(order->beginTransaction(scratch));
    if (take(transaction.insertArc(1)))
      fail("cycle-closing insertion was accepted");
    const std::vector<PnrIndex> expectedWitness{0, 1, 2, 4};
    if (!std::equal(transaction.cycleWitness().begin(),
                    transaction.cycleWitness().end(), expectedWitness.begin(),
                    expectedWitness.end()))
      fail("cycle witness is not the canonical potential-arc set");
    transaction.rollback();
  }

  if (order->isArcActive(1) ||
      !std::equal(order->order().begin(), order->order().end(),
                  originalOrder.begin(), originalOrder.end()) ||
      !std::equal(order->ranks().begin(), order->ranks().end(),
                  originalRank.begin(), originalRank.end()))
    fail("cycle rollback did not restore the exact committed state");
  requireValidOrder(*order);
}

void deletionRollbackRestoresTheCommittedGraph() {
  GraphFixture fixture;
  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(fixture.view(), {0, 2, 4}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(fixture.view()));

  {
    auto transaction = take(order->beginTransaction(scratch));
    requireSuccess(transaction.removeArc(4));
    transaction.rollback();
  }
  if (!order->isArcActive(4))
    fail("arc deletion survived rollback");

  {
    auto transaction = take(order->beginTransaction(scratch));
    requireSuccess(transaction.removeArc(4));
    requireSuccess(transaction.commit());
  }
  if (order->isArcActive(4))
    fail("committed arc deletion was not retained");
  requireSuccess(order->rebuild());
  requireValidOrder(*order);
}

} // namespace

int main() {
  rankViolatingInsertionReordersOnlyTheAffectedState();
  cycleWitnessAndRollbackAreDeterministic();
  deletionRollbackRestoresTheCommittedGraph();
  return 0;
}
