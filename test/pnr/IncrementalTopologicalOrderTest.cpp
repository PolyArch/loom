#include "IncrementalTopologicalOrder.h"
#include "../TestAllocationProbe.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <queue>
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

struct PotentialGraph final {
  PnrIndex nodeCount = 0;
  std::vector<FrozenSpatialHandshakeArc> arcs;
  std::vector<PnrIndex> adjacencyOffsets;
  std::vector<PnrIndex> reverseAdjacencyOffsets;
  std::vector<PnrIndex> reverseArcOrdinals;

  IncrementalTopologicalGraphView view() const {
    return {nodeCount, arcs, adjacencyOffsets, reverseAdjacencyOffsets,
            reverseArcOrdinals};
  }
};

PotentialGraph buildOffsetGraph(PnrIndex nodeCount,
                                const std::vector<PnrIndex> &offsets) {
  PotentialGraph graph;
  graph.nodeCount = nodeCount;
  graph.adjacencyOffsets.reserve(static_cast<std::size_t>(nodeCount) + 1);
  graph.adjacencyOffsets.push_back(0);
  for (PnrIndex source = 0; source < nodeCount; ++source) {
    std::vector<PnrIndex> destinations;
    destinations.reserve(offsets.size());
    for (PnrIndex offset : offsets) {
      const PnrIndex destination = (source + offset) % nodeCount;
      if (destination != source)
        destinations.push_back(destination);
    }
    std::sort(destinations.begin(), destinations.end());
    destinations.erase(std::unique(destinations.begin(), destinations.end()),
                       destinations.end());
    for (PnrIndex destination : destinations)
      graph.arcs.push_back({source, destination});
    graph.adjacencyOffsets.push_back(static_cast<PnrIndex>(graph.arcs.size()));
  }

  std::vector<PnrIndex> reverseCounts(nodeCount, 0);
  for (const FrozenSpatialHandshakeArc arc : graph.arcs)
    ++reverseCounts[arc.destination];
  graph.reverseAdjacencyOffsets.reserve(static_cast<std::size_t>(nodeCount) +
                                        1);
  graph.reverseAdjacencyOffsets.push_back(0);
  for (PnrIndex count : reverseCounts)
    graph.reverseAdjacencyOffsets.push_back(
        graph.reverseAdjacencyOffsets.back() + count);
  graph.reverseArcOrdinals.resize(graph.arcs.size());
  std::vector<PnrIndex> cursors(graph.reverseAdjacencyOffsets.begin(),
                                graph.reverseAdjacencyOffsets.end() - 1);
  for (PnrIndex ordinal = 0; ordinal < graph.arcs.size(); ++ordinal)
    graph.reverseArcOrdinals[cursors[graph.arcs[ordinal].destination]++] =
        ordinal;
  return graph;
}

bool fullKahnIsAcyclic(const PotentialGraph &graph,
                       const std::vector<bool> &active) {
  std::vector<PnrIndex> indegree(graph.nodeCount, 0);
  for (PnrIndex arc = 0; arc < graph.arcs.size(); ++arc)
    if (active[arc])
      ++indegree[graph.arcs[arc].destination];
  std::queue<PnrIndex> ready;
  for (PnrIndex node = 0; node < graph.nodeCount; ++node)
    if (indegree[node] == 0)
      ready.push(node);
  PnrIndex visited = 0;
  while (!ready.empty()) {
    const PnrIndex node = ready.front();
    ready.pop();
    ++visited;
    for (PnrIndex arc = graph.adjacencyOffsets[node];
         arc < graph.adjacencyOffsets[node + 1]; ++arc) {
      if (!active[arc])
        continue;
      const PnrIndex destination = graph.arcs[arc].destination;
      if (--indegree[destination] == 0)
        ready.push(destination);
    }
  }
  return visited == graph.nodeCount;
}

class DeterministicWords final {
public:
  explicit DeterministicWords(std::uint64_t state) : state_(state) {}

  std::uint64_t next() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * UINT64_C(0x2545f4914f6cdd1d);
  }

private:
  std::uint64_t state_;
};

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

void requireExactState(const IncrementalTopologicalOrder &order,
                       const std::vector<bool> &active,
                       const std::vector<PnrIndex> &expectedOrder,
                       const std::vector<PnrIndex> &expectedRanks) {
  for (PnrIndex arc = 0; arc < active.size(); ++arc)
    if (order.isArcActive(arc) != active[arc])
      fail("active arc bits diverge from the full-oracle model");
  if (!std::equal(order.order().begin(), order.order().end(),
                  expectedOrder.begin(), expectedOrder.end()) ||
      !std::equal(order.ranks().begin(), order.ranks().end(),
                  expectedRanks.begin(), expectedRanks.end()))
    fail("rollback did not restore the exact order and rank state");
  requireValidOrder(order);
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

void mixedUpdateRollbackRestoresEveryJournal() {
  const PotentialGraph graph =
      buildOffsetGraph(128, {1, 14, 27, 40, 53, 66, 79, 92});
  const std::vector<PnrIndex> initialArcs{0, 8};
  std::vector<bool> active(graph.arcs.size(), false);
  for (PnrIndex arc : initialArcs)
    active[arc] = true;
  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(graph.view(), initialArcs));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(graph.view()));
  const std::vector<PnrIndex> originalOrder(order->order().begin(),
                                            order->order().end());
  const std::vector<PnrIndex> originalRanks(order->ranks().begin(),
                                            order->ranks().end());

  auto transaction = take(order->beginTransaction(scratch));
  requireSuccess(transaction.removeArc(0));
  if (!take(transaction.insertArc(1)) || !take(transaction.insertArc(16)))
    fail("mixed journal fixture unexpectedly formed a cycle");
  requireSuccess(transaction.removeArc(16));
  if (!take(transaction.insertArc(0)))
    fail("mixed journal fixture could not restore its removed arc");
  transaction.rollback();
  requireExactState(*order, active, originalOrder, originalRanks);
}

void bulkCycleWitnessAndRollbackAreExact() {
  const PotentialGraph graph = buildOffsetGraph(16, {1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<PnrIndex> cycleArcs;
  for (PnrIndex arc = 0; arc < graph.arcs.size(); ++arc)
    if (graph.arcs[arc].destination ==
        (graph.arcs[arc].source + 1) % graph.nodeCount)
      cycleArcs.push_back(arc);
  if (cycleArcs.size() != graph.nodeCount)
    fail("bulk cycle fixture lost its ring");

  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(graph.view(), {}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(graph.view()));
  const std::vector<PnrIndex> originalOrder(order->order().begin(),
                                            order->order().end());
  const std::vector<PnrIndex> originalRanks(order->ranks().begin(),
                                            order->ranks().end());
  const std::vector<bool> active(graph.arcs.size(), false);

  auto transaction = take(order->beginTransaction(scratch));
  if (take(transaction.applyArcChanges({}, cycleArcs)))
    fail("bulk cycle was accepted");
  if (!std::equal(transaction.cycleWitness().begin(),
                  transaction.cycleWitness().end(), cycleArcs.begin(),
                  cycleArcs.end()))
    fail("bulk cycle witness is not the exact canonical ring");
  transaction.rollback();
  requireExactState(*order, active, originalOrder, originalRanks);
}

void randomizedUpdatesAgreeWithFullKahn() {
  const PotentialGraph graph =
      buildOffsetGraph(128, {1, 14, 27, 40, 53, 66, 79, 92});
  std::vector<bool> active(graph.arcs.size(), false);
  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(graph.view(), {}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(graph.view()));
  DeterministicWords words(UINT64_C(0x8a5cd789635d2dff));

  for (std::size_t operation = 0; operation < 4096; ++operation) {
    const PnrIndex arc =
        static_cast<PnrIndex>(words.next() % graph.arcs.size());
    const std::vector<PnrIndex> originalOrder(order->order().begin(),
                                              order->order().end());
    const std::vector<PnrIndex> originalRanks(order->ranks().begin(),
                                              order->ranks().end());
    auto transaction = take(order->beginTransaction(scratch));
    if (active[arc]) {
      requireSuccess(transaction.removeArc(arc));
      const bool commit = (words.next() & 3U) != 0;
      if (commit) {
        requireSuccess(transaction.commit());
        active[arc] = false;
      } else {
        transaction.rollback();
        requireExactState(*order, active, originalOrder, originalRanks);
      }
    } else {
      active[arc] = true;
      const bool fullAcyclic = fullKahnIsAcyclic(graph, active);
      const bool incrementalAcyclic = take(transaction.insertArc(arc));
      if (incrementalAcyclic != fullAcyclic)
        fail("incremental insertion disagrees with the full Kahn oracle");
      if (!incrementalAcyclic) {
        if (transaction.cycleWitness().empty())
          fail("cycle rejection omitted its deterministic witness");
        transaction.rollback();
        active[arc] = false;
        requireExactState(*order, active, originalOrder, originalRanks);
      } else if ((words.next() & 7U) == 0) {
        transaction.rollback();
        active[arc] = false;
        requireExactState(*order, active, originalOrder, originalRanks);
      } else {
        requireSuccess(transaction.commit());
      }
    }
    if (!fullKahnIsAcyclic(graph, active))
      fail("committed randomized graph is cyclic under full Kahn");
    for (PnrIndex selectedArc = 0; selectedArc < active.size(); ++selectedArc)
      if (order->isArcActive(selectedArc) != active[selectedArc])
        fail("randomized active arc bits diverge from the full-oracle model");
    requireValidOrder(*order);
  }
}

using Nanoseconds = std::chrono::nanoseconds;

std::int64_t median(std::vector<std::int64_t> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<PnrIndex> benchmarkBatch(const PotentialGraph &graph,
                                     std::size_t sample) {
  std::vector<PnrIndex> arcs;
  arcs.reserve(32);
  PnrIndex source = static_cast<PnrIndex>(101 + sample * 137 * 2);
  if ((source & 1U) == 0)
    ++source;
  for (PnrIndex local = 0; local < 32; ++local) {
    const PnrIndex selectedSource = source + local * 2;
    const PnrIndex begin = graph.adjacencyOffsets[selectedSource];
    const PnrIndex end = graph.adjacencyOffsets[selectedSource + 1];
    const auto found =
        std::find_if(graph.arcs.begin() + begin, graph.arcs.begin() + end,
                     [&](FrozenSpatialHandshakeArc arc) {
                       return arc.destination + 1 == selectedSource;
                     });
    if (found == graph.arcs.begin() + end)
      fail("pinned benchmark lost its rank-violating arc");
    arcs.push_back(static_cast<PnrIndex>(found - graph.arcs.begin()));
  }
  return arcs;
}

void applyInsertionBatch(IncrementalTopologicalOrder &order,
                         IncrementalTopologicalScratch &scratch,
                         const std::vector<PnrIndex> &arcs) {
  auto transaction = take(order.beginTransaction(scratch));
  for (PnrIndex arc : arcs)
    if (!take(transaction.insertArc(arc)))
      fail("forward benchmark insertion unexpectedly formed a cycle");
  requireSuccess(transaction.commit());
}

void applyRemovalBatch(IncrementalTopologicalOrder &order,
                       IncrementalTopologicalScratch &scratch,
                       const std::vector<PnrIndex> &arcs) {
  auto transaction = take(order.beginTransaction(scratch));
  for (PnrIndex arc : arcs)
    requireSuccess(transaction.removeArc(arc));
  requireSuccess(transaction.commit());
}

void requireReorderedBatch(const IncrementalTopologicalOrder &order,
                           const PotentialGraph &graph,
                           const std::vector<PnrIndex> &arcs) {
  for (PnrIndex arc : arcs) {
    const FrozenSpatialHandshakeArc endpoints = graph.arcs[arc];
    if (endpoints.source <= endpoints.destination)
      fail("pinned batch contains a rank-respecting potential arc");
    if (order.rank(endpoints.source) >= order.rank(endpoints.destination))
      fail("bounded repair did not reorder a rank-violating insertion");
  }
}

void pinnedScaleBenchmarkMeetsNativeContract() {
  if (!loom::test::allocationProbeIsCalibrated())
    fail("heap allocation probe did not observe its calibration calls");
  constexpr PnrIndex nodeCount = 10000;
  constexpr std::size_t potentialArcCount = 50000;
  PotentialGraph graph;
  graph.nodeCount = nodeCount;
  graph.adjacencyOffsets.reserve(nodeCount + 1);
  graph.adjacencyOffsets.push_back(0);
  for (PnrIndex source = 0; source < nodeCount; ++source) {
    std::vector<PnrIndex> destinations;
    destinations.reserve(5);
    if ((source & 1U) != 0)
      destinations.push_back(source - 1);
    const PnrIndex forwardCount = (source & 1U) != 0 ? 4 : 5;
    for (PnrIndex offset = 1; offset <= forwardCount; ++offset)
      destinations.push_back((source + offset) % nodeCount);
    std::sort(destinations.begin(), destinations.end());
    for (PnrIndex destination : destinations)
      graph.arcs.push_back({source, destination});
    graph.adjacencyOffsets.push_back(static_cast<PnrIndex>(graph.arcs.size()));
  }
  std::vector<PnrIndex> reverseCounts(nodeCount, 0);
  for (FrozenSpatialHandshakeArc arc : graph.arcs)
    ++reverseCounts[arc.destination];
  graph.reverseAdjacencyOffsets.reserve(nodeCount + 1);
  graph.reverseAdjacencyOffsets.push_back(0);
  for (PnrIndex count : reverseCounts)
    graph.reverseAdjacencyOffsets.push_back(
        graph.reverseAdjacencyOffsets.back() + count);
  graph.reverseArcOrdinals.resize(graph.arcs.size());
  std::vector<PnrIndex> reverseCursors(graph.reverseAdjacencyOffsets.begin(),
                                       graph.reverseAdjacencyOffsets.end() - 1);
  for (PnrIndex arc = 0; arc < graph.arcs.size(); ++arc)
    graph.reverseArcOrdinals[reverseCursors[graph.arcs[arc].destination]++] =
        arc;
  if (graph.arcs.size() != potentialArcCount)
    fail("pinned benchmark does not contain exactly 50,000 potential arcs");

  IncrementalTopologicalOrderHandle order =
      take(IncrementalTopologicalOrder::create(graph.view(), {}));
  IncrementalTopologicalScratch scratch;
  requireSuccess(scratch.prepare(graph.view()));
  const std::size_t retainedBytes = scratch.retainedStorageBytes();

  for (std::size_t sample = 0; sample < 5; ++sample) {
    const std::vector<PnrIndex> arcs = benchmarkBatch(graph, sample);
    applyInsertionBatch(*order, scratch, arcs);
    applyRemovalBatch(*order, scratch, arcs);
  }

  const std::vector<PnrIndex> allocationProbeArcs = benchmarkBatch(graph, 5);
  loom::test::startAllocationProbe();
  applyInsertionBatch(*order, scratch, allocationProbeArcs);
  requireReorderedBatch(*order, graph, allocationProbeArcs);
  applyRemovalBatch(*order, scratch, allocationProbeArcs);
  if (loom::test::stopAllocationProbe() != 0)
    fail("warm local update performed a heap allocation");
  if (scratch.retainedStorageBytes() != retainedBytes)
    fail("warm local update expanded retained scratch storage");

  std::vector<std::int64_t> incrementalSamples;
  std::vector<std::int64_t> fullSamples;
  incrementalSamples.reserve(25);
  fullSamples.reserve(25);
  for (std::size_t sample = 0; sample < 25; ++sample) {
    const std::vector<PnrIndex> arcs = benchmarkBatch(graph, sample + 6);
    const auto incrementalBegin = std::chrono::steady_clock::now();
    applyInsertionBatch(*order, scratch, arcs);
    const auto incrementalEnd = std::chrono::steady_clock::now();
    incrementalSamples.push_back(std::chrono::duration_cast<Nanoseconds>(
                                     incrementalEnd - incrementalBegin)
                                     .count());
    applyRemovalBatch(*order, scratch, arcs);

    applyInsertionBatch(*order, scratch, arcs);
    const auto fullBegin = std::chrono::steady_clock::now();
    requireSuccess(order->rebuild());
    const auto fullEnd = std::chrono::steady_clock::now();
    fullSamples.push_back(
        std::chrono::duration_cast<Nanoseconds>(fullEnd - fullBegin).count());
    applyRemovalBatch(*order, scratch, arcs);
  }

  const std::int64_t incrementalMedian = median(incrementalSamples);
  const std::int64_t fullMedian = median(fullSamples);
  if (incrementalMedian <= 0 || fullMedian < incrementalMedian * 5)
    fail("pinned incremental median is less than five times faster than full "
         "recomputation");
  requireValidOrder(*order);
  std::cout << "pinned topology benchmark: incremental median "
            << incrementalMedian << " ns, full median " << fullMedian
            << " ns, ratio "
            << static_cast<double>(fullMedian) / incrementalMedian << '\n';
}

} // namespace

int main() {
  rankViolatingInsertionReordersOnlyTheAffectedState();
  cycleWitnessAndRollbackAreDeterministic();
  deletionRollbackRestoresTheCommittedGraph();
  mixedUpdateRollbackRestoresEveryJournal();
  bulkCycleWitnessAndRollbackAreExact();
  randomizedUpdatesAgreeWithFullKahn();
  pinnedScaleBenchmarkMeetsNativeContract();
  return 0;
}
