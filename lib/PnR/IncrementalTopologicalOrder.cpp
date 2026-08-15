#include "IncrementalTopologicalOrder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

llvm::Error topologyError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid incremental topological state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

std::size_t bitWordCount(std::size_t bitCount) {
  return bitCount / 64 + (bitCount % 64 != 0 ? 1 : 0);
}

bool bitIsSet(llvm::ArrayRef<std::uint64_t> words, PnrIndex bit) {
  const std::size_t ordinal = static_cast<std::size_t>(bit);
  return (words[ordinal / 64] & (UINT64_C(1) << (ordinal % 64))) != 0;
}

void setBit(llvm::MutableArrayRef<std::uint64_t> words, PnrIndex bit,
            bool value) {
  const std::size_t ordinal = static_cast<std::size_t>(bit);
  const std::uint64_t mask = UINT64_C(1) << (ordinal % 64);
  if (value)
    words[ordinal / 64] |= mask;
  else
    words[ordinal / 64] &= ~mask;
}

llvm::Error validateGraph(IncrementalTopologicalGraphView graph) {
  const std::size_t nodeCount = static_cast<std::size_t>(graph.nodeCount);
  if (graph.adjacencyOffsets.size() != nodeCount + 1 ||
      graph.reverseAdjacencyOffsets.size() != nodeCount + 1 ||
      graph.adjacencyOffsets.front() != 0 ||
      graph.reverseAdjacencyOffsets.front() != 0 ||
      graph.adjacencyOffsets.back() != graph.arcs.size() ||
      graph.reverseAdjacencyOffsets.back() != graph.arcs.size() ||
      graph.reverseArcOrdinals.size() != graph.arcs.size())
    return topologyError("CSR shape is inconsistent");

  for (auto [ordinal, arc] : llvm::enumerate(graph.arcs)) {
    if (arc.source >= graph.nodeCount || arc.destination >= graph.nodeCount)
      return topologyError("arc endpoint is out of range");
    if (ordinal != 0) {
      const auto &previous = graph.arcs[ordinal - 1];
      if (std::pair(previous.source, previous.destination) >=
          std::pair(arc.source, arc.destination))
        return topologyError("arcs are not unique source-major records");
    }
    if (ordinal < graph.adjacencyOffsets[arc.source] ||
        ordinal >= graph.adjacencyOffsets[arc.source + 1])
      return topologyError("forward CSR does not own its arc");
  }

  std::vector<PnrIndex> reverseCounts(nodeCount, 0);
  std::vector<bool> reverseSeen(graph.arcs.size(), false);
  for (PnrIndex arcOrdinal : graph.reverseArcOrdinals) {
    if (arcOrdinal >= graph.arcs.size())
      return topologyError("reverse CSR arc is out of range");
    if (reverseSeen[arcOrdinal])
      return topologyError("reverse CSR contains a duplicate arc");
    reverseSeen[arcOrdinal] = true;
    ++reverseCounts[graph.arcs[arcOrdinal].destination];
  }
  for (std::size_t node = 0; node < nodeCount; ++node) {
    if (reverseCounts[node] != graph.reverseAdjacencyOffsets[node + 1] -
                                   graph.reverseAdjacencyOffsets[node])
      return topologyError("reverse CSR destination count is inconsistent");
    for (PnrIndex offset = graph.reverseAdjacencyOffsets[node];
         offset < graph.reverseAdjacencyOffsets[node + 1]; ++offset)
      if (graph.arcs[graph.reverseArcOrdinals[offset]].destination != node)
        return topologyError("reverse CSR owns an arc for another node");
  }
  return llvm::Error::success();
}

llvm::Expected<std::pair<std::vector<PnrIndex>, std::vector<PnrIndex>>>
buildCanonicalOrder(IncrementalTopologicalGraphView graph,
                    llvm::ArrayRef<std::uint64_t> activeArcBits) {
  const std::size_t nodeCount = static_cast<std::size_t>(graph.nodeCount);
  std::vector<PnrIndex> indegree(nodeCount, 0);
  for (auto [ordinal, arc] : llvm::enumerate(graph.arcs))
    if (bitIsSet(activeArcBits, static_cast<PnrIndex>(ordinal)))
      ++indegree[arc.destination];

  std::vector<PnrIndex> order;
  order.reserve(nodeCount);
  for (PnrIndex node = 0; node < graph.nodeCount; ++node)
    if (indegree[node] == 0)
      order.push_back(node);

  std::size_t cursor = 0;
  while (cursor < order.size()) {
    const PnrIndex node = order[cursor++];
    for (PnrIndex arc = graph.adjacencyOffsets[node];
         arc < graph.adjacencyOffsets[node + 1]; ++arc) {
      if (!bitIsSet(activeArcBits, arc))
        continue;
      const PnrIndex destination = graph.arcs[arc].destination;
      assert(indegree[destination] != 0);
      if (--indegree[destination] == 0)
        order.push_back(destination);
    }
  }
  if (order.size() != nodeCount)
    return topologyError("active graph contains a directed cycle");

  std::vector<PnrIndex> ranks(nodeCount, 0);
  for (auto [rank, node] : llvm::enumerate(order))
    ranks[node] = static_cast<PnrIndex>(rank);
  return std::make_pair(std::move(order), std::move(ranks));
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

IncrementalTopologicalScratch::~IncrementalTopologicalScratch() {
  if (activeTransaction_)
    activeTransaction_->rollback();
}

llvm::Error
IncrementalTopologicalScratch::prepare(IncrementalTopologicalGraphView graph) {
  if (activeTransaction_)
    return topologyError("cannot prepare scratch during a transaction");
  if (llvm::Error error = validateGraph(graph))
    return error;

  return prepareValidated(graph);
}

llvm::Error IncrementalTopologicalScratch::prepareValidated(
    IncrementalTopologicalGraphView graph) {
  if (activeTransaction_)
    return topologyError("cannot prepare scratch during a transaction");

  const std::size_t nodes = static_cast<std::size_t>(graph.nodeCount);
  const std::size_t arcs = graph.arcs.size();
  forwardMarks_.assign(nodes, 0);
  backwardMarks_.assign(nodes, 0);
  forwardParentArcs_.resize(nodes);
  rankJournalMarks_.assign(nodes, 0);
  arcJournalMarks_.assign(arcs, 0);
  cycleSearchStates_.assign(nodes, 0);
  cycleSearchParents_.resize(nodes);
  cycleSearchCursors_.resize(nodes);
  transactionEpoch_ = 0;
  searchEpoch_ = 0;
  resetTransaction();
  return llvm::Error::success();
}

std::size_t IncrementalTopologicalScratch::retainedStorageBytes() const {
  return retainedBytes(forwardMarks_) + retainedBytes(backwardMarks_) +
         retainedBytes(forwardParentArcs_) + retainedBytes(rankJournalMarks_) +
         retainedBytes(arcJournalMarks_) + retainedBytes(forwardWorklist_) +
         retainedBytes(backwardWorklist_) + retainedBytes(reorderBuffer_) +
         retainedBytes(touchedRanks_) + retainedBytes(oldRankNodes_) +
         retainedBytes(touchedArcs_) + retainedBytes(oldArcActive_) +
         retainedBytes(cycleWitness_) + retainedBytes(cycleSearchStates_) +
         retainedBytes(cycleSearchParents_) +
         retainedBytes(cycleSearchCursors_) + retainedBytes(cycleSearchStack_);
}

void IncrementalTopologicalScratch::beginTransaction() {
  resetTransaction();
  if (++transactionEpoch_ == 0) {
    std::fill(rankJournalMarks_.begin(), rankJournalMarks_.end(), 0);
    std::fill(arcJournalMarks_.begin(), arcJournalMarks_.end(), 0);
    transactionEpoch_ = 1;
  }
}

void IncrementalTopologicalScratch::resetTransaction() {
  forwardWorklist_.clear();
  backwardWorklist_.clear();
  reorderBuffer_.clear();
  touchedRanks_.clear();
  oldRankNodes_.clear();
  touchedArcs_.clear();
  oldArcActive_.clear();
  cycleWitness_.clear();
}

void IncrementalTopologicalScratch::beginSearch() {
  forwardWorklist_.clear();
  backwardWorklist_.clear();
  reorderBuffer_.clear();
  cycleWitness_.clear();
  if (++searchEpoch_ == 0) {
    std::fill(forwardMarks_.begin(), forwardMarks_.end(), 0);
    std::fill(backwardMarks_.begin(), backwardMarks_.end(), 0);
    searchEpoch_ = 1;
  }
}

llvm::Expected<IncrementalTopologicalOrderHandle>
IncrementalTopologicalOrder::create(
    IncrementalTopologicalGraphView graph,
    llvm::ArrayRef<PnrIndex> initiallyActiveArcs) {
  if (llvm::Error error = validateGraph(graph))
    return std::move(error);
  std::vector<std::uint64_t> activeBits(bitWordCount(graph.arcs.size()), 0);
  PnrIndex previous = 0;
  bool hasPrevious = false;
  for (PnrIndex arc : initiallyActiveArcs) {
    if (arc >= graph.arcs.size())
      return topologyError("initial active arc is out of range");
    if (hasPrevious && arc <= previous)
      return topologyError("initial active arcs are not unique canonical IDs");
    setBit(activeBits, arc, true);
    previous = arc;
    hasPrevious = true;
  }
  auto ordering = buildCanonicalOrder(graph, activeBits);
  if (!ordering)
    return ordering.takeError();
  return IncrementalTopologicalOrderHandle(new IncrementalTopologicalOrder(
      graph, std::move(activeBits), std::move(ordering->first),
      std::move(ordering->second)));
}

PnrIndex IncrementalTopologicalOrder::rank(PnrIndex node) const {
  assert(node < graph_.nodeCount);
  return ranks_[node];
}

bool IncrementalTopologicalOrder::isArcActive(PnrIndex arc) const {
  assert(arc < graph_.arcs.size());
  return bitIsSet(activeArcBits_, arc);
}

void IncrementalTopologicalOrder::setArcActive(PnrIndex arc, bool active) {
  setBit(activeArcBits_, arc, active);
}

llvm::Error IncrementalTopologicalOrder::rebuild() {
  if (activeTransaction_)
    return topologyError("cannot rebuild during a transaction");
  auto rebuilt = buildCanonicalOrder(graph_, activeArcBits_);
  if (!rebuilt)
    return rebuilt.takeError();
  order_ = std::move(rebuilt->first);
  ranks_ = std::move(rebuilt->second);
  return llvm::Error::success();
}

llvm::Error IncrementalTopologicalOrder::verify() const {
  if (activeArcBits_.size() != bitWordCount(graph_.arcs.size()) ||
      order_.size() != graph_.nodeCount || ranks_.size() != graph_.nodeCount)
    return topologyError("state shape does not match the potential graph");
  std::vector<bool> seen(graph_.nodeCount, false);
  for (auto [rankOrdinal, node] : llvm::enumerate(order_)) {
    if (node >= graph_.nodeCount || seen[node] || ranks_[node] != rankOrdinal)
      return topologyError("order and rank are not inverse permutations");
    seen[node] = true;
  }
  for (auto [arcOrdinal, arc] : llvm::enumerate(graph_.arcs))
    if (isArcActive(static_cast<PnrIndex>(arcOrdinal)) &&
        ranks_[arc.source] >= ranks_[arc.destination])
      return topologyError("active arc violates topological rank");
  return llvm::Error::success();
}

llvm::Expected<IncrementalTopologicalTransaction>
IncrementalTopologicalOrder::beginTransaction(
    IncrementalTopologicalScratch &scratch) & {
  if (activeTransaction_)
    return topologyError("topological state already has an active transaction");
  if (scratch.activeTransaction_)
    return topologyError(
        "topological scratch already has an active transaction");
  if (scratch.forwardMarks_.size() != graph_.nodeCount ||
      scratch.arcJournalMarks_.size() != graph_.arcs.size())
    return topologyError("topological scratch was not prepared for this graph");
  scratch.beginTransaction();
  return IncrementalTopologicalTransaction(shared_from_this(), scratch);
}

IncrementalTopologicalTransaction::IncrementalTopologicalTransaction(
    IncrementalTopologicalOrderHandle order,
    IncrementalTopologicalScratch &scratch)
    : order_(std::move(order)), scratch_(&scratch) {
  order_->activeTransaction_ = this;
  scratch_->activeTransaction_ = this;
}

IncrementalTopologicalTransaction::IncrementalTopologicalTransaction(
    IncrementalTopologicalTransaction &&other) noexcept
    : order_(std::move(other.order_)), scratch_(other.scratch_),
      hasCycle_(other.hasCycle_) {
  other.scratch_ = nullptr;
  if (order_)
    order_->activeTransaction_ = this;
  if (scratch_)
    scratch_->activeTransaction_ = this;
}

IncrementalTopologicalTransaction::~IncrementalTopologicalTransaction() {
  if (scratch_)
    rollback();
}

void IncrementalTopologicalTransaction::recordArc(PnrIndex arc) {
  if (scratch_->arcJournalMarks_[arc] == scratch_->transactionEpoch_)
    return;
  scratch_->arcJournalMarks_[arc] = scratch_->transactionEpoch_;
  scratch_->touchedArcs_.push_back(arc);
  scratch_->oldArcActive_.push_back(order_->isArcActive(arc));
}

void IncrementalTopologicalTransaction::recordRank(PnrIndex rankOrdinal) {
  if (scratch_->rankJournalMarks_[rankOrdinal] == scratch_->transactionEpoch_)
    return;
  scratch_->rankJournalMarks_[rankOrdinal] = scratch_->transactionEpoch_;
  scratch_->touchedRanks_.push_back(rankOrdinal);
  scratch_->oldRankNodes_.push_back(order_->order_[rankOrdinal]);
}

llvm::Expected<bool>
IncrementalTopologicalTransaction::insertArc(PnrIndex arcOrdinal) {
  if (!scratch_)
    return topologyError("transaction is no longer active");
  if (hasCycle_)
    return topologyError("transaction already contains a directed cycle");
  if (arcOrdinal >= order_->graph_.arcs.size())
    return topologyError("inserted arc is out of range");
  if (order_->isArcActive(arcOrdinal))
    return topologyError("inserted arc is already active");
  recordArc(arcOrdinal);
  order_->setArcActive(arcOrdinal, true);
  return repairAfterInsertion(arcOrdinal);
}

llvm::Error IncrementalTopologicalTransaction::removeArc(PnrIndex arcOrdinal) {
  if (!scratch_)
    return topologyError("transaction is no longer active");
  if (hasCycle_)
    return topologyError("transaction already contains a directed cycle");
  if (arcOrdinal >= order_->graph_.arcs.size())
    return topologyError("removed arc is out of range");
  if (!order_->isArcActive(arcOrdinal))
    return topologyError("removed arc is not active");
  recordArc(arcOrdinal);
  order_->setArcActive(arcOrdinal, false);
  return llvm::Error::success();
}

llvm::Expected<bool> IncrementalTopologicalTransaction::applyArcChanges(
    llvm::ArrayRef<PnrIndex> removals, llvm::ArrayRef<PnrIndex> insertions) {
  if (!scratch_)
    return topologyError("transaction is no longer active");
  if (hasCycle_)
    return topologyError("transaction already contains a directed cycle");

  const auto validate = [&](llvm::ArrayRef<PnrIndex> arcs,
                            bool expectedActive) -> llvm::Error {
    PnrIndex previous = 0;
    bool hasPrevious = false;
    for (PnrIndex arc : arcs) {
      if (arc >= order_->graph_.arcs.size())
        return topologyError("changed arc is out of range");
      if (hasPrevious && arc <= previous)
        return topologyError("changed arcs are not unique canonical IDs");
      if (order_->isArcActive(arc) != expectedActive)
        return topologyError(expectedActive ? "removed arc is not active"
                                            : "inserted arc is already active");
      previous = arc;
      hasPrevious = true;
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validate(removals, true))
    return std::move(error);
  if (llvm::Error error = validate(insertions, false))
    return std::move(error);

  const std::size_t changedCount = removals.size() + insertions.size();
  if (changedCount == 0)
    return true;
  const std::size_t nodeCount = order_->graph_.nodeCount;
  const std::size_t arcCount = order_->graph_.arcs.size();
  const std::size_t graphWork =
      nodeCount > std::numeric_limits<std::size_t>::max() - arcCount
          ? std::numeric_limits<std::size_t>::max()
          : nodeCount + arcCount;
  const bool rebuildOnce = changedCount > graphWork / changedCount;
  if (!rebuildOnce) {
    for (PnrIndex arc : removals)
      if (llvm::Error error = removeArc(arc))
        return std::move(error);
    for (PnrIndex arc : insertions) {
      auto inserted = insertArc(arc);
      if (!inserted)
        return inserted.takeError();
      if (!*inserted)
        return false;
    }
    return true;
  }

  for (PnrIndex arc : removals) {
    recordArc(arc);
    order_->setArcActive(arc, false);
  }
  for (PnrIndex arc : insertions) {
    recordArc(arc);
    order_->setArcActive(arc, true);
  }
  auto rebuilt = buildCanonicalOrder(order_->graph_, order_->activeArcBits_);
  if (!rebuilt) {
    llvm::consumeError(rebuilt.takeError());
    buildCycleWitness();
    hasCycle_ = true;
    return false;
  }
  for (PnrIndex rank = 0; rank < order_->graph_.nodeCount; ++rank)
    if (order_->order_[rank] != rebuilt->first[rank])
      recordRank(rank);
  order_->order_ = std::move(rebuilt->first);
  order_->ranks_ = std::move(rebuilt->second);
  return true;
}

llvm::Expected<bool>
IncrementalTopologicalTransaction::repairAfterInsertion(PnrIndex arcOrdinal) {
  const auto graph = order_->graph_;
  const FrozenSpatialHandshakeArc inserted = graph.arcs[arcOrdinal];
  const PnrIndex sourceRank = order_->ranks_[inserted.source];
  const PnrIndex destinationRank = order_->ranks_[inserted.destination];
  if (sourceRank < destinationRank)
    return true;

  scratch_->beginSearch();
  const std::uint64_t epoch = scratch_->searchEpoch_;
  scratch_->forwardMarks_[inserted.destination] = epoch;
  scratch_->forwardWorklist_.push_back(inserted.destination);
  std::size_t cursor = 0;
  while (cursor < scratch_->forwardWorklist_.size()) {
    const PnrIndex node = scratch_->forwardWorklist_[cursor++];
    for (PnrIndex arc = graph.adjacencyOffsets[node];
         arc < graph.adjacencyOffsets[node + 1]; ++arc) {
      if (!order_->isArcActive(arc))
        continue;
      const PnrIndex destination = graph.arcs[arc].destination;
      if (order_->ranks_[destination] > sourceRank ||
          scratch_->forwardMarks_[destination] == epoch)
        continue;
      scratch_->forwardMarks_[destination] = epoch;
      scratch_->forwardParentArcs_[destination] = arc;
      scratch_->forwardWorklist_.push_back(destination);
    }
  }

  if (scratch_->forwardMarks_[inserted.source] == epoch) {
    scratch_->cycleWitness_.push_back(arcOrdinal);
    PnrIndex node = inserted.source;
    while (node != inserted.destination) {
      const PnrIndex parentArc = scratch_->forwardParentArcs_[node];
      scratch_->cycleWitness_.push_back(parentArc);
      node = graph.arcs[parentArc].source;
    }
    llvm::sort(scratch_->cycleWitness_);
    scratch_->cycleWitness_.erase(std::unique(scratch_->cycleWitness_.begin(),
                                              scratch_->cycleWitness_.end()),
                                  scratch_->cycleWitness_.end());
    hasCycle_ = true;
    return false;
  }

  scratch_->backwardMarks_[inserted.source] = epoch;
  scratch_->backwardWorklist_.push_back(inserted.source);
  cursor = 0;
  while (cursor < scratch_->backwardWorklist_.size()) {
    const PnrIndex node = scratch_->backwardWorklist_[cursor++];
    for (PnrIndex offset = graph.reverseAdjacencyOffsets[node];
         offset < graph.reverseAdjacencyOffsets[node + 1]; ++offset) {
      const PnrIndex arc = graph.reverseArcOrdinals[offset];
      if (!order_->isArcActive(arc))
        continue;
      const PnrIndex source = graph.arcs[arc].source;
      if (order_->ranks_[source] < destinationRank ||
          scratch_->backwardMarks_[source] == epoch)
        continue;
      scratch_->backwardMarks_[source] = epoch;
      scratch_->backwardWorklist_.push_back(source);
    }
  }

  for (PnrIndex node : scratch_->forwardWorklist_)
    if (scratch_->backwardMarks_[node] == epoch)
      return topologyError("bounded searches overlap without a cycle witness");

  const std::size_t firstRank = destinationRank;
  const std::size_t lastRank = sourceRank;
  const std::size_t intervalSize = lastRank - firstRank + 1;
  scratch_->reorderBuffer_.reserve(intervalSize);
  for (std::size_t rank = firstRank; rank <= lastRank; ++rank) {
    const PnrIndex node = order_->order_[rank];
    if (scratch_->backwardMarks_[node] == epoch)
      scratch_->reorderBuffer_.push_back(node);
  }
  for (std::size_t rank = firstRank; rank <= lastRank; ++rank) {
    const PnrIndex node = order_->order_[rank];
    if (scratch_->backwardMarks_[node] != epoch &&
        scratch_->forwardMarks_[node] != epoch)
      scratch_->reorderBuffer_.push_back(node);
  }
  for (std::size_t rank = firstRank; rank <= lastRank; ++rank) {
    const PnrIndex node = order_->order_[rank];
    if (scratch_->forwardMarks_[node] == epoch)
      scratch_->reorderBuffer_.push_back(node);
  }
  if (scratch_->reorderBuffer_.size() != intervalSize)
    return topologyError("bounded reorder lost a rank-interval node");

  for (std::size_t offset = 0; offset < intervalSize; ++offset) {
    const PnrIndex rankOrdinal = static_cast<PnrIndex>(firstRank + offset);
    const PnrIndex node = scratch_->reorderBuffer_[offset];
    if (order_->order_[rankOrdinal] == node)
      continue;
    recordRank(rankOrdinal);
    order_->order_[rankOrdinal] = node;
    order_->ranks_[node] = rankOrdinal;
  }
  if (order_->ranks_[inserted.source] >= order_->ranks_[inserted.destination])
    return topologyError("bounded reorder did not orient the inserted arc");
  return true;
}

void IncrementalTopologicalTransaction::buildCycleWitness() {
  const auto graph = order_->graph_;
  std::fill(scratch_->cycleSearchStates_.begin(),
            scratch_->cycleSearchStates_.end(), 0);
  scratch_->cycleSearchStack_.clear();
  scratch_->cycleWitness_.clear();
  for (PnrIndex root = 0; root < graph.nodeCount; ++root) {
    if (scratch_->cycleSearchStates_[root] != 0)
      continue;
    scratch_->cycleSearchStates_[root] = 1;
    scratch_->cycleSearchParents_[root] = getInvalidPnrIndex();
    scratch_->cycleSearchCursors_[root] = graph.adjacencyOffsets[root];
    scratch_->cycleSearchStack_.push_back(root);
    while (!scratch_->cycleSearchStack_.empty()) {
      const PnrIndex node = scratch_->cycleSearchStack_.back();
      PnrIndex &cursor = scratch_->cycleSearchCursors_[node];
      const PnrIndex end = graph.adjacencyOffsets[node + 1];
      while (cursor < end && !order_->isArcActive(cursor))
        ++cursor;
      if (cursor == end) {
        scratch_->cycleSearchStates_[node] = 2;
        scratch_->cycleSearchStack_.pop_back();
        continue;
      }
      const PnrIndex arc = cursor++;
      const PnrIndex destination = graph.arcs[arc].destination;
      if (scratch_->cycleSearchStates_[destination] == 0) {
        scratch_->cycleSearchStates_[destination] = 1;
        scratch_->cycleSearchParents_[destination] = arc;
        scratch_->cycleSearchCursors_[destination] =
            graph.adjacencyOffsets[destination];
        scratch_->cycleSearchStack_.push_back(destination);
        continue;
      }
      if (scratch_->cycleSearchStates_[destination] != 1)
        continue;
      scratch_->cycleWitness_.push_back(arc);
      PnrIndex ancestor = node;
      while (ancestor != destination) {
        const PnrIndex parentArc = scratch_->cycleSearchParents_[ancestor];
        assert(parentArc != getInvalidPnrIndex());
        scratch_->cycleWitness_.push_back(parentArc);
        ancestor = graph.arcs[parentArc].source;
      }
      llvm::sort(scratch_->cycleWitness_);
      scratch_->cycleWitness_.erase(std::unique(scratch_->cycleWitness_.begin(),
                                                scratch_->cycleWitness_.end()),
                                    scratch_->cycleWitness_.end());
      return;
    }
  }
  llvm_unreachable("failed to recover a cycle from a cyclic active graph");
}

llvm::ArrayRef<PnrIndex>
IncrementalTopologicalTransaction::cycleWitness() const {
  return scratch_ ? llvm::ArrayRef<PnrIndex>(scratch_->cycleWitness_)
                  : llvm::ArrayRef<PnrIndex>();
}

llvm::Error IncrementalTopologicalTransaction::commit() {
  if (!scratch_)
    return topologyError("transaction is no longer active");
  if (hasCycle_)
    return topologyError("cannot commit a cyclic active graph");
  finish();
  return llvm::Error::success();
}

void IncrementalTopologicalTransaction::rollback() noexcept {
  if (!scratch_)
    return;
  for (std::size_t index = 0; index < scratch_->touchedRanks_.size(); ++index)
    order_->order_[scratch_->touchedRanks_[index]] =
        scratch_->oldRankNodes_[index];
  for (std::size_t index = 0; index < scratch_->touchedRanks_.size(); ++index)
    order_->ranks_[scratch_->oldRankNodes_[index]] =
        scratch_->touchedRanks_[index];
  for (std::size_t index = 0; index < scratch_->touchedArcs_.size(); ++index)
    order_->setArcActive(scratch_->touchedArcs_[index],
                         scratch_->oldArcActive_[index] != 0);
  finish();
}

void IncrementalTopologicalTransaction::finish() {
  order_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  order_.reset();
}
