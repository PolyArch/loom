#include "PnR/HandshakeCandidateState.h"

#include "IncrementalTopologicalOrder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace loom::pnr::detail {

struct HandshakeCandidateScratchStorage final {
  IncrementalTopologicalScratch topologyScratch;
  std::optional<IncrementalTopologicalTransaction> topologyTransaction;
  std::vector<PnrIndex> removedArcs;
  std::vector<PnrIndex> insertedArcs;
};

} // namespace loom::pnr::detail

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid handshake candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

detail::IncrementalTopologicalGraphView
topologyView(const FrozenSpatialHandshakeIndex &index) {
  return {index.nodeCount(), index.arcs(), index.adjacencyOffsets(),
          index.reverseAdjacencyOffsets(), index.reverseArcOrdinals()};
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Error increment(PnrIndex &value, llvm::StringRef subject) {
  if (value == std::numeric_limits<PnrIndex>::max())
    return candidateError(subject + " refcount overflows PnrIndex");
  ++value;
  return llvm::Error::success();
}

struct RebuiltHandshakeProjection final {
  std::vector<PnrIndex> fragmentRefcounts;
  std::vector<PnrIndex> arcRefcounts;
  std::vector<PnrIndex> traversalRefcounts;
  std::vector<PnrIndex> allGroupSelectedWitnessCounts;
  std::vector<PnrIndex> activeArcs;
};

llvm::Expected<RebuiltHandshakeProjection>
rebuildHandshakeProjection(const FrozenSpatialHandshakeIndex &index,
                           llvm::ArrayRef<PnrIndex> selectedFragments,
                           llvm::ArrayRef<PnrIndex> traversalUses) {
  const auto traversalFragmentOffsets = index.traversalFragmentOffsets();
  const auto traversalGroupOffsets = index.traversalAllGroupOffsets();
  if (traversalFragmentOffsets.empty() || traversalGroupOffsets.empty())
    return candidateError("projection traversal offsets are empty");
  const std::size_t traversalCount = traversalFragmentOffsets.size() - 1;
  if (traversalUses.size() != traversalCount ||
      traversalGroupOffsets.size() != traversalCount + 1)
    return candidateError(
        "projection traversal dimension does not match its index");

  RebuiltHandshakeProjection result;
  result.fragmentRefcounts.assign(index.fragments().size(), 0);
  result.arcRefcounts.assign(index.arcs().size(), 0);
  result.traversalRefcounts.assign(traversalUses.begin(), traversalUses.end());
  result.allGroupSelectedWitnessCounts.assign(index.allTraversalGroups().size(),
                                              0);
  const auto activateFragment = [&](PnrIndex fragment) -> llvm::Error {
    if (fragment >= result.fragmentRefcounts.size())
      return candidateError("projection fragment is out of range");
    return increment(result.fragmentRefcounts[fragment], "fragment");
  };
  for (PnrIndex fragment : index.fixedFragments())
    if (llvm::Error error = activateFragment(fragment))
      return std::move(error);
  for (PnrIndex fragment : selectedFragments)
    if (llvm::Error error = activateFragment(fragment))
      return std::move(error);

  for (PnrIndex traversal = 0; traversal < traversalCount; ++traversal) {
    if (traversalUses[traversal] == 0)
      continue;
    for (PnrIndex fragment : index.traversalFragments().slice(
             traversalFragmentOffsets[traversal],
             traversalFragmentOffsets[traversal + 1] -
                 traversalFragmentOffsets[traversal]))
      if (llvm::Error error = activateFragment(fragment))
        return std::move(error);
    for (PnrIndex group : index.traversalAllGroups().slice(
             traversalGroupOffsets[traversal],
             traversalGroupOffsets[traversal + 1] -
                 traversalGroupOffsets[traversal])) {
      if (group >= result.allGroupSelectedWitnessCounts.size())
        return candidateError("projection traversal group is out of range");
      if (llvm::Error error =
              increment(result.allGroupSelectedWitnessCounts[group],
                        "all-traversal witness"))
        return std::move(error);
    }
  }
  for (auto [groupOrdinal, group] :
       llvm::enumerate(index.allTraversalGroups())) {
    const PnrIndex selected =
        result.allGroupSelectedWitnessCounts[groupOrdinal];
    if (selected > group.witnessCount)
      return candidateError(
          "projection selects excess all-traversal witnesses");
    if (selected == group.witnessCount)
      if (llvm::Error error = activateFragment(group.fragment))
        return std::move(error);
  }

  const auto fragments = index.fragments();
  const auto fragmentArcs = index.fragmentArcOrdinals();
  for (PnrIndex fragment = 0; fragment < fragments.size(); ++fragment) {
    if (result.fragmentRefcounts[fragment] == 0)
      continue;
    const FrozenSpatialHandshakeFragment record = fragments[fragment];
    if (record.contributionOffset > fragmentArcs.size() ||
        record.contributionCount >
            fragmentArcs.size() - record.contributionOffset)
      return candidateError("projection fragment incidence is out of range");
    for (PnrIndex arc : fragmentArcs.slice(record.contributionOffset,
                                           record.contributionCount)) {
      if (arc >= result.arcRefcounts.size())
        return candidateError("projection fragment arc is out of range");
      if (llvm::Error error = increment(result.arcRefcounts[arc], "arc"))
        return std::move(error);
    }
  }
  result.activeArcs.reserve(result.arcRefcounts.size());
  for (auto [arc, refcount] : llvm::enumerate(result.arcRefcounts))
    if (refcount != 0)
      result.activeArcs.push_back(static_cast<PnrIndex>(arc));
  return result;
}

} // namespace

llvm::Expected<bool> loom::pnr::independentlyVerifyHandshakeProjectionAcyclic(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses) {
  auto projection =
      rebuildHandshakeProjection(index, selectedFragments, traversalUses);
  if (!projection)
    return projection.takeError();
  const auto arcs = index.arcs();
  std::vector<PnrIndex> indegree(index.nodeCount(), 0);
  for (PnrIndex arc : projection->activeArcs)
    if (llvm::Error error = increment(indegree[arcs[arc].destination],
                                      "handshake node indegree"))
      return std::move(error);

  std::vector<PnrIndex> ready;
  ready.reserve(index.nodeCount());
  for (PnrIndex node = 0; node < index.nodeCount(); ++node)
    if (indegree[node] == 0)
      ready.push_back(node);
  std::size_t visited = 0;
  const auto adjacencyOffsets = index.adjacencyOffsets();
  if (adjacencyOffsets.size() !=
      static_cast<std::size_t>(index.nodeCount()) + 1)
    return candidateError(
        "independent projection adjacency shape is inconsistent");
  while (visited < ready.size()) {
    const PnrIndex node = ready[visited];
    ++visited;
    for (PnrIndex arc = adjacencyOffsets[node];
         arc < adjacencyOffsets[node + 1]; ++arc) {
      if (arc >= arcs.size())
        return candidateError(
            "independent projection adjacency arc is out of range");
      if (projection->arcRefcounts[arc] == 0)
        continue;
      PnrIndex &destinationIndegree = indegree[arcs[arc].destination];
      if (destinationIndegree == 0)
        return candidateError("independent projection indegree underflows");
      if (--destinationIndegree == 0)
        ready.push_back(arcs[arc].destination);
    }
  }
  return visited == index.nodeCount();
}

HandshakeCandidateScratch::HandshakeCandidateScratch()
    : storage_(std::make_unique<detail::HandshakeCandidateScratchStorage>()) {}

HandshakeCandidateScratch::~HandshakeCandidateScratch() {
  if (activeTransaction_)
    activeTransaction_->rollback();
}

llvm::Error
HandshakeCandidateScratch::prepare(const FrozenSpatialHandshakeIndex &index) {
  if (activeTransaction_)
    return candidateError("cannot prepare scratch during a transaction");
  if (index.traversalFragmentOffsets().empty())
    return candidateError("handshake traversal offsets are empty");
  if (llvm::Error error =
          storage_->topologyScratch.prepareValidated(topologyView(index)))
    return error;

  const std::size_t traversalCount =
      index.traversalFragmentOffsets().size() - 1;
  fragmentJournalMarks_.assign(index.fragments().size(), 0);
  arcJournalMarks_.assign(index.arcs().size(), 0);
  traversalJournalMarks_.assign(traversalCount, 0);
  groupJournalMarks_.assign(index.allTraversalGroups().size(), 0);
  transactionEpoch_ = 0;
  resetTransaction();
  return llvm::Error::success();
}

std::size_t HandshakeCandidateScratch::retainedStorageBytes() const {
  return storage_->topologyScratch.retainedStorageBytes() +
         retainedBytes(fragmentJournalMarks_) +
         retainedBytes(arcJournalMarks_) +
         retainedBytes(traversalJournalMarks_) +
         retainedBytes(groupJournalMarks_) + retainedBytes(fragmentDeltas_) +
         retainedBytes(arcDeltas_) + retainedBytes(traversalDeltas_) +
         retainedBytes(groupDeltas_) + retainedBytes(storage_->removedArcs) +
         retainedBytes(storage_->insertedArcs);
}

void HandshakeCandidateScratch::beginTransaction() {
  resetTransaction();
  if (++transactionEpoch_ == 0) {
    std::fill(fragmentJournalMarks_.begin(), fragmentJournalMarks_.end(), 0);
    std::fill(arcJournalMarks_.begin(), arcJournalMarks_.end(), 0);
    std::fill(traversalJournalMarks_.begin(), traversalJournalMarks_.end(), 0);
    std::fill(groupJournalMarks_.begin(), groupJournalMarks_.end(), 0);
    transactionEpoch_ = 1;
  }
}

void HandshakeCandidateScratch::resetTransaction() {
  fragmentDeltas_.clear();
  arcDeltas_.clear();
  traversalDeltas_.clear();
  groupDeltas_.clear();
  storage_->removedArcs.clear();
  storage_->insertedArcs.clear();
}

llvm::Expected<HandshakeCandidateStateHandle>
HandshakeCandidateState::create(FrozenSpatialHandshakeIndexHandle index) {
  if (!index)
    return candidateError("FrozenSpatialHandshakeIndex owner is null");
  if (index->traversalFragmentOffsets().empty())
    return candidateError("handshake traversal offsets are empty");
  const std::size_t traversalCount =
      index->traversalFragmentOffsets().size() - 1;
  return create(std::move(index), {}, std::vector<PnrIndex>(traversalCount, 0));
}

llvm::Expected<HandshakeCandidateStateHandle>
HandshakeCandidateState::create(FrozenSpatialHandshakeIndexHandle index,
                                llvm::ArrayRef<PnrIndex> selectedFragments,
                                llvm::ArrayRef<PnrIndex> traversalUses) {
  if (!index)
    return candidateError("FrozenSpatialHandshakeIndex owner is null");
  auto projection =
      rebuildHandshakeProjection(*index, selectedFragments, traversalUses);
  if (!projection)
    return projection.takeError();
  auto topology = detail::IncrementalTopologicalOrder::create(
      topologyView(*index), projection->activeArcs);
  if (!topology)
    return topology.takeError();
  return HandshakeCandidateStateHandle(new HandshakeCandidateState(
      std::move(index), std::move(*topology),
      std::move(projection->fragmentRefcounts),
      std::move(projection->arcRefcounts),
      std::move(projection->traversalRefcounts),
      std::move(projection->allGroupSelectedWitnessCounts)));
}

PnrIndex HandshakeCandidateState::fragmentRefcount(PnrIndex fragment) const {
  assert(fragment < fragmentRefcounts_.size());
  return fragmentRefcounts_[fragment];
}

PnrIndex HandshakeCandidateState::arcRefcount(PnrIndex arc) const {
  assert(arc < arcRefcounts_.size());
  return arcRefcounts_[arc];
}

bool HandshakeCandidateState::isArcActive(PnrIndex arc) const {
  return topology_->isArcActive(arc);
}

PnrIndex HandshakeCandidateState::traversalRefcount(PnrIndex traversal) const {
  assert(traversal < traversalRefcounts_.size());
  return traversalRefcounts_[traversal];
}

bool HandshakeCandidateState::isTraversalSelected(PnrIndex traversal) const {
  return traversalRefcount(traversal) != 0;
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateState::topologicalOrder() const {
  return topology_->order();
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateState::topologicalRanks() const {
  return topology_->ranks();
}

llvm::Error HandshakeCandidateState::verify() const {
  if (!index_ || fragmentRefcounts_.size() != index_->fragments().size() ||
      arcRefcounts_.size() != index_->arcs().size() ||
      traversalRefcounts_.size() + 1 !=
          index_->traversalFragmentOffsets().size() ||
      allGroupSelectedWitnessCounts_.size() !=
          index_->allTraversalGroups().size())
    return candidateError("candidate shape does not match its frozen index");
  if (llvm::Error error = topology_->verify())
    return error;

  std::vector<PnrIndex> expectedArcRefcounts(index_->arcs().size(), 0);
  for (auto [fragmentOrdinal, refcount] : llvm::enumerate(fragmentRefcounts_)) {
    if (refcount == 0)
      continue;
    const auto fragment = index_->fragments()[fragmentOrdinal];
    for (PnrIndex arc : index_->fragmentArcOrdinals().slice(
             fragment.contributionOffset, fragment.contributionCount)) {
      if (expectedArcRefcounts[arc] == std::numeric_limits<PnrIndex>::max())
        return candidateError("recomputed arc refcount overflows PnrIndex");
      ++expectedArcRefcounts[arc];
    }
  }
  if (expectedArcRefcounts != arcRefcounts_)
    return candidateError("arc refcounts do not match active fragments");
  for (auto [arc, refcount] : llvm::enumerate(arcRefcounts_))
    if (topology_->isArcActive(static_cast<PnrIndex>(arc)) != (refcount != 0))
      return candidateError("active arc bit does not match its refcount");

  for (auto [groupOrdinal, group] :
       llvm::enumerate(index_->allTraversalGroups())) {
    PnrIndex selectedWitnesses = 0;
    for (PnrIndex traversal : index_->allTraversalGroupWitnesses().slice(
             group.witnessOffset, group.witnessCount))
      if (isTraversalSelected(traversal))
        ++selectedWitnesses;
    if (selectedWitnesses != allGroupSelectedWitnessCounts_[groupOrdinal])
      return candidateError("all-traversal witness count is stale");
    if (selectedWitnesses == group.witnessCount &&
        fragmentRefcounts_[group.fragment] == 0)
      return candidateError("satisfied all-traversal group is inactive");
  }
  for (PnrIndex fragment : index_->fixedFragments())
    if (fragmentRefcounts_[fragment] == 0)
      return candidateError("fixed handshake fragment is inactive");
  return llvm::Error::success();
}

llvm::Expected<HandshakeCandidateTransaction>
HandshakeCandidateState::beginTransaction(
    HandshakeCandidateScratch &scratch) & {
  if (activeTransaction_)
    return candidateError("candidate already has an active transaction");
  if (scratch.activeTransaction_)
    return candidateError("scratch already has an active transaction");
  if (scratch.fragmentJournalMarks_.size() != fragmentRefcounts_.size() ||
      scratch.arcJournalMarks_.size() != arcRefcounts_.size() ||
      scratch.traversalJournalMarks_.size() !=
          index_->traversalFragmentOffsets().size() - 1 ||
      scratch.groupJournalMarks_.size() !=
          allGroupSelectedWitnessCounts_.size())
    return candidateError("scratch was not prepared for this candidate");
  scratch.beginTransaction();
  auto topologyTransaction =
      topology_->beginTransaction(scratch.storage_->topologyScratch);
  if (!topologyTransaction)
    return topologyTransaction.takeError();
  scratch.storage_->topologyTransaction.emplace(
      std::move(*topologyTransaction));
  return HandshakeCandidateTransaction(shared_from_this(), scratch);
}

HandshakeCandidateTransaction::HandshakeCandidateTransaction(
    HandshakeCandidateStateHandle state, HandshakeCandidateScratch &scratch)
    : state_(std::move(state)), scratch_(&scratch) {
  state_->activeTransaction_ = this;
  scratch_->activeTransaction_ = this;
}

HandshakeCandidateTransaction::HandshakeCandidateTransaction(
    HandshakeCandidateTransaction &&other) noexcept
    : state_(std::move(other.state_)), scratch_(other.scratch_),
      closed_(other.closed_), cycle_(other.cycle_) {
  other.scratch_ = nullptr;
  if (state_)
    state_->activeTransaction_ = this;
  if (scratch_)
    scratch_->activeTransaction_ = this;
}

HandshakeCandidateTransaction::~HandshakeCandidateTransaction() {
  if (scratch_)
    rollback();
}

llvm::Error HandshakeCandidateTransaction::validateFragmentSlice(
    llvm::ArrayRef<PnrIndex> fragments) const {
  PnrIndex previous = 0;
  bool hasPrevious = false;
  for (PnrIndex fragment : fragments) {
    if (fragment >= state_->fragmentRefcounts_.size())
      return candidateError("fragment is out of range");
    if (hasPrevious && fragment <= previous)
      return candidateError("fragment slice is not unique canonical order");
    previous = fragment;
    hasPrevious = true;
  }
  return llvm::Error::success();
}

void HandshakeCandidateTransaction::recordFragment(PnrIndex fragment) {
  if (scratch_->fragmentJournalMarks_[fragment] == scratch_->transactionEpoch_)
    return;
  scratch_->fragmentJournalMarks_[fragment] = scratch_->transactionEpoch_;
  scratch_->fragmentDeltas_.push_back(
      {fragment, state_->fragmentRefcounts_[fragment]});
}

void HandshakeCandidateTransaction::recordArc(PnrIndex arc) {
  if (scratch_->arcJournalMarks_[arc] == scratch_->transactionEpoch_)
    return;
  scratch_->arcJournalMarks_[arc] = scratch_->transactionEpoch_;
  scratch_->arcDeltas_.push_back({arc, state_->arcRefcounts_[arc]});
}

void HandshakeCandidateTransaction::recordTraversal(PnrIndex traversal) {
  if (scratch_->traversalJournalMarks_[traversal] ==
      scratch_->transactionEpoch_)
    return;
  scratch_->traversalJournalMarks_[traversal] = scratch_->transactionEpoch_;
  scratch_->traversalDeltas_.push_back(
      {traversal, state_->traversalRefcounts_[traversal]});
}

void HandshakeCandidateTransaction::recordGroup(PnrIndex group) {
  if (scratch_->groupJournalMarks_[group] == scratch_->transactionEpoch_)
    return;
  scratch_->groupJournalMarks_[group] = scratch_->transactionEpoch_;
  scratch_->groupDeltas_.push_back(
      {group, state_->allGroupSelectedWitnessCounts_[group]});
}

llvm::Error HandshakeCandidateTransaction::changeFragment(PnrIndex fragment,
                                                          bool add) {
  recordFragment(fragment);
  PnrIndex &fragmentRefcount = state_->fragmentRefcounts_[fragment];
  const bool wasActive = fragmentRefcount != 0;
  if (add) {
    if (llvm::Error error = increment(fragmentRefcount, "fragment"))
      return error;
  } else {
    if (fragmentRefcount == 0)
      return candidateError("fragment refcount underflows");
    --fragmentRefcount;
  }
  const bool isActive = fragmentRefcount != 0;
  if (wasActive == isActive)
    return llvm::Error::success();

  const FrozenSpatialHandshakeFragment record =
      state_->index_->fragments()[fragment];
  for (PnrIndex arc : state_->index_->fragmentArcOrdinals().slice(
           record.contributionOffset, record.contributionCount)) {
    recordArc(arc);
    PnrIndex &arcRefcount = state_->arcRefcounts_[arc];
    if (isActive) {
      if (llvm::Error error = increment(arcRefcount, "arc"))
        return error;
    } else {
      if (arcRefcount == 0)
        return candidateError("arc refcount underflows");
      --arcRefcount;
    }
  }
  return llvm::Error::success();
}

llvm::Error HandshakeCandidateTransaction::addFragments(
    llvm::ArrayRef<PnrIndex> fragments) {
  if (!scratch_ || closed_)
    return candidateError("transaction is not collecting changes");
  if (llvm::Error error = validateFragmentSlice(fragments))
    return error;
  for (PnrIndex fragment : fragments)
    if (llvm::Error error = changeFragment(fragment, true))
      return error;
  return llvm::Error::success();
}

llvm::Error HandshakeCandidateTransaction::removeFragments(
    llvm::ArrayRef<PnrIndex> fragments) {
  if (!scratch_ || closed_)
    return candidateError("transaction is not collecting changes");
  if (llvm::Error error = validateFragmentSlice(fragments))
    return error;
  for (PnrIndex fragment : fragments)
    if (llvm::Error error = changeFragment(fragment, false))
      return error;
  return llvm::Error::success();
}

llvm::Error HandshakeCandidateTransaction::addTraversalUses(PnrIndex traversal,
                                                            PnrIndex count) {
  if (!scratch_ || closed_)
    return candidateError("transaction is not collecting changes");
  const std::size_t traversalCount =
      state_->index_->traversalFragmentOffsets().size() - 1;
  if (traversal >= traversalCount)
    return candidateError("selected traversal is out of range");
  if (count == 0)
    return candidateError("traversal use increment is zero");
  PnrIndex &refcount = state_->traversalRefcounts_[traversal];
  if (count > std::numeric_limits<PnrIndex>::max() - refcount)
    return candidateError("traversal use refcount overflows PnrIndex");
  recordTraversal(traversal);
  const bool activate = refcount == 0;
  refcount += count;
  if (!activate)
    return llvm::Error::success();

  const auto fragmentOffsets = state_->index_->traversalFragmentOffsets();
  if (llvm::Error error =
          addFragments(state_->index_->traversalFragments().slice(
              fragmentOffsets[traversal],
              fragmentOffsets[traversal + 1] - fragmentOffsets[traversal])))
    return error;
  const auto groupOffsets = state_->index_->traversalAllGroupOffsets();
  for (PnrIndex group : state_->index_->traversalAllGroups().slice(
           groupOffsets[traversal],
           groupOffsets[traversal + 1] - groupOffsets[traversal])) {
    recordGroup(group);
    PnrIndex &selected = state_->allGroupSelectedWitnessCounts_[group];
    if (llvm::Error error = increment(selected, "all-traversal witness"))
      return error;
    const auto contract = state_->index_->allTraversalGroups()[group];
    if (selected > contract.witnessCount)
      return candidateError("all-traversal witness count exceeds its domain");
    if (selected == contract.witnessCount)
      if (llvm::Error error = changeFragment(contract.fragment, true))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error
HandshakeCandidateTransaction::removeTraversalUses(PnrIndex traversal,
                                                   PnrIndex count) {
  if (!scratch_ || closed_)
    return candidateError("transaction is not collecting changes");
  const std::size_t traversalCount =
      state_->index_->traversalFragmentOffsets().size() - 1;
  if (traversal >= traversalCount)
    return candidateError("deselected traversal is out of range");
  if (count == 0)
    return candidateError("traversal use decrement is zero");
  PnrIndex &refcount = state_->traversalRefcounts_[traversal];
  if (count > refcount)
    return candidateError("traversal use refcount underflows");
  recordTraversal(traversal);
  refcount -= count;
  if (refcount != 0)
    return llvm::Error::success();

  const auto fragmentOffsets = state_->index_->traversalFragmentOffsets();
  if (llvm::Error error =
          removeFragments(state_->index_->traversalFragments().slice(
              fragmentOffsets[traversal],
              fragmentOffsets[traversal + 1] - fragmentOffsets[traversal])))
    return error;
  const auto groupOffsets = state_->index_->traversalAllGroupOffsets();
  for (PnrIndex group : state_->index_->traversalAllGroups().slice(
           groupOffsets[traversal],
           groupOffsets[traversal + 1] - groupOffsets[traversal])) {
    recordGroup(group);
    PnrIndex &selected = state_->allGroupSelectedWitnessCounts_[group];
    const auto contract = state_->index_->allTraversalGroups()[group];
    if (selected == 0)
      return candidateError("all-traversal witness count underflows");
    if (selected == contract.witnessCount)
      if (llvm::Error error = changeFragment(contract.fragment, false))
        return error;
    --selected;
  }
  return llvm::Error::success();
}

llvm::Expected<bool> HandshakeCandidateTransaction::close() {
  if (!scratch_)
    return candidateError("transaction is no longer active");
  if (closed_)
    return !cycle_;

  llvm::sort(scratch_->arcDeltas_, [](const auto &lhs, const auto &rhs) {
    return lhs.index < rhs.index;
  });
  auto &topologyTransaction = *scratch_->storage_->topologyTransaction;
  for (const auto &delta : scratch_->arcDeltas_)
    if (delta.oldValue != 0 && state_->arcRefcounts_[delta.index] == 0)
      scratch_->storage_->removedArcs.push_back(delta.index);
  for (const auto &delta : scratch_->arcDeltas_)
    if (delta.oldValue == 0 && state_->arcRefcounts_[delta.index] != 0)
      scratch_->storage_->insertedArcs.push_back(delta.index);
  auto changed = topologyTransaction.applyArcChanges(
      scratch_->storage_->removedArcs, scratch_->storage_->insertedArcs);
  if (!changed)
    return changed.takeError();
  if (!*changed) {
    cycle_ = true;
    closed_ = true;
    return false;
  }
  closed_ = true;
  return true;
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateTransaction::cycleWitness() const {
  if (!scratch_ || !scratch_->storage_->topologyTransaction)
    return {};
  return scratch_->storage_->topologyTransaction->cycleWitness();
}

llvm::Error HandshakeCandidateTransaction::commit() {
  if (!scratch_)
    return candidateError("transaction is no longer active");
  auto closure = close();
  if (!closure)
    return closure.takeError();
  if (!*closure)
    return candidateError("cannot commit a handshake cycle");
  if (llvm::Error error = scratch_->storage_->topologyTransaction->commit())
    return error;
  scratch_->storage_->topologyTransaction.reset();
  finish();
  return llvm::Error::success();
}

void HandshakeCandidateTransaction::rollback() noexcept {
  if (!scratch_)
    return;
  if (scratch_->storage_->topologyTransaction) {
    scratch_->storage_->topologyTransaction->rollback();
    scratch_->storage_->topologyTransaction.reset();
  }
  for (const auto &delta : scratch_->fragmentDeltas_)
    state_->fragmentRefcounts_[delta.index] = delta.oldValue;
  for (const auto &delta : scratch_->arcDeltas_)
    state_->arcRefcounts_[delta.index] = delta.oldValue;
  for (const auto &delta : scratch_->traversalDeltas_)
    state_->traversalRefcounts_[delta.index] = delta.oldValue;
  for (const auto &delta : scratch_->groupDeltas_)
    state_->allGroupSelectedWitnessCounts_[delta.index] = delta.oldValue;
  finish();
}

void HandshakeCandidateTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  state_.reset();
}
