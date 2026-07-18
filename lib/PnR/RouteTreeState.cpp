#include "PnR/RouteTreeState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

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

namespace {

constexpr llvm::StringLiteral routeTreeArtifact = "RouteTreeState";
constexpr PnrCapacityContext nodeCountContext{routeTreeArtifact, "node_storage",
                                              "reached_endpoints",
                                              PnrCapacityMeasure::Count};
constexpr PnrCapacityContext nodeIndexContext{routeTreeArtifact, "node_storage",
                                              "reached_endpoints",
                                              PnrCapacityMeasure::Index};
constexpr PnrCapacityContext sinkCountContext{
    routeTreeArtifact, "sink_bindings", "logical_sink_obligations",
    PnrCapacityMeasure::Count};

llvm::Error routeTreeError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid route tree state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

std::uint64_t sizeValue(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(size);
}

} // namespace

void RouteTreeTransactionScratch::clearRetainingCapacity() {
  deltas_.clear();
  worklist_.clear();
  childReferences_.clear();
  visited_.clear();
  sinkCounts_.clear();
  pathMarks_.clear();
  pathGeneration_ = 0;
}

llvm::Error loom::pnr::detail::preflightRouteTreeStateCapacity(
    std::uint64_t reachedEndpointCount, std::uint64_t sinkObligationCount) {
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, reachedEndpointCount))
    return error;
  return preflightPnrIndexCapacity(sinkCountContext, sinkObligationCount);
}

llvm::Expected<RouteTreeState>
RouteTreeState::create(const FrozenRoutingGraph &graph,
                       PnrIndex sinkObligationCount) {
  if (llvm::Error error = detail::preflightRouteTreeStateCapacity(
          sizeValue(graph.routingEndpoints().size()), sinkObligationCount))
    return std::move(error);
  return RouteTreeState(
      graph,
      std::vector<SinkBinding>(static_cast<std::size_t>(sinkObligationCount)));
}

RouteTreeState::RouteTreeState(const FrozenRoutingGraph &graph,
                               std::vector<SinkBinding> sinkBindings)
    : graph_(graph), sinkBindings_(std::move(sinkBindings)) {}

RouteTreeState::RouteTreeState(RouteTreeState &&other) noexcept
    : graph_(other.graph_), sourceEndpoint_(other.sourceEndpoint_),
      sinkBindings_(std::move(other.sinkBindings_)),
      nodes_(std::move(other.nodes_)), freeSlots_(std::move(other.freeSlots_)),
      endpointSlots_(std::move(other.endpointSlots_)),
      activeNodeCount_(other.activeNodeCount_), routed_(other.routed_) {
  assert(!other.transactionActive_ &&
         "cannot move route tree with an active transaction");
  other.sourceEndpoint_ = getInvalidPnrIndex();
  other.activeNodeCount_ = 0;
  other.routed_ = false;
}

std::size_t RouteTreeState::hashEndpoint(PnrIndex endpoint) {
  std::uint64_t value = static_cast<std::uint64_t>(endpoint);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  return static_cast<std::size_t>(value);
}

std::optional<PnrIndex> RouteTreeState::lookupSlot(PnrIndex endpoint) const {
  if (endpointSlots_.empty())
    return std::nullopt;
  const std::size_t mask = endpointSlots_.size() - 1;
  std::size_t bucket = hashEndpoint(endpoint) & mask;
  for (std::size_t probe = 0; probe < endpointSlots_.size(); ++probe) {
    const LookupEntry &entry = endpointSlots_[bucket];
    if (!entry.isOccupied())
      return std::nullopt;
    if (entry.endpoint == endpoint)
      return entry.slot;
    bucket = (bucket + 1) & mask;
  }
  return std::nullopt;
}

llvm::Error RouteTreeState::ensureLookupCapacity(PnrIndex requiredCount) {
  if (requiredCount == 0)
    return llvm::Error::success();

  std::size_t capacity = endpointSlots_.empty() ? 8 : endpointSlots_.size();
  const std::size_t required = static_cast<std::size_t>(requiredCount);
  while (required > capacity / 2) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2)
      return routeTreeError("endpoint-to-slot lookup capacity overflow");
    capacity *= 2;
  }
  if (capacity != endpointSlots_.size())
    rehashLookup(capacity);
  return llvm::Error::success();
}

void RouteTreeState::insertLookup(PnrIndex endpoint, PnrIndex slot) {
  assert(!endpointSlots_.empty());
  const std::size_t mask = endpointSlots_.size() - 1;
  std::size_t bucket = hashEndpoint(endpoint) & mask;
  while (endpointSlots_[bucket].isOccupied()) {
    assert(endpointSlots_[bucket].endpoint != endpoint &&
           "duplicate reached endpoint");
    bucket = (bucket + 1) & mask;
  }
  endpointSlots_[bucket] = {endpoint, slot};
}

void RouteTreeState::eraseLookup(PnrIndex endpoint) {
  assert(!endpointSlots_.empty());
  const std::size_t mask = endpointSlots_.size() - 1;
  std::size_t bucket = hashEndpoint(endpoint) & mask;
  while (endpointSlots_[bucket].isOccupied() &&
         endpointSlots_[bucket].endpoint != endpoint)
    bucket = (bucket + 1) & mask;
  assert(endpointSlots_[bucket].isOccupied() &&
         "erasing an endpoint absent from the lookup");
  endpointSlots_[bucket] = {};

  std::size_t next = (bucket + 1) & mask;
  while (endpointSlots_[next].isOccupied()) {
    const LookupEntry displaced = endpointSlots_[next];
    endpointSlots_[next] = {};
    insertLookup(displaced.endpoint, displaced.slot);
    next = (next + 1) & mask;
  }
}

void RouteTreeState::rehashLookup(std::size_t capacity) {
  std::vector<LookupEntry> previous;
  previous.swap(endpointSlots_);
  endpointSlots_.assign(capacity, {});
  for (const LookupEntry &entry : previous)
    if (entry.isOccupied())
      insertLookup(entry.endpoint, entry.slot);
}

std::optional<PnrIndex> RouteTreeState::sourceEndpoint() const {
  if (sourceEndpoint_ == getInvalidPnrIndex())
    return std::nullopt;
  return sourceEndpoint_;
}

std::optional<PnrIndex>
RouteTreeState::sinkEndpoint(PnrIndex obligation) const {
  if (obligation >= sinkBindings_.size() ||
      sinkBindings_[obligation].endpoint == getInvalidPnrIndex())
    return std::nullopt;
  return sinkBindings_[obligation].endpoint;
}

PnrIndex RouteTreeState::arcSourceEndpoint(PnrIndex arc) const {
  const llvm::ArrayRef<PnrIndex> offsets = graph_.adjacencyOffsets();
  const auto sourceEnd = std::upper_bound(offsets.begin(), offsets.end(), arc);
  assert(sourceEnd != offsets.begin() && sourceEnd != offsets.end());
  return static_cast<PnrIndex>(std::distance(offsets.begin(), sourceEnd) - 1);
}

std::optional<PnrIndex> RouteTreeState::findNode(PnrIndex endpoint) const {
  return lookupSlot(endpoint);
}

const RouteTreeNode &RouteTreeState::node(PnrIndex slot) const {
  assert(slot < nodes_.size() && nodes_[slot].isActive());
  return nodes_[slot];
}

llvm::Error RouteTreeState::verify() const {
  if (transactionActive_)
    return routeTreeError("cannot verify while a transaction is active");
  RouteTreeTransactionScratch scratch;
  return verifyCandidate(routed_, scratch);
}

llvm::Error
RouteTreeState::verifyCandidate(bool routedCandidate,
                                RouteTreeTransactionScratch &scratch) const {
  scratch.childReferences_.assign(nodes_.size(), 0);
  scratch.visited_.assign(nodes_.size(), 0);
  scratch.sinkCounts_.assign(nodes_.size(), 0);
  scratch.worklist_.clear();

  std::size_t activeCount = 0;
  std::size_t lookupCount = 0;
  for (const LookupEntry &entry : endpointSlots_)
    lookupCount += entry.isOccupied();
  for (std::size_t slot = 0; slot < nodes_.size(); ++slot) {
    if (!nodes_[slot].isActive())
      continue;
    ++activeCount;
    if (nodes_[slot].endpoint >= graph_.routingEndpoints().size())
      return routeTreeError("node endpoint is outside FrozenRoutingGraph");
    if (lookupSlot(nodes_[slot].endpoint) != slot)
      return routeTreeError("endpoint-to-slot lookup diverges from nodes");
  }
  if (activeCount != activeNodeCount_ || lookupCount != activeCount)
    return routeTreeError("active-node accounting is inconsistent");

  if (!routedCandidate) {
    if (activeCount != 0 || lookupCount != 0)
      return routeTreeError("explicit unrouted state retains route nodes");
    if (sourceEndpoint_ != getInvalidPnrIndex())
      return routeTreeError("explicit unrouted state retains source binding");
    for (const SinkBinding &binding : sinkBindings_)
      if (binding.endpoint != getInvalidPnrIndex() ||
          binding.nodeSlot != getInvalidPnrIndex())
        return routeTreeError("explicit unrouted state retains sink binding");
    return llvm::Error::success();
  }
  if (activeCount == 0)
    return routeTreeError("routed state has no root");
  if (sourceEndpoint_ == getInvalidPnrIndex() ||
      sourceEndpoint_ >= graph_.routingEndpoints().size())
    return routeTreeError("routed state has no valid source binding");

  const std::optional<PnrIndex> rootSlot = lookupSlot(sourceEndpoint_);
  if (!rootSlot)
    return routeTreeError("routed state is missing its source root");
  const RouteTreeNode &root = nodes_[*rootSlot];
  if (root.parentArc != getInvalidPnrIndex() ||
      root.previousSibling != getInvalidPnrIndex() ||
      root.nextSibling != getInvalidPnrIndex())
    return routeTreeError("source root has parent or sibling linkage");

  for (const SinkBinding &binding : sinkBindings_) {
    if (binding.endpoint == getInvalidPnrIndex() ||
        binding.nodeSlot == getInvalidPnrIndex() ||
        binding.nodeSlot >= nodes_.size() ||
        !nodes_[binding.nodeSlot].isActive() ||
        nodes_[binding.nodeSlot].endpoint != binding.endpoint)
      return routeTreeError("sink obligation is not covered");
    ++scratch.sinkCounts_[binding.nodeSlot];
  }
  for (std::size_t slot = 0; slot < nodes_.size(); ++slot)
    if (nodes_[slot].isActive() &&
        nodes_[slot].sinkObligationCount != scratch.sinkCounts_[slot])
      return routeTreeError("node sink metadata diverges from bindings");

  for (std::size_t parentSlot = 0; parentSlot < nodes_.size(); ++parentSlot) {
    const RouteTreeNode &parent = nodes_[parentSlot];
    if (!parent.isActive())
      continue;
    PnrIndex previous = getInvalidPnrIndex();
    PnrIndex child = parent.firstChild;
    std::size_t traversed = 0;
    while (child != getInvalidPnrIndex()) {
      if (child >= nodes_.size() || !nodes_[child].isActive())
        return routeTreeError("child linkage references an inactive slot");
      if (++traversed > activeCount)
        return routeTreeError("child linkage contains a cycle");
      const RouteTreeNode &childNode = nodes_[child];
      if (childNode.previousSibling != previous)
        return routeTreeError("sibling linkage is inconsistent");
      if (childNode.parentArc == getInvalidPnrIndex() ||
          childNode.parentArc >= graph_.routingArcs().size())
        return routeTreeError("non-root node has no valid parent arc");
      if (graph_.routingArcs()[childNode.parentArc].target !=
              childNode.endpoint ||
          arcSourceEndpoint(childNode.parentArc) != parent.endpoint)
        return routeTreeError("parent arc disagrees with tree linkage");
      if (++scratch.childReferences_[child] != 1)
        return routeTreeError("reached endpoint has multiple parents");
      previous = child;
      child = childNode.nextSibling;
    }
  }

  for (std::size_t slot = 0; slot < nodes_.size(); ++slot) {
    if (!nodes_[slot].isActive())
      continue;
    const std::uint8_t expected = slot == *rootSlot ? 0 : 1;
    if (scratch.childReferences_[slot] != expected)
      return routeTreeError("route tree is disconnected or reconvergent");
  }

  scratch.worklist_.push_back(*rootSlot);
  std::size_t visitedCount = 0;
  while (!scratch.worklist_.empty()) {
    const PnrIndex current = scratch.worklist_.back();
    scratch.worklist_.pop_back();
    if (scratch.visited_[current])
      return routeTreeError("route tree contains a cycle or reconvergence");
    scratch.visited_[current] = 1;
    ++visitedCount;
    for (PnrIndex child = nodes_[current].firstChild;
         child != getInvalidPnrIndex(); child = nodes_[child].nextSibling)
      scratch.worklist_.push_back(child);
  }
  if (visitedCount != activeCount)
    return routeTreeError("parent chain does not reach the source root");
  return llvm::Error::success();
}

llvm::Expected<RouteTreeTransaction>
RouteTreeState::beginTransaction(RouteTreeTransactionScratch &scratch) {
  if (transactionActive_)
    return routeTreeError("another transaction is already active");
  if (scratch.inUse_)
    return routeTreeError("transaction scratch is already in use");
  scratch.clearRetainingCapacity();
  scratch.inUse_ = true;
  transactionActive_ = true;
  return RouteTreeTransaction(*this, scratch);
}

RouteTreeTransaction::RouteTreeTransaction(RouteTreeState &state,
                                           RouteTreeTransactionScratch &scratch)
    : state_(&state), scratch_(&scratch),
      initialNodeStorageSize_(state.nodes_.size()),
      initialActiveNodeCount_(state.activeNodeCount_),
      initialRouted_(state.routed_) {}

RouteTreeTransaction::RouteTreeTransaction(
    RouteTreeTransaction &&other) noexcept
    : state_(other.state_), scratch_(other.scratch_),
      initialNodeStorageSize_(other.initialNodeStorageSize_),
      initialActiveNodeCount_(other.initialActiveNodeCount_),
      initialRouted_(other.initialRouted_) {
  other.state_ = nullptr;
  other.scratch_ = nullptr;
}

RouteTreeTransaction::~RouteTreeTransaction() {
  if (state_)
    rollback();
}

void RouteTreeTransaction::recordModifiedNode(PnrIndex slot) {
  scratch_->deltas_.push_back(
      {RouteTreeTransactionScratch::DeltaKind::ModifiedNode, slot,
       state_->nodes_[slot]});
}

void RouteTreeTransaction::setSinkMetadata(PnrIndex slot, PnrIndex count) {
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::SinkMetadata;
  delta.key = slot;
  delta.value0 = state_->nodes_[slot].sinkObligationCount;
  scratch_->deltas_.push_back(delta);
  state_->nodes_[slot].sinkObligationCount = count;
}

void RouteTreeTransaction::setSourceBinding(PnrIndex endpoint) {
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::SourceBinding;
  delta.value0 = state_->sourceEndpoint_;
  scratch_->deltas_.push_back(delta);
  state_->sourceEndpoint_ = endpoint;
}

void RouteTreeTransaction::setSinkBinding(PnrIndex obligation,
                                          PnrIndex endpoint,
                                          PnrIndex nodeSlot) {
  const RouteTreeState::SinkBinding previous =
      state_->sinkBindings_[obligation];
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::SinkBinding;
  delta.key = obligation;
  delta.value0 = previous.endpoint;
  delta.value1 = previous.nodeSlot;
  scratch_->deltas_.push_back(delta);
  state_->sinkBindings_[obligation] = {endpoint, nodeSlot};
}

llvm::Expected<PnrIndex> RouteTreeTransaction::addNode(PnrIndex endpoint,
                                                       PnrIndex parentArc) {
  PnrIndex slot;
  const bool appended = state_->freeSlots_.empty();
  if (appended) {
    auto checkedSlot =
        checkedPnrIndex(nodeIndexContext, sizeValue(state_->nodes_.size()));
    if (!checkedSlot)
      return checkedSlot.takeError();
    slot = *checkedSlot;
    RouteTreeTransactionScratch::Delta delta;
    delta.kind = RouteTreeTransactionScratch::DeltaKind::AddedNode;
    delta.key = slot;
    delta.appended = true;
    scratch_->deltas_.push_back(delta);
    state_->nodes_.push_back({});
  } else {
    slot = state_->freeSlots_.back();
    RouteTreeTransactionScratch::Delta delta;
    delta.kind = RouteTreeTransactionScratch::DeltaKind::AddedNode;
    delta.key = slot;
    scratch_->deltas_.push_back(delta);
    state_->freeSlots_.pop_back();
  }

  RouteTreeNode &node = state_->nodes_[slot];
  node = {};
  node.endpoint = endpoint;
  node.parentArc = parentArc;
  state_->insertLookup(endpoint, slot);
  ++state_->activeNodeCount_;
  return slot;
}

void RouteTreeTransaction::linkChild(PnrIndex parentSlot, PnrIndex childSlot) {
  RouteTreeNode &parent = state_->nodes_[parentSlot];
  const PnrIndex oldFirstChild = parent.firstChild;
  recordModifiedNode(parentSlot);
  if (oldFirstChild != getInvalidPnrIndex()) {
    recordModifiedNode(oldFirstChild);
    state_->nodes_[oldFirstChild].previousSibling = childSlot;
  }
  state_->nodes_[childSlot].nextSibling = oldFirstChild;
  state_->nodes_[childSlot].previousSibling = getInvalidPnrIndex();
  parent.firstChild = childSlot;
}

PnrIndex RouteTreeTransaction::parentSlot(PnrIndex childSlot) const {
  const RouteTreeNode &child = state_->nodes_[childSlot];
  assert(child.parentArc != getInvalidPnrIndex());
  const PnrIndex parentEndpoint = state_->arcSourceEndpoint(child.parentArc);
  const std::optional<PnrIndex> parent = state_->lookupSlot(parentEndpoint);
  assert(parent && "parent arc source is absent from the route tree");
  return *parent;
}

void RouteTreeTransaction::detachNode(PnrIndex slot, PnrIndex parentSlot) {
  const RouteTreeNode &node = state_->nodes_[slot];
  const PnrIndex previous = node.previousSibling;
  const PnrIndex next = node.nextSibling;
  if (previous == getInvalidPnrIndex()) {
    recordModifiedNode(parentSlot);
    assert(state_->nodes_[parentSlot].firstChild == slot);
    state_->nodes_[parentSlot].firstChild = next;
  } else {
    recordModifiedNode(previous);
    state_->nodes_[previous].nextSibling = next;
  }
  if (next != getInvalidPnrIndex()) {
    recordModifiedNode(next);
    state_->nodes_[next].previousSibling = previous;
  }
}

void RouteTreeTransaction::removeNode(PnrIndex slot) {
  const RouteTreeNode snapshot = state_->nodes_[slot];
  scratch_->deltas_.push_back(
      {RouteTreeTransactionScratch::DeltaKind::RemovedNode, slot, snapshot});
  state_->eraseLookup(snapshot.endpoint);
  state_->nodes_[slot] = {};
  state_->freeSlots_.push_back(slot);
  --state_->activeNodeCount_;
}

llvm::Error RouteTreeTransaction::bindSource(PnrIndex endpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (endpoint >= state_->graph_.routingEndpoints().size())
    return routeTreeError("source endpoint is outside FrozenRoutingGraph");
  if (state_->activeNodeCount_ != 0 && state_->sourceEndpoint_ != endpoint)
    return routeTreeError("routed source requires whole-net rip-up");
  if (state_->sourceEndpoint_ != endpoint)
    setSourceBinding(endpoint);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::bindSink(PnrIndex obligation,
                                           PnrIndex endpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (obligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  if (endpoint >= state_->graph_.routingEndpoints().size())
    return routeTreeError("sink endpoint is outside FrozenRoutingGraph");
  const RouteTreeState::SinkBinding &binding =
      state_->sinkBindings_[obligation];
  if (binding.nodeSlot != getInvalidPnrIndex())
    return routeTreeError("attached sink obligation requires rip-up");
  if (binding.endpoint != endpoint)
    setSinkBinding(obligation, endpoint, getInvalidPnrIndex());
  return llvm::Error::success();
}

llvm::Error
RouteTreeTransaction::attachPath(PnrIndex attachmentEndpoint,
                                 llvm::ArrayRef<PnrIndex> forwardArcs,
                                 PnrIndex sinkObligation) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (sinkObligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  const RouteTreeState::SinkBinding &binding =
      state_->sinkBindings_[sinkObligation];
  if (binding.endpoint == getInvalidPnrIndex())
    return routeTreeError("sink obligation has no endpoint binding");
  if (binding.nodeSlot != getInvalidPnrIndex())
    return routeTreeError("sink obligation is already attached");

  const bool emptyTree = state_->activeNodeCount_ == 0;
  std::optional<PnrIndex> attachmentSlot =
      state_->lookupSlot(attachmentEndpoint);
  if (emptyTree) {
    if (state_->sourceEndpoint_ == getInvalidPnrIndex())
      return routeTreeError("route has no source endpoint binding");
    if (attachmentEndpoint != state_->sourceEndpoint_)
      return routeTreeError("first path does not attach at the source root");
  } else if (!attachmentSlot) {
    return routeTreeError("path attachment endpoint is not reached");
  }

  if (scratch_->pathMarks_.size() < state_->graph_.routingEndpoints().size())
    scratch_->pathMarks_.resize(state_->graph_.routingEndpoints().size(), 0);
  if (scratch_->pathGeneration_ == std::numeric_limits<std::uint64_t>::max()) {
    std::fill(scratch_->pathMarks_.begin(), scratch_->pathMarks_.end(), 0);
    scratch_->pathGeneration_ = 1;
  } else {
    ++scratch_->pathGeneration_;
  }
  const std::uint64_t generation = scratch_->pathGeneration_;
  scratch_->pathMarks_[attachmentEndpoint] = generation;
  scratch_->worklist_.clear();

  PnrIndex current = attachmentEndpoint;
  for (PnrIndex forwardArc : forwardArcs) {
    if (forwardArc >= state_->graph_.routingArcs().size())
      return routeTreeError("path arc is outside FrozenRoutingGraph");
    if (state_->arcSourceEndpoint(forwardArc) != current)
      return routeTreeError("path arc does not continue from endpoint");
    const PnrIndex target = state_->graph_.routingArcs()[forwardArc].target;
    if (state_->lookupSlot(target))
      return routeTreeError("path leaves the tree and re-enters it");
    if (scratch_->pathMarks_[target] == generation)
      return routeTreeError("path repeats an endpoint");
    scratch_->pathMarks_[target] = generation;
    scratch_->worklist_.push_back(target);
    current = target;
  }
  if (current != binding.endpoint)
    return routeTreeError("path does not terminate at its sink binding");

  const PnrIndex existingSinkCount =
      !emptyTree && forwardArcs.empty()
          ? state_->nodes_[*attachmentSlot].sinkObligationCount
          : PnrIndex{0};
  auto finalSinkCount =
      checkedPnrIndexAdd(sinkCountContext, existingSinkCount, 1);
  if (!finalSinkCount)
    return finalSinkCount.takeError();

  const std::uint64_t addedCount = sizeValue(scratch_->worklist_.size()) +
                                   static_cast<std::uint64_t>(emptyTree);
  auto finalNodeCount = checkedPnrIndexAdd(
      nodeCountContext, state_->activeNodeCount_, addedCount);
  if (!finalNodeCount)
    return finalNodeCount.takeError();
  if (llvm::Error error = state_->ensureLookupCapacity(*finalNodeCount))
    return error;

  const std::size_t reused =
      std::min<std::size_t>(state_->freeSlots_.size(), addedCount);
  state_->nodes_.reserve(state_->nodes_.size() +
                         static_cast<std::size_t>(addedCount) - reused);
  scratch_->deltas_.reserve(scratch_->deltas_.size() +
                            static_cast<std::size_t>(addedCount) * 3 + 2);

  if (emptyTree) {
    auto root = addNode(state_->sourceEndpoint_, getInvalidPnrIndex());
    if (!root)
      return root.takeError();
    attachmentSlot = *root;
  }

  PnrIndex parent = *attachmentSlot;
  for (auto [forwardArc, target] :
       llvm::zip(forwardArcs, scratch_->worklist_)) {
    auto child = addNode(target, forwardArc);
    if (!child)
      return child.takeError();
    linkChild(parent, *child);
    parent = *child;
  }
  setSinkMetadata(parent, *finalSinkCount);
  setSinkBinding(sinkObligation, binding.endpoint, parent);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpSink(PnrIndex sinkObligation) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (sinkObligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  const RouteTreeState::SinkBinding binding =
      state_->sinkBindings_[sinkObligation];
  if (binding.endpoint == getInvalidPnrIndex())
    return routeTreeError("sink obligation has no endpoint binding");
  if (binding.nodeSlot == getInvalidPnrIndex()) {
    setSinkBinding(sinkObligation, getInvalidPnrIndex(), getInvalidPnrIndex());
    return llvm::Error::success();
  }
  if (binding.nodeSlot >= state_->nodes_.size() ||
      !state_->nodes_[binding.nodeSlot].isActive() ||
      state_->nodes_[binding.nodeSlot].endpoint != binding.endpoint ||
      state_->nodes_[binding.nodeSlot].sinkObligationCount == 0)
    return routeTreeError("sink binding diverges from the route tree");

  scratch_->deltas_.reserve(scratch_->deltas_.size() +
                            state_->nodes_.size() * 4 + 2);
  state_->freeSlots_.reserve(state_->freeSlots_.size() +
                             state_->activeNodeCount_);
  PnrIndex current = binding.nodeSlot;
  setSinkMetadata(current, state_->nodes_[current].sinkObligationCount - 1);
  setSinkBinding(sinkObligation, getInvalidPnrIndex(), getInvalidPnrIndex());

  const std::optional<PnrIndex> root =
      state_->lookupSlot(state_->sourceEndpoint_);
  assert(root);
  while (current != *root &&
         state_->nodes_[current].firstChild == getInvalidPnrIndex() &&
         state_->nodes_[current].sinkObligationCount == 0) {
    const PnrIndex parent = parentSlot(current);
    detachNode(current, parent);
    removeNode(current);
    current = parent;
  }
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpSubtree(PnrIndex subtreeRootEndpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  const std::optional<PnrIndex> subtreeRoot =
      state_->lookupSlot(subtreeRootEndpoint);
  if (!subtreeRoot)
    return routeTreeError("subtree root endpoint is not reached");
  const std::optional<PnrIndex> root =
      state_->lookupSlot(state_->sourceEndpoint_);
  assert(root);
  if (*subtreeRoot == *root)
    return ripUpWholeNet();

  scratch_->worklist_.clear();
  scratch_->worklist_.push_back(*subtreeRoot);
  for (std::size_t index = 0; index < scratch_->worklist_.size(); ++index) {
    const PnrIndex current = scratch_->worklist_[index];
    for (PnrIndex child = state_->nodes_[current].firstChild;
         child != getInvalidPnrIndex();
         child = state_->nodes_[child].nextSibling)
      scratch_->worklist_.push_back(child);
  }
  scratch_->visited_.assign(state_->nodes_.size(), 0);
  for (PnrIndex slot : scratch_->worklist_)
    scratch_->visited_[slot] = 1;
  for (PnrIndex obligation = 0; obligation < state_->sinkBindings_.size();
       ++obligation) {
    const PnrIndex nodeSlot = state_->sinkBindings_[obligation].nodeSlot;
    if (nodeSlot != getInvalidPnrIndex() && scratch_->visited_[nodeSlot])
      setSinkBinding(obligation, getInvalidPnrIndex(), getInvalidPnrIndex());
  }

  scratch_->deltas_.reserve(scratch_->deltas_.size() +
                            scratch_->worklist_.size() + 3);
  state_->freeSlots_.reserve(state_->freeSlots_.size() +
                             scratch_->worklist_.size());
  const PnrIndex parent = parentSlot(*subtreeRoot);
  detachNode(*subtreeRoot, parent);
  for (PnrIndex slot : scratch_->worklist_)
    removeNode(slot);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpWholeNet() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  scratch_->deltas_.reserve(scratch_->deltas_.size() +
                            state_->activeNodeCount_ +
                            state_->sinkBindings_.size() + 1);
  for (PnrIndex obligation = 0; obligation < state_->sinkBindings_.size();
       ++obligation) {
    const RouteTreeState::SinkBinding &binding =
        state_->sinkBindings_[obligation];
    if (binding.endpoint != getInvalidPnrIndex() ||
        binding.nodeSlot != getInvalidPnrIndex())
      setSinkBinding(obligation, getInvalidPnrIndex(), getInvalidPnrIndex());
  }
  if (state_->sourceEndpoint_ != getInvalidPnrIndex())
    setSourceBinding(getInvalidPnrIndex());

  state_->freeSlots_.reserve(state_->freeSlots_.size() +
                             state_->activeNodeCount_);
  for (std::size_t slot = 0; slot < state_->nodes_.size(); ++slot)
    if (state_->nodes_[slot].isActive())
      removeNode(static_cast<PnrIndex>(slot));
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::commit() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  const bool routedCandidate = state_->activeNodeCount_ != 0;
  if (llvm::Error error = state_->verifyCandidate(routedCandidate, *scratch_))
    return error;

  state_->routed_ = routedCandidate;
  if (!routedCandidate) {
    state_->nodes_.clear();
    state_->freeSlots_.clear();
    state_->endpointSlots_.clear();
  }
  finish();
  return llvm::Error::success();
}

void RouteTreeTransaction::finish() {
  state_->transactionActive_ = false;
  scratch_->inUse_ = false;
  scratch_->clearRetainingCapacity();
  state_ = nullptr;
  scratch_ = nullptr;
}

void RouteTreeTransaction::rollback() noexcept {
  if (!state_)
    return;
  for (auto delta = scratch_->deltas_.rbegin();
       delta != scratch_->deltas_.rend(); ++delta) {
    switch (delta->kind) {
    case RouteTreeTransactionScratch::DeltaKind::ModifiedNode:
      state_->nodes_[delta->key] = delta->node;
      break;
    case RouteTreeTransactionScratch::DeltaKind::RemovedNode:
      assert(!state_->freeSlots_.empty() &&
             state_->freeSlots_.back() == delta->key);
      state_->freeSlots_.pop_back();
      state_->nodes_[delta->key] = delta->node;
      state_->insertLookup(delta->node.endpoint, delta->key);
      ++state_->activeNodeCount_;
      break;
    case RouteTreeTransactionScratch::DeltaKind::AddedNode:
      assert(state_->nodes_[delta->key].isActive());
      state_->eraseLookup(state_->nodes_[delta->key].endpoint);
      --state_->activeNodeCount_;
      if (delta->appended) {
        assert(delta->key + 1 == state_->nodes_.size());
        state_->nodes_.pop_back();
      } else {
        state_->nodes_[delta->key] = {};
        state_->freeSlots_.push_back(delta->key);
      }
      break;
    case RouteTreeTransactionScratch::DeltaKind::SinkMetadata:
      state_->nodes_[delta->key].sinkObligationCount = delta->value0;
      break;
    case RouteTreeTransactionScratch::DeltaKind::SourceBinding:
      state_->sourceEndpoint_ = delta->value0;
      break;
    case RouteTreeTransactionScratch::DeltaKind::SinkBinding:
      state_->sinkBindings_[delta->key] = {delta->value0, delta->value1};
      break;
    }
  }
  assert(state_->nodes_.size() == initialNodeStorageSize_);
  assert(state_->activeNodeCount_ == initialActiveNodeCount_);
  state_->routed_ = initialRouted_;
  finish();
}
