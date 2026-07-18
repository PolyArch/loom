#include "PnR/RouteTreeState.h"

#include "llvm/ADT/DenseSet.h"
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
    routeTreeArtifact, "sink_obligations", "logical_sinks",
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

llvm::Error loom::pnr::detail::preflightRouteTreeStateCapacity(
    std::uint64_t reachedEndpointCount, std::uint64_t sinkObligationCount) {
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, reachedEndpointCount))
    return error;
  return preflightPnrIndexCapacity(sinkCountContext, sinkObligationCount);
}

llvm::Expected<RouteTreeState>
RouteTreeState::create(const FrozenRoutingGraph &graph,
                       PnrIndex producerEndpoint,
                       llvm::ArrayRef<PnrIndex> sinkEndpoints) {
  if (llvm::Error error = detail::preflightRouteTreeStateCapacity(
          sizeValue(graph.routingEndpoints().size()),
          sizeValue(sinkEndpoints.size())))
    return std::move(error);
  if (producerEndpoint >= graph.routingEndpoints().size())
    return routeTreeError("producer endpoint is outside FrozenRoutingGraph");

  std::vector<PnrIndex> sortedSinks(sinkEndpoints.begin(), sinkEndpoints.end());
  std::sort(sortedSinks.begin(), sortedSinks.end());
  std::vector<SinkObligation> obligations;
  obligations.reserve(sortedSinks.size());
  for (PnrIndex endpoint : sortedSinks) {
    if (endpoint >= graph.routingEndpoints().size())
      return routeTreeError("sink endpoint is outside FrozenRoutingGraph");
    if (obligations.empty() || obligations.back().endpoint != endpoint) {
      obligations.push_back({endpoint, PnrIndex{1}});
      continue;
    }
    auto count =
        checkedPnrIndexAdd(sinkCountContext, obligations.back().count, 1);
    if (!count)
      return count.takeError();
    obligations.back().count = *count;
  }

  return RouteTreeState(graph, producerEndpoint, std::move(obligations));
}

RouteTreeState::RouteTreeState(const FrozenRoutingGraph &graph,
                               PnrIndex producerEndpoint,
                               std::vector<SinkObligation> sinkObligations)
    : graph_(graph), producerEndpoint_(producerEndpoint),
      sinkObligations_(std::move(sinkObligations)) {}

RouteTreeState::RouteTreeState(RouteTreeState &&other) noexcept
    : graph_(other.graph_), producerEndpoint_(other.producerEndpoint_),
      sinkObligations_(std::move(other.sinkObligations_)),
      nodes_(std::move(other.nodes_)), freeSlots_(std::move(other.freeSlots_)),
      endpointSlots_(std::move(other.endpointSlots_)),
      activeNodeCount_(other.activeNodeCount_), routed_(other.routed_) {
  assert(!other.transactionActive_ &&
         "cannot move route tree with an active transaction");
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
    rebuildLookup(capacity);
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

void RouteTreeState::rebuildLookup(std::size_t capacity) {
  endpointSlots_.assign(capacity, {});
  if (capacity == 0) {
    assert(activeNodeCount_ == 0);
    return;
  }
  for (std::size_t slot = 0; slot < nodes_.size(); ++slot)
    if (nodes_[slot].isActive())
      insertLookup(nodes_[slot].endpoint, static_cast<PnrIndex>(slot));
}

PnrIndex RouteTreeState::requiredSinkCount(PnrIndex endpoint) const {
  const auto found = std::lower_bound(
      sinkObligations_.begin(), sinkObligations_.end(), endpoint,
      [](const SinkObligation &obligation, PnrIndex candidate) {
        return obligation.endpoint < candidate;
      });
  return found != sinkObligations_.end() && found->endpoint == endpoint
             ? found->count
             : PnrIndex{0};
}

PnrIndex RouteTreeState::sourceEndpoint(PnrIndex arc) const {
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
  return verifyCandidate(routed_);
}

llvm::Error RouteTreeState::verifyCandidate(bool routedCandidate) const {
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
    if (nodes_[slot].sinkObligationCount !=
        requiredSinkCount(nodes_[slot].endpoint))
      return routeTreeError("sink obligation is not covered");
  }
  if (activeCount != activeNodeCount_ || lookupCount != activeCount)
    return routeTreeError("active-node accounting is inconsistent");

  if (!routedCandidate) {
    if (activeCount != 0 || lookupCount != 0)
      return routeTreeError("explicit unrouted state retains route nodes");
    if (!transactionActive_ && (!nodes_.empty() || !endpointSlots_.empty()))
      return routeTreeError(
          "committed unrouted state retains sparse route storage");
    return llvm::Error::success();
  }
  if (activeCount == 0)
    return routeTreeError("routed state has no root");

  const std::optional<PnrIndex> rootSlot = lookupSlot(producerEndpoint_);
  if (!rootSlot)
    return routeTreeError("routed state is missing its producer root");
  const RouteTreeNode &root = nodes_[*rootSlot];
  if (root.parentArc != getInvalidPnrIndex() ||
      root.previousSibling != getInvalidPnrIndex() ||
      root.nextSibling != getInvalidPnrIndex())
    return routeTreeError("producer root has parent or sibling linkage");

  std::vector<std::uint8_t> childReferences(nodes_.size(), 0);
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
          sourceEndpoint(childNode.parentArc) != parent.endpoint)
        return routeTreeError("parent arc disagrees with tree linkage");
      if (++childReferences[child] != 1)
        return routeTreeError("reached endpoint has multiple parents");
      previous = child;
      child = childNode.nextSibling;
    }
  }

  for (std::size_t slot = 0; slot < nodes_.size(); ++slot) {
    if (!nodes_[slot].isActive())
      continue;
    const std::uint8_t expected = slot == *rootSlot ? 0 : 1;
    if (childReferences[slot] != expected)
      return routeTreeError("route tree is disconnected or reconvergent");
  }

  std::vector<std::uint8_t> visited(nodes_.size(), 0);
  std::vector<PnrIndex> worklist{*rootSlot};
  std::size_t visitedCount = 0;
  while (!worklist.empty()) {
    const PnrIndex current = worklist.back();
    worklist.pop_back();
    if (visited[current])
      return routeTreeError("route tree contains a cycle or reconvergence");
    visited[current] = 1;
    ++visitedCount;
    for (PnrIndex child = nodes_[current].firstChild;
         child != getInvalidPnrIndex(); child = nodes_[child].nextSibling)
      worklist.push_back(child);
  }
  if (visitedCount != activeCount)
    return routeTreeError("parent chain does not reach the producer root");

  for (const SinkObligation &obligation : sinkObligations_) {
    const std::optional<PnrIndex> sinkSlot = lookupSlot(obligation.endpoint);
    if (!sinkSlot || nodes_[*sinkSlot].sinkObligationCount != obligation.count)
      return routeTreeError("sink obligation is not covered");
  }
  return llvm::Error::success();
}

llvm::Expected<RouteTreeTransaction> RouteTreeState::beginTransaction() {
  if (transactionActive_)
    return routeTreeError("another transaction is already active");
  transactionActive_ = true;
  return RouteTreeTransaction(*this);
}

RouteTreeTransaction::RouteTreeTransaction(RouteTreeState &state)
    : state_(&state), initialLookupCapacity_(state.endpointSlots_.size()),
      initialNodeStorageSize_(state.nodes_.size()),
      initialActiveNodeCount_(state.activeNodeCount_),
      initialRouted_(state.routed_) {}

RouteTreeTransaction::RouteTreeTransaction(
    RouteTreeTransaction &&other) noexcept
    : state_(other.state_), deltas_(std::move(other.deltas_)),
      initialLookupCapacity_(other.initialLookupCapacity_),
      initialNodeStorageSize_(other.initialNodeStorageSize_),
      initialActiveNodeCount_(other.initialActiveNodeCount_),
      initialRouted_(other.initialRouted_) {
  other.state_ = nullptr;
}

RouteTreeTransaction::~RouteTreeTransaction() {
  if (state_)
    rollback();
}

void RouteTreeTransaction::recordModifiedNode(PnrIndex slot) {
  deltas_.push_back(
      {DeltaKind::ModifiedNode, slot, state_->nodes_[slot], false, 0});
}

void RouteTreeTransaction::setSinkMetadata(PnrIndex slot, PnrIndex count) {
  deltas_.push_back({DeltaKind::SinkMetadata,
                     slot,
                     {},
                     false,
                     state_->nodes_[slot].sinkObligationCount});
  state_->nodes_[slot].sinkObligationCount = count;
}

llvm::Expected<PnrIndex> RouteTreeTransaction::addNode(PnrIndex endpoint,
                                                       PnrIndex parentArc) {
  PnrIndex slot;
  bool appended = state_->freeSlots_.empty();
  if (appended) {
    auto checkedSlot =
        checkedPnrIndex(nodeIndexContext, sizeValue(state_->nodes_.size()));
    if (!checkedSlot)
      return checkedSlot.takeError();
    slot = *checkedSlot;
    deltas_.push_back({DeltaKind::AddedNode, slot, {}, true, 0});
    state_->nodes_.push_back({});
  } else {
    slot = state_->freeSlots_.back();
    deltas_.push_back({DeltaKind::AddedNode, slot, {}, false, 0});
    state_->freeSlots_.pop_back();
  }

  RouteTreeNode &node = state_->nodes_[slot];
  node = {};
  node.endpoint = endpoint;
  node.parentArc = parentArc;
  node.sinkObligationCount = state_->requiredSinkCount(endpoint);
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
  const PnrIndex parentEndpoint = state_->sourceEndpoint(child.parentArc);
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
  deltas_.push_back({DeltaKind::RemovedNode, slot, snapshot, false, 0});
  state_->eraseLookup(snapshot.endpoint);
  state_->nodes_[slot] = {};
  state_->freeSlots_.push_back(slot);
  --state_->activeNodeCount_;
}

llvm::Error
RouteTreeTransaction::attachPath(PnrIndex attachmentEndpoint,
                                 llvm::ArrayRef<PnrIndex> forwardArcs,
                                 PnrIndex sinkEndpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (state_->requiredSinkCount(sinkEndpoint) == 0)
    return routeTreeError("path terminal is not a sink obligation");

  const bool emptyTree = state_->activeNodeCount_ == 0;
  std::optional<PnrIndex> attachmentSlot =
      state_->lookupSlot(attachmentEndpoint);
  if (emptyTree) {
    if (attachmentEndpoint != state_->producerEndpoint_)
      return routeTreeError("first path does not attach at the producer root");
  } else if (!attachmentSlot) {
    return routeTreeError("path attachment endpoint is not reached");
  }

  llvm::SmallDenseSet<PnrIndex, 16> pathEndpoints;
  pathEndpoints.insert(attachmentEndpoint);
  std::vector<PnrIndex> targets;
  targets.reserve(forwardArcs.size());
  PnrIndex current = attachmentEndpoint;
  for (PnrIndex forwardArc : forwardArcs) {
    if (forwardArc >= state_->graph_.routingArcs().size())
      return routeTreeError("path arc is outside FrozenRoutingGraph");
    if (state_->sourceEndpoint(forwardArc) != current)
      return routeTreeError("path arc does not continue from endpoint");
    const PnrIndex target = state_->graph_.routingArcs()[forwardArc].target;
    if (state_->lookupSlot(target))
      return routeTreeError("path leaves the tree and re-enters it");
    if (!pathEndpoints.insert(target).second)
      return routeTreeError("path repeats an endpoint");
    targets.push_back(target);
    current = target;
  }
  if (current != sinkEndpoint)
    return routeTreeError("path does not terminate at its sink obligation");
  if (!emptyTree && forwardArcs.empty() &&
      state_->nodes_[*attachmentSlot].sinkObligationCount != 0)
    return routeTreeError("sink obligation is already attached");

  const std::uint64_t addedCount =
      sizeValue(targets.size()) + static_cast<std::uint64_t>(emptyTree);
  auto finalCount = checkedPnrIndexAdd(nodeCountContext,
                                       state_->activeNodeCount_, addedCount);
  if (!finalCount)
    return finalCount.takeError();
  if (llvm::Error error = state_->ensureLookupCapacity(*finalCount))
    return error;

  const std::size_t reused =
      std::min<std::size_t>(state_->freeSlots_.size(), addedCount);
  state_->nodes_.reserve(state_->nodes_.size() +
                         static_cast<std::size_t>(addedCount) - reused);
  deltas_.reserve(deltas_.size() + static_cast<std::size_t>(addedCount) * 3 +
                  1);

  if (emptyTree) {
    auto root = addNode(state_->producerEndpoint_, getInvalidPnrIndex());
    if (!root)
      return root.takeError();
    attachmentSlot = *root;
  }

  PnrIndex parent = *attachmentSlot;
  for (auto [forwardArc, target] : llvm::zip(forwardArcs, targets)) {
    auto child = addNode(target, forwardArc);
    if (!child)
      return child.takeError();
    linkChild(parent, *child);
    parent = *child;
  }
  if (forwardArcs.empty() && state_->nodes_[parent].sinkObligationCount == 0)
    setSinkMetadata(parent, state_->requiredSinkCount(sinkEndpoint));
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpSink(PnrIndex sinkEndpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  const std::optional<PnrIndex> found = state_->lookupSlot(sinkEndpoint);
  if (!found || state_->nodes_[*found].sinkObligationCount == 0)
    return routeTreeError("sink obligation is not reached");

  deltas_.reserve(deltas_.size() + state_->nodes_.size() * 4 + 1);
  state_->freeSlots_.reserve(state_->freeSlots_.size() +
                             state_->activeNodeCount_);
  PnrIndex current = *found;
  setSinkMetadata(current, 0);
  const std::optional<PnrIndex> root =
      state_->lookupSlot(state_->producerEndpoint_);
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
      state_->lookupSlot(state_->producerEndpoint_);
  assert(root);
  if (*subtreeRoot == *root)
    return ripUpWholeNet();

  std::vector<PnrIndex> subtree;
  subtree.push_back(*subtreeRoot);
  for (std::size_t index = 0; index < subtree.size(); ++index) {
    const PnrIndex current = subtree[index];
    for (PnrIndex child = state_->nodes_[current].firstChild;
         child != getInvalidPnrIndex();
         child = state_->nodes_[child].nextSibling)
      subtree.push_back(child);
  }

  deltas_.reserve(deltas_.size() + subtree.size() + 3);
  state_->freeSlots_.reserve(state_->freeSlots_.size() + subtree.size());
  const PnrIndex parent = parentSlot(*subtreeRoot);
  detachNode(*subtreeRoot, parent);
  for (PnrIndex slot : subtree)
    removeNode(slot);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpWholeNet() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  deltas_.reserve(deltas_.size() + state_->activeNodeCount_);
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
  if (llvm::Error error = state_->verifyCandidate(routedCandidate))
    return error;

  state_->routed_ = routedCandidate;
  if (!routedCandidate) {
    state_->nodes_.clear();
    state_->freeSlots_.clear();
    state_->endpointSlots_.clear();
  }
  state_->transactionActive_ = false;
  deltas_.clear();
  state_ = nullptr;
  return llvm::Error::success();
}

void RouteTreeTransaction::rollback() noexcept {
  if (!state_)
    return;
  for (auto delta = deltas_.rbegin(); delta != deltas_.rend(); ++delta) {
    switch (delta->kind) {
    case DeltaKind::ModifiedNode:
      state_->nodes_[delta->slot] = delta->node;
      break;
    case DeltaKind::RemovedNode:
      assert(!state_->freeSlots_.empty() &&
             state_->freeSlots_.back() == delta->slot);
      state_->freeSlots_.pop_back();
      state_->nodes_[delta->slot] = delta->node;
      ++state_->activeNodeCount_;
      break;
    case DeltaKind::AddedNode:
      assert(state_->nodes_[delta->slot].isActive());
      --state_->activeNodeCount_;
      if (delta->appended) {
        assert(delta->slot + 1 == state_->nodes_.size());
        state_->nodes_.pop_back();
      } else {
        state_->nodes_[delta->slot] = {};
        state_->freeSlots_.push_back(delta->slot);
      }
      break;
    case DeltaKind::SinkMetadata:
      state_->nodes_[delta->slot].sinkObligationCount =
          delta->sinkObligationCount;
      break;
    }
  }
  assert(state_->nodes_.size() == initialNodeStorageSize_);
  assert(state_->activeNodeCount_ == initialActiveNodeCount_);
  state_->rebuildLookup(initialLookupCapacity_);
  state_->routed_ = initialRouted_;
  state_->transactionActive_ = false;
  deltas_.clear();
  state_ = nullptr;
}
