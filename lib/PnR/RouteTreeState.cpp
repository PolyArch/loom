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
constexpr PnrCapacityContext traversalDeltaCountContext{
    routeTreeArtifact, "transaction_traversal_deltas", "route_tree_nodes",
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

RouteTreeTransactionScratch::~RouteTreeTransactionScratch() {
  if (activeTransaction_)
    activeTransaction_->rollback();
}

std::size_t
RouteTreeTransactionScratch::retainedLookupRollbackStorageBytes() const {
  return lookupBaseline_.capacity() * sizeof(detail::RouteTreeLookupEntry);
}

void RouteTreeTransactionScratch::resetTransaction() {
  deltas_.clear();
  traversalDeltas_.clear();
  worklist_.clear();
  lookupBaselineActive_ = false;
}

llvm::Error loom::pnr::detail::preflightRouteTreeStateCapacity(
    std::uint64_t reachedEndpointCount, std::uint64_t sinkObligationCount) {
  if (llvm::Error error =
          preflightPnrIndexCapacity(nodeCountContext, reachedEndpointCount))
    return error;
  return preflightPnrIndexCapacity(sinkCountContext, sinkObligationCount);
}

llvm::Expected<RouteTreeStateHandle>
RouteTreeState::create(FrozenSpatialRoutingGraphHandle graph,
                       PnrIndex sinkObligationCount) {
  if (!graph)
    return routeTreeError("FrozenSpatialRoutingGraph owner is null");
  if (llvm::Error error = detail::preflightRouteTreeStateCapacity(
          sizeValue(graph->routingEndpoints().size()), sinkObligationCount))
    return std::move(error);
  return RouteTreeStateHandle(new RouteTreeState(
      std::move(graph),
      std::vector<SinkBinding>(static_cast<std::size_t>(sinkObligationCount))));
}

RouteTreeState::RouteTreeState(FrozenSpatialRoutingGraphHandle graph,
                               std::vector<SinkBinding> sinkBindings)
    : graph_(std::move(graph)), sinkBindings_(std::move(sinkBindings)) {}

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
    if (entry.isEmpty())
      return std::nullopt;
    if (entry.isOccupied() && entry.endpoint == endpoint)
      return entry.slot;
    bucket = (bucket + 1) & mask;
  }
  return std::nullopt;
}

llvm::Error RouteTreeTransaction::ensureLookupCapacity(PnrIndex requiredCount) {
  if (requiredCount == 0)
    return llvm::Error::success();

  std::size_t capacity =
      state_->endpointSlots_.empty() ? 8 : state_->endpointSlots_.size();
  const std::size_t required = static_cast<std::size_t>(requiredCount);
  while (required > capacity / 2) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2)
      return routeTreeError("endpoint-to-slot lookup capacity overflow");
    capacity *= 2;
  }
  const bool shouldPurgeTombstones =
      state_->lookupTombstoneCount_ > capacity / 2 - required;
  if (capacity != state_->endpointSlots_.size() || shouldPurgeTombstones)
    rehashLookup(capacity);
  return llvm::Error::success();
}

void RouteTreeTransaction::recordLookupBucket(std::size_t bucket) {
  if (scratch_->lookupBaselineActive_)
    return;
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::LookupBucket;
  delta.lookupIndex = bucket;
  delta.lookupTombstoneCount = state_->lookupTombstoneCount_;
  delta.lookupEntry = state_->endpointSlots_[bucket];
  scratch_->deltas_.push_back(delta);
}

void RouteTreeTransaction::insertLookupWithoutDelta(PnrIndex endpoint,
                                                    PnrIndex slot) {
  assert(!state_->endpointSlots_.empty());
  const std::size_t mask = state_->endpointSlots_.size() - 1;
  const std::size_t noBucket = std::numeric_limits<std::size_t>::max();
  std::size_t firstTombstone = noBucket;
  std::size_t bucket = RouteTreeState::hashEndpoint(endpoint) & mask;
  for (std::size_t probe = 0; probe < state_->endpointSlots_.size(); ++probe) {
    RouteTreeState::LookupEntry &entry = state_->endpointSlots_[bucket];
    if (entry.isOccupied()) {
      assert(entry.endpoint != endpoint && "duplicate reached endpoint");
    } else if (entry.tombstone) {
      if (firstTombstone == noBucket)
        firstTombstone = bucket;
    } else {
      const std::size_t destination =
          firstTombstone == noBucket ? bucket : firstTombstone;
      if (firstTombstone != noBucket)
        --state_->lookupTombstoneCount_;
      state_->endpointSlots_[destination] = {endpoint, slot, false};
      return;
    }
    bucket = (bucket + 1) & mask;
  }
  assert(firstTombstone != noBucket && "endpoint lookup has no insert slot");
  --state_->lookupTombstoneCount_;
  state_->endpointSlots_[firstTombstone] = {endpoint, slot, false};
}

void RouteTreeTransaction::insertLookup(PnrIndex endpoint, PnrIndex slot) {
  assert(!state_->endpointSlots_.empty());
  const std::size_t mask = state_->endpointSlots_.size() - 1;
  const std::size_t noBucket = std::numeric_limits<std::size_t>::max();
  std::size_t firstTombstone = noBucket;
  std::size_t bucket = RouteTreeState::hashEndpoint(endpoint) & mask;
  for (std::size_t probe = 0; probe < state_->endpointSlots_.size(); ++probe) {
    const RouteTreeState::LookupEntry &entry = state_->endpointSlots_[bucket];
    if (entry.isOccupied()) {
      assert(entry.endpoint != endpoint && "duplicate reached endpoint");
    } else if (entry.tombstone) {
      if (firstTombstone == noBucket)
        firstTombstone = bucket;
    } else {
      const std::size_t destination =
          firstTombstone == noBucket ? bucket : firstTombstone;
      recordLookupBucket(destination);
      if (firstTombstone != noBucket)
        --state_->lookupTombstoneCount_;
      state_->endpointSlots_[destination] = {endpoint, slot, false};
      return;
    }
    bucket = (bucket + 1) & mask;
  }
  assert(firstTombstone != noBucket && "endpoint lookup has no insert slot");
  recordLookupBucket(firstTombstone);
  --state_->lookupTombstoneCount_;
  state_->endpointSlots_[firstTombstone] = {endpoint, slot, false};
}

void RouteTreeTransaction::eraseLookup(PnrIndex endpoint) {
  assert(!state_->endpointSlots_.empty());
  const std::size_t mask = state_->endpointSlots_.size() - 1;
  std::size_t bucket = RouteTreeState::hashEndpoint(endpoint) & mask;
  for (std::size_t probe = 0; probe < state_->endpointSlots_.size(); ++probe) {
    RouteTreeState::LookupEntry &entry = state_->endpointSlots_[bucket];
    if (entry.isEmpty())
      break;
    if (entry.isOccupied() && entry.endpoint == endpoint) {
      recordLookupBucket(bucket);
      entry = {getInvalidPnrIndex(), getInvalidPnrIndex(), true};
      ++state_->lookupTombstoneCount_;
      return;
    }
    bucket = (bucket + 1) & mask;
  }
  assert(false && "erasing an endpoint absent from the lookup");
}

void RouteTreeTransaction::rehashLookup(std::size_t capacity) {
  std::vector<RouteTreeState::LookupEntry> replacement(capacity);
  std::vector<RouteTreeState::LookupEntry> discarded;
  const bool hadBaseline = scratch_->lookupBaselineActive_;
  if (!hadBaseline) {
    RouteTreeTransactionScratch::Delta delta;
    delta.kind = RouteTreeTransactionScratch::DeltaKind::LookupBaseline;
    delta.lookupTombstoneCount = state_->lookupTombstoneCount_;
    scratch_->deltas_.push_back(delta);
    scratch_->lookupBaselineActive_ = true;
    scratch_->lookupBaseline_.swap(state_->endpointSlots_);
  } else {
    discarded.swap(state_->endpointSlots_);
  }
  replacement.swap(state_->endpointSlots_);
  state_->lookupTombstoneCount_ = 0;
  const std::vector<RouteTreeState::LookupEntry> &previous =
      hadBaseline ? discarded : scratch_->lookupBaseline_;
  for (const RouteTreeState::LookupEntry &entry : previous)
    if (entry.isOccupied())
      insertLookupWithoutDelta(entry.endpoint, entry.slot);
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
  assert(arc < graph_->arcSources().size());
  return graph_->arcSources()[arc];
}

std::optional<PnrIndex> RouteTreeState::findNode(PnrIndex endpoint) const {
  return lookupSlot(endpoint);
}

const RouteTreeNode &RouteTreeState::node(PnrIndex slot) const {
  assert(slot < nodes_.size() && nodes_[slot].isActive());
  return nodes_[slot];
}

llvm::Error RouteTreeState::verify() const {
  if (activeTransaction_)
    return routeTreeError("cannot verify while a transaction is active");
  return verifyState();
}

llvm::Error RouteTreeState::verifyReplicationBranches() const {
  if (graph_->traversalReplicationGroups().size() !=
      graph_->traversals().size())
    return routeTreeError(
        "FrozenSpatialRoutingGraph has no exact replication projection");

  for (const RouteTreeNode &parent : nodes_) {
    if (!parent.isActive())
      continue;
    PnrIndex child = parent.firstChild;
    PnrIndex replicationGroup = getInvalidPnrIndex();
    std::size_t childCount = 0;
    while (child != getInvalidPnrIndex()) {
      if (child >= nodes_.size() || !nodes_[child].isActive())
        return routeTreeError("child linkage references an inactive slot");
      if (++childCount > activeNodeCount_)
        return routeTreeError("child linkage contains a cycle");
      const PnrIndex parentArc = nodes_[child].parentArc;
      if (parentArc == getInvalidPnrIndex() ||
          parentArc >= graph_->routingArcs().size())
        return routeTreeError("non-root node has no valid parent arc");
      const PnrIndex traversal = graph_->routingArcs()[parentArc].traversal;
      if (traversal >= graph_->traversalReplicationGroups().size())
        return routeTreeError("parent arc names an invalid traversal");
      const PnrIndex childGroup =
          graph_->traversalReplicationGroups()[traversal];
      if (childCount == 1) {
        replicationGroup = childGroup;
      } else if (replicationGroup == getInvalidPnrIndex() ||
                 childGroup != replicationGroup) {
        return routeTreeError(
            "route branch is not one explicit Fabric replication group");
      }
      child = nodes_[child].nextSibling;
    }
  }
  return llvm::Error::success();
}

llvm::Error RouteTreeState::verifyState() const {
  if (llvm::Error error = verifyReplicationBranches())
    return error;
  if (!endpointSlots_.empty() &&
      (endpointSlots_.size() < 8 ||
       (endpointSlots_.size() & (endpointSlots_.size() - 1)) != 0))
    return routeTreeError("endpoint-to-slot lookup capacity is invalid");

  std::size_t lookupCount = 0;
  std::size_t tombstoneCount = 0;
  for (const LookupEntry &entry : endpointSlots_) {
    if (entry.isOccupied()) {
      ++lookupCount;
      if (entry.tombstone || entry.slot >= nodes_.size() ||
          !nodes_[entry.slot].isActive() ||
          nodes_[entry.slot].endpoint != entry.endpoint ||
          lookupSlot(entry.endpoint) != entry.slot)
        return routeTreeError("endpoint-to-slot lookup diverges from nodes");
      continue;
    }
    if (entry.tombstone) {
      ++tombstoneCount;
      if (entry.slot != getInvalidPnrIndex())
        return routeTreeError("lookup tombstone retains a node slot");
    } else if (entry.slot != getInvalidPnrIndex()) {
      return routeTreeError("empty lookup bucket retains a node slot");
    }
  }
  if (tombstoneCount != lookupTombstoneCount_)
    return routeTreeError("lookup tombstone accounting is inconsistent");
  if (lookupCount + tombstoneCount > endpointSlots_.size() / 2)
    return routeTreeError("endpoint lookup probe occupancy is saturated");

  std::vector<std::uint8_t> visited(nodes_.size(), 0);
  for (PnrIndex slot : freeSlots_) {
    if (slot >= nodes_.size() || nodes_[slot].isActive())
      return routeTreeError("free slot is outside inactive node storage");
    if (visited[slot])
      return routeTreeError("free slot is duplicated");
    visited[slot] = 1;
  }

  std::size_t activeCount = 0;
  for (std::size_t slot = 0; slot < nodes_.size(); ++slot) {
    const RouteTreeNode &node = nodes_[slot];
    if (!node.isActive()) {
      if (!visited[slot])
        return routeTreeError("inactive node is absent from free slots");
      continue;
    }
    if (visited[slot])
      return routeTreeError("active node appears in free slots");
    ++activeCount;
    if (node.endpoint >= graph_->routingEndpoints().size())
      return routeTreeError(
          "node endpoint is outside FrozenSpatialRoutingGraph");
    if (lookupSlot(node.endpoint) != slot)
      return routeTreeError("endpoint-to-slot lookup diverges from nodes");
  }
  if (activeCount != activeNodeCount_ || lookupCount != activeCount)
    return routeTreeError("active-node accounting is inconsistent");

  if (activeCount == 0) {
    if (sourceEndpoint_ != getInvalidPnrIndex() ||
        boundSinkObligationCount_ != 0 || attachedSinkObligationCount_ != 0)
      return routeTreeError("explicit unrouted state retains bindings");
    for (const SinkBinding &binding : sinkBindings_)
      if (binding.endpoint != getInvalidPnrIndex() ||
          binding.nodeSlot != getInvalidPnrIndex() ||
          binding.previousAtNode != getInvalidPnrIndex() ||
          binding.nextAtNode != getInvalidPnrIndex())
        return routeTreeError("explicit unrouted state retains sink binding");
    if (!nodes_.empty() || !freeSlots_.empty() || !endpointSlots_.empty())
      return routeTreeError("explicit unrouted state retains route storage");
    return llvm::Error::success();
  }

  if (sourceEndpoint_ == getInvalidPnrIndex() ||
      sourceEndpoint_ >= graph_->routingEndpoints().size())
    return routeTreeError("routed state has no valid source binding");
  const std::optional<PnrIndex> rootSlot = lookupSlot(sourceEndpoint_);
  if (!rootSlot)
    return routeTreeError("routed state is missing its source root");
  const RouteTreeNode &root = nodes_[*rootSlot];
  if (root.parentArc != getInvalidPnrIndex() ||
      root.previousSibling != getInvalidPnrIndex() ||
      root.nextSibling != getInvalidPnrIndex())
    return routeTreeError("source root has parent or sibling linkage");

  std::vector<PnrIndex> sinkCounts(nodes_.size(), 0);
  std::vector<std::uint8_t> obligationReferences(sinkBindings_.size(), 0);
  std::size_t boundCount = 0;
  std::size_t attachedCount = 0;
  for (const SinkBinding &binding : sinkBindings_) {
    if (binding.endpoint == getInvalidPnrIndex()) {
      if (binding.nodeSlot != getInvalidPnrIndex() ||
          binding.previousAtNode != getInvalidPnrIndex() ||
          binding.nextAtNode != getInvalidPnrIndex())
        return routeTreeError("unbound sink retains node linkage");
      continue;
    }
    ++boundCount;
    if (binding.endpoint >= graph_->routingEndpoints().size())
      return routeTreeError(
          "sink binding is outside FrozenSpatialRoutingGraph");
    if (binding.nodeSlot == getInvalidPnrIndex()) {
      if (binding.previousAtNode != getInvalidPnrIndex() ||
          binding.nextAtNode != getInvalidPnrIndex())
        return routeTreeError("unattached sink retains node linkage");
      continue;
    }
    ++attachedCount;
    if (binding.nodeSlot >= nodes_.size() ||
        !nodes_[binding.nodeSlot].isActive() ||
        nodes_[binding.nodeSlot].endpoint != binding.endpoint)
      return routeTreeError("sink obligation is not covered");
    ++sinkCounts[binding.nodeSlot];
  }
  if (boundCount != boundSinkObligationCount_ ||
      attachedCount != attachedSinkObligationCount_)
    return routeTreeError("sink binding accounting is inconsistent");
  if (attachedCount != sinkBindings_.size())
    return routeTreeError("sink obligation is not covered");

  for (std::size_t slot = 0; slot < nodes_.size(); ++slot) {
    const RouteTreeNode &node = nodes_[slot];
    if (!node.isActive())
      continue;
    PnrIndex previous = getInvalidPnrIndex();
    PnrIndex obligation = node.firstSinkObligation;
    PnrIndex count = 0;
    while (obligation != getInvalidPnrIndex()) {
      if (obligation >= sinkBindings_.size())
        return routeTreeError("node sink linkage is outside the freeze");
      const SinkBinding &binding = sinkBindings_[obligation];
      if (binding.nodeSlot != slot || binding.endpoint != node.endpoint ||
          binding.previousAtNode != previous)
        return routeTreeError("node sink linkage is inconsistent");
      if (obligationReferences[obligation])
        return routeTreeError("sink obligation appears at multiple nodes");
      obligationReferences[obligation] = 1;
      previous = obligation;
      obligation = binding.nextAtNode;
      ++count;
    }
    if (count != node.sinkObligationCount || count != sinkCounts[slot])
      return routeTreeError("node sink metadata diverges from bindings");
  }
  for (std::size_t obligation = 0; obligation < sinkBindings_.size();
       ++obligation) {
    const bool attached =
        sinkBindings_[obligation].nodeSlot != getInvalidPnrIndex();
    if (obligationReferences[obligation] != static_cast<std::uint8_t>(attached))
      return routeTreeError("sink obligation is absent from node metadata");
  }

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
          childNode.parentArc >= graph_->routingArcs().size())
        return routeTreeError("non-root node has no valid parent arc");
      if (graph_->routingArcs()[childNode.parentArc].target !=
              childNode.endpoint ||
          arcSourceEndpoint(childNode.parentArc) != parent.endpoint)
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

  std::fill(visited.begin(), visited.end(), 0);
  std::vector<PnrIndex> worklist;
  worklist.push_back(*rootSlot);
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
    return routeTreeError("parent chain does not reach the source root");
  return llvm::Error::success();
}

llvm::Expected<RouteTreeTransaction>
RouteTreeState::beginTransaction(RouteTreeTransactionScratch &scratch) & {
  if (activeTransaction_)
    return routeTreeError("another transaction is already active");
  if (scratch.activeTransaction_)
    return routeTreeError("transaction scratch is already in use");
  scratch.resetTransaction();
  return RouteTreeTransaction(shared_from_this(), scratch);
}

RouteTreeTransaction::RouteTreeTransaction(RouteTreeStateHandle state,
                                           RouteTreeTransactionScratch &scratch)
    : state_(std::move(state)), scratch_(&scratch),
      initialNodeStorageSize_(state_->nodes_.size()),
      initialActiveNodeCount_(state_->activeNodeCount_),
      initialBoundSinkObligationCount_(state_->boundSinkObligationCount_),
      initialAttachedSinkObligationCount_(
          state_->attachedSinkObligationCount_) {
  state_->activeTransaction_ = this;
  scratch.activeTransaction_ = this;
}

RouteTreeTransaction::RouteTreeTransaction(
    RouteTreeTransaction &&other) noexcept
    : state_(std::move(other.state_)), scratch_(other.scratch_),
      initialNodeStorageSize_(other.initialNodeStorageSize_),
      initialActiveNodeCount_(other.initialActiveNodeCount_),
      initialBoundSinkObligationCount_(other.initialBoundSinkObligationCount_),
      initialAttachedSinkObligationCount_(
          other.initialAttachedSinkObligationCount_),
      prepared_(other.prepared_) {
  if (state_)
    state_->activeTransaction_ = this;
  if (scratch_)
    scratch_->activeTransaction_ = this;
  other.scratch_ = nullptr;
}

RouteTreeTransaction::~RouteTreeTransaction() {
  if (state_)
    rollback();
}

void RouteTreeTransaction::recordModifiedNode(PnrIndex slot) {
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::ModifiedNode;
  delta.key = slot;
  delta.node = state_->nodes_[slot];
  scratch_->deltas_.push_back(delta);
}

void RouteTreeTransaction::setSourceBinding(PnrIndex endpoint) {
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::SourceBinding;
  delta.value0 = state_->sourceEndpoint_;
  scratch_->deltas_.push_back(delta);
  state_->sourceEndpoint_ = endpoint;
}

void RouteTreeTransaction::setSinkBinding(PnrIndex obligation,
                                          PnrIndex endpoint, PnrIndex nodeSlot,
                                          PnrIndex previousAtNode,
                                          PnrIndex nextAtNode) {
  const RouteTreeState::SinkBinding previous =
      state_->sinkBindings_[obligation];
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::SinkBinding;
  delta.key = obligation;
  delta.value0 = previous.endpoint;
  delta.value1 = previous.nodeSlot;
  delta.value2 = previous.previousAtNode;
  delta.value3 = previous.nextAtNode;
  scratch_->deltas_.push_back(delta);

  const bool wasBound = previous.endpoint != getInvalidPnrIndex();
  const bool isBound = endpoint != getInvalidPnrIndex();
  const bool wasAttached = previous.nodeSlot != getInvalidPnrIndex();
  const bool isAttached = nodeSlot != getInvalidPnrIndex();
  if (!wasBound && isBound)
    ++state_->boundSinkObligationCount_;
  else if (wasBound && !isBound)
    --state_->boundSinkObligationCount_;
  if (!wasAttached && isAttached)
    ++state_->attachedSinkObligationCount_;
  else if (wasAttached && !isAttached)
    --state_->attachedSinkObligationCount_;
  state_->sinkBindings_[obligation] = {endpoint, nodeSlot, previousAtNode,
                                       nextAtNode};
}

void RouteTreeTransaction::attachSinkBinding(PnrIndex obligation,
                                             PnrIndex nodeSlot,
                                             PnrIndex finalSinkCount) {
  RouteTreeNode &node = state_->nodes_[nodeSlot];
  const PnrIndex oldHead = node.firstSinkObligation;
  const PnrIndex endpoint = state_->sinkBindings_[obligation].endpoint;
  recordModifiedNode(nodeSlot);
  if (oldHead != getInvalidPnrIndex()) {
    const RouteTreeState::SinkBinding head = state_->sinkBindings_[oldHead];
    setSinkBinding(oldHead, head.endpoint, head.nodeSlot, obligation,
                   head.nextAtNode);
  }
  setSinkBinding(obligation, endpoint, nodeSlot, getInvalidPnrIndex(), oldHead);
  node.firstSinkObligation = obligation;
  node.sinkObligationCount = finalSinkCount;
}

void RouteTreeTransaction::unlinkSinkBinding(PnrIndex obligation) {
  const RouteTreeState::SinkBinding binding = state_->sinkBindings_[obligation];
  RouteTreeNode &node = state_->nodes_[binding.nodeSlot];
  recordModifiedNode(binding.nodeSlot);
  if (binding.previousAtNode == getInvalidPnrIndex()) {
    node.firstSinkObligation = binding.nextAtNode;
  } else {
    const RouteTreeState::SinkBinding previous =
        state_->sinkBindings_[binding.previousAtNode];
    setSinkBinding(binding.previousAtNode, previous.endpoint, previous.nodeSlot,
                   previous.previousAtNode, binding.nextAtNode);
  }
  if (binding.nextAtNode != getInvalidPnrIndex()) {
    const RouteTreeState::SinkBinding next =
        state_->sinkBindings_[binding.nextAtNode];
    setSinkBinding(binding.nextAtNode, next.endpoint, next.nodeSlot,
                   binding.previousAtNode, next.nextAtNode);
  }
  --node.sinkObligationCount;
  setSinkBinding(obligation, getInvalidPnrIndex(), getInvalidPnrIndex(),
                 getInvalidPnrIndex(), getInvalidPnrIndex());
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
  insertLookup(endpoint, slot);
  ++state_->activeNodeCount_;
  recordTraversalDelta(parentArc, true);
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
  RouteTreeTransactionScratch::Delta delta;
  delta.kind = RouteTreeTransactionScratch::DeltaKind::RemovedNode;
  delta.key = slot;
  delta.node = snapshot;
  scratch_->deltas_.push_back(delta);
  eraseLookup(snapshot.endpoint);
  state_->nodes_[slot] = {};
  state_->freeSlots_.push_back(slot);
  --state_->activeNodeCount_;
  recordTraversalDelta(snapshot.parentArc, false);
}

void RouteTreeTransaction::recordTraversalDelta(PnrIndex parentArc,
                                                bool added) {
  if (parentArc == getInvalidPnrIndex())
    return;
  assert(parentArc < state_->graph_->routingArcs().size());
  RouteTreeTraversalDelta delta;
  delta.traversal = state_->graph_->routingArcs()[parentArc].traversal;
  delta.added = added ? 1 : 0;
  delta.removed = added ? 0 : 1;
  scratch_->traversalDeltas_.push_back(delta);
}

llvm::Error RouteTreeTransaction::bindSource(PnrIndex endpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  if (endpoint >= state_->graph_->routingEndpoints().size())
    return routeTreeError(
        "source endpoint is outside FrozenSpatialRoutingGraph");
  if (state_->activeNodeCount_ != 0 && state_->sourceEndpoint_ != endpoint)
    return routeTreeError("routed source requires whole-net rip-up");
  if (state_->sourceEndpoint_ != endpoint) {
    scratch_->deltas_.reserve(scratch_->deltas_.size() + 1);
    setSourceBinding(endpoint);
  }
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::bindSink(PnrIndex obligation,
                                           PnrIndex endpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  if (obligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  if (endpoint >= state_->graph_->routingEndpoints().size())
    return routeTreeError("sink endpoint is outside FrozenSpatialRoutingGraph");
  const RouteTreeState::SinkBinding &binding =
      state_->sinkBindings_[obligation];
  if (binding.nodeSlot != getInvalidPnrIndex())
    return routeTreeError("attached sink obligation requires rip-up");
  if (binding.endpoint != endpoint) {
    scratch_->deltas_.reserve(scratch_->deltas_.size() + 1);
    setSinkBinding(obligation, endpoint, getInvalidPnrIndex(),
                   getInvalidPnrIndex(), getInvalidPnrIndex());
  }
  return llvm::Error::success();
}

llvm::Error
RouteTreeTransaction::attachPath(PnrIndex attachmentEndpoint,
                                 llvm::ArrayRef<PnrIndex> forwardArcs,
                                 PnrIndex sinkObligation) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  if (sinkObligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  const RouteTreeState::SinkBinding binding =
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

  if (scratch_->pathMarks_.size() < state_->graph_->routingEndpoints().size())
    scratch_->pathMarks_.resize(state_->graph_->routingEndpoints().size(), 0);
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
    if (forwardArc >= state_->graph_->routingArcs().size())
      return routeTreeError("path arc is outside FrozenSpatialRoutingGraph");
    if (state_->arcSourceEndpoint(forwardArc) != current)
      return routeTreeError("path arc does not continue from endpoint");
    const PnrIndex target = state_->graph_->routingArcs()[forwardArc].target;
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

  const std::size_t reused =
      std::min<std::size_t>(state_->freeSlots_.size(), addedCount);
  state_->nodes_.reserve(state_->nodes_.size() +
                         static_cast<std::size_t>(addedCount) - reused);
  scratch_->deltas_.reserve(scratch_->deltas_.size() +
                            static_cast<std::size_t>(addedCount) * 3 + 5);
  scratch_->traversalDeltas_.reserve(scratch_->traversalDeltas_.size() +
                                     forwardArcs.size());
  if (llvm::Error error = ensureLookupCapacity(*finalNodeCount))
    return error;

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
  attachSinkBinding(sinkObligation, parent, *finalSinkCount);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpSink(PnrIndex sinkObligation) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  if (sinkObligation >= state_->sinkBindings_.size())
    return routeTreeError("sink obligation ordinal is outside the freeze");
  const RouteTreeState::SinkBinding binding =
      state_->sinkBindings_[sinkObligation];
  if (binding.endpoint == getInvalidPnrIndex())
    return routeTreeError("sink obligation has no endpoint binding");
  if (binding.nodeSlot == getInvalidPnrIndex()) {
    scratch_->deltas_.reserve(scratch_->deltas_.size() + 1);
    setSinkBinding(sinkObligation, getInvalidPnrIndex(), getInvalidPnrIndex(),
                   getInvalidPnrIndex(), getInvalidPnrIndex());
    return llvm::Error::success();
  }
  if (binding.nodeSlot >= state_->nodes_.size() ||
      !state_->nodes_[binding.nodeSlot].isActive() ||
      state_->nodes_[binding.nodeSlot].endpoint != binding.endpoint ||
      state_->nodes_[binding.nodeSlot].sinkObligationCount == 0)
    return routeTreeError("sink binding diverges from the route tree");

  const std::optional<PnrIndex> root =
      state_->lookupSlot(state_->sourceEndpoint_);
  if (!root)
    return routeTreeError("routed state is missing its source root");
  scratch_->worklist_.clear();
  if (state_->nodes_[binding.nodeSlot].firstChild == getInvalidPnrIndex() &&
      state_->nodes_[binding.nodeSlot].sinkObligationCount == 1) {
    PnrIndex child = binding.nodeSlot;
    while (child != *root) {
      scratch_->worklist_.push_back(child);
      const PnrIndex parent = parentSlot(child);
      const RouteTreeNode &childNode = state_->nodes_[child];
      const RouteTreeNode &parentNode = state_->nodes_[parent];
      if (parentNode.sinkObligationCount != 0 ||
          parentNode.firstChild != child ||
          childNode.previousSibling != getInvalidPnrIndex() ||
          childNode.nextSibling != getInvalidPnrIndex())
        break;
      child = parent;
    }
  }

  const std::size_t pruneCount = scratch_->worklist_.size();
  scratch_->deltas_.reserve(scratch_->deltas_.size() + 4 + pruneCount * 5);
  scratch_->traversalDeltas_.reserve(scratch_->traversalDeltas_.size() +
                                     pruneCount);
  state_->freeSlots_.reserve(state_->freeSlots_.size() + pruneCount);
  unlinkSinkBinding(sinkObligation);
  for (PnrIndex slot : scratch_->worklist_) {
    const PnrIndex parent = parentSlot(slot);
    detachNode(slot, parent);
    removeNode(slot);
  }
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpSubtree(PnrIndex subtreeRootEndpoint) {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  const std::optional<PnrIndex> subtreeRoot =
      state_->lookupSlot(subtreeRootEndpoint);
  if (!subtreeRoot)
    return routeTreeError("subtree root endpoint is not reached");
  const std::optional<PnrIndex> root =
      state_->lookupSlot(state_->sourceEndpoint_);
  if (!root)
    return routeTreeError("routed state is missing its source root");
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

  std::size_t bindingCount = 0;
  for (PnrIndex slot : scratch_->worklist_)
    for (PnrIndex obligation = state_->nodes_[slot].firstSinkObligation;
         obligation != getInvalidPnrIndex();
         obligation = state_->sinkBindings_[obligation].nextAtNode)
      ++bindingCount;
  scratch_->deltas_.reserve(scratch_->deltas_.size() + bindingCount +
                            scratch_->worklist_.size() * 2 + 3);
  scratch_->traversalDeltas_.reserve(scratch_->traversalDeltas_.size() +
                                     scratch_->worklist_.size());
  state_->freeSlots_.reserve(state_->freeSlots_.size() +
                             scratch_->worklist_.size());

  for (PnrIndex slot : scratch_->worklist_) {
    PnrIndex obligation = state_->nodes_[slot].firstSinkObligation;
    while (obligation != getInvalidPnrIndex()) {
      const PnrIndex next = state_->sinkBindings_[obligation].nextAtNode;
      setSinkBinding(obligation, getInvalidPnrIndex(), getInvalidPnrIndex(),
                     getInvalidPnrIndex(), getInvalidPnrIndex());
      obligation = next;
    }
  }
  const PnrIndex parent = parentSlot(*subtreeRoot);
  detachNode(*subtreeRoot, parent);
  for (PnrIndex slot : scratch_->worklist_)
    removeNode(slot);
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::ripUpWholeNet() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return routeTreeError("transaction is already prepared");
  const std::size_t activeNodeCount =
      static_cast<std::size_t>(state_->activeNodeCount_);
  const std::size_t boundSinkCount =
      static_cast<std::size_t>(state_->boundSinkObligationCount_);
  scratch_->deltas_.reserve(scratch_->deltas_.size() + activeNodeCount * 2 +
                            boundSinkCount + 1);
  if (activeNodeCount != 0)
    scratch_->traversalDeltas_.reserve(scratch_->traversalDeltas_.size() +
                                       activeNodeCount - 1);
  state_->freeSlots_.reserve(state_->freeSlots_.size() + activeNodeCount);

  for (PnrIndex obligation = 0; obligation < state_->sinkBindings_.size();
       ++obligation) {
    const RouteTreeState::SinkBinding &binding =
        state_->sinkBindings_[obligation];
    if (binding.endpoint != getInvalidPnrIndex())
      setSinkBinding(obligation, getInvalidPnrIndex(), getInvalidPnrIndex(),
                     getInvalidPnrIndex(), getInvalidPnrIndex());
  }
  if (state_->sourceEndpoint_ != getInvalidPnrIndex())
    setSourceBinding(getInvalidPnrIndex());
  for (std::size_t slot = 0; slot < state_->nodes_.size(); ++slot)
    if (state_->nodes_[slot].isActive())
      removeNode(static_cast<PnrIndex>(slot));
  return llvm::Error::success();
}

llvm::Error RouteTreeTransaction::verify() const {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  return state_->verifyState();
}

llvm::Expected<llvm::ArrayRef<RouteTreeTraversalDelta>>
RouteTreeTransaction::prepare() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (prepared_)
    return llvm::ArrayRef<RouteTreeTraversalDelta>(scratch_->traversalDeltas_);

  if (state_->activeNodeCount_ == 0) {
    if (state_->sourceEndpoint_ != getInvalidPnrIndex() ||
        state_->boundSinkObligationCount_ != 0 ||
        state_->attachedSinkObligationCount_ != 0)
      return routeTreeError(
          "explicit unrouted candidate retains physical bindings");
  } else {
    if (state_->sourceEndpoint_ == getInvalidPnrIndex() ||
        state_->boundSinkObligationCount_ != state_->sinkBindings_.size() ||
        state_->attachedSinkObligationCount_ != state_->sinkBindings_.size())
      return routeTreeError("sink obligation is not covered");
    const std::optional<PnrIndex> root =
        state_->lookupSlot(state_->sourceEndpoint_);
    if (!root || state_->nodes_[*root].parentArc != getInvalidPnrIndex())
      return routeTreeError("routed state is missing its source root");
  }
  if (llvm::Error error = state_->verifyReplicationBranches())
    return std::move(error);

  auto &deltas = scratch_->traversalDeltas_;
  llvm::sort(deltas, [](const RouteTreeTraversalDelta &lhs,
                        const RouteTreeTraversalDelta &rhs) {
    return lhs.traversal < rhs.traversal;
  });
  std::size_t write = 0;
  for (std::size_t begin = 0; begin < deltas.size();) {
    const PnrIndex traversal = deltas[begin].traversal;
    PnrIndex removed = 0;
    PnrIndex added = 0;
    std::size_t end = begin;
    for (; end < deltas.size() && deltas[end].traversal == traversal; ++end) {
      auto nextRemoved = checkedPnrIndexAdd(traversalDeltaCountContext, removed,
                                            deltas[end].removed);
      if (!nextRemoved)
        return nextRemoved.takeError();
      removed = *nextRemoved;
      auto nextAdded = checkedPnrIndexAdd(traversalDeltaCountContext, added,
                                          deltas[end].added);
      if (!nextAdded)
        return nextAdded.takeError();
      added = *nextAdded;
    }
    const PnrIndex cancelled = std::min(removed, added);
    removed -= cancelled;
    added -= cancelled;
    if (removed != 0 || added != 0)
      deltas[write++] = {traversal, removed, added};
    begin = end;
  }
  deltas.resize(write);
  prepared_ = true;
  return llvm::ArrayRef<RouteTreeTraversalDelta>(deltas);
}

llvm::Expected<const RouteTreeState *>
RouteTreeTransaction::preparedState() const {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  if (!prepared_)
    return routeTreeError("transaction has not been prepared");
  if (state_->isRouted())
    if (llvm::Error error = state_->verifyState())
      return error;
  return state_.get();
}

llvm::Error RouteTreeTransaction::commit() {
  if (!state_)
    return routeTreeError("transaction is no longer active");
  auto prepared = prepare();
  if (!prepared)
    return prepared.takeError();
  if (state_->activeNodeCount_ == 0) {
    state_->nodes_.clear();
    state_->freeSlots_.clear();
    state_->endpointSlots_.clear();
    state_->lookupTombstoneCount_ = 0;
    finish();
    return llvm::Error::success();
  }

  finish();
  return llvm::Error::success();
}

void RouteTreeTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
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
      ++state_->activeNodeCount_;
      break;
    case RouteTreeTransactionScratch::DeltaKind::AddedNode:
      assert(state_->nodes_[delta->key].isActive());
      --state_->activeNodeCount_;
      if (delta->appended) {
        assert(delta->key + 1 == state_->nodes_.size());
        state_->nodes_.pop_back();
      } else {
        state_->nodes_[delta->key] = {};
        state_->freeSlots_.push_back(delta->key);
      }
      break;
    case RouteTreeTransactionScratch::DeltaKind::SourceBinding:
      state_->sourceEndpoint_ = delta->value0;
      break;
    case RouteTreeTransactionScratch::DeltaKind::SinkBinding:
      state_->sinkBindings_[delta->key] = {delta->value0, delta->value1,
                                           delta->value2, delta->value3};
      break;
    case RouteTreeTransactionScratch::DeltaKind::LookupBucket:
      state_->endpointSlots_[delta->lookupIndex] = delta->lookupEntry;
      state_->lookupTombstoneCount_ = delta->lookupTombstoneCount;
      break;
    case RouteTreeTransactionScratch::DeltaKind::LookupBaseline:
      state_->endpointSlots_.swap(scratch_->lookupBaseline_);
      state_->lookupTombstoneCount_ = delta->lookupTombstoneCount;
      break;
    }
  }
  assert(state_->nodes_.size() == initialNodeStorageSize_);
  assert(state_->activeNodeCount_ == initialActiveNodeCount_);
  state_->boundSinkObligationCount_ = initialBoundSinkObligationCount_;
  state_->attachedSinkObligationCount_ = initialAttachedSinkObligationCount_;
  finish();
}
