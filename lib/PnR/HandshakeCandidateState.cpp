#include "PnR/HandshakeCandidateState.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace loom::pnr::detail {

struct HandshakeNodeIdentity final {
  std::optional<::loom::fabric::HandshakeSignalRef> boundarySignal;
  PnrIndex owner = 0;
  std::uint32_t localNode = 0;

  friend bool operator==(const HandshakeNodeIdentity &lhs,
                         const HandshakeNodeIdentity &rhs) {
    return lhs.boundarySignal == rhs.boundarySignal && lhs.owner == rhs.owner &&
           lhs.localNode == rhs.localNode;
  }
};

struct HandshakeArcIdentity final {
  HandshakeNodeIdentity source;
  HandshakeNodeIdentity destination;

  friend bool operator==(const HandshakeArcIdentity &lhs,
                         const HandshakeArcIdentity &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

struct MaterializedHandshakeGraph final {
  std::vector<std::optional<::loom::fabric::HandshakeSignalRef>> nodeSignals;
  std::vector<HandshakeNodeIdentity> nodeIdentities;
  std::vector<FrozenSpatialHandshakeArc> arcs;
  std::vector<HandshakeArcIdentity> arcIdentities;
  std::vector<std::vector<PnrIndex>> arcContributors;
  std::vector<std::vector<PnrIndex>> outgoingArcs;
  std::vector<std::vector<PnrIndex>> reverseArcs;
  std::map<std::string, PnrIndex> nodeOrdinals;
  std::map<std::pair<PnrIndex, PnrIndex>, PnrIndex> arcOrdinals;
  std::vector<PnrIndex> order;
  std::vector<PnrIndex> ranks;
  std::vector<PnrIndex> cycleWitness;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
};

struct ChangedArcContribution final {
  HandshakeArcIdentity arc;
  std::string sourceKey;
  std::string destinationKey;
  PnrIndex fragment = 0;
  bool add = false;
};

struct HandshakeArcChange final {
  HandshakeArcIdentity arc;
  std::string sourceKey;
  std::string destinationKey;
  std::size_t contributionOffset = 0;
  std::size_t contributionCount = 0;
  PnrIndex additionCount = 0;
  PnrIndex removalCount = 0;
};

struct HandshakeCandidateScratchStorage final {
  std::vector<ChangedArcContribution> changedContributions;
  std::vector<HandshakeArcChange> arcChanges;
  std::vector<PnrIndex> removedArcOrdinals;
  std::vector<PnrIndex> insertedArcChanges;
  std::map<std::string, PnrIndex> newNodeOrdinals;
  std::vector<HandshakeNodeIdentity> newNodes;
  std::vector<FrozenSpatialHandshakeArc> insertedArcOrdinals;
  std::vector<std::uint64_t> reachabilityMarks;
  std::vector<PnrIndex> reachabilityWorklist;
  std::vector<std::uint64_t> backwardMarks;
  std::vector<PnrIndex> backwardWorklist;
  std::vector<PnrIndex> reorderedNodes;
  std::uint64_t reachabilityEpoch = 0;
  std::uint64_t backwardEpoch = 0;
};

} // namespace loom::pnr::detail

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid handshake candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

template <typename T>
std::size_t retainedNestedBytes(const std::vector<std::vector<T>> &values) {
  std::size_t bytes = retainedBytes(values);
  for (const auto &value : values)
    bytes += retainedBytes(value);
  return bytes;
}

template <typename Key, typename Value>
std::size_t retainedMapBytes(const std::map<Key, Value> &values) {
  return values.size() * (sizeof(typename std::map<Key, Value>::value_type) +
                          4 * sizeof(void *));
}

llvm::Error increment(PnrIndex &value, llvm::StringRef subject) {
  if (value == std::numeric_limits<PnrIndex>::max())
    return candidateError(subject + " refcount overflows PnrIndex");
  ++value;
  return llvm::Error::success();
}

llvm::Expected<PnrIndex> checkedIndex(std::size_t value,
                                      llvm::StringRef subject) {
  if (value >= static_cast<std::size_t>(getInvalidPnrIndex()))
    return candidateError(subject + " exceeds PnrIndex");
  return static_cast<PnrIndex>(value);
}

void addWork(std::uint64_t &work, std::uint64_t amount = 1) {
  work = amount > std::numeric_limits<std::uint64_t>::max() - work
             ? std::numeric_limits<std::uint64_t>::max()
             : work + amount;
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin) {
  const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - begin)
                         .count();
  return count <= 0 ? 0 : static_cast<std::uint64_t>(count);
}

std::string signalKey(const ::loom::fabric::HandshakeSignalRef &signal) {
  std::vector<std::uint8_t> bytes =
      ::loom::fabric::canonicalFabricBytes(signal.endpoint);
  bytes.push_back(static_cast<std::uint8_t>(signal.signal));
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::string nodeKey(const detail::HandshakeNodeIdentity &identity) {
  if (identity.boundarySignal)
    return std::string(1, '\1') + signalKey(*identity.boundarySignal);
  std::string key(9, '\0');
  key[0] = '\2';
  const auto append = [&](std::size_t offset, std::uint32_t value) {
    for (std::size_t byte = 0; byte != 4; ++byte)
      key[offset + byte] = static_cast<char>(value >> (24 - byte * 8));
  };
  append(1, identity.owner);
  append(5, identity.localNode);
  return key;
}

llvm::Expected<detail::HandshakeNodeIdentity>
nodeIdentity(PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
             std::uint32_t localNode) {
  if (localNode >= model.nodeCount())
    return candidateError("active handshake node is out of range");
  const ::loom::fabric::HandshakeOwnerNode node = model.node(localNode);
  detail::HandshakeNodeIdentity identity;
  identity.boundarySignal = node.boundarySignal;
  if (!node.boundarySignal) {
    identity.owner = owner;
    identity.localNode = localNode;
  }
  return identity;
}

llvm::Expected<detail::HandshakeArcIdentity>
arcIdentity(PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
            const ::loom::fabric::HandshakeOwnerArc &arc) {
  auto source = nodeIdentity(owner, model, arc.source);
  if (!source)
    return source.takeError();
  auto destination = nodeIdentity(owner, model, arc.destination);
  if (!destination)
    return destination.takeError();
  return detail::HandshakeArcIdentity{std::move(*source),
                                      std::move(*destination)};
}

struct RebuiltHandshakeSelection final {
  std::vector<PnrIndex> fragmentRefcounts;
  std::vector<PnrIndex> activeFragments;
  std::vector<PnrIndex> traversalRefcounts;
  std::vector<PnrIndex> allGroupSelectedWitnessCounts;
};

llvm::Expected<RebuiltHandshakeSelection>
rebuildHandshakeSelection(const FrozenSpatialHandshakeIndex &index,
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

  RebuiltHandshakeSelection result;
  result.fragmentRefcounts.assign(index.fragments().size(), 0);
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
  result.activeFragments.reserve(result.fragmentRefcounts.size());
  for (auto [fragment, refcount] : llvm::enumerate(result.fragmentRefcounts))
    if (refcount != 0)
      result.activeFragments.push_back(static_cast<PnrIndex>(fragment));
  return result;
}

void buildCycleWitness(detail::MaterializedHandshakeGraph &graph) {
  struct Frame final {
    PnrIndex node = 0;
    std::size_t next = 0;
  };
  std::vector<std::uint8_t> colors(graph.nodeSignals.size(), 0);
  std::vector<PnrIndex> parentArcs(graph.nodeSignals.size(),
                                   getInvalidPnrIndex());
  std::vector<Frame> stack;
  for (PnrIndex root = 0; root < graph.nodeSignals.size(); ++root) {
    if (colors[root] != 0)
      continue;
    colors[root] = 1;
    stack.push_back({root, 0});
    while (!stack.empty()) {
      Frame &frame = stack.back();
      if (frame.next == graph.outgoingArcs[frame.node].size()) {
        colors[frame.node] = 2;
        stack.pop_back();
        continue;
      }
      const PnrIndex arc = graph.outgoingArcs[frame.node][frame.next++];
      const PnrIndex destination = graph.arcs[arc].destination;
      if (colors[destination] == 0) {
        colors[destination] = 1;
        parentArcs[destination] = arc;
        stack.push_back({destination, 0});
        continue;
      }
      if (colors[destination] != 1)
        continue;
      graph.cycleWitness.push_back(arc);
      PnrIndex node = frame.node;
      while (node != destination) {
        const PnrIndex parent = parentArcs[node];
        if (parent == getInvalidPnrIndex()) {
          graph.cycleWitness.clear();
          return;
        }
        graph.cycleWitness.push_back(parent);
        node = graph.arcs[parent].source;
      }
      std::reverse(graph.cycleWitness.begin(), graph.cycleWitness.end());
      return;
    }
  }
}

llvm::Expected<detail::MaterializedHandshakeGraph>
materializeHandshakeGraph(const FrozenSpatialHandshakeIndex &index,
                          llvm::ArrayRef<PnrIndex> activeFragments) {
  const auto begin = std::chrono::steady_clock::now();
  detail::MaterializedHandshakeGraph graph;
  const auto models = index.ownerModels();
  const auto fragments = index.fragments();
  std::map<std::pair<PnrIndex, PnrIndex>, std::vector<PnrIndex>> arcFragments;
  std::vector<std::vector<PnrIndex>> localNodeOrdinals(models.size());

  const auto resolveNode =
      [&](PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
          std::uint32_t localNode) -> llvm::Expected<PnrIndex> {
    if (localNode >= model.nodeCount())
      return candidateError("active handshake node is out of range");
    std::vector<PnrIndex> &cache = localNodeOrdinals[owner];
    if (cache.empty())
      cache.assign(model.nodeCount(), getInvalidPnrIndex());
    if (cache[localNode] != getInvalidPnrIndex())
      return cache[localNode];
    auto identity = nodeIdentity(owner, model, localNode);
    if (!identity)
      return identity.takeError();
    const std::string key = nodeKey(*identity);
    auto found = graph.nodeOrdinals.find(key);
    if (found != graph.nodeOrdinals.end()) {
      cache[localNode] = found->second;
      return found->second;
    }
    auto ordinal = checkedIndex(graph.nodeSignals.size(), "handshake node");
    if (!ordinal)
      return ordinal.takeError();
    graph.nodeOrdinals.emplace(key, *ordinal);
    graph.nodeSignals.push_back(identity->boundarySignal);
    graph.nodeIdentities.push_back(std::move(*identity));
    graph.outgoingArcs.emplace_back();
    graph.reverseArcs.emplace_back();
    cache[localNode] = *ordinal;
    addWork(graph.deterministicWork);
    return *ordinal;
  };

  PnrIndex previousFragment = 0;
  bool hasPrevious = false;
  for (PnrIndex fragmentOrdinal : activeFragments) {
    if (fragmentOrdinal >= fragments.size())
      return candidateError("active handshake fragment is out of range");
    if (hasPrevious && fragmentOrdinal <= previousFragment)
      return candidateError(
          "active handshake fragments are not unique canonical order");
    previousFragment = fragmentOrdinal;
    hasPrevious = true;
    const FrozenSpatialHandshakeFragment fragment = fragments[fragmentOrdinal];
    if (fragment.owner >= models.size())
      return candidateError("active handshake fragment owner is out of range");
    const ::loom::fabric::HandshakeOwnerModel &model = models[fragment.owner];
    if (fragment.localFragment >= model.fragmentCount())
      return candidateError("active local handshake fragment is out of range");
    const ::loom::fabric::HandshakeActivationFragment local =
        model.fragment(fragment.localFragment);
    if (local.contributionCount != fragment.contributionCount ||
        local.contributionOffset > model.fragmentContributionCount() ||
        local.contributionCount >
            model.fragmentContributionCount() - local.contributionOffset)
      return candidateError("active handshake fragment contribution is stale");
    for (std::uint32_t index = 0; index < local.contributionCount; ++index) {
      const std::uint32_t localArc =
          model.fragmentContributionOrdinal(local.contributionOffset + index);
      if (localArc >= model.arcCount())
        return candidateError("active handshake arc is out of range");
      const ::loom::fabric::HandshakeOwnerArc arc = model.arc(localArc);
      auto source = resolveNode(fragment.owner, model, arc.source);
      if (!source)
        return source.takeError();
      auto destination = resolveNode(fragment.owner, model, arc.destination);
      if (!destination)
        return destination.takeError();
      arcFragments[{*source, *destination}].push_back(fragmentOrdinal);
      addWork(graph.deterministicWork);
    }
  }

  graph.arcs.reserve(arcFragments.size());
  graph.arcIdentities.reserve(arcFragments.size());
  graph.arcContributors.reserve(arcFragments.size());
  for (auto &[arc, contributors] : arcFragments) {
    llvm::sort(contributors);
    contributors.erase(std::unique(contributors.begin(), contributors.end()),
                       contributors.end());
    auto ordinal = checkedIndex(graph.arcs.size(), "handshake arc");
    if (!ordinal)
      return ordinal.takeError();
    graph.arcs.push_back({arc.first, arc.second});
    graph.arcIdentities.push_back(
        {graph.nodeIdentities[arc.first], graph.nodeIdentities[arc.second]});
    graph.arcContributors.push_back(std::move(contributors));
    graph.arcOrdinals.emplace(arc, *ordinal);
    graph.outgoingArcs[arc.first].push_back(*ordinal);
    graph.reverseArcs[arc.second].push_back(*ordinal);
    addWork(graph.deterministicWork);
  }

  std::vector<PnrIndex> indegree(graph.nodeSignals.size(), 0);
  for (const FrozenSpatialHandshakeArc arc : graph.arcs) {
    if (llvm::Error error =
            increment(indegree[arc.destination], "handshake node indegree"))
      return std::move(error);
    addWork(graph.deterministicWork);
  }
  std::vector<PnrIndex> ready;
  ready.reserve(graph.nodeSignals.size());
  for (PnrIndex node = 0; node < graph.nodeSignals.size(); ++node)
    if (indegree[node] == 0)
      ready.push_back(node);
  graph.order.reserve(graph.nodeSignals.size());
  std::size_t cursor = 0;
  while (cursor < ready.size()) {
    const PnrIndex node = ready[cursor++];
    graph.order.push_back(node);
    for (PnrIndex arc : graph.outgoingArcs[node]) {
      PnrIndex &destinationIndegree = indegree[graph.arcs[arc].destination];
      if (destinationIndegree == 0)
        return candidateError("handshake indegree underflows");
      if (--destinationIndegree == 0)
        ready.push_back(graph.arcs[arc].destination);
      addWork(graph.deterministicWork);
    }
  }
  if (graph.order.size() != graph.nodeSignals.size()) {
    buildCycleWitness(graph);
    if (graph.cycleWitness.empty())
      return candidateError("cyclic handshake graph has no cycle witness");
    graph.constructionNanoseconds = elapsedNanoseconds(begin);
    return graph;
  }
  graph.ranks.resize(graph.nodeSignals.size());
  for (auto [rank, node] : llvm::enumerate(graph.order))
    graph.ranks[node] = static_cast<PnrIndex>(rank);
  graph.constructionNanoseconds = elapsedNanoseconds(begin);
  return graph;
}

std::optional<PnrIndex> findArc(const detail::MaterializedHandshakeGraph &graph,
                                const std::string &sourceKey,
                                const std::string &destinationKey) {
  const auto source = graph.nodeOrdinals.find(sourceKey);
  const auto destination = graph.nodeOrdinals.find(destinationKey);
  if (source == graph.nodeOrdinals.end() ||
      destination == graph.nodeOrdinals.end())
    return std::nullopt;
  const auto found =
      graph.arcOrdinals.find({source->second, destination->second});
  return found == graph.arcOrdinals.end()
             ? std::nullopt
             : std::optional<PnrIndex>(found->second);
}

bool containsContributor(const detail::MaterializedHandshakeGraph &graph,
                         PnrIndex arc, PnrIndex fragment) {
  return llvm::binary_search(graph.arcContributors[arc], fragment);
}

struct HandshakeDeltaClosureStatistics final {
  bool acyclic = true;
  std::uint64_t insertedArcCount = 0;
  std::uint64_t removedArcCount = 0;
  std::uint64_t affectedNodeCount = 0;
  std::uint64_t affectedRankSpan = 0;
  std::uint64_t deterministicWork = 0;
};

HandshakeDeltaClosureStatistics summarizeRebuiltHandshakeGraphDelta(
    const detail::MaterializedHandshakeGraph &before,
    const detail::MaterializedHandshakeGraph &after) {
  HandshakeDeltaClosureStatistics result;
  result.acyclic = after.cycleWitness.empty();
  std::set<std::string> affectedNodes;
  const auto observeMissingArc = [&](
                                     const detail::MaterializedHandshakeGraph
                                         &sourceGraph,
                                     const detail::MaterializedHandshakeGraph
                                         &targetGraph,
                                     PnrIndex arc,
                                     std::uint64_t &missingCount) {
    const detail::HandshakeArcIdentity &identity =
        sourceGraph.arcIdentities[arc];
    const std::string sourceKey = nodeKey(identity.source);
    const std::string destinationKey = nodeKey(identity.destination);
    const std::optional<PnrIndex> targetArc =
        findArc(targetGraph, sourceKey, destinationKey);
    addWork(result.deterministicWork);
    if (targetArc && !targetGraph.arcContributors[*targetArc].empty())
      return;
    addWork(missingCount);
    affectedNodes.insert(sourceKey);
    affectedNodes.insert(destinationKey);
  };
  for (PnrIndex arc = 0; arc < after.arcs.size(); ++arc)
    if (!after.arcContributors[arc].empty())
      observeMissingArc(after, before, arc, result.insertedArcCount);
  for (PnrIndex arc = 0; arc < before.arcs.size(); ++arc)
    if (!before.arcContributors[arc].empty())
      observeMissingArc(before, after, arc, result.removedArcCount);

  result.affectedNodeCount = affectedNodes.size();
  PnrIndex minimumRank = getInvalidPnrIndex();
  PnrIndex maximumRank = 0;
  const auto observeRank = [&](const detail::MaterializedHandshakeGraph &graph,
                               const std::string &key) {
    const auto node = graph.nodeOrdinals.find(key);
    if (node == graph.nodeOrdinals.end() || node->second >= graph.ranks.size())
      return;
    minimumRank = std::min(minimumRank, graph.ranks[node->second]);
    maximumRank = std::max(maximumRank, graph.ranks[node->second]);
  };
  for (const std::string &node : affectedNodes) {
    observeRank(before, node);
    observeRank(after, node);
    addWork(result.deterministicWork);
  }
  if (minimumRank != getInvalidPnrIndex())
    result.affectedRankSpan =
        static_cast<std::uint64_t>(maximumRank) - minimumRank + 1;
  return result;
}

llvm::Error appendChangedFragmentContributions(
    const FrozenSpatialHandshakeIndex &index, PnrIndex fragmentOrdinal,
    bool add, detail::HandshakeCandidateScratchStorage &storage,
    std::uint64_t &work) {
  if (fragmentOrdinal >= index.fragments().size())
    return candidateError("changed handshake fragment is out of range");
  const FrozenSpatialHandshakeFragment fragment =
      index.fragments()[fragmentOrdinal];
  if (fragment.owner >= index.ownerModels().size())
    return candidateError("changed handshake fragment owner is out of range");
  const ::loom::fabric::HandshakeOwnerModel &model =
      index.ownerModels()[fragment.owner];
  if (fragment.localFragment >= model.fragmentCount())
    return candidateError("changed local handshake fragment is out of range");
  const ::loom::fabric::HandshakeActivationFragment local =
      model.fragment(fragment.localFragment);
  if (local.contributionCount != fragment.contributionCount ||
      local.contributionOffset > model.fragmentContributionCount() ||
      local.contributionCount >
          model.fragmentContributionCount() - local.contributionOffset)
    return candidateError("changed handshake fragment contribution is stale");

  for (std::uint32_t index = 0; index < local.contributionCount; ++index) {
    const std::uint32_t localArc =
        model.fragmentContributionOrdinal(local.contributionOffset + index);
    if (localArc >= model.arcCount())
      return candidateError("changed handshake arc is out of range");
    auto identity = arcIdentity(fragment.owner, model, model.arc(localArc));
    if (!identity)
      return identity.takeError();
    std::string sourceKey = nodeKey(identity->source);
    std::string destinationKey = nodeKey(identity->destination);
    addWork(work);
    storage.changedContributions.push_back(
        {std::move(*identity), std::move(sourceKey), std::move(destinationKey),
         fragmentOrdinal, add});
  }
  return llvm::Error::success();
}

llvm::Expected<HandshakeDeltaClosureStatistics>
closeHandshakeArcDelta(const detail::MaterializedHandshakeGraph &graph,
                       detail::HandshakeCandidateScratchStorage &storage) {
  HandshakeDeltaClosureStatistics result;
  storage.arcChanges.clear();
  storage.removedArcOrdinals.clear();
  storage.insertedArcChanges.clear();
  storage.newNodeOrdinals.clear();
  storage.newNodes.clear();
  storage.insertedArcOrdinals.clear();
  storage.reachabilityWorklist.clear();

  llvm::sort(storage.changedContributions,
             [](const detail::ChangedArcContribution &lhs,
                const detail::ChangedArcContribution &rhs) {
               return std::tie(lhs.sourceKey, lhs.destinationKey, lhs.fragment,
                               lhs.add) < std::tie(rhs.sourceKey,
                                                   rhs.destinationKey,
                                                   rhs.fragment, rhs.add);
             });
  storage.changedContributions.erase(
      std::unique(storage.changedContributions.begin(),
                  storage.changedContributions.end(),
                  [](const detail::ChangedArcContribution &lhs,
                     const detail::ChangedArcContribution &rhs) {
                    return lhs.sourceKey == rhs.sourceKey &&
                           lhs.destinationKey == rhs.destinationKey &&
                           lhs.fragment == rhs.fragment && lhs.add == rhs.add;
                  }),
      storage.changedContributions.end());
  for (std::size_t offset = 0; offset != storage.changedContributions.size();) {
    const detail::ChangedArcContribution &first =
        storage.changedContributions[offset];
    std::size_t end = offset;
    PnrIndex additionCount = 0;
    PnrIndex removalCount = 0;
    while (end != storage.changedContributions.size() &&
           storage.changedContributions[end].sourceKey == first.sourceKey &&
           storage.changedContributions[end].destinationKey ==
               first.destinationKey) {
      PnrIndex &count =
          storage.changedContributions[end].add ? additionCount : removalCount;
      if (llvm::Error error = increment(count, "changed arc contribution"))
        return std::move(error);
      ++end;
      addWork(result.deterministicWork);
    }
    storage.arcChanges.push_back({first.arc, first.sourceKey,
                                  first.destinationKey, offset, end - offset,
                                  additionCount, removalCount});
    offset = end;
  }

  for (auto changeRecord : llvm::enumerate(storage.arcChanges)) {
    const std::size_t changeOrdinal = changeRecord.index();
    const detail::HandshakeArcChange &change = changeRecord.value();
    const std::optional<PnrIndex> currentArc =
        findArc(graph, change.sourceKey, change.destinationKey);
    const PnrIndex currentCount =
        currentArc
            ? static_cast<PnrIndex>(graph.arcContributors[*currentArc].size())
            : 0;
    for (const detail::ChangedArcContribution &contribution :
         llvm::ArrayRef(storage.changedContributions)
             .slice(change.contributionOffset, change.contributionCount)) {
      const bool present =
          currentArc &&
          containsContributor(graph, *currentArc, contribution.fragment);
      if (present == contribution.add)
        return candidateError(contribution.add
                                  ? "activated fragment already contributes "
                                    "to its handshake arc"
                                  : "deactivated fragment does not contribute "
                                    "to its handshake arc");
      addWork(result.deterministicWork);
    }
    if (change.removalCount > currentCount)
      return candidateError("changed handshake arc refcount underflows");
    const std::uint64_t retained = currentCount - change.removalCount;
    const std::uint64_t proposed = retained + change.additionCount;
    if (proposed >= static_cast<std::uint64_t>(getInvalidPnrIndex()))
      return candidateError("changed handshake arc refcount exceeds PnrIndex");
    if (currentCount != 0 && proposed == 0) {
      storage.removedArcOrdinals.push_back(*currentArc);
    } else if (currentCount == 0 && proposed != 0) {
      auto ordinal = checkedIndex(changeOrdinal, "inserted handshake arc");
      if (!ordinal)
        return ordinal.takeError();
      storage.insertedArcChanges.push_back(*ordinal);
    }
  }
  llvm::sort(storage.removedArcOrdinals);
  result.insertedArcCount = storage.insertedArcChanges.size();
  result.removedArcCount = storage.removedArcOrdinals.size();

  const auto resolveNode =
      [&](const detail::HandshakeNodeIdentity &identity,
          const std::string &key) -> llvm::Expected<PnrIndex> {
    const auto current = graph.nodeOrdinals.find(key);
    if (current != graph.nodeOrdinals.end())
      return current->second;
    const auto pending = storage.newNodeOrdinals.find(key);
    if (pending != storage.newNodeOrdinals.end())
      return pending->second;
    auto ordinal =
        checkedIndex(graph.nodeIdentities.size() + storage.newNodes.size(),
                     "prospective handshake node");
    if (!ordinal)
      return ordinal.takeError();
    storage.newNodeOrdinals.emplace(key, *ordinal);
    storage.newNodes.push_back(identity);
    return *ordinal;
  };
  for (PnrIndex changeOrdinal : storage.insertedArcChanges) {
    const detail::HandshakeArcChange &change =
        storage.arcChanges[changeOrdinal];
    auto source = resolveNode(change.arc.source, change.sourceKey);
    if (!source)
      return source.takeError();
    auto destination =
        resolveNode(change.arc.destination, change.destinationKey);
    if (!destination)
      return destination.takeError();
    storage.insertedArcOrdinals.push_back({*source, *destination});
  }
  llvm::sort(storage.insertedArcOrdinals,
             [](const FrozenSpatialHandshakeArc &lhs,
                const FrozenSpatialHandshakeArc &rhs) {
               return std::tie(lhs.source, lhs.destination) <
                      std::tie(rhs.source, rhs.destination);
             });

  const std::size_t prospectiveNodeCount =
      graph.nodeIdentities.size() + storage.newNodes.size();
  if (storage.reachabilityMarks.size() < prospectiveNodeCount)
    storage.reachabilityMarks.resize(prospectiveNodeCount, 0);
  if (storage.backwardMarks.size() < prospectiveNodeCount)
    storage.backwardMarks.resize(prospectiveNodeCount, 0);
  storage.reachabilityWorklist.reserve(prospectiveNodeCount);
  storage.backwardWorklist.reserve(prospectiveNodeCount);
  storage.reorderedNodes.reserve(prospectiveNodeCount);
  const auto removed = [&](PnrIndex arc) {
    return llvm::binary_search(storage.removedArcOrdinals, arc);
  };
  const auto rank = [&](PnrIndex node) {
    if (node < graph.ranks.size())
      return graph.ranks[node];
    return node;
  };

  for (const FrozenSpatialHandshakeArc inserted : storage.insertedArcOrdinals) {
    if (++storage.reachabilityEpoch == 0) {
      std::fill(storage.reachabilityMarks.begin(),
                storage.reachabilityMarks.end(), 0);
      storage.reachabilityEpoch = 1;
    }
    const std::uint64_t epoch = storage.reachabilityEpoch;
    storage.reachabilityWorklist.clear();
    storage.reachabilityMarks[inserted.destination] = epoch;
    storage.reachabilityWorklist.push_back(inserted.destination);
    PnrIndex minimumRank =
        std::min(rank(inserted.source), rank(inserted.destination));
    PnrIndex maximumRank =
        std::max(rank(inserted.source), rank(inserted.destination));
    std::size_t cursor = 0;
    while (cursor < storage.reachabilityWorklist.size()) {
      const PnrIndex node = storage.reachabilityWorklist[cursor++];
      addWork(result.affectedNodeCount);
      minimumRank = std::min(minimumRank, rank(node));
      maximumRank = std::max(maximumRank, rank(node));
      if (node == inserted.source) {
        result.acyclic = false;
        addWork(result.affectedRankSpan,
                static_cast<std::uint64_t>(maximumRank) - minimumRank + 1);
        return result;
      }
      const auto visit = [&](PnrIndex destination) {
        addWork(result.deterministicWork);
        if (storage.reachabilityMarks[destination] == epoch)
          return;
        storage.reachabilityMarks[destination] = epoch;
        storage.reachabilityWorklist.push_back(destination);
      };
      if (node < graph.nodeIdentities.size()) {
        if (graph.outgoingArcs.size() != graph.nodeIdentities.size())
          return candidateError("active handshake adjacency is stale");
        for (PnrIndex arc : graph.outgoingArcs[node])
          if (!graph.arcContributors[arc].empty() && !removed(arc))
            visit(graph.arcs[arc].destination);
      }
      const auto first = llvm::lower_bound(
          storage.insertedArcOrdinals, node,
          [](const FrozenSpatialHandshakeArc &candidate, PnrIndex source) {
            return candidate.source < source;
          });
      for (auto candidate = first;
           candidate != storage.insertedArcOrdinals.end() &&
           candidate->source == node;
           ++candidate)
        visit(candidate->destination);
    }
    addWork(result.affectedRankSpan,
            static_cast<std::uint64_t>(maximumRank) - minimumRank + 1);
  }
  return result;
}

bool arcIsActive(const detail::MaterializedHandshakeGraph &graph,
                 PnrIndex arc) {
  return !graph.arcContributors[arc].empty();
}

llvm::Expected<PnrIndex>
ensureHandshakeNode(detail::MaterializedHandshakeGraph &graph,
                    const detail::HandshakeNodeIdentity &identity,
                    const std::string &key) {
  const auto found = graph.nodeOrdinals.find(key);
  if (found != graph.nodeOrdinals.end())
    return found->second;
  auto ordinal = checkedIndex(graph.nodeIdentities.size(), "handshake node");
  if (!ordinal)
    return ordinal.takeError();
  graph.nodeOrdinals.emplace(key, *ordinal);
  graph.nodeSignals.push_back(identity.boundarySignal);
  graph.nodeIdentities.push_back(identity);
  graph.outgoingArcs.emplace_back();
  graph.reverseArcs.emplace_back();
  graph.order.push_back(*ordinal);
  graph.ranks.push_back(*ordinal);
  return *ordinal;
}

llvm::Expected<PnrIndex>
ensureHandshakeNode(detail::MaterializedHandshakeGraph &graph,
                    const detail::HandshakeNodeIdentity &identity) {
  return ensureHandshakeNode(graph, identity, nodeKey(identity));
}

llvm::Expected<PnrIndex>
ensureHandshakeArc(detail::MaterializedHandshakeGraph &graph,
                   const detail::HandshakeArcIdentity &identity) {
  auto source = ensureHandshakeNode(graph, identity.source);
  if (!source)
    return source.takeError();
  auto destination = ensureHandshakeNode(graph, identity.destination);
  if (!destination)
    return destination.takeError();
  const std::pair<PnrIndex, PnrIndex> key{*source, *destination};
  const auto found = graph.arcOrdinals.find(key);
  if (found != graph.arcOrdinals.end())
    return found->second;
  auto ordinal = checkedIndex(graph.arcs.size(), "handshake arc");
  if (!ordinal)
    return ordinal.takeError();
  graph.arcOrdinals.emplace(key, *ordinal);
  graph.arcs.push_back({*source, *destination});
  graph.arcIdentities.push_back(identity);
  graph.arcContributors.emplace_back();
  graph.outgoingArcs[*source].push_back(*ordinal);
  graph.reverseArcs[*destination].push_back(*ordinal);
  return *ordinal;
}

llvm::Error reorderForInsertedHandshakeArc(
    detail::MaterializedHandshakeGraph &graph, PnrIndex insertedArc,
    detail::HandshakeCandidateScratchStorage &storage, std::uint64_t &work) {
  if (insertedArc >= graph.arcs.size() || !arcIsActive(graph, insertedArc))
    return candidateError("inserted handshake arc is not active");
  const FrozenSpatialHandshakeArc edge = graph.arcs[insertedArc];
  if (graph.ranks[edge.source] < graph.ranks[edge.destination])
    return llvm::Error::success();

  const PnrIndex lower = graph.ranks[edge.destination];
  const PnrIndex upper = graph.ranks[edge.source];
  if (storage.reachabilityMarks.size() < graph.nodeIdentities.size())
    storage.reachabilityMarks.resize(graph.nodeIdentities.size(), 0);
  if (storage.backwardMarks.size() < graph.nodeIdentities.size())
    storage.backwardMarks.resize(graph.nodeIdentities.size(), 0);
  if (++storage.reachabilityEpoch == 0) {
    std::fill(storage.reachabilityMarks.begin(),
              storage.reachabilityMarks.end(), 0);
    storage.reachabilityEpoch = 1;
  }
  if (++storage.backwardEpoch == 0) {
    std::fill(storage.backwardMarks.begin(), storage.backwardMarks.end(), 0);
    storage.backwardEpoch = 1;
  }
  const std::uint64_t forwardEpoch = storage.reachabilityEpoch;
  const std::uint64_t backwardEpoch = storage.backwardEpoch;

  storage.reachabilityWorklist.clear();
  storage.reachabilityWorklist.push_back(edge.destination);
  storage.reachabilityMarks[edge.destination] = forwardEpoch;
  for (std::size_t cursor = 0; cursor != storage.reachabilityWorklist.size();
       ++cursor) {
    const PnrIndex node = storage.reachabilityWorklist[cursor];
    for (PnrIndex arc : graph.outgoingArcs[node]) {
      if (!arcIsActive(graph, arc))
        continue;
      const PnrIndex destination = graph.arcs[arc].destination;
      addWork(work);
      if (graph.ranks[destination] > upper ||
          storage.reachabilityMarks[destination] == forwardEpoch)
        continue;
      storage.reachabilityMarks[destination] = forwardEpoch;
      storage.reachabilityWorklist.push_back(destination);
    }
  }
  if (storage.reachabilityMarks[edge.source] == forwardEpoch)
    return candidateError("inserted handshake arc creates a cycle");

  storage.backwardWorklist.clear();
  storage.backwardWorklist.push_back(edge.source);
  storage.backwardMarks[edge.source] = backwardEpoch;
  for (std::size_t cursor = 0; cursor != storage.backwardWorklist.size();
       ++cursor) {
    const PnrIndex node = storage.backwardWorklist[cursor];
    for (PnrIndex arc : graph.reverseArcs[node]) {
      if (!arcIsActive(graph, arc))
        continue;
      const PnrIndex source = graph.arcs[arc].source;
      addWork(work);
      if (graph.ranks[source] < lower ||
          storage.backwardMarks[source] == backwardEpoch)
        continue;
      storage.backwardMarks[source] = backwardEpoch;
      storage.backwardWorklist.push_back(source);
    }
  }

  storage.reorderedNodes.clear();
  storage.reorderedNodes.reserve(static_cast<std::size_t>(upper) - lower + 1);
  const auto appendClass = [&](unsigned kind) {
    for (PnrIndex rank = lower; rank <= upper; ++rank) {
      const PnrIndex node = graph.order[rank];
      const bool forward = storage.reachabilityMarks[node] == forwardEpoch;
      const bool backward = storage.backwardMarks[node] == backwardEpoch;
      if (forward && backward)
        continue;
      if ((kind == 0 && backward) || (kind == 1 && !forward && !backward) ||
          (kind == 2 && forward))
        storage.reorderedNodes.push_back(node);
    }
  };
  appendClass(0);
  appendClass(1);
  appendClass(2);
  if (storage.reorderedNodes.size() !=
      static_cast<std::size_t>(upper) - lower + 1)
    return candidateError("handshake topology reorder found a cycle");
  for (auto [offset, node] : llvm::enumerate(storage.reorderedNodes)) {
    const PnrIndex rank = lower + static_cast<PnrIndex>(offset);
    graph.order[rank] = node;
    graph.ranks[node] = rank;
    addWork(work);
  }
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
applyHandshakeArcDelta(detail::MaterializedHandshakeGraph &graph,
                       detail::HandshakeCandidateScratchStorage &storage) {
  std::uint64_t work = 0;
  if (storage.newNodes.size() > static_cast<std::size_t>(getInvalidPnrIndex()) -
                                    graph.nodeIdentities.size() ||
      storage.insertedArcChanges.size() >
          static_cast<std::size_t>(getInvalidPnrIndex()) - graph.arcs.size())
    return candidateError("handshake delta exceeds PnrIndex");
  for (const detail::HandshakeNodeIdentity &identity : storage.newNodes) {
    auto node = ensureHandshakeNode(graph, identity);
    if (!node)
      return node.takeError();
  }
  for (const detail::ChangedArcContribution &change :
       storage.changedContributions) {
    auto source =
        ensureHandshakeNode(graph, change.arc.source, change.sourceKey);
    if (!source)
      return source.takeError();
    auto destination = ensureHandshakeNode(graph, change.arc.destination,
                                           change.destinationKey);
    if (!destination)
      return destination.takeError();
    const auto foundArc = graph.arcOrdinals.find({*source, *destination});
    llvm::Expected<PnrIndex> arc =
        foundArc == graph.arcOrdinals.end()
            ? ensureHandshakeArc(graph, change.arc)
            : llvm::Expected<PnrIndex>(foundArc->second);
    if (!arc)
      return arc.takeError();
    std::vector<PnrIndex> &contributors = graph.arcContributors[*arc];
    auto found = llvm::lower_bound(contributors, change.fragment);
    if (change.add) {
      if (found != contributors.end() && *found == change.fragment)
        return candidateError("handshake contribution is already active");
      contributors.insert(found, change.fragment);
    } else {
      if (found == contributors.end() || *found != change.fragment)
        return candidateError("handshake contribution is not active");
      contributors.erase(found);
    }
    addWork(work);
  }
  for (PnrIndex changeOrdinal : storage.insertedArcChanges) {
    const detail::HandshakeArcChange &change =
        storage.arcChanges[changeOrdinal];
    const std::optional<PnrIndex> arc =
        findArc(graph, change.sourceKey, change.destinationKey);
    if (!arc)
      return candidateError("inserted handshake arc was not materialized");
    if (llvm::Error error =
            reorderForInsertedHandshakeArc(graph, *arc, storage, work))
      return std::move(error);
  }
  return work;
}

llvm::Error setFragmentActive(std::vector<PnrIndex> &activeFragments,
                              PnrIndex fragment, bool active) {
  auto found = llvm::lower_bound(activeFragments, fragment);
  const bool wasActive = found != activeFragments.end() && *found == fragment;
  if (wasActive == active)
    return candidateError("active fragment index is inconsistent");
  if (active)
    activeFragments.insert(found, fragment);
  else
    activeFragments.erase(found);
  return llvm::Error::success();
}

void restoreFragmentActive(std::vector<PnrIndex> &activeFragments,
                           PnrIndex fragment, bool active) noexcept {
  auto found = llvm::lower_bound(activeFragments, fragment);
  const bool wasActive = found != activeFragments.end() && *found == fragment;
  if (wasActive == active)
    return;
  if (active)
    activeFragments.insert(found, fragment);
  else
    activeFragments.erase(found);
}

} // namespace

llvm::Expected<bool> loom::pnr::independentlyVerifyHandshakeProjectionAcyclic(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> selectedFragments,
    llvm::ArrayRef<PnrIndex> traversalUses) {
  auto selection =
      rebuildHandshakeSelection(index, selectedFragments, traversalUses);
  if (!selection)
    return selection.takeError();
  auto graph = materializeHandshakeGraph(index, selection->activeFragments);
  if (!graph)
    return graph.takeError();
  return graph->cycleWitness.empty();
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
  const std::size_t traversalCount =
      index.traversalFragmentOffsets().size() - 1;
  fragmentJournalMarks_.assign(index.fragments().size(), 0);
  traversalJournalMarks_.assign(traversalCount, 0);
  groupJournalMarks_.assign(index.allTraversalGroups().size(), 0);
  transactionEpoch_ = 0;
  resetTransaction();
  return llvm::Error::success();
}

std::size_t HandshakeCandidateScratch::retainedStorageBytes() const {
  return retainedBytes(storage_->changedContributions) +
         retainedBytes(storage_->arcChanges) +
         retainedBytes(storage_->removedArcOrdinals) +
         retainedBytes(storage_->insertedArcChanges) +
         retainedMapBytes(storage_->newNodeOrdinals) +
         retainedBytes(storage_->newNodes) +
         retainedBytes(storage_->insertedArcOrdinals) +
         retainedBytes(storage_->reachabilityMarks) +
         retainedBytes(storage_->reachabilityWorklist) +
         retainedBytes(storage_->backwardMarks) +
         retainedBytes(storage_->backwardWorklist) +
         retainedBytes(storage_->reorderedNodes) +
         retainedBytes(fragmentJournalMarks_) +
         retainedBytes(traversalJournalMarks_) +
         retainedBytes(groupJournalMarks_) + retainedBytes(fragmentDeltas_) +
         retainedBytes(traversalDeltas_) + retainedBytes(groupDeltas_);
}

void HandshakeCandidateScratch::beginTransaction() {
  resetTransaction();
  if (++transactionEpoch_ == 0) {
    std::fill(fragmentJournalMarks_.begin(), fragmentJournalMarks_.end(), 0);
    std::fill(traversalJournalMarks_.begin(), traversalJournalMarks_.end(), 0);
    std::fill(groupJournalMarks_.begin(), groupJournalMarks_.end(), 0);
    transactionEpoch_ = 1;
  }
}

void HandshakeCandidateScratch::resetTransaction() {
  storage_->changedContributions.clear();
  storage_->arcChanges.clear();
  storage_->removedArcOrdinals.clear();
  storage_->insertedArcChanges.clear();
  storage_->newNodeOrdinals.clear();
  storage_->newNodes.clear();
  storage_->insertedArcOrdinals.clear();
  storage_->reachabilityWorklist.clear();
  storage_->backwardWorklist.clear();
  storage_->reorderedNodes.clear();
  fragmentDeltas_.clear();
  traversalDeltas_.clear();
  groupDeltas_.clear();
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
  auto selection =
      rebuildHandshakeSelection(*index, selectedFragments, traversalUses);
  if (!selection)
    return selection.takeError();
  auto graph = materializeHandshakeGraph(*index, selection->activeFragments);
  if (!graph)
    return graph.takeError();
  if (!graph->cycleWitness.empty())
    return candidateError("initial handshake selection is cyclic");
  auto graphOwner =
      std::make_shared<detail::MaterializedHandshakeGraph>(std::move(*graph));
  auto state = HandshakeCandidateStateHandle(new HandshakeCandidateState(
      std::move(index), std::move(graphOwner),
      std::move(selection->fragmentRefcounts),
      std::move(selection->activeFragments),
      std::move(selection->traversalRefcounts),
      std::move(selection->allGroupSelectedWitnessCounts)));
  state->materializationConstructionCount_ = 1;
  state->materializationConstructionNanoseconds_ =
      state->graph_->constructionNanoseconds;
  state->materializationDeterministicWork_ = state->graph_->deterministicWork;
  return state;
}

PnrIndex HandshakeCandidateState::fragmentRefcount(PnrIndex fragment) const {
  assert(fragment < fragmentRefcounts_.size());
  return fragmentRefcounts_[fragment];
}

PnrIndex HandshakeCandidateState::traversalRefcount(PnrIndex traversal) const {
  assert(traversal < traversalRefcounts_.size());
  return traversalRefcounts_[traversal];
}

bool HandshakeCandidateState::isTraversalSelected(PnrIndex traversal) const {
  return traversalRefcount(traversal) != 0;
}

llvm::ArrayRef<std::optional<::loom::fabric::HandshakeSignalRef>>
HandshakeCandidateState::activeNodeSignals() const {
  if (activeTransaction_ && activeTransaction_->pendingGraph_)
    return activeTransaction_->pendingGraph_->nodeSignals;
  return graph_->nodeSignals;
}

llvm::ArrayRef<FrozenSpatialHandshakeArc>
HandshakeCandidateState::activeArcs() const {
  if (activeTransaction_ && activeTransaction_->pendingGraph_)
    return activeTransaction_->pendingGraph_->arcs;
  return graph_->arcs;
}

llvm::ArrayRef<PnrIndex>
HandshakeCandidateState::activeArcContributors(PnrIndex arc) const {
  const detail::MaterializedHandshakeGraph &visible =
      activeTransaction_ && activeTransaction_->pendingGraph_
          ? *activeTransaction_->pendingGraph_
          : *graph_;
  assert(arc < visible.arcContributors.size());
  return visible.arcContributors[arc];
}

std::size_t HandshakeCandidateState::activeArcContributionCount() const {
  const detail::MaterializedHandshakeGraph &visible =
      activeTransaction_ && activeTransaction_->pendingGraph_
          ? *activeTransaction_->pendingGraph_
          : *graph_;
  std::size_t count = 0;
  for (const auto &contributors : visible.arcContributors)
    count += contributors.size();
  return count;
}

HandshakeActiveDemandStatistics
HandshakeCandidateState::materializationStatistics() const {
  HandshakeActiveDemandStatistics result;
  result.constructionCount = materializationConstructionCount_;
  result.constructionNanoseconds = materializationConstructionNanoseconds_;
  result.deterministicWork = materializationDeterministicWork_;
  result.activeFragmentCount = activeFragments_.size();
  result.materializedNodeCount = graph_->nodeSignals.size();
  result.materializedArcCount =
      llvm::count_if(graph_->arcContributors, [](const auto &contributors) {
        return !contributors.empty();
      });
  result.materializedContributionCount = activeArcContributionCount();
  result.transactionClosureCount = transactionClosureCount_;
  result.transactionInsertedArcCount = transactionInsertedArcCount_;
  result.transactionRemovedArcCount = transactionRemovedArcCount_;
  result.transactionAffectedNodeCount = transactionAffectedNodeCount_;
  result.transactionAffectedRankSpan = transactionAffectedRankSpan_;
  const auto addBytes = [&](std::size_t bytes) {
    addWork(result.retainedBytes, static_cast<std::uint64_t>(bytes));
  };
  addBytes(retainedBytes(fragmentRefcounts_));
  addBytes(retainedBytes(activeFragments_));
  addBytes(retainedBytes(traversalRefcounts_));
  addBytes(retainedBytes(allGroupSelectedWitnessCounts_));
  addBytes(retainedBytes(graph_->nodeSignals));
  addBytes(retainedBytes(graph_->nodeIdentities));
  addBytes(retainedBytes(graph_->arcs));
  addBytes(retainedBytes(graph_->arcIdentities));
  addBytes(retainedNestedBytes(graph_->arcContributors));
  addBytes(retainedNestedBytes(graph_->outgoingArcs));
  addBytes(retainedNestedBytes(graph_->reverseArcs));
  addBytes(retainedMapBytes(graph_->nodeOrdinals));
  addBytes(retainedMapBytes(graph_->arcOrdinals));
  addBytes(retainedBytes(graph_->order));
  addBytes(retainedBytes(graph_->ranks));
  addBytes(retainedBytes(graph_->cycleWitness));
  return result;
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateState::topologicalOrder() const {
  if (activeTransaction_ && activeTransaction_->pendingGraph_)
    return activeTransaction_->pendingGraph_->order;
  return graph_->order;
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateState::topologicalRanks() const {
  if (activeTransaction_ && activeTransaction_->pendingGraph_)
    return activeTransaction_->pendingGraph_->ranks;
  return graph_->ranks;
}

llvm::Error HandshakeCandidateState::verify() const {
  if (!index_ || !graph_ ||
      fragmentRefcounts_.size() != index_->fragments().size() ||
      traversalRefcounts_.size() + 1 !=
          index_->traversalFragmentOffsets().size() ||
      allGroupSelectedWitnessCounts_.size() !=
          index_->allTraversalGroups().size())
    return candidateError("candidate shape does not match its frozen index");
  std::vector<PnrIndex> expectedActive;
  expectedActive.reserve(activeFragments_.size());
  for (auto [fragment, refcount] : llvm::enumerate(fragmentRefcounts_))
    if (refcount != 0)
      expectedActive.push_back(static_cast<PnrIndex>(fragment));
  if (expectedActive != activeFragments_)
    return candidateError("active fragment index is stale");

  if (graph_->nodeSignals.size() != graph_->nodeIdentities.size() ||
      graph_->nodeSignals.size() != graph_->outgoingArcs.size() ||
      graph_->nodeSignals.size() != graph_->reverseArcs.size() ||
      graph_->nodeSignals.size() != graph_->order.size() ||
      graph_->nodeSignals.size() != graph_->ranks.size() ||
      graph_->arcs.size() != graph_->arcIdentities.size() ||
      graph_->arcs.size() != graph_->arcContributors.size())
    return candidateError("materialized handshake graph shape is stale");
  std::vector<std::uint8_t> observedNodes(graph_->order.size(), 0);
  for (auto [rank, node] : llvm::enumerate(graph_->order)) {
    if (node >= graph_->ranks.size() || observedNodes[node] != 0 ||
        graph_->ranks[node] != rank)
      return candidateError("materialized handshake topology is not a rank");
    observedNodes[node] = 1;
  }
  for (auto [arc, record] : llvm::enumerate(graph_->arcs)) {
    if (record.source >= graph_->nodeIdentities.size() ||
        record.destination >= graph_->nodeIdentities.size())
      return candidateError("materialized handshake arc is out of range");
    const auto &contributors = graph_->arcContributors[arc];
    if (!llvm::is_sorted(contributors) ||
        std::adjacent_find(contributors.begin(), contributors.end()) !=
            contributors.end())
      return candidateError(
          "materialized handshake contributors are not canonical");
    if (!contributors.empty() &&
        graph_->ranks[record.source] >= graph_->ranks[record.destination])
      return candidateError("materialized handshake topology is cyclic");
  }

  auto rebuilt = materializeHandshakeGraph(*index_, activeFragments_);
  if (!rebuilt)
    return rebuilt.takeError();
  if (!rebuilt->cycleWitness.empty())
    return candidateError("committed handshake graph is cyclic");
  const auto activeArcSnapshot = [](const detail::MaterializedHandshakeGraph
                                        &g) {
    std::map<std::pair<std::string, std::string>, std::vector<PnrIndex>> result;
    for (auto [arc, identity] : llvm::enumerate(g.arcIdentities))
      if (!g.arcContributors[arc].empty())
        result.emplace(std::make_pair(nodeKey(identity.source),
                                      nodeKey(identity.destination)),
                       g.arcContributors[arc]);
    return result;
  };
  if (activeArcSnapshot(*rebuilt) != activeArcSnapshot(*graph_))
    return candidateError("materialized handshake graph is stale");

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
      scratch.traversalJournalMarks_.size() !=
          index_->traversalFragmentOffsets().size() - 1 ||
      scratch.groupJournalMarks_.size() !=
          allGroupSelectedWitnessCounts_.size())
    return candidateError("scratch was not prepared for this candidate");
  scratch.beginTransaction();
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
      closed_(other.closed_), cycle_(other.cycle_),
      pendingGraph_(std::move(other.pendingGraph_)) {
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
  return setFragmentActive(state_->activeFragments_, fragment, isActive);
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
  detail::HandshakeCandidateScratchStorage &storage = *scratch_->storage_;
  storage.changedContributions.clear();
  std::uint64_t projectionWork = 0;
  for (const HandshakeCandidateScratch::IndexDelta &delta :
       scratch_->fragmentDeltas_) {
    const bool wasActive = delta.oldValue != 0;
    const bool isActive = state_->fragmentRefcounts_[delta.index] != 0;
    if (wasActive == isActive)
      continue;
    if (llvm::Error error = appendChangedFragmentContributions(
            *state_->index_, delta.index, isActive, storage, projectionWork))
      return error;
  }
  if (storage.changedContributions.empty()) {
    closed_ = true;
    return true;
  }
  const std::size_t changedContributionCount =
      storage.changedContributions.size();
  const std::size_t activeContributionCount =
      state_->activeArcContributionCount();
  const std::size_t nodeCount = state_->graph_->nodeIdentities.size();
  const std::size_t arcCount = state_->graph_->arcs.size();
  const bool exceedsActiveDemand =
      changedContributionCount > activeContributionCount;
  const bool coversGraphInventory =
      changedContributionCount >= nodeCount &&
      changedContributionCount - nodeCount >= arcCount;
  if (exceedsActiveDemand || coversGraphInventory) {
    auto graph =
        materializeHandshakeGraph(*state_->index_, state_->activeFragments_);
    if (!graph)
      return graph.takeError();
    const HandshakeDeltaClosureStatistics closure =
        summarizeRebuiltHandshakeGraphDelta(*state_->graph_, *graph);
    addWork(state_->transactionClosureCount_);
    addWork(state_->transactionInsertedArcCount_, closure.insertedArcCount);
    addWork(state_->transactionRemovedArcCount_, closure.removedArcCount);
    addWork(state_->transactionAffectedNodeCount_, closure.affectedNodeCount);
    addWork(state_->transactionAffectedRankSpan_, closure.affectedRankSpan);
    addWork(state_->materializationConstructionCount_);
    addWork(state_->materializationConstructionNanoseconds_,
            graph->constructionNanoseconds);
    addWork(state_->materializationDeterministicWork_, projectionWork);
    addWork(state_->materializationDeterministicWork_,
            closure.deterministicWork);
    addWork(state_->materializationDeterministicWork_,
            graph->deterministicWork);
    cycle_ = !closure.acyclic;
    pendingGraph_ = std::make_shared<detail::MaterializedHandshakeGraph>(
        std::move(*graph));
    closed_ = true;
    return !cycle_;
  }
  auto closure = closeHandshakeArcDelta(*state_->graph_, storage);
  if (!closure)
    return closure.takeError();
  addWork(state_->transactionClosureCount_);
  addWork(state_->transactionInsertedArcCount_, closure->insertedArcCount);
  addWork(state_->transactionRemovedArcCount_, closure->removedArcCount);
  addWork(state_->transactionAffectedNodeCount_, closure->affectedNodeCount);
  addWork(state_->transactionAffectedRankSpan_, closure->affectedRankSpan);
  addWork(state_->materializationDeterministicWork_, projectionWork);
  addWork(state_->materializationDeterministicWork_,
          closure->deterministicWork);
  cycle_ = !closure->acyclic;
  if (cycle_) {
    auto graph =
        materializeHandshakeGraph(*state_->index_, state_->activeFragments_);
    if (!graph)
      return graph.takeError();
    if (graph->cycleWitness.empty())
      return candidateError(
          "delta closure reported a cycle absent from exact materialization");
    addWork(state_->materializationConstructionCount_);
    addWork(state_->materializationConstructionNanoseconds_,
            graph->constructionNanoseconds);
    addWork(state_->materializationDeterministicWork_,
            graph->deterministicWork);
    pendingGraph_ = std::make_shared<detail::MaterializedHandshakeGraph>(
        std::move(*graph));
  }
  closed_ = true;
  return !cycle_;
}

llvm::ArrayRef<PnrIndex> HandshakeCandidateTransaction::cycleWitness() const {
  return pendingGraph_ ? llvm::ArrayRef(pendingGraph_->cycleWitness)
                       : llvm::ArrayRef<PnrIndex>();
}

llvm::Error HandshakeCandidateTransaction::commit() {
  if (!scratch_)
    return candidateError("transaction is no longer active");
  auto closure = close();
  if (!closure)
    return closure.takeError();
  if (!*closure)
    return candidateError("cannot commit a handshake cycle");
  if (pendingGraph_) {
    state_->graph_ = std::move(pendingGraph_);
    finish();
    return llvm::Error::success();
  }
  if (scratch_->storage_->changedContributions.empty()) {
    pendingGraph_.reset();
    finish();
    return llvm::Error::success();
  }
  auto work = applyHandshakeArcDelta(*state_->graph_, *scratch_->storage_);
  if (!work)
    return work.takeError();
  addWork(state_->materializationDeterministicWork_, *work);
  pendingGraph_.reset();
  finish();
  return llvm::Error::success();
}

void HandshakeCandidateTransaction::rollback() noexcept {
  if (!scratch_)
    return;
  for (const auto &delta : scratch_->fragmentDeltas_) {
    state_->fragmentRefcounts_[delta.index] = delta.oldValue;
    restoreFragmentActive(state_->activeFragments_, delta.index,
                          delta.oldValue != 0);
  }
  for (const auto &delta : scratch_->traversalDeltas_)
    state_->traversalRefcounts_[delta.index] = delta.oldValue;
  for (const auto &delta : scratch_->groupDeltas_)
    state_->allGroupSelectedWitnessCounts_[delta.index] = delta.oldValue;
  pendingGraph_.reset();
  finish();
}

void HandshakeCandidateTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  state_.reset();
}
