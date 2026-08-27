#include "PnR/HandshakeCandidateState.h"

#include "Common/MappingDebugLog.h"
#include "HandshakeProjectionInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::pnr;
using loom::pnr::detail::rebuildHandshakeSelection;

namespace loom::pnr::detail {

/// The active handshake subgraph over compact node and arc numbering. Every
/// node and arc also names its ordinal in the frozen dense projection, which
/// is the single identity owner; no byte key is derived after freeze.
struct MaterializedHandshakeGraph final {
  std::vector<std::optional<::loom::fabric::HandshakeSignalRef>> nodeSignals;
  /// Compact node ordinal to frozen dense projection node ordinal.
  std::vector<PnrIndex> nodeFrozenIds;
  std::vector<FrozenSpatialHandshakeArc> arcs;
  /// Compact arc ordinal to frozen dense projection arc ordinal.
  std::vector<PnrIndex> arcFrozenIds;
  std::vector<std::uint8_t> fixedArcs;
  std::vector<std::vector<PnrIndex>> arcContributors;
  std::vector<std::vector<PnrIndex>> outgoingArcs;
  std::vector<std::vector<PnrIndex>> reverseArcs;
  /// Frozen projection ordinal to compact ordinal. Unmaterialized entries use
  /// the invalid ordinal; compact ordinals retain first-encounter order.
  std::vector<PnrIndex> nodeOrdinals;
  std::vector<PnrIndex> arcOrdinals;
  std::vector<PnrIndex> order;
  std::vector<PnrIndex> ranks;
  std::vector<PnrIndex> cycleWitness;
  std::vector<PnrIndex> constructionIndegree;
  std::vector<PnrIndex> constructionReady;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
};

struct ChangedArcContribution final {
  /// Frozen dense projection arc ordinal.
  PnrIndex arc = 0;
  PnrIndex fragment = 0;
  bool add = false;
};

struct HandshakeArcChange final {
  /// Frozen dense projection arc ordinal.
  PnrIndex arc = 0;
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
  /// Frozen node ordinal to prospective compact ordinal for nodes the delta
  /// would introduce.
  llvm::DenseMap<PnrIndex, PnrIndex> newNodeOrdinals;
  /// Frozen node ordinals of prospective nodes in assignment order.
  std::vector<PnrIndex> newNodes;
  std::vector<FrozenSpatialHandshakeArc> insertedArcOrdinals;
  std::vector<std::uint64_t> reachabilityMarks;
  std::vector<PnrIndex> reachabilityWorklist;
  std::vector<std::uint64_t> backwardMarks;
  std::vector<PnrIndex> backwardWorklist;
  std::vector<PnrIndex> reorderedNodes;
  std::vector<PnrIndex> unaffectedReorderedNodes;
  std::vector<PnrIndex> forwardReorderedNodes;
  std::shared_ptr<MaterializedHandshakeGraph> reusableGraph;
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
std::size_t retainedDenseMapBytes(const llvm::DenseMap<Key, Value> &values) {
  return values.getMemorySize();
}

std::size_t retainedMaterializedHandshakeGraphBytes(
    const detail::MaterializedHandshakeGraph &graph) {
  return retainedBytes(graph.nodeSignals) + retainedBytes(graph.nodeFrozenIds) +
         retainedBytes(graph.arcs) + retainedBytes(graph.arcFrozenIds) +
         retainedBytes(graph.fixedArcs) +
         retainedNestedBytes(graph.arcContributors) +
         retainedNestedBytes(graph.outgoingArcs) +
         retainedNestedBytes(graph.reverseArcs) +
         retainedBytes(graph.nodeOrdinals) + retainedBytes(graph.arcOrdinals) +
         retainedBytes(graph.order) + retainedBytes(graph.ranks) +
         retainedBytes(graph.cycleWitness) +
         retainedBytes(graph.constructionIndegree) +
         retainedBytes(graph.constructionReady);
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

void resetMaterializedHandshakeGraph(detail::MaterializedHandshakeGraph &graph,
                                     std::size_t frozenNodeCount,
                                     std::size_t frozenArcCount) {
  if (graph.nodeOrdinals.size() == frozenNodeCount) {
    for (PnrIndex frozenNode : graph.nodeFrozenIds)
      graph.nodeOrdinals[frozenNode] = getInvalidPnrIndex();
  } else {
    graph.nodeOrdinals.assign(frozenNodeCount, getInvalidPnrIndex());
  }
  if (graph.arcOrdinals.size() == frozenArcCount) {
    for (PnrIndex frozenArc : graph.arcFrozenIds)
      graph.arcOrdinals[frozenArc] = getInvalidPnrIndex();
  } else {
    graph.arcOrdinals.assign(frozenArcCount, getInvalidPnrIndex());
  }
  graph.nodeSignals.clear();
  graph.nodeFrozenIds.clear();
  graph.arcs.clear();
  graph.arcFrozenIds.clear();
  graph.fixedArcs.clear();
  for (std::vector<PnrIndex> &contributors : graph.arcContributors)
    contributors.clear();
  for (std::vector<PnrIndex> &arcs : graph.outgoingArcs)
    arcs.clear();
  for (std::vector<PnrIndex> &arcs : graph.reverseArcs)
    arcs.clear();
  graph.order.clear();
  graph.ranks.clear();
  graph.cycleWitness.clear();
  graph.constructionIndegree.clear();
  graph.constructionReady.clear();
  graph.constructionNanoseconds = 0;
  graph.deterministicWork = 0;
}

llvm::Error
materializeHandshakeGraphInto(const FrozenSpatialHandshakeIndex &index,
                              llvm::ArrayRef<PnrIndex> activeFragments,
                              detail::MaterializedHandshakeGraph &graph) {
  const auto begin = std::chrono::steady_clock::now();
  const auto projectionArcs = index.projectionArcs();
  const auto nodeSignals = index.projectionNodeSignals();
  if (nodeSignals.size() !=
      static_cast<std::size_t>(index.projectionNodeCount()))
    return candidateError("frozen handshake node signals are incomplete");
  resetMaterializedHandshakeGraph(graph, nodeSignals.size(),
                                  projectionArcs.size());

  const auto resolveNode =
      [&](PnrIndex frozenNode) -> llvm::Expected<PnrIndex> {
    if (frozenNode >= nodeSignals.size())
      return candidateError("frozen handshake node is out of range");
    if (graph.nodeOrdinals[frozenNode] != getInvalidPnrIndex())
      return graph.nodeOrdinals[frozenNode];
    auto ordinal = checkedIndex(graph.nodeFrozenIds.size(), "handshake node");
    if (!ordinal)
      return ordinal.takeError();
    graph.nodeOrdinals[frozenNode] = *ordinal;
    graph.nodeFrozenIds.push_back(frozenNode);
    graph.nodeSignals.push_back(nodeSignals[frozenNode]);
    if (*ordinal == graph.outgoingArcs.size()) {
      graph.outgoingArcs.emplace_back();
      graph.reverseArcs.emplace_back();
    }
    addWork(graph.deterministicWork);
    return *ordinal;
  };
  const auto resolveArc = [&](PnrIndex frozenArc) -> llvm::Expected<PnrIndex> {
    if (frozenArc >= projectionArcs.size())
      return candidateError("frozen handshake arc is out of range");
    if (graph.arcOrdinals[frozenArc] != getInvalidPnrIndex())
      return graph.arcOrdinals[frozenArc];
    auto source = resolveNode(projectionArcs[frozenArc].source);
    if (!source)
      return source.takeError();
    auto destination = resolveNode(projectionArcs[frozenArc].destination);
    if (!destination)
      return destination.takeError();
    auto ordinal = checkedIndex(graph.arcs.size(), "handshake arc");
    if (!ordinal)
      return ordinal.takeError();
    graph.arcOrdinals[frozenArc] = *ordinal;
    graph.arcs.push_back({*source, *destination});
    graph.arcFrozenIds.push_back(frozenArc);
    graph.fixedArcs.push_back(0);
    if (*ordinal == graph.arcContributors.size())
      graph.arcContributors.emplace_back();
    graph.outgoingArcs[*source].push_back(*ordinal);
    graph.reverseArcs[*destination].push_back(*ordinal);
    addWork(graph.deterministicWork);
    return *ordinal;
  };

  if (!index.fabricContext())
    return candidateError("handshake index has no Fabric static context");
  for (PnrIndex frozenArc : index.projectionFixedArcs()) {
    auto arc = resolveArc(frozenArc);
    if (!arc)
      return arc.takeError();
    graph.fixedArcs[*arc] = 1;
    addWork(graph.deterministicWork);
  }

  const auto fragmentOffsets = index.projectionFragmentArcOffsets();
  const auto fragmentArcs = index.projectionFragmentArcs();
  if (fragmentOffsets.size() != index.fragments().size() + 1)
    return candidateError("frozen handshake fragment arc index is incomplete");
  PnrIndex previousFragment = 0;
  bool hasPrevious = false;
  for (PnrIndex fragmentOrdinal : activeFragments) {
    if (fragmentOrdinal >= index.fragments().size())
      return candidateError("active handshake fragment is out of range");
    if (hasPrevious && fragmentOrdinal <= previousFragment)
      return candidateError(
          "active handshake fragments are not unique canonical order");
    previousFragment = fragmentOrdinal;
    hasPrevious = true;
    // Ascending fragments append each contributor once per arc, so every
    // contributor list stays sorted and unique without a per-arc sort.
    for (PnrIndex frozenArc :
         fragmentArcs.slice(fragmentOffsets[fragmentOrdinal],
                            fragmentOffsets[fragmentOrdinal + 1] -
                                fragmentOffsets[fragmentOrdinal])) {
      auto arc = resolveArc(frozenArc);
      if (!arc)
        return arc.takeError();
      graph.arcContributors[*arc].push_back(fragmentOrdinal);
      addWork(graph.deterministicWork);
    }
  }

  graph.arcContributors.resize(graph.arcs.size());
  graph.outgoingArcs.resize(graph.nodeSignals.size());
  graph.reverseArcs.resize(graph.nodeSignals.size());
  graph.constructionIndegree.assign(graph.nodeSignals.size(), 0);
  for (const FrozenSpatialHandshakeArc arc : graph.arcs) {
    if (llvm::Error error =
            increment(graph.constructionIndegree[arc.destination],
                      "handshake node indegree"))
      return error;
    addWork(graph.deterministicWork);
  }
  graph.constructionReady.clear();
  graph.constructionReady.reserve(graph.nodeSignals.size());
  for (PnrIndex node = 0; node < graph.nodeSignals.size(); ++node)
    if (graph.constructionIndegree[node] == 0)
      graph.constructionReady.push_back(node);
  graph.order.reserve(graph.nodeSignals.size());
  std::size_t cursor = 0;
  while (cursor < graph.constructionReady.size()) {
    const PnrIndex node = graph.constructionReady[cursor++];
    graph.order.push_back(node);
    for (PnrIndex arc : graph.outgoingArcs[node]) {
      PnrIndex &destinationIndegree =
          graph.constructionIndegree[graph.arcs[arc].destination];
      if (destinationIndegree == 0)
        return candidateError("handshake indegree underflows");
      if (--destinationIndegree == 0)
        graph.constructionReady.push_back(graph.arcs[arc].destination);
      addWork(graph.deterministicWork);
    }
  }
  if (graph.order.size() != graph.nodeSignals.size()) {
    buildCycleWitness(graph);
    if (graph.cycleWitness.empty())
      return candidateError("cyclic handshake graph has no cycle witness");
    graph.constructionNanoseconds = elapsedNanoseconds(begin);
    return llvm::Error::success();
  }
  graph.ranks.resize(graph.nodeSignals.size());
  for (auto [rank, node] : llvm::enumerate(graph.order))
    graph.ranks[node] = static_cast<PnrIndex>(rank);
  graph.constructionNanoseconds = elapsedNanoseconds(begin);
  return llvm::Error::success();
}

llvm::Expected<detail::MaterializedHandshakeGraph>
materializeHandshakeGraph(const FrozenSpatialHandshakeIndex &index,
                          llvm::ArrayRef<PnrIndex> activeFragments) {
  detail::MaterializedHandshakeGraph graph;
  if (llvm::Error error =
          materializeHandshakeGraphInto(index, activeFragments, graph))
    return std::move(error);
  return graph;
}

llvm::Expected<std::shared_ptr<detail::MaterializedHandshakeGraph>>
materializeReusableHandshakeGraph(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> activeFragments,
    std::shared_ptr<detail::MaterializedHandshakeGraph> &reusableGraph) {
  std::shared_ptr<detail::MaterializedHandshakeGraph> graph =
      std::move(reusableGraph);
  if (!graph || graph.use_count() != 1)
    graph = std::make_shared<detail::MaterializedHandshakeGraph>();
  if (llvm::Error error =
          materializeHandshakeGraphInto(index, activeFragments, *graph)) {
    reusableGraph = std::move(graph);
    return std::move(error);
  }
  return graph;
}

void recycleMaterializedHandshakeGraph(
    std::shared_ptr<detail::MaterializedHandshakeGraph> graph,
    detail::HandshakeCandidateScratchStorage &storage) {
  if (!storage.reusableGraph && graph && graph.use_count() == 1)
    storage.reusableGraph = std::move(graph);
}

std::optional<PnrIndex> findArc(const detail::MaterializedHandshakeGraph &graph,
                                PnrIndex frozenArc) {
  if (frozenArc >= graph.arcOrdinals.size() ||
      graph.arcOrdinals[frozenArc] == getInvalidPnrIndex())
    return std::nullopt;
  return graph.arcOrdinals[frozenArc];
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

bool arcIsActive(const detail::MaterializedHandshakeGraph &graph, PnrIndex arc);

HandshakeDeltaClosureStatistics summarizeRebuiltHandshakeGraphDelta(
    const detail::MaterializedHandshakeGraph &before,
    const detail::MaterializedHandshakeGraph &after) {
  HandshakeDeltaClosureStatistics result;
  result.acyclic = after.cycleWitness.empty();
  llvm::SmallDenseSet<PnrIndex, 16> affectedNodes;
  const auto observeMissingArc =
      [&](const detail::MaterializedHandshakeGraph &sourceGraph,
          const detail::MaterializedHandshakeGraph &targetGraph, PnrIndex arc,
          std::uint64_t &missingCount) {
        const std::optional<PnrIndex> targetArc =
            findArc(targetGraph, sourceGraph.arcFrozenIds[arc]);
        addWork(result.deterministicWork);
        if (targetArc && (targetGraph.fixedArcs[*targetArc] ||
                          !targetGraph.arcContributors[*targetArc].empty()))
          return;
        addWork(missingCount);
        affectedNodes.insert(
            sourceGraph.nodeFrozenIds[sourceGraph.arcs[arc].source]);
        affectedNodes.insert(
            sourceGraph.nodeFrozenIds[sourceGraph.arcs[arc].destination]);
      };
  for (PnrIndex arc = 0; arc < after.arcs.size(); ++arc)
    if (after.fixedArcs[arc] || !after.arcContributors[arc].empty())
      observeMissingArc(after, before, arc, result.insertedArcCount);
  for (PnrIndex arc = 0; arc < before.arcs.size(); ++arc)
    if (before.fixedArcs[arc] || !before.arcContributors[arc].empty())
      observeMissingArc(before, after, arc, result.removedArcCount);

  result.affectedNodeCount = affectedNodes.size();
  PnrIndex minimumRank = getInvalidPnrIndex();
  PnrIndex maximumRank = 0;
  const auto observeRank = [&](const detail::MaterializedHandshakeGraph &graph,
                               PnrIndex frozenNode) {
    if (frozenNode >= graph.nodeOrdinals.size())
      return;
    const PnrIndex node = graph.nodeOrdinals[frozenNode];
    if (node == getInvalidPnrIndex() || node >= graph.ranks.size())
      return;
    minimumRank = std::min(minimumRank, graph.ranks[node]);
    maximumRank = std::max(maximumRank, graph.ranks[node]);
  };
  for (PnrIndex node : affectedNodes) {
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
  const auto fragmentOffsets = index.projectionFragmentArcOffsets();
  const auto fragmentArcs = index.projectionFragmentArcs();
  if (fragmentOffsets.size() != index.fragments().size() + 1)
    return candidateError("frozen handshake fragment arc index is incomplete");
  for (PnrIndex frozenArc :
       fragmentArcs.slice(fragmentOffsets[fragmentOrdinal],
                          fragmentOffsets[fragmentOrdinal + 1] -
                              fragmentOffsets[fragmentOrdinal])) {
    if (frozenArc >= index.projectionArcs().size())
      return candidateError("changed handshake arc is out of range");
    addWork(work);
    storage.changedContributions.push_back({frozenArc, fragmentOrdinal, add});
  }
  return llvm::Error::success();
}

llvm::Expected<HandshakeDeltaClosureStatistics>
closeHandshakeArcDelta(const FrozenSpatialHandshakeIndex &graphIndex,
                       const detail::MaterializedHandshakeGraph &graph,
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
               return std::tie(lhs.arc, lhs.fragment, lhs.add) <
                      std::tie(rhs.arc, rhs.fragment, rhs.add);
             });
  storage.changedContributions.erase(
      std::unique(storage.changedContributions.begin(),
                  storage.changedContributions.end(),
                  [](const detail::ChangedArcContribution &lhs,
                     const detail::ChangedArcContribution &rhs) {
                    return lhs.arc == rhs.arc && lhs.fragment == rhs.fragment &&
                           lhs.add == rhs.add;
                  }),
      storage.changedContributions.end());
  for (std::size_t offset = 0; offset != storage.changedContributions.size();) {
    const detail::ChangedArcContribution &first =
        storage.changedContributions[offset];
    std::size_t end = offset;
    PnrIndex additionCount = 0;
    PnrIndex removalCount = 0;
    while (end != storage.changedContributions.size() &&
           storage.changedContributions[end].arc == first.arc) {
      PnrIndex &count =
          storage.changedContributions[end].add ? additionCount : removalCount;
      if (llvm::Error error = increment(count, "changed arc contribution"))
        return std::move(error);
      ++end;
      addWork(result.deterministicWork);
    }
    storage.arcChanges.push_back(
        {first.arc, offset, end - offset, additionCount, removalCount});
    offset = end;
  }

  for (auto changeRecord : llvm::enumerate(storage.arcChanges)) {
    const std::size_t changeOrdinal = changeRecord.index();
    const detail::HandshakeArcChange &change = changeRecord.value();
    const std::optional<PnrIndex> currentArc = findArc(graph, change.arc);
    const PnrIndex currentCount =
        currentArc
            ? static_cast<PnrIndex>(graph.arcContributors[*currentArc].size())
            : 0;
    const bool fixed = currentArc && graph.fixedArcs[*currentArc];
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
    const bool currentlyActive = fixed || currentCount != 0;
    const bool proposedActive = fixed || proposed != 0;
    if (currentlyActive && !proposedActive) {
      storage.removedArcOrdinals.push_back(*currentArc);
    } else if (!currentlyActive && proposedActive) {
      auto ordinal = checkedIndex(changeOrdinal, "inserted handshake arc");
      if (!ordinal)
        return ordinal.takeError();
      storage.insertedArcChanges.push_back(*ordinal);
    }
  }
  llvm::sort(storage.removedArcOrdinals);
  result.insertedArcCount = storage.insertedArcChanges.size();
  result.removedArcCount = storage.removedArcOrdinals.size();

  const auto projectionArcs = graphIndex.projectionArcs();
  const auto resolveNode =
      [&](PnrIndex frozenNode) -> llvm::Expected<PnrIndex> {
    if (frozenNode >= graph.nodeOrdinals.size())
      return candidateError("changed handshake node is out of range");
    if (graph.nodeOrdinals[frozenNode] != getInvalidPnrIndex())
      return graph.nodeOrdinals[frozenNode];
    const auto pending = storage.newNodeOrdinals.find(frozenNode);
    if (pending != storage.newNodeOrdinals.end())
      return pending->second;
    auto ordinal =
        checkedIndex(graph.nodeFrozenIds.size() + storage.newNodes.size(),
                     "prospective handshake node");
    if (!ordinal)
      return ordinal.takeError();
    storage.newNodeOrdinals.insert({frozenNode, *ordinal});
    storage.newNodes.push_back(frozenNode);
    return *ordinal;
  };
  for (PnrIndex changeOrdinal : storage.insertedArcChanges) {
    const detail::HandshakeArcChange &change =
        storage.arcChanges[changeOrdinal];
    if (change.arc >= projectionArcs.size())
      return candidateError("changed handshake arc is out of range");
    auto source = resolveNode(projectionArcs[change.arc].source);
    if (!source)
      return source.takeError();
    auto destination = resolveNode(projectionArcs[change.arc].destination);
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
      graph.nodeFrozenIds.size() + storage.newNodes.size();
  if (storage.reachabilityMarks.size() < prospectiveNodeCount)
    storage.reachabilityMarks.resize(prospectiveNodeCount, 0);
  if (storage.backwardMarks.size() < prospectiveNodeCount)
    storage.backwardMarks.resize(prospectiveNodeCount, 0);
  storage.reachabilityWorklist.reserve(prospectiveNodeCount);
  storage.backwardWorklist.reserve(prospectiveNodeCount);
  storage.reorderedNodes.reserve(prospectiveNodeCount);
  storage.unaffectedReorderedNodes.reserve(prospectiveNodeCount);
  storage.forwardReorderedNodes.reserve(prospectiveNodeCount);
  const auto removed = [&](PnrIndex arc) {
    return llvm::binary_search(storage.removedArcOrdinals, arc);
  };
  const auto rank = [&](PnrIndex node) {
    if (node < graph.ranks.size())
      return graph.ranks[node];
    return node;
  };

  for (const FrozenSpatialHandshakeArc inserted : storage.insertedArcOrdinals) {
    if (rank(inserted.source) < rank(inserted.destination))
      continue;
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
      if (node < graph.nodeFrozenIds.size()) {
        if (graph.outgoingArcs.size() != graph.nodeFrozenIds.size())
          return candidateError("active handshake adjacency is stale");
        for (PnrIndex arc : graph.outgoingArcs[node])
          if (arcIsActive(graph, arc) && !removed(arc))
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
  return graph.fixedArcs[arc] || !graph.arcContributors[arc].empty();
}

llvm::Expected<PnrIndex>
ensureHandshakeNode(detail::MaterializedHandshakeGraph &graph,
                    const FrozenSpatialHandshakeIndex &index,
                    PnrIndex frozenNode) {
  const auto nodeSignals = index.projectionNodeSignals();
  if (frozenNode >= nodeSignals.size() ||
      graph.nodeOrdinals.size() != nodeSignals.size())
    return candidateError("frozen handshake node is out of range");
  if (graph.nodeOrdinals[frozenNode] != getInvalidPnrIndex())
    return graph.nodeOrdinals[frozenNode];
  auto ordinal = checkedIndex(graph.nodeFrozenIds.size(), "handshake node");
  if (!ordinal)
    return ordinal.takeError();
  graph.nodeOrdinals[frozenNode] = *ordinal;
  graph.nodeFrozenIds.push_back(frozenNode);
  graph.nodeSignals.push_back(nodeSignals[frozenNode]);
  graph.outgoingArcs.emplace_back();
  graph.reverseArcs.emplace_back();
  graph.order.push_back(*ordinal);
  graph.ranks.push_back(*ordinal);
  return *ordinal;
}

llvm::Expected<PnrIndex>
ensureHandshakeArc(detail::MaterializedHandshakeGraph &graph,
                   const FrozenSpatialHandshakeIndex &index,
                   PnrIndex frozenArc) {
  const auto projectionArcs = index.projectionArcs();
  if (frozenArc >= projectionArcs.size() ||
      graph.arcOrdinals.size() != projectionArcs.size())
    return candidateError("frozen handshake arc is out of range");
  if (graph.arcOrdinals[frozenArc] != getInvalidPnrIndex())
    return graph.arcOrdinals[frozenArc];
  auto source =
      ensureHandshakeNode(graph, index, projectionArcs[frozenArc].source);
  if (!source)
    return source.takeError();
  auto destination =
      ensureHandshakeNode(graph, index, projectionArcs[frozenArc].destination);
  if (!destination)
    return destination.takeError();
  auto ordinal = checkedIndex(graph.arcs.size(), "handshake arc");
  if (!ordinal)
    return ordinal.takeError();
  graph.arcOrdinals[frozenArc] = *ordinal;
  graph.arcs.push_back({*source, *destination});
  graph.arcFrozenIds.push_back(frozenArc);
  graph.fixedArcs.push_back(false);
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
  if (storage.reachabilityMarks.size() < graph.nodeFrozenIds.size())
    storage.reachabilityMarks.resize(graph.nodeFrozenIds.size(), 0);
  if (storage.backwardMarks.size() < graph.nodeFrozenIds.size())
    storage.backwardMarks.resize(graph.nodeFrozenIds.size(), 0);
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
  storage.unaffectedReorderedNodes.clear();
  storage.forwardReorderedNodes.clear();
  const std::size_t rankSpan = static_cast<std::size_t>(upper) - lower + 1;
  storage.reorderedNodes.reserve(rankSpan);
  storage.unaffectedReorderedNodes.reserve(rankSpan);
  storage.forwardReorderedNodes.reserve(rankSpan);
  for (PnrIndex rank = lower; rank <= upper; ++rank) {
    const PnrIndex node = graph.order[rank];
    const bool forward = storage.reachabilityMarks[node] == forwardEpoch;
    const bool backward = storage.backwardMarks[node] == backwardEpoch;
    if (forward && backward)
      continue;
    if (backward)
      storage.reorderedNodes.push_back(node);
    else if (forward)
      storage.forwardReorderedNodes.push_back(node);
    else
      storage.unaffectedReorderedNodes.push_back(node);
  }
  storage.reorderedNodes.insert(storage.reorderedNodes.end(),
                                storage.unaffectedReorderedNodes.begin(),
                                storage.unaffectedReorderedNodes.end());
  storage.reorderedNodes.insert(storage.reorderedNodes.end(),
                                storage.forwardReorderedNodes.begin(),
                                storage.forwardReorderedNodes.end());
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
applyHandshakeArcDelta(const FrozenSpatialHandshakeIndex &index,
                       detail::MaterializedHandshakeGraph &graph,
                       detail::HandshakeCandidateScratchStorage &storage) {
  std::uint64_t work = 0;
  if (storage.newNodes.size() > static_cast<std::size_t>(getInvalidPnrIndex()) -
                                    graph.nodeFrozenIds.size() ||
      storage.insertedArcChanges.size() >
          static_cast<std::size_t>(getInvalidPnrIndex()) - graph.arcs.size())
    return candidateError("handshake delta exceeds PnrIndex");
  for (PnrIndex frozenNode : storage.newNodes) {
    auto node = ensureHandshakeNode(graph, index, frozenNode);
    if (!node)
      return node.takeError();
  }
  for (const detail::ChangedArcContribution &change :
       storage.changedContributions) {
    auto arc = ensureHandshakeArc(graph, index, change.arc);
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
    const std::optional<PnrIndex> arc = findArc(graph, change.arc);
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
         retainedDenseMapBytes(storage_->newNodeOrdinals) +
         retainedBytes(storage_->newNodes) +
         retainedBytes(storage_->insertedArcOrdinals) +
         retainedBytes(storage_->reachabilityMarks) +
         retainedBytes(storage_->reachabilityWorklist) +
         retainedBytes(storage_->backwardMarks) +
         retainedBytes(storage_->backwardWorklist) +
         retainedBytes(storage_->reorderedNodes) +
         retainedBytes(storage_->unaffectedReorderedNodes) +
         retainedBytes(storage_->forwardReorderedNodes) +
         (storage_->reusableGraph ? retainedMaterializedHandshakeGraphBytes(
                                        *storage_->reusableGraph)
                                  : 0) +
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
  storage_->unaffectedReorderedNodes.clear();
  storage_->forwardReorderedNodes.clear();
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
  for (PnrIndex arc = 0; arc < graph_->arcs.size(); ++arc) {
    result.materializedArcCount += arcIsActive(*graph_, arc);
    result.fabricUnconditionalArcCount += graph_->fixedArcs[arc];
  }
  result.materializedContributionCount = activeArcContributionCount();
  result.transactionClosureCount = transactionClosureCount_;
  result.transactionInsertedArcCount = transactionInsertedArcCount_;
  result.transactionRemovedArcCount = transactionRemovedArcCount_;
  result.transactionAffectedNodeCount = transactionAffectedNodeCount_;
  result.transactionAffectedRankSpan = transactionAffectedRankSpan_;
  result.cachedVerificationCount = cachedVerificationCount_;
  result.coldVerificationConstructionCount = coldVerificationConstructionCount_;
  result.coldVerificationConstructionNanoseconds =
      coldVerificationConstructionNanoseconds_;
  const auto addBytes = [&](std::size_t bytes) {
    addWork(result.retainedBytes, static_cast<std::uint64_t>(bytes));
  };
  addBytes(retainedBytes(fragmentRefcounts_));
  addBytes(retainedBytes(activeFragments_));
  addBytes(retainedBytes(traversalRefcounts_));
  addBytes(retainedBytes(allGroupSelectedWitnessCounts_));
  addBytes(retainedMaterializedHandshakeGraphBytes(*graph_));
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

llvm::Error HandshakeCandidateState::verifyCachedState() const {
  addWork(cachedVerificationCount_);
  if (activeTransaction_)
    return candidateError("cannot verify during a handshake transaction");
  if (!index_ || !graph_ ||
      fragmentRefcounts_.size() != index_->fragments().size() ||
      traversalRefcounts_.size() + 1 !=
          index_->traversalFragmentOffsets().size() ||
      allGroupSelectedWitnessCounts_.size() !=
          index_->allTraversalGroups().size())
    return candidateError("candidate shape does not match its frozen index");
  std::size_t activeOrdinal = 0;
  for (auto [fragment, refcount] : llvm::enumerate(fragmentRefcounts_)) {
    if (refcount == 0)
      continue;
    if (activeOrdinal >= activeFragments_.size() ||
        activeFragments_[activeOrdinal] != static_cast<PnrIndex>(fragment))
      return candidateError("active fragment index is stale");
    ++activeOrdinal;
  }
  if (activeOrdinal != activeFragments_.size())
    return candidateError("active fragment index is stale");

  const auto frozenNodeSignals = index_->projectionNodeSignals();
  const auto frozenArcs = index_->projectionArcs();
  if (graph_->nodeSignals.size() != graph_->nodeFrozenIds.size() ||
      graph_->nodeSignals.size() != graph_->outgoingArcs.size() ||
      graph_->nodeSignals.size() != graph_->reverseArcs.size() ||
      graph_->nodeSignals.size() != graph_->order.size() ||
      graph_->nodeSignals.size() != graph_->ranks.size() ||
      graph_->nodeOrdinals.size() != frozenNodeSignals.size() ||
      graph_->arcs.size() != graph_->arcFrozenIds.size() ||
      graph_->arcs.size() != graph_->fixedArcs.size() ||
      graph_->arcs.size() != graph_->arcContributors.size() ||
      graph_->arcOrdinals.size() != frozenArcs.size() ||
      !graph_->cycleWitness.empty())
    return candidateError("materialized handshake graph shape is stale");
  for (PnrIndex node = 0; node < graph_->nodeFrozenIds.size(); ++node) {
    const PnrIndex frozenNode = graph_->nodeFrozenIds[node];
    if (frozenNode >= frozenNodeSignals.size() ||
        graph_->nodeOrdinals[frozenNode] != node ||
        graph_->nodeSignals[node] != frozenNodeSignals[frozenNode])
      return candidateError("materialized handshake node signal is stale");
  }
  for (auto [rank, node] : llvm::enumerate(graph_->order)) {
    if (node >= graph_->ranks.size() || graph_->ranks[node] != rank)
      return candidateError("materialized handshake topology is not a rank");
  }
  for (auto [arc, record] : llvm::enumerate(graph_->arcs)) {
    const PnrIndex frozenArc = graph_->arcFrozenIds[arc];
    if (record.source >= graph_->nodeFrozenIds.size() ||
        record.destination >= graph_->nodeFrozenIds.size() ||
        frozenArc >= frozenArcs.size() ||
        graph_->arcOrdinals[frozenArc] != arc ||
        graph_->nodeFrozenIds[record.source] != frozenArcs[frozenArc].source ||
        graph_->nodeFrozenIds[record.destination] !=
            frozenArcs[frozenArc].destination ||
        graph_->fixedArcs[arc] > 1)
      return candidateError("materialized handshake arc is out of range");
    const auto &contributors = graph_->arcContributors[arc];
    if (!llvm::is_sorted(contributors) ||
        std::adjacent_find(contributors.begin(), contributors.end()) !=
            contributors.end())
      return candidateError("materialized handshake contributors are stale");
    if (arcIsActive(*graph_, static_cast<PnrIndex>(arc)) &&
        graph_->ranks[record.source] >= graph_->ranks[record.destination])
      return candidateError("materialized handshake topology is cyclic");
  }
  std::size_t outgoingArcCount = 0;
  std::size_t reverseArcCount = 0;
  for (PnrIndex node = 0; node < graph_->nodeFrozenIds.size(); ++node) {
    const auto verifyAdjacency = [&](llvm::ArrayRef<PnrIndex> adjacency,
                                     bool outgoing) -> llvm::Error {
      PnrIndex previous = getInvalidPnrIndex();
      for (PnrIndex arc : adjacency) {
        if (arc >= graph_->arcs.size() ||
            (previous != getInvalidPnrIndex() && arc <= previous) ||
            (outgoing ? graph_->arcs[arc].source
                      : graph_->arcs[arc].destination) != node)
          return candidateError("materialized handshake adjacency is stale");
        previous = arc;
      }
      return llvm::Error::success();
    };
    if (llvm::Error error = verifyAdjacency(graph_->outgoingArcs[node], true))
      return error;
    if (llvm::Error error = verifyAdjacency(graph_->reverseArcs[node], false))
      return error;
    if (graph_->outgoingArcs[node].size() >
            std::numeric_limits<std::size_t>::max() - outgoingArcCount ||
        graph_->reverseArcs[node].size() >
            std::numeric_limits<std::size_t>::max() - reverseArcCount)
      return candidateError("materialized handshake adjacency overflows");
    outgoingArcCount += graph_->outgoingArcs[node].size();
    reverseArcCount += graph_->reverseArcs[node].size();
  }
  if (outgoingArcCount != graph_->arcs.size() ||
      reverseArcCount != graph_->arcs.size())
    return candidateError("materialized handshake adjacency is incomplete");
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

llvm::Error HandshakeCandidateState::verify() const {
  if (llvm::Error error = verifyCachedState())
    return error;

  const auto begin = std::chrono::steady_clock::now();
  addWork(coldVerificationConstructionCount_);
  auto rebuilt = materializeHandshakeGraph(*index_, activeFragments_);
  addWork(coldVerificationConstructionNanoseconds_, elapsedNanoseconds(begin));
  if (!rebuilt)
    return rebuilt.takeError();
  if (!rebuilt->cycleWitness.empty())
    return candidateError("committed handshake graph is cyclic");
  std::size_t rebuiltActiveArcCount = 0;
  for (PnrIndex rebuiltArc = 0; rebuiltArc < rebuilt->arcs.size();
       ++rebuiltArc) {
    if (!arcIsActive(*rebuilt, rebuiltArc))
      continue;
    ++rebuiltActiveArcCount;
    const std::optional<PnrIndex> cachedArc =
        findArc(*graph_, rebuilt->arcFrozenIds[rebuiltArc]);
    if (!cachedArc || !arcIsActive(*graph_, *cachedArc) ||
        rebuilt->fixedArcs[rebuiltArc] != graph_->fixedArcs[*cachedArc] ||
        rebuilt->arcContributors[rebuiltArc] !=
            graph_->arcContributors[*cachedArc])
      return candidateError("materialized handshake graph is stale");
  }
  std::size_t cachedActiveArcCount = 0;
  for (PnrIndex arc = 0; arc < graph_->arcs.size(); ++arc)
    cachedActiveArcCount += arcIsActive(*graph_, arc);
  if (rebuiltActiveArcCount != cachedActiveArcCount)
    return candidateError("materialized handshake graph is stale");
  return llvm::Error::success();
}

llvm::Expected<HandshakeCandidateTransaction>
HandshakeCandidateState::beginTransaction(
    HandshakeCandidateScratch &scratch,
    HandshakeProjectionScratch *projectionScratch) & {
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
  return HandshakeCandidateTransaction(shared_from_this(), scratch,
                                       projectionScratch);
}

HandshakeCandidateTransaction::HandshakeCandidateTransaction(
    HandshakeCandidateStateHandle state, HandshakeCandidateScratch &scratch,
    HandshakeProjectionScratch *projectionScratch)
    : state_(std::move(state)), scratch_(&scratch),
      projectionScratch_(projectionScratch) {
  state_->activeTransaction_ = this;
  scratch_->activeTransaction_ = this;
}

HandshakeCandidateTransaction::HandshakeCandidateTransaction(
    HandshakeCandidateTransaction &&other) noexcept
    : state_(std::move(other.state_)), scratch_(other.scratch_),
      projectionScratch_(other.projectionScratch_), closed_(other.closed_),
      cycle_(other.cycle_), pendingGraph_(std::move(other.pendingGraph_)) {
  other.scratch_ = nullptr;
  other.projectionScratch_ = nullptr;
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
  const std::size_t changedContributionCount =
      storage.changedContributions.size();
  if (changedContributionCount == 0) {
    closed_ = true;
    return true;
  }
  const std::size_t nodeCount = state_->graph_->nodeFrozenIds.size();
  const std::size_t arcCount = state_->graph_->arcs.size();
  // A full rebuild includes the immutable Fabric handshake inventory, not
  // only the currently selected fragment contributions. Comparing a local
  // delta with active contributions alone makes sparse workloads on a large
  // Fabric rebuild that immutable inventory for almost every rejected probe.
  // Retain the rebuild path only when the delta itself covers the complete
  // materialized graph inventory.
  const bool coversGraphInventory =
      changedContributionCount >= nodeCount &&
      changedContributionCount - nodeCount >= arcCount;
  if (coversGraphInventory && projectionScratch_) {
    const HandshakeProjectionStatistics projectionBefore =
        projectionScratch_->statistics();
    auto acyclic = projectionScratch_->projectActiveFragmentsAcyclic(
        *state_->index_, state_->activeFragments_);
    if (!acyclic)
      return acyclic.takeError();
    const HandshakeProjectionStatistics projectionAfter =
        projectionScratch_->statistics();
    std::uint64_t denseProjectionWork = 0;
    if (projectionAfter.deterministicWork >= projectionBefore.deterministicWork)
      denseProjectionWork = projectionAfter.deterministicWork -
                            projectionBefore.deterministicWork;
    addWork(state_->transactionClosureCount_);
    addWork(state_->materializationDeterministicWork_, projectionWork);
    addWork(state_->materializationDeterministicWork_, denseProjectionWork);
    cycle_ = !*acyclic;
    if (cycle_) {
      if (!mapping_debug::enabled(mapping_debug::Level::Decision)) {
        closed_ = true;
        return false;
      }
      auto graph = materializeReusableHandshakeGraph(
          *state_->index_, state_->activeFragments_, storage.reusableGraph);
      if (!graph)
        return graph.takeError();
      if ((*graph)->cycleWitness.empty())
        return candidateError(
            "dense projection reported a cycle absent from exact "
            "materialization");
      addWork(state_->materializationConstructionCount_);
      addWork(state_->materializationConstructionNanoseconds_,
              (*graph)->constructionNanoseconds);
      addWork(state_->materializationDeterministicWork_,
              (*graph)->deterministicWork);
      pendingGraph_ = std::move(*graph);
      closed_ = true;
      return false;
    }

    auto graph = materializeReusableHandshakeGraph(
        *state_->index_, state_->activeFragments_, storage.reusableGraph);
    if (!graph)
      return graph.takeError();
    if (!(*graph)->cycleWitness.empty())
      return candidateError(
          "dense projection reported an acyclic selection with an exact "
          "cycle");
    const HandshakeDeltaClosureStatistics closure =
        summarizeRebuiltHandshakeGraphDelta(*state_->graph_, **graph);
    addWork(state_->transactionInsertedArcCount_, closure.insertedArcCount);
    addWork(state_->transactionRemovedArcCount_, closure.removedArcCount);
    addWork(state_->transactionAffectedNodeCount_, closure.affectedNodeCount);
    addWork(state_->transactionAffectedRankSpan_, closure.affectedRankSpan);
    addWork(state_->materializationConstructionCount_);
    addWork(state_->materializationConstructionNanoseconds_,
            (*graph)->constructionNanoseconds);
    addWork(state_->materializationDeterministicWork_,
            closure.deterministicWork);
    addWork(state_->materializationDeterministicWork_,
            (*graph)->deterministicWork);
    pendingGraph_ = std::move(*graph);
    closed_ = true;
    return true;
  }
  if (coversGraphInventory) {
    auto graph = materializeReusableHandshakeGraph(
        *state_->index_, state_->activeFragments_, storage.reusableGraph);
    if (!graph)
      return graph.takeError();
    const HandshakeDeltaClosureStatistics closure =
        summarizeRebuiltHandshakeGraphDelta(*state_->graph_, **graph);
    addWork(state_->transactionClosureCount_);
    addWork(state_->transactionInsertedArcCount_, closure.insertedArcCount);
    addWork(state_->transactionRemovedArcCount_, closure.removedArcCount);
    addWork(state_->transactionAffectedNodeCount_, closure.affectedNodeCount);
    addWork(state_->transactionAffectedRankSpan_, closure.affectedRankSpan);
    addWork(state_->materializationConstructionCount_);
    addWork(state_->materializationConstructionNanoseconds_,
            (*graph)->constructionNanoseconds);
    addWork(state_->materializationDeterministicWork_, projectionWork);
    addWork(state_->materializationDeterministicWork_,
            closure.deterministicWork);
    addWork(state_->materializationDeterministicWork_,
            (*graph)->deterministicWork);
    cycle_ = !closure.acyclic;
    pendingGraph_ = std::move(*graph);
    closed_ = true;
    return !cycle_;
  }
  auto closure =
      closeHandshakeArcDelta(*state_->index_, *state_->graph_, storage);
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
  // The incremental closure is the hot legality check. Constructing the
  // entire graph again merely to format a rejected probe's optional witness
  // turns a local negative decision into Fabric-sized work. Final candidate
  // verification remains an independent whole-graph rebuild; materialize a
  // transient witness here only when diagnostics will actually consume it.
  if (cycle_ && mapping_debug::enabled(mapping_debug::Level::Decision)) {
    auto graph = materializeReusableHandshakeGraph(
        *state_->index_, state_->activeFragments_, storage.reusableGraph);
    if (!graph)
      return graph.takeError();
    if ((*graph)->cycleWitness.empty())
      return candidateError(
          "delta closure reported a cycle absent from exact materialization");
    addWork(state_->materializationConstructionCount_);
    addWork(state_->materializationConstructionNanoseconds_,
            (*graph)->constructionNanoseconds);
    addWork(state_->materializationDeterministicWork_,
            (*graph)->deterministicWork);
    pendingGraph_ = std::move(*graph);
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
    std::shared_ptr<detail::MaterializedHandshakeGraph> previous =
        std::move(state_->graph_);
    state_->graph_ = std::move(pendingGraph_);
    recycleMaterializedHandshakeGraph(std::move(previous), *scratch_->storage_);
    finish();
    return llvm::Error::success();
  }
  if (scratch_->storage_->changedContributions.empty()) {
    pendingGraph_.reset();
    finish();
    return llvm::Error::success();
  }
  auto work = applyHandshakeArcDelta(*state_->index_, *state_->graph_,
                                     *scratch_->storage_);
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
  recycleMaterializedHandshakeGraph(std::move(pendingGraph_),
                                    *scratch_->storage_);
  finish();
}

void HandshakeCandidateTransaction::finish() {
  state_->activeTransaction_ = nullptr;
  scratch_->activeTransaction_ = nullptr;
  scratch_->resetTransaction();
  scratch_ = nullptr;
  projectionScratch_ = nullptr;
  state_.reset();
}
