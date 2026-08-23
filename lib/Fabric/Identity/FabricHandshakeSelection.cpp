#include "Fabric/Identity/FabricHandshake.h"

#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

struct ResolvedSelectedHandshakeGraph final {
  using Arc = std::pair<std::size_t, std::size_t>;

  std::map<std::vector<std::uint8_t>, std::size_t> boundaryNodes;
  std::size_t nodeCount = 0;
  std::vector<Arc> arcs;
};

llvm::Expected<ResolvedSelectedHandshakeGraph> resolveSelectedHandshakeGraph(
    const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeOwnerModel> models,
    llvm::ArrayRef<HandshakeDependencyArc> unconditionalArcs,
    ExecutionControlView executionControl = {}) {
  if (executionControl.stopRequested())
    return invalid("selected handshake resolution was interrupted");
  if (llvm::Error error = detail::verifyMemoryInternalHandshakeClosure(
          selection.memoryOperations))
    return std::move(error);

  using OwnerKey = std::vector<std::uint8_t>;
  std::map<OwnerKey, FabricHandshakeSelection> ownerSelections;
  std::set<std::vector<std::uint8_t>> traversalKeys;
  for (const FabricPhysicalTraversalRef &traversal : selection.traversals) {
    if (executionControl.stopRequested())
      return invalid("selected handshake resolution was interrupted");
    if (!traversalKeys.insert(canonicalFabricBytes(traversal)).second)
      return invalid("selected traversal relation contains a duplicate");
    const auto owner = detail::handshakeTraversalOwner(traversal);
    if (!owner)
      return invalid("selected traversal has no handshake owner");
    ownerSelections[detail::handshakeOwnerKey(*owner)].traversals.push_back(
        traversal);
  }
  for (const FabricSwitchHandshakeActivationSelection &activation :
       selection.switchActivations)
    ownerSelections[detail::handshakeOwnerKey(
                        FabricHandshakeOwner::switchResource(
                            activation.key.occurrence))]
        .switchActivations.push_back(activation);
  for (const FabricFuHandshakeSelection &selected : selection.fuCapabilities)
    ownerSelections[detail::handshakeOwnerKey(
                        FabricHandshakeOwner::fu(selected.occurrence()))]
        .fuCapabilities.push_back(selected);
  for (const FabricMemoryHandshakeSelection &selected :
       selection.memoryOperations)
    ownerSelections[detail::handshakeOwnerKey(FabricHandshakeOwner::memory(
                        selected.capability().port.memory))]
        .memoryOperations.push_back(selected);

  ResolvedSelectedHandshakeGraph graph;
  const auto boundaryNode = [&](const HandshakeSignalRef &signal) {
    const auto [found, inserted] = graph.boundaryNodes.try_emplace(
        detail::handshakeSignalKey(signal), graph.nodeCount);
    if (inserted)
      ++graph.nodeCount;
    return found->second;
  };
  for (const HandshakeDependencyArc &arc : unconditionalArcs)
    graph.arcs.emplace_back(boundaryNode(arc.source),
                            boundaryNode(arc.destination));

  std::map<OwnerKey, const HandshakeOwnerModel *> modelsByOwner;
  for (const HandshakeOwnerModel &model : models)
    if (!modelsByOwner.emplace(detail::handshakeOwnerKey(model.owner()), &model)
             .second)
      return invalid("handshake model inventory repeats an owner");

  for (const auto &[key, ownerSelection] : ownerSelections) {
    if (executionControl.stopRequested())
      return invalid("selected handshake resolution was interrupted");
    const auto foundModel = modelsByOwner.find(key);
    if (foundModel == modelsByOwner.end())
      return invalid("selected handshake relation names a stale owner");
    const HandshakeOwnerModel &model = *foundModel->second;

    std::vector<std::uint32_t> activeArcs;
    if (model.owner().kind() == FabricHandshakeOwnerKind::FuOccurrence &&
        ownerSelection.fuCapabilities.empty()) {
      for (std::uint32_t fragmentOrdinal = 0;
           fragmentOrdinal != model.fragmentCount(); ++fragmentOrdinal) {
        const HandshakeActivationFragment fragment =
            model.fragment(fragmentOrdinal);
        if (fragment.activationKind != HandshakeActivationKind::Always)
          continue;
        for (std::uint32_t index = 0; index < fragment.contributionCount;
             ++index)
          activeArcs.push_back(model.fragmentContributionOrdinal(
              fragment.contributionOffset + index));
      }
    } else if (model.owner().kind() == FabricHandshakeOwnerKind::FuOccurrence) {
      for (const FabricFuHandshakeSelection &fuSelection :
           ownerSelection.fuCapabilities) {
        FabricHandshakeSelection local;
        local.traversals = ownerSelection.traversals;
        local.fuCapabilities.push_back(fuSelection);
        auto active = resolveSelectedHandshake(model, local);
        if (!active)
          return active.takeError();
        activeArcs.insert(activeArcs.end(), active->arcOrdinals().begin(),
                          active->arcOrdinals().end());
      }
      llvm::sort(activeArcs);
      activeArcs.erase(std::unique(activeArcs.begin(), activeArcs.end()),
                       activeArcs.end());
    } else {
      auto active = resolveSelectedHandshake(model, ownerSelection);
      if (!active)
        return active.takeError();
      activeArcs.assign(active->arcOrdinals().begin(),
                        active->arcOrdinals().end());
    }

    std::map<std::uint32_t, std::size_t> modelNodes;
    const auto selectedNode =
        [&](std::uint32_t ordinal) -> llvm::Expected<std::size_t> {
      if (ordinal >= model.nodeCount())
        return invalid("selected handshake arc endpoint is out of range");
      const auto known = modelNodes.find(ordinal);
      if (known != modelNodes.end())
        return known->second;
      const HandshakeOwnerNode node = model.node(ordinal);
      const std::size_t resolved = node.boundarySignal
                                       ? boundaryNode(*node.boundarySignal)
                                       : graph.nodeCount++;
      modelNodes.emplace(ordinal, resolved);
      return resolved;
    };
    for (std::uint32_t arcOrdinal : activeArcs) {
      if (executionControl.stopRequested())
        return invalid("selected handshake resolution was interrupted");
      if (arcOrdinal >= model.arcCount())
        return invalid("selected handshake arc is out of range");
      const HandshakeOwnerArc arc = model.arc(arcOrdinal);
      auto source = selectedNode(arc.source);
      if (!source)
        return source.takeError();
      auto destination = selectedNode(arc.destination);
      if (!destination)
        return destination.takeError();
      graph.arcs.emplace_back(*source, *destination);
    }
  }
  llvm::sort(graph.arcs);
  graph.arcs.erase(std::unique(graph.arcs.begin(), graph.arcs.end()),
                   graph.arcs.end());
  return graph;
}

llvm::Expected<std::vector<std::vector<std::size_t>>>
acyclicAdjacency(const ResolvedSelectedHandshakeGraph &graph,
                 std::vector<std::size_t> *topologicalOrder = nullptr,
                 ExecutionControlView executionControl = {}) {
  if (executionControl.stopRequested())
    return invalid("selected handshake acyclicity was interrupted");
  std::vector<std::vector<std::size_t>> adjacency(graph.nodeCount);
  std::vector<std::size_t> indegree(graph.nodeCount, 0);
  for (const auto &[source, destination] : graph.arcs) {
    adjacency[source].push_back(destination);
    ++indegree[destination];
  }
  std::vector<std::size_t> worklist;
  worklist.reserve(graph.nodeCount);
  std::vector<std::size_t> order;
  if (topologicalOrder)
    order.reserve(graph.nodeCount);
  for (std::size_t node = 0; node < graph.nodeCount; ++node)
    if (indegree[node] == 0)
      worklist.push_back(node);
  std::size_t visited = 0;
  while (!worklist.empty()) {
    if ((visited & 4095U) == 0 && executionControl.stopRequested())
      return invalid("selected handshake acyclicity was interrupted");
    const std::size_t node = worklist.back();
    worklist.pop_back();
    ++visited;
    if (topologicalOrder)
      order.push_back(node);
    for (std::size_t destination : adjacency[node])
      if (--indegree[destination] == 0)
        worklist.push_back(destination);
  }
  if (visited != graph.nodeCount)
    return invalid("SelectedCombinationalHandshakeCycle");
  if (topologicalOrder)
    *topologicalOrder = std::move(order);
  return adjacency;
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveSelectedHandshakeReachabilityWithModels(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeSignalRef> terminals,
    llvm::ArrayRef<HandshakeOwnerModel> models,
    llvm::ArrayRef<HandshakeDependencyArc> unconditionalArcs,
    ExecutionControlView executionControl = {}) {
  auto graph = resolveSelectedHandshakeGraph(
      selection, models, unconditionalArcs, executionControl);
  if (!graph)
    return graph.takeError();
  std::vector<std::size_t> topologicalOrder;
  auto adjacency =
      acyclicAdjacency(*graph, &topologicalOrder, executionControl);
  if (!adjacency)
    return adjacency.takeError();

  std::set<std::vector<std::uint8_t>> terminalKeys;
  std::vector<std::optional<std::size_t>> terminalNodes;
  terminalNodes.reserve(terminals.size());
  for (const HandshakeSignalRef &terminal : terminals) {
    if (llvm::Error error = validateFabricRef(view, terminal.endpoint))
      return std::move(error);
    const auto key = detail::handshakeSignalKey(terminal);
    if (!terminalKeys.insert(key).second)
      return invalid("selected handshake terminal inventory has a duplicate");
    const auto found = graph->boundaryNodes.find(key);
    terminalNodes.push_back(found == graph->boundaryNodes.end()
                                ? std::nullopt
                                : std::optional<std::size_t>(found->second));
  }

  // Propagate one machine word of terminal reachability at a time through the
  // selected DAG. This keeps temporary memory linear in the selected graph
  // while replacing one full graph traversal per source terminal with one
  // traversal per 64 destination terminals.
  constexpr std::size_t reachabilityBatchWidth = 64;
  std::vector<HandshakeDependencyArc> result;
  std::vector<std::uint64_t> reachable(graph->nodeCount, 0);
  for (std::size_t batch = 0; batch < terminals.size();
       batch += reachabilityBatchWidth) {
    if (executionControl.stopRequested())
      return invalid("selected handshake reachability was interrupted");
    const std::size_t batchSize =
        std::min(reachabilityBatchWidth, terminals.size() - batch);
    std::fill(reachable.begin(), reachable.end(), 0);
    for (std::size_t local = 0; local < batchSize; ++local)
      if (terminalNodes[batch + local])
        reachable[*terminalNodes[batch + local]] |= std::uint64_t{1} << local;
    std::size_t visitedNodes = 0;
    for (std::size_t node : llvm::reverse(topologicalOrder)) {
      if ((visitedNodes++ & 4095U) == 0 && executionControl.stopRequested())
        return invalid("selected handshake reachability was interrupted");
      for (std::size_t destination : (*adjacency)[node])
        reachable[node] |= reachable[destination];
    }
    for (std::size_t source = 0; source < terminals.size(); ++source) {
      if (!terminalNodes[source])
        continue;
      const std::uint64_t word = reachable[*terminalNodes[source]];
      for (std::size_t local = 0; local < batchSize; ++local) {
        const std::size_t destination = batch + local;
        if (source != destination && (word & (std::uint64_t{1} << local)) != 0)
          result.push_back({terminals[source], terminals[destination]});
      }
    }
  }

  detail::sortHandshakeDependencyArcs(result, false);
  return result;
}

} // namespace

llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection) {
  auto context = buildFabricHandshakeContext(view);
  if (!context)
    return context.takeError();
  auto graph =
      resolveSelectedHandshakeGraph(selection, context->ownerModels(),
                                    context->unconditionalDependencyArcs());
  if (!graph)
    return graph.takeError();
  auto adjacency = acyclicAdjacency(*graph);
  if (!adjacency)
    return adjacency.takeError();
  return llvm::Error::success();
}

llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    const FabricHandshakeContext &context) {
  if (llvm::Error error = revalidateFabricHandshakeContext(context, view))
    return error;
  auto graph = resolveSelectedHandshakeGraph(
      selection, context.ownerModels(), context.unconditionalDependencyArcs());
  if (!graph)
    return graph.takeError();
  auto adjacency = acyclicAdjacency(*graph);
  if (!adjacency)
    return adjacency.takeError();
  return llvm::Error::success();
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveSelectedHandshakeReachability(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeSignalRef> terminals,
    ExecutionControlView executionControl) {
  auto context = buildFabricHandshakeContext(view);
  if (!context)
    return context.takeError();
  return deriveSelectedHandshakeReachabilityWithModels(
      view, selection, terminals, context->ownerModels(),
      context->unconditionalDependencyArcs(), executionControl);
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveSelectedHandshakeReachability(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection,
    llvm::ArrayRef<HandshakeSignalRef> terminals,
    const FabricHandshakeContext &context,
    ExecutionControlView executionControl) {
  if (llvm::Error error = revalidateFabricHandshakeContext(context, view))
    return std::move(error);
  return deriveSelectedHandshakeReachabilityWithModels(
      view, selection, terminals, context.ownerModels(),
      context.unconditionalDependencyArcs(), executionControl);
}

} // namespace loom::fabric
