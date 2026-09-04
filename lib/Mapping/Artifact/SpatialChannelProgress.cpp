#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "MappingProgressInternal.h"

#include "ConfiguredHardwareProjectionInternal.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::mapping {
using namespace progress_detail;
namespace {

struct ActorDependencyGraph final {
  struct InitializedFeedbackEdge final {
    ::dataflow::ActorTokenResultRef producer;
    ::dataflow::ActorTokenOperandRef consumer;
    std::uint32_t producerActor = 0;
    std::uint32_t consumerActor = 0;
    bool closesCycle = false;
  };

  llvm::DenseMap<std::uint64_t, std::uint32_t> actorOrdinals;
  /// The actor DAG after initialized-feedback edges have been removed.
  std::vector<std::size_t> offsets;
  std::vector<std::uint32_t> destinations;
  std::vector<::dataflow::ActorRef> actorRefs;
  std::vector<InitializedFeedbackEdge> initializedFeedbackEdges;
  std::vector<::dataflow::ActorRef> postInitializationCycle;
  bool acyclic = false;
  bool postInitializationAcyclic = false;
};

llvm::Expected<bool>
closeActorEdges(std::size_t actorCount,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> edges,
                std::vector<std::size_t> *closedOffsets = nullptr,
                std::vector<std::uint32_t> *closedDestinations = nullptr) {
  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::vector<std::size_t> offsets(actorCount + 1, 0);
  std::vector<std::uint32_t> indegrees(actorCount, 0);
  for (const auto &[source, sink] : edges) {
    if (source >= actorCount || sink >= actorCount)
      return invalid("actor dependency edge is out of range");
    ++offsets[static_cast<std::size_t>(source) + 1];
    if (indegrees[sink] == std::numeric_limits<std::uint32_t>::max())
      return invalid("actor dependency indegree overflows");
    ++indegrees[sink];
  }
  for (std::size_t index = 1; index < offsets.size(); ++index)
    offsets[index] += offsets[index - 1];
  std::vector<std::uint32_t> destinations(edges.size());
  std::vector<std::size_t> cursors = offsets;
  cursors.pop_back();
  for (const auto &[source, sink] : edges)
    destinations[cursors[source]++] = sink;

  std::vector<std::uint32_t> ready;
  ready.reserve(actorCount);
  for (std::uint32_t actor = 0; actor != actorCount; ++actor)
    if (indegrees[actor] == 0)
      ready.push_back(actor);
  std::size_t visited = 0;
  for (std::size_t cursor = 0; cursor != ready.size(); ++cursor) {
    const std::uint32_t source = ready[cursor];
    ++visited;
    for (std::size_t edge = offsets[source]; edge != offsets[source + 1];
         ++edge) {
      const std::uint32_t sink = destinations[edge];
      if (--indegrees[sink] == 0)
        ready.push_back(sink);
    }
  }
  if (closedOffsets)
    *closedOffsets = std::move(offsets);
  if (closedDestinations)
    *closedDestinations = std::move(destinations);
  return visited == actorCount;
}

bool actorReachable(const ActorDependencyGraph &graph, std::uint32_t source,
                    std::uint32_t target) {
  if (source == target)
    return true;
  std::vector<bool> visited(graph.actorOrdinals.size(), false);
  std::vector<std::uint32_t> worklist{source};
  visited[source] = true;
  for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
    for (std::size_t edge = graph.offsets[worklist[cursor]];
         edge != graph.offsets[worklist[cursor] + 1]; ++edge) {
      const std::uint32_t sink = graph.destinations[edge];
      if (sink == target)
        return true;
      if (!visited[sink]) {
        visited[sink] = true;
        worklist.push_back(sink);
      }
    }
  return false;
}

llvm::Expected<ActorDependencyGraph> buildActorDependencyGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) {
  llvm::DenseMap<std::uint64_t, bool> covered;
  covered.reserve(coveredGraphs.size());
  for (const ::dataflow::GraphRef graph : coveredGraphs) {
    if (!covered.try_emplace(graph.entity.value(), true).second)
      return invalid("covered graph inventory contains a duplicate");
    auto resolved = dataflow.resolve(graph);
    if (!resolved)
      return resolved.takeError();
  }

  std::vector<::dataflow::CanonicalActorView> actors;
  for (const ::dataflow::CanonicalActorView &actor : dataflow.actors())
    if (covered.count(actor.graph.entity.value()) != 0)
      actors.push_back(actor);
  if (actors.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("actor inventory exceeds the native analysis domain");

  ActorDependencyGraph result;
  result.actorOrdinals.reserve(actors.size());
  result.actorRefs.reserve(actors.size());
  std::vector<std::vector<std::uint32_t>> initializedFeedbackInputs;
  initializedFeedbackInputs.reserve(actors.size());
  for (const auto [ordinal, actor] : llvm::enumerate(actors))
    if (!result.actorOrdinals
             .try_emplace(actor.ref.entity.value(),
                          static_cast<std::uint32_t>(ordinal))
             .second) {
      return invalid("canonical actor inventory contains a duplicate");
    } else {
      result.actorRefs.push_back(actor.ref);
      auto inputs = initializedFeedbackInputOrdinals(actor);
      if (!inputs)
        return inputs.takeError();
      initializedFeedbackInputs.push_back(std::move(*inputs));
    }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> allEdges;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> nonFeedbackEdges;
  if (llvm::Error error = dataflow.forEachGraphEdge(
          [&](const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
              const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer)
              -> llvm::Error {
            const auto *source =
                std::get_if<::dataflow::ActorTokenResultRef>(&producer);
            const auto *sink =
                std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
            if (!source || !sink)
              return llvm::Error::success();
            const auto sourceOrdinal =
                result.actorOrdinals.find(source->actor.entity.value());
            const auto sinkOrdinal =
                result.actorOrdinals.find(sink->actor.entity.value());
            if (sourceOrdinal == result.actorOrdinals.end() &&
                sinkOrdinal == result.actorOrdinals.end())
              return llvm::Error::success();
            if (sourceOrdinal == result.actorOrdinals.end() ||
                sinkOrdinal == result.actorOrdinals.end())
              return invalid("covered graph edge crosses the actor catalog");
            allEdges.emplace_back(sourceOrdinal->second, sinkOrdinal->second);
            const bool initializedFeedback = llvm::is_contained(
                initializedFeedbackInputs[sinkOrdinal->second], sink->ordinal);
            if (initializedFeedback) {
              result.initializedFeedbackEdges.push_back(
                  {*source, *sink, sourceOrdinal->second, sinkOrdinal->second,
                   false});
            } else {
              nonFeedbackEdges.emplace_back(sourceOrdinal->second,
                                            sinkOrdinal->second);
            }
            return llvm::Error::success();
          }))
    return std::move(error);

  auto acyclic = closeActorEdges(actors.size(), std::move(allEdges));
  if (!acyclic)
    return acyclic.takeError();
  result.acyclic = *acyclic;
  auto postInitializationAcyclic =
      closeActorEdges(actors.size(), std::move(nonFeedbackEdges),
                      &result.offsets, &result.destinations);
  if (!postInitializationAcyclic)
    return postInitializationAcyclic.takeError();
  result.postInitializationAcyclic = *postInitializationAcyclic;
  if (!result.postInitializationAcyclic) {
    std::vector<std::vector<std::uint32_t>> edges(actors.size());
    for (std::uint32_t source = 0; source != actors.size(); ++source)
      edges[source].assign(result.destinations.begin() + result.offsets[source],
                           result.destinations.begin() +
                               result.offsets[source + 1]);
    std::vector<std::uint32_t> cycle = findDirectedCycle(edges);
    if (!cycle.empty() && cycle.front() == cycle.back())
      cycle.pop_back();
    result.postInitializationCycle.reserve(cycle.size());
    for (std::uint32_t ordinal : cycle)
      result.postInitializationCycle.push_back(result.actorRefs[ordinal]);
  }
  if (result.postInitializationAcyclic)
    for (ActorDependencyGraph::InitializedFeedbackEdge &edge :
         result.initializedFeedbackEdges)
      edge.closesCycle =
          actorReachable(result, edge.consumerActor, edge.producerActor);
  return result;
}

std::optional<std::uint32_t>
actorOrdinal(const ActorDependencyGraph &graph,
             const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint) {
  const auto *operand =
      std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
  if (!operand)
    return std::nullopt;
  const auto found = graph.actorOrdinals.find(operand->actor.entity.value());
  if (found == graph.actorOrdinals.end())
    return std::nullopt;
  return found->second;
}

llvm::Expected<bool> dependentBranchHasDurableBoundary(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const SpatialRouteTreeView &route,
    const SpatialRouteProgressPrerequisite &prerequisite,
    std::uint64_t dependentSink) {
  const auto *external =
      std::get_if<SpatialRouteExternalSinkPrerequisite>(&prerequisite);
  if ((external && external->sinkOrdinal >= route.sinks.size()) ||
      dependentSink >= route.sinks.size())
    return invalid("route progress dependency names an absent sink");
  const SpatialRouteSinkView &dependent = route.sinks[dependentSink];
  if (dependent.nodeOrdinal >= route.nodes.size())
    return invalid("route progress dependency names an absent node");
  auto localBoundary = deriveSpatialSinkDurableProgressBoundary(
      techMapping, fabric, computeBindings, route, dependent);
  if (!localBoundary)
    return localBoundary.takeError();
  if (*localBoundary)
    return true;

  if (!external) {
    std::uint64_t node = dependent.nodeOrdinal;
    for (std::size_t visited = 0;; ++visited) {
      if (visited >= route.nodes.size())
        return invalid("route dependent ancestry is cyclic");
      if (isBuffered(route.nodes[node].incomingTraversal))
        return true;
      const auto parent = route.nodes[node].parentOrdinal;
      if (!parent)
        return false;
      if (*parent >= route.nodes.size())
        return invalid("route dependent parent is out of range");
      node = *parent;
    }
  }

  const SpatialRouteSinkView &prerequisiteSink =
      route.sinks[external->sinkOrdinal];
  if (prerequisiteSink.nodeOrdinal >= route.nodes.size())
    return invalid("route progress prerequisite names an absent node");

  std::vector<bool> prerequisiteAncestors(route.nodes.size(), false);
  std::uint64_t node = prerequisiteSink.nodeOrdinal;
  for (std::size_t visited = 0;; ++visited) {
    if (visited >= route.nodes.size() || prerequisiteAncestors[node])
      return invalid("route prerequisite ancestry is cyclic");
    prerequisiteAncestors[node] = true;
    const auto parent = route.nodes[node].parentOrdinal;
    if (!parent)
      break;
    if (*parent >= route.nodes.size())
      return invalid("route prerequisite parent is out of range");
    node = *parent;
  }

  node = dependent.nodeOrdinal;
  for (std::size_t visited = 0; !prerequisiteAncestors[node]; ++visited) {
    if (visited >= route.nodes.size())
      return invalid("route dependent ancestry is cyclic");
    if (isBuffered(route.nodes[node].incomingTraversal))
      return true;
    const auto parent = route.nodes[node].parentOrdinal;
    if (!parent || *parent >= route.nodes.size())
      return invalid("route sink branches have no common ancestor");
    node = *parent;
  }
  return false;
}

} // namespace

llvm::Expected<MappingDataflowProgressBasis> deriveMappingDataflowProgressBasis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) {
  auto graph = buildActorDependencyGraph(dataflow, coveredGraphs);
  if (!graph)
    return graph.takeError();
  MappingDataflowProgressBasisKind kind =
      MappingDataflowProgressBasisKind::Cyclic;
  if (graph->acyclic)
    kind = MappingDataflowProgressBasisKind::Acyclic;
  else if (graph->postInitializationAcyclic)
    kind = MappingDataflowProgressBasisKind::InitializedFeedback;
  return MappingDataflowProgressBasis{
      kind, graph->actorRefs.size(), graph->initializedFeedbackEdges.size(),
      std::move(graph->postInitializationCycle)};
}

void emitMappingDataflowProgressBasisDiagnostic(
    const MappingDataflowProgressBasis &basis,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    mapping_debug::Stage stage) {
  if (basis.kind != MappingDataflowProgressBasisKind::Cyclic)
    return;
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "dataflow_progress_basis";
        fields["closure_status"] = "proof_not_established";
        fields["covered_actor_count"] = basis.coveredActorCount;
        fields["initialized_feedback_edge_count"] =
            basis.initializedFeedbackEdgeCount;
        llvm::json::Array cycle;
        for (const ::dataflow::ActorRef actor : basis.residualCycle) {
          llvm::json::Object member;
          member["actor"] = actor.entity.value();
          auto resolved = dataflow.resolve(actor);
          if (resolved) {
            member["operation"] = resolved->op->getName().getStringRef().str();
          } else {
            llvm::consumeError(resolved.takeError());
            member["operation"] = "<unresolved>";
          }
          cycle.push_back(std::move(member));
        }
        fields["residual_cycle"] = std::move(cycle);
      });
}

llvm::Expected<std::vector<SpatialRouteProgressDependency>>
deriveSpatialRouteProgressDependencies(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  auto graph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!graph)
    return graph.takeError();

  std::vector<SpatialRouteProgressDependency> result;
  std::vector<std::uint64_t> reachabilityMarks(graph->actorOrdinals.size(), 0);
  std::vector<std::uint32_t> worklist;
  worklist.reserve(graph->actorOrdinals.size());
  std::uint64_t generation = 0;
  for (const auto [netOrdinal, net] :
       llvm::enumerate(techMapping.residualLogicalNets())) {
    struct PrerequisiteCandidate final {
      ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
      SpatialRouteProgressPrerequisite prerequisite;
    };
    std::vector<PrerequisiteCandidate> candidates;
    candidates.reserve(net.sinks.size());
    for (const auto [sinkOrdinal, sink] : llvm::enumerate(net.sinks))
      candidates.push_back(
          {sink, SpatialRouteExternalSinkPrerequisite{sinkOrdinal}});
    for (const auto [realizationOrdinal, realization] :
         llvm::enumerate(techMapping.memoryRealizations()))
      for (const auto &edgeRecord :
           llvm::enumerate(realization.internalEdges)) {
        const std::size_t edgeOrdinal = edgeRecord.index();
        const TechMemoryInternalEdgeView &edge = edgeRecord.value();
        if (edge.producer == net.producer &&
            !llvm::any_of(candidates, [&](const auto &candidate) {
              return candidate.consumer == edge.consumer;
            }))
          candidates.push_back(
              {edge.consumer, SpatialRouteInternalMemoryConnectionPrerequisite{
                                  realizationOrdinal, edgeOrdinal}});
      }
    std::vector<std::optional<std::uint32_t>> sinkActors;
    sinkActors.reserve(candidates.size());
    for (const PrerequisiteCandidate &candidate : candidates)
      sinkActors.push_back(actorOrdinal(*graph, candidate.consumer));
    for (std::size_t prerequisite = 0; prerequisite != candidates.size();
         ++prerequisite) {
      if (!sinkActors[prerequisite])
        continue;
      ++generation;
      if (generation == 0) {
        std::fill(reachabilityMarks.begin(), reachabilityMarks.end(), 0);
        generation = 1;
      }
      worklist.clear();
      reachabilityMarks[*sinkActors[prerequisite]] = generation;
      worklist.push_back(*sinkActors[prerequisite]);
      for (std::size_t cursor = 0; cursor != worklist.size(); ++cursor)
        for (std::size_t edge = graph->offsets[worklist[cursor]];
             edge != graph->offsets[worklist[cursor] + 1]; ++edge) {
          const std::uint32_t successor = graph->destinations[edge];
          if (reachabilityMarks[successor] == generation)
            continue;
          reachabilityMarks[successor] = generation;
          worklist.push_back(successor);
        }
      for (std::size_t dependent = 0; dependent != net.sinks.size();
           ++dependent) {
        if (dependent == prerequisite || !sinkActors[dependent] ||
            *sinkActors[dependent] == *sinkActors[prerequisite] ||
            reachabilityMarks[*sinkActors[dependent]] != generation)
          continue;
        result.push_back(
            {netOrdinal, candidates[prerequisite].prerequisite, dependent});
      }
    }
  }
  for (const ActorDependencyGraph::InitializedFeedbackEdge &feedback :
       graph->initializedFeedbackEdges) {
    if (!feedback.closesCycle)
      continue;
    bool found = false;
    for (const auto [netOrdinal, net] :
         llvm::enumerate(techMapping.residualLogicalNets())) {
      if (net.producer !=
          ::dataflow::CanonicalGraphProducerEndpointRef(feedback.producer))
        continue;
      for (const auto [sinkOrdinal, sink] : llvm::enumerate(net.sinks)) {
        if (sink !=
            ::dataflow::CanonicalGraphConsumerEndpointRef(feedback.consumer))
          continue;
        if (found)
          return invalid("initialized feedback edge has multiple residual "
                         "physical dispositions");
        result.push_back({netOrdinal,
                          SpatialRouteInitializedFeedbackPrerequisite{},
                          sinkOrdinal});
        found = true;
      }
    }
    if (!found)
      return invalid("initialized feedback edge has no residual physical "
                     "disposition");
  }
  const auto prerequisiteKey = [](const auto &prerequisite) {
    return std::visit(
        [](const auto &typed) {
          using T = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<T, SpatialRouteExternalSinkPrerequisite>)
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{
                0, typed.sinkOrdinal, 0};
          else if constexpr (
              std::is_same_v<T,
                             SpatialRouteInternalMemoryConnectionPrerequisite>)
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{
                1, typed.memoryRealizationOrdinal, typed.internalEdgeOrdinal};
          else
            return std::tuple<std::uint8_t, std::uint64_t, std::uint64_t>{2, 0,
                                                                          0};
        },
        prerequisite);
  };
  llvm::sort(result, [&](const auto &lhs, const auto &rhs) {
    const auto lhsPrefix =
        std::tie(lhs.logicalNetOrdinal, lhs.dependentSinkOrdinal);
    const auto rhsPrefix =
        std::tie(rhs.logicalNetOrdinal, rhs.dependentSinkOrdinal);
    if (lhsPrefix != rhsPrefix)
      return lhsPrefix < rhsPrefix;
    return prerequisiteKey(lhs.prerequisite) <
           prerequisiteKey(rhs.prerequisite);
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [&](const auto &lhs, const auto &rhs) {
                             return lhs.logicalNetOrdinal ==
                                        rhs.logicalNetOrdinal &&
                                    lhs.dependentSinkOrdinal ==
                                        rhs.dependentSinkOrdinal &&
                                    prerequisiteKey(lhs.prerequisite) ==
                                        prerequisiteKey(rhs.prerequisite);
                           }),
               result.end());
  return result;
}

std::uint64_t countSpatialSharedFiniteBuffers(
    llvm::ArrayRef<SpatialFiniteBufferSelection> selections) {
  struct SelectedFifo final {
    ::loom::fabric::FabricFifoOccurrenceRef fifo;
    std::uint64_t logicalNetOrdinal = 0;
    bool shared = false;
  };
  std::vector<SelectedFifo> selectedFifos;
  std::uint64_t count = 0;
  for (const SpatialFiniteBufferSelection &selection : selections) {
    auto selected = llvm::find_if(selectedFifos, [&](const auto &use) {
      return use.fifo == selection.fifo;
    });
    if (selected == selectedFifos.end()) {
      selectedFifos.push_back(
          {selection.fifo, selection.logicalNetOrdinal, false});
      continue;
    }
    if (selected->logicalNetOrdinal == selection.logicalNetOrdinal ||
        selected->shared)
      continue;
    ++count;
    selected->shared = true;
  }
  return count;
}

MappingRouteProgressObligationProjection projectSpatialFiniteBufferRecurrence(
    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> logicalNets;
  std::vector<SpatialFiniteBufferSelection> selections;
  for (const SpatialRouteTreeView &route : routes) {
    auto logicalNet = llvm::find(logicalNets, route.logicalNet);
    if (logicalNet == logicalNets.end()) {
      logicalNets.push_back(route.logicalNet);
      logicalNet = std::prev(logicalNets.end());
    }
    const std::uint64_t logicalNetOrdinal =
        std::distance(logicalNets.begin(), logicalNet);
    const auto append =
        [&](const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                &traversal) {
          if (!traversal)
            return;
          const auto *fifo =
              std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                  &traversal->payload);
          if (fifo &&
              fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered)
            selections.push_back({logicalNetOrdinal, fifo->owner});
        };
    append(route.localTraversal);
    for (const SpatialRouteNodeView &node : route.nodes)
      append(node.incomingTraversal);
    for (const SpatialRouteSinkView &sink : route.sinks)
      append(sink.localTraversal);
  }
  return {MappingRouteProgressObligationKind::FiniteBufferRecurrence,
          countSpatialSharedFiniteBuffers(selections) == 0};
}

namespace {

/// One selected Buffered FIFO occurrence and its queue class along one
/// producer-to-sink channel path, in token flow order.
struct SpatialChannelStorageStop final {
  MappingStorageQueueProgressNode node;
  ::loom::fabric::FabricPhysicalTraversalRef traversal;
};

/// The buffered storage sequence of one residual channel path.
struct SpatialChannelStorageChain final {
  std::optional<::dataflow::ActorRef> producer;
  std::optional<::dataflow::ActorRef> consumer;
  std::optional<::dataflow::CanonicalGraphProducerEndpointRef> producerEndpoint;
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumerEndpoint;
  std::vector<SpatialChannelStorageStop> stops;
  std::uint64_t netOrdinal = 0;
  bool primed = false;
};

/// The complete per-channel storage inventory of the selected graphs, shared
/// by the buffer-dependency obligation and the reconvergent capacity proof.
struct SpatialChannelStorageInventory final {
  std::vector<SpatialChannelStorageChain> channels;
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> logicalNets;
};

} // namespace

/// Rebuilds the per-channel buffered-storage inventory of the selected graphs.
/// Channels primed by initialized feedback are marked but retained: their
/// initial tokens occupy the queue classes even though their wait facts never
/// participate in a closed wait. Returns a disengaged optional when a queue
/// class, tag residency, or relation domain is indeterminate.
llvm::Expected<std::optional<SpatialChannelStorageInventory>>
projectSpatialChannelStorageInventory(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs) {
    if (!llvm::is_contained(techMapping.covers(), graph))
      return invalid("selected progress graph is absent from TechMapping");
    if (!selectedGraphEntities.insert(graph.entity.value()).second)
      return invalid("selected progress graph inventory contains a duplicate");
  }

  // Channels primed by initialized feedback carry an initial token, so their
  // queue order and acceptance facts cannot participate in a closed wait.
  auto actorGraph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!actorGraph)
    return actorGraph.takeError();
  std::set<std::pair<std::string, std::string>> primedChannels;
  for (const ActorDependencyGraph::InitializedFeedbackEdge &feedback :
       actorGraph->initializedFeedbackEdges) {
    auto producerKey = ::dataflow::encodeDataflowReference(
        dataflow.identity(),
        ::dataflow::CanonicalGraphProducerEndpointRef(feedback.producer));
    if (!producerKey)
      return producerKey.takeError();
    auto consumerKey = ::dataflow::encodeDataflowReference(
        dataflow.identity(),
        ::dataflow::CanonicalGraphConsumerEndpointRef(feedback.consumer));
    if (!consumerKey)
      return consumerKey.takeError();
    primedChannels.emplace(
        std::string(reinterpret_cast<const char *>(producerKey->data()),
                    producerKey->size()),
        std::string(reinterpret_cast<const char *>(consumerKey->data()),
                    consumerKey->size()));
  }

  SpatialChannelStorageInventory inventory;
  const auto producerActorOf =
      [](const ::dataflow::CanonicalGraphProducerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token = std::get_if<::dataflow::ActorTokenResultRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };
  const auto consumerActorOf =
      [](const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token =
        std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };

  for (const auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    auto graph = dataflow.graphOf(route.logicalNet);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    auto logicalNet = llvm::find(inventory.logicalNets, route.logicalNet);
    if (logicalNet == inventory.logicalNets.end()) {
      inventory.logicalNets.push_back(route.logicalNet);
      logicalNet = std::prev(inventory.logicalNets.end());
    }
    const std::uint64_t netOrdinal = static_cast<std::uint64_t>(
        std::distance(inventory.logicalNets.begin(), logicalNet));
    const std::optional<::dataflow::ActorRef> producerActor =
        producerActorOf(route.logicalNet);

    for (const SpatialRouteSinkView &sink : route.sinks) {
      const std::optional<::dataflow::ActorRef> consumerActor =
          consumerActorOf(sink.sink);
      // The buffered storage sequence along this producer-to-sink path, in
      // token flow order: the root arc, the ancestor node arcs, then the sink
      // arc. Each stop names the traversal position whose node carries the
      // Physical Tag assignment for its queue class.
      std::vector<
          std::pair<std::optional<::loom::fabric::FabricPhysicalTraversalRef>,
                    std::uint64_t>>
          arcs;
      arcs.push_back({route.localTraversal, 0});
      std::vector<
          std::pair<std::optional<::loom::fabric::FabricPhysicalTraversalRef>,
                    std::uint64_t>>
          nodeArcs;
      std::optional<std::uint64_t> cursor = sink.nodeOrdinal;
      while (cursor) {
        if (*cursor >= route.nodes.size())
          return invalid("RouteTree sink names an absent node");
        const SpatialRouteNodeView &node = route.nodes[*cursor];
        if (node.incomingTraversal)
          nodeArcs.push_back({node.incomingTraversal, node.ordinal});
        cursor = node.parentOrdinal;
      }
      arcs.insert(arcs.end(), nodeArcs.rbegin(), nodeArcs.rend());
      arcs.push_back({sink.localTraversal, sink.nodeOrdinal});

      SpatialChannelStorageChain channel;
      channel.producer = producerActor;
      channel.consumer = consumerActor;
      channel.producerEndpoint = route.logicalNet;
      channel.consumerEndpoint = sink.sink;
      channel.netOrdinal = netOrdinal;
      std::set<std::string> channelTraversals;
      for (const auto &[traversal, tagNodeOrdinal] : arcs) {
        if (!traversal)
          continue;
        const auto *fifo =
            std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                &traversal->payload);
        if (!fifo ||
            fifo->mode != ::loom::fabric::FabricFifoTraversalMode::Buffered)
          continue;
        const std::vector<std::uint8_t> traversalBytes =
            ::loom::fabric::canonicalFabricBytes(*traversal);
        if (!channelTraversals
                 .insert(std::string(
                     reinterpret_cast<const char *>(traversalBytes.data()),
                     traversalBytes.size()))
                 .second)
          continue;
        const ::fabric::FifoQueueDiscipline discipline =
            fabric.fifoQueueDiscipline(fifo->owner)
                .value_or(::fabric::FifoQueueDiscipline::StrictFifo);
        MappingStaticQueueClass queueClass{MappingStaticQueueClassKind::Global,
                                           llvm::APInt(1, 0)};
        if (discipline == ::fabric::FifoQueueDiscipline::PerTagVirtualChannel) {
          // A virtual-channel class is keyed by the exact resident tag bit
          // value; an untagged path or a missing tag assignment makes the
          // class indeterminate and the whole construction unestablished.
          if (route.nodes.empty())
            return std::optional<SpatialChannelStorageInventory>{};
          const auto &endpoint = route.nodes[tagNodeOrdinal].endpoint;
          const auto path = fabric.transportEndpointDataPath(endpoint);
          if (!path || path->tagWidthBits == 0)
            return std::optional<SpatialChannelStorageInventory>{};
          auto tag = detail::resolveConfiguredHardwarePhysicalTag(
              fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
              tagNodeOrdinal);
          if (!tag)
            return std::optional<SpatialChannelStorageInventory>{};
          queueClass = MappingStaticQueueClass{
              MappingStaticQueueClassKind::PhysicalTag, *tag};
        }
        channel.stops.push_back(SpatialChannelStorageStop{
            MappingStorageQueueProgressNode{fifo->owner, queueClass},
            *traversal});
      }

      // A channel primed by initialized feedback carries its initial token;
      // none of its wait facts can participate in a closed wait.
      if (producerActor && consumerActor) {
        auto producerKey = ::dataflow::encodeDataflowReference(
            dataflow.identity(), route.logicalNet);
        if (!producerKey)
          return producerKey.takeError();
        auto consumerKey =
            ::dataflow::encodeDataflowReference(dataflow.identity(), sink.sink);
        if (!consumerKey)
          return consumerKey.takeError();
        channel.primed =
            primedChannels.count({std::string(reinterpret_cast<const char *>(
                                                  producerKey->data()),
                                              producerKey->size()),
                                  std::string(reinterpret_cast<const char *>(
                                                  consumerKey->data()),
                                              consumerKey->size())}) != 0;
      }
      inventory.channels.push_back(std::move(channel));
    }
  }
  return std::optional<SpatialChannelStorageInventory>(std::move(inventory));
}

llvm::Expected<std::optional<std::vector<MappingBufferDependencyEdge>>>
projectSpatialBufferDependencyEdges(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  auto inventory = projectSpatialChannelStorageInventory(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!inventory)
    return inventory.takeError();
  if (!*inventory)
    return std::optional<std::vector<MappingBufferDependencyEdge>>{};
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs)
    selectedGraphEntities.insert(graph.entity.value());

  std::vector<MappingBufferDependencyEdge> edges;
  std::set<std::string> edgeKeys;
  const auto emit =
      [&](MappingStaticWaitNode from, MappingStaticWaitNode to,
          MappingBufferDependencyEdgeKind kind, std::uint64_t logicalNetOrdinal,
          std::optional<::loom::fabric::FabricPhysicalTraversalRef> routeAnchor)
      -> llvm::Error {
    std::string key = staticWaitNodeKey(from) + staticWaitNodeKey(to);
    key.push_back(static_cast<char>(kind));
    appendU64(key, logicalNetOrdinal);
    if (routeAnchor) {
      const std::vector<std::uint8_t> anchor =
          ::loom::fabric::canonicalFabricBytes(*routeAnchor);
      key.append(reinterpret_cast<const char *>(anchor.data()), anchor.size());
    }
    if (!edgeKeys.insert(std::move(key)).second)
      return llvm::Error::success();
    edges.push_back(MappingBufferDependencyEdge{std::move(from), std::move(to),
                                                kind, logicalNetOrdinal,
                                                std::move(routeAnchor)});
    return llvm::Error::success();
  };
  const auto consumerActorOf =
      [](const ::dataflow::CanonicalGraphConsumerEndpointRef &endpoint)
      -> std::optional<::dataflow::ActorRef> {
    const auto *token =
        std::get_if<::dataflow::ActorTokenOperandRef>(&endpoint);
    return token ? std::optional<::dataflow::ActorRef>(token->actor)
                 : std::nullopt;
  };

  // A queue class couples the nets resident in it. The terminal head-release
  // wait (queue class to consumer) exists only when the class couples at
  // least two nets: with one net the head always belongs to that net and the
  // delivery is the same handshake as the consumer's input join, never an
  // independent wait.
  std::map<std::string, std::set<std::uint64_t>> classNets;
  for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
    if (channel.primed)
      continue;
    for (const SpatialChannelStorageStop &stop : channel.stops)
      classNets[staticWaitNodeKey(stop.node)].insert(channel.netOrdinal);
  }

  for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
    if (channel.primed)
      continue;
    if (channel.stops.empty()) {
      if (channel.producer && channel.consumer) {
        if (llvm::Error error =
                emit(*channel.consumer, *channel.producer,
                     MappingBufferDependencyEdgeKind::ActorInputJoin,
                     channel.netOrdinal, std::nullopt))
          return std::move(error);
      }
      continue;
    }
    if (channel.producer) {
      if (llvm::Error error =
              emit(*channel.producer, channel.stops.front().node,
                   MappingBufferDependencyEdgeKind::OutputCausalRelease,
                   channel.netOrdinal, channel.stops.front().traversal))
        return std::move(error);
    }
    for (std::size_t stop = 1; stop != channel.stops.size(); ++stop)
      if (llvm::Error error =
              emit(channel.stops[stop - 1].node, channel.stops[stop].node,
                   MappingBufferDependencyEdgeKind::DownstreamCapacity,
                   channel.netOrdinal, channel.stops[stop].traversal))
        return std::move(error);
    if (channel.consumer) {
      const SpatialChannelStorageStop &terminal = channel.stops.back();
      if (llvm::Error error =
              emit(*channel.consumer, terminal.node,
                   MappingBufferDependencyEdgeKind::ActorInputJoin,
                   channel.netOrdinal, terminal.traversal))
        return std::move(error);
      const auto coupled = classNets.find(staticWaitNodeKey(terminal.node));
      if (coupled != classNets.end() && coupled->second.size() >= 2) {
        if (llvm::Error error =
                emit(terminal.node, *channel.consumer,
                     MappingBufferDependencyEdgeKind::ActorInputJoin,
                     channel.netOrdinal, terminal.traversal))
          return std::move(error);
      }
    }
  }

  for (const SpatialPeOperandQueueMatchGroupView &group : operandQueueGroups) {
    auto graph = dataflow.graphOf(group.logicalNet);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    auto logicalNet = llvm::find((*inventory)->logicalNets, group.logicalNet);
    if (logicalNet == (*inventory)->logicalNets.end()) {
      (*inventory)->logicalNets.push_back(group.logicalNet);
      logicalNet = std::prev((*inventory)->logicalNets.end());
    }
    const std::uint64_t netOrdinal = static_cast<std::uint64_t>(
        std::distance((*inventory)->logicalNets.begin(), logicalNet));
    for (const SpatialPeOperandQueueMatchView &match : group.matches) {
      const MappingOperandQueueProgressNode queueNode{match.queue, match.fu};
      for (const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer :
           match.consumers) {
        const std::optional<::dataflow::ActorRef> consumerActor =
            consumerActorOf(consumer);
        if (!consumerActor)
          continue;
        if (llvm::Error error =
                emit(*consumerActor, queueNode,
                     MappingBufferDependencyEdgeKind::OperandQueueOwner,
                     netOrdinal, std::nullopt))
          return std::move(error);
      }
    }
  }
  return std::optional<std::vector<MappingBufferDependencyEdge>>(
      std::move(edges));
}

llvm::Expected<std::vector<MappingReconvergentCapacityObligation>>
deriveMappingReconvergentCapacityProof(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  auto inventory = projectSpatialChannelStorageInventory(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!inventory)
    return inventory.takeError();

  struct OwnerInventory final {
    ::loom::fabric::FabricFifoOccurrenceRef owner;
    std::map<std::string, MappingStaticQueueClass> queueClasses;
    std::map<std::string, ::loom::fabric::FabricPhysicalTraversalRef>
        routeAnchors;
    std::set<std::uint64_t> logicalNets;
    bool simpleTopology = true;
  };
  const auto ownerKey =
      [](const ::loom::fabric::FabricFifoOccurrenceRef owner) {
        const std::vector<std::uint8_t> bytes =
            ::loom::fabric::canonicalFabricBytes(owner);
        return std::string(reinterpret_cast<const char *>(bytes.data()),
                           bytes.size());
      };
  std::map<std::string, OwnerInventory> ownerByKey;
  if (*inventory)
    for (const SpatialChannelStorageChain &channel : (*inventory)->channels) {
      std::set<std::string> channelOwners;
      for (const SpatialChannelStorageStop &stop : channel.stops) {
        const std::string key = ownerKey(stop.node.owner);
        auto [position, inserted] = ownerByKey.try_emplace(
            key, OwnerInventory{stop.node.owner, {}, {}, {}, true});
        if (!inserted && position->second.owner != stop.node.owner)
          return invalid("FIFO capacity owner key collision");
        if (!channelOwners.insert(key).second)
          position->second.simpleTopology = false;
        position->second.queueClasses.emplace(staticWaitNodeKey(stop.node),
                                              stop.node.queueClass);
        position->second.logicalNets.insert(channel.netOrdinal);
        const std::vector<std::uint8_t> anchorBytes =
            ::loom::fabric::canonicalFabricBytes(stop.traversal);
        position->second.routeAnchors.emplace(
            std::string(reinterpret_cast<const char *>(anchorBytes.data()),
                        anchorBytes.size()),
            stop.traversal);
      }
    }
  std::vector<OwnerInventory> owners;
  owners.reserve(ownerByKey.size());
  for (auto &[key, owner] : ownerByKey) {
    (void)key;
    owners.push_back(std::move(owner));
  }
  const std::size_t ownerCount = owners.size();

  auto actorGraph = buildActorDependencyGraph(dataflow, techMapping.covers());
  if (!actorGraph)
    return actorGraph.takeError();
  std::vector<bool> established(ownerCount,
                                actorGraph->postInitializationAcyclic);
  for (std::size_t ordinal = 0; ordinal != ownerCount; ++ordinal)
    established[ordinal] =
        established[ordinal] && owners[ordinal].simpleTopology;

  std::vector<MappingReconvergentCapacityObligation> obligations;
  obligations.reserve(ownerCount);
  for (std::size_t ordinal = 0; ordinal != ownerCount; ++ordinal) {
    const OwnerInventory &owner = owners[ordinal];
    // Read the shared queue-slot pool through the FIFO contract's typed state
    // and capacity-domain owners. Key-ordered storage is an implementation
    // detail of ResourceContract, not this proof's semantic selector.
    const ::fabric::ResourceContract *contract = fabric.resourceContract(
        ::loom::fabric::FabricInventoryOwnerRef::of(owner.owner));
    std::optional<std::uint64_t> pool;
    const ::fabric::StateKey bufferedQueue =
        ::fabric::fifoResourceState(::fabric::FifoResourceState::BufferedQueue);
    const ::fabric::CapacityDimensionKey queueSlot =
        ::fabric::fifoBufferedCapacity(
            ::fabric::FifoBufferedCapacity::QueueSlot);
    if (contract && contract->stateCount() > bufferedQueue.ordinal()) {
      const auto dimensions = contract->capacityDimensions(bufferedQueue);
      if (dimensions.size() > queueSlot.ordinal())
        pool = dimensions[queueSlot.ordinal()].capacity.value();
    }
    // One producer binding cannot own a second active transfer until every
    // sink of its current token has reached durable acceptance. Therefore one
    // logical net contributes at most one resident token to a selected FIFO
    // occurrence, including a distance-one initialized-feedback token. A pool
    // with one slot per distinct selected net removes shared-pool capacity
    // from the wait graph for every firing interleaving. Queue classes still
    // own dequeue order, but they share this one physical capacity bound.
    const bool proven =
        established[ordinal] && pool.has_value() && !owner.logicalNets.empty();
    const std::optional<std::uint64_t> sufficient =
        proven ? std::optional<std::uint64_t>(owner.logicalNets.size())
               : std::nullopt;
    std::vector<MappingStaticQueueClass> queueClasses;
    queueClasses.reserve(owner.queueClasses.size());
    for (const auto &[key, queueClass] : owner.queueClasses)
      queueClasses.push_back(queueClass);
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> routeAnchors;
    routeAnchors.reserve(owner.routeAnchors.size());
    for (const auto &[key, anchor] : owner.routeAnchors)
      routeAnchors.push_back(anchor);
    MappingReconvergentCapacityObligation obligation{
        owner.owner, std::move(queueClasses), std::move(routeAnchors),
        pool.value_or(0), sufficient};
    obligations.push_back(std::move(obligation));
  }
  return obligations;
}

llvm::Expected<MappingProgressProjection> projectSpatialMappingProgress(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups,
    llvm::ArrayRef<::dataflow::GraphRef> selectedGraphs) {
  std::set<std::uint64_t> selectedGraphEntities;
  for (const ::dataflow::GraphRef graph : selectedGraphs) {
    if (!llvm::is_contained(techMapping.covers(), graph))
      return invalid("selected progress graph is absent from TechMapping");
    if (!selectedGraphEntities.insert(graph.entity.value()).second)
      return invalid("selected progress graph inventory contains a duplicate");
  }
  auto basis = deriveMappingDataflowProgressBasis(dataflow, selectedGraphs);
  if (!basis)
    return basis.takeError();
  MappingProgressProjection result;
  result.basis = *basis;
  auto bufferDependencyEdges = projectSpatialBufferDependencyEdges(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      operandQueueGroups, selectedGraphs);
  if (!bufferDependencyEdges)
    return bufferDependencyEdges.takeError();
  result.bufferDependencyEdges = std::move(*bufferDependencyEdges);
  auto capacityObligations = deriveMappingReconvergentCapacityProof(
      dataflow, techMapping, fabric, routes, resourceUses, physicalTagSegments,
      selectedGraphs);
  if (!capacityObligations)
    return capacityObligations.takeError();
  result.reconvergentCapacityObligations = std::move(*capacityObligations);
  auto temporalDispatchDomains =
      deriveSpatialTemporalPeDispatchDomains(fabric, computeBindings);
  if (!temporalDispatchDomains)
    return temporalDispatchDomains.takeError();
  std::set<std::uint64_t> dispatchedRealizations;
  for (const SpatialTemporalPeDispatchDomainView &domain :
       *temporalDispatchDomains) {
    if (domain.candidates.empty() ||
        domain.resetPosition >= domain.candidates.size())
      return invalid("temporal dispatch progress domain is malformed");
    for (const SpatialTemporalPeDispatchCandidateView &candidate :
         domain.candidates)
      if (!dispatchedRealizations.insert(candidate.realization).second)
        return invalid("temporal compute realization belongs to multiple fair "
                       "dispatch domains");
  }
  auto dependencies =
      deriveSpatialRouteProgressDependencies(dataflow, techMapping);
  if (!dependencies)
    return dependencies.takeError();
  const auto nets = techMapping.residualLogicalNets();
  for (const SpatialRouteProgressDependency &dependency : *dependencies) {
    if (dependency.logicalNetOrdinal >= nets.size())
      return invalid("route progress dependency names an absent logical net");
    const TechResidualLogicalNetView &net = nets[dependency.logicalNetOrdinal];
    auto graph = dataflow.graphOf(net.producer);
    if (!graph)
      return graph.takeError();
    if (selectedGraphEntities.count(graph->entity.value()) == 0)
      continue;
    const auto *external = std::get_if<SpatialRouteExternalSinkPrerequisite>(
        &dependency.prerequisite);
    if ((external && external->sinkOrdinal >= net.sinks.size()) ||
        dependency.dependentSinkOrdinal >= net.sinks.size())
      return invalid("route progress dependency names an absent logical sink");
    const auto localTransfer =
        llvm::find_if(registerFifoTransfers, [&](const auto &transfer) {
          return transfer.logicalNet == net.producer &&
                 transfer.sink == net.sinks[dependency.dependentSinkOrdinal];
        });
    if (localTransfer != registerFifoTransfers.end()) {
      result.routeObligations.push_back(
          {MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence,
           true});
      continue;
    }
    const auto route = llvm::find_if(routes, [&](const auto &candidate) {
      return candidate.logicalNet == net.producer;
    });
    if (route == routes.end())
      return invalid("route progress dependency has no selected route");
    if (const auto *internal =
            std::get_if<SpatialRouteInternalMemoryConnectionPrerequisite>(
                &dependency.prerequisite)) {
      const auto realizations = techMapping.memoryRealizations();
      if (internal->memoryRealizationOrdinal >= realizations.size() ||
          internal->internalEdgeOrdinal >=
              realizations[internal->memoryRealizationOrdinal]
                  .internalEdges.size() ||
          realizations[internal->memoryRealizationOrdinal]
                  .internalEdges[internal->internalEdgeOrdinal]
                  .producer != net.producer)
        return invalid(
            "route progress dependency names an absent internal connection");
    }
    const auto prerequisite =
        external ? llvm::find_if(route->sinks,
                                 [&](const auto &sink) {
                                   return sink.sink ==
                                          net.sinks[external->sinkOrdinal];
                                 })
                 : route->sinks.end();
    const auto dependent = llvm::find_if(route->sinks, [&](const auto &sink) {
      return sink.sink == net.sinks[dependency.dependentSinkOrdinal];
    });
    if ((external && prerequisite == route->sinks.end()) ||
        dependent == route->sinks.end())
      return invalid("selected route omits a progress dependency sink");
    SpatialRouteProgressPrerequisite routedPrerequisite =
        dependency.prerequisite;
    if (external)
      routedPrerequisite =
          SpatialRouteExternalSinkPrerequisite{static_cast<std::uint64_t>(
              std::distance(route->sinks.begin(), prerequisite))};
    auto durable = dependentBranchHasDurableBoundary(
        techMapping, fabric, computeBindings, *route, routedPrerequisite,
        std::distance(route->sinks.begin(), dependent));
    if (!durable)
      return durable.takeError();
    result.routeObligations.push_back(
        {MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence,
         *durable});
  }
  return result;
}

llvm::Expected<MappingProgressClosure> deriveSpatialMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> operandQueueGroups) {
  if (!operandQueueGroups.empty()) {
    auto feedback = deriveSpatialPeOperandProgressFeedback(
        dataflow, techMapping, operandQueueGroups);
    if (!feedback)
      return feedback.takeError();
    mapping_debug::emit(
        mapping_debug::Level::Decision, mapping_debug::Stage::SpatialPnr,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "temporal_operand_queue_progress";
          fields["group_count"] = feedback->groupCount;
          fields["potentially_blocking_group_count"] =
              feedback->potentiallyBlockingGroupCount;
          fields["pairing_opportunity_count"] =
              feedback->pairingOpportunityCount;
          fields["distinct_ingress_count"] = feedback->distinctIngressCount;
          fields["shared_ingress_count"] = feedback->sharedIngressCount;
          fields["shared_ingress_pressure"] = feedback->sharedIngressPressure;
          fields["pairing_key_count"] = feedback->pairingKeyCount;
          fields["distinct_pairing_key_count"] =
              feedback->distinctPairingKeyCount;
          fields["status"] = static_cast<std::uint64_t>(feedback->status);
          fields["support"] = static_cast<std::uint64_t>(feedback->support);
        });
  }
  // The queue/pairing projection is a ranking and runtime-witness input at
  // this boundary. Analytic risk is deliberately not a Mapping gate; only an
  // exact queue-level closed-wait witness may change closure status.
  auto projection = projectSpatialMappingProgress(
      dataflow, techMapping, fabric, computeBindings, registerFifoTransfers,
      routes, resourceUses, physicalTagSegments, operandQueueGroups,
      techMapping.covers());
  if (!projection)
    return projection.takeError();
  auto model = freezeMappingProgressModel(dataflow, {});
  if (!model)
    return model.takeError();
  return deriveMappingProgressClosure(*model, *projection);
}

} // namespace loom::mapping
