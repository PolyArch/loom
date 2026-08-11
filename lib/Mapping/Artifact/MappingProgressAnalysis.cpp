#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_progress_analysis_invalid: " + message);
}

struct ActorDependencyGraph final {
  llvm::DenseMap<std::uint64_t, std::uint32_t> actorOrdinals;
  std::vector<std::size_t> offsets;
  std::vector<std::uint32_t> destinations;
  bool acyclic = false;
};

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
  for (const auto [ordinal, actor] : llvm::enumerate(actors))
    if (!result.actorOrdinals
             .try_emplace(actor.ref.entity.value(),
                          static_cast<std::uint32_t>(ordinal))
             .second)
      return invalid("canonical actor inventory contains a duplicate");

  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
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
            edges.emplace_back(sourceOrdinal->second, sinkOrdinal->second);
            return llvm::Error::success();
          }))
    return std::move(error);

  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  result.offsets.assign(actors.size() + 1, 0);
  std::vector<std::uint32_t> indegrees(actors.size(), 0);
  for (const auto &[source, sink] : edges) {
    ++result.offsets[static_cast<std::size_t>(source) + 1];
    if (indegrees[sink] == std::numeric_limits<std::uint32_t>::max())
      return invalid("actor dependency indegree overflows");
    ++indegrees[sink];
  }
  for (std::size_t index = 1; index < result.offsets.size(); ++index)
    result.offsets[index] += result.offsets[index - 1];
  result.destinations.resize(edges.size());
  std::vector<std::size_t> cursors = result.offsets;
  cursors.pop_back();
  for (const auto &[source, sink] : edges)
    result.destinations[cursors[source]++] = sink;

  std::vector<std::uint32_t> ready;
  ready.reserve(actors.size());
  for (std::uint32_t actor = 0; actor != actors.size(); ++actor)
    if (indegrees[actor] == 0)
      ready.push_back(actor);
  std::size_t visited = 0;
  for (std::size_t cursor = 0; cursor != ready.size(); ++cursor) {
    const std::uint32_t source = ready[cursor];
    ++visited;
    for (std::size_t edge = result.offsets[source];
         edge != result.offsets[source + 1]; ++edge) {
      const std::uint32_t sink = result.destinations[edge];
      if (--indegrees[sink] == 0)
        ready.push_back(sink);
    }
  }
  result.acyclic = visited == actors.size();
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

bool isBuffered(const std::optional<::loom::fabric::FabricPhysicalTraversalRef>
                    &traversal) {
  if (!traversal)
    return false;
  const auto *fifo = std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
      &traversal->payload);
  return fifo &&
         fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered;
}

llvm::Expected<bool>
dependentBranchIsBuffered(const SpatialRouteTreeView &route,
                          std::uint64_t prerequisiteSink,
                          std::uint64_t dependentSink) {
  if (prerequisiteSink >= route.sinks.size() ||
      dependentSink >= route.sinks.size())
    return invalid("route progress dependency names an absent sink");
  const SpatialRouteSinkView &prerequisite = route.sinks[prerequisiteSink];
  const SpatialRouteSinkView &dependent = route.sinks[dependentSink];
  if (prerequisite.nodeOrdinal >= route.nodes.size() ||
      dependent.nodeOrdinal >= route.nodes.size())
    return invalid("route progress dependency names an absent node");
  if (isBuffered(dependent.localTraversal))
    return true;

  std::vector<bool> prerequisiteAncestors(route.nodes.size(), false);
  std::uint64_t node = prerequisite.nodeOrdinal;
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

llvm::Expected<MappingProgressClosure> deriveMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) {
  auto graph = buildActorDependencyGraph(dataflow, coveredGraphs);
  if (!graph)
    return graph.takeError();
  return MappingProgressClosure{
      graph->acyclic ? MappingProgressClosureKind::ProvenNoClosedWaitSet
                     : MappingProgressClosureKind::ProofNotEstablished};
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
    std::vector<std::optional<std::uint32_t>> sinkActors;
    sinkActors.reserve(net.sinks.size());
    for (const auto &sink : net.sinks)
      sinkActors.push_back(actorOrdinal(*graph, sink));
    for (std::size_t prerequisite = 0; prerequisite != net.sinks.size();
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
        result.push_back({netOrdinal, prerequisite, dependent});
      }
    }
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.logicalNetOrdinal, lhs.dependentSinkOrdinal,
                    lhs.prerequisiteSinkOrdinal) <
           std::tie(rhs.logicalNetOrdinal, rhs.dependentSinkOrdinal,
                    rhs.prerequisiteSinkOrdinal);
  });
  return result;
}

llvm::Expected<MappingProgressClosure> deriveSpatialMappingProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  auto basis = deriveMappingProgressClosure(dataflow, techMapping.covers());
  if (!basis ||
      basis->kind != MappingProgressClosureKind::ProvenNoClosedWaitSet)
    return basis;
  auto dependencies =
      deriveSpatialRouteProgressDependencies(dataflow, techMapping);
  if (!dependencies)
    return dependencies.takeError();
  const auto nets = techMapping.residualLogicalNets();
  for (const SpatialRouteProgressDependency &dependency : *dependencies) {
    if (dependency.logicalNetOrdinal >= nets.size())
      return invalid("route progress dependency names an absent logical net");
    const TechResidualLogicalNetView &net = nets[dependency.logicalNetOrdinal];
    const auto route = llvm::find_if(routes, [&](const auto &candidate) {
      return candidate.logicalNet == net.producer;
    });
    if (route == routes.end())
      return invalid("route progress dependency has no selected route");
    if (dependency.prerequisiteSinkOrdinal >= net.sinks.size() ||
        dependency.dependentSinkOrdinal >= net.sinks.size())
      return invalid("route progress dependency names an absent logical sink");
    const auto prerequisite =
        llvm::find_if(route->sinks, [&](const auto &sink) {
          return sink.sink == net.sinks[dependency.prerequisiteSinkOrdinal];
        });
    const auto dependent = llvm::find_if(route->sinks, [&](const auto &sink) {
      return sink.sink == net.sinks[dependency.dependentSinkOrdinal];
    });
    if (prerequisite == route->sinks.end() || dependent == route->sinks.end())
      return invalid("selected route omits a progress dependency sink");
    auto buffered = dependentBranchIsBuffered(
        *route, std::distance(route->sinks.begin(), prerequisite),
        std::distance(route->sinks.begin(), dependent));
    if (!buffered)
      return buffered.takeError();
    if (!*buffered)
      return MappingProgressClosure{
          MappingProgressClosureKind::ProvenClosedWaitSet};
  }
  return basis;
}

} // namespace loom::mapping
