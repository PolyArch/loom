#include "Dataflow/IR/DataflowStaticScheduleAnalysis.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace dataflow {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "dataflow_static_schedule_invalid: " + message);
}

llvm::Expected<std::uint64_t>
checkedSum(std::uint64_t left, std::uint64_t right, llvm::StringRef subject) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return invalid(subject + " exceeds u64");
  return left + right;
}

llvm::Error addTo(std::uint64_t &value, std::uint64_t increment,
                  llvm::StringRef subject) {
  auto sum = checkedSum(value, increment, subject);
  if (!sum)
    return sum.takeError();
  value = *sum;
  return llvm::Error::success();
}

using ActorKey = std::uint64_t;
using GraphKey = std::uint64_t;
using EdgeKey =
    std::tuple<ActorKey, std::uint64_t, ActorKey, std::uint64_t>;

ActorKey actorKey(ActorRef actor) { return actor.entity.value(); }

GraphKey graphKey(GraphRef graph) { return graph.entity.value(); }

EdgeKey edgeKey(const ActorTokenResultRef &producer,
                const ActorTokenOperandRef &consumer) {
  return {actorKey(producer.actor), producer.ordinal, actorKey(consumer.actor),
          consumer.ordinal};
}

bool isTemporalStateCarrier(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::DataflowStream:
  case OperationSchemaId::DataflowCarry:
  case OperationSchemaId::DataflowInvariant:
  case OperationSchemaId::DataflowGate:
    return true;
  default:
    return false;
  }
}

struct WorkingEdge final {
  std::size_t source = 0;
  std::size_t sink = 0;
  std::size_t analysisEdge = 0;
  bool initializedFeedback = false;
  bool timingFeedback = false;
};

struct WorkingGraph final {
  GraphRef graph;
  std::vector<std::size_t> actors;
  std::vector<std::size_t> edges;
};

struct AnalysisBuilder final {
  StaticScheduleAnalysis result;
  std::vector<llvm::SmallVector<
      semantics::InitializedFeedbackInputDescriptor, 3>>
      initializedFeedbackInputs;
  std::vector<WorkingEdge> edges;
  std::vector<WorkingGraph> graphs;
  std::map<ActorKey, std::size_t> actorOrdinals;
  std::map<GraphKey, std::size_t> graphOrdinals;

  llvm::Error analyzeGraph(const WorkingGraph &graph) {
    const std::size_t actorCount = graph.actors.size();
    if (actorCount == 0)
      return llvm::Error::success();

    std::map<std::size_t, std::size_t> localActor;
    for (const auto &[local, actor] : llvm::enumerate(graph.actors))
      localActor.emplace(actor, local);

    std::vector<std::vector<std::size_t>> adjacency(actorCount);
    for (std::size_t edgeOrdinal : graph.edges) {
      const WorkingEdge &edge = edges[edgeOrdinal];
      if (edge.initializedFeedback)
        continue;
      const auto source = localActor.find(edge.source);
      const auto sink = localActor.find(edge.sink);
      if (source == localActor.end() || sink == localActor.end())
        return invalid("graph edge escapes its owning graph");
      adjacency[source->second].push_back(sink->second);
    }

    std::vector<std::size_t> discovery(actorCount,
                                       std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> low(actorCount, 0);
    std::vector<std::size_t> stack;
    std::vector<bool> onStack(actorCount, false);
    std::vector<std::size_t> component(actorCount,
                                       std::numeric_limits<std::size_t>::max());
    std::size_t nextDiscovery = 0;
    std::size_t componentCount = 0;
    std::function<void(std::size_t)> visit = [&](std::size_t actor) {
      discovery[actor] = low[actor] = nextDiscovery++;
      stack.push_back(actor);
      onStack[actor] = true;
      for (std::size_t sink : adjacency[actor]) {
        if (discovery[sink] == std::numeric_limits<std::size_t>::max()) {
          visit(sink);
          low[actor] = std::min(low[actor], low[sink]);
        } else if (onStack[sink]) {
          low[actor] = std::min(low[actor], discovery[sink]);
        }
      }
      if (low[actor] != discovery[actor])
        return;
      while (true) {
        const std::size_t member = stack.back();
        stack.pop_back();
        onStack[member] = false;
        component[member] = componentCount;
        if (member == actor)
          break;
      }
      ++componentCount;
    };
    for (std::size_t actor = 0; actor < actorCount; ++actor)
      if (discovery[actor] == std::numeric_limits<std::size_t>::max())
        visit(actor);

    std::vector<std::uint64_t> componentWeight(componentCount, 0);
    for (std::size_t actor = 0; actor < actorCount; ++actor)
      ++componentWeight[component[actor]];

    bool postInitializationAcyclic = llvm::all_of(
        componentWeight, [](std::uint64_t weight) { return weight == 1; });
    std::set<std::pair<std::size_t, std::size_t>> componentEdgeSet;
    for (std::size_t edgeOrdinal : graph.edges) {
      const WorkingEdge &edge = edges[edgeOrdinal];
      if (edge.initializedFeedback)
        continue;
      const std::size_t source = component[localActor.at(edge.source)];
      const std::size_t sink = component[localActor.at(edge.sink)];
      if (edge.source == edge.sink)
        postInitializationAcyclic = false;
      if (source != sink)
        componentEdgeSet.emplace(source, sink);
    }
    result.recurrenceTopologies_.push_back(
        {graph.graph, postInitializationAcyclic});

    std::vector<std::vector<std::size_t>> componentSuccessors(componentCount);
    std::vector<std::size_t> indegree(componentCount, 0);
    for (const auto &[source, sink] : componentEdgeSet) {
      componentSuccessors[source].push_back(sink);
      ++indegree[sink];
    }
    std::priority_queue<std::size_t, std::vector<std::size_t>,
                        std::greater<std::size_t>>
        ready;
    for (std::size_t ordinal = 0; ordinal < componentCount; ++ordinal)
      if (indegree[ordinal] == 0)
        ready.push(ordinal);
    std::vector<std::size_t> topological;
    topological.reserve(componentCount);
    while (!ready.empty()) {
      const std::size_t source = ready.top();
      ready.pop();
      topological.push_back(source);
      for (std::size_t sink : componentSuccessors[source])
        if (--indegree[sink] == 0)
          ready.push(sink);
    }
    if (topological.size() != componentCount)
      return invalid("SCC condensation is cyclic");

    std::vector<std::uint64_t> prefix = componentWeight;
    for (std::size_t source : topological)
      for (std::size_t sink : componentSuccessors[source]) {
        auto path = checkedSum(prefix[source], componentWeight[sink],
                               "graph critical path length");
        if (!path)
          return path.takeError();
        prefix[sink] = std::max(prefix[sink], *path);
      }
    std::vector<std::uint64_t> suffix = componentWeight;
    for (std::size_t source : llvm::reverse(topological))
      for (std::size_t sink : componentSuccessors[source]) {
        auto path = checkedSum(componentWeight[source], suffix[sink],
                               "graph critical path length");
        if (!path)
          return path.takeError();
        suffix[source] = std::max(suffix[source], *path);
      }
    const std::uint64_t graphLength =
        *std::max_element(prefix.begin(), prefix.end());
    std::vector<bool> graphCriticalComponent(componentCount, false);
    for (std::size_t ordinal = 0; ordinal < componentCount; ++ordinal) {
      auto joined = checkedSum(prefix[ordinal], suffix[ordinal],
                               "graph critical path test");
      if (!joined)
        return joined.takeError();
      graphCriticalComponent[ordinal] =
          *joined - componentWeight[ordinal] == graphLength;
    }
    std::set<std::pair<std::size_t, std::size_t>> graphCriticalEdges;
    for (const auto &[source, sink] : componentEdgeSet) {
      auto joined =
          checkedSum(prefix[source], suffix[sink], "graph critical edge test");
      if (!joined)
        return joined.takeError();
      if (*joined == graphLength)
        graphCriticalEdges.emplace(source, sink);
    }
    for (std::size_t local = 0; local < actorCount; ++local)
      if (graphCriticalComponent[component[local]])
        result.actors_[graph.actors[local]].graphCriticalLength = graphLength;
    for (std::size_t edgeOrdinal : graph.edges) {
      const WorkingEdge &edge = edges[edgeOrdinal];
      if (edge.initializedFeedback)
        continue;
      const std::size_t source = component[localActor.at(edge.source)];
      const std::size_t sink = component[localActor.at(edge.sink)];
      if ((source == sink && graphCriticalComponent[source]) ||
          graphCriticalEdges.count({source, sink}))
        if (llvm::Error error =
                addTo(result.edges_[edge.analysisEdge].weight, graphLength,
                      "graph critical edge weight"))
          return error;
    }

    for (std::size_t feedbackOrdinal : graph.edges) {
      const WorkingEdge &feedback = edges[feedbackOrdinal];
      if (!feedback.timingFeedback)
        continue;
      const std::size_t recurrenceSource =
          component[localActor.at(feedback.sink)];
      const std::size_t recurrenceSink =
          component[localActor.at(feedback.source)];
      std::vector<std::uint64_t> from(componentCount, 0);
      from[recurrenceSource] = componentWeight[recurrenceSource];
      for (std::size_t source : topological) {
        if (from[source] == 0)
          continue;
        for (std::size_t sink : componentSuccessors[source]) {
          auto path = checkedSum(from[source], componentWeight[sink],
                                 "recurrence critical path length");
          if (!path)
            return path.takeError();
          from[sink] = std::max(from[sink], *path);
        }
      }
      const std::uint64_t recurrenceLength = from[recurrenceSink];
      if (recurrenceLength == 0)
        continue;
      std::vector<std::uint64_t> to(componentCount, 0);
      to[recurrenceSink] = componentWeight[recurrenceSink];
      for (std::size_t source : llvm::reverse(topological))
        for (std::size_t sink : componentSuccessors[source]) {
          if (to[sink] == 0)
            continue;
          auto path = checkedSum(componentWeight[source], to[sink],
                                 "recurrence critical path length");
          if (!path)
            return path.takeError();
          to[source] = std::max(to[source], *path);
        }
      std::vector<bool> recurrenceCriticalComponent(componentCount, false);
      for (std::size_t ordinal = 0; ordinal < componentCount; ++ordinal) {
        if (from[ordinal] == 0 || to[ordinal] == 0)
          continue;
        auto joined = checkedSum(from[ordinal], to[ordinal],
                                 "recurrence critical path test");
        if (!joined)
          return joined.takeError();
        recurrenceCriticalComponent[ordinal] =
            *joined - componentWeight[ordinal] == recurrenceLength;
      }
      std::set<std::pair<std::size_t, std::size_t>> recurrenceCriticalEdges;
      for (const auto &[source, sink] : componentEdgeSet) {
        if (from[source] == 0 || to[sink] == 0)
          continue;
        auto joined = checkedSum(from[source], to[sink],
                                 "recurrence critical edge test");
        if (!joined)
          return joined.takeError();
        if (*joined == recurrenceLength)
          recurrenceCriticalEdges.emplace(source, sink);
      }
      for (std::size_t local = 0; local < actorCount; ++local)
        if (recurrenceCriticalComponent[component[local]])
          if (llvm::Error error = addTo(
                  result.actors_[graph.actors[local]].recurrenceCriticalLength,
                  recurrenceLength, "actor recurrence criticality"))
            return error;
      for (std::size_t edgeOrdinal : graph.edges) {
        const WorkingEdge &edge = edges[edgeOrdinal];
        if (edge.initializedFeedback) {
          if (edgeOrdinal == feedbackOrdinal)
            if (llvm::Error error = addTo(
                    result.edges_[edge.analysisEdge].weight, recurrenceLength,
                    "feedback edge recurrence criticality"))
              return error;
          continue;
        }
        const std::size_t source = component[localActor.at(edge.source)];
        const std::size_t sink = component[localActor.at(edge.sink)];
        if ((source == sink && recurrenceCriticalComponent[source]) ||
            recurrenceCriticalEdges.count({source, sink}))
          if (llvm::Error error = addTo(
                  result.edges_[edge.analysisEdge].weight, recurrenceLength,
                  "edge recurrence criticality"))
            return error;
      }
    }
    return llvm::Error::success();
  }
};

} // namespace

const StaticActorCriticality *
StaticScheduleAnalysis::findActor(ActorRef actor) const {
  const auto found = std::lower_bound(
      actors_.begin(), actors_.end(), actorKey(actor),
      [](const StaticActorCriticality &candidate, ActorKey key) {
        return actorKey(candidate.actor) < key;
      });
  return found != actors_.end() && found->actor == actor ? &*found : nullptr;
}

std::uint64_t StaticScheduleAnalysis::edgeWeight(
    const ActorTokenResultRef &producer,
    const ActorTokenOperandRef &consumer) const {
  const EdgeKey key = edgeKey(producer, consumer);
  const auto found = std::lower_bound(
      edges_.begin(), edges_.end(), key,
      [](const StaticActorEdgeCriticality &candidate, const EdgeKey &target) {
        return edgeKey(candidate.producer, candidate.consumer) < target;
      });
  return found != edges_.end() &&
                 edgeKey(found->producer, found->consumer) == key
             ? found->weight
             : 0;
}

std::uint64_t
StaticScheduleAnalysis::graphCriticalLength(GraphRef graph) const {
  std::uint64_t result = 0;
  for (const StaticActorCriticality &actor : actors_)
    if (actor.graph == graph)
      result = std::max(result, actor.graphCriticalLength);
  return result;
}

llvm::Expected<StaticScheduleAnalysis>
deriveStaticScheduleAnalysis(const CanonicalDataflowProgramView &dataflow,
                             llvm::ArrayRef<GraphRef> covers) {
  AnalysisBuilder builder;
  std::set<GraphKey> covered;
  for (GraphRef graph : covers) {
    auto resolved = dataflow.resolve(graph);
    if (!resolved)
      return resolved.takeError();
    if (!covered.insert(graphKey(graph)).second)
      return invalid("covered graph inventory contains a duplicate");
    builder.graphOrdinals.emplace(graphKey(graph), builder.graphs.size());
    builder.graphs.push_back({graph, {}, {}});
  }
  for (const CanonicalActorView &actor : dataflow.actors()) {
    const auto graph = builder.graphOrdinals.find(graphKey(actor.graph));
    if (graph == builder.graphOrdinals.end())
      continue;
    const std::size_t ordinal = builder.result.actors_.size();
    if (!builder.actorOrdinals.emplace(actorKey(actor.ref), ordinal).second)
      return invalid("covered actor inventory contains a duplicate");
    const OperationSchemaId schema = requireOperationSchema(actor.op);
    auto feedbackInputs = semantics::projectActorInitializedFeedbackInputs(
        schema, actor.op->getNumOperands(), actor.op->getNumResults());
    if (!feedbackInputs)
      return feedbackInputs.takeError();
    builder.result.actors_.push_back(
        {actor.ref, actor.graph, 0, 0, isTemporalStateCarrier(schema)});
    builder.initializedFeedbackInputs.push_back(std::move(*feedbackInputs));
    builder.graphs[graph->second].actors.push_back(ordinal);
  }
  if (llvm::Error error = dataflow.forEachGraphEdge(
          [&](const CanonicalGraphProducerEndpointRef &producer,
              const CanonicalGraphConsumerEndpointRef &consumer)
              -> llvm::Error {
            const auto *actorProducer =
                std::get_if<ActorTokenResultRef>(&producer);
            const auto *actorConsumer =
                std::get_if<ActorTokenOperandRef>(&consumer);
            if (!actorProducer || !actorConsumer)
              return llvm::Error::success();
            const auto source =
                builder.actorOrdinals.find(actorKey(actorProducer->actor));
            const auto sink =
                builder.actorOrdinals.find(actorKey(actorConsumer->actor));
            if (source == builder.actorOrdinals.end() &&
                sink == builder.actorOrdinals.end())
              return llvm::Error::success();
            if (source == builder.actorOrdinals.end() ||
                sink == builder.actorOrdinals.end())
              return invalid("covered graph edge has one foreign actor");
            const auto &sourceActor = builder.result.actors_[source->second];
            const auto &sinkActor = builder.result.actors_[sink->second];
            if (sourceActor.graph != sinkActor.graph)
              return invalid("canonical actor edge crosses graph ownership");
            const auto feedbackDescriptor = llvm::find_if(
                builder.initializedFeedbackInputs[sink->second],
                [&](const auto &candidate) {
                  return candidate.inputOrdinal == actorConsumer->ordinal;
                });
            const bool initializedFeedback =
                feedbackDescriptor !=
                builder.initializedFeedbackInputs[sink->second].end();
            const bool timingFeedback =
                initializedFeedback &&
                feedbackDescriptor->timingDependenceDistance.has_value();
            const std::size_t analysisEdge = builder.result.edges_.size();
            builder.result.edges_.push_back({*actorProducer, *actorConsumer,
                                             sourceActor.graph, 0,
                                             initializedFeedback});
            if (timingFeedback)
              builder.result.feedbacks_.push_back(
                  {*actorProducer, *actorConsumer, sourceActor.graph,
                   *feedbackDescriptor->timingDependenceDistance});
            const std::size_t edgeOrdinal = builder.edges.size();
            builder.edges.push_back({source->second, sink->second, analysisEdge,
                                     initializedFeedback, timingFeedback});
            builder
                .graphs[builder.graphOrdinals.at(graphKey(sourceActor.graph))]
                .edges.push_back(edgeOrdinal);
            return llvm::Error::success();
          }))
    return std::move(error);
  for (const WorkingGraph &graph : builder.graphs)
    if (llvm::Error error = builder.analyzeGraph(graph))
      return std::move(error);
  llvm::sort(builder.result.actors_, [](const auto &left, const auto &right) {
    return actorKey(left.actor) < actorKey(right.actor);
  });
  llvm::sort(builder.result.edges_, [](const auto &left, const auto &right) {
    return edgeKey(left.producer, left.consumer) <
           edgeKey(right.producer, right.consumer);
  });
  llvm::sort(builder.result.feedbacks_,
             [](const auto &left, const auto &right) {
               return edgeKey(left.producer, left.consumer) <
                      edgeKey(right.producer, right.consumer);
             });
  llvm::sort(builder.result.recurrenceTopologies_,
             [](const auto &left, const auto &right) {
               return graphKey(left.graph) < graphKey(right.graph);
             });
  return std::move(builder.result);
}

} // namespace dataflow
