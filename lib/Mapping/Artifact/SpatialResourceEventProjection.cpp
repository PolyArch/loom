#include "Mapping/Artifact/SpatialResourceEventProjection.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <type_traits>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_resource_event_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendSized(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

} // namespace

llvm::Expected<std::vector<::dataflow::EventFamilyKey>>
projectRootedSpatialActivityEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    const SpatialActivityEventRef &event) {
  return std::visit(
      [&](const auto &typed)
          -> llvm::Expected<std::vector<::dataflow::EventFamilyKey>> {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event, SpatialActorTransitionEventRef>) {
          ::dataflow::EventFamilyKey projected(
              ::dataflow::ContextualActorTransitionEventRef{
                  ::dataflow::ContextualActorRef{graph, typed.actor},
                  typed.transition});
          if (llvm::Error error = dataflow.validate(projected))
            return std::move(error);
          return std::vector<::dataflow::EventFamilyKey>{std::move(projected)};
        } else {
          return dataflow.projectRootedGraphEndpointEventFamilies(graph, typed);
        }
      },
      event);
}

llvm::Expected<::dataflow::GraphRef> resolveSpatialActivityEventGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SpatialActivityEventRef &event) {
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<::dataflow::GraphRef> {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event, SpatialActorTransitionEventRef>) {
          auto actor = dataflow.resolve(typed.actor);
          if (!actor)
            return actor.takeError();
          return actor->graph;
        } else {
          return dataflow.graphOf(typed);
        }
      },
      event);
}

llvm::Expected<std::vector<MappingCausalReleasePointProjection>>
projectRootedSpatialCausalRelease(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<SpatialEventPointView> release,
    llvm::ArrayRef<SpatialComputeResultHandoffView> resultHandoffs) {
  auto launchedGraph = dataflow.resolve(graph);
  if (!launchedGraph)
    return launchedGraph.takeError();
  std::vector<MappingCausalReleasePointProjection> result;
  result.reserve(release.size());
  for (const auto &point : release) {
    auto ownerGraph = resolveSpatialActivityEventGraph(dataflow, point.event);
    if (!ownerGraph)
      return ownerGraph.takeError();
    if (*ownerGraph != *launchedGraph)
      continue;
    const auto *produced =
        std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
            &point.event);
    const auto *actorResult =
        produced ? std::get_if<::dataflow::ActorTokenResultRef>(produced)
                 : nullptr;
    if (!actorResult)
      return invalid("Spatial causal release does not name a compute result");
    const auto handoff = llvm::find_if(resultHandoffs, [&](const auto &entry) {
      return entry.producer == *actorResult;
    });
    if (handoff == resultHandoffs.end())
      return invalid("Spatial causal release has no selected result handoff");
    result.push_back({RootedSpatialResultHandoffProjection{graph, *handoff},
                      point.guaranteedOffset});
  }
  if (!release.empty() && result.empty())
    return invalid("Spatial activation has no release in its rooted graph");
  return result;
}

llvm::Expected<std::vector<MappingProgressCausalReleaseProjection>>
projectMappingCausalReleasePrerequisites(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<MappingCausalReleasePointProjection> release) {
  std::map<std::vector<std::uint8_t>, MappingProgressCausalReleaseProjection>
      clauses;
  const auto append =
      [&](std::vector<::dataflow::EventFamilyKey> alternatives) -> llvm::Error {
    auto key =
        encodeMappingCausalReleaseEventKey(dataflow.identity(), alternatives);
    if (!key)
      return key.takeError();
    clauses.try_emplace(std::move(*key), MappingProgressCausalReleaseProjection{
                                             std::move(alternatives)});
    return llvm::Error::success();
  };
  for (const auto &point : release) {
    if (const auto *events =
            std::get_if<std::vector<::dataflow::EventFamilyKey>>(
                &point.event)) {
      if (llvm::Error error = append(*events))
        return std::move(error);
      continue;
    }
    const auto &handoff =
        std::get<RootedSpatialResultHandoffProjection>(point.event);
    for (const auto &sink : handoff.result.sinks) {
      if (sink.durableBoundary)
        continue;
      auto events = dataflow.projectRootedGraphEndpointEventFamilies(
          handoff.graph, sink.sink);
      if (!events)
        return events.takeError();
      if (llvm::Error error = append(std::move(*events)))
        return std::move(error);
    }
  }
  std::vector<MappingProgressCausalReleaseProjection> result;
  result.reserve(clauses.size());
  for (auto &[key, clause] : clauses)
    result.push_back(std::move(clause));
  return result;
}

llvm::Expected<std::vector<std::uint8_t>> encodeMappingCausalReleaseEventKey(
    const ArtifactIdentity &dataflowIdentity,
    const MappingCausalReleaseEventProjection &event) {
  std::vector<std::uint8_t> result;
  appendU64(result, event.index());
  if (const auto *events =
          std::get_if<std::vector<::dataflow::EventFamilyKey>>(&event)) {
    appendU64(result, events->size());
    for (const auto &alternative : *events) {
      auto key =
          ::dataflow::encodeDataflowReference(dataflowIdentity, alternative);
      if (!key)
        return key.takeError();
      appendSized(result, *key);
    }
  } else {
    const auto &handoff = std::get<RootedSpatialResultHandoffProjection>(event);
    auto graph =
        ::dataflow::encodeDataflowReference(dataflowIdentity, handoff.graph);
    if (!graph)
      return graph.takeError();
    appendSized(result, *graph);
    auto producer = ::dataflow::encodeDataflowReference(
        dataflowIdentity,
        ::dataflow::CanonicalGraphProducerEndpointRef(handoff.result.producer));
    if (!producer)
      return producer.takeError();
    appendSized(result, *producer);
  }
  return result;
}

} // namespace loom::mapping
