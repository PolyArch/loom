#include "Mapping/Artifact/SpatialProgressAnalysis.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_progress_analysis_invalid: " + message);
}

} // namespace

llvm::Expected<SpatialProgressClosure> deriveSpatialProgressClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  const auto actors = dataflow.actors();
  if (actors.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("actor inventory exceeds the native analysis domain");

  llvm::DenseMap<std::uint64_t, std::uint32_t> actorOrdinals;
  actorOrdinals.reserve(actors.size());
  for (auto [ordinal, actor] : llvm::enumerate(actors)) {
    if (!actorOrdinals
             .try_emplace(actor.ref.entity.value(),
                          static_cast<std::uint32_t>(ordinal))
             .second)
      return invalid("canonical actor inventory contains a duplicate");
  }

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
            auto sourceOrdinal =
                actorOrdinals.find(source->actor.entity.value());
            auto sinkOrdinal = actorOrdinals.find(sink->actor.entity.value());
            if (sourceOrdinal == actorOrdinals.end() ||
                sinkOrdinal == actorOrdinals.end())
              return invalid("graph edge names an actor outside the catalog");
            edges.emplace_back(sourceOrdinal->second, sinkOrdinal->second);
            return llvm::Error::success();
          }))
    return std::move(error);

  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::vector<std::size_t> offsets(actors.size() + 1, 0);
  std::vector<std::uint32_t> indegrees(actors.size(), 0);
  for (const auto &[source, sink] : edges) {
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
  ready.reserve(actors.size());
  for (std::uint32_t actor = 0; actor != actors.size(); ++actor)
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

  return SpatialProgressClosure{
      visited == actors.size()
          ? SpatialProgressClosureKind::ProvenNoClosedWaitSet
          : SpatialProgressClosureKind::ProofNotEstablished};
}

} // namespace loom::mapping
