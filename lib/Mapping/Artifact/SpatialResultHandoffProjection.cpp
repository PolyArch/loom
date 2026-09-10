#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <utility>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_result_handoff_invalid: " + message);
}

} // namespace

llvm::Expected<std::vector<SpatialComputeResultConnectionsView>>
deriveSpatialComputeResultConnections(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  if (techMapping.dataflowIdentity() != dataflow.identity())
    return invalid("compute connection inputs have inconsistent owners");

  using ProducerKey = std::pair<std::uint64_t, std::uint64_t>;
  struct Result final {
    ::dataflow::ActorTokenResultRef producer;
    std::map<std::vector<std::uint8_t>, SpatialResultConnectionSinkView> sinks;
  };
  std::map<ProducerKey, Result> results;
  std::map<std::uint64_t, std::uint64_t> realizationByActor;
  for (const auto &realization : techMapping.computeRealizations())
    for (const auto &binding : realization.actors) {
      realizationByActor.emplace(binding.actor.entity.value(),
                                 realization.entityId);
      auto actor = dataflow.resolve(binding.actor);
      if (!actor)
        return actor.takeError();
      for (std::uint64_t ordinal = 0; ordinal < actor->op->getNumResults();
           ++ordinal)
        results.emplace(ProducerKey{binding.actor.entity.value(), ordinal},
                        Result{{binding.actor, ordinal}, {}});
    }
  if (llvm::Error error = dataflow.forEachGraphEdge([&](const auto &producer,
                                                        const auto &consumer)
                                                        -> llvm::Error {
        const auto *source =
            std::get_if<::dataflow::ActorTokenResultRef>(&producer);
        if (!source)
          return llvm::Error::success();
        const auto owner =
            realizationByActor.find(source->actor.entity.value());
        if (owner == realizationByActor.end())
          return llvm::Error::success();

        SpatialResultConnectionSinkView handoff{consumer, std::nullopt};
        const auto residual = llvm::find_if(
            techMapping.residualLogicalNets(), [&](const auto &net) {
              return net.producer == producer &&
                     llvm::is_contained(net.sinks, consumer);
            });
        if (residual == techMapping.residualLogicalNets().end()) {
          const auto *sink =
              std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
          const auto target =
              sink ? realizationByActor.find(sink->actor.entity.value())
                   : realizationByActor.end();
          if (target == realizationByActor.end() ||
              target->second != owner->second)
            return invalid("unrouted compute result has no internal sink");
        } else {
          handoff.residualBranch = SpatialResultResidualBranch{
              static_cast<std::uint64_t>(
                  residual - techMapping.residualLogicalNets().begin()),
              static_cast<std::uint64_t>(llvm::find(residual->sinks, consumer) -
                                         residual->sinks.begin())};
        }

        auto key =
            ::dataflow::encodeDataflowReference(dataflow.identity(), consumer);
        if (!key)
          return key.takeError();
        auto result = results.find(
            ProducerKey{source->actor.entity.value(), source->ordinal});
        if (result == results.end())
          return invalid("compute edge names an absent actor result");
        if (!result->second.sinks.emplace(std::move(*key), std::move(handoff))
                 .second)
          return invalid("compute result repeats a consumer");
        return llvm::Error::success();
      }))
    return std::move(error);

  std::vector<SpatialComputeResultConnectionsView> projection;
  projection.reserve(results.size());
  for (auto &[key, result] : results) {
    SpatialComputeResultConnectionsView handoff{result.producer, {}};
    handoff.sinks.reserve(result.sinks.size());
    for (auto &[sinkKey, sink] : result.sinks)
      handoff.sinks.push_back(std::move(sink));
    projection.push_back(std::move(handoff));
  }
  return projection;
}

llvm::Expected<std::vector<SpatialComputeResultHandoffView>>
deriveSpatialComputeResultHandoffs(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("compute handoff inputs have inconsistent owners");
  auto connections =
      deriveSpatialComputeResultConnections(dataflow, techMapping);
  if (!connections)
    return connections.takeError();
  std::vector<SpatialComputeResultHandoffView> projection;
  projection.reserve(connections->size());
  for (const auto &result : *connections) {
    SpatialComputeResultHandoffView handoff{result.producer, {}};
    const ::dataflow::CanonicalGraphProducerEndpointRef producer(
        result.producer);
    for (const auto &connection : result.sinks) {
      SpatialResultHandoffSinkView sink{connection.sink, std::nullopt};
      if (connection.residualBranch) {
        const auto local =
            llvm::find_if(registerFifoTransfers, [&](const auto &transfer) {
              return transfer.logicalNet == producer &&
                     transfer.sink == connection.sink;
            });
        if (local != registerFifoTransfers.end()) {
          auto kind = classifySpatialAttachmentDurableProgressBoundary(
              fabric, local->writeTraversal, std::nullopt);
          if (!kind)
            return kind.takeError();
          if (*kind != SpatialDurableProgressBoundaryKind::RegisterFifo)
            return invalid("register transfer has no durable write");
          sink.durableBoundary = SpatialDurableProgressBoundaryView{
              *kind, local->writeTraversal, std::nullopt};
        } else {
          const auto route = llvm::find_if(routes, [&](const auto &tree) {
            return tree.logicalNet == producer;
          });
          if (route == routes.end())
            return invalid("residual compute result has no route");
          const auto terminal =
              llvm::find_if(route->sinks, [&](const auto &candidate) {
                return candidate.sink == connection.sink;
              });
          if (terminal == route->sinks.end())
            return invalid("result route omits a residual consumer");
          auto traversals = spatialRouteBranchTraversals(*route, *terminal);
          if (!traversals)
            return traversals.takeError();
          for (const auto &traversal : *traversals) {
            auto kind = classifySpatialAttachmentDurableProgressBoundary(
                fabric, traversal, std::nullopt);
            if (!kind)
              return kind.takeError();
            if (*kind != SpatialDurableProgressBoundaryKind::None) {
              sink.durableBoundary = SpatialDurableProgressBoundaryView{
                  *kind, traversal, std::nullopt};
              break;
            }
          }
          if (!sink.durableBoundary) {
            auto boundary = deriveSpatialSinkDurableProgressBoundary(
                techMapping, fabric, computeBindings, *route, *terminal);
            if (!boundary)
              return boundary.takeError();
            sink.durableBoundary = std::move(*boundary);
          }
        }
      }
      handoff.sinks.push_back(std::move(sink));
    }
    projection.push_back(std::move(handoff));
  }
  return projection;
}

} // namespace loom::mapping
