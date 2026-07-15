#include "Mapping/StructureValidator.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::mapping;

char MappingError::ID;

void MappingError::log(llvm::raw_ostream &stream) const { stream << message_; }

std::error_code MappingError::convertToErrorCode() const {
  return std::make_error_code(std::errc::invalid_argument);
}

namespace {

constexpr SchemaVersion supportedSchemaVersion{1, 0};

enum class EntityKind {
  Graph,
  Actor,
  Fu,
  FabricOp,
  Encoding,
  StructuralRealization,
};

using EntityKinds = std::map<std::uint64_t, EntityKind>;

llvm::Error mappingError(MappingErrorCode code, const llvm::Twine &message) {
  return llvm::make_error<MappingError>(code, message.str());
}

llvm::Error addEntity(EntityKinds &entities, std::uint64_t id,
                      EntityKind kind) {
  if (!entities.emplace(id, kind).second)
    return mappingError(MappingErrorCode::DuplicateEntityId,
                        "duplicate local entity ID");
  return llvm::Error::success();
}

struct EndpointKey {
  bool actor;
  std::uint64_t owner;
  PortDirection direction;
  std::uint32_t index;

  friend bool operator==(const EndpointKey &lhs, const EndpointKey &rhs) {
    return lhs.actor == rhs.actor && lhs.owner == rhs.owner &&
           lhs.direction == rhs.direction && lhs.index == rhs.index;
  }

  friend bool operator<(const EndpointKey &lhs, const EndpointKey &rhs) {
    return std::tie(lhs.actor, lhs.owner, lhs.direction, lhs.index) <
           std::tie(rhs.actor, rhs.owner, rhs.direction, rhs.index);
  }
};

struct EdgeKey {
  EndpointKey source;
  EndpointKey target;

  friend bool operator<(const EdgeKey &lhs, const EdgeKey &rhs) {
    return std::tie(lhs.source, lhs.target) < std::tie(rhs.source, rhs.target);
  }
};

struct DataflowPortInfo {
  std::uint64_t graph;
  EndpointKey key;
  const PortDescriptor *descriptor;
};

struct ResolvedDataflowEdge {
  DataflowPortInfo source;
  DataflowPortInfo target;
};

struct DataflowIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const GraphDescriptor *> graphs;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
  std::vector<ResolvedDataflowEdge> edges;
};

struct FabricIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const FuDescriptor *> functionalUnits;
  std::map<std::uint64_t, const FabricOpDescriptor *> operations;
  std::map<std::uint64_t, const EncodingDescriptor *> encodings;
};

llvm::Error requireLocalKind(const EntityKinds &entities, std::uint64_t id,
                             EntityKind expected) {
  const auto entity = entities.find(id);
  if (entity == entities.end() || entity->second != expected)
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "semantic view contains an invalid local reference");
  return llvm::Error::success();
}

llvm::Expected<DataflowPortInfo>
resolveDataflowPort(const DataflowEndpoint &endpoint,
                    const DataflowIndex &index) {
  if (const auto *port = std::get_if<GraphPort>(&endpoint)) {
    if (llvm::Error error = requireLocalKind(index.kinds, port->graph.value(),
                                             EntityKind::Graph))
      return std::move(error);
    const GraphDescriptor &graph = *index.graphs.at(port->graph.value());
    const auto &ports = port->direction == PortDirection::Input
                            ? graph.inputPorts
                            : graph.outputPorts;
    if (port->index >= ports.size())
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "dataflow graph port index is out of range");
    return DataflowPortInfo{
        port->graph.value(),
        EndpointKey{false, port->graph.value(), port->direction, port->index},
        &ports[port->index]};
  }

  const auto &port = std::get<ActorPort>(endpoint);
  if (llvm::Error error =
          requireLocalKind(index.kinds, port.actor.value(), EntityKind::Actor))
    return std::move(error);
  const ActorDescriptor &actor = *index.actors.at(port.actor.value());
  const auto &ports = port.direction == PortDirection::Input
                          ? actor.inputPorts
                          : actor.outputPorts;
  if (port.index >= ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "dataflow actor port index is out of range");
  return DataflowPortInfo{
      actor.graph.value(),
      EndpointKey{true, actor.id.value(), port.direction, port.index},
      &ports[port.index]};
}

llvm::Expected<DataflowIndex>
buildDataflowIndex(const DataflowProgramView &dataflow) {
  if (dataflow.identity.empty())
    return mappingError(MappingErrorCode::InvalidArtifactIdentity,
                        "Dataflow artifact identity is empty");

  DataflowIndex index;
  for (const GraphDescriptor &graph : dataflow.graphs) {
    if (llvm::Error error =
            addEntity(index.kinds, graph.id.value(), EntityKind::Graph))
      return std::move(error);
    index.graphs.emplace(graph.id.value(), &graph);
  }
  for (const ActorDescriptor &actor : dataflow.actors) {
    if (llvm::Error error =
            addEntity(index.kinds, actor.id.value(), EntityKind::Actor))
      return std::move(error);
    index.actors.emplace(actor.id.value(), &actor);
  }

  for (const ActorDescriptor &actor : dataflow.actors) {
    if (llvm::Error error = requireLocalKind(index.kinds, actor.graph.value(),
                                             EntityKind::Graph))
      return std::move(error);
  }

  std::set<EdgeKey> edges;
  for (const DataflowEdge &edge : dataflow.edges) {
    auto source = resolveDataflowPort(edge.source, index);
    if (!source)
      return source.takeError();
    auto target = resolveDataflowPort(edge.target, index);
    if (!target)
      return target.takeError();

    const bool validSource =
        source->key.actor ? source->key.direction == PortDirection::Output
                          : source->key.direction == PortDirection::Input;
    const bool validTarget =
        target->key.actor ? target->key.direction == PortDirection::Input
                          : target->key.direction == PortDirection::Output;
    if (!validSource || !validTarget || source->graph != target->graph)
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "dataflow edge has invalid direction or ownership");
    if (*source->descriptor != *target->descriptor)
      return mappingError(MappingErrorCode::PortSignatureMismatch,
                          "dataflow edge port kind or type does not match");

    const EdgeKey edgeKey{source->key, target->key};
    if (!edges.insert(edgeKey).second)
      return mappingError(MappingErrorCode::DuplicateEdge,
                          "dataflow edge is duplicated");

    index.edges.push_back(ResolvedDataflowEdge{*source, *target});
  }

  return index;
}

llvm::Expected<FabricIndex> buildFabricIndex(const FabricHardwareView &fabric) {
  if (fabric.identity.empty())
    return mappingError(MappingErrorCode::InvalidArtifactIdentity,
                        "Fabric artifact identity is empty");

  FabricIndex index;
  for (const FuDescriptor &fu : fabric.functionalUnits) {
    if (llvm::Error error =
            addEntity(index.kinds, fu.id.value(), EntityKind::Fu))
      return std::move(error);
    index.functionalUnits.emplace(fu.id.value(), &fu);
  }
  for (const FabricOpDescriptor &operation : fabric.operations) {
    if (llvm::Error error =
            addEntity(index.kinds, operation.id.value(), EntityKind::FabricOp))
      return std::move(error);
    index.operations.emplace(operation.id.value(), &operation);
  }
  for (const EncodingDescriptor &encoding : fabric.encodings) {
    if (llvm::Error error =
            addEntity(index.kinds, encoding.id.value(), EntityKind::Encoding))
      return std::move(error);
    index.encodings.emplace(encoding.id.value(), &encoding);
  }

  for (const FabricOpDescriptor &operation : fabric.operations) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, operation.fu.value(), EntityKind::Fu))
      return std::move(error);
  }
  for (const EncodingDescriptor &encoding : fabric.encodings) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, encoding.fu.value(), EntityKind::Fu))
      return std::move(error);
  }

  return index;
}

template <typename Id, typename Descriptor>
llvm::Expected<const Descriptor *>
resolveReference(const EntityReference<Id> &reference,
                 const ArtifactIdentity &artifact, const EntityKinds &kinds,
                 EntityKind expected,
                 const std::map<std::uint64_t, const Descriptor *> &entities) {
  if (reference.artifact != artifact)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "reference names a foreign artifact");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "reference names an unresolved entity ID");
  if (kind->second != expected)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "reference names an entity of the wrong kind");
  return entities.at(reference.entity.value());
}

llvm::Expected<DataflowPortInfo>
resolveActorPortReference(const ActorPortRef &port,
                          const ArtifactIdentity &artifact,
                          const DataflowIndex &index) {
  auto actor = resolveReference(port.actor, artifact, index.kinds,
                                EntityKind::Actor, index.actors);
  if (!actor)
    return actor.takeError();
  const auto &ports = port.direction == PortDirection::Input
                          ? (*actor)->inputPorts
                          : (*actor)->outputPorts;
  if (port.index >= ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "actor boundary port index is out of range");
  return DataflowPortInfo{
      (*actor)->graph.value(),
      EndpointKey{true, (*actor)->id.value(), port.direction, port.index},
      &ports[port.index]};
}

struct FuPortInfo {
  std::uint64_t fu;
  PortDirection direction;
  std::uint32_t index;
  const PortDescriptor *descriptor;
};

llvm::Expected<FuPortInfo>
resolveFuPortReference(const FuPortRef &port, const ArtifactIdentity &artifact,
                       const FabricIndex &index) {
  auto fu = resolveReference(port.fu, artifact, index.kinds, EntityKind::Fu,
                             index.functionalUnits);
  if (!fu)
    return fu.takeError();
  const auto &ports = port.direction == PortDirection::Input
                          ? (*fu)->inputPorts
                          : (*fu)->outputPorts;
  if (port.index >= ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "FU boundary port index is out of range");
  return FuPortInfo{(*fu)->id.value(), port.direction, port.index,
                    &ports[port.index]};
}

struct RealizationActors {
  const StructuralRealizationDraft *record;
  std::uint64_t graph;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
};

llvm::Expected<std::vector<RealizationActors>> resolveRealizationActors(
    const TechMappingDraft &mapping, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex,
    const std::set<std::uint64_t> &coveredGraphs,
    std::map<std::uint64_t, std::size_t> &actorToRealization) {
  std::vector<RealizationActors> resolved;
  resolved.reserve(mapping.realizations.size());

  for (const StructuralRealizationDraft &realization : mapping.realizations) {
    if (realization.actors.empty())
      return mappingError(MappingErrorCode::EmptyActorGroup,
                          "structural realization draft actor group is empty");

    RealizationActors actors{&realization, 0, {}};
    bool firstActor = true;
    for (const ActorRef &actorReference : realization.actors) {
      auto actor = resolveReference(actorReference, dataflow.identity,
                                    dataflowIndex.kinds, EntityKind::Actor,
                                    dataflowIndex.actors);
      if (!actor)
        return actor.takeError();
      if (!actors.actors.emplace((*actor)->id.value(), *actor).second ||
          !actorToRealization.emplace((*actor)->id.value(), resolved.size())
               .second)
        return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                            "actor is covered more than once");
      if (firstActor) {
        actors.graph = (*actor)->graph.value();
        firstActor = false;
      } else if ((*actor)->graph.value() != actors.graph) {
        return mappingError(MappingErrorCode::CrossGraphActorGroup,
                            "structural realization draft crosses graphs");
      }
    }
    if (!coveredGraphs.count(actors.graph))
      return mappingError(
          MappingErrorCode::IncompleteGraphCoverage,
          "structural realization draft uses an uncovered graph");
    resolved.push_back(std::move(actors));
  }

  return resolved;
}

llvm::Expected<const FuDescriptor *> validateActorToOpCorrespondence(
    const RealizationActors &realization, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex, const FabricHardwareView &fabric,
    const FabricIndex &fabricIndex) {
  auto fu = resolveReference(realization.record->fu, fabric.identity,
                             fabricIndex.kinds, EntityKind::Fu,
                             fabricIndex.functionalUnits);
  if (!fu)
    return fu.takeError();
  auto encoding = resolveReference(realization.record->encoding,
                                   fabric.identity, fabricIndex.kinds,
                                   EntityKind::Encoding, fabricIndex.encodings);
  if (!encoding)
    return encoding.takeError();
  if ((*encoding)->fu != (*fu)->id)
    return mappingError(MappingErrorCode::SelectedFuMismatch,
                        "selected encoding belongs to a different FU");

  if (realization.record->actorToOps.size() != realization.actors.size())
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-fabric.op correspondence is not complete");

  std::set<std::uint64_t> mappedActors;
  std::set<std::uint64_t> mappedOperations;
  for (const ActorToFabricOp &correspondence : realization.record->actorToOps) {
    auto actor = resolveReference(correspondence.actor, dataflow.identity,
                                  dataflowIndex.kinds, EntityKind::Actor,
                                  dataflowIndex.actors);
    if (!actor)
      return actor.takeError();
    auto operation = resolveReference(correspondence.fabricOp, fabric.identity,
                                      fabricIndex.kinds, EntityKind::FabricOp,
                                      fabricIndex.operations);
    if (!operation)
      return operation.takeError();

    if (!realization.actors.count((*actor)->id.value()) ||
        !mappedActors.insert((*actor)->id.value()).second ||
        !mappedOperations.insert((*operation)->id.value()).second)
      return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                          "actor-to-fabric.op correspondence is not bijective");
    if ((*operation)->fu != (*fu)->id)
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "fabric.op belongs to a different FU");
    if ((*actor)->inputPorts != (*operation)->inputPorts ||
        (*actor)->outputPorts != (*operation)->outputPorts)
      return mappingError(MappingErrorCode::PortSignatureMismatch,
                          "actor and fabric.op port signatures do not match");
  }

  if (mappedActors.size() != realization.actors.size())
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-fabric.op correspondence is not complete");
  return *fu;
}

std::set<EndpointKey> deriveBoundaryPorts(const RealizationActors &realization,
                                          const DataflowIndex &dataflowIndex) {
  std::set<EndpointKey> boundaryPorts;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    const bool sourceInside = edge.source.key.actor &&
                              realization.actors.count(edge.source.key.owner);
    const bool targetInside = edge.target.key.actor &&
                              realization.actors.count(edge.target.key.owner);
    if (sourceInside == targetInside)
      continue;
    boundaryPorts.insert(sourceInside ? edge.source.key : edge.target.key);
  }
  return boundaryPorts;
}

llvm::Error validateBoundaryCorrespondence(const RealizationActors &realization,
                                           const FuDescriptor &selectedFu,
                                           const DataflowProgramView &dataflow,
                                           const DataflowIndex &dataflowIndex,
                                           const FabricHardwareView &fabric,
                                           const FabricIndex &fabricIndex) {
  const std::set<EndpointKey> expected =
      deriveBoundaryPorts(realization, dataflowIndex);
  std::set<EndpointKey> mappedActorPorts;
  std::set<std::tuple<std::uint64_t, PortDirection, std::uint32_t>>
      mappedFuPorts;

  for (const BoundaryPortCorrespondence &correspondence :
       realization.record->boundaryPorts) {
    auto actorPort = resolveActorPortReference(
        correspondence.actorPort, dataflow.identity, dataflowIndex);
    if (!actorPort)
      return actorPort.takeError();
    auto fuPort = resolveFuPortReference(correspondence.fuPort, fabric.identity,
                                         fabricIndex);
    if (!fuPort)
      return fuPort.takeError();

    const auto fuKey =
        std::make_tuple(fuPort->fu, fuPort->direction, fuPort->index);
    if (!realization.actors.count(actorPort->key.owner) ||
        !expected.count(actorPort->key) ||
        !mappedActorPorts.insert(actorPort->key).second ||
        !mappedFuPorts.insert(fuKey).second)
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "software-to-FU boundary correspondence is not bijective");
    if (fuPort->fu != selectedFu.id.value())
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "boundary correspondence uses a different FU");
    if (actorPort->key.direction != fuPort->direction)
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "boundary correspondence reverses port direction");
    if (*actorPort->descriptor != *fuPort->descriptor)
      return mappingError(MappingErrorCode::PortSignatureMismatch,
                          "boundary port kind or type does not match");
  }

  if (mappedActorPorts != expected)
    return mappingError(
        MappingErrorCode::UnaccountedGraphEdge,
        "declared graph edge has an unmapped boundary endpoint");
  return llvm::Error::success();
}

llvm::Error validateCoveredEdgeAccounting(
    const DataflowIndex &dataflowIndex,
    const std::set<std::uint64_t> &coveredGraphs,
    const std::map<std::uint64_t, std::size_t> &actorToRealization) {
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    if (!coveredGraphs.count(edge.source.graph))
      continue;
    if (edge.source.key.actor &&
        !actorToRealization.count(edge.source.key.owner))
      return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                          "covered edge source actor is not realized");
    if (edge.target.key.actor &&
        !actorToRealization.count(edge.target.key.owner))
      return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                          "covered edge target actor is not realized");
  }
  return llvm::Error::success();
}

llvm::Error
validateCoveredSinkAccounting(const DataflowIndex &dataflowIndex,
                              const std::set<std::uint64_t> &coveredGraphs) {
  std::map<EndpointKey, std::size_t> driverCounts;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    if (!coveredGraphs.count(edge.source.graph))
      continue;
    if (!edge.source.key.actor && !edge.target.key.actor)
      return mappingError(
          MappingErrorCode::ActorlessGraphPassthrough,
          llvm::Twine("covered graph ") + llvm::Twine(edge.source.graph) +
              " has actorless passthrough requiring a typed obligation");
    ++driverCounts[edge.target.key];
  }

  for (std::uint64_t graphId : coveredGraphs) {
    const GraphDescriptor &graph = *dataflowIndex.graphs.at(graphId);
    for (const auto &entry : dataflowIndex.actors) {
      const ActorDescriptor &actor = *entry.second;
      if (actor.graph.value() != graphId)
        continue;
      for (std::size_t index = 0; index < actor.inputPorts.size(); ++index) {
        const EndpointKey sink{true, actor.id.value(), PortDirection::Input,
                               static_cast<std::uint32_t>(index)};
        const std::size_t count = driverCounts[sink];
        if (count == 0)
          return mappingError(
              MappingErrorCode::MissingSinkDriver,
              llvm::Twine("covered graph ") + llvm::Twine(graphId) + " actor " +
                  llvm::Twine(actor.id.value()) + " input port " +
                  llvm::Twine(index) + " has no driver");
        if (count != 1)
          return mappingError(MappingErrorCode::MultipleSinkDrivers,
                              llvm::Twine("covered graph ") +
                                  llvm::Twine(graphId) + " actor " +
                                  llvm::Twine(actor.id.value()) +
                                  " input port " + llvm::Twine(index) +
                                  " has " + llvm::Twine(count) + " drivers");
      }
    }

    for (std::size_t index = 0; index < graph.outputPorts.size(); ++index) {
      const EndpointKey sink{false, graphId, PortDirection::Output,
                             static_cast<std::uint32_t>(index)};
      const std::size_t count = driverCounts[sink];
      if (count == 0)
        return mappingError(MappingErrorCode::MissingSinkDriver,
                            llvm::Twine("covered graph ") +
                                llvm::Twine(graphId) + " output port " +
                                llvm::Twine(index) + " has no driver");
      if (count != 1)
        return mappingError(MappingErrorCode::MultipleSinkDrivers,
                            llvm::Twine("covered graph ") +
                                llvm::Twine(graphId) + " output port " +
                                llvm::Twine(index) + " has " +
                                llvm::Twine(count) + " drivers");
    }
  }

  return llvm::Error::success();
}

} // namespace

llvm::Expected<ValidatedTechMappingStructure>
loom::mapping::validateTechMappingStructure(const TechMappingDraft &mapping,
                                            const DataflowProgramView &dataflow,
                                            const FabricHardwareView &fabric) {
  if (mapping.header.schemaVersion != supportedSchemaVersion)
    return mappingError(MappingErrorCode::UnsupportedSchemaVersion,
                        "Mapping structural validator supports schema 1.0");
  if (mapping.header.profile != MappingProfile::TechMapping)
    return mappingError(
        MappingErrorCode::WrongMappingProfile,
        "Mapping structural validator requires the TechMapping profile");
  if (mapping.header.dataflowIdentity.empty() ||
      mapping.header.fabricIdentity.empty() || dataflow.identity.empty() ||
      fabric.identity.empty())
    return mappingError(MappingErrorCode::InvalidArtifactIdentity,
                        "Mapping inputs require non-empty artifact identities");
  if (mapping.header.dataflowIdentity != dataflow.identity ||
      mapping.header.fabricIdentity != fabric.identity)
    return mappingError(MappingErrorCode::ArtifactIdentityMismatch,
                        "Mapping draft inputs do not match its exact header");

  auto dataflowIndex = buildDataflowIndex(dataflow);
  if (!dataflowIndex)
    return dataflowIndex.takeError();
  auto fabricIndex = buildFabricIndex(fabric);
  if (!fabricIndex)
    return fabricIndex.takeError();

  EntityKinds mappingEntities;
  for (const StructuralRealizationDraft &realization : mapping.realizations) {
    if (llvm::Error error = addEntity(mappingEntities, realization.id.value(),
                                      EntityKind::StructuralRealization))
      return std::move(error);
  }

  if (mapping.coveredGraphs.empty())
    return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                        "Mapping draft declares no covered graph");
  std::set<std::uint64_t> coveredGraphs;
  for (const GraphRef &graphReference : mapping.coveredGraphs) {
    auto graph = resolveReference(graphReference, dataflow.identity,
                                  dataflowIndex->kinds, EntityKind::Graph,
                                  dataflowIndex->graphs);
    if (!graph)
      return graph.takeError();
    if (!coveredGraphs.insert((*graph)->id.value()).second)
      return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                          "Mapping draft repeats a covered graph");
  }

  if (llvm::Error error =
          validateCoveredSinkAccounting(*dataflowIndex, coveredGraphs))
    return std::move(error);

  std::map<std::uint64_t, std::size_t> actorToRealization;
  auto realizations = resolveRealizationActors(
      mapping, dataflow, *dataflowIndex, coveredGraphs, actorToRealization);
  if (!realizations)
    return realizations.takeError();

  std::set<std::uint64_t> expectedActors;
  for (const ActorDescriptor &actor : dataflow.actors) {
    if (coveredGraphs.count(actor.graph.value()))
      expectedActors.insert(actor.id.value());
  }
  std::set<std::uint64_t> realizedActors;
  for (const auto &entry : actorToRealization)
    realizedActors.insert(entry.first);
  if (realizedActors != expectedActors)
    return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                        "declared graphs do not have closed actor coverage");

  for (const RealizationActors &realization : *realizations) {
    auto selectedFu = validateActorToOpCorrespondence(
        realization, dataflow, *dataflowIndex, fabric, *fabricIndex);
    if (!selectedFu)
      return selectedFu.takeError();
    if (llvm::Error error = validateBoundaryCorrespondence(
            realization, **selectedFu, dataflow, *dataflowIndex, fabric,
            *fabricIndex))
      return std::move(error);
  }

  if (llvm::Error error = validateCoveredEdgeAccounting(
          *dataflowIndex, coveredGraphs, actorToRealization))
    return std::move(error);

  return ValidatedTechMappingStructure(mapping);
}
