#include "Mapping/Verifier.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "FabricOccurrenceIndex.h"
#include "MemoryRealizationProjection.h"
#include "VerifierInternal.h"
#include "VerifierState.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>
using namespace loom::mapping;
using namespace loom::mapping::detail;
char MappingError::ID;
void MappingError::log(llvm::raw_ostream &stream) const { stream << message_; }
std::error_code MappingError::convertToErrorCode() const {
  return std::make_error_code(std::errc::invalid_argument);
}
llvm::Error loom::mapping::detail::mappingError(MappingErrorCode code,
                                                const llvm::Twine &message) {
  return llvm::make_error<MappingError>(code, message.str());
}
llvm::Error loom::mapping::detail::addEntity(EntityKinds &entities,
                                             std::uint64_t id,
                                             EntityKind kind) {
  if (!entities.emplace(id, kind).second)
    return mappingError(MappingErrorCode::DuplicateEntityId,
                        "duplicate local entity ID");
  return llvm::Error::success();
}
llvm::Error loom::mapping::detail::requireLocalKind(const EntityKinds &entities,
                                                    std::uint64_t id,
                                                    EntityKind expected) {
  const auto entity = entities.find(id);
  if (entity == entities.end() || entity->second != expected)
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "semantic view contains an invalid local reference");
  return llvm::Error::success();
}
bool loom::mapping::detail::physicalPortAdmitsSemanticPort(
    const PortDescriptor &semantic, const PortDescriptor &physical) {
  return semantic.kind == physical.kind && semantic.type == physical.type &&
         semantic.role == physical.role &&
         semantic.payloadWidthBits <= physical.payloadWidthBits &&
         semantic.tagWidthBits <= physical.tagWidthBits;
}
bool loom::mapping::detail::physicalPortsMayConnect(
    const PortDescriptor &source, const PortDescriptor &destination) {
  if (source.kind != destination.kind || source.type != destination.type ||
      source.role != destination.role)
    return false;
  if (source.kind == PortKind::Memory)
    return source.payloadWidthBits == destination.payloadWidthBits &&
           source.tagWidthBits == destination.tagWidthBits;
  return (source.tagWidthBits == 0) == (destination.tagWidthBits == 0);
}
namespace {
const std::set<MemoryAccessPortRole> &
requiredMemoryAccessRoles(MemoryOperationKind operation) {
  static const std::set<MemoryAccessPortRole> loadRoles{
      MemoryAccessPortRole::Address, MemoryAccessPortRole::Control,
      MemoryAccessPortRole::Result, MemoryAccessPortRole::Done};
  static const std::set<MemoryAccessPortRole> storeRoles{
      MemoryAccessPortRole::Address, MemoryAccessPortRole::Data,
      MemoryAccessPortRole::Control, MemoryAccessPortRole::Done};
  return operation == MemoryOperationKind::Load ? loadRoles : storeRoles;
}
bool isAllowedMemoryAccessRole(MemoryOperationKind operation,
                               MemoryAccessPortRole role) {
  return requiredMemoryAccessRoles(operation).count(role) ||
         role == MemoryAccessPortRole::Mask;
}
PortDirection memoryAccessRoleDirection(MemoryAccessPortRole role) {
  switch (role) {
  case MemoryAccessPortRole::Address:
  case MemoryAccessPortRole::Data:
  case MemoryAccessPortRole::Mask:
  case MemoryAccessPortRole::Control:
    return PortDirection::Input;
  case MemoryAccessPortRole::Result:
  case MemoryAccessPortRole::Done:
    return PortDirection::Output;
  }
  llvm_unreachable("unknown memory access port role");
}
llvm::Expected<MemoryActorInfo>
validateMemoryActorView(const ActorDescriptor &actor) {
  const CanonicalMemoryActorView &memory = *actor.memory;
  if (memory.accessWidthBits == 0 || memory.accessSizeBytes == 0 ||
      memory.alignmentBytes == 0 ||
      memory.accessWidthBits !=
          static_cast<std::uint64_t>(memory.accessSizeBytes) * 8)
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory actor has an invalid one-beat access shape");
  std::map<ActorPortKey, MemoryAccessPortRole> ports;
  std::set<MemoryAccessPortRole> roles;
  for (const MemoryAccessPortDescriptor &port : memory.ports) {
    const auto &descriptors = port.direction == PortDirection::Input
                                  ? actor.inputPorts
                                  : actor.outputPorts;
    if (port.index >= descriptors.size() ||
        memoryAccessRoleDirection(port.role) != port.direction ||
        !isAllowedMemoryAccessRole(memory.operation, port.role) ||
        !ports.emplace(ActorPortKey{port.direction, port.index}, port.role)
             .second ||
        !roles.insert(port.role).second)
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "memory actor has invalid or duplicate port semantics");
  }
  const std::set<MemoryAccessPortRole> &required =
      requiredMemoryAccessRoles(memory.operation);
  if (ports.size() != actor.inputPorts.size() + actor.outputPorts.size() ||
      !std::includes(roles.begin(), roles.end(), required.begin(),
                     required.end()))
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory actor port semantics are incomplete");
  return MemoryActorInfo{&memory, std::move(ports)};
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
  DataflowIndex index;
  std::set<std::uint32_t> pointerAddressSpaces;
  for (const ::loom::PointerLayout &layout : dataflow.pointerLayouts) {
    if (layout.representationBits == 0 || layout.addressBits == 0 ||
        layout.addressBits > layout.representationBits ||
        !pointerAddressSpaces.insert(layout.addressSpace).second)
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "Dataflow pointer-layout projection is malformed or duplicated");
  }
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
    if (actor.semantics.type.getNumInputs() != actor.inputPorts.size() ||
        actor.semantics.type.getNumResults() != actor.outputPorts.size())
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "actor port inventory disagrees with its canonical projection");
    auto semanticBytes =
        ::dataflow::encodeCanonicalActorSchemaProjection(actor.semantics);
    if (!semanticBytes)
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          llvm::Twine("actor has an invalid canonical projection: ") +
              llvm::toString(semanticBytes.takeError()));
    index.actors.emplace(actor.id.value(), &actor);
  }
  for (const LogicalMemoryRootDescriptor &root : dataflow.logicalMemoryRoots) {
    if (llvm::Error error = addEntity(index.kinds, root.id.value(),
                                      EntityKind::LogicalMemoryRoot))
      return std::move(error);
    index.logicalMemoryRoots.emplace(root.id.value(), &root);
  }
  std::size_t graphMemoryPortCount = 0;
  for (const GraphDescriptor &graph : dataflow.graphs) {
    graphMemoryPortCount +=
        llvm::count_if(graph.inputPorts, [](const PortDescriptor &port) {
          return port.kind == PortKind::Memory;
        });
    graphMemoryPortCount +=
        llvm::count_if(graph.outputPorts, [](const PortDescriptor &port) {
          return port.kind == PortKind::Memory;
        });
  }
  std::set<EndpointKey> rootedMemoryPorts;
  for (const LogicalMemoryRootDescriptor &root : dataflow.logicalMemoryRoots) {
    if (llvm::Error error = requireLocalKind(index.kinds, root.graph.value(),
                                             EntityKind::Graph))
      return std::move(error);
    auto addRootPort = [&](const GraphPort &port,
                           PortDirection direction) -> llvm::Error {
      auto resolved = resolveDataflowPort(DataflowEndpoint{port}, index);
      if (!resolved)
        return resolved.takeError();
      if (port.graph != root.graph || port.direction != direction ||
          resolved->descriptor->kind != PortKind::Memory ||
          !rootedMemoryPorts.insert(resolved->key).second)
        return mappingError(
            MappingErrorCode::InvalidPortConnection,
            "logical memory root has an invalid graph boundary port");
      return llvm::Error::success();
    };
    for (const GraphPort &port : root.importPorts)
      if (llvm::Error error = addRootPort(port, PortDirection::Input))
        return std::move(error);
    for (const GraphPort &port : root.exportPorts)
      if (llvm::Error error = addRootPort(port, PortDirection::Output))
        return std::move(error);
  }
  if (rootedMemoryPorts.size() != graphMemoryPortCount)
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "graph memory ports are not exactly rooted");
  for (const ActorDescriptor &actor : dataflow.actors) {
    if (llvm::Error error = requireLocalKind(index.kinds, actor.graph.value(),
                                             EntityKind::Graph))
      return std::move(error);
    if (!actor.memory)
      continue;
    if (llvm::Error error =
            requireLocalKind(index.kinds, actor.memory->root.value(),
                             EntityKind::LogicalMemoryRoot))
      return std::move(error);
    if (index.logicalMemoryRoots.at(actor.memory->root.value())->graph !=
        actor.graph)
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "memory actor uses a root from another graph");
    auto memoryActor = validateMemoryActorView(actor);
    if (!memoryActor)
      return memoryActor.takeError();
    index.memoryActors.emplace(actor.id.value(), std::move(*memoryActor));
  }
  index.edges.reserve(dataflow.edges.size());
  for (const DataflowEdge &edge : dataflow.edges) {
    auto source = resolveDataflowPort(edge.source, index);
    if (!source)
      return source.takeError();
    auto target = resolveDataflowPort(edge.target, index);
    if (!target)
      return target.takeError();
    if (source->descriptor->kind == PortKind::Memory ||
        target->descriptor->kind == PortKind::Memory)
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "memory capability ports cannot participate in dataflow edges");
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
    if (!index.edgesByKey.emplace(edgeKey, index.edges.size()).second)
      return mappingError(MappingErrorCode::DuplicateEdge,
                          "dataflow edge is duplicated");
    index.edges.push_back(ResolvedDataflowEdge{*source, *target});
  }
  return index;
}
llvm::Error validateMemoryOperationPortTemplate(
    const MemoryOperationPortTemplateDescriptor &operation) {
  std::set<MemoryAccessPortRole> roles;
  for (const MemoryOperationPortDescriptor &port : operation.ports) {
    if (memoryAccessRoleDirection(port.role) != port.direction ||
        !isAllowedMemoryAccessRole(operation.operation, port.role) ||
        !roles.insert(port.role).second ||
        (port.direction == PortDirection::Input && port.maxInternalFanout != 0))
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "memory operation template has invalid or duplicate ports");
  }
  const std::set<MemoryAccessPortRole> &required =
      requiredMemoryAccessRoles(operation.operation);
  if (!std::includes(roles.begin(), roles.end(), required.begin(),
                     required.end()))
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory operation template is missing a required port");
  if (operation.physicalDataWidthBits == 0 ||
      operation.accessCapabilities.empty())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory operation template has no access contract");
  std::set<std::uint32_t> accessSizes;
  for (const MemoryAccessCapability &capability :
       operation.accessCapabilities) {
    if (capability.accessSizeBytes == 0 ||
        capability.requiredAlignmentBytes == 0 ||
        static_cast<std::uint64_t>(capability.accessSizeBytes) * 8 >
            operation.physicalDataWidthBits ||
        !accessSizes.insert(capability.accessSizeBytes).second)
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "memory operation template has an invalid access capability");
  }
  return llvm::Error::success();
}
struct LocalMemoryOperationPortInfo {
  const MemoryOperationPortTemplateDescriptor *operation;
  const MemoryOperationPortDescriptor *port;
};
llvm::Expected<LocalMemoryOperationPortInfo>
resolveLocalMemoryOperationPort(const MemoryOperationPort &reference,
                                const FabricIndex &index) {
  const auto operation =
      index.memoryOperationPortTemplates.find(reference.operation.value());
  if (operation == index.memoryOperationPortTemplates.end() ||
      reference.index >= operation->second->ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "Fabric memory connectivity has an invalid port");
  return LocalMemoryOperationPortInfo{
      operation->second, &operation->second->ports[reference.index]};
}
struct MemoryInternalEndpointKey {
  bool boundary;
  std::uint64_t owner;
  std::uint32_t index;
  friend bool operator==(const MemoryInternalEndpointKey &lhs,
                         const MemoryInternalEndpointKey &rhs) {
    return lhs.boundary == rhs.boundary && lhs.owner == rhs.owner &&
           lhs.index == rhs.index;
  }
  friend bool operator!=(const MemoryInternalEndpointKey &lhs,
                         const MemoryInternalEndpointKey &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const MemoryInternalEndpointKey &lhs,
                        const MemoryInternalEndpointKey &rhs) {
    return std::tie(lhs.boundary, lhs.owner, lhs.index) <
           std::tie(rhs.boundary, rhs.owner, rhs.index);
  }
};
struct LocalMemoryInternalEndpointInfo {
  MemoryInternalEndpointKey key;
  const PortDescriptor *port;
  std::uint32_t maxFanout;
  bool source;
  const MemoryOperationPortTemplateDescriptor *operation;
};
llvm::Expected<LocalMemoryInternalEndpointInfo>
resolveLocalMemoryInternalEndpoint(const MemoryInternalEndpoint &endpoint,
                                   MemoryImplementationId implementation,
                                   const FabricIndex &index) {
  if (const auto *boundary =
          std::get_if<MemoryImplementationBoundaryPort>(&endpoint)) {
    const auto memory =
        index.memoryImplementations.find(implementation.value());
    if (memory == index.memoryImplementations.end() ||
        boundary->index >= memory->second->boundaryPorts.size())
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "memory connectivity has an invalid boundary port");
    const auto &port = memory->second->boundaryPorts[boundary->index];
    return LocalMemoryInternalEndpointInfo{
        {true, implementation.value(), boundary->index},
        &port.port,
        port.maxInternalFanout,
        port.direction == PortDirection::Input,
        nullptr};
  }
  const auto &operationPort = std::get<MemoryOperationPort>(endpoint);
  auto port = resolveLocalMemoryOperationPort(operationPort, index);
  if (!port)
    return port.takeError();
  if (port->operation->implementation != implementation)
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory connectivity crosses implementations");
  return LocalMemoryInternalEndpointInfo{
      {false, operationPort.operation.value(), operationPort.index},
      &port->port->port,
      port->port->maxInternalFanout,
      port->port->direction == PortDirection::Output,
      port->operation};
}
llvm::Expected<FabricIndex> buildFabricIndex(const FabricHardwareView &fabric) {
  FabricIndex index;
  for (const FuDescriptor &fu : fabric.functionalUnits) {
    if (llvm::Error error = addEntity(index.kinds, fu.id.id(), EntityKind::Fu))
      return std::move(error);
    auto normalized =
        ::loom::fabric::normalizeFabricFuCapabilityTemplateInventory(
            fu.capabilityTemplates);
    if (!normalized)
      return mappingError(
          MappingErrorCode::InvalidCapabilityTemplateReference,
          llvm::Twine("invalid Fabric FU capability-template inventory: ") +
              llvm::toString(normalized.takeError()));
    if (*normalized != fu.capabilityTemplates)
      return mappingError(
          MappingErrorCode::InvalidCapabilityTemplateReference,
          "Fabric FU capability-template inventory is not canonical");
    index.functionalUnits.emplace(fu.id.id(), &fu);
  }
  for (const FabricOpDescriptor &operation : fabric.operations) {
    if (operation.id.node != ::loom::fabric::FabricFuNodeKind::Op ||
        !index.functionalUnits.count(operation.id.fu.id()) ||
        static_cast<std::uint32_t>(operation.family) >=
            ::fabric::implementationFamilyCount() ||
        ::fabric::capabilityParamsSchema(operation.capability) !=
            ::fabric::implementationFamily(operation.family)
                .capabilityParamsSchema ||
        !index.operations.emplace(nodeKey(operation.id), &operation).second)
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "Fabric operation inventory has an invalid or duplicate owner");
    if (operation.enabledOperationSchemas.empty())
      return mappingError(MappingErrorCode::CapabilityTemplateMismatch,
                          "Fabric operation has no enabled operation schema");
    std::optional<::dataflow::OperationSchemaId> previous;
    for (::dataflow::OperationSchemaId schema :
         operation.enabledOperationSchemas) {
      if (static_cast<std::uint32_t>(schema) >=
              ::dataflow::operationSchemaCount() ||
          !::fabric::admitsOperationSchema(operation.family, schema) ||
          (previous && static_cast<std::uint32_t>(*previous) >=
                           static_cast<std::uint32_t>(schema)))
        return mappingError(
            MappingErrorCode::CapabilityTemplateMismatch,
            "Fabric operation enabled-schema inventory is not canonical");
      previous = schema;
    }
  }
  for (const MemoryServiceDomainDescriptor &service :
       fabric.memoryServiceDomains) {
    if (llvm::Error error = addEntity(index.kinds, service.id.value(),
                                      EntityKind::MemoryServiceDomain))
      return std::move(error);
    index.memoryServiceDomains.emplace(service.id.value(), &service);
  }
  for (const MemoryImplementationDescriptor &implementation :
       fabric.memoryImplementations) {
    if (llvm::Error error = addEntity(index.kinds, implementation.id.value(),
                                      EntityKind::MemoryImplementation))
      return std::move(error);
    index.memoryImplementations.emplace(implementation.id.value(),
                                        &implementation);
  }
  for (const MemoryOperationPortTemplateDescriptor &operation :
       fabric.memoryOperationPortTemplates) {
    if (llvm::Error error = addEntity(index.kinds, operation.id.value(),
                                      EntityKind::MemoryOperationPortTemplate))
      return std::move(error);
    index.memoryOperationPortTemplates.emplace(operation.id.value(),
                                               &operation);
  }
  for (const MemoryInternalConnectionDescriptor &connection :
       fabric.memoryInternalConnections) {
    if (llvm::Error error = addEntity(index.kinds, connection.id.value(),
                                      EntityKind::MemoryInternalConnection))
      return std::move(error);
    index.memoryInternalConnections.emplace(connection.id.value(), &connection);
  }
  for (const MemorySemanticEncodingDescriptor &encoding :
       fabric.memorySemanticEncodings) {
    if (llvm::Error error = addEntity(index.kinds, encoding.id.value(),
                                      EntityKind::MemorySemanticEncoding))
      return std::move(error);
    index.memorySemanticEncodings.emplace(encoding.id.value(), &encoding);
  }
  auto projection =
      buildValidatedFabricProjection(fabric, index.kinds, index.functionalUnits,
                                     index.memoryOperationPortTemplates);
  if (!projection)
    return projection.takeError();
  auto routing =
      buildValidatedFabricRoutingProjection(fabric, index.kinds, **projection);
  if (!routing)
    return routing.takeError();
  (*projection)->routing = std::move(*routing);
  index.projection =
      std::shared_ptr<const ValidatedFabricProjection>(std::move(*projection));
  for (const MemoryImplementationDescriptor &implementation :
       fabric.memoryImplementations) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, implementation.service.value(),
                             EntityKind::MemoryServiceDomain))
      return std::move(error);
    for (const MemoryImplementationBoundaryPortDescriptor &port :
         implementation.boundaryPorts)
      if (port.direction == PortDirection::Output &&
          port.maxInternalFanout != 0)
        return mappingError(
            MappingErrorCode::InvalidPortConnection,
            "memory boundary sink declares internal fanout capacity");
  }
  for (const MemoryOperationPortTemplateDescriptor &operation :
       fabric.memoryOperationPortTemplates) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, operation.implementation.value(),
                             EntityKind::MemoryImplementation))
      return std::move(error);
    if (llvm::Error error = validateMemoryOperationPortTemplate(operation))
      return std::move(error);
  }
  for (const MemoryInternalConnectionDescriptor &connection :
       fabric.memoryInternalConnections) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, connection.implementation.value(),
                             EntityKind::MemoryImplementation))
      return std::move(error);
    auto source = resolveLocalMemoryInternalEndpoint(
        connection.source, connection.implementation, index);
    if (!source)
      return source.takeError();
    auto sink = resolveLocalMemoryInternalEndpoint(
        connection.sink, connection.implementation, index);
    if (!sink)
      return sink.takeError();
    if (!source->source || sink->source ||
        !physicalPortsMayConnect(*source->port, *sink->port) ||
        source->maxFanout == 0 || (source->key.boundary && sink->key.boundary))
      return mappingError(
          MappingErrorCode::InvalidPortConnection,
          "memory internal connectivity has incompatible endpoints");
  }

  for (const MemorySemanticEncodingDescriptor &encoding :
       fabric.memorySemanticEncodings) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, encoding.implementation.value(),
                             EntityKind::MemoryImplementation))
      return std::move(error);
    std::set<std::uint64_t> operationTemplates;
    for (MemoryOperationPortTemplateId operationId :
         encoding.operationTemplates) {
      if (llvm::Error error =
              requireLocalKind(index.kinds, operationId.value(),
                               EntityKind::MemoryOperationPortTemplate))
        return std::move(error);
      const auto *operation =
          index.memoryOperationPortTemplates.at(operationId.value());
      if (operation->implementation != encoding.implementation ||
          !operationTemplates.insert(operationId.value()).second)
        return mappingError(
            MappingErrorCode::InvalidPortConnection,
            "memory encoding has an invalid operation template");
    }
    if (operationTemplates.empty())
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "memory encoding has no operation template");

    std::set<std::uint64_t> internalConnections;
    std::map<MemoryInternalEndpointKey, std::size_t> selectedFanout;
    for (MemoryInternalConnectionId connectionId :
         encoding.internalConnections) {
      if (llvm::Error error =
              requireLocalKind(index.kinds, connectionId.value(),
                               EntityKind::MemoryInternalConnection))
        return std::move(error);
      const auto *connection =
          index.memoryInternalConnections.at(connectionId.value());
      auto source = resolveLocalMemoryInternalEndpoint(
          connection->source, encoding.implementation, index);
      if (!source)
        return source.takeError();
      auto sink = resolveLocalMemoryInternalEndpoint(
          connection->sink, encoding.implementation, index);
      if (!sink)
        return sink.takeError();
      const bool sourceSelected =
          !source->operation ||
          operationTemplates.count(source->operation->id.value());
      const bool sinkSelected =
          !sink->operation ||
          operationTemplates.count(sink->operation->id.value());
      if (connection->implementation != encoding.implementation ||
          !sourceSelected || !sinkSelected ||
          !internalConnections.insert(connectionId.value()).second ||
          ++selectedFanout[source->key] > source->maxFanout)
        return mappingError(
            MappingErrorCode::InvalidPortConnection,
            "memory encoding has an invalid internal connection");
    }
  }

  return index;
}
} // namespace

llvm::Expected<DataflowPortInfo>
loom::mapping::detail::resolveActorPortReference(
    const ActorPortRef &port, const ArtifactIdentity &artifact,
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

namespace {
llvm::Expected<DataflowPortInfo>
resolveGraphPortReference(const GraphPortRef &port,
                          const ArtifactIdentity &artifact,
                          const DataflowIndex &index) {
  auto graph = resolveReference(port.graph, artifact, index.kinds,
                                EntityKind::Graph, index.graphs);
  if (!graph)
    return graph.takeError();
  const auto &ports = port.direction == PortDirection::Input
                          ? (*graph)->inputPorts
                          : (*graph)->outputPorts;
  if (port.index >= ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "graph boundary port index is out of range");
  return DataflowPortInfo{
      (*graph)->id.value(),
      EndpointKey{false, (*graph)->id.value(), port.direction, port.index},
      &ports[port.index]};
}

struct MemoryImplementationBoundaryPortInfo {
  const MemoryImplementationDescriptor *implementation;
  std::uint32_t index;
  const MemoryImplementationBoundaryPortDescriptor *descriptor;
};

llvm::Expected<MemoryImplementationBoundaryPortInfo>
resolveMemoryImplementationBoundaryPortReference(
    const MemoryImplementationBoundaryPortRef &port,
    const ArtifactIdentity &artifact, const FabricIndex &index) {
  auto implementation = resolveReference(
      port.implementation, artifact, index.kinds,
      EntityKind::MemoryImplementation, index.memoryImplementations);
  if (!implementation)
    return implementation.takeError();
  if (port.index >= (*implementation)->boundaryPorts.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory boundary port index is out of range");
  return MemoryImplementationBoundaryPortInfo{
      *implementation, port.index,
      &(*implementation)->boundaryPorts[port.index]};
}

struct MemoryOperationPortInfo {
  const MemoryOperationPortTemplateDescriptor *operation;
  std::uint32_t index;
  const MemoryOperationPortDescriptor *descriptor;
};

llvm::Expected<MemoryOperationPortInfo>
resolveMemoryOperationPortReference(const MemoryOperationPortRef &port,
                                    const ArtifactIdentity &artifact,
                                    const FabricIndex &index) {
  auto operation = resolveReference(port.operation, artifact, index.kinds,
                                    EntityKind::MemoryOperationPortTemplate,
                                    index.memoryOperationPortTemplates);
  if (!operation)
    return operation.takeError();
  if (port.index >= (*operation)->ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "memory operation port index is out of range");
  return MemoryOperationPortInfo{*operation, port.index,
                                 &(*operation)->ports[port.index]};
}

llvm::Expected<const ResolvedDataflowEdge *>
resolveEdgeReference(const DataflowEdgeRef &reference,
                     const DataflowProgramView &dataflow,
                     const DataflowIndex &index) {
  if (reference.artifact != dataflow.identity)
    return mappingError(MappingErrorCode::ForeignReference,
                        "reference names a foreign artifact");
  auto source = resolveDataflowPort(reference.edge.source, index);
  if (!source)
    return source.takeError();
  auto target = resolveDataflowPort(reference.edge.target, index);
  if (!target)
    return target.takeError();
  const auto edge = index.edgesByKey.find(EdgeKey{source->key, target->key});
  if (edge == index.edgesByKey.end())
    return mappingError(MappingErrorCode::UnresolvedEdgeReference,
                        "reference names a non-canonical endpoint pair");
  return &index.edges[edge->second];
}

llvm::Expected<std::vector<RealizationActors>> resolveRealizationActors(
    const TechMappingDraft &mapping, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex,
    const std::set<std::uint64_t> &coveredGraphs,
    std::map<std::uint64_t, std::size_t> &actorToRealization) {
  std::vector<RealizationActors> resolved;
  resolved.reserve(mapping.realizations.size());

  for (const ComputeRealizationDraft &realization : mapping.realizations) {
    if (realization.actorToOps.empty())
      return mappingError(MappingErrorCode::EmptyActorGroup,
                          "Compute Realization actor group is empty");

    RealizationActors actors{&realization, 0, {}};
    bool firstActor = true;
    for (const ActorToFabricOp &binding : realization.actorToOps) {
      auto actor = resolveReference(binding.actor, dataflow.identity,
                                    dataflowIndex.kinds, EntityKind::Actor,
                                    dataflowIndex.actors);
      if (!actor)
        return actor.takeError();
      if ((*actor)->memory)
        return mappingError(MappingErrorCode::WrongActorRealizationKind,
                            "Compute Realization covers a memory actor");
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
                            "Compute Realization crosses graphs");
      }
    }
    if (!coveredGraphs.count(actors.graph))
      return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                          "Compute Realization uses an uncovered graph");
    resolved.push_back(std::move(actors));
  }

  return resolved;
}

struct MemoryRealizationActors {
  const MemoryRealizationDraft *record;
  std::uint64_t graph;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
};

llvm::Expected<std::vector<MemoryRealizationActors>>
resolveMemoryRealizationActors(
    const TechMappingDraft &mapping, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex,
    const std::set<std::uint64_t> &coveredGraphs,
    std::map<std::uint64_t, std::size_t> &actorToRealization) {
  std::vector<MemoryRealizationActors> resolved;
  resolved.reserve(mapping.memoryRealizations.size());

  for (const MemoryRealizationDraft &realization : mapping.memoryRealizations) {
    if (realization.actors.empty())
      return mappingError(MappingErrorCode::InvalidMemoryRealization,
                          "Memory Realization actor group is empty");

    MemoryRealizationActors actors{&realization, 0, {}};
    bool firstActor = true;
    for (const ActorRef &actorReference : realization.actors) {
      auto actor = resolveReference(actorReference, dataflow.identity,
                                    dataflowIndex.kinds, EntityKind::Actor,
                                    dataflowIndex.actors);
      if (!actor)
        return actor.takeError();
      if (!(*actor)->memory)
        return mappingError(MappingErrorCode::WrongActorRealizationKind,
                            "Memory Realization covers a compute actor");
      const std::size_t realizationIndex =
          mapping.realizations.size() + resolved.size();
      if (!actors.actors.emplace((*actor)->id.value(), *actor).second ||
          !actorToRealization.emplace((*actor)->id.value(), realizationIndex)
               .second)
        return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                            "actor is covered more than once");
      if (firstActor) {
        actors.graph = (*actor)->graph.value();
        firstActor = false;
      } else if ((*actor)->graph.value() != actors.graph) {
        return mappingError(MappingErrorCode::CrossGraphActorGroup,
                            "Memory Realization crosses graphs");
      }
    }
    if (!coveredGraphs.count(actors.graph))
      return mappingError(MappingErrorCode::IncompleteGraphCoverage,
                          "Memory Realization uses an uncovered graph");
    resolved.push_back(std::move(actors));
  }
  return resolved;
}

llvm::Error validateMemoryRealization(
    const MemoryRealizationActors &realization,
    const DataflowProgramView &dataflow, const DataflowIndex &dataflowIndex,
    const FabricHardwareView &fabric, const FabricIndex &fabricIndex,
    std::map<std::uint64_t, std::uint64_t> &rootServices) {
  auto encoding = resolveReference(
      realization.record->encoding, fabric.identity, fabricIndex.kinds,
      EntityKind::MemorySemanticEncoding, fabricIndex.memorySemanticEncodings);
  if (!encoding)
    return encoding.takeError();
  const MemoryImplementationDescriptor &implementation =
      *fabricIndex.memoryImplementations.at(
          (*encoding)->implementation.value());

  if (realization.record->actorToOperations.size() != realization.actors.size())
    return mappingError(
        MappingErrorCode::InvalidMemoryRealization,
        "actor-to-memory-operation correspondence is not complete");

  std::set<std::uint64_t> encodingOperations;
  for (MemoryOperationPortTemplateId operation :
       (*encoding)->operationTemplates)
    encodingOperations.insert(operation.value());

  std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
      actorOperations;
  std::map<EndpointKey, MemoryInternalEndpointKey> actorInternalEndpoints;
  std::set<std::uint64_t> mappedActors;
  std::set<std::uint64_t> mappedOperations;
  std::set<std::uint64_t> usedRoots;
  for (const ActorToMemoryOperation &correspondence :
       realization.record->actorToOperations) {
    auto actor = resolveReference(correspondence.actor, dataflow.identity,
                                  dataflowIndex.kinds, EntityKind::Actor,
                                  dataflowIndex.actors);
    if (!actor)
      return actor.takeError();
    auto root = resolveReference(
        correspondence.root, dataflow.identity, dataflowIndex.kinds,
        EntityKind::LogicalMemoryRoot, dataflowIndex.logicalMemoryRoots);
    if (!root)
      return root.takeError();
    auto operation = resolveReference(correspondence.operation, fabric.identity,
                                      fabricIndex.kinds,
                                      EntityKind::MemoryOperationPortTemplate,
                                      fabricIndex.memoryOperationPortTemplates);
    if (!operation)
      return operation.takeError();

    if (!realization.actors.count((*actor)->id.value()) || !(*actor)->memory ||
        !mappedActors.insert((*actor)->id.value()).second)
      return mappingError(
          MappingErrorCode::InvalidMemoryRealization,
          "actor-to-memory-operation correspondence is not exact");
    const MemoryActorInfo &memory =
        dataflowIndex.memoryActors.at((*actor)->id.value());
    if (memory.view->root != (*root)->id)
      return mappingError(MappingErrorCode::InvalidMemoryRealization,
                          "memory actor names the wrong logical root");
    if ((*operation)->operation != memory.view->operation)
      return mappingError(MappingErrorCode::MemoryOperationMismatch,
                          "memory operation kind does not match actor");
    if ((*operation)->implementation != implementation.id ||
        !encodingOperations.count((*operation)->id.value()))
      return mappingError(MappingErrorCode::MemoryEncodingMismatch,
                          "memory operation is not selected by encoding");

    std::map<MemoryAccessPortRole,
             std::pair<std::uint32_t, const MemoryOperationPortDescriptor *>>
        hardwarePorts;
    for (std::size_t index = 0; index < (*operation)->ports.size(); ++index) {
      const MemoryOperationPortDescriptor &port = (*operation)->ports[index];
      hardwarePorts.emplace(
          port.role, std::make_pair(static_cast<std::uint32_t>(index), &port));
    }
    if (hardwarePorts.size() != memory.ports.size())
      return mappingError(MappingErrorCode::MemoryOperationMismatch,
                          "memory operation signature does not match actor");
    for (const auto &entry : memory.ports) {
      const auto hardware = hardwarePorts.find(entry.second);
      const auto &softwarePorts = entry.first.first == PortDirection::Input
                                      ? (*actor)->inputPorts
                                      : (*actor)->outputPorts;
      if (hardware == hardwarePorts.end() ||
          hardware->second.second->direction != entry.first.first ||
          !physicalPortAdmitsSemanticPort(softwarePorts[entry.first.second],
                                          hardware->second.second->port))
        return mappingError(MappingErrorCode::MemoryOperationMismatch,
                            "memory operation signature does not match actor");
      actorInternalEndpoints.emplace(
          EndpointKey{true, (*actor)->id.value(), entry.first.first,
                      entry.first.second},
          MemoryInternalEndpointKey{false, (*operation)->id.value(),
                                    hardware->second.first});
    }

    const auto access = llvm::find_if(
        (*operation)->accessCapabilities,
        [&](const MemoryAccessCapability &capability) {
          return capability.accessSizeBytes == memory.view->accessSizeBytes;
        });
    if (memory.view->accessWidthBits > (*operation)->physicalDataWidthBits ||
        access == (*operation)->accessCapabilities.end() ||
        memory.view->alignmentBytes % access->requiredAlignmentBytes != 0)
      return mappingError(MappingErrorCode::MemoryAccessIncompatible,
                          "memory access is not one-beat compatible");

    actorOperations.emplace((*actor)->id.value(), *operation);
    if (!mappedOperations.insert((*operation)->id.value()).second)
      return mappingError(
          MappingErrorCode::InvalidMemoryRealization,
          "memory operation template is assigned to multiple actors");
    usedRoots.insert((*root)->id.value());
  }

  if (mappedActors.size() != realization.actors.size() ||
      mappedOperations != encodingOperations)
    return mappingError(MappingErrorCode::InvalidMemoryRealization,
                        "memory encoding operations are not exactly covered");

  std::set<std::uint64_t> declaredRoots;
  for (const LogicalMemoryRootRef &rootReference : realization.record->roots) {
    auto root = resolveReference(
        rootReference, dataflow.identity, dataflowIndex.kinds,
        EntityKind::LogicalMemoryRoot, dataflowIndex.logicalMemoryRoots);
    if (!root)
      return root.takeError();
    if (!declaredRoots.insert((*root)->id.value()).second)
      return mappingError(MappingErrorCode::InvalidMemoryRealization,
                          "Memory Realization repeats a logical root");
  }
  if (declaredRoots != usedRoots)
    return mappingError(MappingErrorCode::InvalidMemoryRealization,
                        "Memory Realization root set is not exact");
  for (std::uint64_t root : declaredRoots) {
    auto [service, inserted] =
        rootServices.emplace(root, implementation.service.value());
    if (!inserted && service->second != implementation.service.value())
      return mappingError(MappingErrorCode::MemoryServiceMismatch,
                          "logical memory root uses unrelated services");
  }

  std::set<std::uint64_t> activeConnections;
  for (MemoryInternalConnectionId connection : (*encoding)->internalConnections)
    activeConnections.insert(connection.value());

  std::map<EndpointKey, MemoryInternalEndpointKey> graphInternalEndpoints;
  std::set<MemoryInternalEndpointKey> mappedImplementationBoundaries;
  for (const MemoryGraphBoundaryPortCorrespondence &correspondence :
       realization.record->graphBoundaryPorts) {
    auto graphPort = resolveGraphPortReference(
        correspondence.graphPort, dataflow.identity, dataflowIndex);
    if (!graphPort)
      return graphPort.takeError();
    auto implementationPort = resolveMemoryImplementationBoundaryPortReference(
        correspondence.implementationPort, fabric.identity, fabricIndex);
    if (!implementationPort)
      return implementationPort.takeError();
    const MemoryInternalEndpointKey endpoint{
        true, implementationPort->implementation->id.value(),
        implementationPort->index};
    if (graphPort->graph != realization.graph ||
        implementationPort->implementation->id != implementation.id ||
        graphPort->key.direction != implementationPort->descriptor->direction ||
        !physicalPortAdmitsSemanticPort(*graphPort->descriptor,
                                        implementationPort->descriptor->port) ||
        !graphInternalEndpoints.emplace(graphPort->key, endpoint).second ||
        !mappedImplementationBoundaries.insert(endpoint).second)
      return mappingError(
          MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
          "memory graph boundary correspondence is not exact");
  }

  std::set<EdgeKey> internalEdges;
  std::set<std::uint64_t> witnessedConnections;
  std::set<EndpointKey> usedGraphBoundaries;
  for (const MemoryInternalEdgeWitness &witness :
       realization.record->internalEdges) {
    auto edge = resolveEdgeReference(witness.edge, dataflow, dataflowIndex);
    if (!edge)
      return edge.takeError();
    auto connection =
        resolveReference(witness.connection, fabric.identity, fabricIndex.kinds,
                         EntityKind::MemoryInternalConnection,
                         fabricIndex.memoryInternalConnections);
    if (!connection)
      return connection.takeError();
    if ((*edge)->source.graph != realization.graph ||
        ((*edge)->source.key.actor &&
         !realization.actors.count((*edge)->source.key.owner)) ||
        ((*edge)->target.key.actor &&
         !realization.actors.count((*edge)->target.key.owner)) ||
        !internalEdges.insert(EdgeKey{(*edge)->source.key, (*edge)->target.key})
             .second ||
        !witnessedConnections.insert((*connection)->id.value()).second ||
        !activeConnections.count((*connection)->id.value()) ||
        (*connection)->implementation != implementation.id)
      return mappingError(MappingErrorCode::InvalidInternalEdgeWitness,
                          "memory internal edge witness is not selected");

    auto sourcePort = resolveLocalMemoryInternalEndpoint(
        (*connection)->source, implementation.id, fabricIndex);
    if (!sourcePort)
      return sourcePort.takeError();
    auto sinkPort = resolveLocalMemoryInternalEndpoint(
        (*connection)->sink, implementation.id, fabricIndex);
    if (!sinkPort)
      return sinkPort.takeError();

    const auto sourceExpected =
        (*edge)->source.key.actor
            ? actorInternalEndpoints.find((*edge)->source.key)
            : graphInternalEndpoints.find((*edge)->source.key);
    const auto sourceEnd = (*edge)->source.key.actor
                               ? actorInternalEndpoints.end()
                               : graphInternalEndpoints.end();
    const auto sinkExpected =
        (*edge)->target.key.actor
            ? actorInternalEndpoints.find((*edge)->target.key)
            : graphInternalEndpoints.find((*edge)->target.key);
    const auto sinkEnd = (*edge)->target.key.actor
                             ? actorInternalEndpoints.end()
                             : graphInternalEndpoints.end();
    if (sourceExpected == sourceEnd || sinkExpected == sinkEnd ||
        sourceExpected->second != sourcePort->key ||
        sinkExpected->second != sinkPort->key ||
        !physicalPortAdmitsSemanticPort(*(*edge)->source.descriptor,
                                        *sourcePort->port) ||
        !physicalPortAdmitsSemanticPort(*(*edge)->target.descriptor,
                                        *sinkPort->port))
      return mappingError(MappingErrorCode::InvalidInternalEdgeWitness,
                          "memory internal edge does not match capability");
    if (!(*edge)->source.key.actor)
      usedGraphBoundaries.insert((*edge)->source.key);
    if (!(*edge)->target.key.actor)
      usedGraphBoundaries.insert((*edge)->target.key);
  }
  std::set<EndpointKey> mappedGraphBoundaries;
  for (const auto &entry : graphInternalEndpoints)
    mappedGraphBoundaries.insert(entry.first);
  if (witnessedConnections != activeConnections ||
      mappedGraphBoundaries != usedGraphBoundaries)
    return mappingError(MappingErrorCode::InvalidInternalEdgeWitness,
                        "memory internal witness set is not exact");

  std::set<EndpointKey> expectedBoundaryPorts;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    if (edge.source.graph != realization.graph ||
        internalEdges.count(EdgeKey{edge.source.key, edge.target.key}))
      continue;
    if (edge.source.key.actor &&
        realization.actors.count(edge.source.key.owner))
      expectedBoundaryPorts.insert(edge.source.key);
    if (edge.target.key.actor &&
        realization.actors.count(edge.target.key.owner))
      expectedBoundaryPorts.insert(edge.target.key);
  }

  std::set<EndpointKey> mappedBoundaryPorts;
  for (const MemoryBoundaryPortCorrespondence &correspondence :
       realization.record->boundaryPorts) {
    auto actorPort = resolveActorPortReference(
        correspondence.actorPort, dataflow.identity, dataflowIndex);
    if (!actorPort)
      return actorPort.takeError();
    auto operationPort = resolveMemoryOperationPortReference(
        correspondence.operationPort, fabric.identity, fabricIndex);
    if (!operationPort)
      return operationPort.takeError();
    const ActorPortKey actorPortKey{actorPort->key.direction,
                                    actorPort->key.index};
    if (!realization.actors.count(actorPort->key.owner) ||
        !expectedBoundaryPorts.count(actorPort->key) ||
        !mappedBoundaryPorts.insert(actorPort->key).second)
      return mappingError(
          MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
          "memory boundary correspondence is not exact");
    const auto &memory = dataflowIndex.memoryActors.at(actorPort->key.owner);
    const auto role = memory.ports.find(actorPortKey);
    if (role == memory.ports.end() ||
        actorOperations.at(actorPort->key.owner)->id !=
            operationPort->operation->id)
      return mappingError(
          MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
          "memory boundary correspondence is not exact");
    if (role->second != operationPort->descriptor->role ||
        actorPort->key.direction != operationPort->descriptor->direction)
      return mappingError(
          MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
          "memory boundary correspondence uses the wrong role");
    if (!physicalPortAdmitsSemanticPort(*actorPort->descriptor,
                                        operationPort->descriptor->port))
      return mappingError(MappingErrorCode::PortSignatureMismatch,
                          "memory boundary port type does not match");
  }
  if (mappedBoundaryPorts != expectedBoundaryPorts)
    return mappingError(
        MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
        "memory boundary correspondence is not complete");
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
      if (graph.outputPorts[index].kind == PortKind::Memory)
        continue;
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

llvm::Expected<ValidatedTechMapping> loom::mapping::validateTechMapping(
    ArtifactIdentity identity, const TechMappingDraft &mapping,
    const DataflowProgramView &dataflow, const FabricHardwareView &fabric) {
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
  for (const ComputeRealizationDraft &realization : mapping.realizations) {
    if (llvm::Error error = addEntity(mappingEntities, realization.id.value(),
                                      EntityKind::ComputeRealization))
      return std::move(error);
  }
  for (const MemoryRealizationDraft &realization : mapping.memoryRealizations) {
    if (llvm::Error error = addEntity(mappingEntities, realization.id.value(),
                                      EntityKind::MemoryRealization))
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
  auto memoryRealizations = resolveMemoryRealizationActors(
      mapping, dataflow, *dataflowIndex, coveredGraphs, actorToRealization);
  if (!memoryRealizations)
    return memoryRealizations.takeError();

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

  auto mappingProjection = std::make_shared<ValidatedTechMappingProjection>();
  mappingProjection->computeRealizations.reserve(realizations->size());
  for (const RealizationActors &realization : *realizations) {
    auto activeBoundaryPorts = verifyComputeRealization(
        realization, dataflow, *dataflowIndex, *fabricIndex);
    if (!activeBoundaryPorts)
      return activeBoundaryPorts.takeError();
    mappingProjection->computeRealizations.push_back(
        ValidatedComputeRealizationProjection{
            realization.record->id, realization.record->capabilityTemplate,
            std::move(*activeBoundaryPorts)});
  }
  std::map<std::uint64_t, std::uint64_t> rootServices;
  for (const MemoryRealizationActors &realization : *memoryRealizations) {
    if (llvm::Error error =
            validateMemoryRealization(realization, dataflow, *dataflowIndex,
                                      fabric, *fabricIndex, rootServices))
      return std::move(error);
  }
  auto memoryProjection = buildMemoryRealizationProjections(
      mapping.memoryRealizations, fabricIndex->memorySemanticEncodings,
      fabricIndex->memoryImplementations,
      fabricIndex->memoryOperationPortTemplates);
  if (!memoryProjection)
    return memoryProjection.takeError();
  mappingProjection->memoryRealizations = std::move(*memoryProjection);

  if (llvm::Error error = validateCoveredEdgeAccounting(
          *dataflowIndex, coveredGraphs, actorToRealization))
    return std::move(error);

  std::sort(mappingProjection->computeRealizations.begin(),
            mappingProjection->computeRealizations.end(),
            [](const ValidatedComputeRealizationProjection &lhs,
               const ValidatedComputeRealizationProjection &rhs) {
              return lhs.id.value() < rhs.id.value();
            });
  return ValidatedTechMapping(std::move(identity), mapping,
                              fabricIndex->projection,
                              std::move(mappingProjection));
}
