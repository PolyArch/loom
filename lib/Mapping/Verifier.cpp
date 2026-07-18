#include "Mapping/Verifier.h"
#include "FabricOccurrenceIndex.h"
#include "MemoryRealizationProjection.h"
#include "VerifierInternal.h"
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
namespace {
constexpr SchemaVersion supportedSchemaVersion{2, 0};
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
  EdgeId id;
  DataflowPortInfo source;
  DataflowPortInfo target;
};
using ActorPortKey = std::pair<PortDirection, std::uint32_t>;
struct MemoryActorInfo {
  const CanonicalMemoryActorView *view;
  std::map<ActorPortKey, MemoryAccessPortRole> ports;
};
struct DataflowIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const GraphDescriptor *> graphs;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
  std::map<std::uint64_t, const LogicalMemoryRootDescriptor *>
      logicalMemoryRoots;
  std::map<std::uint64_t, MemoryActorInfo> memoryActors;
  std::map<std::uint64_t, std::size_t> edgesById;
  std::vector<ResolvedDataflowEdge> edges;
};
struct FabricIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const FuDescriptor *> functionalUnits;
  std::map<std::uint64_t, const FabricOpDescriptor *> operations;
  std::map<std::uint64_t, const EncodingDescriptor *> encodings;
  std::map<std::uint64_t, ValidatedPairedLaneCapability> pairedLaneCapabilities;
  std::map<std::uint64_t, ValidatedConfiguredBoundaryIndex>
      configuredBoundaryIndexes;
  std::map<std::uint64_t, const MemoryServiceDomainDescriptor *>
      memoryServiceDomains;
  std::map<std::uint64_t, const MemoryImplementationDescriptor *>
      memoryImplementations;
  std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
      memoryOperationPortTemplates;
  std::map<std::uint64_t, const MemoryInternalConnectionDescriptor *>
      memoryInternalConnections;
  std::map<std::uint64_t, const MemorySemanticEncodingDescriptor *>
      memorySemanticEncodings;
  std::shared_ptr<const ValidatedFabricProjection> projection;
};
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
bool samePortClasses(const std::vector<PortDescriptor> &lhs,
                     const std::vector<PortDescriptor> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.kind != right.kind || left.role != right.role)
      return false;
  return true;
}
llvm::Expected<const PortDescriptor *> resolveConfiguredValue(
    const ConfiguredValue &value,
    const std::map<std::uint32_t, const PortDescriptor *> &inputs,
    const std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
        &operations) {
  if (const auto *input = std::get_if<FuInputValue>(&value)) {
    auto descriptor = inputs.find(input->index);
    if (descriptor == inputs.end())
      return mappingError(MappingErrorCode::InvalidConfiguredFunction,
                          "configured value names an inactive FU input");
    return descriptor->second;
  }
  const auto &result = std::get<FabricOpResultValue>(value);
  auto operation = operations.find(result.operation.value());
  if (operation == operations.end() ||
      result.index >= operation->second->outputPorts.size())
    return mappingError(MappingErrorCode::InvalidConfiguredFunction,
                        "configured value names an invalid fabric.op result");
  return &operation->second->outputPorts[result.index];
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
  for (const LogicalMemoryRootDescriptor &root : dataflow.logicalMemoryRoots) {
    if (llvm::Error error = addEntity(index.kinds, root.id.value(),
                                      EntityKind::LogicalMemoryRoot))
      return std::move(error);
    index.logicalMemoryRoots.emplace(root.id.value(), &root);
  }
  for (const DataflowEdge &edge : dataflow.edges) {
    if (llvm::Error error =
            addEntity(index.kinds, edge.id.value(), EntityKind::Edge))
      return std::move(error);
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
  std::set<EdgeKey> edges;
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
    if (!edges.insert(edgeKey).second)
      return mappingError(MappingErrorCode::DuplicateEdge,
                          "dataflow edge is duplicated");
    index.edgesById.emplace(edge.id.value(), index.edges.size());
    index.edges.push_back(ResolvedDataflowEdge{edge.id, *source, *target});
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
  for (const FabricOpDescriptor &operation : fabric.operations) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, operation.fu.value(), EntityKind::Fu))
      return std::move(error);
    if (!operation.pairedLanes.empty()) {
      auto capability = buildValidatedPairedLaneCapability(operation);
      if (!capability)
        return capability.takeError();
      index.pairedLaneCapabilities.emplace(operation.id.value(),
                                           std::move(*capability));
    }
  }
  for (const EncodingDescriptor &encoding : fabric.encodings) {
    if (llvm::Error error =
            requireLocalKind(index.kinds, encoding.fu.value(), EntityKind::Fu))
      return std::move(error);
    const FuDescriptor &fu = *index.functionalUnits.at(encoding.fu.value());
    std::map<std::uint32_t, const PortDescriptor *> configuredInputs;
    for (const ConfiguredInputDescriptor &input : encoding.inputs) {
      if (input.fuPort >= fu.inputPorts.size() ||
          fu.inputPorts[input.fuPort].kind != input.port.kind ||
          fu.inputPorts[input.fuPort].role != input.port.role ||
          !configuredInputs.emplace(input.fuPort, &input.port).second)
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "encoding has an invalid or duplicate configured FU input");
    }
    std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
        configuredOperations;
    for (const ConfiguredFabricOpDescriptor &configured : encoding.operations) {
      auto operation = index.operations.find(configured.operation.value());
      if (operation == index.operations.end() ||
          operation->second->fu != encoding.fu ||
          configured.operands.size() != configured.inputPorts.size() ||
          !configuredOperations
               .emplace(configured.operation.value(), &configured)
               .second)
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "encoding has an invalid or duplicate configured fabric.op");
      const bool validPorts =
          operation->second->pairedLanes.empty()
              ? samePortClasses(configured.inputPorts,
                                operation->second->inputPorts) &&
                    samePortClasses(configured.outputPorts,
                                    operation->second->outputPorts)
              : validPairedConfiguredPorts(configured, *operation->second);
      if (!validPorts)
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "configured fabric.op ports do not match its physical capability");
    }
    for (const ConfiguredFabricOpDescriptor &configured : encoding.operations) {
      for (auto [operand, expected] :
           llvm::zip(configured.operands, configured.inputPorts)) {
        auto source = resolveConfiguredValue(operand, configuredInputs,
                                             configuredOperations);
        if (!source)
          return source.takeError();
        if (**source != expected)
          return mappingError(
              MappingErrorCode::InvalidConfiguredFunction,
              "configured fabric.op operand has the wrong semantic type");
      }
    }
    std::set<std::uint32_t> configuredOutputs;
    for (const ConfiguredOutputDescriptor &output : encoding.outputs) {
      if (output.fuPort >= fu.outputPorts.size() ||
          fu.outputPorts[output.fuPort].kind != output.port.kind ||
          fu.outputPorts[output.fuPort].role != output.port.role ||
          !configuredOutputs.insert(output.fuPort).second)
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "encoding has an invalid or duplicate configured FU output");
      auto source = resolveConfiguredValue(output.value, configuredInputs,
                                           configuredOperations);
      if (!source)
        return source.takeError();
      if (**source != output.port)
        return mappingError(MappingErrorCode::InvalidConfiguredFunction,
                            "configured FU output has the wrong semantic type");
    }
    auto boundaryIndex = buildValidatedConfiguredBoundaryIndex(encoding);
    if (!boundaryIndex)
      return boundaryIndex.takeError();
    index.configuredBoundaryIndexes.emplace(encoding.id.value(),
                                            std::move(*boundaryIndex));
  }
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
    if (!source->source || sink->source || *source->port != *sink->port ||
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
resolveEdgeReference(const EdgeRef &reference,
                     const DataflowProgramView &dataflow,
                     const DataflowIndex &index) {
  if (reference.artifact != dataflow.identity)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "reference names a foreign artifact");
  const auto kind = index.kinds.find(reference.entity.value());
  if (kind == index.kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "reference names an unresolved entity ID");
  if (kind->second != EntityKind::Edge)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "reference names an entity of the wrong kind");
  return &index.edges[index.edgesById.at(reference.entity.value())];
}

struct RealizationActors {
  const ComputeRealizationDraft *record;
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

  for (const ComputeRealizationDraft &realization : mapping.realizations) {
    if (realization.actors.empty())
      return mappingError(MappingErrorCode::EmptyActorGroup,
                          "Compute Realization actor group is empty");

    RealizationActors actors{&realization, 0, {}};
    bool firstActor = true;
    for (const ActorRef &actorReference : realization.actors) {
      auto actor = resolveReference(actorReference, dataflow.identity,
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

struct ResolvedRealization {
  const FuDescriptor *fu;
  const EncodingDescriptor *encoding;
  std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *> actorToOp;
  std::map<std::uint64_t, PairedLaneProjection> actorToLaneProjections;
  std::vector<ValidatedConfiguredBoundaryPort> activeBoundaryPorts;
};

std::uint32_t configuredPortIndex(const ResolvedRealization &selected,
                                  std::uint64_t actor,
                                  std::uint32_t softwarePort) {
  auto lanes = selected.actorToLaneProjections.find(actor);
  return lanes == selected.actorToLaneProjections.end()
             ? softwarePort
             : lanes->second.laneIndices[softwarePort];
}

llvm::Expected<ResolvedRealization> validateActorToOpCorrespondence(
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

  if (realization.record->actorToOps.size() != realization.actors.size() ||
      (*encoding)->operations.size() != realization.actors.size())
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-fabric.op correspondence is not complete");

  std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
      configuredOperations;
  for (const ConfiguredFabricOpDescriptor &operation : (*encoding)->operations)
    configuredOperations.emplace(operation.operation.value(), &operation);

  std::set<std::uint64_t> mappedActors;
  std::set<std::uint64_t> mappedOperations;
  ResolvedRealization resolved{*fu, *encoding, {}, {}, {}};
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
    auto configured = configuredOperations.find((*operation)->id.value());
    if (configured == configuredOperations.end())
      return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                          "actor correspondence names an inactive fabric.op");
    if ((*actor)->operation != configured->second->semantics ||
        (*actor)->attributes != configured->second->attributes)
      return mappingError(
          MappingErrorCode::ConfiguredFunctionMismatch,
          "actor semantics do not match the configured fabric.op");

    const bool subsetArity =
        (*actor)->inputPorts.size() != configured->second->inputPorts.size() ||
        (*actor)->outputPorts.size() != configured->second->outputPorts.size();
    if ((*operation)->pairedLanes.empty()) {
      if (subsetArity)
        return mappingError(
            MappingErrorCode::ConfiguredFunctionMismatch,
            "subset arity requires an explicit paired-lane capability");
      if (!correspondence.laneSelections.empty() ||
          (*actor)->inputPorts != configured->second->inputPorts ||
          (*actor)->outputPorts != configured->second->outputPorts)
        return mappingError(
            MappingErrorCode::ConfiguredFunctionMismatch,
            "ordinary actor semantics do not match the configured fabric.op");
    } else {
      if ((*actor)->inputPorts.size() != correspondence.laneSelections.size() ||
          (*actor)->outputPorts.size() != correspondence.laneSelections.size())
        return mappingError(MappingErrorCode::ConfiguredFunctionMismatch,
                            "paired-lane correspondence is incomplete");
      auto capability =
          fabricIndex.pairedLaneCapabilities.find((*operation)->id.value());
      if (capability == fabricIndex.pairedLaneCapabilities.end())
        return mappingError(MappingErrorCode::InternalError,
                            "validated paired-lane capability is missing");
      auto projection = validateAndProjectPairedLaneSelection(
          fabric.identity, **operation, capability->second, correspondence);
      if (!projection)
        return projection.takeError();
      for (std::size_t softwarePort = 0;
           softwarePort < projection->laneIndices.size(); ++softwarePort) {
        const std::uint32_t lane = projection->laneIndices[softwarePort];
        if ((*actor)->inputPorts[softwarePort] !=
                configured->second->inputPorts[lane] ||
            (*actor)->outputPorts[softwarePort] !=
                configured->second->outputPorts[lane])
          return mappingError(
              MappingErrorCode::ConfiguredFunctionMismatch,
              "actor lane type does not match the configured fabric.op");
      }
      resolved.actorToLaneProjections.emplace((*actor)->id.value(),
                                              std::move(*projection));
    }
    resolved.actorToOp.emplace((*actor)->id.value(), configured->second);
  }

  if (mappedActors.size() != realization.actors.size() ||
      mappedOperations.size() != configuredOperations.size())
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-fabric.op correspondence is not complete");
  return resolved;
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

struct ResolvedBoundary {
  std::map<EndpointKey, std::uint32_t> actorToFuPort;
};

llvm::Expected<ResolvedBoundary> validateBoundaryCorrespondence(
    const RealizationActors &realization, const ResolvedRealization &selected,
    const DataflowProgramView &dataflow, const DataflowIndex &dataflowIndex,
    const FabricHardwareView &fabric, const FabricIndex &fabricIndex) {
  const std::set<EndpointKey> expected =
      deriveBoundaryPorts(realization, dataflowIndex);
  std::set<EndpointKey> mappedActorPorts;
  std::set<std::tuple<std::uint64_t, PortDirection, std::uint32_t>>
      mappedFuPorts;
  std::map<std::pair<PortDirection, std::uint32_t>, const PortDescriptor *>
      configuredPorts;
  for (const ConfiguredInputDescriptor &input : selected.encoding->inputs)
    configuredPorts.emplace(std::make_pair(PortDirection::Input, input.fuPort),
                            &input.port);
  for (const ConfiguredOutputDescriptor &output : selected.encoding->outputs)
    configuredPorts.emplace(
        std::make_pair(PortDirection::Output, output.fuPort), &output.port);

  ResolvedBoundary resolved;

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

    if (!realization.actors.count(actorPort->key.owner) ||
        !expected.count(actorPort->key) ||
        !mappedActorPorts.insert(actorPort->key).second)
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "software boundary endpoint correspondence is not exact");
    if (fuPort->fu != selected.fu->id.value())
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "boundary correspondence uses a different FU");
    if (actorPort->key.direction != fuPort->direction)
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "boundary correspondence reverses port direction");
    auto configured =
        configuredPorts.find(std::make_pair(fuPort->direction, fuPort->index));
    if (configured == configuredPorts.end())
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "boundary correspondence names an inactive configured FU port");
    if (*actorPort->descriptor != *configured->second)
      return mappingError(
          MappingErrorCode::ConfiguredFunctionMismatch,
          "software boundary type does not match the configured FU port");
    const auto fuKey =
        std::make_tuple(fuPort->fu, fuPort->direction, fuPort->index);
    const bool repeatedFuPort = !mappedFuPorts.insert(fuKey).second;
    if (repeatedFuPort && fuPort->direction == PortDirection::Output)
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "configured FU output correspondence is not one-to-one");
    resolved.actorToFuPort.emplace(actorPort->key, fuPort->index);
  }

  if (mappedActorPorts != expected)
    return mappingError(
        MappingErrorCode::UnaccountedGraphEdge,
        "declared graph edge has an unmapped boundary endpoint");
  if (mappedFuPorts.size() != selected.activeBoundaryPorts.size())
    return mappingError(
        MappingErrorCode::IncompleteBoundaryCorrespondence,
        "active configured FU boundary correspondence is not complete");
  for (const ValidatedConfiguredBoundaryPort &port :
       selected.activeBoundaryPorts) {
    const auto key =
        std::make_tuple(selected.fu->id.value(), port.direction, port.fuPort);
    if (!mappedFuPorts.count(key))
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "active configured FU boundary correspondence is not complete");
  }
  return resolved;
}

llvm::Error validateConfiguredFunctionTopology(
    const RealizationActors &realization, const ResolvedRealization &selected,
    const ResolvedBoundary &boundary, const DataflowIndex &dataflowIndex) {
  std::map<EndpointKey, const DataflowPortInfo *> drivers;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges)
    drivers.emplace(edge.target.key, &edge.source);
  std::map<std::uint32_t, EndpointKey> configuredInputSources;

  for (const auto &entry : selected.actorToOp) {
    const ActorDescriptor &actor = *realization.actors.at(entry.first);
    const ConfiguredFabricOpDescriptor &configured = *entry.second;
    for (std::size_t input = 0; input < actor.inputPorts.size(); ++input) {
      const EndpointKey target{true, actor.id.value(), PortDirection::Input,
                               static_cast<std::uint32_t>(input)};
      const DataflowPortInfo &source = *drivers.at(target);
      ConfiguredValue expected = FuInputValue{0};
      if (source.key.actor && realization.actors.count(source.key.owner)) {
        const ConfiguredFabricOpDescriptor &sourceOperation =
            *selected.actorToOp.at(source.key.owner);
        expected = FabricOpResultValue{
            sourceOperation.operation,
            configuredPortIndex(selected, source.key.owner, source.key.index)};
      } else {
        auto port = boundary.actorToFuPort.find(target);
        if (port == boundary.actorToFuPort.end())
          return mappingError(
              MappingErrorCode::ConfiguredFunctionMismatch,
              "software operand has no configured FU input correspondence");
        auto [inputSource, inserted] =
            configuredInputSources.emplace(port->second, source.key);
        if (!inserted && !(inputSource->second == source.key))
          return mappingError(
              MappingErrorCode::ConfiguredFunctionMismatch,
              "configured FU input correspondence merges distinct software "
              "values");
        expected = FuInputValue{port->second};
      }
      const std::uint32_t configuredInput = configuredPortIndex(
          selected, actor.id.value(), static_cast<std::uint32_t>(input));
      if (configured.operands[configuredInput] != expected)
        return mappingError(
            MappingErrorCode::ConfiguredFunctionMismatch,
            "configured fabric.op operand topology does not match software");
    }
  }

  std::map<std::uint32_t, const ConfiguredOutputDescriptor *> configuredOutputs;
  for (const ConfiguredOutputDescriptor &output : selected.encoding->outputs)
    configuredOutputs.emplace(output.fuPort, &output);
  for (const auto &entry : boundary.actorToFuPort) {
    if (entry.first.direction != PortDirection::Output)
      continue;
    const ConfiguredFabricOpDescriptor &source =
        *selected.actorToOp.at(entry.first.owner);
    const ConfiguredValue expected = FabricOpResultValue{
        source.operation,
        configuredPortIndex(selected, entry.first.owner, entry.first.index)};
    auto output = configuredOutputs.find(entry.second);
    if (output == configuredOutputs.end() || output->second->value != expected)
      return mappingError(
          MappingErrorCode::ConfiguredFunctionMismatch,
          "configured FU output topology does not match software");
  }
  return llvm::Error::success();
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
          hardware->second.second->port != softwarePorts[entry.first.second])
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
        *graphPort->descriptor != implementationPort->descriptor->port ||
        !graphInternalEndpoints.emplace(graphPort->key, endpoint).second ||
        !mappedImplementationBoundaries.insert(endpoint).second)
      return mappingError(
          MappingErrorCode::IncompleteMemoryBoundaryCorrespondence,
          "memory graph boundary correspondence is not exact");
  }

  std::set<std::uint64_t> internalEdges;
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
        !internalEdges.insert((*edge)->id.value()).second ||
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
        *(*edge)->source.descriptor != *sourcePort->port ||
        *(*edge)->target.descriptor != *sinkPort->port)
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
        internalEdges.count(edge.id.value()))
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
    if (*actorPort->descriptor != operationPort->descriptor->port)
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
  if (mapping.header.schemaVersion != supportedSchemaVersion)
    return mappingError(MappingErrorCode::UnsupportedSchemaVersion,
                        "Mapping verifier supports schema 2.0");
  if (mapping.header.profile != MappingProfile::TechMapping)
    return mappingError(MappingErrorCode::WrongMappingProfile,
                        "Mapping verifier requires the TechMapping profile");
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
    auto selected = validateActorToOpCorrespondence(
        realization, dataflow, *dataflowIndex, fabric, *fabricIndex);
    if (!selected)
      return selected.takeError();
    auto boundaryIndex = fabricIndex->configuredBoundaryIndexes.find(
        selected->encoding->id.value());
    if (boundaryIndex == fabricIndex->configuredBoundaryIndexes.end())
      return mappingError(MappingErrorCode::InternalError,
                          "validated configured boundary index is missing");
    selected->activeBoundaryPorts = deriveActiveConfiguredBoundaryPorts(
        boundaryIndex->second, selected->actorToOp,
        selected->actorToLaneProjections);
    auto boundary = validateBoundaryCorrespondence(
        realization, *selected, dataflow, *dataflowIndex, fabric, *fabricIndex);
    if (!boundary)
      return boundary.takeError();
    if (llvm::Error error = validateConfiguredFunctionTopology(
            realization, *selected, *boundary, *dataflowIndex))
      return std::move(error);

    ValidatedComputeRealizationProjection projected{
        realization.record->id,
        selected->fu->id,
        selected->encoding->id,
        std::move(selected->activeBoundaryPorts),
        {}};
    projected.pairedLaneProjections.reserve(
        selected->actorToLaneProjections.size());
    for (auto &entry : selected->actorToLaneProjections) {
      PairedLaneProjection &lanes = entry.second;
      projected.pairedLaneProjections.push_back(
          {ActorId(entry.first), selected->actorToOp.at(entry.first)->operation,
           std::move(lanes.laneIndices), std::move(lanes.bitmask)});
    }
    mappingProjection->computeRealizations.push_back(std::move(projected));
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
