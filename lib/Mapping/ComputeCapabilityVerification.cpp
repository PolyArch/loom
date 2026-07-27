#include "VerifierState.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;

namespace {

struct FuPortInfo {
  ::loom::fabric::FabricFuTemplateRef fu;
  PortDirection direction;
  std::uint32_t index;
  const PortDescriptor *descriptor;
};

llvm::Expected<FuPortInfo>
resolveFuPortReference(const ::loom::fabric::FabricFuTemplatePortRef &port,
                       const FabricIndex &index) {
  auto fu = index.functionalUnits.find(port.fu.id());
  if (fu == index.functionalUnits.end())
    return mappingError(MappingErrorCode::MissingFuImplementation,
                        "FU boundary port names a missing implementation");
  const PortDirection direction =
      port.direction == ::loom::fabric::FabricPortDirection::Input
          ? PortDirection::Input
          : PortDirection::Output;
  const auto &ports = direction == PortDirection::Input
                          ? fu->second->inputPorts
                          : fu->second->outputPorts;
  if (port.ordinal >= ports.size())
    return mappingError(MappingErrorCode::InvalidPortConnection,
                        "FU boundary port index is out of range");
  return FuPortInfo{fu->second->id, direction,
                    static_cast<std::uint32_t>(port.ordinal),
                    &ports[port.ordinal]};
}

struct ResolvedRealization {
  const FuDescriptor *fu;
  const ::loom::fabric::FabricFuCapabilityTemplateRecord *capabilityTemplate;
  std::map<std::uint64_t, const FabricOpDescriptor *> actorToOp;
  struct PortCorrespondence {
    std::vector<std::uint32_t> operandPorts;
    std::vector<std::uint32_t> resultPorts;
  };
  std::map<std::uint64_t, PortCorrespondence> actorToPorts;
  std::vector<ValidatedComputeBoundaryPort> activeBoundaryPorts;
};

bool validPortCorrespondence(const std::vector<PortDescriptor> &actorPorts,
                             const std::vector<PortDescriptor> &physicalPorts,
                             const std::vector<std::uint32_t> &selected) {
  if (selected.size() != actorPorts.size())
    return false;
  std::vector<bool> taken(physicalPorts.size(), false);
  for (std::size_t softwarePort = 0; softwarePort < selected.size();
       ++softwarePort) {
    const std::uint32_t physicalPort = selected[softwarePort];
    if (physicalPort >= physicalPorts.size() || taken[physicalPort] ||
        !physicalPortAdmitsSemanticPort(actorPorts[softwarePort],
                                        physicalPorts[physicalPort]))
      return false;
    taken[physicalPort] = true;
  }
  return true;
}

llvm::Expected<ResolvedRealization> validateActorToOpCorrespondence(
    const RealizationActors &realization, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex, const FabricIndex &fabricIndex) {
  const auto &templateRef = realization.record->capabilityTemplate;
  auto fu = fabricIndex.functionalUnits.find(templateRef.fu.id());
  if (fu == fabricIndex.functionalUnits.end())
    return mappingError(MappingErrorCode::InvalidCapabilityTemplateReference,
                        "selected template names a missing FU definition");
  if (llvm::Error error = ::loom::fabric::validateFabricFuCapabilityTemplateRef(
          fu->second->capabilityTemplates, templateRef))
    return mappingError(
        MappingErrorCode::InvalidCapabilityTemplateReference,
        llvm::Twine("invalid selected FU capability-template reference: ") +
            llvm::toString(std::move(error)));
  const auto &capabilityTemplate =
      fu->second->capabilityTemplates[templateRef.ordinal];

  std::set<FabricNodeKey> activeOperations;
  for (const ::loom::fabric::FabricFuTemplateNodeRef &node :
       capabilityTemplate.activeNodes) {
    if (node.node != ::loom::fabric::FabricFuNodeKind::Op)
      continue;
    const FabricNodeKey key = nodeKey(node);
    if (!fabricIndex.operations.count(key))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "selected capability template names an unknown Fabric operation");
    activeOperations.insert(key);
  }
  if (activeOperations.size() != realization.actors.size())
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-Fabric operation correspondence is not "
                        "complete over the selected template");

  std::set<std::uint64_t> mappedActors;
  std::set<FabricNodeKey> mappedOperations;
  ResolvedRealization resolved{fu->second, &capabilityTemplate, {}, {}, {}};
  for (const ActorToFabricOp &correspondence : realization.record->actorToOps) {
    auto actor = resolveReference(correspondence.actor, dataflow.identity,
                                  dataflowIndex.kinds, EntityKind::Actor,
                                  dataflowIndex.actors);
    if (!actor)
      return actor.takeError();
    const FabricNodeKey operationKey = nodeKey(correspondence.fabricOp);
    auto operation = fabricIndex.operations.find(operationKey);

    if (!realization.actors.count((*actor)->id.value()) ||
        !mappedActors.insert((*actor)->id.value()).second)
      return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                          "actor-to-Fabric operation correspondence is not "
                          "bijective");
    if (operation == fabricIndex.operations.end())
      return mappingError(MappingErrorCode::CapabilityTemplateMismatch,
                          "actor correspondence names an unknown Fabric "
                          "operation");
    if (!mappedOperations.insert(operationKey).second)
      return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                          "actor-to-Fabric operation correspondence is not "
                          "bijective");
    if (correspondence.fabricOp.fu != templateRef.fu)
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "Fabric operation belongs to a different FU");
    if (!activeOperations.count(operationKey))
      return mappingError(MappingErrorCode::CapabilityTemplateMismatch,
                          "actor correspondence names an inactive Fabric "
                          "operation");
    if (llvm::Error error = ::fabric::verifyImplementationFamilyAdmission(
            operation->second->family, &operation->second->capability,
            (*actor)->semantics, dataflow.indexBitWidth))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          llvm::Twine("actor is not admitted by the selected Fabric "
                      "operation: ") +
              llvm::toString(std::move(error)));
    if (!validPortCorrespondence((*actor)->inputPorts,
                                 operation->second->inputPorts,
                                 correspondence.operandPorts) ||
        !validPortCorrespondence((*actor)->outputPorts,
                                 operation->second->outputPorts,
                                 correspondence.resultPorts))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "ordered actor-to-operation port correspondence is invalid");
    resolved.actorToPorts.emplace(
        (*actor)->id.value(),
        ResolvedRealization::PortCorrespondence{correspondence.operandPorts,
                                                correspondence.resultPorts});
    resolved.actorToOp.emplace((*actor)->id.value(), operation->second);
  }

  if (mappedActors.size() != realization.actors.size() ||
      mappedOperations != activeOperations)
    return mappingError(MappingErrorCode::IncompleteActorToOpCorrespondence,
                        "actor-to-Fabric operation correspondence is not "
                        "complete");
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
  std::map<EndpointKey, ::loom::fabric::FabricFuTemplatePortRef> actorToFuPort;
};

llvm::Expected<ResolvedBoundary> validateBoundaryCorrespondence(
    const RealizationActors &realization, const ResolvedRealization &selected,
    const DataflowProgramView &dataflow, const DataflowIndex &dataflowIndex,
    const FabricIndex &fabricIndex) {
  const std::set<EndpointKey> expected =
      deriveBoundaryPorts(realization, dataflowIndex);
  std::set<EndpointKey> mappedActorPorts;
  std::set<std::pair<PortDirection, std::uint32_t>> mappedFuPorts;
  ResolvedBoundary resolved;

  for (const BoundaryPortCorrespondence &correspondence :
       realization.record->boundaryPorts) {
    auto actorPort = resolveActorPortReference(
        correspondence.actorPort, dataflow.identity, dataflowIndex);
    if (!actorPort)
      return actorPort.takeError();
    auto fuPort = resolveFuPortReference(correspondence.fuPort, fabricIndex);
    if (!fuPort)
      return fuPort.takeError();

    if (!realization.actors.count(actorPort->key.owner) ||
        !expected.count(actorPort->key) ||
        !mappedActorPorts.insert(actorPort->key).second)
      return mappingError(
          MappingErrorCode::IncompleteBoundaryCorrespondence,
          "software boundary endpoint correspondence is not exact");
    if (fuPort->fu != selected.fu->id)
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "boundary correspondence uses a different FU");
    if (actorPort->key.direction != fuPort->direction)
      return mappingError(MappingErrorCode::InvalidPortConnection,
                          "boundary correspondence reverses port direction");
    if (!physicalPortAdmitsSemanticPort(*actorPort->descriptor,
                                        *fuPort->descriptor))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "software boundary type does not match the Fabric FU port");
    const auto fuKey = std::make_pair(fuPort->direction, fuPort->index);
    const bool repeatedFuPort = !mappedFuPorts.insert(fuKey).second;
    if (repeatedFuPort && fuPort->direction == PortDirection::Output)
      return mappingError(MappingErrorCode::IncompleteBoundaryCorrespondence,
                          "FU output correspondence is not one-to-one");
    resolved.actorToFuPort.emplace(actorPort->key, correspondence.fuPort);
  }

  if (mappedActorPorts != expected)
    return mappingError(
        MappingErrorCode::UnaccountedGraphEdge,
        "declared graph edge has an unmapped boundary endpoint");
  if (mappedFuPorts.size() != selected.activeBoundaryPorts.size())
    return mappingError(MappingErrorCode::IncompleteBoundaryCorrespondence,
                        "active FU boundary correspondence is not complete");
  for (const ValidatedComputeBoundaryPort &port :
       selected.activeBoundaryPorts) {
    const auto key = std::make_pair(port.direction, port.fuPort);
    if (!mappedFuPorts.count(key))
      return mappingError(MappingErrorCode::IncompleteBoundaryCorrespondence,
                          "active FU boundary correspondence is not complete");
  }

  std::map<std::pair<PortDirection, std::uint32_t>, EndpointKey> inputSources;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    if (!edge.target.key.actor ||
        !realization.actors.count(edge.target.key.owner))
      continue;
    auto boundary = resolved.actorToFuPort.find(edge.target.key);
    if (boundary == resolved.actorToFuPort.end())
      continue;
    const auto key =
        std::make_pair(PortDirection::Input,
                       static_cast<std::uint32_t>(boundary->second.ordinal));
    auto [source, inserted] = inputSources.emplace(key, edge.source.key);
    if (!inserted && !(source->second == edge.source.key))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "one FU input correspondence merges distinct software values");
  }
  return resolved;
}

struct TemplateEndpointKey {
  bool boundary;
  ::loom::fabric::FabricFuNodeKind nodeKind;
  std::uint64_t fu;
  std::uint64_t nodeOrdinal;
  ::loom::fabric::FabricPortDirection direction;
  std::uint64_t portOrdinal;

  friend bool operator==(TemplateEndpointKey lhs, TemplateEndpointKey rhs) {
    return std::tie(lhs.boundary, lhs.nodeKind, lhs.fu, lhs.nodeOrdinal,
                    lhs.direction, lhs.portOrdinal) ==
           std::tie(rhs.boundary, rhs.nodeKind, rhs.fu, rhs.nodeOrdinal,
                    rhs.direction, rhs.portOrdinal);
  }
  friend bool operator<(TemplateEndpointKey lhs, TemplateEndpointKey rhs) {
    return std::tie(lhs.boundary, lhs.nodeKind, lhs.fu, lhs.nodeOrdinal,
                    lhs.direction, lhs.portOrdinal) <
           std::tie(rhs.boundary, rhs.nodeKind, rhs.fu, rhs.nodeOrdinal,
                    rhs.direction, rhs.portOrdinal);
  }
};

TemplateEndpointKey endpointKey(
    const ::loom::fabric::FabricFuCapabilityTemplateEndpointRef &endpoint) {
  if (endpoint.kind() ==
      ::loom::fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
    const auto &port =
        std::get<::loom::fabric::FabricFuTemplatePortRef>(endpoint.payload);
    return {true,           ::loom::fabric::FabricFuNodeKind::Op,
            port.fu.id(),   0,
            port.direction, port.ordinal};
  }
  const auto &port =
      std::get<::loom::fabric::FabricFuNodePortRef>(endpoint.payload);
  return {false,          port.node.node, port.node.fu.id(), port.node.ordinal,
          port.direction, port.ordinal};
}

TemplateEndpointKey
endpointKey(const ::loom::fabric::FabricFuTemplatePortRef &port) {
  return {true,           ::loom::fabric::FabricFuNodeKind::Op,
          port.fu.id(),   0,
          port.direction, port.ordinal};
}

TemplateEndpointKey actorPortEndpoint(const ResolvedRealization &selected,
                                      std::uint64_t actor,
                                      PortDirection direction,
                                      std::uint32_t softwarePort) {
  const FabricOpDescriptor &operation = *selected.actorToOp.at(actor);
  const auto &ports = selected.actorToPorts.at(actor);
  const std::uint32_t physicalPort = direction == PortDirection::Input
                                         ? ports.operandPorts[softwarePort]
                                         : ports.resultPorts[softwarePort];
  return {false,
          operation.id.node,
          operation.id.fu.id(),
          operation.id.ordinal,
          direction == PortDirection::Input
              ? ::loom::fabric::FabricPortDirection::Input
              : ::loom::fabric::FabricPortDirection::Output,
          physicalPort};
}

using TemplateConnection = std::pair<TemplateEndpointKey, TemplateEndpointKey>;

llvm::Expected<std::set<TemplateConnection>>
projectTemplateConnections(ResolvedRealization &selected) {
  std::map<TemplateEndpointKey, std::set<TemplateEndpointKey>> adjacency;
  std::map<TemplateEndpointKey, std::set<TemplateEndpointKey>> reverse;
  std::set<TemplateEndpointKey> sourceTerminals;
  std::set<TemplateEndpointKey> destinationTerminals;
  std::map<FabricNodeKey, std::set<TemplateEndpointKey>> structuralInputs;
  std::map<FabricNodeKey, std::set<TemplateEndpointKey>> structuralOutputs;

  auto validateEndpoint = [&](const TemplateEndpointKey &endpoint,
                              bool source) -> llvm::Error {
    if (endpoint.fu != selected.fu->id.id())
      return mappingError(MappingErrorCode::SelectedFuMismatch,
                          "template edge names a different FU definition");
    if (endpoint.boundary) {
      const auto &ports =
          endpoint.direction == ::loom::fabric::FabricPortDirection::Input
              ? selected.fu->inputPorts
              : selected.fu->outputPorts;
      if (endpoint.portOrdinal >= ports.size())
        return mappingError(MappingErrorCode::CapabilityTemplateMismatch,
                            "template edge names an invalid FU boundary port");
      (source ? sourceTerminals : destinationTerminals).insert(endpoint);
      return llvm::Error::success();
    }
    const FabricNodeKey key{endpoint.nodeKind, endpoint.fu,
                            endpoint.nodeOrdinal};
    if (endpoint.nodeKind == ::loom::fabric::FabricFuNodeKind::Op) {
      auto operation = selected.actorToOp.end();
      for (auto candidate = selected.actorToOp.begin();
           candidate != selected.actorToOp.end(); ++candidate) {
        if (nodeKey(candidate->second->id) == key) {
          operation = candidate;
          break;
        }
      }
      if (operation == selected.actorToOp.end())
        return mappingError(MappingErrorCode::CapabilityTemplateMismatch,
                            "template edge names an unbound Fabric operation");
      const auto &ports =
          endpoint.direction == ::loom::fabric::FabricPortDirection::Input
              ? operation->second->inputPorts
              : operation->second->outputPorts;
      if (endpoint.portOrdinal >= ports.size())
        return mappingError(
            MappingErrorCode::CapabilityTemplateMismatch,
            "template edge names an invalid Fabric operation port");
      (source ? sourceTerminals : destinationTerminals).insert(endpoint);
      return llvm::Error::success();
    }
    (source ? structuralOutputs : structuralInputs)[key].insert(endpoint);
    return llvm::Error::success();
  };

  for (const auto &edge : selected.capabilityTemplate->activeEdges) {
    const TemplateEndpointKey source = endpointKey(edge.source);
    const TemplateEndpointKey destination = endpointKey(edge.destination);
    if (llvm::Error error = validateEndpoint(source, true))
      return std::move(error);
    if (llvm::Error error = validateEndpoint(destination, false))
      return std::move(error);
    adjacency[source].insert(destination);
    reverse[destination].insert(source);
  }
  for (const auto &entry : structuralInputs) {
    auto outputs = structuralOutputs.find(entry.first);
    if (outputs == structuralOutputs.end())
      continue;
    for (const TemplateEndpointKey &input : entry.second)
      for (const TemplateEndpointKey &output : outputs->second) {
        adjacency[input].insert(output);
        reverse[output].insert(input);
      }
  }

  auto reachable = [](const std::set<TemplateEndpointKey> &roots,
                      const auto &graph) {
    std::set<TemplateEndpointKey> visited = roots;
    std::vector<TemplateEndpointKey> worklist(roots.begin(), roots.end());
    while (!worklist.empty()) {
      TemplateEndpointKey current = worklist.back();
      worklist.pop_back();
      auto next = graph.find(current);
      if (next == graph.end())
        continue;
      for (const TemplateEndpointKey &endpoint : next->second)
        if (visited.insert(endpoint).second)
          worklist.push_back(endpoint);
    }
    return visited;
  };
  const std::set<TemplateEndpointKey> forward =
      reachable(sourceTerminals, adjacency);
  const std::set<TemplateEndpointKey> backward =
      reachable(destinationTerminals, reverse);
  for (const auto &edge : selected.capabilityTemplate->activeEdges) {
    if (!forward.count(endpointKey(edge.source)) ||
        !backward.count(endpointKey(edge.destination)))
      return mappingError(
          MappingErrorCode::CapabilityTemplateMismatch,
          "template contains an edge outside any complete terminal path");
  }

  std::set<TemplateConnection> connections;
  for (const TemplateEndpointKey &source : sourceTerminals) {
    std::set<TemplateEndpointKey> visited{source};
    std::vector<TemplateEndpointKey> worklist{source};
    while (!worklist.empty()) {
      TemplateEndpointKey current = worklist.back();
      worklist.pop_back();
      auto next = adjacency.find(current);
      if (next == adjacency.end())
        continue;
      for (const TemplateEndpointKey &endpoint : next->second) {
        if (destinationTerminals.count(endpoint)) {
          connections.emplace(source, endpoint);
          continue;
        }
        if (visited.insert(endpoint).second)
          worklist.push_back(endpoint);
      }
    }
  }

  std::set<std::pair<PortDirection, std::uint32_t>> boundaryPorts;
  for (const TemplateConnection &connection : connections) {
    for (const TemplateEndpointKey &endpoint :
         {connection.first, connection.second}) {
      if (!endpoint.boundary)
        continue;
      const PortDirection direction =
          endpoint.direction == ::loom::fabric::FabricPortDirection::Input
              ? PortDirection::Input
              : PortDirection::Output;
      boundaryPorts.emplace(direction,
                            static_cast<std::uint32_t>(endpoint.portOrdinal));
    }
  }
  selected.activeBoundaryPorts.clear();
  for (const auto &[direction, ordinal] : boundaryPorts) {
    const auto &ports = direction == PortDirection::Input
                            ? selected.fu->inputPorts
                            : selected.fu->outputPorts;
    selected.activeBoundaryPorts.push_back(
        {direction, ordinal, ports[ordinal]});
  }
  return connections;
}

llvm::Error validateCapabilityTemplateTopology(
    const RealizationActors &realization, const ResolvedRealization &selected,
    const ResolvedBoundary &boundary, const DataflowIndex &dataflowIndex,
    const std::set<TemplateConnection> &actual) {
  std::set<TemplateConnection> expected;
  for (const ResolvedDataflowEdge &edge : dataflowIndex.edges) {
    const bool sourceInside = edge.source.key.actor &&
                              realization.actors.count(edge.source.key.owner);
    const bool targetInside = edge.target.key.actor &&
                              realization.actors.count(edge.target.key.owner);
    if (!sourceInside && !targetInside)
      continue;
    if (sourceInside && targetInside) {
      expected.emplace(
          actorPortEndpoint(selected, edge.source.key.owner,
                            PortDirection::Output, edge.source.key.index),
          actorPortEndpoint(selected, edge.target.key.owner,
                            PortDirection::Input, edge.target.key.index));
      continue;
    }
    const EndpointKey actorPort =
        sourceInside ? edge.source.key : edge.target.key;
    auto fuPort = boundary.actorToFuPort.find(actorPort);
    if (fuPort == boundary.actorToFuPort.end())
      return mappingError(MappingErrorCode::IncompleteBoundaryCorrespondence,
                          "software edge has no FU boundary correspondence");
    if (sourceInside)
      expected.emplace(actorPortEndpoint(selected, actorPort.owner,
                                         PortDirection::Output,
                                         actorPort.index),
                       endpointKey(fuPort->second));
    else
      expected.emplace(endpointKey(fuPort->second),
                       actorPortEndpoint(selected, actorPort.owner,
                                         PortDirection::Input,
                                         actorPort.index));
  }
  if (actual != expected)
    return mappingError(
        MappingErrorCode::CapabilityTemplateMismatch,
        "selected Fabric template topology does not exactly match software");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<ValidatedComputeBoundaryPort>>
loom::mapping::detail::verifyComputeRealization(
    const RealizationActors &realization, const DataflowProgramView &dataflow,
    const DataflowIndex &dataflowIndex, const FabricIndex &fabricIndex) {
  auto selected = validateActorToOpCorrespondence(realization, dataflow,
                                                  dataflowIndex, fabricIndex);
  if (!selected)
    return selected.takeError();
  auto templateConnections = projectTemplateConnections(*selected);
  if (!templateConnections)
    return templateConnections.takeError();
  auto boundary = validateBoundaryCorrespondence(
      realization, *selected, dataflow, dataflowIndex, fabricIndex);
  if (!boundary)
    return boundary.takeError();
  if (llvm::Error error = validateCapabilityTemplateTopology(
          realization, *selected, *boundary, dataflowIndex,
          *templateConnections))
    return std::move(error);
  return std::move(selected->activeBoundaryPorts);
}
