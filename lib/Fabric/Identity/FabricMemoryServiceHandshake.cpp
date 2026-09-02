#include "Fabric/Identity/FabricMemoryServiceHandshake.h"

#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_memory_service_handshake_invalid: " + message);
}

std::string byteString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Reference>
void appendReference(std::string &key, const Reference &reference) {
  const auto bytes = canonicalFabricBytes(reference);
  key.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

std::string transportSignalKey(const HandshakeSignalRef &signal) {
  std::string key(1, 't');
  const auto bytes = detail::handshakeSignalKey(signal);
  key.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  return key;
}

std::string memorySignalKey(const MemoryServiceHandshakeSignalRef &signal) {
  std::string key;
  key.push_back('m');
  key.push_back(static_cast<char>(signal.channel));
  key.push_back(static_cast<char>(signal.signal));
  appendReference(key, signal.endpoint);
  return key;
}

std::string boundarySignalKey(const ModuleBoundaryHandshakeSignalRef &signal) {
  std::string key;
  key.push_back('b');
  key.push_back(signal.memoryChannel
                    ? static_cast<char>(1 +
                                        static_cast<std::uint8_t>(
                                            *signal.memoryChannel))
                    : 0);
  key.push_back(static_cast<char>(signal.signal));
  appendReference(key, signal.boundary);
  return key;
}

std::string placementKey(const FabricMemoryHandshakePlacement &placement) {
  std::string key;
  key.push_back(static_cast<char>(placement.index()));
  std::visit([&](const auto &selected) { appendReference(key, selected); },
             placement);
  return key;
}

std::string targetKey(const FabricMemoryHandshakeServiceTarget &target) {
  std::string key;
  key.push_back(static_cast<char>(target.index()));
  std::visit([&](const auto &selected) { appendReference(key, selected); },
             target);
  return key;
}

struct MemoryEndpointInventory final {
  std::vector<ManagerEndpointRef> managers;
  std::vector<SubordinateEndpointRef> subordinates;
};

llvm::Expected<MemoryEndpointInventory>
memoryEndpointInventory(const FabricArtifactView &view,
                        FabricMemoryOccurrenceRef occurrence) {
  MemoryEndpointInventory result;
  const FabricMemoryEndpointOwnerRef owner =
      FabricMemoryEndpointOwnerRef::of(occurrence);
  for (FabricOrdinal ordinal = 0;
       ordinal != view.memoryEndpointCount(owner); ++ordinal) {
    const FabricMemoryEndpointRef endpoint{owner, ordinal};
    const auto role = view.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("memory endpoint has no role");
    if (*role == FabricMemoryEndpointRole::Manager)
      result.managers.emplace_back(endpoint);
    else
      result.subordinates.emplace_back(endpoint);
  }
  return result;
}

bool targetMatches(const ::fabric::MemoryDispatchTarget &declared,
                   const FabricMemoryHandshakeServiceTarget &selected,
                   const MemoryEndpointInventory &endpoints) {
  if (std::holds_alternative<LocalMemoryServiceRef>(selected))
    return std::holds_alternative<::fabric::LocalMemoryDispatchTarget>(
        declared);
  const auto *manager =
      std::get_if<::fabric::ManagerMemoryDispatchTarget>(&declared);
  return manager && manager->endpointOrdinal < endpoints.managers.size() &&
         endpoints.managers[manager->endpointOrdinal] ==
             std::get<ManagerEndpointRef>(selected);
}

llvm::Error validateTarget(
    const FabricArtifactView &view, FabricMemoryOccurrenceRef occurrence,
    llvm::ArrayRef<::fabric::MemoryDispatchTarget> domain,
    const FabricMemoryHandshakeServiceTarget &target,
    const MemoryEndpointInventory &endpoints) {
  if (const auto *local = std::get_if<LocalMemoryServiceRef>(&target)) {
    if (local->underlying() != FabricMemoryServiceRef::local(occurrence) ||
        !view.declaresLocalMemoryService(occurrence))
      return invalid("memory handshake selects a foreign local service");
    if (llvm::Error error = validateFabricRef(view, *local))
      return error;
  } else {
    const auto &manager = std::get<ManagerEndpointRef>(target);
    if (manager.underlying().owner !=
        FabricMemoryEndpointOwnerRef::of(occurrence))
      return invalid("memory handshake selects a foreign manager endpoint");
    if (llvm::Error error = validateFabricRef(view, manager))
      return error;
    if (view.memoryEndpointRole(manager.underlying()) !=
        FabricMemoryEndpointRole::Manager)
      return invalid("memory handshake target is not a manager endpoint");
  }
  if (!llvm::any_of(domain, [&](const auto &candidate) {
        return targetMatches(candidate, target, endpoints);
      }))
    return invalid("memory handshake target is outside H_dispatch");
  return llvm::Error::success();
}

const FabricMemoryHandshakeSelection *findOperation(
    llvm::ArrayRef<FabricMemoryHandshakeSelection> operations,
    const FabricMemoryHandshakePlacement &placement) {
  const auto found = llvm::find_if(operations, [&](const auto &candidate) {
    return candidate.placement() == placement;
  });
  return found == operations.end() ? nullptr : &*found;
}

llvm::Error validateSelection(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection) {
  std::set<std::string> operationKeys;
  for (const FabricMemoryHandshakeSelection &operation :
       transportSelection.memoryOperations)
    if (!operationKeys.insert(placementKey(operation.placement())).second)
      return invalid("memory operation handshake placement is duplicated");
  if (memorySelection.operations.size() != operationKeys.size())
    return invalid("memory operation service selection is incomplete");

  std::set<std::string> selectedOperationKeys;
  for (const auto &selected : memorySelection.operations) {
    const std::string key = placementKey(selected.placement);
    if (!selectedOperationKeys.insert(key).second)
      return invalid("memory operation service selection is duplicated");
    const FabricMemoryHandshakeSelection *operation =
        findOperation(transportSelection.memoryOperations,
                      selected.placement);
    if (!operation || operation->capability() != selected.capability)
      return invalid("memory operation service selection has no exact plan");
    const FabricMemoryOperationPortRef port = selected.capability.port;
    const auto *connectivity = view.memoryConnectivity(port.memory);
    const auto *capability =
        view.memoryCapabilityAlternative(selected.capability);
    if (!connectivity || !capability ||
        port.ordinal >= connectivity->operationPorts().size() ||
        selected.capability.ordinal >=
            connectivity->operationPorts()[port.ordinal]
                .capabilityTargetDomains.size())
      return invalid("memory operation service selection is stale");
    auto endpoints = memoryEndpointInventory(view, port.memory);
    if (!endpoints)
      return endpoints.takeError();
    if (llvm::Error error = validateTarget(
            view, port.memory,
            connectivity->operationPorts()[port.ordinal]
                .capabilityTargetDomains[selected.capability.ordinal],
            selected.target, *endpoints))
      return error;
  }
  if (selectedOperationKeys != operationKeys)
    return invalid("memory operation service selection has a foreign plan");

  std::set<std::string> providerKeys;
  for (const auto &provider : memorySelection.providers) {
    const FabricMemoryEndpointRef subordinate = provider.subordinate.underlying();
    const std::string key = byteString(canonicalFabricBytes(subordinate));
    if (!providerKeys.insert(key).second || provider.targets.empty())
      return invalid("memory provider service selection is noncanonical");
    if (subordinate.owner.kind() !=
            FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence ||
        view.memoryEndpointRole(subordinate) !=
            FabricMemoryEndpointRole::Subordinate)
      return invalid("memory provider selection has a foreign subordinate");
    const auto occurrence =
        std::get<FabricMemoryOccurrenceRef>(subordinate.owner.payload);
    const auto *connectivity = view.memoryConnectivity(occurrence);
    auto endpoints = memoryEndpointInventory(view, occurrence);
    if (!connectivity || !endpoints)
      return endpoints ? invalid("memory provider has no connectivity")
                       : endpoints.takeError();
    const auto found = llvm::find(endpoints->subordinates,
                                  provider.subordinate);
    if (found == endpoints->subordinates.end())
      return invalid("memory provider subordinate is absent");
    const std::size_t ordinal =
        static_cast<std::size_t>(std::distance(endpoints->subordinates.begin(),
                                              found));
    if (ordinal >= connectivity->subordinateEndpoints().size())
      return invalid("memory provider dispatch row is absent");
    std::optional<std::string> previous;
    for (const auto &target : provider.targets) {
      const std::string targetBytes = targetKey(target);
      if (previous && *previous >= targetBytes)
        return invalid("memory provider targets are not canonical");
      previous = targetBytes;
      if (llvm::Error error = validateTarget(
              view, occurrence,
              connectivity->subordinateEndpoints()[ordinal].targetDomain,
              target, *endpoints))
        return error;
    }
  }
  return llvm::Error::success();
}

const ::fabric::MemoryRoleEndpointBindingRecord *
bindingForRole(const MemoryCapabilityAlternativeView &capability,
               std::size_t role) {
  const auto found = llvm::find_if(capability.roleToEndpoint,
                                   [&](const auto &candidate) {
    return static_cast<std::size_t>(candidate.role) == role;
  });
  return found == capability.roleToEndpoint.end() ? nullptr : &*found;
}

struct Graph final {
  std::set<std::string> nodes;
  std::set<std::pair<std::string, std::string>> arcs;

  void addNode(const std::string &node) { nodes.insert(node); }
  void addArc(const std::string &source, const std::string &destination) {
    nodes.insert(source);
    nodes.insert(destination);
    arcs.emplace(source, destination);
  }
};

std::string memorySignalKey(FabricMemoryEndpointRef endpoint,
                            MemoryServiceHandshakeChannel channel,
                            HandshakeSignalKind signal) {
  return memorySignalKey({endpoint, channel, signal});
}

std::string boundarySignalKey(
    FabricModuleBoundaryEndpointRef endpoint,
    std::optional<MemoryServiceHandshakeChannel> channel,
    HandshakeSignalKind signal) {
  return boundarySignalKey({endpoint, channel, signal});
}

using MemoryNetworkFace =
    std::variant<FabricMemoryEndpointRef, FabricModuleBoundaryEndpointRef>;

std::string faceKey(const MemoryNetworkFace &face) {
  std::string key;
  key.push_back(static_cast<char>(face.index()));
  std::visit([&](const auto &selected) { appendReference(key, selected); },
             face);
  return key;
}

std::string faceSignalKey(const MemoryNetworkFace &face,
                          MemoryServiceHandshakeChannel channel,
                          HandshakeSignalKind signal) {
  return std::visit(
      [&](const auto &selected) {
        using Face = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Face, FabricMemoryEndpointRef>)
          return memorySignalKey(selected, channel, signal);
        else
          return boundarySignalKey(selected, channel, signal);
      },
      face);
}

struct MemoryNetworkConnection final {
  MemoryNetworkFace source;
  MemoryNetworkFace destination;
};

llvm::Error appendMemoryOwnerArcs(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection,
    std::set<std::string> &activeManagers,
    std::set<std::string> &activeSubordinates, Graph &graph) {
  for (const auto &selected : memorySelection.operations) {
    const FabricMemoryHandshakeSelection *operation =
        findOperation(transportSelection.memoryOperations,
                      selected.placement);
    if (!operation)
      return invalid("memory service operation lost its exact plan");
    const auto *capability =
        view.memoryCapabilityAlternative(selected.capability);
    if (!capability)
      return invalid("memory service operation capability is stale");
    const FabricMemoryOccurrenceRef memory = selected.capability.port.memory;
    const FabricTransportEndpointOwnerRef tokenOwner =
        FabricTransportEndpointOwnerRef::of(memory);
    std::vector<FabricTransportEndpointRef> inputs;
    std::vector<FabricTransportEndpointRef> outputs;
    for (std::size_t role = 0; role != operation->roleSources().size(); ++role) {
      const auto *binding = bindingForRole(*capability, role);
      if (operation->roleSources()[role]) {
        if (!binding)
          return invalid("memory service input role has no endpoint");
        inputs.push_back({tokenOwner, binding->endpointOrdinal});
      }
      if (operation->roleDestinations()[role]) {
        if (!binding)
          return invalid("memory service output role has no endpoint");
        outputs.push_back({tokenOwner, binding->endpointOrdinal});
      }
    }

    const bool spatial =
        view.memorySchedule(memory) == ::fabric::Schedule::Spatial;
    if (spatial)
      for (const FabricTransportEndpointRef input : inputs)
        for (const FabricTransportEndpointRef driver : inputs)
          graph.addArc(
              transportSignalKey({driver, HandshakeSignalKind::Valid}),
              transportSignalKey({input, HandshakeSignalKind::Ready}));

    const auto *manager = std::get_if<ManagerEndpointRef>(&selected.target);
    if (!manager) {
      if (spatial)
        for (const FabricTransportEndpointRef output : outputs)
          for (const FabricTransportEndpointRef input : inputs)
            graph.addArc(
                transportSignalKey({output, HandshakeSignalKind::Ready}),
                transportSignalKey({input, HandshakeSignalKind::Ready}));
      continue;
    }

    const FabricMemoryEndpointRef endpoint = manager->underlying();
    activeManagers.insert(byteString(canonicalFabricBytes(endpoint)));
    const std::string requestValid = memorySignalKey(
        endpoint, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Valid);
    const std::string requestReady = memorySignalKey(
        endpoint, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Ready);
    if (spatial)
      for (const FabricTransportEndpointRef input : inputs) {
        graph.addArc(
            transportSignalKey({input, HandshakeSignalKind::Valid}),
            requestValid);
        graph.addArc(
            requestReady,
            transportSignalKey({input, HandshakeSignalKind::Ready}));
        graph.addArc(
            requestValid,
            transportSignalKey({input, HandshakeSignalKind::Ready}));
      }
    for (const FabricTransportEndpointRef output : outputs)
      graph.addArc(
          transportSignalKey({output, HandshakeSignalKind::Ready}),
          requestValid);
  }

  for (const auto &provider : memorySelection.providers) {
    const FabricMemoryEndpointRef subordinate = provider.subordinate.underlying();
    activeSubordinates.insert(byteString(canonicalFabricBytes(subordinate)));
    const std::string subordinateRequestValid = memorySignalKey(
        subordinate, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Valid);
    const std::string subordinateRequestReady = memorySignalKey(
        subordinate, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Ready);
    const std::string subordinateResponseReady = memorySignalKey(
        subordinate, MemoryServiceHandshakeChannel::Response,
        HandshakeSignalKind::Ready);
    for (const auto &target : provider.targets) {
      const auto *manager = std::get_if<ManagerEndpointRef>(&target);
      if (!manager) {
        graph.addArc(subordinateRequestValid, subordinateRequestReady);
        graph.addArc(subordinateResponseReady, subordinateRequestReady);
        continue;
      }
      const FabricMemoryEndpointRef endpoint = manager->underlying();
      activeManagers.insert(byteString(canonicalFabricBytes(endpoint)));
      const std::string managerRequestValid = memorySignalKey(
          endpoint, MemoryServiceHandshakeChannel::Request,
          HandshakeSignalKind::Valid);
      const std::string managerRequestReady = memorySignalKey(
          endpoint, MemoryServiceHandshakeChannel::Request,
          HandshakeSignalKind::Ready);
      graph.addArc(subordinateRequestValid, managerRequestValid);
      graph.addArc(subordinateResponseReady, managerRequestValid);
      graph.addArc(managerRequestReady, subordinateRequestReady);
      graph.addArc(managerRequestValid, subordinateRequestReady);
    }
  }
  return llvm::Error::success();
}

llvm::Error appendMemoryNetworkArcs(
    const FabricArtifactView &view,
    const std::set<std::string> &activeManagers,
    const std::set<std::string> &activeSubordinates, Graph &graph) {
  std::vector<MemoryNetworkConnection> connections;
  connections.reserve(view.memoryServiceConnections().size() +
                      view.moduleBoundaryMemoryAttachments().size());
  for (const FabricMemoryServiceConnectionPayload &connection :
       view.memoryServiceConnections())
    connections.push_back({connection.source, connection.destination});
  for (const FabricModuleBoundaryMemoryAttachmentView &attachment :
       view.moduleBoundaryMemoryAttachments()) {
    if (attachment.boundary.direction == FabricPortDirection::Input)
      connections.push_back({attachment.endpoint, attachment.boundary});
    else
      connections.push_back({attachment.boundary, attachment.endpoint});
  }

  std::map<std::string, std::vector<MemoryNetworkFace>> sourcesBySink;
  std::set<std::string> connectedActiveManagers;
  for (const MemoryNetworkConnection &connection : connections) {
    bool active = false;
    if (const auto *endpoint =
            std::get_if<FabricMemoryEndpointRef>(&connection.source)) {
      const std::string key = byteString(canonicalFabricBytes(*endpoint));
      active = activeManagers.count(key) != 0;
      if (active)
        connectedActiveManagers.insert(key);
    } else if (const auto *sink =
                   std::get_if<FabricMemoryEndpointRef>(
                       &connection.destination)) {
      active = activeSubordinates.count(
                   byteString(canonicalFabricBytes(*sink))) != 0;
    }
    if (active)
      sourcesBySink[faceKey(connection.destination)].push_back(
          connection.source);
  }
  if (connectedActiveManagers != activeManagers)
    return invalid("selected memory manager has no physical service edge");

  for (const auto &[sinkKey, sources] : sourcesBySink) {
    const std::string sinkIdentity = sinkKey;
    const auto connection = llvm::find_if(
        connections, [&](const auto &candidate) {
          return faceKey(candidate.destination) == sinkIdentity;
        });
    if (connection == connections.end())
      return invalid("memory service network sink is stale");
    const MemoryNetworkFace &sink = connection->destination;
    const std::string sinkRequestValid = faceSignalKey(
        sink, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Valid);
    const std::string sinkRequestReady = faceSignalKey(
        sink, MemoryServiceHandshakeChannel::Request,
        HandshakeSignalKind::Ready);
    const std::string sinkResponseValid = faceSignalKey(
        sink, MemoryServiceHandshakeChannel::Response,
        HandshakeSignalKind::Valid);
    const std::string sinkResponseReady = faceSignalKey(
        sink, MemoryServiceHandshakeChannel::Response,
        HandshakeSignalKind::Ready);
    for (const MemoryNetworkFace &source : sources) {
      const std::string sourceRequestValid = faceSignalKey(
          source, MemoryServiceHandshakeChannel::Request,
          HandshakeSignalKind::Valid);
      const std::string sourceRequestReady = faceSignalKey(
          source, MemoryServiceHandshakeChannel::Request,
          HandshakeSignalKind::Ready);
      const std::string sourceResponseValid = faceSignalKey(
          source, MemoryServiceHandshakeChannel::Response,
          HandshakeSignalKind::Valid);
      const std::string sourceResponseReady = faceSignalKey(
          source, MemoryServiceHandshakeChannel::Response,
          HandshakeSignalKind::Ready);
      graph.addArc(sourceRequestValid, sinkRequestValid);
      graph.addArc(sinkRequestReady, sourceRequestReady);
      graph.addArc(sinkResponseValid, sourceResponseValid);
      graph.addArc(sourceResponseReady, sinkResponseReady);
      for (const MemoryNetworkFace &readySource : sources)
        graph.addArc(sourceRequestValid,
                     faceSignalKey(readySource,
                                   MemoryServiceHandshakeChannel::Request,
                                   HandshakeSignalKind::Ready));
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::string>>
topologicalOrder(const Graph &graph, ExecutionControlView executionControl) {
  std::map<std::string, std::size_t> indegree;
  std::map<std::string, std::vector<std::string>> adjacency;
  for (const std::string &node : graph.nodes)
    indegree.emplace(node, 0);
  for (const auto &[source, destination] : graph.arcs) {
    adjacency[source].push_back(destination);
    ++indegree[destination];
  }
  std::vector<std::string> ready;
  for (const auto &[node, degree] : indegree)
    if (degree == 0)
      ready.push_back(node);
  std::vector<std::string> order;
  order.reserve(graph.nodes.size());
  while (!ready.empty()) {
    if ((order.size() & 4095U) == 0 && executionControl.stopRequested())
      return invalid("memory-service handshake closure was interrupted");
    std::string node = std::move(ready.back());
    ready.pop_back();
    order.push_back(node);
    for (const std::string &destination : adjacency[node])
      if (--indegree[destination] == 0)
        ready.push_back(destination);
  }
  if (order.size() != graph.nodes.size())
    return invalid("SelectedCombinationalHandshakeCycle");
  return order;
}

llvm::Expected<std::pair<Graph, std::map<std::string, std::string>>>
buildGraph(const FabricArtifactView &view,
           const FabricHandshakeSelection &transportSelection,
           const FabricMemoryServiceHandshakeSelection &memorySelection,
           const FabricHandshakeContext &context,
           ExecutionControlView executionControl) {
  if (view.rootKind() != FabricRootKind::Module)
    return invalid("memory-service boundary projection requires a Module");
  if (llvm::Error error = validateSelection(view, transportSelection,
                                            memorySelection))
    return std::move(error);

  std::map<std::string, HandshakeSignalRef> transportTerminals;
  const auto addTransportTerminal = [&](FabricTransportEndpointRef endpoint) {
    for (HandshakeSignalKind signal : {HandshakeSignalKind::Valid,
                                       HandshakeSignalKind::Ready}) {
      HandshakeSignalRef reference{endpoint, signal};
      transportTerminals.try_emplace(transportSignalKey(reference), reference);
    }
  };
  std::map<std::string, std::string> boundaryAliases;
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       view.moduleBoundaryTransportAttachments()) {
    addTransportTerminal(attachment.endpoint);
    for (HandshakeSignalKind signal : {HandshakeSignalKind::Valid,
                                       HandshakeSignalKind::Ready})
      boundaryAliases.emplace(
          boundarySignalKey(attachment.boundary, std::nullopt, signal),
          transportSignalKey({attachment.endpoint, signal}));
  }
  for (const FabricMemoryHandshakeSelection &operation :
       transportSelection.memoryOperations) {
    const auto *capability =
        view.memoryCapabilityAlternative(operation.capability());
    if (!capability)
      return invalid("selected memory operation capability is stale");
    const FabricTransportEndpointOwnerRef owner =
        FabricTransportEndpointOwnerRef::of(operation.capability().port.memory);
    for (std::size_t role = 0; role != operation.roleSources().size(); ++role) {
      if (!operation.roleSources()[role] &&
          !operation.roleDestinations()[role])
        continue;
      const auto *binding = bindingForRole(*capability, role);
      if (!binding)
        return invalid("selected memory role has no token endpoint");
      addTransportTerminal({owner, binding->endpointOrdinal});
    }
  }

  std::vector<HandshakeSignalRef> terminalValues;
  terminalValues.reserve(transportTerminals.size());
  for (const auto &[key, signal] : transportTerminals) {
    (void)key;
    terminalValues.push_back(signal);
  }
  auto transportReachability = deriveSelectedHandshakeReachability(
      view, transportSelection, terminalValues, context, executionControl);
  if (!transportReachability)
    return transportReachability.takeError();

  Graph graph;
  for (const auto &[key, signal] : transportTerminals) {
    (void)signal;
    graph.addNode(key);
  }
  for (const HandshakeDependencyArc &arc : *transportReachability)
    graph.addArc(transportSignalKey(arc.source),
                 transportSignalKey(arc.destination));

  for (const FabricModuleBoundaryTransportPassthroughView &passthrough :
       view.moduleBoundaryTransportPassthroughs()) {
    const auto node = [&](FabricModuleBoundaryEndpointRef boundary,
                          HandshakeSignalKind signal) {
      const std::string key = boundarySignalKey(boundary, std::nullopt, signal);
      const auto alias = boundaryAliases.find(key);
      return alias == boundaryAliases.end() ? key : alias->second;
    };
    graph.addArc(node(passthrough.input, HandshakeSignalKind::Valid),
                 node(passthrough.output, HandshakeSignalKind::Valid));
    graph.addArc(node(passthrough.output, HandshakeSignalKind::Ready),
                 node(passthrough.input, HandshakeSignalKind::Ready));
  }

  std::set<std::string> activeManagers;
  std::set<std::string> activeSubordinates;
  if (llvm::Error error = appendMemoryOwnerArcs(
          view, transportSelection, memorySelection, activeManagers,
          activeSubordinates, graph))
    return std::move(error);
  if (llvm::Error error = appendMemoryNetworkArcs(
          view, activeManagers, activeSubordinates, graph))
    return std::move(error);

  return std::make_pair(std::move(graph), std::move(boundaryAliases));
}

} // namespace

llvm::Expected<std::vector<ModuleBoundaryHandshakeDependencyArc>>
deriveSelectedModuleBoundaryHandshakeReachability(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection,
    const FabricHandshakeContext &context,
    ExecutionControlView executionControl) {
  auto built = buildGraph(view, transportSelection, memorySelection, context,
                          executionControl);
  if (!built)
    return built.takeError();
  Graph &graph = built->first;
  const auto &aliases = built->second;
  auto order = topologicalOrder(graph, executionControl);
  if (!order)
    return order.takeError();

  std::vector<ModuleBoundaryHandshakeSignalRef> terminals;
  const auto module = view.moduleRootTemplate();
  if (!module)
    return invalid("memory-service boundary projection has no Module root");
  for (FabricPortDirection direction : {FabricPortDirection::Input,
                                        FabricPortDirection::Output}) {
    const std::uint64_t count =
        view.moduleBoundaryEndpointCount(*module, direction);
    for (FabricOrdinal ordinal = 0; ordinal != count; ++ordinal) {
      const FabricModuleBoundaryEndpointRef boundary{*module, direction,
                                                      ordinal};
      const auto plane = view.moduleBoundaryEndpointPlane(boundary);
      if (!plane)
        return invalid("Module boundary handshake endpoint is stale");
      if (*plane == FabricSpatialAttachmentEndpointRef::Plane::Transport) {
        for (HandshakeSignalKind signal : {HandshakeSignalKind::Valid,
                                           HandshakeSignalKind::Ready})
          terminals.push_back({boundary, std::nullopt, signal});
      } else {
        for (MemoryServiceHandshakeChannel channel :
             {MemoryServiceHandshakeChannel::Request,
              MemoryServiceHandshakeChannel::Response})
          for (HandshakeSignalKind signal : {HandshakeSignalKind::Valid,
                                             HandshakeSignalKind::Ready})
            terminals.push_back({boundary, channel, signal});
      }
    }
  }

  const auto graphKey = [&](const ModuleBoundaryHandshakeSignalRef &signal) {
    const std::string key = boundarySignalKey(signal);
    const auto alias = aliases.find(key);
    return alias == aliases.end() ? key : alias->second;
  };
  std::map<std::string, std::vector<std::string>> adjacency;
  for (const auto &[source, destination] : graph.arcs)
    adjacency[source].push_back(destination);

  std::vector<ModuleBoundaryHandshakeDependencyArc> result;
  for (const ModuleBoundaryHandshakeSignalRef &source : terminals) {
    if (executionControl.stopRequested())
      return invalid("memory-service boundary reachability was interrupted");
    const std::string sourceKey = graphKey(source);
    if (!graph.nodes.count(sourceKey))
      continue;
    std::set<std::string> visited{sourceKey};
    std::vector<std::string> worklist{sourceKey};
    while (!worklist.empty()) {
      std::string node = std::move(worklist.back());
      worklist.pop_back();
      for (const std::string &destination : adjacency[node])
        if (visited.insert(destination).second)
          worklist.push_back(destination);
    }
    for (const ModuleBoundaryHandshakeSignalRef &destination : terminals)
      if (!(source == destination) && visited.count(graphKey(destination)))
        result.push_back({source, destination});
  }
  return result;
}

llvm::Error verifySelectedMemoryServiceHandshakeAcyclic(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection,
    const FabricHandshakeContext &context,
    ExecutionControlView executionControl) {
  auto built = buildGraph(view, transportSelection, memorySelection, context,
                          executionControl);
  if (!built)
    return built.takeError();
  auto order = topologicalOrder(built->first, executionControl);
  if (!order)
    return order.takeError();
  return llvm::Error::success();
}

} // namespace loom::fabric
