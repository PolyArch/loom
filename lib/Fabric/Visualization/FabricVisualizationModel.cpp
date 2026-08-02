#include "FabricVisualizationInternal.h"

#include "../Artifact/FabricArtifactBytecodeInternal.h"
#include "Common/ArtifactText.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace loom::fabric::visualization {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_visualization_invalid: " + message);
}

std::string entityLabel(FabricEntityKind kind, FabricEntityId id) {
  std::string prefix;
  switch (kind) {
  case FabricEntityKind::FabricModuleTemplate:
    prefix = "SpatialCore";
    break;
  case FabricEntityKind::FabricPeOccurrence:
    prefix = "PE";
    break;
  case FabricEntityKind::FabricFuTemplate:
    prefix = "FU template";
    break;
  case FabricEntityKind::FabricFuOccurrence:
    prefix = "FU";
    break;
  case FabricEntityKind::FabricMemoryOccurrence:
    prefix = "Memory";
    break;
  case FabricEntityKind::FabricSwitchOccurrence:
    prefix = "Switch";
    break;
  case FabricEntityKind::FabricFifoOccurrence:
    prefix = "FIFO";
    break;
  case FabricEntityKind::FabricBoundaryOccurrence:
    prefix = "Boundary";
    break;
  case FabricEntityKind::HostCoreOccurrence:
    prefix = "HostCore";
    break;
  case FabricEntityKind::AccCoreOccurrence:
    prefix = "AccCore";
    break;
  case FabricEntityKind::SystemMemoryService:
    prefix = "Memory service";
    break;
  case FabricEntityKind::SystemServiceEndpoint:
    prefix = "Service endpoint";
    break;
  case FabricEntityKind::SystemServiceTransform:
    prefix = "Service transform";
    break;
  case FabricEntityKind::SystemTransportResource:
    prefix = "Transport";
    break;
  case FabricEntityKind::HardwareDomain:
    prefix = "Hardware domain";
    break;
  case FabricEntityKind::ExternalBoundary:
    prefix = "External boundary";
    break;
  case FabricEntityKind::FabricMemoryEngineTemplate:
    prefix = "Memory engine template";
    break;
  }
  return prefix + " " + std::to_string(id);
}

std::optional<FabricEntityId>
transportOwnerEntity(const FabricTransportEndpointOwnerRef &owner) {
  return std::visit(
      [](const auto &reference) -> std::optional<FabricEntityId> {
        using T = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<T, SpatialCoreOccurrenceRef>)
          return reference.core.id();
        else
          return reference.id();
      },
      owner.payload);
}

std::optional<FabricEntityId>
memoryOwnerEntity(const FabricMemoryEndpointOwnerRef &owner) {
  return std::visit(
      [](const auto &reference) -> std::optional<FabricEntityId> {
        using T = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<T, SpatialCoreOccurrenceRef>)
          return reference.core.id();
        else
          return reference.id();
      },
      owner.payload);
}

std::optional<FabricEntityId>
inventoryOwnerEntity(const FabricInventoryOwnerRef &owner) {
  return std::visit(
      [](const auto &reference) -> std::optional<FabricEntityId> {
        using T = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<T, SpatialCoreOccurrenceRef> ||
                      std::is_same_v<T, InstructionCoreContextRef>) {
          return reference.core.id();
        } else if constexpr (std::is_same_v<T, InstructionContextRef>) {
          return reference.pe.id();
        } else if constexpr (std::is_same_v<T, FabricFuTemplateNodeRef>) {
          return reference.fu.id();
        } else if constexpr (std::is_same_v<T, FabricFuOccurrenceNodeRef>) {
          return reference.fu.id();
        } else if constexpr (std::is_same_v<T, FabricMemoryOperationPortRef>) {
          return reference.memory.id();
        } else if constexpr (std::is_same_v<T, FabricMemoryServiceRef>) {
          return std::visit(
              [](const auto &service) -> FabricEntityId {
                return service.id();
              },
              reference.payload);
        } else if constexpr (std::is_same_v<T, FabricTransferPatternRef>) {
          return reference.resource.id();
        } else if constexpr (std::is_same_v<T, FabricFuTemplateNodeRef> ||
                             std::is_same_v<T, FabricFuOccurrenceNodeRef>) {
          return std::nullopt;
        } else {
          return reference.id();
        }
      },
      owner.payload);
}

std::optional<FabricEntityId> moduleEntityId(mlir::Operation *operation) {
  auto id = operation->getAttrOfType<::fabric::EntityIdAttr>(
      ::fabric::kEntityIdAttrName);
  if (!id)
    return std::nullopt;
  return id.getId();
}

std::string typeText(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return result;
}

void addEdge(Graph &graph,
             std::set<std::tuple<std::size_t, std::size_t, std::string>> &seen,
             std::size_t source, std::size_t destination, llvm::StringRef kind,
             llvm::StringRef label = {}) {
  std::string kindText = kind.str();
  if (!seen.emplace(source, destination, kindText).second)
    return;
  graph.edges.push_back(
      {source, destination, label.str(), std::move(kindText), {}});
}

std::string moduleNodeDetail(mlir::Operation *operation) {
  if (auto pe = mlir::dyn_cast<::fabric::PeOp>(operation)) {
    std::size_t fuCount = 0;
    for (mlir::Operation &nested : pe.getBody().front())
      fuCount += mlir::isa<::fabric::FuOp>(nested);
    return ::fabric::stringifySchedule(pe.getSchedule()).str() + " | " +
           std::to_string(fuCount) + " FU" + (fuCount == 1 ? "" : "s");
  }
  if (auto memory = mlir::dyn_cast<::fabric::MemOp>(operation)) {
    std::string detail = "memory engine";
    if (memory.getMemoryContract().getLocalService())
      detail += " + local service";
    return detail;
  }
  if (auto sw = mlir::dyn_cast<::fabric::SwitchOp>(operation))
    return ::fabric::stringifySchedule(sw.getSchedule()).str() + " | " +
           std::to_string(sw.getNumOperands()) + " in / " +
           std::to_string(sw.getNumResults()) + " out";
  if (auto fifo = mlir::dyn_cast<::fabric::FifoOp>(operation))
    return "depth " + std::to_string(fifo.getMaxDepth()) +
           (fifo.getBypassable() ? " | bypassable" : " | buffered");
  if (auto boundary = mlir::dyn_cast<::fabric::BoundaryOp>(operation))
    return ::fabric::stringifyBoundaryDirection(boundary.getDirection()).str();
  return operation->getName().getStringRef().str();
}

bool isModuleResource(mlir::Operation *operation) {
  return mlir::isa<::fabric::PeOp, ::fabric::MemOp, ::fabric::SwitchOp,
                   ::fabric::FifoOp, ::fabric::BoundaryOp>(operation);
}

llvm::Expected<Graph>
buildModuleGraph(const FinalizedFabricRoot &root,
                 detail::ParsedFabricBytecodeModule &parsed) {
  mlir::Operation &only = parsed.module->getBody()->front();
  auto module = mlir::dyn_cast<::fabric::ModuleOp>(&only);
  if (!module)
    return invalid("Module projection received a non-Module root");

  const std::string identity =
      formatArtifactIdentityHex(root.reference().artifact);
  Graph graph{"module-" + identity.substr(0, 16),
              "SpatialCore " + identity.substr(0, 12),
              "Canonical Fabric module | " + identity,
              "spatial-core",
              identity,
              {},
              {},
              0.0,
              0.0};
  llvm::DenseMap<mlir::Operation *, std::size_t> nodeByOperation;
  llvm::DenseMap<FabricEntityId, std::size_t> nodeByEntity;

  mlir::Block &body = module.getBody().front();
  for (std::size_t ordinal = 0; ordinal < body.getNumArguments(); ++ordinal) {
    const std::size_t index = graph.nodes.size();
    graph.nodes.push_back(
        {"input-" + std::to_string(ordinal), "Input " + std::to_string(ordinal),
         typeText(body.getArgument(ordinal).getType()), "fabric.module_input"});
    (void)index;
  }

  for (mlir::Operation &operation : body) {
    if (!isModuleResource(&operation))
      continue;
    auto id = moduleEntityId(&operation);
    if (!id)
      return invalid("canonical Module resource has no entity ID");
    auto kind = root.view().entityKind(*id);
    if (!kind)
      return invalid("canonical Module resource has an unknown entity ID");
    const std::size_t index = graph.nodes.size();
    graph.nodes.push_back(
        {"entity-" + std::to_string(*id), entityLabel(*kind, *id),
         moduleNodeDetail(&operation), fabricRefKeyword(*kind).str()});
    nodeByOperation[&operation] = index;
    nodeByEntity[*id] = index;
  }

  auto yield = mlir::dyn_cast<::fabric::YieldOp>(body.getTerminator());
  if (!yield)
    return invalid("canonical Module has no Fabric yield");
  const std::size_t outputBase = graph.nodes.size();
  for (auto [ordinal, value] : llvm::enumerate(yield.getOperands()))
    graph.nodes.push_back({"output-" + std::to_string(ordinal),
                           "Output " + std::to_string(ordinal),
                           typeText(value.getType()), "fabric.module_output"});

  std::set<std::tuple<std::size_t, std::size_t, std::string>> seen;
  for (const auto &entry : nodeByOperation) {
    mlir::Operation *destination = entry.first;
    for (mlir::Value operand : destination->getOperands()) {
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        if (argument.getOwner() == &body)
          addEdge(graph, seen, argument.getArgNumber(), entry.second,
                  "transport");
        continue;
      }
      auto source = nodeByOperation.find(operand.getDefiningOp());
      if (source != nodeByOperation.end())
        addEdge(graph, seen, source->second, entry.second, "transport");
    }
  }
  for (auto [ordinal, value] : llvm::enumerate(yield.getOperands())) {
    const std::size_t destination = outputBase + ordinal;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() == &body)
        addEdge(graph, seen, argument.getArgNumber(), destination, "transport");
      continue;
    }
    auto source = nodeByOperation.find(value.getDefiningOp());
    if (source != nodeByOperation.end())
      addEdge(graph, seen, source->second, destination, "transport");
  }
  return graph;
}

std::string fuNodeLabel(mlir::Operation *operation) {
  if (auto op = mlir::dyn_cast<::fabric::OpOp>(operation)) {
    if (auto family = op.getImplementationFamily())
      return ::fabric::stringifyImplementationFamilyId(*family).str();
    return "Operation";
  }
  if (mlir::isa<::fabric::MuxOp>(operation))
    return "Mux";
  if (mlir::isa<::fabric::DemuxOp>(operation))
    return "Demux";
  return operation->getName().getStringRef().str();
}

Graph buildFuGraph(const ArtifactIdentity &artifact, ::fabric::FuOp fu,
                   FabricEntityId templateId) {
  const std::string identity = formatArtifactIdentityHex(artifact);
  Graph graph{"fu-" + identity.substr(0, 12) + "-" + std::to_string(templateId),
              "FU template " + std::to_string(templateId),
              "Configured physical graph | SpatialCore " +
                  identity.substr(0, 12),
              "fu",
              identity,
              {},
              {},
              0.0,
              0.0};
  mlir::Block &body = fu.getBody().front();
  llvm::DenseMap<mlir::Operation *, std::size_t> nodeByOperation;
  for (std::size_t ordinal = 0; ordinal < body.getNumArguments(); ++ordinal)
    graph.nodes.push_back(
        {"input-" + std::to_string(ordinal), "Input " + std::to_string(ordinal),
         typeText(body.getArgument(ordinal).getType()), "fabric.fu_input"});
  std::size_t operationOrdinal = 0;
  for (mlir::Operation &operation : body) {
    if (mlir::isa<::fabric::YieldOp>(operation))
      continue;
    const std::size_t index = graph.nodes.size();
    graph.nodes.push_back(
        {"node-" + std::to_string(operationOrdinal++), fuNodeLabel(&operation),
         operation.getName().getStringRef().str(), "fabric.fu_node"});
    nodeByOperation[&operation] = index;
  }
  auto yield = mlir::dyn_cast<::fabric::YieldOp>(body.getTerminator());
  const std::size_t outputBase = graph.nodes.size();
  if (yield)
    for (auto [ordinal, value] : llvm::enumerate(yield.getOperands()))
      graph.nodes.push_back({"output-" + std::to_string(ordinal),
                             "Output " + std::to_string(ordinal),
                             typeText(value.getType()), "fabric.fu_output"});

  std::set<std::tuple<std::size_t, std::size_t, std::string>> seen;
  for (const auto &entry : nodeByOperation)
    for (mlir::Value operand : entry.first->getOperands()) {
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        if (argument.getOwner() == &body)
          addEdge(graph, seen, argument.getArgNumber(), entry.second,
                  "fu-route");
      } else if (auto source = nodeByOperation.find(operand.getDefiningOp());
                 source != nodeByOperation.end()) {
        addEdge(graph, seen, source->second, entry.second, "fu-route");
      }
    }
  if (yield)
    for (auto [ordinal, value] : llvm::enumerate(yield.getOperands())) {
      const std::size_t destination = outputBase + ordinal;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        if (argument.getOwner() == &body)
          addEdge(graph, seen, argument.getArgNumber(), destination,
                  "fu-route");
      } else if (auto source = nodeByOperation.find(value.getDefiningOp());
                 source != nodeByOperation.end()) {
        addEdge(graph, seen, source->second, destination, "fu-route");
      }
    }
  return graph;
}

llvm::Expected<std::vector<Graph>>
buildModuleGraphs(const FinalizedFabricRoot &root) {
  auto decoded = decodeFabricArtifactEnvelope(root.canonicalBytes().bytes());
  if (!decoded)
    return decoded.takeError();
  auto parsed =
      detail::parseFabricBytecodeModule(decoded->canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  auto moduleGraph = buildModuleGraph(root, *parsed);
  if (!moduleGraph)
    return moduleGraph.takeError();

  std::vector<Graph> result;
  result.push_back(std::move(*moduleGraph));
  std::map<FabricEntityId, ::fabric::FuOp> representativeByTemplate;
  parsed->module->walk([&](::fabric::FuOp fu) {
    auto templateId = fu->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kFuTemplateIdAttrName);
    if (templateId)
      representativeByTemplate.try_emplace(templateId.getId(), fu);
  });
  for (const auto &entry : representativeByTemplate)
    result.push_back(
        buildFuGraph(root.reference().artifact, entry.second, entry.first));
  return result;
}

llvm::Expected<Graph> buildSystemGraph(const FinalizedFabricRoot &root) {
  auto system = requireSystemRoot(root.view());
  if (!system)
    return system.takeError();
  const std::string identity =
      formatArtifactIdentityHex(root.reference().artifact);
  Graph graph{"system-" + identity.substr(0, 16),
              "Fabric System",
              "Multi-AccCore architecture | " + identity,
              "system",
              identity,
              {},
              {},
              0.0,
              0.0};
  llvm::DenseMap<FabricEntityId, std::size_t> nodeByEntity;
  for (FabricEntityId id = 0;; ++id) {
    auto kind = root.view().entityKind(id);
    if (!kind)
      break;
    std::string detail = "Entity " + std::to_string(id);
    if (*kind == FabricEntityKind::AccCoreOccurrence)
      detail = "InstructionCore + SpatialCore";
    else if (*kind == FabricEntityKind::HardwareDomain)
      if (auto domainKind =
              root.view().hardwareDomainKind(HardwareDomainRef(id)))
        detail = fabricRefKeyword(*domainKind).str() + " domain";
    const std::size_t index = graph.nodes.size();
    graph.nodes.push_back({"entity-" + std::to_string(id),
                           entityLabel(*kind, id), std::move(detail),
                           fabricRefKeyword(*kind).str()});
    nodeByEntity[id] = index;
  }

  std::vector<std::size_t> moduleNodes;
  for (auto [ordinal, dependency] :
       llvm::enumerate(root.directDependencies())) {
    if (dependency.role != FabricDependencyRole::ImportedModule)
      continue;
    const std::string dependencyIdentity =
        formatArtifactIdentityHex(dependency.root.artifact);
    moduleNodes.resize(std::max(moduleNodes.size(), ordinal + 1));
    moduleNodes[ordinal] = graph.nodes.size();
    graph.nodes.push_back({"module-" + std::to_string(ordinal),
                           "SpatialCore type " + std::to_string(ordinal),
                           dependencyIdentity, "fabric.module_dependency"});
  }

  std::set<std::tuple<std::size_t, std::size_t, std::string>> seen;
  for (const FabricPointConnectionPayload &connection :
       root.view().pointConnections()) {
    auto sourceId = transportOwnerEntity(connection.source.owner);
    auto destinationId = transportOwnerEntity(connection.destination.owner);
    if (!sourceId || !destinationId)
      continue;
    auto source = nodeByEntity.find(*sourceId);
    auto destination = nodeByEntity.find(*destinationId);
    if (source != nodeByEntity.end() && destination != nodeByEntity.end())
      addEdge(graph, seen, source->second, destination->second, "transport");
  }

  for (const HardwareDomainRef &domain : system->hardwareDomains()) {
    auto domainNode = nodeByEntity.find(domain.id());
    if (domainNode == nodeByEntity.end())
      continue;
    for (const FabricInventoryOwnerRef &member :
         system->hardwareDomainMembers(domain)) {
      auto memberId = inventoryOwnerEntity(member);
      if (!memberId)
        continue;
      auto memberNode = nodeByEntity.find(*memberId);
      if (memberNode != nodeByEntity.end() &&
          memberNode->second != domainNode->second)
        addEdge(graph, seen, domainNode->second, memberNode->second, "domain");
    }
  }

  for (const FabricSpatialAttachmentRecordView &attachment :
       system->spatialAttachments()) {
    std::optional<FabricEntityId> core;
    if (const auto *transport = attachment.spatialEndpoint.transport())
      core = transportOwnerEntity(transport->owner);
    else if (const auto *memory = attachment.spatialEndpoint.memory())
      core = memoryOwnerEntity(memory->owner);
    if (!core ||
        attachment.moduleEndpoint.dependencyOrdinal >= moduleNodes.size())
      continue;
    auto source = nodeByEntity.find(*core);
    if (source != nodeByEntity.end())
      addEdge(graph, seen, source->second,
              moduleNodes[attachment.moduleEndpoint.dependencyOrdinal],
              "attachment", "uses");
  }
  return graph;
}

bool isHiddenNocDetailNode(const Node &node) {
  return node.kind == "fabric.system_transport_resource" ||
         node.kind == "fabric.hardware_domain";
}

bool isSystemOverviewNode(const Node &node) {
  return node.kind == "fabric.host_core_occurrence" ||
         node.kind == "fabric.acc_core_occurrence" ||
         node.kind == "fabric.system_memory_service" ||
         node.kind == "fabric.external_boundary";
}

Graph buildSystemNocGraph(const Graph &detail) {
  Graph graph{"system-noc-" + detail.artifactIdentity.substr(0, 16),
              "NoC Topology",
              "Directed System connectivity | " + detail.artifactIdentity,
              "system-noc",
              detail.artifactIdentity,
              {},
              {},
              0.0,
              0.0};
  std::vector<std::optional<std::size_t>> projected(detail.nodes.size());
  for (auto [index, node] : llvm::enumerate(detail.nodes)) {
    if (!isSystemOverviewNode(node))
      continue;
    projected[index] = graph.nodes.size();
    graph.nodes.push_back(node);
  }

  std::vector<std::vector<std::size_t>> transportAdjacency(detail.nodes.size());
  std::set<std::tuple<std::size_t, std::size_t, std::string>> seen;
  for (const Edge &edge : detail.edges) {
    if (edge.kind == "transport") {
      transportAdjacency[edge.source].push_back(edge.destination);
      continue;
    }
    if (projected[edge.source] && projected[edge.destination])
      addEdge(graph, seen, *projected[edge.source],
              *projected[edge.destination], edge.kind, edge.label);
  }

  for (std::size_t source = 0; source != detail.nodes.size(); ++source) {
    if (!projected[source])
      continue;
    llvm::SmallVector<std::size_t, 8> pending(
        transportAdjacency[source].begin(), transportAdjacency[source].end());
    std::set<std::size_t> visited;
    while (!pending.empty()) {
      const std::size_t destination = pending.pop_back_val();
      if (projected[destination]) {
        addEdge(graph, seen, *projected[source], *projected[destination],
                "transport", "NoC");
        continue;
      }
      if (!isHiddenNocDetailNode(detail.nodes[destination]) ||
          !visited.insert(destination).second)
        continue;
      pending.append(transportAdjacency[destination].begin(),
                     transportAdjacency[destination].end());
    }
  }
  return graph;
}

Graph buildSystemOverviewGraph(const Graph &detail, const Graph &noc) {
  const std::size_t transportCount =
      llvm::count_if(detail.nodes, [](const Node &node) {
        return node.kind == "fabric.system_transport_resource";
      });
  const std::size_t accCoreCount =
      llvm::count_if(detail.nodes, [](const Node &node) {
        return node.kind == "fabric.acc_core_occurrence";
      });
  Graph graph{"system-overview-" + detail.artifactIdentity.substr(0, 16),
              "System Architecture",
              std::to_string(accCoreCount) + " AccCores | " +
                  std::to_string(transportCount) +
                  " directed NoC resources | " + detail.artifactIdentity,
              "system-overview",
              detail.artifactIdentity,
              {},
              {},
              0.0,
              0.0};
  std::vector<std::optional<std::size_t>> projected(noc.nodes.size());
  for (auto [index, node] : llvm::enumerate(noc.nodes)) {
    if (!isSystemOverviewNode(node))
      continue;
    projected[index] = graph.nodes.size();
    graph.nodes.push_back(node);
  }

  std::optional<std::size_t> nocNode;
  if (transportCount != 0) {
    nocNode = graph.nodes.size();
    graph.nodes.push_back(
        {"noc", "NoC",
         std::to_string(transportCount) + " directed transport resources",
         "visual.noc_summary"});
  }

  std::set<std::tuple<std::size_t, std::size_t, std::string>> seen;
  std::set<std::size_t> nocParticipants;
  for (const Edge &edge : noc.edges) {
    if (edge.kind == "transport" && nocNode) {
      if (projected[edge.source])
        nocParticipants.insert(*projected[edge.source]);
      if (projected[edge.destination])
        nocParticipants.insert(*projected[edge.destination]);
      continue;
    }
    if (projected[edge.source] && projected[edge.destination])
      addEdge(graph, seen, *projected[edge.source],
              *projected[edge.destination], edge.kind, edge.label);
  }
  if (nocNode)
    for (std::size_t participant : nocParticipants)
      addEdge(graph, seen, participant, *nocNode, "summary");
  return graph;
}

} // namespace

llvm::Expected<Document> buildDocument(const FinalizedFabricRoot &root,
                                       const ArtifactStore &store) {
  Document document;
  document.rootIdentity = formatArtifactIdentityHex(root.reference().artifact);
  document.title = root.view().rootKind() == FabricRootKind::System
                       ? "Loom Fabric System"
                       : "Loom SpatialCore";

  if (root.view().rootKind() == FabricRootKind::Module) {
    auto graphs = buildModuleGraphs(root);
    if (!graphs)
      return graphs.takeError();
    document.graphs = std::move(*graphs);
  } else if (root.view().rootKind() == FabricRootKind::System) {
    auto systemDetail = buildSystemGraph(root);
    if (!systemDetail)
      return systemDetail.takeError();
    systemDetail->id =
        "system-detail-" + systemDetail->artifactIdentity.substr(0, 16);
    systemDetail->title = "Fabric Detail";
    systemDetail->subtitle = "Exact resources, endpoints, and domains | " +
                             systemDetail->artifactIdentity;
    Graph noc = buildSystemNocGraph(*systemDetail);
    document.graphs.push_back(buildSystemOverviewGraph(*systemDetail, noc));
    document.graphs.push_back(std::move(noc));
    document.graphs.push_back(std::move(*systemDetail));
    std::set<std::string> imported;
    for (const FabricDirectDependency &dependency : root.directDependencies()) {
      if (dependency.role != FabricDependencyRole::ImportedModule)
        continue;
      const std::string identity =
          formatArtifactIdentityHex(dependency.root.artifact);
      if (!imported.insert(identity).second)
        continue;
      auto module = importEntireFabricRoot(dependency.root, store);
      if (!module)
        return module.takeError();
      auto graphs = buildModuleGraphs(*module);
      if (!graphs)
        return graphs.takeError();
      document.graphs.insert(document.graphs.end(),
                             std::make_move_iterator(graphs->begin()),
                             std::make_move_iterator(graphs->end()));
    }
  } else {
    return invalid("InterconnectImplementation visualization is unavailable");
  }

  for (Graph &graph : document.graphs) {
    if (graph.kind == "system-overview")
      computeSystemOverviewLayout(graph);
    else
      computeLayeredLayout(graph);
  }
  return document;
}

} // namespace loom::fabric::visualization
