#include "PnR/FrozenRealizationGraph.h"
#include "FrozenComputeDomains.h"

#include "Mapping/Verifier.h"
#include "PnR/PnrProblemInputs.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

char FrozenMappingInfeasibility::ID;

void FrozenMappingInfeasibility::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code FrozenMappingInfeasibility::convertToErrorCode() const {
  return std::make_error_code(std::errc::operation_not_permitted);
}

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenRealizationGraph";

constexpr PnrCapacityContext actorCountContext{
    frozenArtifact, "actor_ownerships", "actors", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext computeCountContext{
    frozenArtifact, "compute_realizations", "compute_realizations",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext memoryCountContext{
    frozenArtifact, "memory_realizations", "memory_realizations",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext externalEdgeCountContext{
    frozenArtifact, "external_edges", "canonical_edges",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext terminalCountContext{
    frozenArtifact, "template_terminals", "template_terminals",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext terminalIndexContext{
    frozenArtifact, "template_terminals", "template_terminals",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext realizationIndexContext{
    frozenArtifact, "actor_ownerships", "realizations",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext portIndexContext{
    frozenArtifact, "terminals", "ports", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext netCountContext{
    frozenArtifact, "logical_nets", "logical_nets", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext sinkCountContext{
    frozenArtifact, "logical_net_sinks", "logical_net_sinks",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext sinkOffsetContext{frozenArtifact, "logical_nets",
                                               "logical_net_sinks",
                                               PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext obligationCountContext{
    frozenArtifact, "memory_service_obligations", "logical_memory_roots",
    PnrCapacityMeasure::Count};

llvm::Error freezeError(std::string message) {
  return llvm::make_error<llvm::StringError>(
      std::move(message), std::make_error_code(std::errc::invalid_argument));
}

std::uint64_t sizeValue(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(size);
}

llvm::Error preflight(PnrCapacityContext context, std::size_t size) {
  return preflightPnrIndexCapacity(context, sizeValue(size));
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, sizeValue(value));
}

llvm::Expected<PnrIndex> checkedPort(std::uint32_t value) {
  return checkedPnrIndex(portIndexContext, value);
}

llvm::Error addCapacityCount(PnrCapacityContext context, std::uint64_t &total,
                             std::size_t count) {
  auto sum = checkedPnrIndexAdd(context, total, sizeValue(count));
  if (!sum)
    return sum.takeError();
  total = *sum;
  return llvm::Error::success();
}

struct OwnershipInfo {
  FrozenRealizationKind kind;
  PnrIndex realization;
};

struct ComputeTerminalKey {
  ComputeRealizationId realizationId;
  PnrIndex realization;
  FuId fu;
  PortDirection direction;
  std::uint32_t port;
};

struct MemoryTerminalKey {
  MemoryRealizationId realizationId;
  PnrIndex realization;
  MemoryOperationPortTemplateId operation;
  PortDirection direction;
  std::uint32_t port;
};

using TemplateTerminalKey = std::variant<ComputeTerminalKey, MemoryTerminalKey>;

struct GraphTerminalKey {
  GraphId graph;
  PortDirection direction;
  std::uint32_t port;
};

using TerminalKey = std::variant<GraphTerminalKey, TemplateTerminalKey>;

struct EndpointLess {
  bool operator()(const DataflowEndpoint &lhs,
                  const DataflowEndpoint &rhs) const {
    if (lhs.index() != rhs.index())
      return lhs.index() < rhs.index();
    if (const auto *lhsGraph = std::get_if<GraphPort>(&lhs)) {
      const auto &rhsGraph = std::get<GraphPort>(rhs);
      return std::make_tuple(lhsGraph->graph.value(), lhsGraph->direction,
                             lhsGraph->index) <
             std::make_tuple(rhsGraph.graph.value(), rhsGraph.direction,
                             rhsGraph.index);
    }
    const auto &lhsActor = std::get<ActorPort>(lhs);
    const auto &rhsActor = std::get<ActorPort>(rhs);
    return std::make_tuple(lhsActor.actor.value(), lhsActor.direction,
                           lhsActor.index) <
           std::make_tuple(rhsActor.actor.value(), rhsActor.direction,
                           rhsActor.index);
  }
};

bool sameEndpoint(const DataflowEndpoint &lhs, const DataflowEndpoint &rhs) {
  EndpointLess less;
  return !less(lhs, rhs) && !less(rhs, lhs);
}

struct ActorPortLess {
  bool operator()(const ActorPort &lhs, const ActorPort &rhs) const {
    return std::make_tuple(lhs.actor.value(), lhs.direction, lhs.index) <
           std::make_tuple(rhs.actor.value(), rhs.direction, rhs.index);
  }
};

struct TemplateTerminalLess {
  bool operator()(const TemplateTerminalKey &lhs,
                  const TemplateTerminalKey &rhs) const {
    if (lhs.index() != rhs.index())
      return lhs.index() < rhs.index();
    if (const auto *lhsCompute = std::get_if<ComputeTerminalKey>(&lhs)) {
      const auto &rhsCompute = std::get<ComputeTerminalKey>(rhs);
      return std::make_tuple(lhsCompute->fu.value(),
                             lhsCompute->realizationId.value(),
                             lhsCompute->direction, lhsCompute->port) <
             std::make_tuple(rhsCompute.fu.value(),
                             rhsCompute.realizationId.value(),
                             rhsCompute.direction, rhsCompute.port);
    }
    const auto &lhsMemory = std::get<MemoryTerminalKey>(lhs);
    const auto &rhsMemory = std::get<MemoryTerminalKey>(rhs);
    return std::make_tuple(lhsMemory.operation.value(),
                           lhsMemory.realizationId.value(), lhsMemory.direction,
                           lhsMemory.port) <
           std::make_tuple(rhsMemory.operation.value(),
                           rhsMemory.realizationId.value(), rhsMemory.direction,
                           rhsMemory.port);
  }
};

struct ExternalEdge {
  EdgeId edge;
  DataflowEndpoint sourceEndpoint;
  DataflowEndpoint targetEndpoint;
  TerminalKey source;
  TerminalKey target;
};

struct ExternalEdgeLess {
  bool operator()(const ExternalEdge &lhs, const ExternalEdge &rhs) const {
    EndpointLess endpointLess;
    if (endpointLess(lhs.sourceEndpoint, rhs.sourceEndpoint))
      return true;
    if (endpointLess(rhs.sourceEndpoint, lhs.sourceEndpoint))
      return false;
    if (endpointLess(lhs.targetEndpoint, rhs.targetEndpoint))
      return true;
    if (endpointLess(rhs.targetEndpoint, lhs.targetEndpoint))
      return false;
    return lhs.edge.value() < rhs.edge.value();
  }
};

template <typename Descriptor, typename Id>
const Descriptor *findDescriptor(const std::vector<Descriptor> &descriptors,
                                 Id id) {
  auto iterator = std::find_if(
      descriptors.begin(), descriptors.end(),
      [&](const Descriptor &descriptor) { return descriptor.id == id; });
  return iterator == descriptors.end() ? nullptr : &*iterator;
}

llvm::Expected<GraphId> endpointGraph(
    const DataflowEndpoint &endpoint,
    const std::map<std::uint64_t, const ActorDescriptor *> &actorsById) {
  if (const auto *graph = std::get_if<GraphPort>(&endpoint))
    return graph->graph;
  const ActorPort &actorPort = std::get<ActorPort>(endpoint);
  auto actor = actorsById.find(actorPort.actor.value());
  if (actor == actorsById.end())
    return freezeError("cannot freeze realization graph: unresolved actor in "
                       "canonical edge endpoint");
  return actor->second->graph;
}

llvm::Expected<PortDescriptor> graphPortDescriptor(
    const GraphTerminalKey &terminal,
    const std::map<std::uint64_t, const GraphDescriptor *> &graphsById) {
  auto graph = graphsById.find(terminal.graph.value());
  if (graph == graphsById.end())
    return freezeError("cannot freeze realization graph: unresolved graph "
                       "boundary terminal");
  const std::vector<PortDescriptor> &ports =
      terminal.direction == PortDirection::Input ? graph->second->inputPorts
                                                 : graph->second->outputPorts;
  if (terminal.port >= ports.size())
    return freezeError("cannot freeze realization graph: graph boundary "
                       "terminal index is out of range");
  return ports[terminal.port];
}

llvm::Expected<FrozenTemplateTerminal>
freezeTemplateTerminal(const TemplateTerminalKey &terminal) {
  if (const auto *compute = std::get_if<ComputeTerminalKey>(&terminal)) {
    auto port = checkedPort(compute->port);
    if (!port)
      return port.takeError();
    return FrozenComputeTemplateTerminal{compute->realization, compute->fu,
                                         compute->direction, *port};
  }
  const auto &memory = std::get<MemoryTerminalKey>(terminal);
  auto port = checkedPort(memory.port);
  if (!port)
    return port.takeError();
  return FrozenMemoryTemplateTerminal{memory.realization, memory.operation,
                                      memory.direction, *port};
}

llvm::Expected<FrozenTerminal>
freezeTerminal(const TerminalKey &terminal,
               const std::vector<TemplateTerminalKey> &templateTerminals) {
  if (const auto *graph = std::get_if<GraphTerminalKey>(&terminal)) {
    auto port = checkedPort(graph->port);
    if (!port)
      return port.takeError();
    return FrozenGraphBoundaryTerminal{graph->graph, graph->direction, *port};
  }

  const TemplateTerminalKey &templateTerminal =
      std::get<TemplateTerminalKey>(terminal);
  auto iterator =
      std::lower_bound(templateTerminals.begin(), templateTerminals.end(),
                       templateTerminal, TemplateTerminalLess{});
  if (iterator == templateTerminals.end() ||
      TemplateTerminalLess{}(templateTerminal, *iterator) ||
      TemplateTerminalLess{}(*iterator, templateTerminal))
    return freezeError("cannot freeze realization graph: unresolved template "
                       "terminal index");
  auto index =
      checked(terminalIndexContext,
              static_cast<std::size_t>(iterator - templateTerminals.begin()));
  if (!index)
    return index.takeError();
  return FrozenTemplateTerminalRef{*index};
}

} // namespace

llvm::Error loom::pnr::detail::preflightFrozenRangeCapacity(
    PnrCapacityContext context, std::uint64_t offset, std::uint64_t count) {
  auto end = checkedPnrIndexAdd(context, offset, count);
  if (!end)
    return end.takeError();
  return llvm::Error::success();
}

llvm::Error loom::pnr::detail::preflightFrozenRealizationGraphCapacity(
    llvm::ArrayRef<ComputeRealizationDraft> computeRealizations,
    llvm::ArrayRef<MemoryRealizationDraft> memoryRealizations,
    std::uint64_t canonicalEdgeCount) {
  if (llvm::Error error = preflightPnrIndexCapacity(computeCountContext,
                                                    computeRealizations.size()))
    return error;
  if (llvm::Error error = preflightPnrIndexCapacity(memoryCountContext,
                                                    memoryRealizations.size()))
    return error;

  std::uint64_t actorUpperBound = 0;
  for (const ComputeRealizationDraft &realization : computeRealizations) {
    if (llvm::Error error = addCapacityCount(actorCountContext, actorUpperBound,
                                             realization.actors.size()))
      return error;
  }
  for (const MemoryRealizationDraft &realization : memoryRealizations) {
    if (llvm::Error error = addCapacityCount(actorCountContext, actorUpperBound,
                                             realization.actors.size()))
      return error;
  }

  if (llvm::Error error = preflightPnrIndexCapacity(externalEdgeCountContext,
                                                    canonicalEdgeCount))
    return error;
  if (llvm::Error error =
          preflightPnrIndexCapacity(netCountContext, canonicalEdgeCount))
    return error;
  if (llvm::Error error =
          preflightPnrIndexCapacity(sinkCountContext, canonicalEdgeCount))
    return error;
  if (llvm::Error error =
          preflightPnrIndexCapacity(sinkOffsetContext, canonicalEdgeCount))
    return error;

  std::uint64_t memoryRootUpperBound = 0;
  for (const MemoryRealizationDraft &realization : memoryRealizations) {
    if (llvm::Error error =
            addCapacityCount(obligationCountContext, memoryRootUpperBound,
                             realization.roots.size()))
      return error;
  }

  auto templateTerminalUpperBound = checkedPnrIndexMultiply(
      terminalCountContext, canonicalEdgeCount, std::uint64_t{2});
  if (!templateTerminalUpperBound)
    return templateTerminalUpperBound.takeError();
  return llvm::Error::success();
}

llvm::Expected<FrozenRealizationGraph>
loom::pnr::freezeRealizationGraph(const PnrProblemInputs &inputs) {
  if (llvm::Error error = validatePnrProblemInputs(inputs))
    return std::move(error);

  const DataflowProgramView &dataflow = inputs.dataflow;
  const FabricHardwareView &fabric = inputs.fabric;
  const ValidatedTechMapping &mapping = inputs.techMapping;

  if (llvm::Error error = detail::preflightFrozenRealizationGraphCapacity(
          mapping.realizations(), mapping.memoryRealizations(),
          sizeValue(dataflow.edges.size())))
    return std::move(error);

  std::vector<const ComputeRealizationDraft *> computeDrafts;
  computeDrafts.reserve(mapping.realizations().size());
  for (const ComputeRealizationDraft &realization : mapping.realizations())
    computeDrafts.push_back(&realization);
  std::sort(computeDrafts.begin(), computeDrafts.end(),
            [](const ComputeRealizationDraft *lhs,
               const ComputeRealizationDraft *rhs) {
              return lhs->id.value() < rhs->id.value();
            });

  std::vector<const MemoryRealizationDraft *> memoryDrafts;
  memoryDrafts.reserve(mapping.memoryRealizations().size());
  for (const MemoryRealizationDraft &realization : mapping.memoryRealizations())
    memoryDrafts.push_back(&realization);
  std::sort(
      memoryDrafts.begin(), memoryDrafts.end(),
      [](const MemoryRealizationDraft *lhs, const MemoryRealizationDraft *rhs) {
        return lhs->id.value() < rhs->id.value();
      });

  if (llvm::Error error = preflight(memoryCountContext, memoryDrafts.size()))
    return std::move(error);

  auto frozenCompute =
      detail::buildFrozenComputeDomains(fabric, mapping, computeDrafts);
  if (!frozenCompute)
    return frozenCompute.takeError();
  detail::FrozenComputeDomains compute = std::move(*frozenCompute);

  std::map<std::uint64_t, OwnershipInfo> ownershipByActor;
  std::map<ActorPort, TemplateTerminalKey, ActorPortLess> actorTerminals;
  for (std::size_t index = 0; index < computeDrafts.size(); ++index) {
    const ComputeRealizationDraft &draft = *computeDrafts[index];
    auto denseIndex = checked(realizationIndexContext, index);
    if (!denseIndex)
      return denseIndex.takeError();
    for (const ActorRef &actor : draft.actors) {
      if (!ownershipByActor
               .emplace(
                   actor.entity.value(),
                   OwnershipInfo{FrozenRealizationKind::Compute, *denseIndex})
               .second)
        return freezeError("cannot freeze realization graph: actor ownership "
                           "is not unique");
    }
    for (const BoundaryPortCorrespondence &boundary : draft.boundaryPorts) {
      ActorPort actor{boundary.actorPort.actor.entity,
                      boundary.actorPort.direction, boundary.actorPort.index};
      TemplateTerminalKey terminal =
          ComputeTerminalKey{draft.id, *denseIndex, boundary.fuPort.fu.entity,
                             boundary.fuPort.direction, boundary.fuPort.index};
      if (!actorTerminals.emplace(actor, std::move(terminal)).second)
        return freezeError("cannot freeze realization graph: actor terminal "
                           "correspondence is not unique");
    }
  }

  std::map<std::uint64_t, MemoryServiceDomainId> servicesByRoot;
  std::vector<FrozenMemoryRealization> memoryRealizations;
  memoryRealizations.reserve(memoryDrafts.size());

  for (std::size_t index = 0; index < memoryDrafts.size(); ++index) {
    const MemoryRealizationDraft &draft = *memoryDrafts[index];
    auto denseIndex = checked(realizationIndexContext, index);
    if (!denseIndex)
      return denseIndex.takeError();
    const MemorySemanticEncodingDescriptor *encoding =
        findDescriptor(fabric.memorySemanticEncodings, draft.encoding.entity);
    if (!encoding)
      return freezeError("cannot freeze realization graph: selected memory "
                         "encoding is unresolved");
    const MemoryImplementationDescriptor *implementation =
        findDescriptor(fabric.memoryImplementations, encoding->implementation);
    if (!implementation)
      return freezeError("cannot freeze realization graph: selected memory "
                         "implementation is unresolved");
    if (!findDescriptor(fabric.memoryServiceDomains, implementation->service))
      return freezeError("cannot freeze realization graph: selected memory "
                         "service is unresolved");

    memoryRealizations.push_back({draft.id, draft.encoding.entity,
                                  implementation->id, implementation->service});
    for (const ActorRef &actor : draft.actors) {
      if (!ownershipByActor
               .emplace(
                   actor.entity.value(),
                   OwnershipInfo{FrozenRealizationKind::Memory, *denseIndex})
               .second)
        return freezeError("cannot freeze realization graph: actor ownership "
                           "is not unique");
    }
    for (const MemoryBoundaryPortCorrespondence &boundary :
         draft.boundaryPorts) {
      ActorPort actor{boundary.actorPort.actor.entity,
                      boundary.actorPort.direction, boundary.actorPort.index};
      TemplateTerminalKey terminal = MemoryTerminalKey{
          draft.id, *denseIndex, boundary.operationPort.operation.entity,
          boundary.actorPort.direction, boundary.operationPort.index};
      if (!actorTerminals.emplace(actor, std::move(terminal)).second)
        return freezeError("cannot freeze realization graph: actor terminal "
                           "correspondence is not unique");
    }
    for (const LogicalMemoryRootRef &root : draft.roots) {
      auto [iterator, inserted] =
          servicesByRoot.emplace(root.entity.value(), implementation->service);
      if (!inserted && iterator->second != implementation->service)
        return freezeError("cannot freeze realization graph: logical memory "
                           "root resolves to inconsistent services");
    }
  }

  if (llvm::Error error = preflight(actorCountContext, ownershipByActor.size()))
    return std::move(error);

  std::vector<FrozenActorOwnership> actorOwnerships;
  actorOwnerships.reserve(ownershipByActor.size());
  for (const auto &[actor, ownership] : ownershipByActor)
    actorOwnerships.push_back(
        {ActorId(actor), ownership.kind, ownership.realization});

  std::map<std::uint64_t, const ActorDescriptor *> actorsById;
  for (const ActorDescriptor &actor : dataflow.actors)
    actorsById.emplace(actor.id.value(), &actor);
  std::map<std::uint64_t, const GraphDescriptor *> graphsById;
  for (const GraphDescriptor &graph : dataflow.graphs)
    graphsById.emplace(graph.id.value(), &graph);
  std::set<std::uint64_t> coveredGraphs;
  for (const GraphRef &graph : mapping.coveredGraphs())
    coveredGraphs.insert(graph.entity.value());

  std::set<std::uint64_t> memoryInternalEdges;
  for (const MemoryRealizationDraft *draft : memoryDrafts) {
    for (const MemoryInternalEdgeWitness &witness : draft->internalEdges)
      memoryInternalEdges.insert(witness.edge.entity.value());
  }

  auto resolveTerminal =
      [&](const DataflowEndpoint &endpoint) -> llvm::Expected<TerminalKey> {
    if (const auto *graph = std::get_if<GraphPort>(&endpoint)) {
      GraphTerminalKey terminal{graph->graph, graph->direction, graph->index};
      auto descriptor = graphPortDescriptor(terminal, graphsById);
      if (!descriptor)
        return descriptor.takeError();
      if (descriptor->kind == PortKind::Memory)
        return freezeError("cannot freeze realization graph: graph memory "
                           "capability port cannot become a token terminal");
      return TerminalKey{terminal};
    }
    const ActorPort &actor = std::get<ActorPort>(endpoint);
    auto terminal = actorTerminals.find(actor);
    if (terminal == actorTerminals.end())
      return freezeError("cannot freeze realization graph: external actor "
                         "endpoint has no boundary correspondence");
    return TerminalKey{terminal->second};
  };

  std::vector<ExternalEdge> externalEdges;
  externalEdges.reserve(dataflow.edges.size());
  for (const DataflowEdge &edge : dataflow.edges) {
    auto graph = endpointGraph(edge.source, actorsById);
    if (!graph)
      return graph.takeError();
    if (!coveredGraphs.count(graph->value()))
      continue;

    bool internal = memoryInternalEdges.count(edge.id.value()) != 0;
    const auto *sourceActor = std::get_if<ActorPort>(&edge.source);
    const auto *targetActor = std::get_if<ActorPort>(&edge.target);
    if (!internal && sourceActor && targetActor) {
      auto sourceOwner = ownershipByActor.find(sourceActor->actor.value());
      auto targetOwner = ownershipByActor.find(targetActor->actor.value());
      if (sourceOwner == ownershipByActor.end() ||
          targetOwner == ownershipByActor.end())
        return freezeError("cannot freeze realization graph: covered actor "
                           "has no realization ownership");
      internal =
          sourceOwner->second.kind == FrozenRealizationKind::Compute &&
          targetOwner->second.kind == FrozenRealizationKind::Compute &&
          sourceOwner->second.realization == targetOwner->second.realization;
    }
    if (internal)
      continue;

    auto source = resolveTerminal(edge.source);
    if (!source)
      return source.takeError();
    auto target = resolveTerminal(edge.target);
    if (!target)
      return target.takeError();
    externalEdges.push_back(
        {edge.id, edge.source, edge.target, *source, *target});
  }
  std::sort(externalEdges.begin(), externalEdges.end(), ExternalEdgeLess{});

  std::vector<TemplateTerminalKey> templateTerminalKeys;
  templateTerminalKeys.reserve(externalEdges.size());
  for (const ExternalEdge &edge : externalEdges) {
    if (const auto *terminal = std::get_if<TemplateTerminalKey>(&edge.source))
      templateTerminalKeys.push_back(*terminal);
    if (const auto *terminal = std::get_if<TemplateTerminalKey>(&edge.target))
      templateTerminalKeys.push_back(*terminal);
  }
  std::sort(templateTerminalKeys.begin(), templateTerminalKeys.end(),
            TemplateTerminalLess{});
  templateTerminalKeys.erase(
      std::unique(
          templateTerminalKeys.begin(), templateTerminalKeys.end(),
          [](const TemplateTerminalKey &lhs, const TemplateTerminalKey &rhs) {
            TemplateTerminalLess less;
            return !less(lhs, rhs) && !less(rhs, lhs);
          }),
      templateTerminalKeys.end());

  std::size_t logicalNetCount = 0;
  std::size_t maximumNetSinkCount = 0;
  for (std::size_t begin = 0; begin < externalEdges.size();) {
    std::size_t end = begin + 1;
    while (end < externalEdges.size() &&
           sameEndpoint(externalEdges[begin].sourceEndpoint,
                        externalEdges[end].sourceEndpoint))
      ++end;
    ++logicalNetCount;
    maximumNetSinkCount = std::max(maximumNetSinkCount, end - begin);
    begin = end;
  }

  if (llvm::Error error =
          preflight(terminalCountContext, templateTerminalKeys.size()))
    return std::move(error);
  if (llvm::Error error = preflight(netCountContext, logicalNetCount))
    return std::move(error);
  if (llvm::Error error = preflight(sinkCountContext, externalEdges.size()))
    return std::move(error);
  if (llvm::Error error = preflight(sinkCountContext, maximumNetSinkCount))
    return std::move(error);
  if (llvm::Error error = preflight(sinkOffsetContext, externalEdges.size()))
    return std::move(error);
  if (llvm::Error error =
          preflight(obligationCountContext, servicesByRoot.size()))
    return std::move(error);

  std::vector<FrozenTemplateTerminal> templateTerminals;
  templateTerminals.reserve(templateTerminalKeys.size());
  for (const TemplateTerminalKey &terminal : templateTerminalKeys) {
    auto frozen = freezeTemplateTerminal(terminal);
    if (!frozen)
      return frozen.takeError();
    templateTerminals.push_back(*frozen);
  }

  std::vector<FrozenLogicalNet> logicalNets;
  logicalNets.reserve(logicalNetCount);
  std::vector<FrozenLogicalNetSink> logicalNetSinks;
  logicalNetSinks.reserve(externalEdges.size());
  for (std::size_t begin = 0; begin < externalEdges.size();) {
    std::size_t end = begin + 1;
    while (end < externalEdges.size() &&
           sameEndpoint(externalEdges[begin].sourceEndpoint,
                        externalEdges[end].sourceEndpoint))
      ++end;

    auto source =
        freezeTerminal(externalEdges[begin].source, templateTerminalKeys);
    if (!source)
      return source.takeError();
    auto sinkOffset = checked(sinkOffsetContext, logicalNetSinks.size());
    if (!sinkOffset)
      return sinkOffset.takeError();
    auto sinkCount = checked(sinkCountContext, end - begin);
    if (!sinkCount)
      return sinkCount.takeError();
    logicalNets.push_back({*source, *sinkOffset, *sinkCount});

    for (std::size_t index = begin; index < end; ++index) {
      auto terminal =
          freezeTerminal(externalEdges[index].target, templateTerminalKeys);
      if (!terminal)
        return terminal.takeError();
      logicalNetSinks.push_back({externalEdges[index].edge, *terminal});
    }
    begin = end;
  }

  std::vector<FrozenMemoryServiceObligation> memoryServiceObligations;
  memoryServiceObligations.reserve(servicesByRoot.size());
  for (const auto &[root, service] : servicesByRoot)
    memoryServiceObligations.push_back({LogicalMemoryRootId(root), service});

  return FrozenRealizationGraph(
      std::move(actorOwnerships), std::move(compute.realizations),
      std::move(compute.occurrences),
      std::move(compute.occurrenceFuMemberships), std::move(compute.endpoints),
      std::move(compute.endpointCompatibleTypes), std::move(compute.localArcs),
      std::move(compute.implementationOccurrences),
      std::move(compute.portDemands), std::move(compute.compatibleEndpoints),
      std::move(memoryRealizations), std::move(templateTerminals),
      std::move(logicalNets), std::move(logicalNetSinks),
      std::move(memoryServiceObligations));
}
