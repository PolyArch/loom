#include "TechMappingCandidateDomain.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using Direction = ::loom::fabric::FabricPortDirection;
using TemplateEdge = ::loom::fabric::FabricFuCapabilityTemplateEdge;
using TemplateEndpoint = ::loom::fabric::FabricFuCapabilityTemplateEndpointRef;

struct ActorPortKey final {
  ::dataflow::ActorRef actor;
  Direction direction;
  std::uint64_t ordinal;
};

struct BoundaryRequirement final {
  ActorPortKey software;
  std::vector<::loom::fabric::FabricFuTemplatePortRef> candidates;
};

struct CanonicalActorOption final {
  TechComputeActorView actor;
  std::size_t operationOrdinal;
  std::vector<std::uint8_t> key;
};

bool actorPortLess(const ActorPortKey &lhs, const ActorPortKey &rhs) {
  return std::make_tuple(lhs.actor.entity.value(), lhs.direction, lhs.ordinal) <
         std::make_tuple(rhs.actor.entity.value(), rhs.direction, rhs.ordinal);
}

llvm::Expected<std::optional<::loom::PointerLayout>>
pointerLayoutFor(const TechMappingGenerationInputs &inputs,
                 const ::dataflow::CanonicalActorSchemaProjection &projection) {
  auto addressSpace = ::dataflow::projectActorPointerAddressSpace(projection);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<::loom::PointerLayout>{};
  auto layout = inputs.dataflow.pointerLayout(**addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<::loom::PointerLayout>(*layout);
}

std::vector<const ::loom::fabric::ResolvedFabricOpPhysicalPortView *>
physicalPorts(const ::loom::fabric::ResolvedFabricOpCapabilityView &capability,
              Direction direction) {
  std::vector<const ::loom::fabric::ResolvedFabricOpPhysicalPortView *> result;
  for (const auto &port : capability.physicalPorts)
    if (port.reference.direction == direction)
      result.push_back(&port);
  llvm::sort(result, [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  });
  return result;
}

llvm::Error enumeratePortMaps(
    const ::loom::fabric::ResolvedFabricOpCapabilityView &capability,
    Direction direction, std::size_t arity,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<std::uint64_t>)>
        emit) {
  const auto ports = physicalPorts(capability, direction);
  if (arity > ports.size())
    return llvm::Error::success();

  std::vector<std::uint64_t> current(arity);
  llvm::SmallDenseSet<std::uint64_t, 8> used;
  bool continueEnumeration = true;
  std::function<llvm::Error(std::size_t)> visit =
      [&](std::size_t ordinal) -> llvm::Error {
    if (!continueEnumeration)
      return llvm::Error::success();
    if (ordinal == arity) {
      auto result = emit(current);
      if (!result)
        return result.takeError();
      continueEnumeration = *result;
      return llvm::Error::success();
    }
    for (const auto *port : ports) {
      if (!used.insert(port->reference.ordinal).second)
        continue;
      current[ordinal] = port->reference.ordinal;
      if (llvm::Error error = visit(ordinal + 1))
        return error;
      used.erase(port->reference.ordinal);
      if (!continueEnumeration)
        break;
    }
    return llvm::Error::success();
  };
  return visit(0);
}

llvm::Error enumerateActorOptions(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    const ::loom::fabric::FabricFuTemplateNodeRef &operation,
    llvm::function_ref<llvm::Expected<bool>(TechComputeActorView)> emit) {
  const auto *capability = inputs.fabric.resolvedFabricOpCapability(operation);
  if (!capability)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_generation_invalid: active FU template operation has "
        "no resolved capability");

  bool continueEnumeration = true;
  return enumeratePortMaps(
      *capability, Direction::Input, actor.op->getNumOperands(),
      [&](llvm::ArrayRef<std::uint64_t> operandMap) -> llvm::Expected<bool> {
        if (llvm::Error error = enumeratePortMaps(
                *capability, Direction::Output, actor.op->getNumResults(),
                [&](llvm::ArrayRef<std::uint64_t> resultMap)
                    -> llvm::Expected<bool> {
                  auto result = emit(TechComputeActorView{
                      actor.ref, operation,
                      std::vector<std::uint64_t>(operandMap),
                      std::vector<std::uint64_t>(resultMap)});
                  if (!result)
                    return result.takeError();
                  continueEnumeration = *result;
                  return *result;
                }))
          return std::move(error);
        return continueEnumeration;
      });
}

llvm::Expected<std::vector<CanonicalActorOption>> canonicalActorOptions(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    llvm::ArrayRef<::loom::fabric::FabricFuTemplateNodeRef> operations) {
  std::vector<CanonicalActorOption> options;
  for (auto indexedOperation : llvm::enumerate(operations)) {
    const std::size_t operationOrdinal = indexedOperation.index();
    const auto &operation = indexedOperation.value();
    if (llvm::Error error = enumerateActorOptions(
            inputs, actor, operation,
            [&](TechComputeActorView option) -> llvm::Expected<bool> {
              auto key = canonicalTechMatchActorKey(option,
                                                    inputs.dataflow.identity());
              if (!key)
                return key.takeError();
              options.push_back(CanonicalActorOption{
                  std::move(option), operationOrdinal, std::move(*key)});
              return true;
            }))
      return std::move(error);
  }
  llvm::sort(options, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  options.erase(std::unique(options.begin(), options.end(),
                            [](const auto &lhs, const auto &rhs) {
                              return lhs.key == rhs.key;
                            }),
                options.end());
  return options;
}

llvm::Expected<bool>
admitActorCorrespondence(const TechMappingGenerationInputs &inputs,
                         const TechComputeActorView &selected) {
  auto actor = inputs.dataflow.resolve(selected.actor);
  if (!actor)
    return actor.takeError();
  const auto *capability =
      inputs.fabric.resolvedFabricOpCapability(selected.fabricOperation);
  if (!capability)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_generation_invalid: selected fabric.op has no resolved "
        "capability");
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
  if (!projection)
    return projection.takeError();
  auto indexBitWidth = ::loom::getIndexBitWidth(actor->op);
  if (!indexBitWidth)
    return indexBitWidth.takeError();
  auto pointerLayout = pointerLayoutFor(inputs, *projection);
  if (!pointerLayout)
    return pointerLayout.takeError();
  if (llvm::Error error = capability->admitCorrespondence(
          *projection, *indexBitWidth, selected.operandPorts,
          selected.resultPorts, *pointerLayout ? &**pointerLayout : nullptr)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

llvm::Expected<bool>
admitActorCorrespondences(const TechMappingGenerationInputs &inputs,
                          llvm::ArrayRef<TechComputeActorView> actors) {
  for (const TechComputeActorView &selected : actors) {
    auto admitted = admitActorCorrespondence(inputs, selected);
    if (!admitted || !*admitted)
      return admitted;
  }
  return true;
}

TemplateEndpoint nodeEndpoint(const TechComputeActorView &actor,
                              Direction direction,
                              std::uint64_t softwareOrdinal) {
  const auto &ports =
      direction == Direction::Input ? actor.operandPorts : actor.resultPorts;
  return TemplateEndpoint::nodePort(::loom::fabric::FabricFuNodePortRef{
      actor.fabricOperation, direction, ports[softwareOrdinal]});
}

bool rowContains(llvm::ArrayRef<TechComputeActorView> actors,
                 ::dataflow::ActorRef actor) {
  return llvm::any_of(
      actors, [&](const auto &candidate) { return candidate.actor == actor; });
}

const TechComputeActorView *
findActor(llvm::ArrayRef<TechComputeActorView> actors,
          ::dataflow::ActorRef actor) {
  auto found = llvm::find_if(
      actors, [&](const auto &candidate) { return candidate.actor == actor; });
  return found == actors.end() ? nullptr : &*found;
}

std::vector<::loom::fabric::FabricFuTemplatePortRef>
boundaryCandidates(llvm::ArrayRef<TemplateEdge> topology,
                   const TemplateEndpoint &operation, Direction direction) {
  std::vector<::loom::fabric::FabricFuTemplatePortRef> result;
  for (const TemplateEdge &edge : topology) {
    const TemplateEndpoint &candidate =
        direction == Direction::Input ? edge.source : edge.destination;
    const TemplateEndpoint &op =
        direction == Direction::Input ? edge.destination : edge.source;
    if (op != operation)
      continue;
    const auto *boundary = std::get_if<::loom::fabric::FabricFuTemplatePortRef>(
        &candidate.payload);
    if (boundary && boundary->direction == direction)
      result.push_back(*boundary);
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool sameEdgeSet(llvm::ArrayRef<TemplateEdge> lhs,
                 llvm::ArrayRef<TemplateEdge> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(lhs, [&](const TemplateEdge &edge) {
    return llvm::is_contained(rhs, edge);
  });
}

std::vector<TemplateEdge>
selectedTopology(llvm::ArrayRef<TemplateEdge> topology,
                 llvm::ArrayRef<TechComputeActorView> actors) {
  std::vector<TemplateEndpoint> sources;
  std::vector<TemplateEndpoint> sinks;
  for (const TechComputeActorView &actor : actors) {
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal)
      sinks.push_back(nodeEndpoint(actor, Direction::Input, ordinal));
    for (std::uint64_t ordinal = 0; ordinal < actor.resultPorts.size();
         ++ordinal)
      sources.push_back(nodeEndpoint(actor, Direction::Output, ordinal));
  }
  std::vector<TemplateEdge> selected;
  for (const TemplateEdge &edge : topology) {
    const bool sourceBoundary =
        std::holds_alternative<::loom::fabric::FabricFuTemplatePortRef>(
            edge.source.payload);
    const bool sinkBoundary =
        std::holds_alternative<::loom::fabric::FabricFuTemplatePortRef>(
            edge.destination.payload);
    if ((!sourceBoundary && !llvm::is_contained(sources, edge.source)) ||
        (!sinkBoundary && !llvm::is_contained(sinks, edge.destination)))
      continue;
    selected.push_back(edge);
  }
  return selected;
}

llvm::Expected<bool>
internalTopologyCompatible(const TechMappingGenerationInputs &inputs,
                           llvm::ArrayRef<TechComputeActorView> actors,
                           llvm::ArrayRef<TemplateEdge> topology) {
  std::vector<TemplateEdge> expected;
  for (const TechComputeActorView &actor : actors) {
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor.actor, ordinal};
      auto producer = inputs.dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      const auto *result =
          std::get_if<::dataflow::ActorTokenResultRef>(&*producer);
      if (!result || !rowContains(actors, result->actor))
        continue;
      const TechComputeActorView *source = findActor(actors, result->actor);
      if (!source || result->ordinal >= source->resultPorts.size())
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "tech_mapping_generation_invalid: partial compute topology uses "
            "an unmapped result");
      expected.push_back(
          {nodeEndpoint(*source, Direction::Output, result->ordinal),
           nodeEndpoint(actor, Direction::Input, ordinal)});
    }
  }

  std::vector<TemplateEdge> physical;
  for (const TemplateEdge &edge : selectedTopology(topology, actors))
    if (std::holds_alternative<::loom::fabric::FabricFuNodePortRef>(
            edge.source.payload) &&
        std::holds_alternative<::loom::fabric::FabricFuNodePortRef>(
            edge.destination.payload))
      physical.push_back(edge);
  return sameEdgeSet(expected, physical);
}

llvm::Expected<bool>
mandatoryBoundariesAvailable(const TechMappingGenerationInputs &inputs,
                             llvm::ArrayRef<TechComputeActorView> actors,
                             llvm::ArrayRef<TemplateEdge> topology) {
  const std::uint64_t lastActor = actors.back().actor.entity.value();
  for (const TechComputeActorView &actor : actors) {
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor.actor, ordinal};
      auto producer = inputs.dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      bool boundaryRequired =
          std::holds_alternative<::dataflow::GraphIngressTokenRef>(*producer);
      if (const auto *result =
              std::get_if<::dataflow::ActorTokenResultRef>(&*producer))
        boundaryRequired = !rowContains(actors, result->actor) &&
                           result->actor.entity.value() <= lastActor;
      if (boundaryRequired &&
          boundaryCandidates(topology,
                             nodeEndpoint(actor, Direction::Input, ordinal),
                             Direction::Input)
              .empty())
        return false;
    }

    for (std::uint64_t ordinal = 0; ordinal < actor.resultPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor.actor, ordinal};
      auto consumers = inputs.dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      const bool boundaryRequired =
          llvm::any_of(*consumers, [&](const auto &consumer) {
            if (std::holds_alternative<::dataflow::GraphEgressTokenRef>(
                    consumer))
              return true;
            const auto *operand =
                std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
            return operand && !rowContains(actors, operand->actor) &&
                   operand->actor.entity.value() <= lastActor;
          });
      if (boundaryRequired &&
          boundaryCandidates(topology,
                             nodeEndpoint(actor, Direction::Output, ordinal),
                             Direction::Output)
              .empty())
        return false;
    }
  }
  return true;
}

llvm::Expected<bool>
selectionCanComplete(const TechMappingGenerationInputs &inputs,
                     llvm::ArrayRef<::dataflow::CanonicalActorView> actors,
                     llvm::ArrayRef<TemplateEdge> topology,
                     llvm::ArrayRef<TechComputeActorView> selection,
                     std::size_t nextActor, ::dataflow::GraphRef graph,
                     std::size_t remaining) {
  auto internal = internalTopologyCompatible(inputs, selection, topology);
  if (!internal || !*internal)
    return internal;
  auto boundaries = mandatoryBoundariesAvailable(inputs, selection, topology);
  if (!boundaries || !*boundaries)
    return boundaries;
  if (remaining == 0)
    return true;

  std::size_t availableActors = 0;
  for (std::size_t actor = nextActor; actor < actors.size(); ++actor)
    availableActors += actors[actor].graph == graph;
  return availableActors >= remaining;
}

llvm::Expected<std::vector<BoundaryRequirement>>
deriveBoundaryRequirements(const TechMappingGenerationInputs &inputs,
                           llvm::ArrayRef<TechComputeActorView> actors,
                           llvm::ArrayRef<TemplateEdge> topology,
                           std::vector<TemplateEdge> &fixedEdges) {
  std::vector<BoundaryRequirement> requirements;
  for (const TechComputeActorView &actor : actors) {
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor.actor, ordinal};
      auto producer = inputs.dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      if (const auto *result =
              std::get_if<::dataflow::ActorTokenResultRef>(&*producer);
          result && rowContains(actors, result->actor)) {
        const TechComputeActorView *source = findActor(actors, result->actor);
        if (!source || result->ordinal >= source->resultPorts.size())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "tech_mapping_generation_invalid: internal compute edge uses "
              "an unmapped result");
        fixedEdges.push_back(TemplateEdge{
            nodeEndpoint(*source, Direction::Output, result->ordinal),
            nodeEndpoint(actor, Direction::Input, ordinal)});
        continue;
      }
      auto candidates = boundaryCandidates(
          topology, nodeEndpoint(actor, Direction::Input, ordinal),
          Direction::Input);
      if (candidates.empty())
        return std::vector<BoundaryRequirement>{};
      requirements.push_back(BoundaryRequirement{
          ActorPortKey{actor.actor, Direction::Input, ordinal},
          std::move(candidates)});
    }

    for (std::uint64_t ordinal = 0; ordinal < actor.resultPorts.size();
         ++ordinal) {
      const ::dataflow::CanonicalGraphProducerEndpointRef producer =
          ::dataflow::ActorTokenResultRef{actor.actor, ordinal};
      auto consumers = inputs.dataflow.graphConsumers(producer);
      if (!consumers)
        return consumers.takeError();
      if (!llvm::any_of(*consumers, [&](const auto &consumer) {
            const auto *operand =
                std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
            return !operand || !rowContains(actors, operand->actor);
          }))
        continue;
      auto candidates = boundaryCandidates(
          topology, nodeEndpoint(actor, Direction::Output, ordinal),
          Direction::Output);
      if (candidates.empty())
        return std::vector<BoundaryRequirement>{};
      requirements.push_back(BoundaryRequirement{
          ActorPortKey{actor.actor, Direction::Output, ordinal},
          std::move(candidates)});
    }
  }
  llvm::sort(requirements, [](const auto &lhs, const auto &rhs) {
    return actorPortLess(lhs.software, rhs.software);
  });
  return requirements;
}

llvm::Error emitComputeSeeds(
    const TechMappingGenerationInputs &inputs,
    const ::loom::fabric::FabricFuCapabilityTemplateRef &capabilityTemplate,
    llvm::ArrayRef<TemplateEdge> completeTopology,
    std::vector<TechComputeActorView> actorSelection,
    TechMatchRowCollector &collector) {
  llvm::sort(actorSelection, [](const auto &lhs, const auto &rhs) {
    return lhs.actor.entity.value() < rhs.actor.entity.value();
  });
  const std::vector<TemplateEdge> topology =
      selectedTopology(completeTopology, actorSelection);
  std::vector<TemplateEdge> fixedEdges;
  auto requirements =
      deriveBoundaryRequirements(inputs, actorSelection, topology, fixedEdges);
  if (!requirements)
    return requirements.takeError();
  if (requirements->empty() && fixedEdges.size() != topology.size())
    return llvm::Error::success();

  std::vector<TechComputeBoundaryView> boundaries;
  std::vector<TemplateEdge> selectedEdges = fixedEdges;
  std::function<llvm::Error(std::size_t)> visit =
      [&](std::size_t ordinal) -> llvm::Error {
    if (collector.truncated())
      return llvm::Error::success();
    if (ordinal != requirements->size()) {
      const BoundaryRequirement &requirement = (*requirements)[ordinal];
      const TechComputeActorView *actor =
          findActor(actorSelection, requirement.software.actor);
      const TemplateEndpoint operation = nodeEndpoint(
          *actor, requirement.software.direction, requirement.software.ordinal);
      for (const auto &boundary : requirement.candidates) {
        boundaries.push_back(TechComputeBoundaryView{
            requirement.software.actor, requirement.software.direction,
            requirement.software.ordinal, boundary});
        const TemplateEndpoint boundaryEndpoint =
            TemplateEndpoint::boundaryPort(boundary);
        selectedEdges.push_back(
            requirement.software.direction == Direction::Input
                ? TemplateEdge{boundaryEndpoint, operation}
                : TemplateEdge{operation, boundaryEndpoint});
        if (llvm::Error error = visit(ordinal + 1))
          return error;
        selectedEdges.pop_back();
        boundaries.pop_back();
        if (collector.truncated())
          break;
      }
      return llvm::Error::success();
    }

    TechComputeRealizationView realization{0, capabilityTemplate,
                                           actorSelection, boundaries};
    auto key =
        canonicalTechMatchRowKey(realization, inputs.dataflow.identity());
    if (!key)
      return key.takeError();
    auto entered = collector.beginSeed(std::move(*key));
    if (!entered)
      return entered.takeError();
    if (!*entered)
      return llvm::Error::success();
    auto capabilityAdmitted = admitActorCorrespondences(inputs, actorSelection);
    if (!capabilityAdmitted)
      return capabilityAdmitted.takeError();
    if (!*capabilityAdmitted)
      return collector.reject(
          TechMatchSeedRejectionReason::CapabilityInadmissible);
    if (!sameEdgeSet(topology, selectedEdges))
      return collector.reject(
          TechMatchSeedRejectionReason::CorrespondenceInadmissible);
    if (llvm::Error error = verifyTechComputeRealizationClosure(
            realization, inputs.dataflow, inputs.fabric)) {
      llvm::consumeError(std::move(error));
      return collector.reject(
          TechMatchSeedRejectionReason::RealizationInadmissible);
    }
    std::vector<::dataflow::ActorRef> covered;
    covered.reserve(actorSelection.size());
    for (const auto &actor : actorSelection)
      covered.push_back(actor.actor);
    return collector.admit(std::move(realization), covered);
  };
  return visit(0);
}

} // namespace

llvm::Error enumerateCanonicalComputeSelections(
    const TechMappingGenerationInputs &inputs,
    llvm::ArrayRef<::dataflow::CanonicalActorView> actors,
    llvm::ArrayRef<::loom::fabric::FabricFuTemplateNodeRef> operations,
    llvm::ArrayRef<TemplateEdge> topology,
    llvm::function_ref<
        llvm::Expected<bool>(llvm::ArrayRef<TechComputeActorView>)>
        emitSelection) {
  if (operations.empty())
    return llvm::Error::success();
  const std::size_t operationCount = operations.size();
  const std::size_t actorCount = actors.size();
  if (actorCount < operationCount)
    return llvm::Error::success();

  std::vector<std::uint8_t> usedOperations(operationCount, false);
  std::vector<std::vector<CanonicalActorOption>> optionsByActor;
  optionsByActor.reserve(actorCount);
  for (const auto &actor : actors) {
    auto options = canonicalActorOptions(inputs, actor, operations);
    if (!options)
      return options.takeError();
    optionsByActor.push_back(std::move(*options));
  }
  std::vector<TechComputeActorView> selection;
  bool continueEnumeration = true;
  std::function<llvm::Error(std::size_t, std::size_t,
                            std::optional<::dataflow::GraphRef>)>
      visit = [&](std::size_t start, std::size_t remaining,
                  std::optional<::dataflow::GraphRef> graph) -> llvm::Error {
    if (remaining == 0) {
      auto result = emitSelection(selection);
      if (!result)
        return result.takeError();
      continueEnumeration = *result;
      return llvm::Error::success();
    }

    for (std::size_t actor = start;
         continueEnumeration && actor + remaining <= actorCount; ++actor) {
      if (graph && actors[actor].graph != *graph)
        continue;
      const ::dataflow::GraphRef selectedGraph =
          graph ? *graph : actors[actor].graph;
      for (const CanonicalActorOption &option : optionsByActor[actor]) {
        if (!continueEnumeration)
          break;
        if (usedOperations[option.operationOrdinal])
          continue;
        usedOperations[option.operationOrdinal] = true;
        selection.push_back(option.actor);
        auto viable =
            selectionCanComplete(inputs, actors, topology, selection, actor + 1,
                                 selectedGraph, remaining - 1);
        if (!viable)
          return viable.takeError();
        if (*viable)
          if (llvm::Error error =
                  visit(actor + 1, remaining - 1, selectedGraph))
            return error;
        selection.pop_back();
        usedOperations[option.operationOrdinal] = false;
      }
    }
    return llvm::Error::success();
  };
  return visit(0, operationCount, std::nullopt);
}

llvm::Error
deriveComputeRows(const TechMappingGenerationInputs &inputs,
                  llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
                  TechMatchRowCollector &collector) {
  std::vector<::dataflow::CanonicalActorView> actors;
  for (const auto &actor : selectedActors)
    if (actor.kind != ::dataflow::CanonicalDataflowActorKind::Memory)
      actors.push_back(actor);
  llvm::sort(actors, [](const auto &lhs, const auto &rhs) {
    return lhs.ref.entity.value() < rhs.ref.entity.value();
  });

  std::vector<::loom::fabric::FabricFuTemplateRef> fuTemplates(
      inputs.fabric.fuTemplates().begin(), inputs.fabric.fuTemplates().end());
  llvm::sort(fuTemplates, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  for (const ::loom::fabric::FabricFuTemplateRef fu : fuTemplates) {
    const auto inventory = inputs.fabric.fuCapabilityTemplates(fu);
    for (std::size_t templateOrdinal = 0; templateOrdinal < inventory.size();
         ++templateOrdinal) {
      const auto &record = inventory[templateOrdinal];
      std::vector<::loom::fabric::FabricFuTemplateNodeRef> operations;
      for (const auto &node : record.activeNodes)
        if (node.node == ::loom::fabric::FabricFuNodeKind::Op)
          operations.push_back(node);
      llvm::sort(operations, [](const auto &lhs, const auto &rhs) {
        return ::loom::fabric::canonicalFabricBytes(lhs) <
               ::loom::fabric::canonicalFabricBytes(rhs);
      });
      if (operations.empty())
        continue;
      auto topology =
          ::loom::fabric::projectFabricFuCapabilityTemplateTerminalEdges(
              record);
      if (!topology)
        return topology.takeError();

      if (actors.size() < operations.size())
        continue;
      if (llvm::Error error = enumerateCanonicalComputeSelections(
              inputs, actors, operations, *topology,
              [&](llvm::ArrayRef<TechComputeActorView> selection)
                  -> llvm::Expected<bool> {
                if (collector.truncated())
                  return false;
                if (llvm::Error error = emitComputeSeeds(
                        inputs,
                        ::loom::fabric::FabricFuCapabilityTemplateRef{
                            fu, static_cast<std::uint64_t>(templateOrdinal)},
                        *topology,
                        std::vector<TechComputeActorView>(selection.begin(),
                                                          selection.end()),
                        collector))
                  return std::move(error);
                return !collector.truncated();
              }))
        return error;
      if (collector.truncated())
        return llvm::Error::success();
    }
  }
  return llvm::Error::success();
}

} // namespace loom::mapping::detail
