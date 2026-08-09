#include "TechMappingCandidateDomain.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
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

struct CachedActorTopology final {
  ::dataflow::ActorRef actor;
  std::vector<::dataflow::CanonicalGraphProducerEndpointRef> producers;
  std::vector<std::vector<::dataflow::CanonicalGraphConsumerEndpointRef>>
      consumers;
};

struct SeedRangeCardinality final {
  std::uint64_t count = 1;
  bool overflow = false;

  void multiply(std::size_t factor) {
    if (overflow)
      return;
    if (factor > std::numeric_limits<std::uint64_t>::max() / count) {
      count = std::numeric_limits<std::uint64_t>::max();
      overflow = true;
      return;
    }
    count *= factor;
  }
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

std::vector<std::uint64_t> physicalPortOrdinals(
    const ::loom::fabric::ResolvedFabricOpCapabilityView &capability,
    Direction direction) {
  const auto ports = physicalPorts(capability, direction);
  std::vector<std::uint64_t> result;
  result.reserve(ports.size());
  for (const auto *port : ports)
    result.push_back(port->reference.ordinal);
  return result;
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

  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
  if (!projection)
    return projection.takeError();
  const std::vector<std::uint64_t> inputPorts =
      physicalPortOrdinals(*capability, Direction::Input);
  const std::vector<std::uint64_t> resultPorts =
      physicalPortOrdinals(*capability, Direction::Output);
  return ::fabric::forEachImplementationFamilyPortCorrespondence(
      capability->implementationFamily, *projection, inputPorts, resultPorts,
      [&](llvm::ArrayRef<std::uint64_t> operandMap,
          llvm::ArrayRef<std::uint64_t> resultMap) {
        return emit(TechComputeActorView{
            actor.ref, operation, std::vector<std::uint64_t>(operandMap),
            std::vector<std::uint64_t>(resultMap)});
      });
}

llvm::Expected<std::vector<CanonicalActorOption>> canonicalActorOptions(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    llvm::ArrayRef<::loom::fabric::FabricFuTemplateNodeRef> operations) {
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
  if (!projection)
    return projection.takeError();
  std::vector<CanonicalActorOption> options;
  for (auto indexedOperation : llvm::enumerate(operations)) {
    const std::size_t operationOrdinal = indexedOperation.index();
    const auto &operation = indexedOperation.value();
    const auto *capability =
        inputs.fabric.resolvedFabricOpCapability(operation);
    if (!capability)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "tech_mapping_generation_invalid: active FU template operation has "
          "no resolved capability");
    if (!llvm::is_contained(capability->enabledOperationSchemas,
                            projection->schema))
      continue;
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

const CachedActorTopology &
cachedTopology(llvm::ArrayRef<CachedActorTopology> topology,
               ::dataflow::ActorRef actor) {
  const auto found = llvm::lower_bound(
      topology, actor.entity.value(),
      [](const CachedActorTopology &candidate, std::uint64_t value) {
        return candidate.actor.entity.value() < value;
      });
  assert(found != topology.end() && found->actor == actor);
  return *found;
}

bool actorUsesNodeEndpoint(const TechComputeActorView &actor,
                           const TemplateEndpoint &endpoint) {
  const auto *node =
      std::get_if<::loom::fabric::FabricFuNodePortRef>(&endpoint.payload);
  if (!node || actor.fabricOperation != node->node)
    return false;
  const auto &ports = node->direction == Direction::Input ? actor.operandPorts
                                                          : actor.resultPorts;
  return llvm::is_contained(ports, node->ordinal);
}

bool selectedNodeEndpoint(llvm::ArrayRef<TechComputeActorView> actors,
                          const TemplateEndpoint &endpoint) {
  for (const TechComputeActorView &actor : actors) {
    if (actorUsesNodeEndpoint(actor, endpoint))
      return true;
  }
  return false;
}

bool boundaryAvailable(llvm::ArrayRef<TemplateEdge> topology,
                       const TemplateEndpoint &operation, Direction direction) {
  return llvm::any_of(topology, [&](const TemplateEdge &edge) {
    const TemplateEndpoint &candidate =
        direction == Direction::Input ? edge.source : edge.destination;
    const TemplateEndpoint &op =
        direction == Direction::Input ? edge.destination : edge.source;
    const auto *boundary = std::get_if<::loom::fabric::FabricFuTemplatePortRef>(
        &candidate.payload);
    return op == operation && boundary && boundary->direction == direction;
  });
}

llvm::Expected<bool>
internalTopologyCompatible(llvm::ArrayRef<TechComputeActorView> actors,
                           llvm::ArrayRef<TemplateEdge> topology,
                           llvm::ArrayRef<CachedActorTopology> cached) {
  std::size_t expectedEdgeCount = 0;
  for (const TechComputeActorView &actor : actors) {
    const CachedActorTopology &actorTopology =
        cachedTopology(cached, actor.actor);
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal) {
      const auto *result = std::get_if<::dataflow::ActorTokenResultRef>(
          &actorTopology.producers[ordinal]);
      if (!result || !rowContains(actors, result->actor))
        continue;
      const TechComputeActorView *source = findActor(actors, result->actor);
      if (!source || result->ordinal >= source->resultPorts.size())
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "tech_mapping_generation_invalid: partial compute topology uses "
            "an unmapped result");
      const TemplateEdge edge{
          nodeEndpoint(*source, Direction::Output, result->ordinal),
          nodeEndpoint(actor, Direction::Input, ordinal)};
      if (!llvm::is_contained(topology, edge))
        return false;
      ++expectedEdgeCount;
    }
  }

  std::size_t selectedPhysicalEdgeCount = 0;
  for (const TemplateEdge &edge : topology)
    selectedPhysicalEdgeCount += selectedNodeEndpoint(actors, edge.source) &&
                                 selectedNodeEndpoint(actors, edge.destination);
  return expectedEdgeCount == selectedPhysicalEdgeCount;
}

llvm::Expected<bool>
mandatoryBoundariesAvailable(llvm::ArrayRef<TechComputeActorView> actors,
                             llvm::ArrayRef<TemplateEdge> topology,
                             llvm::ArrayRef<CachedActorTopology> cached) {
  const std::uint64_t lastActor = actors.back().actor.entity.value();
  for (const TechComputeActorView &actor : actors) {
    const CachedActorTopology &actorTopology =
        cachedTopology(cached, actor.actor);
    for (std::uint64_t ordinal = 0; ordinal < actor.operandPorts.size();
         ++ordinal) {
      bool boundaryRequired =
          std::holds_alternative<::dataflow::GraphIngressTokenRef>(
              actorTopology.producers[ordinal]);
      if (const auto *result = std::get_if<::dataflow::ActorTokenResultRef>(
              &actorTopology.producers[ordinal]))
        boundaryRequired = !rowContains(actors, result->actor) &&
                           result->actor.entity.value() <= lastActor;
      if (boundaryRequired &&
          !boundaryAvailable(topology,
                             nodeEndpoint(actor, Direction::Input, ordinal),
                             Direction::Input))
        return false;
    }

    for (std::uint64_t ordinal = 0; ordinal < actor.resultPorts.size();
         ++ordinal) {
      const bool boundaryRequired = llvm::any_of(
          actorTopology.consumers[ordinal], [&](const auto &consumer) {
            if (std::holds_alternative<::dataflow::GraphEgressTokenRef>(
                    consumer))
              return true;
            const auto *operand =
                std::get_if<::dataflow::ActorTokenOperandRef>(&consumer);
            return operand && !rowContains(actors, operand->actor) &&
                   operand->actor.entity.value() <= lastActor;
          });
      if (boundaryRequired &&
          !boundaryAvailable(topology,
                             nodeEndpoint(actor, Direction::Output, ordinal),
                             Direction::Output))
        return false;
    }
  }
  return true;
}

llvm::Expected<bool>
selectionCanComplete(llvm::ArrayRef<::dataflow::CanonicalActorView> actors,
                     llvm::ArrayRef<TemplateEdge> topology,
                     llvm::ArrayRef<TechComputeActorView> selection,
                     llvm::ArrayRef<CachedActorTopology> cached,
                     std::size_t nextActor, ::dataflow::GraphRef graph,
                     std::size_t remaining) {
  auto internal = internalTopologyCompatible(selection, topology, cached);
  if (!internal || !*internal)
    return internal;
  auto boundaries = mandatoryBoundariesAvailable(selection, topology, cached);
  if (!boundaries || !*boundaries)
    return boundaries;
  if (remaining == 0)
    return true;

  std::size_t availableActors = 0;
  for (std::size_t actor = nextActor; actor < actors.size(); ++actor)
    availableActors += actors[actor].graph == graph;
  return availableActors >= remaining;
}

bool directedActorPairCompatible(const TechComputeActorView &source,
                                 const TechComputeActorView &sink,
                                 llvm::ArrayRef<TemplateEdge> topology,
                                 llvm::ArrayRef<CachedActorTopology> cached) {
  const CachedActorTopology &sinkTopology = cachedTopology(cached, sink.actor);
  std::size_t softwareEdgeCount = 0;
  for (std::uint64_t ordinal = 0; ordinal < sink.operandPorts.size();
       ++ordinal) {
    const auto *producer = std::get_if<::dataflow::ActorTokenResultRef>(
        &sinkTopology.producers[ordinal]);
    if (!producer || producer->actor != source.actor)
      continue;
    if (producer->ordinal >= source.resultPorts.size())
      return false;
    const TemplateEdge edge{
        nodeEndpoint(source, Direction::Output, producer->ordinal),
        nodeEndpoint(sink, Direction::Input, ordinal)};
    if (!llvm::is_contained(topology, edge))
      return false;
    ++softwareEdgeCount;
  }

  std::size_t physicalEdgeCount = 0;
  for (const TemplateEdge &edge : topology)
    physicalEdgeCount += actorUsesNodeEndpoint(source, edge.source) &&
                         actorUsesNodeEndpoint(sink, edge.destination);
  return softwareEdgeCount == physicalEdgeCount;
}

bool actorPairCompatible(const TechComputeActorView &lhs,
                         const TechComputeActorView &rhs,
                         llvm::ArrayRef<TemplateEdge> topology,
                         llvm::ArrayRef<CachedActorTopology> cached) {
  return directedActorPairCompatible(lhs, rhs, topology, cached) &&
         (lhs.actor == rhs.actor ||
          directedActorPairCompatible(rhs, lhs, topology, cached));
}

llvm::Expected<std::vector<CachedActorTopology>>
cacheActorTopology(const TechMappingGenerationInputs &inputs,
                   llvm::ArrayRef<::dataflow::CanonicalActorView> actors) {
  std::vector<CachedActorTopology> result;
  result.reserve(actors.size());
  for (const auto &actor : actors) {
    CachedActorTopology cached{actor.ref, {}, {}};
    cached.producers.reserve(actor.op->getNumOperands());
    for (std::uint64_t ordinal = 0; ordinal < actor.op->getNumOperands();
         ++ordinal) {
      auto producer = inputs.dataflow.graphProducer(
          ::dataflow::CanonicalGraphConsumerEndpointRef{
              ::dataflow::ActorTokenOperandRef{actor.ref, ordinal}});
      if (!producer)
        return producer.takeError();
      cached.producers.push_back(std::move(*producer));
    }
    cached.consumers.reserve(actor.op->getNumResults());
    for (std::uint64_t ordinal = 0; ordinal < actor.op->getNumResults();
         ++ordinal) {
      auto consumers = inputs.dataflow.graphConsumers(
          ::dataflow::CanonicalGraphProducerEndpointRef{
              ::dataflow::ActorTokenResultRef{actor.ref, ordinal}});
      if (!consumers)
        return consumers.takeError();
      cached.consumers.emplace_back(consumers->begin(), consumers->end());
    }
    result.push_back(std::move(cached));
  }
  return result;
}

llvm::Expected<std::vector<BoundaryRequirement>>
deriveBoundaryRequirements(const TechMappingGenerationInputs &inputs,
                           llvm::ArrayRef<TechComputeActorView> actors,
                           llvm::ArrayRef<TemplateEdge> topology) {
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
  auto requirements =
      deriveBoundaryRequirements(inputs, actorSelection, topology);
  if (!requirements)
    return requirements.takeError();

  std::vector<TechComputeBoundaryView> boundaries;
  SeedRangeCardinality cardinality;
  bool branchingBoundary = false;
  for (const BoundaryRequirement &requirement : *requirements) {
    cardinality.multiply(requirement.candidates.size());
    branchingBoundary |= requirement.candidates.size() > 1;
    if (!requirement.candidates.empty())
      boundaries.push_back(TechComputeBoundaryView{
          requirement.software.actor, requirement.software.direction,
          requirement.software.ordinal, requirement.candidates.front()});
  }

  auto capabilityAdmitted = admitActorCorrespondences(inputs, actorSelection);
  if (!capabilityAdmitted)
    return capabilityAdmitted.takeError();
  if (branchingBoundary) {
    TechComputeRealizationView first{0, capabilityTemplate, actorSelection,
                                     boundaries};
    auto firstKey = canonicalTechMatchRowKey(first, inputs.dataflow.identity());
    if (!firstKey)
      return firstKey.takeError();
    for (auto [boundary, requirement] :
         llvm::zip_equal(boundaries, *requirements))
      boundary.fabricPort = requirement.candidates.back();
    TechComputeRealizationView last{0, capabilityTemplate, actorSelection,
                                    boundaries};
    auto lastKey = canonicalTechMatchRowKey(last, inputs.dataflow.identity());
    if (!lastKey)
      return lastKey.takeError();
    return collector.rejectCanonicalSeedRange(
        std::move(*firstKey), std::move(*lastKey), cardinality.count,
        cardinality.overflow,
        *capabilityAdmitted
            ? TechMatchSeedRejectionReason::RealizationInadmissible
            : TechMatchSeedRejectionReason::CapabilityInadmissible);
  }

  TechComputeRealizationView realization{0, capabilityTemplate, actorSelection,
                                         boundaries};
  auto key = canonicalTechMatchRowKey(realization, inputs.dataflow.identity());
  if (!key)
    return key.takeError();
  auto entered = collector.beginSeed(std::move(*key));
  if (!entered)
    return entered.takeError();
  if (!*entered)
    return llvm::Error::success();
  if (!*capabilityAdmitted)
    return collector.reject(
        TechMatchSeedRejectionReason::CapabilityInadmissible);
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

  auto cached = cacheActorTopology(inputs, actors);
  if (!cached)
    return cached.takeError();

  std::vector<std::vector<CanonicalActorOption>> optionsByActor;
  optionsByActor.reserve(actorCount);
  for (const auto &actor : actors) {
    auto options = canonicalActorOptions(inputs, actor, operations);
    if (!options)
      return options.takeError();
    optionsByActor.push_back(std::move(*options));
  }

  std::vector<std::uint8_t> usedOperations(operationCount, false);
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
            selectionCanComplete(actors, topology, selection, *cached,
                                 actor + 1, selectedGraph, remaining - 1);
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

  std::vector<std::vector<std::size_t>> operationNeighbors(operationCount);
  const auto operationOrdinal =
      [&](const auto &operation) -> std::optional<std::size_t> {
    const auto found = llvm::find(operations, operation);
    if (found == operations.end())
      return std::nullopt;
    return static_cast<std::size_t>(found - operations.begin());
  };
  for (const TemplateEdge &edge : topology) {
    const auto *source =
        std::get_if<::loom::fabric::FabricFuNodePortRef>(&edge.source.payload);
    const auto *sink = std::get_if<::loom::fabric::FabricFuNodePortRef>(
        &edge.destination.payload);
    if (!source || !sink)
      continue;
    const auto sourceOrdinal = operationOrdinal(source->node);
    const auto sinkOrdinal = operationOrdinal(sink->node);
    if (!sourceOrdinal || !sinkOrdinal || *sourceOrdinal == *sinkOrdinal)
      continue;
    operationNeighbors[*sourceOrdinal].push_back(*sinkOrdinal);
    operationNeighbors[*sinkOrdinal].push_back(*sourceOrdinal);
  }
  for (auto &neighbors : operationNeighbors) {
    llvm::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  std::vector<std::uint8_t> reached(operationCount, false);
  std::vector<std::size_t> queue = {0};
  reached[0] = true;
  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor)
    for (std::size_t neighbor : operationNeighbors[queue[cursor]])
      if (!reached[neighbor]) {
        reached[neighbor] = true;
        queue.push_back(neighbor);
      }
  if (!llvm::all_of(reached, [](std::uint8_t value) { return value != 0; }))
    return visit(0, operationCount, std::nullopt);

  struct IndexedOption final {
    std::size_t actorOrdinal;
    const CanonicalActorOption *option;
  };
  struct Completion final {
    std::vector<TechComputeActorView> actors;
    std::vector<std::vector<std::uint8_t>> key;
  };

  std::vector<std::vector<IndexedOption>> optionsByOperation(operationCount);
  for (auto [actorOrdinal, options] : llvm::enumerate(optionsByActor))
    for (const CanonicalActorOption &option : options)
      if (actorPairCompatible(option.actor, option.actor, topology, *cached))
        optionsByOperation[option.operationOrdinal].push_back(
            {actorOrdinal, &option});

  std::vector<std::optional<IndexedOption>> assignment(operationCount);
  std::vector<std::uint8_t> usedActors(actorCount, false);
  for (std::size_t anchorActor = 0;
       continueEnumeration && anchorActor + operationCount <= actorCount;
       ++anchorActor) {
    for (const CanonicalActorOption &anchor : optionsByActor[anchorActor]) {
      if (!continueEnumeration)
        break;
      if (!actorPairCompatible(anchor.actor, anchor.actor, topology, *cached))
        continue;
      selection = {anchor.actor};
      auto anchorViable = selectionCanComplete(
          actors, topology, selection, *cached, anchorActor + 1,
          actors[anchorActor].graph, operationCount - 1);
      if (!anchorViable)
        return anchorViable.takeError();
      if (!*anchorViable)
        continue;

      assignment[anchor.operationOrdinal] = IndexedOption{anchorActor, &anchor};
      usedActors[anchorActor] = true;
      std::vector<Completion> completions;
      std::function<llvm::Error(std::size_t)> search =
          [&](std::size_t assignedCount) -> llvm::Error {
        if (assignedCount == operationCount) {
          Completion completion;
          completion.actors.reserve(operationCount);
          completion.key.reserve(operationCount);
          for (const auto &selected : assignment) {
            assert(selected);
            completion.actors.push_back(selected->option->actor);
            completion.key.push_back(selected->option->key);
          }
          llvm::sort(completion.actors, [](const auto &lhs, const auto &rhs) {
            return lhs.actor.entity.value() < rhs.actor.entity.value();
          });
          llvm::sort(completion.key);
          auto viable =
              selectionCanComplete(actors, topology, completion.actors, *cached,
                                   actorCount, actors[anchorActor].graph, 0);
          if (!viable)
            return viable.takeError();
          if (*viable)
            completions.push_back(std::move(completion));
          return llvm::Error::success();
        }

        std::size_t selectedOperation = operationCount;
        std::vector<IndexedOption> selectedDomain;
        for (std::size_t operation = 0; operation < operationCount;
             ++operation) {
          if (assignment[operation])
            continue;
          const bool onFrontier = llvm::any_of(
              operationNeighbors[operation], [&](std::size_t neighbor) {
                return assignment[neighbor].has_value();
              });
          if (!onFrontier)
            continue;

          std::vector<IndexedOption> domain;
          for (const IndexedOption &candidate : optionsByOperation[operation]) {
            if (candidate.actorOrdinal <= anchorActor ||
                usedActors[candidate.actorOrdinal] ||
                actors[candidate.actorOrdinal].graph !=
                    actors[anchorActor].graph)
              continue;
            bool compatible = true;
            for (const auto &selected : assignment)
              if (selected && !actorPairCompatible(candidate.option->actor,
                                                   selected->option->actor,
                                                   topology, *cached)) {
                compatible = false;
                break;
              }
            if (compatible)
              domain.push_back(candidate);
          }
          if (domain.empty())
            return llvm::Error::success();
          if (selectedOperation == operationCount ||
              domain.size() < selectedDomain.size()) {
            selectedOperation = operation;
            selectedDomain = std::move(domain);
          }
        }
        if (selectedOperation == operationCount)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "tech_mapping_generation_invalid: connected template search "
              "lost its operation frontier");

        for (const IndexedOption &candidate : selectedDomain) {
          assignment[selectedOperation] = candidate;
          usedActors[candidate.actorOrdinal] = true;
          if (llvm::Error error = search(assignedCount + 1))
            return error;
          usedActors[candidate.actorOrdinal] = false;
          assignment[selectedOperation].reset();
        }
        return llvm::Error::success();
      };
      if (llvm::Error error = search(1))
        return error;
      usedActors[anchorActor] = false;
      assignment[anchor.operationOrdinal].reset();

      llvm::sort(completions, [](const auto &lhs, const auto &rhs) {
        return lhs.key < rhs.key;
      });
      completions.erase(std::unique(completions.begin(), completions.end(),
                                    [](const auto &lhs, const auto &rhs) {
                                      return lhs.key == rhs.key;
                                    }),
                        completions.end());
      for (const Completion &completion : completions) {
        auto result = emitSelection(completion.actors);
        if (!result)
          return result.takeError();
        continueEnumeration = *result;
        if (!continueEnumeration)
          break;
      }
    }
  }
  return llvm::Error::success();
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
