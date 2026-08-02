#include "TechMappingCandidateDomain.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

struct MemoryActorOption final {
  TechMemoryActorView actor;
  std::vector<TechMemoryGraphBoundaryView> boundaries;
  std::vector<std::uint8_t> key;
  bool capabilityAdmitted = false;
};

struct MemoryActorDomain final {
  const ::dataflow::CanonicalActorView *actor;
  std::vector<MemoryActorOption> options;
};

llvm::Expected<std::optional<std::vector<TechMemoryGraphBoundaryView>>>
mergeBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                const ArtifactIdentity &owner);

llvm::Expected<std::vector<TechMemoryGraphBoundaryView>>
collectBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                  const ArtifactIdentity &owner);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_generation_invalid: " + message);
}

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
endpointForRole(
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    const ::loom::fabric::MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::ServiceValueRole role) {
  const auto binding = llvm::find_if(
      capability.roleToEndpoint,
      [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
        return candidate.role == role;
      });
  if (binding == capability.roleToEndpoint.end())
    return invalid("memory capability omits a canonical service role");
  return ::loom::fabric::FabricMemoryEngineTemplateEndpointRef{
      engine, binding->endpointOrdinal};
}

llvm::Expected<std::vector<std::uint8_t>>
boundaryTerminalKey(const TechMemoryGraphEndpointRef &terminal,
                    const ArtifactIdentity &owner) {
  std::vector<std::uint8_t> key;
  if (const auto *producer =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
              &terminal)) {
    appendU32(key, 0);
    if (llvm::Error error = appendDataflowRef(key, owner, *producer))
      return std::move(error);
  } else {
    appendU32(key, 1);
    if (llvm::Error error = appendDataflowRef(
            key, owner,
            std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(terminal)))
      return std::move(error);
  }
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
internalEdgeKey(const TechMemoryInternalEdgeView &edge,
                const ArtifactIdentity &owner) {
  std::vector<std::uint8_t> key;
  if (llvm::Error error = appendDataflowRef(key, owner, edge.producer))
    return std::move(error);
  if (llvm::Error error = appendDataflowRef(key, owner, edge.consumer))
    return std::move(error);
  appendFabricRef(key, edge.connection);
  return key;
}

llvm::Expected<MemoryActorOption> buildActorOption(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef port,
    ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef
        alternative,
    const ::loom::fabric::MemoryCapabilityAlternativeView &capability,
    const ::dataflow::semantics::CanonicalService &service,
    bool capabilityAdmitted) {
  MemoryActorOption option{
      TechMemoryActorView{actor.ref, port, alternative, {}, {}},
      {},
      {},
      capabilityAdmitted};
  for (const auto &value : service.arguments()) {
    auto endpoint = endpointForRole(engine, capability, value.role);
    if (!endpoint)
      return endpoint.takeError();
    option.actor.operandPorts.push_back(*endpoint);
  }
  for (const auto &value : service.results()) {
    auto endpoint = endpointForRole(engine, capability, value.role);
    if (!endpoint)
      return endpoint.takeError();
    option.actor.resultPorts.push_back(*endpoint);
  }

  for (auto [ordinal, endpoint] : llvm::enumerate(option.actor.operandPorts)) {
    auto operand = service.argumentValue(actor.op, ordinal);
    if (!operand)
      return operand.takeError();
    const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
        ::dataflow::ActorTokenOperandRef{actor.ref,
                                         (*operand)->getOperandNumber()};
    auto producer = inputs.dataflow.graphProducer(consumer);
    if (!producer)
      return producer.takeError();
    if (std::holds_alternative<::dataflow::GraphIngressTokenRef>(*producer))
      option.boundaries.push_back(
          TechMemoryGraphBoundaryView{*producer, endpoint});
  }
  for (auto [ordinal, endpoint] : llvm::enumerate(option.actor.resultPorts)) {
    auto result = service.resultValue(actor.op, ordinal);
    if (!result)
      return result.takeError();
    const ::dataflow::CanonicalGraphProducerEndpointRef producer =
        ::dataflow::ActorTokenResultRef{actor.ref, result->getResultNumber()};
    auto consumers = inputs.dataflow.graphConsumers(producer);
    if (!consumers)
      return consumers.takeError();
    for (const auto &consumer : *consumers)
      if (std::holds_alternative<::dataflow::GraphEgressTokenRef>(consumer))
        option.boundaries.push_back(
            TechMemoryGraphBoundaryView{consumer, endpoint});
  }

  const std::array<const MemoryActorOption *, 1> singleton{&option};
  auto boundaries = collectBoundaries(singleton, inputs.dataflow.identity());
  if (!boundaries)
    return boundaries.takeError();
  option.boundaries = std::move(*boundaries);

  auto key =
      canonicalTechMatchActorKey(option.actor, inputs.dataflow.identity());
  if (!key)
    return key.takeError();
  option.key = std::move(*key);
  return option;
}

llvm::Expected<std::vector<MemoryActorOption>> actorOptions(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    const ::loom::fabric::FabricMemoryEngineTemplateRecord &engineRecord) {
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
  if (!projection)
    return projection.takeError();
  auto service = ::dataflow::semantics::CanonicalService::forActor(actor.op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actor.op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }

  std::vector<MemoryActorOption> options;
  for (std::uint64_t portOrdinal = 0;
       portOrdinal < engineRecord.operationPorts.size(); ++portOrdinal) {
    const ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef port{
        engine, portOrdinal};
    const auto *portView =
        inputs.fabric.memoryEngineTemplateOperationPort(port);
    if (!portView)
      return invalid("sealed Fabric memory operation port does not resolve");
    auto matches =
        portView->matchingCapabilities(*projection, *service, access);
    if (!matches)
      return matches.takeError();
    llvm::sort(*matches, [](const auto &lhs, const auto &rhs) {
      return lhs.alternativeOrdinal < rhs.alternativeOrdinal;
    });
    for (std::uint64_t alternativeOrdinal = 0;
         alternativeOrdinal < portView->capabilityAlternatives().size();
         ++alternativeOrdinal) {
      const ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef
          alternative{port, alternativeOrdinal};
      const auto *capability =
          inputs.fabric.memoryEngineTemplateCapabilityAlternative(alternative);
      if (!capability)
        return invalid("memory capability alternative does not resolve");
      const bool admitted = llvm::any_of(
          *matches, [&](const ::fabric::MemoryCapabilityMatch &match) {
            return match.alternativeOrdinal == alternativeOrdinal;
          });
      const bool hasCompleteRoleMap =
          llvm::all_of(service->arguments(),
                       [&](const auto &value) {
                         return llvm::any_of(capability->roleToEndpoint,
                                             [&](const auto &binding) {
                                               return binding.role ==
                                                      value.role;
                                             });
                       }) &&
          llvm::all_of(service->results(), [&](const auto &value) {
            return llvm::any_of(capability->roleToEndpoint,
                                [&](const auto &binding) {
                                  return binding.role == value.role;
                                });
          });
      if (!hasCompleteRoleMap)
        continue;
      auto option = buildActorOption(inputs, actor, engine, port, alternative,
                                     *capability, *service, admitted);
      if (!option)
        return option.takeError();
      options.push_back(std::move(*option));
    }
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

llvm::Expected<std::optional<std::vector<TechMemoryGraphBoundaryView>>>
mergeBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                const ArtifactIdentity &owner) {
  auto merged = collectBoundaries(selection, owner);
  if (!merged)
    return merged.takeError();
  for (std::size_t index = 1; index < merged->size(); ++index) {
    auto previous = boundaryTerminalKey((*merged)[index - 1].terminal, owner);
    if (!previous)
      return previous.takeError();
    auto current = boundaryTerminalKey((*merged)[index].terminal, owner);
    if (!current)
      return current.takeError();
    if (*previous == *current &&
        (*merged)[index - 1].endpoint != (*merged)[index].endpoint)
      return std::optional<std::vector<TechMemoryGraphBoundaryView>>{};
  }
  return std::optional<std::vector<TechMemoryGraphBoundaryView>>(
      std::move(*merged));
}

llvm::Expected<std::vector<TechMemoryGraphBoundaryView>>
collectBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                  const ArtifactIdentity &owner) {
  using KeyedBoundary =
      std::pair<std::vector<std::uint8_t>, TechMemoryGraphBoundaryView>;
  std::vector<KeyedBoundary> keyed;
  for (const MemoryActorOption *option : selection)
    for (const TechMemoryGraphBoundaryView &boundary : option->boundaries) {
      auto key = boundaryTerminalKey(boundary.terminal, owner);
      if (!key)
        return key.takeError();
      keyed.emplace_back(std::move(*key), boundary);
    }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    return ::loom::fabric::canonicalFabricBytes(lhs.second.endpoint) <
           ::loom::fabric::canonicalFabricBytes(rhs.second.endpoint);
  });

  std::vector<TechMemoryGraphBoundaryView> result;
  for (std::size_t index = 0; index < keyed.size(); ++index) {
    if (index != 0 && keyed[index - 1].first == keyed[index].first &&
        keyed[index - 1].second.endpoint == keyed[index].second.endpoint)
      continue;
    result.push_back(std::move(keyed[index].second));
  }
  return result;
}

const MemoryActorOption *
findActor(llvm::ArrayRef<const MemoryActorOption *> selection,
          ::dataflow::ActorRef actor) {
  auto found = llvm::find_if(selection, [&](const MemoryActorOption *option) {
    return option->actor.actor == actor;
  });
  return found == selection.end() ? nullptr : *found;
}

llvm::Expected<std::vector<TechMemoryInternalEdgeView>>
deriveInternalEdges(const TechMappingGenerationInputs &inputs,
                    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
                    llvm::ArrayRef<const MemoryActorOption *> selection) {
  using KeyedEdge =
      std::pair<std::vector<std::uint8_t>, TechMemoryInternalEdgeView>;
  std::vector<KeyedEdge> keyed;
  for (const MemoryActorOption *source : selection) {
    auto sourceActor = inputs.dataflow.resolve(source->actor.actor);
    if (!sourceActor)
      return sourceActor.takeError();
    auto sourceService =
        ::dataflow::semantics::CanonicalService::forActor(sourceActor->op);
    if (!sourceService)
      return sourceService.takeError();
    for (auto [resultOrdinal, sourceEndpoint] :
         llvm::enumerate(source->actor.resultPorts)) {
      auto result = sourceService->resultValue(sourceActor->op, resultOrdinal);
      if (!result)
        return result.takeError();
      const ::dataflow::ActorTokenResultRef producer{source->actor.actor,
                                                     result->getResultNumber()};
      auto consumers = inputs.dataflow.graphConsumers(
          ::dataflow::CanonicalGraphProducerEndpointRef{producer});
      if (!consumers)
        return consumers.takeError();
      for (const auto &consumerRef : *consumers) {
        const auto *consumer =
            std::get_if<::dataflow::ActorTokenOperandRef>(&consumerRef);
        if (!consumer)
          continue;
        const MemoryActorOption *sink = findActor(selection, consumer->actor);
        if (!sink)
          continue;
        auto sinkActor = inputs.dataflow.resolve(sink->actor.actor);
        if (!sinkActor)
          return sinkActor.takeError();
        auto sinkService =
            ::dataflow::semantics::CanonicalService::forActor(sinkActor->op);
        if (!sinkService)
          return sinkService.takeError();
        for (auto [argumentOrdinal, sinkEndpoint] :
             llvm::enumerate(sink->actor.operandPorts)) {
          auto operand =
              sinkService->argumentValue(sinkActor->op, argumentOrdinal);
          if (!operand)
            return operand.takeError();
          if ((*operand)->getOperandNumber() != consumer->ordinal)
            continue;
          const ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef
              connection{engine, sourceEndpoint, sinkEndpoint};
          if (!inputs.fabric.hasMemoryEngineTemplateInternalConnection(
                  connection))
            break;
          TechMemoryInternalEdgeView edge{
              ::dataflow::CanonicalGraphProducerEndpointRef{producer},
              ::dataflow::CanonicalGraphConsumerEndpointRef{*consumer},
              connection};
          auto key = internalEdgeKey(edge, inputs.dataflow.identity());
          if (!key)
            return key.takeError();
          keyed.emplace_back(std::move(*key), std::move(edge));
          break;
        }
      }
    }
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  std::vector<TechMemoryInternalEdgeView> edges;
  for (std::size_t index = 0; index < keyed.size(); ++index) {
    if (index != 0 && keyed[index - 1].first == keyed[index].first)
      continue;
    edges.push_back(std::move(keyed[index].second));
  }
  return edges;
}

llvm::Error emitInternalEdgeSubsets(
    const TechMappingGenerationInputs &inputs, TechMemoryRealizationView base,
    llvm::ArrayRef<TechMemoryInternalEdgeView> eligibleEdges,
    llvm::ArrayRef<const MemoryActorOption *> selection,
    TechMatchRowCollector &collector) {
  std::vector<TechMemoryInternalEdgeView> selectedEdges;
  std::vector<::dataflow::ActorRef> covered;
  covered.reserve(base.actors.size());
  for (const TechMemoryActorView &actor : base.actors)
    covered.push_back(actor.actor);

  for (std::size_t count = 0; count <= eligibleEdges.size(); ++count) {
    std::function<llvm::Error(std::size_t, std::size_t)> choose =
        [&](std::size_t start, std::size_t remaining) -> llvm::Error {
      if (collector.truncated())
        return llvm::Error::success();
      if (remaining != 0) {
        for (std::size_t index = start;
             index + remaining <= eligibleEdges.size(); ++index) {
          selectedEdges.push_back(eligibleEdges[index]);
          if (llvm::Error error = choose(index + 1, remaining - 1))
            return error;
          selectedEdges.pop_back();
          if (collector.truncated())
            break;
        }
        return llvm::Error::success();
      }

      TechMemoryRealizationView row = base;
      row.internalEdges = selectedEdges;
      auto key = canonicalTechMatchRowKey(row, inputs.dataflow.identity());
      if (!key)
        return key.takeError();
      auto entered = collector.beginSeed(std::move(*key));
      if (!entered)
        return entered.takeError();
      if (!*entered)
        return llvm::Error::success();
      if (llvm::any_of(selection, [](const MemoryActorOption *option) {
            return !option->capabilityAdmitted;
          }))
        return collector.reject(
            TechMatchSeedRejectionReason::CapabilityInadmissible);
      auto boundaries = mergeBoundaries(selection, inputs.dataflow.identity());
      if (!boundaries)
        return boundaries.takeError();
      if (!*boundaries)
        return collector.reject(
            TechMatchSeedRejectionReason::CorrespondenceInadmissible);
      row.graphBoundaries = std::move(**boundaries);
      if (llvm::Error error = verifyTechMemoryRealizationClosure(
              row, inputs.dataflow, inputs.fabric)) {
        llvm::consumeError(std::move(error));
        return collector.reject(
            TechMatchSeedRejectionReason::RealizationInadmissible);
      }
      return collector.admit(std::move(row), covered);
    };
    if (llvm::Error error = choose(0, count))
      return error;
    if (collector.truncated())
      break;
  }
  return llvm::Error::success();
}

llvm::Error emitCanonicalMemorySelections(
    const TechMappingGenerationInputs &inputs,
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    llvm::ArrayRef<MemoryActorDomain> domains, std::size_t actorCount,
    TechMatchRowCollector &collector) {
  std::vector<const MemoryActorOption *> selection;
  std::function<llvm::Error(std::size_t, std::size_t,
                            std::optional<::dataflow::GraphRef>)>
      choose = [&](std::size_t start, std::size_t remaining,
                   std::optional<::dataflow::GraphRef> graph) -> llvm::Error {
    if (collector.truncated())
      return llvm::Error::success();
    if (remaining != 0) {
      for (std::size_t actor = start; actor + remaining <= domains.size();
           ++actor) {
        if (graph && domains[actor].actor->graph != *graph)
          continue;
        const ::dataflow::GraphRef selectedGraph =
            graph ? *graph : domains[actor].actor->graph;
        for (const MemoryActorOption &option : domains[actor].options) {
          if (collector.truncated())
            break;
          selection.push_back(&option);
          if (llvm::Error error =
                  choose(actor + 1, remaining - 1, selectedGraph))
            return error;
          selection.pop_back();
        }
        if (collector.truncated())
          break;
      }
      return llvm::Error::success();
    }

    auto boundaries = collectBoundaries(selection, inputs.dataflow.identity());
    if (!boundaries)
      return boundaries.takeError();
    auto internalEdges = deriveInternalEdges(inputs, engine, selection);
    if (!internalEdges)
      return internalEdges.takeError();

    std::vector<TechMemoryActorView> actors;
    actors.reserve(selection.size());
    for (const MemoryActorOption *option : selection)
      actors.push_back(option->actor);
    TechMemoryRealizationView base{
        0, engine, std::move(actors), std::move(*boundaries), {}};
    return emitInternalEdgeSubsets(inputs, std::move(base), *internalEdges,
                                   selection, collector);
  };
  return choose(0, actorCount, std::nullopt);
}

} // namespace

llvm::Error
deriveMemoryRows(const TechMappingGenerationInputs &inputs,
                 llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
                 TechMatchRowCollector &collector) {
  std::vector<::loom::fabric::FabricMemoryEngineTemplateRef> engines(
      inputs.fabric.memoryEngineTemplates().begin(),
      inputs.fabric.memoryEngineTemplates().end());
  llvm::sort(engines, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  for (const ::loom::fabric::FabricMemoryEngineTemplateRef engine : engines) {
    const auto *engineRecord = inputs.fabric.memoryEngineTemplate(engine);
    if (!engineRecord)
      return invalid("sealed Fabric memory template does not resolve");

    std::vector<MemoryActorDomain> domains;
    std::map<std::uint64_t, std::size_t> graphActorCounts;
    for (const ::dataflow::CanonicalActorView &actor : selectedActors) {
      if (actor.kind != ::dataflow::CanonicalDataflowActorKind::Memory)
        continue;
      auto options = actorOptions(inputs, actor, engine, *engineRecord);
      if (!options)
        return options.takeError();
      domains.push_back(MemoryActorDomain{&actor, std::move(*options)});
      ++graphActorCounts[actor.graph.entity.value()];
    }
    llvm::sort(domains, [](const auto &lhs, const auto &rhs) {
      return lhs.actor->ref.entity.value() < rhs.actor->ref.entity.value();
    });
    const std::size_t maxActorCount =
        graphActorCounts.empty()
            ? 0
            : llvm::max_element(graphActorCounts, [](const auto &lhs,
                                                     const auto &rhs) {
                return lhs.second < rhs.second;
              })->second;

    for (std::size_t actorCount = 1; actorCount <= maxActorCount;
         ++actorCount) {
      if (llvm::Error error = emitCanonicalMemorySelections(
              inputs, engine, domains, actorCount, collector))
        return error;
      if (collector.truncated())
        return llvm::Error::success();
    }
  }
  return llvm::Error::success();
}

} // namespace loom::mapping::detail
