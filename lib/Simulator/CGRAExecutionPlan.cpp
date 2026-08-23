#include "CGRAExecutionPlan.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Error add(std::uint64_t value, std::uint64_t &total,
                llvm::StringRef label) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    return llvm::createStringError(std::errc::value_too_large,
                                   "CGRA %s count overflow",
                                   label.str().c_str());
  total += value;
  return llvm::Error::success();
}

template <typename Realization>
llvm::Expected<std::uint64_t>
realizationGraph(const Realization &realization,
                 const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (realization.actors.empty())
    return invalid("CGRA preparation found an empty Tech realization");
  std::optional<std::uint64_t> graph;
  for (const auto &actor : realization.actors) {
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    const std::uint64_t current = resolved->graph.entity.value();
    if (graph && *graph != current)
      return invalid("CGRA preparation found a cross-graph Tech realization");
    graph = current;
  }
  return *graph;
}

llvm::Expected<std::vector<::dataflow::GraphRef>>
deriveMappedGraphs(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const ::loom::mapping::TechMappingView &tech,
                   const ::loom::mapping::SpatialMappingView &spatial) {
  llvm::DenseMap<std::uint64_t, std::uint64_t> computeGraphs;
  computeGraphs.reserve(tech.computeRealizations().size());
  for (const auto &realization : tech.computeRealizations()) {
    auto graph = realizationGraph(realization, dataflow);
    if (!graph)
      return graph.takeError();
    if (!computeGraphs.try_emplace(realization.entityId, *graph).second)
      return invalid("CGRA preparation found duplicate compute realizations");
  }

  llvm::DenseMap<std::uint64_t, std::uint64_t> memoryGraphs;
  memoryGraphs.reserve(tech.memoryRealizations().size());
  for (const auto &realization : tech.memoryRealizations()) {
    auto graph = realizationGraph(realization, dataflow);
    if (!graph)
      return graph.takeError();
    if (!memoryGraphs.try_emplace(realization.entityId, *graph).second)
      return invalid("CGRA preparation found duplicate memory realizations");
  }

  llvm::DenseSet<std::uint64_t> selectedGraphs;
  selectedGraphs.reserve(tech.covers().size());
  for (const auto &binding : spatial.computeBindings()) {
    auto graph = computeGraphs.find(binding.realization);
    if (graph == computeGraphs.end())
      return invalid("CGRA preparation found an unknown compute realization");
    selectedGraphs.insert(graph->second);
  }
  for (const auto &binding : spatial.memoryEngineBindings()) {
    auto graph = memoryGraphs.find(binding.realization);
    if (graph == memoryGraphs.end())
      return invalid("CGRA preparation found an unknown memory realization");
    selectedGraphs.insert(graph->second);
  }

  std::vector<::dataflow::GraphRef> result;
  result.reserve(tech.covers().size());
  for (::dataflow::GraphRef graph : tech.covers()) {
    if (!selectedGraphs.contains(graph.entity.value()))
      return invalid("CGRA preparation found a covered graph without a "
                     "selected physical realization");
    result.push_back(graph);
  }
  if (result.empty())
    return invalid("CGRA preparation requires a nonempty covered graph set");
  return result;
}

llvm::Expected<CgraExecutionPlanSummary>
deriveSummary(const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const ::loom::mapping::TechMappingView &tech,
              const ::loom::mapping::SpatialMappingView &spatial,
              std::uint64_t mappedGraphCount) {
  llvm::DenseMap<std::uint64_t,
                 const ::loom::mapping::TechComputeRealizationView *>
      realizations;
  realizations.reserve(tech.computeRealizations().size());
  for (const auto &realization : tech.computeRealizations())
    realizations.try_emplace(realization.entityId, &realization);

  CgraExecutionPlanSummary summary;
  summary.mappedGraphCount = mappedGraphCount;
  summary.semanticConfigurationFieldCount =
      spatial.configuredHardware().fields().size();
  for (const auto &binding : spatial.computeBindings()) {
    auto found = realizations.find(binding.realization);
    if (found == realizations.end())
      return invalid("CGRA preparation cannot resolve a compute binding");
    const auto &realization = *found->second;
    if (llvm::Error error = add(realization.actors.size(),
                                summary.computeActorCount, "compute actor"))
      return std::move(error);
    for (const auto &actorBinding : realization.actors) {
      auto actor = dataflow.resolve(actorBinding.actor);
      if (!actor)
        return actor.takeError();
      auto projection =
          ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
      if (!projection)
        return projection.takeError();
      auto transitions = ::dataflow::semantics::projectActorHandshakeCases(
          projection->schema, actorBinding.operandPorts.size(),
          actorBinding.resultPorts.size());
      if (!transitions)
        return transitions.takeError();
      if (llvm::Error error =
              add(transitions->size(), summary.actorTransitionCount,
                  "actor transition"))
        return std::move(error);
    }
  }
  return summary;
}

struct ComputeTriggerKey final {
  std::uint64_t realization = 0;
  std::vector<std::uint8_t> event;

  friend bool operator<(const ComputeTriggerKey &lhs,
                        const ComputeTriggerKey &rhs) {
    return std::tie(lhs.realization, lhs.event) <
           std::tie(rhs.realization, rhs.event);
  }
};

struct SelectedComputeActor final {
  std::uint64_t realization = 0;
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricFuOccurrenceRef occurrence;
  ::loom::fabric::InstructionContextRef context;
  std::optional<std::uint64_t> temporalDispatchDomain;
  std::uint32_t temporalDispatchPosition = 0;
};

struct ComputeExecutionProjection final {
  std::vector<CgraComputeActorPlan> actors;
  std::vector<CgraComputeTransitionPlan> transitions;
  std::vector<CgraTemporalDispatchDomainPlan> temporalDispatchDomains;
  std::vector<std::uint64_t> physicalUses;
};

bool sameTiming(const CgraPhysicalUseTiming &lhs,
                const CgraPhysicalUseTiming &rhs) {
  return lhs.acquireRank == rhs.acquireRank &&
         lhs.commitRank == rhs.commitRank &&
         lhs.releaseRank == rhs.releaseRank &&
         lhs.acquireEventOrdinal == rhs.acquireEventOrdinal &&
         lhs.releaseEventOrdinal == rhs.releaseEventOrdinal &&
         lhs.commitEventOrdinal == rhs.commitEventOrdinal &&
         lhs.requiresCausalRelease == rhs.requiresCausalRelease;
}

llvm::Expected<std::uint64_t> appendPhysicalActivation(
    llvm::ArrayRef<::loom::fabric::FabricUsePatternRef> inputPatterns,
    CgraPhysicalUseClientKind client, bool requiresCausalRelease,
    const ::loom::fabric::FabricArtifactView &fabric,
    const std::map<std::vector<std::uint8_t>, std::uint64_t> &ownerOrdinals,
    CgraFrozenExecutionPlan &result,
    std::vector<CgraResourcePatternSelection> &selectedPatterns,
    std::vector<CgraResourceActivationSelection> &activations,
    llvm::DenseSet<std::uint64_t> &selectedOwners) {
  if (inputPatterns.empty())
    return invalid("CGRA physical activation has no exact UsePattern");
  std::map<std::vector<std::uint8_t>, ::loom::fabric::FabricUsePatternRef>
      patterns;
  for (const auto &reference : inputPatterns)
    patterns.try_emplace(::loom::fabric::canonicalFabricBytes(reference),
                         reference);
  if (patterns.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("CGRA physical activation pattern count exceeds u32");

  std::optional<::loom::fabric::FabricInventoryOwnerRef> activationOwner;
  std::optional<CgraPhysicalUseTiming> timing;
  std::uint64_t ownerOrdinal = 0;
  const std::uint64_t actionOrdinal = result.physicalUses.size();
  const std::uint64_t patternOffset = result.physicalUsePatterns.size();
  const std::uint64_t selectionOffset = selectedPatterns.size();
  for (const auto &[key, reference] : patterns) {
    (void)key;
    const auto owner = reference.owner.catalog();
    if (activationOwner && *activationOwner != owner)
      return invalid("CGRA physical activation spans Fabric owners");
    if (!activationOwner) {
      activationOwner = owner;
      auto found =
          ownerOrdinals.find(::loom::fabric::canonicalFabricBytes(owner));
      if (found == ownerOrdinals.end())
        return invalid("CGRA physical activation owner is absent from Fabric");
      ownerOrdinal = found->second;
    }
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract || reference.ordinal >= contract->usePatternCount())
      return invalid("CGRA physical activation has no exact resource contract");
    const ::fabric::UsePattern pattern =
        contract->usePattern(::fabric::UsePatternKey(reference.ordinal));
    const auto ranks = contract->eventOrder(pattern.timingAndProgress);
    CgraPhysicalUseTiming current{
        actionOrdinal,
        ranks[pattern.acquire.ordinal()],
        pattern.commit ? std::optional<std::uint32_t>(
                             ranks[pattern.commit->event.ordinal()])
                       : std::nullopt,
        ranks[pattern.release.ordinal()],
        pattern.acquire.ordinal(),
        pattern.release.ordinal(),
        pattern.commit
            ? std::optional<std::uint32_t>(pattern.commit->event.ordinal())
            : std::nullopt,
        requiresCausalRelease};
    if (timing && !sameTiming(*timing, current))
      return invalid("CGRA atomic activation has inconsistent owner timing");
    timing = current;
    result.physicalUsePatterns.push_back(reference);
    selectedPatterns.push_back(
        {ownerOrdinal, ::fabric::UsePatternKey(reference.ordinal)});
  }

  result.physicalUses.push_back({patternOffset,
                                 static_cast<std::uint32_t>(patterns.size()),
                                 ownerOrdinal});
  result.physicalUseClients.push_back(client);
  result.physicalUseTimings.push_back(*timing);
  if (llvm::Error error = add(timing->acquireRank,
                              result.summary.physicalUseAcquireRankSum,
                              "physical acquire rank"))
    return std::move(error);
  if (llvm::Error error = add(timing->releaseRank,
                              result.summary.physicalUseReleaseRankSum,
                              "physical release rank"))
    return std::move(error);
  result.summary.physicalUseMaxAcquireRank =
      std::max<std::uint64_t>(result.summary.physicalUseMaxAcquireRank,
                              timing->acquireRank);
  result.summary.physicalUseMaxReleaseRank =
      std::max<std::uint64_t>(result.summary.physicalUseMaxReleaseRank,
                              timing->releaseRank);
  if (timing->requiresCausalRelease)
    if (llvm::Error error = add(1, result.summary.physicalUseCausalReleaseCount,
                                "causal-release physical use"))
      return std::move(error);
  std::uint64_t *timingCount = nullptr;
  std::uint64_t *maxReleaseRank = nullptr;
  switch (client) {
  case CgraPhysicalUseClientKind::ComputeTransition:
    timingCount = &result.summary.computeTransitionTimingCount;
    maxReleaseRank = &result.summary.computeTransitionMaxReleaseRank;
    break;
  case CgraPhysicalUseClientKind::MemoryTransition:
    timingCount = &result.summary.memoryTransitionTimingCount;
    maxReleaseRank = &result.summary.memoryTransitionMaxReleaseRank;
    break;
  case CgraPhysicalUseClientKind::ProducedTransport:
    timingCount = &result.summary.producedTransportTimingCount;
    maxReleaseRank = &result.summary.producedTransportMaxReleaseRank;
    break;
  case CgraPhysicalUseClientKind::ConsumedTransport:
    timingCount = &result.summary.consumedTransportTimingCount;
    maxReleaseRank = &result.summary.consumedTransportMaxReleaseRank;
    break;
  case CgraPhysicalUseClientKind::TraversalTransport:
    timingCount = &result.summary.traversalTransportTimingCount;
    maxReleaseRank = &result.summary.traversalTransportMaxReleaseRank;
    break;
  }
  if (llvm::Error error = add(1, *timingCount, "typed physical-use client"))
    return std::move(error);
  *maxReleaseRank =
      std::max<std::uint64_t>(*maxReleaseRank, timing->releaseRank);
  activations.push_back(
      {selectionOffset, static_cast<std::uint32_t>(patterns.size())});
  selectedOwners.insert(ownerOrdinal);
  return actionOrdinal;
}

llvm::Expected<std::vector<CgraPhysicalUseClientKind>> derivePhysicalUseClients(
    llvm::ArrayRef<::loom::mapping::SpatialResourceUseView> uses,
    CgraExecutionPlanSummary &summary) {
  std::vector<CgraPhysicalUseClientKind> result;
  result.reserve(uses.size());
  for (const auto &use : uses) {
    if (std::holds_alternative<::loom::mapping::SpatialActorTransitionEventRef>(
            use.activation.trigger.event)) {
      if (std::holds_alternative<
              ::loom::mapping::SpatialComputeResourceOwnerRef>(use.owner)) {
        result.push_back(CgraPhysicalUseClientKind::ComputeTransition);
        if (llvm::Error error =
                add(1, summary.computeTransitionPhysicalUseCount,
                    "compute-transition physical use"))
          return std::move(error);
        continue;
      }
      if (std::holds_alternative<
              ::loom::mapping::SpatialMemoryEngineResourceOwnerRef>(
              use.owner) ||
          std::holds_alternative<
              ::loom::mapping::SpatialMemoryBindingResourceOwnerRef>(
              use.owner)) {
        result.push_back(CgraPhysicalUseClientKind::MemoryTransition);
        if (llvm::Error error = add(1, summary.memoryTransitionPhysicalUseCount,
                                    "memory-transition physical use"))
          return std::move(error);
        continue;
      }
      return invalid(
          "CGRA actor-transition ResourceUse has no execution owner");
    }
    if (std::holds_alternative<::dataflow::CanonicalGraphProducerEndpointRef>(
            use.activation.trigger.event)) {
      result.push_back(CgraPhysicalUseClientKind::ProducedTransport);
      if (llvm::Error error =
              add(1, summary.producedPhysicalUseCount, "produced physical use"))
        return std::move(error);
      continue;
    }
    if (!std::holds_alternative<::dataflow::CanonicalGraphConsumerEndpointRef>(
            use.activation.trigger.event))
      return invalid("CGRA ResourceUse has an unknown activity event");
    result.push_back(CgraPhysicalUseClientKind::ConsumedTransport);
    if (llvm::Error error =
            add(1, summary.consumedPhysicalUseCount, "consumed physical use"))
      return std::move(error);
  }
  return result;
}

struct TemporalDispatchProjection final {
  struct Selection final {
    std::uint64_t domain = 0;
    std::uint32_t position = 0;
  };

  std::vector<CgraTemporalDispatchDomainPlan> domains;
  llvm::DenseMap<std::uint64_t, Selection> byRealization;
};

llvm::Expected<TemporalDispatchProjection> deriveTemporalDispatchProjection(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<::loom::mapping::SpatialComputeBindingView> bindings) {
  auto projected =
      ::loom::mapping::deriveSpatialTemporalPeDispatchDomains(fabric, bindings);
  if (!projected)
    return projected.takeError();

  TemporalDispatchProjection result;
  result.domains.reserve(projected->size());
  result.byRealization.reserve(bindings.size());
  for (const auto &domain : *projected) {
    if (domain.candidates.empty() ||
        domain.candidates.size() > std::numeric_limits<std::uint32_t>::max() ||
        domain.resetPosition >= domain.candidates.size())
      return invalid("CGRA temporal dispatch domain has invalid cardinality");

    const std::uint64_t domainOrdinal = result.domains.size();
    result.domains.push_back({domain.pe, domain.allocationUnit,
                              static_cast<std::uint32_t>(
                                  domain.candidates.size()),
                              domain.resetPosition});
    for (auto [position, candidate] : llvm::enumerate(domain.candidates))
      if (!result.byRealization
               .try_emplace(
                   candidate.realization,
                   TemporalDispatchProjection::Selection{
                       domainOrdinal, static_cast<std::uint32_t>(position)})
               .second)
        return invalid("CGRA realization belongs to multiple temporal "
                       "dispatch domains");
  }
  return result;
}

llvm::Expected<ComputeExecutionProjection> deriveComputeExecutionProjection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients) {
  auto dispatch =
      deriveTemporalDispatchProjection(fabric, spatial.computeBindings());
  if (!dispatch)
    return dispatch.takeError();
  llvm::DenseMap<std::uint64_t,
                 const ::loom::mapping::TechComputeRealizationView *>
      realizations;
  realizations.reserve(tech.computeRealizations().size());
  for (const auto &realization : tech.computeRealizations())
    if (!realizations.try_emplace(realization.entityId, &realization).second)
      return invalid("CGRA compute execution found duplicate realizations");

  std::map<std::vector<std::uint8_t>, SelectedComputeActor> selectedActors;
  for (const auto &binding : spatial.computeBindings()) {
    auto realization = realizations.find(binding.realization);
    if (realization == realizations.end())
      return invalid("CGRA compute execution found an unknown realization");
    std::optional<std::uint64_t> dispatchDomain;
    std::uint32_t dispatchPosition = 0;
    if (const auto selected = dispatch->byRealization.find(binding.realization);
        selected != dispatch->byRealization.end()) {
      dispatchDomain = selected->second.domain;
      dispatchPosition = selected->second.position;
    } else {
      const auto pe = fabric.parentPeOf(binding.occurrence);
      if (!pe)
        return invalid("CGRA compute binding FU has no parent PE");
      if (fabric.peSchedule(*pe) == ::fabric::Schedule::Temporal)
        return invalid("CGRA temporal compute binding has no dispatch domain");
    }
    for (const auto &actor : realization->second->actors) {
      auto key =
          ::dataflow::encodeDataflowReference(dataflow.identity(), actor.actor);
      if (!key)
        return key.takeError();
      if (!selectedActors
               .try_emplace(
                   std::move(*key),
                   SelectedComputeActor{binding.realization, actor.actor,
                                        binding.occurrence, binding.context,
                                        dispatchDomain, dispatchPosition})
               .second)
        return invalid("CGRA compute actor has multiple physical bindings");
    }
  }

  std::map<ComputeTriggerKey, std::vector<std::uint64_t>> triggerUses;
  for (auto [actionOrdinal, use] : llvm::enumerate(spatial.resourceUses())) {
    if (actionOrdinal >= physicalUseClients.size())
      return invalid("CGRA physical-use client projection is incomplete");
    if (physicalUseClients[actionOrdinal] !=
        CgraPhysicalUseClientKind::ComputeTransition)
      continue;
    const auto *owner =
        std::get_if<::loom::mapping::SpatialComputeResourceOwnerRef>(
            &use.owner);
    const auto *trigger =
        std::get_if<::loom::mapping::SpatialActorTransitionEventRef>(
            &use.activation.trigger.event);
    if (!owner || !trigger)
      continue;
    auto event = ::loom::mapping::encodeSpatialActivityEventKey(
        dataflow.identity(), use.activation.trigger.event);
    if (!event)
      return event.takeError();
    triggerUses[ComputeTriggerKey{owner->realization, std::move(*event)}]
        .push_back(actionOrdinal);
  }

  ComputeExecutionProjection result;
  result.temporalDispatchDomains = std::move(dispatch->domains);
  result.actors.reserve(selectedActors.size());
  for (const auto &[key, selected] : selectedActors) {
    (void)key;
    auto actor = dataflow.resolve(selected.actor);
    if (!actor)
      return actor.takeError();
    auto projection =
        ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
    if (!projection)
      return projection.takeError();
    auto cases = ::dataflow::semantics::projectActorHandshakeCases(
        projection->schema, actor->op->getNumOperands(),
        actor->op->getNumResults());
    if (!cases)
      return cases.takeError();
    if (cases->size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA compute actor transition count exceeds u32");

    const std::uint64_t transitionOffset = result.transitions.size();
    for (const auto &transition : *cases) {
      const ::loom::mapping::SpatialActivityEventRef event =
          ::loom::mapping::SpatialActorTransitionEventRef{selected.actor,
                                                          transition.ordinal};
      auto encoded = ::loom::mapping::encodeSpatialActivityEventKey(
          dataflow.identity(), event);
      if (!encoded)
        return encoded.takeError();
      auto uses = triggerUses.find(
          ComputeTriggerKey{selected.realization, std::move(*encoded)});
      if (uses == triggerUses.end() || uses->second.empty())
        return invalid(
            "CGRA compute transition has no selected physical ResourceUse");
      if (uses->second.size() > std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA compute transition ResourceUse count exceeds u32");
      const std::uint64_t useOffset = result.physicalUses.size();
      result.physicalUses.insert(result.physicalUses.end(),
                                 uses->second.begin(), uses->second.end());
      result.transitions.push_back(CgraComputeTransitionPlan{
          transition.ordinal, useOffset,
          static_cast<std::uint32_t>(uses->second.size())});
      triggerUses.erase(uses);
    }
    result.actors.push_back(CgraComputeActorPlan{
        selected.actor, actor->graph, selected.occurrence, selected.context,
        transitionOffset, static_cast<std::uint32_t>(cases->size()),
        selected.temporalDispatchDomain, selected.temporalDispatchPosition});
  }
  if (!triggerUses.empty())
    return invalid(
        "CGRA compute ResourceUse trigger has no selected actor transition");
  return result;
}

} // namespace

llvm::Expected<CgraFrozenExecutionPlan> freezeCgraExecutionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial) {
  auto mappedGraphs = deriveMappedGraphs(dataflow, tech, spatial);
  if (!mappedGraphs)
    return mappedGraphs.takeError();
  auto summary =
      deriveSummary(dataflow, tech, spatial,
                    static_cast<std::uint64_t>(mappedGraphs->size()));
  if (!summary)
    return summary.takeError();
  auto physicalUseClients =
      derivePhysicalUseClients(spatial.resourceUses(), *summary);
  if (!physicalUseClients)
    return physicalUseClients.takeError();
  auto compute = deriveComputeExecutionProjection(dataflow, tech, fabric,
                                                  spatial, *physicalUseClients);
  if (!compute)
    return compute.takeError();
  if (compute->actors.size() != summary->computeActorCount ||
      compute->transitions.size() != summary->actorTransitionCount ||
      compute->physicalUses.size() !=
          summary->computeTransitionPhysicalUseCount)
    return invalid("CGRA compute execution projection count drifted");
  auto memory = freezeCgraMemoryPlan(dataflow, tech, fabric, spatial,
                                     *physicalUseClients);
  if (!memory)
    return memory.takeError();

  std::map<std::vector<std::uint8_t>, std::uint64_t> ownerOrdinals;
  std::vector<const ::fabric::ResourceContract *> ownerContracts;
  ownerContracts.reserve(fabric.moduleResourceOwners().size());
  for (auto [ordinal, owner] : llvm::enumerate(fabric.moduleResourceOwners())) {
    if (!ownerOrdinals
             .try_emplace(::loom::fabric::canonicalFabricBytes(owner),
                          static_cast<std::uint64_t>(ordinal))
             .second)
      return invalid("CGRA preparation found duplicate resource owners");
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract)
      return invalid("CGRA resource owner has no ResourceContract");
    ownerContracts.push_back(contract);
  }

  CgraFrozenExecutionPlan result;
  result.summary = *summary;
  result.mappedGraphs = std::move(*mappedGraphs);
  result.computeActors = std::move(compute->actors);
  result.computeTransitions = std::move(compute->transitions);
  result.temporalDispatchDomains = std::move(compute->temporalDispatchDomains);
  result.actorTransitionPhysicalUses = std::move(compute->physicalUses);
  result.memory = std::move(*memory);
  result.physicalUses.reserve(spatial.resourceUses().size());
  result.physicalUsePatterns.reserve(spatial.resourceUses().size());
  result.physicalUseClients.reserve(spatial.resourceUses().size());
  result.physicalUseTimings.reserve(spatial.resourceUses().size());
  std::vector<CgraResourcePatternSelection> selectedPatterns;
  selectedPatterns.reserve(spatial.resourceUses().size());
  std::vector<CgraResourceActivationSelection> activations;
  activations.reserve(spatial.resourceUses().size());
  llvm::DenseSet<std::uint64_t> selectedOwners;
  for (auto [ordinal, use] : llvm::enumerate(spatial.resourceUses())) {
    auto action = appendPhysicalActivation(
        llvm::ArrayRef<::loom::fabric::FabricUsePatternRef>(&use.useSite, 1),
        (*physicalUseClients)[ordinal], !use.activation.release.empty(), fabric,
        ownerOrdinals, result, selectedPatterns, activations, selectedOwners);
    if (!action)
      return action.takeError();
    if (*action != ordinal)
      return invalid("CGRA Mapping physical-use ordinal drifted");
  }
  auto transport =
      freezeCgraTransportPlan(dataflow, tech, fabric, spatial,
                              result.mappedGraphs, result.physicalUseClients);
  if (!transport)
    return transport.takeError();
  result.transport = std::move(*transport);
  std::map<std::uint64_t, std::vector<std::uint64_t>> traversalActivationUses;
  for (auto [ordinal, use] : llvm::enumerate(result.transport.traversalUses)) {
    if (use.activationInstanceOrdinal == invalidCgraTransportOrdinal)
      return invalid("CGRA traversal use has no activation instance");
    traversalActivationUses[use.activationInstanceOrdinal].push_back(ordinal);
  }
  for (const auto &[key, useOrdinals] : traversalActivationUses) {
    (void)key;
    std::vector<::loom::fabric::FabricUsePatternRef> patterns;
    patterns.reserve(useOrdinals.size());
    for (std::uint64_t ordinal : useOrdinals)
      patterns.push_back(result.transport.traversalUses[ordinal].pattern);
    auto action = appendPhysicalActivation(
        patterns, CgraPhysicalUseClientKind::TraversalTransport, false, fabric,
        ownerOrdinals, result, selectedPatterns, activations, selectedOwners);
    if (!action)
      return action.takeError();
    if (llvm::Error error = add(1, result.summary.traversalPhysicalUseCount,
                                "traversal physical use"))
      return std::move(error);
    for (std::uint64_t ordinal : useOrdinals)
      result.transport.traversalUses[ordinal].physicalUseOrdinal = *action;
  }
  std::map<std::vector<std::uint8_t>, std::uint64_t> traversalPatternActions;
  for (const CgraTraversalUsePlan &use : result.transport.traversalUses) {
    if (use.physicalUseOrdinal == invalidCgraTransportOrdinal)
      return invalid("CGRA traversal UsePattern has no physical action");
    if (use.requesterGroup.kind ==
        ::loom::fabric::FabricTraversalRequesterGroupKind::SwitchRequester)
      continue;
    auto [position, inserted] = traversalPatternActions.try_emplace(
        ::loom::fabric::canonicalFabricBytes(use.pattern),
        use.physicalUseOrdinal);
    if (!inserted && position->second != use.physicalUseOrdinal)
      return invalid("CGRA traversal UsePattern has multiple physical actions");
  }
  const auto storageAction =
      [&](const ::loom::fabric::FabricUsePatternRef &pattern)
      -> llvm::Expected<std::uint64_t> {
    auto found = traversalPatternActions.find(
        ::loom::fabric::canonicalFabricBytes(pattern));
    if (found != traversalPatternActions.end())
      return found->second;
    auto action = appendPhysicalActivation(
        llvm::ArrayRef<::loom::fabric::FabricUsePatternRef>(&pattern, 1),
        CgraPhysicalUseClientKind::TraversalTransport, false, fabric,
        ownerOrdinals, result, selectedPatterns, activations, selectedOwners);
    if (!action)
      return action.takeError();
    traversalPatternActions.emplace(
        ::loom::fabric::canonicalFabricBytes(pattern), *action);
    if (llvm::Error error = add(1, result.summary.traversalPhysicalUseCount,
                                "traversal storage physical use"))
      return std::move(error);
    return *action;
  };
  for (CgraTraversalStoragePlan &storage : result.transport.traversalStorages) {
    auto enqueue = storageAction(storage.enqueuePattern);
    if (!enqueue)
      return enqueue.takeError();
    auto dequeue = storageAction(storage.dequeuePattern);
    if (!dequeue)
      return dequeue.takeError();
    storage.enqueuePhysicalUseOrdinal = *enqueue;
    storage.dequeuePhysicalUseOrdinal = *dequeue;
    if (storage.simultaneousPattern) {
      auto simultaneous = storageAction(*storage.simultaneousPattern);
      if (!simultaneous)
        return simultaneous.takeError();
      storage.simultaneousPhysicalUseOrdinal = *simultaneous;
    }
  }
  auto resources = freezeCgraResourceRuntimePlan(ownerContracts,
                                                 selectedPatterns, activations);
  if (!resources)
    return resources.takeError();
  for (CgraTraversalStoragePlan &storage : result.transport.traversalStorages) {
    if (storage.enqueuePhysicalUseOrdinal >= resources->selectedUses.size() ||
        storage.dequeuePhysicalUseOrdinal >= resources->selectedUses.size())
      return invalid("CGRA storage actions are absent from resource plan");
    const CgraResourceUsePlan &enqueue =
        resources->selectedUses[storage.enqueuePhysicalUseOrdinal];
    const CgraResourceUsePlan &dequeue =
        resources->selectedUses[storage.dequeuePhysicalUseOrdinal];
    if (enqueue.claimOffset > resources->claims.size() ||
        enqueue.claimCount > resources->claims.size() - enqueue.claimOffset ||
        dequeue.claimOffset > resources->claims.size() ||
        dequeue.claimCount > resources->claims.size() - dequeue.claimOffset)
      return invalid("CGRA storage action claim slice is malformed");
    llvm::SmallDenseSet<std::uint64_t, 4> enqueueDimensions;
    for (const CgraResourceClaimPlan &claim :
         llvm::ArrayRef(resources->claims)
             .slice(enqueue.claimOffset, enqueue.claimCount))
      enqueueDimensions.insert(claim.dimensionOrdinal);
    storage.independentReadWriteServices = true;
    for (const CgraResourceClaimPlan &claim :
         llvm::ArrayRef(resources->claims)
             .slice(dequeue.claimOffset, dequeue.claimCount))
      if (enqueueDimensions.contains(claim.dimensionOrdinal)) {
        storage.independentReadWriteServices = false;
        break;
      }
  }
  result.resources = std::move(*resources);
  result.summary.physicalUseCount =
      static_cast<std::uint64_t>(result.physicalUses.size());
  result.summary.temporalComputeActorCount = llvm::count_if(
      result.computeActors,
      [](const CgraComputeActorPlan &actor) {
        return actor.temporalDispatchDomain.has_value();
      });
  result.summary.spatialComputeActorCount =
      result.computeActors.size() - result.summary.temporalComputeActorCount;
  result.summary.temporalDispatchDomainCount =
      result.temporalDispatchDomains.size();
  result.summary.operandBufferCount = result.transport.operandBuffers.size();
  result.summary.resourceOwnerCount =
      static_cast<std::uint64_t>(selectedOwners.size());
  result.summary.claimCount =
      static_cast<std::uint64_t>(result.resources.claims.size());
  result.summary.routeTreeCount = result.transport.routes.size();
  result.summary.routeNodeCount = result.transport.routeNodes.size();
  result.summary.routeSinkCount = result.transport.routeSinks.size();
  for (const CgraRoutePlan &route : result.transport.routes) {
    std::vector<std::uint64_t> depths(route.nodeCount, 0);
    for (std::uint32_t local = 0; local != route.nodeCount; ++local) {
      const CgraRouteNodePlan &node =
          result.transport.routeNodes[route.nodeOffset + local];
      std::uint64_t depth = 1;
      if (node.parentOrdinal != std::numeric_limits<std::uint32_t>::max()) {
        if (node.parentOrdinal >= local)
          return invalid("CGRA route parent is not topologically ordered");
        depth = depths[node.parentOrdinal] + 1;
      }
      depths[local] = depth;
      result.summary.maximumRouteNodeDepth =
          std::max(result.summary.maximumRouteNodeDepth, depth);
    }
  }
  result.summary.selectedTraversalCount = result.transport.traversals.size();
  result.summary.localTransferCount = result.transport.localTransfers.size();
  result.summary.localTransferSinkCount =
      result.transport.localTransferSinks.size();
  result.summary.physicalTagSegmentCount = result.transport.physicalTags.size();
  result.summary.taggedRouteNodeCount = llvm::count_if(
      result.transport.routeNodes, [](const CgraRouteNodePlan &node) {
        return node.physicalTagOrdinal != invalidCgraTransportOrdinal;
      });
  result.summary.memoryActorCount = result.memory.actors.size();
  result.summary.memoryRootedUseCount = result.memory.rootedUses.size();
  result.summary.memoryChildTransactionCount =
      result.memory.childTransactions.size();
  result.summary.memoryResultAssemblyCount =
      result.memory.resultAssemblies.size();
  return result;
}

} // namespace loom::sim::detail
