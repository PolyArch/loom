#include "CGRAExecutionPlan.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

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
};

struct ComputeExecutionProjection final {
  std::vector<CgraComputeActorPlan> actors;
  std::vector<CgraComputeTransitionPlan> transitions;
  std::vector<std::uint64_t> physicalUses;
};

llvm::Expected<ComputeExecutionProjection> deriveComputeExecutionProjection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::mapping::SpatialMappingView &spatial) {
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
    for (const auto &actor : realization->second->actors) {
      auto key =
          ::dataflow::encodeDataflowReference(dataflow.identity(), actor.actor);
      if (!key)
        return key.takeError();
      if (!selectedActors
               .try_emplace(
                   std::move(*key),
                   SelectedComputeActor{binding.realization, actor.actor,
                                        binding.occurrence, binding.context})
               .second)
        return invalid("CGRA compute actor has multiple physical bindings");
    }
  }

  std::map<ComputeTriggerKey, std::vector<std::uint64_t>> triggerUses;
  for (auto [actionOrdinal, use] : llvm::enumerate(spatial.resourceUses())) {
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
        transitionOffset, static_cast<std::uint32_t>(cases->size())});
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
  auto compute = deriveComputeExecutionProjection(dataflow, tech, spatial);
  if (!compute)
    return compute.takeError();
  if (compute->actors.size() != summary->computeActorCount ||
      compute->transitions.size() != summary->actorTransitionCount)
    return invalid("CGRA compute execution projection count drifted");

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
  result.summary.actorTriggeredPhysicalUseCount = compute->physicalUses.size();
  result.mappedGraphs = std::move(*mappedGraphs);
  result.computeActors = std::move(compute->actors);
  result.computeTransitions = std::move(compute->transitions);
  result.actorTransitionPhysicalUses = std::move(compute->physicalUses);
  result.physicalUses.reserve(spatial.resourceUses().size());
  result.physicalUseTimings.reserve(spatial.resourceUses().size());
  std::vector<CgraResourcePatternSelection> selectedPatterns;
  selectedPatterns.reserve(spatial.resourceUses().size());
  llvm::DenseSet<std::uint64_t> selectedOwners;
  for (const auto &use : spatial.resourceUses()) {
    const auto &owner = use.useSite.owner.catalog();
    auto ownerOrdinal =
        ownerOrdinals.find(::loom::fabric::canonicalFabricBytes(owner));
    if (ownerOrdinal == ownerOrdinals.end())
      return invalid("CGRA ResourceUse owner is absent from Fabric");
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract || use.useSite.ordinal >= contract->usePatternCount())
      return invalid("CGRA ResourceUse has no exact resource contract");
    const ::fabric::UsePattern pattern =
        contract->usePattern(::fabric::UsePatternKey(use.useSite.ordinal));
    const auto ranks = contract->eventOrder(pattern.timingAndProgress);

    CgraPhysicalUsePlan plan;
    plan.reference = use.useSite;
    plan.resourceOwnerOrdinal = ownerOrdinal->second;
    plan.requesterOrdinal = pattern.requester.ordinal();
    plan.eligibilityOrdinal = pattern.eligibility.ordinal();
    if (pattern.commit) {
      plan.transitionOrdinal = pattern.commit->transition.ordinal();
    }
    result.physicalUses.push_back(std::move(plan));
    result.physicalUseTimings.push_back(CgraPhysicalUseTiming{
        static_cast<std::uint64_t>(result.physicalUseTimings.size()),
        ranks[pattern.acquire.ordinal()],
        pattern.commit ? std::optional<std::uint32_t>(
                             ranks[pattern.commit->event.ordinal()])
                       : std::nullopt,
        ranks[pattern.release.ordinal()], pattern.acquire.ordinal(),
        pattern.release.ordinal(),
        pattern.commit
            ? std::optional<std::uint32_t>(pattern.commit->event.ordinal())
            : std::nullopt});
    selectedPatterns.push_back(
        {ownerOrdinal->second, ::fabric::UsePatternKey(use.useSite.ordinal)});
    selectedOwners.insert(ownerOrdinal->second);
  }
  auto resources =
      freezeCgraResourceRuntimePlan(ownerContracts, selectedPatterns);
  if (!resources)
    return resources.takeError();
  result.resources = std::move(*resources);
  result.summary.physicalUseCount =
      static_cast<std::uint64_t>(result.physicalUses.size());
  result.summary.resourceOwnerCount =
      static_cast<std::uint64_t>(selectedOwners.size());
  result.summary.claimCount =
      static_cast<std::uint64_t>(result.resources.claims.size());
  auto transport =
      freezeCgraTransportPlan(dataflow, fabric, spatial, result.mappedGraphs);
  if (!transport)
    return transport.takeError();
  result.transport = std::move(*transport);
  result.summary.routeTreeCount = result.transport.routes.size();
  result.summary.routeNodeCount = result.transport.routeNodes.size();
  result.summary.routeSinkCount = result.transport.routeSinks.size();
  result.summary.selectedTraversalCount = result.transport.traversals.size();
  result.summary.localTransferCount = result.transport.localTransfers.size();
  result.summary.localTransferSinkCount =
      result.transport.localTransferSinks.size();
  return result;
}

} // namespace loom::sim::detail
