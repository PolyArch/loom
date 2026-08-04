#include "CGRAExecutionPlan.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <system_error>
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

  std::map<std::vector<std::uint8_t>, std::uint64_t> ownerOrdinals;
  for (auto [ordinal, owner] : llvm::enumerate(fabric.moduleResourceOwners())) {
    if (!ownerOrdinals
             .try_emplace(::loom::fabric::canonicalFabricBytes(owner),
                          static_cast<std::uint64_t>(ordinal))
             .second)
      return invalid("CGRA preparation found duplicate resource owners");
  }

  CgraFrozenExecutionPlan result;
  result.summary = *summary;
  result.mappedGraphs = std::move(*mappedGraphs);
  result.physicalUses.reserve(spatial.resourceUses().size());
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
    plan.acquireRank = ranks[pattern.acquire.ordinal()];
    plan.releaseRank = ranks[pattern.release.ordinal()];
    if (pattern.commit) {
      plan.commitRank = ranks[pattern.commit->event.ordinal()];
      plan.transitionOrdinal = pattern.commit->transition.ordinal();
    }
    plan.claimOffset = static_cast<std::uint64_t>(result.claims.size());
    if (pattern.claims.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA ResourceUse claim count exceeds u32");
    plan.claimCount = static_cast<std::uint32_t>(pattern.claims.size());
    for (const ::fabric::Claim &claim : pattern.claims)
      result.claims.push_back(CgraResourceClaimPlan{claim.state.ordinal(),
                                                    claim.dimension.ordinal(),
                                                    claim.amount.value()});
    result.physicalUses.push_back(std::move(plan));
    selectedOwners.insert(ownerOrdinal->second);
  }
  result.summary.physicalUseCount =
      static_cast<std::uint64_t>(result.physicalUses.size());
  result.summary.resourceOwnerCount =
      static_cast<std::uint64_t>(selectedOwners.size());
  result.summary.claimCount = static_cast<std::uint64_t>(result.claims.size());
  return result;
}

} // namespace loom::sim::detail
