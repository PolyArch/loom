#include "DSE/JointMappingMigration.h"

#include "Common/ArtifactStore.h"
#include "DSE/HardwareDecision.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <set>
#include <system_error>
#include <type_traits>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_mapping_migration_invalid: " + message);
}

void canonicalize(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

const CompletedDsePlanExecution &
availableExecution(const DsePlanExecutionOutcome &outcome) {
  if (const auto *completed = std::get_if<CompletedDsePlanExecution>(&outcome))
    return *completed;
  return std::get<IncompleteDsePlanExecution>(outcome).availableExecution();
}

void appendAvailableRoots(const CompletedDsePlanExecution &execution,
                          llvm::ArrayRef<PlanOutputRef> outputs,
                          std::vector<ArtifactRootReference> &roots) {
  for (PlanOutputRef output : outputs) {
    if (!execution.hasOutput(output))
      continue;
    const auto available = execution.resolve(output);
    roots.insert(roots.end(), available.begin(), available.end());
  }
}

std::vector<ArtifactRootReference>
mappingRoots(const JointDesignExecution &execution) {
  std::vector<ArtifactRootReference> roots;
  for (const JointMappedPair &pair : execution.mappedPairs)
    roots.insert(roots.end(), pair.systemMappings.begin(),
                 pair.systemMappings.end());
  canonicalize(roots);
  return roots;
}

struct TechRecord final {
  ArtifactIdentity parentIdentity;
  mapping::FinalizedTechMapping child;
};

std::uint64_t spatialDecisionCount(const mapping::SpatialMappingView &mapping) {
  return mapping.computeBindings().size() +
         mapping.memoryEngineBindings().size() +
         mapping.memoryBindings().size() +
         mapping.registerFifoTransfers().size() + mapping.routeTrees().size() +
         mapping.resourceUses().size();
}

std::uint64_t
spatialRouteNodeCount(const mapping::SpatialMappingView &mapping) {
  std::uint64_t count = 0;
  for (const auto &route : mapping.routeTrees())
    count += route.nodes.size() + route.sinks.size();
  return count;
}

std::uint64_t techDecisionCount(const mapping::TechMappingView &mapping) {
  return mapping.computeRealizations().size() +
         mapping.memoryRealizations().size();
}

template <typename Ref>
bool physicalOwnerMatches(const fabric::FabricModulePhysicalOwnerRef &target,
                          const Ref &reference) {
  auto candidate = fabric::FabricModulePhysicalOwnerRef::create(reference);
  return candidate && *candidate == target;
}

bool endpointUsesOwner(const fabric::FabricTransportEndpointRef &endpoint,
    const fabric::FabricModulePhysicalOwnerRef &target) {
  return std::visit(
      [&](const auto &owner) {
        using Owner = std::decay_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner, fabric::FabricPeOccurrenceRef> ||
            std::is_same_v<Owner, fabric::FabricFuOccurrenceRef> ||
                      std::is_same_v<Owner,
                                     fabric::FabricMemoryOccurrenceRef> ||
                      std::is_same_v<Owner,
                                     fabric::FabricSwitchOccurrenceRef> ||
            std::is_same_v<Owner, fabric::FabricFifoOccurrenceRef> ||
                      std::is_same_v<Owner,
                                     fabric::FabricBoundaryOccurrenceRef>)
          return physicalOwnerMatches(target, owner);
        return false;
      },
      endpoint.owner.payload);
}

bool traversalUsesOwner(const fabric::FabricPhysicalTraversalRef &traversal,
    const fabric::FabricModulePhysicalOwnerRef &target) {
  return std::visit(
      [&](const auto &payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload,
                                     fabric::FabricPointConnectionPayload>)
          return endpointUsesOwner(payload.source, target) ||
                 endpointUsesOwner(payload.destination, target);
        if constexpr (std::is_same_v<Payload, fabric::FabricPeSelectorPayload>)
          return physicalOwnerMatches(target, payload.owner) ||
                 endpointUsesOwner(payload.source, target) ||
                 endpointUsesOwner(payload.destination, target);
        if constexpr (std::is_same_v<Payload,
                                     fabric::FabricPeRegisterFifoPayload>)
          return physicalOwnerMatches(target, payload.owner);
        if constexpr (std::is_same_v<Payload,
                                     fabric::FabricSwitchTraversalPayload>)
          return physicalOwnerMatches(target, payload.owner);
        if constexpr (std::is_same_v<Payload,
                                     fabric::FabricFifoTraversalPayload>)
          return physicalOwnerMatches(target, payload.owner);
        if constexpr (std::is_same_v<Payload,
                                     fabric::FabricBoundaryTraversalPayload>)
          return physicalOwnerMatches(target, payload.owner);
        return false;
      },
      traversal.payload);
}

bool spatialMappingUsesOwner(
    const mapping::SpatialMappingView &mapping,
    const fabric::FabricArtifactView &fabric,
    const fabric::FabricModulePhysicalOwnerRef &target) {
  for (const auto &binding : mapping.computeBindings()) {
    if (physicalOwnerMatches(target, binding.occurrence))
      return true;
    auto parent = fabric.parentPeOf(binding.occurrence);
    if (parent && physicalOwnerMatches(target, *parent))
      return true;
  }
  for (const auto &binding : mapping.memoryEngineBindings())
    if (physicalOwnerMatches(target, binding.occurrence))
      return true;
  for (const auto &transfer : mapping.registerFifoTransfers())
    if (physicalOwnerMatches(target, transfer.pe))
      return true;
  for (const auto &route : mapping.routeTrees()) {
    if (endpointUsesOwner(route.rootEndpoint, target) ||
        (route.localTraversal &&
         traversalUsesOwner(*route.localTraversal, target)))
      return true;
    for (const auto &node : route.nodes) {
      if (endpointUsesOwner(node.endpoint, target) ||
          (node.incomingTraversal &&
           traversalUsesOwner(*node.incomingTraversal, target)))
        return true;
    }
    for (const auto &sink : route.sinks)
      if (sink.localTraversal &&
          traversalUsesOwner(*sink.localTraversal, target))
        return true;
  }
  return false;
}

bool spatialMappingUsesImpact(
    const mapping::SpatialMappingView &mapping,
    const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<fabric::FabricModulePhysicalOwnerRef> placementRoots) {
  if (placementRoots.empty())
    return true;
  bool used = false;
  for (const auto &root : placementRoots)
    used |= spatialMappingUsesOwner(mapping, fabric, root);
  return used;
}

std::uint64_t systemServiceLegCount(const mapping::SystemMappingView &mapping) {
  std::uint64_t count = 0;
  for (const auto &service : mapping.serviceRealizations())
    for (const auto &plan : service.plans)
      count += plan.transferLegs.size();
  return count;
}

std::uint64_t
systemServiceLegCount(const mapping::SystemServiceRealizationView &service) {
  std::uint64_t count = 0;
  for (const auto &plan : service.plans)
    count += plan.transferLegs.size();
  return count;
}

bool bindingTargetsAccCore(
    const mapping::SystemThreadExecutionBindingView &binding,
    llvm::ArrayRef<fabric::AccCoreOccurrenceRef> targets) {
  const auto matches = [&](fabric::AccCoreOccurrenceRef target) {
    return llvm::is_contained(targets, target);
  };
  if (binding.defaultTarget && matches(*binding.defaultTarget))
    return true;
  if (llvm::any_of(binding.clauses,
                   [&](const auto &clause) { return matches(clause.target); }))
    return true;
  return llvm::any_of(binding.stableKeyEntries,
                      [&](const auto &entry) { return matches(entry.target); });
}

const TechRecord *findTechRecord(llvm::ArrayRef<TechRecord> records,
                                 const ArtifactIdentity &parent) {
  const auto found = llvm::find_if(records, [&](const TechRecord &record) {
    return record.parentIdentity == parent;
  });
  return found == records.end() ? nullptr : &*found;
}

std::uint64_t invalidationRootCount(const HardwareImpactProjection &impact) {
  return impact.tech.realizationRoots.size() +
         impact.spatial.placementRoots.size() +
         impact.spatial.routeRoots.size() +
         impact.system.executionRoots.size() +
         impact.system.instructionContextRoots.size() +
         impact.system.transportRoots.size() + impact.system.routeRoots.size() +
         impact.system.serviceRoots.size() +
         impact.system.memoryServiceRoots.size() +
         impact.system.memoryRoots.size();
}

template <typename T>
void appendUnique(std::vector<T> &destination, llvm::ArrayRef<T> source) {
  for (const T &value : source)
    if (!llvm::is_contained(destination, value))
      destination.push_back(value);
}

HardwareMappingImpactKind strongestImpactKind(HardwareMappingImpactKind lhs,
                                              HardwareMappingImpactKind rhs) {
  if (lhs == HardwareMappingImpactKind::Reopen ||
      rhs == HardwareMappingImpactKind::Reopen)
    return HardwareMappingImpactKind::Reopen;
  if (lhs == HardwareMappingImpactKind::Rebase ||
      rhs == HardwareMappingImpactKind::Rebase)
    return HardwareMappingImpactKind::Rebase;
  return HardwareMappingImpactKind::Unchanged;
}

HardwareImpactProjection
aggregateColdFallbackImpact(llvm::ArrayRef<HardwareImpactProjection> impacts,
                            const ArtifactRootReference &childSystem) {
  HardwareImpactProjection aggregate = impacts.front();
  aggregate.child = childSystem;
  aggregate.locality = HardwareMutationLocality::GlobalReopen;
  aggregate.tech = {};
  aggregate.spatial = {};
  aggregate.system = {};
  aggregate.moduleEntities.clear();
  for (const HardwareImpactProjection &impact : impacts) {
    aggregate.tech.kind =
        strongestImpactKind(aggregate.tech.kind, impact.tech.kind);
    appendUnique(aggregate.tech.realizationRoots,
                 llvm::ArrayRef(impact.tech.realizationRoots));
    aggregate.spatial.kind =
        strongestImpactKind(aggregate.spatial.kind, impact.spatial.kind);
    appendUnique(aggregate.spatial.placementRoots,
                 llvm::ArrayRef(impact.spatial.placementRoots));
    appendUnique(aggregate.spatial.routeRoots,
                 llvm::ArrayRef(impact.spatial.routeRoots));
    aggregate.system.kind =
        strongestImpactKind(aggregate.system.kind, impact.system.kind);
    appendUnique(aggregate.system.executionRoots,
                 llvm::ArrayRef(impact.system.executionRoots));
    appendUnique(aggregate.system.instructionContextRoots,
                 llvm::ArrayRef(impact.system.instructionContextRoots));
    appendUnique(aggregate.system.transportRoots,
                 llvm::ArrayRef(impact.system.transportRoots));
    appendUnique(aggregate.system.routeRoots,
                 llvm::ArrayRef(impact.system.routeRoots));
    appendUnique(aggregate.system.serviceRoots,
                 llvm::ArrayRef(impact.system.serviceRoots));
    appendUnique(aggregate.system.memoryServiceRoots,
                 llvm::ArrayRef(impact.system.memoryServiceRoots));
    appendUnique(aggregate.system.memoryRoots,
                 llvm::ArrayRef(impact.system.memoryRoots));
  }
  return aggregate;
}

llvm::Error
accountColdFallbackCone(const JointDesignExplorationPlan &parentPlan,
                        const JointDesignExecution &parentExecution,
                        const HardwareImpactProjection &impact,
                        JointMappingRebaseAccounting &accounting,
                        const ArtifactStore &artifacts,
                        llvm::ArrayRef<ArtifactRootReference> parentMappings) {
  const JointDesignPlanPair &pair = parentPlan.pairOutputs.front();
  const CompletedDsePlanExecution &available =
      availableExecution(parentExecution.planExecution);
  std::vector<ArtifactRootReference> parentTech = pair.immutableTechMappings;
  std::vector<ArtifactRootReference> parentSpatial =
      pair.immutableSpatialMappings;
  appendAvailableRoots(available, pair.techMappings, parentTech);
  appendAvailableRoots(available, pair.spatialMappings, parentSpatial);
  canonicalize(parentSpatial);
  for (const ArtifactRootReference &spatialReference : parentSpatial) {
    auto spatial = mapping::importSpatialMapping(spatialReference, artifacts);
    if (!spatial)
      return spatial.takeError();
    parentTech.push_back({mapping::mappingArtifactSchema.identity.str(),
                          mapping::mappingArtifactSchema.version,
                          spatial->view().techMappingIdentity()});
  }
  canonicalize(parentTech);

  accounting.parentTechMappings = parentTech.size();
  accounting.parentSpatialMappings = parentSpatial.size();
  accounting.invalidatedTechMappings = parentTech.size();
  accounting.invalidatedSpatialMappings = parentSpatial.size();
  accounting.invalidationRootCount =
      projectJointHardwareInvalidationRootCount(
          llvm::ArrayRef<HardwareImpactProjection>(impact));
  for (const ArtifactRootReference &reference : parentTech) {
    auto imported = mapping::importTechMapping(reference, artifacts);
    if (!imported)
      return imported.takeError();
    accounting.parentTechDecisions += techDecisionCount(imported->view());
  }
  accounting.reopenedTechDecisions = accounting.parentTechDecisions;
  for (const ArtifactRootReference &reference : parentSpatial) {
    auto imported = mapping::importSpatialMapping(reference, artifacts);
    if (!imported)
      return imported.takeError();
    accounting.parentSpatialDecisions += spatialDecisionCount(imported->view());
    accounting.parentRouteNodeCount += spatialRouteNodeCount(imported->view());
  }
  accounting.reopenedSpatialDecisions = accounting.parentSpatialDecisions;
  accounting.reopenedRouteNodeCount = accounting.parentRouteNodeCount;

  for (const ArtifactRootReference &reference : parentMappings) {
    auto imported = mapping::importSystemMapping(reference, artifacts);
    if (!imported)
      return imported.takeError();
    const auto &view = imported->view();
    accounting.parentThreadBindingCount +=
        view.executionBindings().threadBindings().size();
    accounting.parentGraphBindingCount +=
        view.executionBindings().graphBindings().size();
    accounting.parentResourceUseCount += view.resourceUses().size();
    accounting.parentServiceRealizationCount +=
        view.serviceRealizations().size();
    accounting.parentServiceLegCount += systemServiceLegCount(view);
  }
  accounting.reopenedThreadBindingCount = accounting.parentThreadBindingCount;
  accounting.reopenedGraphBindingCount = accounting.parentGraphBindingCount;
  accounting.reopenedResourceUseCount = accounting.parentResourceUseCount;
  accounting.reopenedServiceRealizationCount =
      accounting.parentServiceRealizationCount;
  accounting.reopenedServiceLegCount = accounting.parentServiceLegCount;
  accounting.invalidationConeDecisionCount =
      accounting.reopenedTechDecisions + accounting.reopenedSpatialDecisions +
      accounting.reopenedRouteNodeCount +
      accounting.reopenedThreadBindingCount +
      accounting.reopenedGraphBindingCount +
      accounting.reopenedResourceUseCount +
      accounting.reopenedServiceRealizationCount +
      accounting.reopenedServiceLegCount;
  return llvm::Error::success();
}

} // namespace

llvm::StringRef jointMappingRebaseFailureReasonSpelling(
    JointMappingRebaseFailureReason reason) {
  switch (reason) {
  case JointMappingRebaseFailureReason::MissingParentFrontier:
    return "missing_parent_frontier";
  case JointMappingRebaseFailureReason::MissingImpactProjection:
    return "missing_impact_projection";
  case JointMappingRebaseFailureReason::ImpactRequiresColdFallback:
    return "impact_requires_cold_fallback";
  case JointMappingRebaseFailureReason::ModuleCorrespondence:
    return "module_correspondence";
  case JointMappingRebaseFailureReason::TechImpactReopened:
    return "tech_impact_reopened";
  case JointMappingRebaseFailureReason::SpatialImpactReopened:
    return "spatial_impact_reopened";
  case JointMappingRebaseFailureReason::TechRebaseRejected:
    return "tech_rebase_rejected";
  case JointMappingRebaseFailureReason::SpatialRebaseRejected:
    return "spatial_rebase_rejected";
  }
  llvm_unreachable("unknown joint Mapping rebase failure reason");
}

llvm::StringRef
jointMappingReuseDispositionSpelling(JointMappingReuseDisposition disposition) {
  switch (disposition) {
  case JointMappingReuseDisposition::Preserved:
    return "preserved";
  case JointMappingReuseDisposition::LocalRepair:
    return "local_repair";
  case JointMappingReuseDisposition::ColdFallback:
    return "cold_fallback";
  }
  llvm_unreachable("unknown joint Mapping reuse disposition");
}

llvm::Error validateJointMappingRebaseAccounting(
    const JointMappingRebaseAccounting &accounting) {
  const auto closedSum = [](std::initializer_list<std::uint64_t> values)
      -> std::optional<std::uint64_t> {
    std::uint64_t total = 0;
    for (const std::uint64_t value : values) {
      auto next = llvm::checkedAddUnsigned(total, value);
      if (!next)
        return std::nullopt;
      total = *next;
    }
    return total;
  };
  const auto requirePartition = [&](std::uint64_t parent,
                                    std::initializer_list<std::uint64_t> parts,
                                    llvm::StringRef name) -> llvm::Error {
    const auto total = closedSum(parts);
    if (!total)
      return invalid(name + " accounting overflows");
    if (*total != parent)
      return invalid(name + " accounting is not closed");
    return llvm::Error::success();
  };

  if (llvm::Error error = requirePartition(accounting.parentTechMappings,
                                           {accounting.preservedTechMappings,
                                            accounting.repairedTechMappings,
                                            accounting.invalidatedTechMappings},
                                           "TechMapping"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentSpatialMappings,
                           {accounting.preservedSpatialMappings,
                            accounting.repairedSpatialMappings,
                            accounting.invalidatedSpatialMappings},
                           "SpatialMapping"))
    return error;
  if (llvm::Error error = requirePartition(accounting.parentTechDecisions,
                                           {accounting.preservedTechDecisions,
                                            accounting.repairedTechDecisions,
                                            accounting.reopenedTechDecisions},
                                           "Tech decision"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentSpatialDecisions,
                           {accounting.preservedSpatialDecisions,
                            accounting.repairedSpatialDecisions,
                            accounting.reopenedSpatialDecisions},
                           "Spatial decision"))
    return error;
  if (llvm::Error error = requirePartition(accounting.parentRouteNodeCount,
                                           {accounting.preservedRouteNodeCount,
                                            accounting.repairedRouteNodeCount,
                                            accounting.reopenedRouteNodeCount},
                                           "route node"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentThreadBindingCount,
                           {accounting.preservedThreadBindingCount,
                            accounting.reopenedThreadBindingCount},
                           "thread binding"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentGraphBindingCount,
                           {accounting.preservedGraphBindingCount,
                            accounting.reopenedGraphBindingCount},
                           "graph binding"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentResourceUseCount,
                           {accounting.preservedResourceUseCount,
                            accounting.reopenedResourceUseCount},
                           "ResourceUse"))
    return error;
  if (llvm::Error error =
          requirePartition(accounting.parentServiceRealizationCount,
                           {accounting.preservedServiceRealizationCount,
                            accounting.reopenedServiceRealizationCount},
                           "service realization"))
    return error;
  if (llvm::Error error = requirePartition(accounting.parentServiceLegCount,
                                           {accounting.preservedServiceLegCount,
                                            accounting.reopenedServiceLegCount},
                                           "service leg"))
    return error;

  const auto cone = closedSum(
      {accounting.reopenedTechDecisions, accounting.reopenedSpatialDecisions,
       accounting.reopenedRouteNodeCount, accounting.repairedTechDecisions,
       accounting.repairedSpatialDecisions, accounting.repairedRouteNodeCount,
       accounting.reopenedThreadBindingCount,
       accounting.reopenedGraphBindingCount,
       accounting.reopenedResourceUseCount,
       accounting.reopenedServiceRealizationCount,
       accounting.reopenedServiceLegCount});
  if (!cone)
    return invalid("invalidation cone accounting overflows");
  if (*cone != accounting.invalidationConeDecisionCount)
    return invalid("invalidation cone accounting is not closed");
  return llvm::Error::success();
}

llvm::Expected<std::vector<ArtifactRootReference>>
resolveJointSpatialMappingFrontier(const JointDesignExplorationPlan &plan,
                                   const JointDesignExecution &execution) {
  if (plan.pairOutputs.size() != 1)
    return invalid("Spatial frontier reuse requires one exact Mapping pair");
  const JointDesignPlanPair &pair = plan.pairOutputs.front();
  std::vector<ArtifactRootReference> mappings = pair.immutableSpatialMappings;
  appendAvailableRoots(availableExecution(execution.planExecution),
                       pair.spatialMappings, mappings);
  canonicalize(mappings);
  if (mappings.empty())
    return invalid("Mapping execution has no reusable Spatial frontier");
  return mappings;
}

std::vector<::dataflow::RootThreadLaunchRef> projectJointSystemReopenRoots(
    const mapping::SystemMappingView &parentMapping,
    llvm::ArrayRef<HardwareImpactProjection> impacts) {
  std::vector<fabric::AccCoreOccurrenceRef> executionRoots;
  bool reopenEveryRoot = false;
  for (const HardwareImpactProjection &impact : impacts) {
    if (impact.system.kind != HardwareMappingImpactKind::Reopen)
      continue;
    appendUnique(executionRoots, llvm::ArrayRef(impact.system.executionRoots));
    for (fabric::InstructionCoreContextRef context :
         impact.system.instructionContextRoots)
      if (!llvm::is_contained(executionRoots, context.core))
        executionRoots.push_back(context.core);
    reopenEveryRoot |= !impact.system.transportRoots.empty() ||
                       !impact.system.routeRoots.empty() ||
                       !impact.system.serviceRoots.empty() ||
                       !impact.system.memoryServiceRoots.empty() ||
                       !impact.system.memoryRoots.empty();
  }

  std::vector<::dataflow::RootThreadLaunchRef> roots;
  const auto append = [&](::dataflow::RootThreadLaunchRef root) {
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  };
  const mapping::SystemExecutionBindingView &execution =
      parentMapping.executionBindings();
  if (reopenEveryRoot) {
    for (const auto root : execution.rootThreadLaunches())
      append(root);
    return roots;
  }
  for (const mapping::SystemThreadExecutionBindingView &binding :
       execution.threadBindings())
    if (bindingTargetsAccCore(binding, executionRoots))
      append(binding.key);
  return roots;
}

std::uint64_t projectJointHardwareInvalidationRootCount(
    llvm::ArrayRef<HardwareImpactProjection> impacts) {
  if (impacts.empty())
    return 0;
  if (impacts.size() == 1)
    return invalidationRootCount(impacts.front());
  const ArtifactRootReference &terminal =
      impacts.back().child ? *impacts.back().child : impacts.back().parent;
  return invalidationRootCount(aggregateColdFallbackImpact(impacts, terminal));
}

llvm::Expected<JointMappingRebaseResult> rebaseJointMappingFrontier(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const ArtifactRootReference &childSystem,
    llvm::ArrayRef<pnr::SystemModuleCorrespondence> moduleCorrespondences,
    llvm::ArrayRef<HardwareImpactProjection> impacts,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> selectedParentMapping) {
  JointMappingRebaseResult result;
  std::vector<ArtifactRootReference> parentMappings =
      mappingRoots(parentExecution);
  if (selectedParentMapping) {
    if (!llvm::is_contained(parentMappings, *selectedParentMapping))
      return invalid("selected parent Mapping is not in its execution");
    parentMappings = {*selectedParentMapping};
  }
  if (parentMappings.empty())
    return invalid("Mapping rebase requires one exact parent Mapping");
  if (impacts.empty()) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::MissingImpactProjection, std::nullopt,
         "Mapping rebase requires a typed hardware impact projection"});
    return result;
  }
  if (llvm::any_of(impacts,
                   [](const HardwareImpactProjection &impact) {
                     return !impact.child;
                   }) ||
      *impacts.back().child != childSystem)
    return invalid("hardware impact lineage does not end at its child System");
  if (parentPlan.pairOutputs.size() != 1)
    return invalid("Mapping rebase requires one exact parent pair");
  if (impacts.size() != 1) {
    const HardwareImpactProjection aggregate =
        aggregateColdFallbackImpact(impacts, childSystem);
    if (llvm::Error error = accountColdFallbackCone(
            parentPlan, parentExecution, aggregate, result.accounting,
            artifacts, parentMappings))
      return std::move(error);
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ImpactRequiresColdFallback,
         std::nullopt,
         "composed hardware lineage requires cold Mapping repair"});
    if (llvm::Error error =
            validateJointMappingRebaseAccounting(result.accounting))
      return std::move(error);
    return result;
  }
  const HardwareImpactProjection &impact = impacts.front();
  if (impact.locality == HardwareMutationLocality::GlobalReopen &&
      (impact.tech.kind == HardwareMappingImpactKind::Reopen ||
       impact.spatial.kind == HardwareMappingImpactKind::Reopen)) {
    if (llvm::Error error = accountColdFallbackCone(parentPlan, parentExecution,
                                                    impact, result.accounting,
                                                    artifacts, parentMappings))
      return std::move(error);
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ImpactRequiresColdFallback,
         std::nullopt, "typed hardware impact requires global Mapping repair"});
    if (llvm::Error error =
            validateJointMappingRebaseAccounting(result.accounting))
      return std::move(error);
    return result;
  }
  if (impact.tech.kind == HardwareMappingImpactKind::Reopen &&
      impact.tech.realizationRoots.empty())
    return invalid("typed Tech impact has no realization root");
  if (impact.spatial.kind == HardwareMappingImpactKind::Reopen &&
      impact.spatial.placementRoots.empty() &&
      impact.spatial.routeRoots.empty())
    return invalid("typed Spatial impact has no placement or route root");
  if (moduleCorrespondences.empty()) {
    if (llvm::Error error = accountColdFallbackCone(parentPlan, parentExecution,
                                                    impact, result.accounting,
                                                    artifacts, parentMappings))
      return std::move(error);
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ModuleCorrespondence, std::nullopt,
         "hardware child has no exact functional Module correspondence"});
    if (llvm::Error error =
            validateJointMappingRebaseAccounting(result.accounting))
      return std::move(error);
    return result;
  }

  auto targetModules = projectJointDesignTargetModules(childSystem, artifacts);
  if (!targetModules)
    return targetModules.takeError();
  std::set<ArtifactIdentity::Storage> targetModuleIdentities;
  for (const ArtifactRootReference &module : *targetModules)
    targetModuleIdentities.insert(module.artifact.bytes());

  std::map<ArtifactIdentity::Storage, ArtifactRootReference>
      childModuleByParent;
  std::set<ArtifactIdentity::Storage> childModules;
  for (const pnr::SystemModuleCorrespondence &correspondence :
       moduleCorrespondences) {
    if (correspondence.parent.schemaIdentity !=
            fabric::fabricArtifactSchema.identity ||
        correspondence.parent.schemaVersion !=
            fabric::fabricArtifactSchema.version ||
        correspondence.child.schemaIdentity !=
            fabric::fabricArtifactSchema.identity ||
        correspondence.child.schemaVersion !=
            fabric::fabricArtifactSchema.version)
      return invalid("Module correspondence has a foreign schema");
    if (!childModuleByParent
             .emplace(correspondence.parent.artifact.bytes(),
                      correspondence.child)
             .second ||
        !childModules.insert(correspondence.child.artifact.bytes()).second)
      return invalid("Module correspondence is not one-to-one");
    if (targetModuleIdentities.find(correspondence.child.artifact.bytes()) ==
        targetModuleIdentities.end())
      return invalid("Module correspondence child is not attached to System");
    auto parent =
        fabric::importEntireFabricRoot(correspondence.parent, artifacts);
    if (!parent)
      return parent.takeError();
    auto child =
        fabric::importEntireFabricRoot(correspondence.child, artifacts);
    if (!child)
      return child.takeError();
    if (parent->view().rootKind() != fabric::FabricRootKind::Module ||
        child->view().rootKind() != fabric::FabricRootKind::Module)
      return invalid("Module correspondence names a non-Module root");
  }

  const JointDesignPlanPair &pair = parentPlan.pairOutputs.front();
  const CompletedDsePlanExecution &available =
      availableExecution(parentExecution.planExecution);
  std::vector<ArtifactRootReference> parentTech = pair.immutableTechMappings;
  std::vector<ArtifactRootReference> parentSpatial =
      pair.immutableSpatialMappings;
  appendAvailableRoots(available, pair.techMappings, parentTech);
  appendAvailableRoots(available, pair.spatialMappings, parentSpatial);
  canonicalize(parentSpatial);
  for (const ArtifactRootReference &spatialReference : parentSpatial) {
    auto spatial = mapping::importSpatialMapping(spatialReference, artifacts);
    if (!spatial)
      return spatial.takeError();
    parentTech.push_back({mapping::mappingArtifactSchema.identity.str(),
         mapping::mappingArtifactSchema.version,
         spatial->view().techMappingIdentity()});
  }
  canonicalize(parentTech);

  result.accounting.parentTechMappings = parentTech.size();
  result.accounting.parentSpatialMappings = parentSpatial.size();
  result.accounting.invalidationRootCount =
      projectJointHardwareInvalidationRootCount(
          llvm::ArrayRef<HardwareImpactProjection>(impact));
  if (parentTech.empty() || parentSpatial.empty()) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::MissingParentFrontier, std::nullopt,
         "parent execution has no complete Tech/Spatial frontier"});
    return result;
  }

  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(pair.pair.software.dataflow, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();

  std::vector<TechRecord> techRecords;
  techRecords.reserve(parentTech.size());
  for (const ArtifactRootReference &parentReference : parentTech) {
    auto parent = mapping::importTechMapping(parentReference, artifacts);
    if (!parent)
      return parent.takeError();
    const std::uint64_t decisions = techDecisionCount(parent->view());
    result.accounting.parentTechDecisions += decisions;
    const bool impactedModule =
        impact.parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact.parent.artifact == parent->view().fabricIdentity();
    if (impactedModule &&
        impact.tech.kind == HardwareMappingImpactKind::Reopen) {
      result.accounting.reopenedTechDecisions += decisions;
      ++result.accounting.invalidatedTechMappings;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::TechImpactReopened, parentReference,
           "typed hardware delta reopens this TechMapping owner"});
      continue;
    }
    const auto module =
        childModuleByParent.find(parent->view().fabricIdentity().bytes());
    if (module == childModuleByParent.end()) {
      ++result.accounting.invalidatedTechMappings;
      result.accounting.reopenedTechDecisions += decisions;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::ModuleCorrespondence,
           parentReference,
           "TechMapping owner has no parent-to-child Module lineage"});
      continue;
    }
    auto childModule =
        fabric::importEntireFabricRoot(module->second, artifacts);
    if (!childModule)
      return childModule.takeError();
    auto child =
        mapping::rebaseTechMapping(*parent, childModule->view(), artifacts);
    if (!child) {
      ++result.accounting.invalidatedTechMappings;
      result.accounting.reopenedTechDecisions += decisions;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::TechRebaseRejected, parentReference,
           llvm::toString(child.takeError())});
      continue;
    }
    result.seed.techMappings.push_back(child->reference());
    techRecords.push_back({parent->view().identity(), std::move(*child)});
    const bool moduleRepair =
        impact.parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact.parent.artifact == parent->view().fabricIdentity();
    if (moduleRepair)
      ++result.accounting.repairedTechMappings;
    else
      ++result.accounting.preservedTechMappings;
    if (moduleRepair)
      result.accounting.repairedTechDecisions += decisions;
    else
      result.accounting.preservedTechDecisions += decisions;
  }

  for (const ArtifactRootReference &parentReference : parentSpatial) {
    auto parent = mapping::importSpatialMapping(parentReference, artifacts);
    if (!parent)
      return parent.takeError();
    const std::uint64_t decisions = spatialDecisionCount(parent->view());
    const std::uint64_t routes = spatialRouteNodeCount(parent->view());
    result.accounting.parentSpatialDecisions += decisions;
    result.accounting.parentRouteNodeCount += routes;
    const TechRecord *childTech =
        findTechRecord(techRecords, parent->view().techMappingIdentity());
    if (!childTech) {
      ++result.accounting.invalidatedSpatialMappings;
      result.accounting.reopenedSpatialDecisions += decisions;
      result.accounting.reopenedRouteNodeCount += routes;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::TechRebaseRejected, parentReference,
           "SpatialMapping depends on an invalidated TechMapping"});
      continue;
    }
    const auto module =
        childModuleByParent.find(parent->view().fabricIdentity().bytes());
    if (module == childModuleByParent.end()) {
      ++result.accounting.invalidatedSpatialMappings;
      result.accounting.reopenedSpatialDecisions += decisions;
      result.accounting.reopenedRouteNodeCount += routes;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::ModuleCorrespondence,
           parentReference,
           "SpatialMapping owner has no parent-to-child Module lineage"});
      continue;
    }
    auto childModule =
        fabric::importEntireFabricRoot(module->second, artifacts);
    if (!childModule)
      return childModule.takeError();
    const bool moduleRepair =
        impact.parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact.parent.artifact == parent->view().fabricIdentity();
    bool spatialImpactUsed = moduleRepair;
    if (moduleRepair && impact.spatial.routeRoots.empty()) {
      auto parentModule = fabric::importEntireFabricRoot(
          {fabric::fabricArtifactSchema.identity.str(),
           fabric::fabricArtifactSchema.version,
           parent->view().fabricIdentity()},
          artifacts);
      if (!parentModule)
        return parentModule.takeError();
      spatialImpactUsed = spatialMappingUsesImpact(
          parent->view(), parentModule->view(), impact.spatial.placementRoots);
    }
    auto constraints = mapping::finalizeEmptySpatialMappingConstraintSet(
        *dataflow, childTech->child.view(), childModule->view(), artifacts);
    if (!constraints)
      return constraints.takeError();
    auto child = mapping::rebaseSpatialMapping(
        *parent, childTech->child, childModule->view(), constraints->view(),
        artifacts, nullptr,
        moduleRepair
            ? llvm::ArrayRef<loom::fabric::FabricModuleEntityCorrespondence>(
                  impact.moduleEntities)
                     : llvm::ArrayRef<loom::fabric::FabricModuleEntityCorrespondence>());
    if (!child) {
      ++result.accounting.invalidatedSpatialMappings;
      result.accounting.reopenedSpatialDecisions += decisions;
      result.accounting.reopenedRouteNodeCount += routes;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::SpatialRebaseRejected,
           parentReference, llvm::toString(child.takeError())});
      continue;
    }
    result.seed.spatialMappings.push_back(child->reference());
    const bool mappingRepair = moduleRepair && spatialImpactUsed;
    if (mappingRepair)
      ++result.accounting.repairedSpatialMappings;
    else
      ++result.accounting.preservedSpatialMappings;
    if (mappingRepair) {
      result.accounting.repairedSpatialDecisions += decisions;
      result.accounting.repairedRouteNodeCount += routes;
    } else {
      result.accounting.preservedSpatialDecisions += decisions;
      result.accounting.preservedRouteNodeCount += routes;
    }
  }
  for (const ArtifactRootReference &parentReference : parentMappings) {
    auto mapping = mapping::importSystemMapping(parentReference, artifacts);
    if (!mapping)
      return mapping.takeError();
    const auto &systemView = mapping->view();
    const std::uint64_t threadBindings =
        systemView.executionBindings().threadBindings().size();
    const std::uint64_t graphBindings =
        systemView.executionBindings().graphBindings().size();
    const std::uint64_t resourceUses = systemView.resourceUses().size();
    const std::uint64_t serviceRealizations =
        systemView.serviceRealizations().size();
    result.accounting.parentThreadBindingCount += threadBindings;
    result.accounting.parentGraphBindingCount += graphBindings;
    result.accounting.parentResourceUseCount += resourceUses;
    result.accounting.parentServiceRealizationCount += serviceRealizations;
    const std::vector<::dataflow::RootThreadLaunchRef> reopenedRoots =
        projectJointSystemReopenRoots(mapping->view(), impacts);
    std::set<std::vector<std::uint8_t>> reopenedServices;
    if (!reopenedRoots.empty()) {
      auto obligations =
          mapping::projectSystemServiceObligations(*dataflow, reopenedRoots);
      if (!obligations)
        return obligations.takeError();
      for (const mapping::SystemServiceObligationProjection &obligation :
           *obligations) {
        auto encoded = mapping::encodeSystemServiceObligationKey(
            dataflow->identity(), obligation.key);
        if (!encoded)
          return encoded.takeError();
        reopenedServices.insert(std::move(*encoded));
      }
    }
    const auto serviceIsReopened =
        [&](const mapping::SystemServiceObligationKey &service)
        -> llvm::Expected<bool> {
      auto encoded = mapping::encodeSystemServiceObligationKey(
          dataflow->identity(), service);
      if (!encoded)
        return encoded.takeError();
      return reopenedServices.find(*encoded) != reopenedServices.end();
    };
    const std::uint64_t reopenedThreadBindings = llvm::count_if(
        systemView.executionBindings().threadBindings(), [&](const auto &item) {
          return llvm::is_contained(reopenedRoots, item.key);
        });
    const std::uint64_t reopenedGraphBindings = llvm::count_if(
        systemView.executionBindings().graphBindings(), [&](const auto &item) {
          return llvm::is_contained(reopenedRoots, item.key.rootThreadLaunch);
        });
    result.accounting.reopenedThreadBindingCount += reopenedThreadBindings;
    result.accounting.preservedThreadBindingCount +=
        threadBindings - reopenedThreadBindings;
    result.accounting.reopenedGraphBindingCount += reopenedGraphBindings;
    result.accounting.preservedGraphBindingCount +=
        graphBindings - reopenedGraphBindings;
    std::uint64_t reopenedResourceUses = 0;
    for (const mapping::SystemResourceUseView &use :
         systemView.resourceUses()) {
      bool reopened = false;
      if (const auto *instruction =
              std::get_if<mapping::SystemInstructionResourceOwnerView>(
                  &use.owner))
        reopened = llvm::is_contained(reopenedRoots, instruction->root);
      else {
        const auto &service =
            std::get<mapping::SystemServicePlanResourceOwnerView>(use.owner);
        auto affected = serviceIsReopened(service.service);
        if (!affected)
          return affected.takeError();
        reopened = *affected;
      }
      reopenedResourceUses += reopened;
    }
    result.accounting.reopenedResourceUseCount += reopenedResourceUses;
    result.accounting.preservedResourceUseCount +=
        resourceUses - reopenedResourceUses;

    std::uint64_t reopenedRealizations = 0;
    std::uint64_t reopenedLegs = 0;
    for (const mapping::SystemServiceRealizationView &service :
         systemView.serviceRealizations()) {
      auto affected = serviceIsReopened(service.key);
      if (!affected)
        return affected.takeError();
      if (*affected) {
        ++reopenedRealizations;
        reopenedLegs += systemServiceLegCount(service);
      }
    }
    result.accounting.reopenedServiceRealizationCount += reopenedRealizations;
    result.accounting.preservedServiceRealizationCount +=
        serviceRealizations - reopenedRealizations;
    const std::uint64_t legs = systemServiceLegCount(systemView);
    result.accounting.parentServiceLegCount += legs;
    result.accounting.reopenedServiceLegCount += reopenedLegs;
    result.accounting.preservedServiceLegCount += legs - reopenedLegs;
  }
  result.accounting.invalidationConeDecisionCount =
      result.accounting.reopenedTechDecisions +
      result.accounting.reopenedSpatialDecisions +
      result.accounting.reopenedRouteNodeCount +
      result.accounting.repairedTechDecisions +
      result.accounting.repairedSpatialDecisions +
      result.accounting.repairedRouteNodeCount +
      result.accounting.reopenedThreadBindingCount +
      result.accounting.reopenedGraphBindingCount +
      result.accounting.reopenedResourceUseCount +
      result.accounting.reopenedServiceRealizationCount +
      result.accounting.reopenedServiceLegCount;
  canonicalize(result.seed.techMappings);
  canonicalize(result.seed.spatialMappings);
  if (result.accounting.invalidatedTechMappings == 0 &&
      result.accounting.invalidatedSpatialMappings == 0)
    result.disposition = JointMappingReuseDisposition::Preserved;
  else if (result.accounting.invalidatedSpatialMappings ==
               result.accounting.parentSpatialMappings &&
           impact.spatial.kind != HardwareMappingImpactKind::Reopen)
    // A lower-layer seed is not enough to call a Spatial repair incremental.
    // If the affected Spatial cone has no proven child correspondence, the
    // caller must report a cold fallback even when an unaffected Tech seed is
    // still useful to the child generator.
    result.disposition = JointMappingReuseDisposition::ColdFallback;
  else if (result.accounting.invalidatedTechMappings ==
               result.accounting.parentTechMappings &&
           impact.tech.kind != HardwareMappingImpactKind::Reopen)
    result.disposition = JointMappingReuseDisposition::ColdFallback;
  else if (!result.seed.techMappings.empty() ||
           !result.seed.spatialMappings.empty())
    result.disposition = JointMappingReuseDisposition::LocalRepair;
  if (llvm::Error error =
          validateJointMappingRebaseAccounting(result.accounting))
    return std::move(error);
  return result;
}

} // namespace loom::dse
