#include "DSE/JointMappingMigration.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "DSE/HardwareDecision.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <system_error>

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
  if (const auto *completed =
          std::get_if<CompletedDsePlanExecution>(&outcome))
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
         mapping.memoryEngineBindings().size() + mapping.memoryBindings().size() +
         mapping.registerFifoTransfers().size() + mapping.routeTrees().size() +
         mapping.resourceUses().size();
}

std::uint64_t spatialRouteNodeCount(const mapping::SpatialMappingView &mapping) {
  std::uint64_t count = 0;
  for (const auto &route : mapping.routeTrees())
    count += route.nodes.size() + route.sinks.size();
  return count;
}

std::uint64_t techDecisionCount(const mapping::TechMappingView &mapping) {
  return mapping.computeRealizations().size() +
         mapping.memoryRealizations().size();
}

std::uint64_t systemServiceLegCount(
    const mapping::SystemMappingView &mapping) {
  std::uint64_t count = 0;
  for (const auto &service : mapping.serviceRealizations())
    for (const auto &plan : service.plans)
      count += plan.transferLegs.size();
  return count;
}

const TechRecord *findTechRecord(llvm::ArrayRef<TechRecord> records,
                                 const ArtifactIdentity &parent) {
  const auto found = llvm::find_if(records, [&](const TechRecord &record) {
    return record.parentIdentity == parent;
  });
  return found == records.end() ? nullptr : &*found;
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

llvm::StringRef jointMappingReuseDispositionSpelling(
    JointMappingReuseDisposition disposition) {
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

llvm::Expected<std::vector<ArtifactRootReference>>
resolveJointSpatialMappingFrontier(const JointDesignExplorationPlan &plan,
                                   const JointDesignExecution &execution) {
  if (plan.pairOutputs.size() != 1)
    return invalid("Spatial frontier reuse requires one exact Mapping pair");
  const JointDesignPlanPair &pair = plan.pairOutputs.front();
  std::vector<ArtifactRootReference> mappings =
      pair.immutableSpatialMappings;
  appendAvailableRoots(availableExecution(execution.planExecution),
                       pair.spatialMappings, mappings);
  canonicalize(mappings);
  if (mappings.empty())
    return invalid("Mapping execution has no reusable Spatial frontier");
  return mappings;
}

llvm::Expected<JointMappingRebaseResult> rebaseJointMappingFrontier(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const ArtifactRootReference &childSystem,
    llvm::ArrayRef<pnr::SystemModuleCorrespondence> moduleCorrespondences,
    const HardwareImpactProjection *impact,
    const ArtifactStore &artifacts) {
  JointMappingRebaseResult result;
  if (!impact) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::MissingImpactProjection,
         std::nullopt,
         "Mapping rebase requires a typed hardware impact projection"});
    return result;
  }
  if (impact->child && *impact->child != childSystem)
    return invalid("hardware impact projection names a different child System");
  if (impact->locality == HardwareMutationLocality::GlobalReopen) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ImpactRequiresColdFallback,
         std::nullopt,
         "typed hardware impact requires global Mapping repair"});
    return result;
  }
  if (impact->tech.kind == HardwareMappingImpactKind::Reopen &&
      impact->tech.realizationRoots.empty()) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ImpactRequiresColdFallback,
         std::nullopt,
         "typed Tech impact has no realization root"});
    return result;
  }
  if (impact->spatial.kind == HardwareMappingImpactKind::Reopen &&
      impact->spatial.placementRoots.empty() &&
      impact->spatial.routeRoots.empty()) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::ImpactRequiresColdFallback,
         std::nullopt,
         "typed Spatial impact has no placement or route root"});
    return result;
  }
  if (parentPlan.pairOutputs.size() != 1)
    return invalid("Mapping rebase requires one exact parent pair");
  if (moduleCorrespondences.empty())
    return JointMappingRebaseResult{
        {}, {}, {{JointMappingRebaseFailureReason::ModuleCorrespondence,
                  std::nullopt,
                  "hardware child published no Module correspondence"}}};

  auto targetModules =
      projectJointDesignTargetModules(childSystem, artifacts);
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
    auto child = fabric::importEntireFabricRoot(correspondence.child, artifacts);
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
    parentTech.push_back(
        {mapping::mappingArtifactSchema.identity.str(),
         mapping::mappingArtifactSchema.version,
         spatial->view().techMappingIdentity()});
  }
  canonicalize(parentTech);

  result.accounting.parentTechMappings = parentTech.size();
  result.accounting.parentSpatialMappings = parentSpatial.size();
  result.accounting.invalidationRootCount =
      impact->tech.realizationRoots.size() +
      impact->spatial.placementRoots.size() + impact->spatial.routeRoots.size() +
      impact->system.executionRoots.size();
  if (parentTech.empty() || parentSpatial.empty()) {
    result.failures.push_back(
        {JointMappingRebaseFailureReason::MissingParentFrontier, std::nullopt,
         "parent execution has no complete Tech/Spatial frontier"});
    return result;
  }

  auto dataflowArtifact = dataflow::importCanonicalDataflow(
      pair.pair.software.dataflow, artifacts);
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
        impact->parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact->parent.artifact == parent->view().fabricIdentity();
    if (impactedModule &&
        impact->tech.kind == HardwareMappingImpactKind::Reopen) {
      result.accounting.reopenedTechDecisions += decisions;
      ++result.accounting.invalidatedTechMappings;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::TechImpactReopened,
           parentReference,
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
          {JointMappingRebaseFailureReason::TechRebaseRejected,
           parentReference, llvm::toString(child.takeError())});
      continue;
    }
    result.seed.techMappings.push_back(child->reference());
    techRecords.push_back(
        {parent->view().identity(), std::move(*child)});
    const bool moduleRepair =
        impact->parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact->parent.artifact == parent->view().fabricIdentity();
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
    const bool impactedModule =
        impact->parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact->parent.artifact == parent->view().fabricIdentity();
    if (impactedModule &&
        impact->spatial.kind == HardwareMappingImpactKind::Reopen) {
      result.accounting.reopenedSpatialDecisions += decisions;
      result.accounting.reopenedRouteNodeCount += routes;
      ++result.accounting.invalidatedSpatialMappings;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::SpatialImpactReopened,
           parentReference,
           "typed hardware delta reopens this SpatialMapping owner"});
      continue;
    }
    const TechRecord *childTech =
        findTechRecord(techRecords, parent->view().techMappingIdentity());
    if (!childTech) {
      ++result.accounting.invalidatedSpatialMappings;
      result.accounting.reopenedSpatialDecisions += decisions;
      result.accounting.reopenedRouteNodeCount += routes;
      result.failures.push_back(
          {JointMappingRebaseFailureReason::TechRebaseRejected,
           parentReference,
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
    auto constraints = mapping::finalizeEmptySpatialMappingConstraintSet(
        *dataflow, childTech->child.view(), childModule->view(), artifacts);
    if (!constraints)
      return constraints.takeError();
    const bool moduleRepair =
        impact->parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        impact->parent.artifact == parent->view().fabricIdentity();
    auto child = mapping::rebaseSpatialMapping(
        *parent, childTech->child, childModule->view(), constraints->view(),
        artifacts, nullptr,
        moduleRepair ? llvm::ArrayRef<loom::fabric::FabricModuleEntityCorrespondence>(
                            impact->moduleEntities)
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
    if (moduleRepair)
      ++result.accounting.repairedSpatialMappings;
    else
      ++result.accounting.preservedSpatialMappings;
    if (moduleRepair) {
      result.accounting.repairedSpatialDecisions += decisions;
      result.accounting.repairedRouteNodeCount += routes;
    } else {
      result.accounting.preservedSpatialDecisions += decisions;
      result.accounting.preservedRouteNodeCount += routes;
    }
  }
  for (const ArtifactRootReference &parentReference : mappingRoots(parentExecution)) {
    auto mapping = mapping::importSystemMapping(parentReference, artifacts);
    if (!mapping)
      return mapping.takeError();
    const std::uint64_t legs = systemServiceLegCount(mapping->view());
    result.accounting.parentServiceLegCount += legs;
    if (impact->system.kind == HardwareMappingImpactKind::Reopen)
      result.accounting.reopenedServiceLegCount += legs;
    else
      result.accounting.preservedServiceLegCount += legs;
  }
  result.accounting.invalidationConeDecisionCount =
      result.accounting.reopenedTechDecisions +
      result.accounting.reopenedSpatialDecisions +
      result.accounting.reopenedRouteNodeCount;
  canonicalize(result.seed.techMappings);
  canonicalize(result.seed.spatialMappings);
  if (result.accounting.invalidatedTechMappings == 0 &&
      result.accounting.invalidatedSpatialMappings == 0)
    result.disposition = JointMappingReuseDisposition::Preserved;
  else if (result.accounting.invalidatedSpatialMappings ==
               result.accounting.parentSpatialMappings &&
           impact->spatial.kind != HardwareMappingImpactKind::Reopen)
    // A lower-layer seed is not enough to call a Spatial repair incremental.
    // If the affected Spatial cone has no proven child correspondence, the
    // caller must report a cold fallback even when an unaffected Tech seed is
    // still useful to the child generator.
    result.disposition = JointMappingReuseDisposition::ColdFallback;
  else if (result.accounting.invalidatedTechMappings ==
               result.accounting.parentTechMappings &&
           impact->tech.kind != HardwareMappingImpactKind::Reopen)
    result.disposition = JointMappingReuseDisposition::ColdFallback;
  else if (!result.seed.techMappings.empty() ||
           !result.seed.spatialMappings.empty())
    result.disposition = JointMappingReuseDisposition::LocalRepair;
  return result;
}

} // namespace loom::dse
