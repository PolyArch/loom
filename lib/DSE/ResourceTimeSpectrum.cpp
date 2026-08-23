#include "DSE/ResourceTimeSpectrum.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <system_error>
#include <utility>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resource_time_spectrum_invalid: " + message);
}

bool referenceLess(const ArtifactRootReference &lhs,
                   const ArtifactRootReference &rhs) {
  return artifactRootReferenceLess(lhs, rhs);
}

bool rootLess(::dataflow::RootThreadLaunchRef lhs,
              ::dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

std::string
physicalKey(const ::loom::fabric::FabricPhysicalOccurrenceOwnerRef &resource) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(resource);
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

void canonicalizeResources(
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> &resources) {
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return physicalKey(lhs) < physicalKey(rhs);
  });
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
}

llvm::Expected<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>
physicalAccCore(::loom::fabric::AccCoreOccurrenceRef core) {
  return ::loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
      ::loom::fabric::FabricInventoryOwnerRef::of(core));
}

struct ImportedMappingProjection final {
  ArtifactRootReference reference;
  ArtifactRootReference dataflow;
  ArtifactRootReference fabric;
  std::map<std::uint64_t,
           std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>>
      resourcesByRoot;
  ::loom::mapping::MappingProgressClosure resourceTimeProgress;
};

llvm::Expected<ImportedMappingProjection>
importProjection(const ArtifactRootReference &reference,
                 llvm::ArrayRef<ResourceTimeRegionMapping> regions,
                 const ArtifactStore &store) {
  auto mapping = ::loom::mapping::importSystemMapping(reference, store);
  if (!mapping)
    return mapping.takeError();
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      mapping->view().dataflowIdentity()};
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      mapping->view().fabricIdentity()};
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto contexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, mapping->view().executionBindings());
  if (!contexts)
    return contexts.takeError();

  auto fabricArtifact =
      ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();
  auto resourceTimeProgress =
      ::loom::mapping::qualifySystemMappingResourceTimeProgress(
          *mapping, *dataflow, *system);
  if (!resourceTimeProgress)
    return resourceTimeProgress.takeError();
  ImportedMappingProjection result{reference, dataflowReference,
                                   fabricReference, {},
                                   std::move(*resourceTimeProgress)};
  for (const ResourceTimeRegionMapping &region : regions) {
    if (region.root.artifact != dataflowReference.artifact)
      return invalid("region correspondence has a foreign Dataflow root");
    auto resolved = dataflow->resolve(region.root);
    if (!resolved)
      return resolved.takeError();
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
    for (const auto &domain : contexts->instructionDomains) {
      if (domain.root != region.root)
        continue;
      auto physical = physicalAccCore(domain.context.accCore);
      if (!physical)
        return physical.takeError();
      resources.push_back(std::move(*physical));
    }
    for (const auto &domain : contexts->spatialDomains) {
      if (domain.graph.rootThreadLaunch != region.root)
        continue;
      auto physical = physicalAccCore(domain.context.accCore);
      if (!physical)
        return physical.takeError();
      resources.push_back(std::move(*physical));
    }
    canonicalizeResources(resources);
    if (resources.empty())
      return invalid("verified SystemMapping has no AccCore allocation for a "
                     "resource-time region");
    result.resourcesByRoot.emplace(region.root.entity.value(),
                                   std::move(resources));
  }
  return result;
}

IncompleteResourceTimeSpectrum
incomplete(ResourceTimeSpectrumIncompleteReason reason, llvm::Twine diagnostic,
           std::uint64_t imported) {
  return {reason, diagnostic.str(), imported};
}

const ResourceTimeHintAllocation *
findHintAllocation(const ResourceTimeHintState &state,
                   ::dataflow::RootThreadLaunchRef root) {
  const auto found = llvm::find_if(state.active, [&](const auto &allocation) {
    return allocation.region == root;
  });
  return found == state.active.end() ? nullptr : &*found;
}

bool mappingMatchesState(const ImportedMappingProjection &mapping,
                         const ResourceTimeHintState &state,
                         ResourceTimeSpectrumFunnelAccounting &accounting) {
  ++accounting.matchingMappingChecks;
  std::set<std::string> used;
  for (const ResourceTimeHintAllocation &allocation : state.active) {
    if (allocation.resourceUnits.size() != 1)
      return false;
    const auto found =
        mapping.resourcesByRoot.find(allocation.region.entity.value());
    if (found == mapping.resourcesByRoot.end() ||
        found->second.size() != allocation.resourceUnits.front())
      return false;
    for (const auto &resource : found->second)
      if (!used.insert(physicalKey(resource)).second)
        return false;
  }
  return true;
}

bool mappingMatchesHint(const ImportedMappingProjection &mapping,
                        const ResourceTimeScheduleHint &hint,
                        ResourceTimeSpectrumFunnelAccounting &accounting) {
  return llvm::all_of(hint.states, [&](const auto &state) {
    return mappingMatchesState(mapping, state, accounting);
  });
}

llvm::Expected<std::optional<::loom::pnr::ResourceTimeScheduleScenario>>
materializeHint(const ResourceTimeScheduleHint &hint,
                llvm::ArrayRef<ResourceTimeRegionFeature> regions,
                llvm::ArrayRef<ImportedMappingProjection> mappings,
                ResourceTimeSpectrumFunnelAccounting &accounting) {
  if (hint.states.size() != hint.actions.size() + 1)
    return invalid("resource-time hint action/state lineage is incomplete");
  const ImportedMappingProjection *selected = nullptr;
  for (const ImportedMappingProjection &mapping : mappings)
    if (mappingMatchesHint(mapping, hint, accounting)) {
      selected = &mapping;
      break;
    }
  if (!selected)
    return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};

  ::loom::pnr::ResourceTimeScheduleScenario scenario;
  scenario.makespanPicoseconds = hint.estimatedMakespanPicoseconds;
  scenario.executions.reserve(regions.size());
  for (const ResourceTimeRegionFeature &region : regions) {
    std::optional<std::uint64_t> start;
    std::optional<std::uint64_t> completion;
    for (std::size_t action = 0; action != hint.actions.size(); ++action) {
      const ResourceTimeActionDelta &delta = hint.actions[action];
      if (delta.kind != ResourceTimeActionKind::AdmitRegion ||
          delta.admittedRegion != region.region)
        continue;
      if (start)
        return invalid("resource-time hint admits one region more than once");
      start = delta.beforeTimePicoseconds;
      const ResourceTimeHintAllocation *allocation =
          findHintAllocation(hint.states[action + 1], region.region);
      if (!allocation)
        return invalid("resource-time admission has no active allocation");
      completion = allocation->completionTimePicoseconds;
    }
    if (!start || !completion)
      return invalid("resource-time hint omits a covered region execution");
    scenario.executions.push_back({region.region, {}, 0, *start, *completion});
  }
  for (std::size_t ordinal = 0; ordinal != regions.size(); ++ordinal) {
    const ResourceTimeRegionFeature &feature = regions[ordinal];
    auto &execution = scenario.executions[ordinal];
    std::uint64_t ready = 0;
    for (const ResourceTimeDependencyFeature &dependency :
         feature.dependencies) {
      const auto producer =
          llvm::find_if(scenario.executions, [&](const auto &candidate) {
            return candidate.region == dependency.producer;
          });
      if (producer == scenario.executions.end())
        return invalid("resource-time hint dependency has no execution");
      std::uint64_t release = producer->completionPicoseconds;
      if (dependency.readiness ==
          ::loom::pnr::ResourceTimeReadinessKind::FifoToken) {
        std::optional<std::uint64_t> tokenTime;
        for (const ResourceTimeActionDelta &action : hint.actions)
          if (llvm::is_contained(action.tokenReadyProducers,
                                 dependency.producer)) {
            tokenTime = action.afterTimePicoseconds;
            break;
          }
        if (!tokenTime)
          return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
        release = *tokenTime;
      }
      ready = std::max(ready, release);
      execution.prerequisites.push_back(
          {dependency.producer, dependency.readiness});
    }
    if (execution.startPicoseconds < ready)
      return invalid("resource-time hint starts a region before readiness");
    execution.readyPicoseconds = ready;
  }

  for (std::size_t snapshot = 1; snapshot != hint.states.size(); ++snapshot) {
    const ResourceTimeActionDelta &action = hint.actions[snapshot - 1];
    if (action.kind != ResourceTimeActionKind::AdmitRegion &&
        action.completedRegions.empty())
      continue;
    const ResourceTimeHintState &state = hint.states[snapshot];
    ::dataflow::EventFamilyKey event =
        action.kind == ResourceTimeActionKind::AdmitRegion
            ? ::dataflow::rootThreadStartEventFamily(*action.admittedRegion)
            : ::dataflow::rootThreadCompletionEventFamily(
                  action.completedRegions.front());
    std::vector<::loom::pnr::ResourceTimeRegionAllocation> active;
    active.reserve(state.active.size());
    for (const ResourceTimeHintAllocation &allocation : state.active) {
      const auto resources =
          selected->resourcesByRoot.find(allocation.region.entity.value());
      if (resources == selected->resourcesByRoot.end())
        return invalid("matched Mapping lost a resource-time region");
      active.push_back({allocation.region, resources->second});
    }
    ::loom::pnr::ResourceTimeScheduleState materialized{
        selected->reference, std::move(event), state.timePicoseconds,
        std::move(active)};
    if (!scenario.states.empty() &&
        scenario.states.back().timePicoseconds == state.timePicoseconds)
      scenario.states.back() = std::move(materialized);
    else
      scenario.states.push_back(std::move(materialized));
  }
  if (scenario.states.empty())
    return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
  return std::optional<::loom::pnr::ResourceTimeScheduleScenario>(
      std::move(scenario));
}

bool hasUnrepresentableCompositeEvent(const ResourceTimeScheduleHint &hint) {
  return llvm::any_of(hint.actions, [](const ResourceTimeActionDelta &action) {
    return action.kind == ResourceTimeActionKind::AdvanceEvent &&
           ((!action.completedRegions.empty() &&
             action.completedRegions.size() > 1) ||
            (action.completedRegions.empty() &&
             !action.tokenReadyProducers.empty()));
  });
}

} // namespace

llvm::Expected<ResourceTimeSpectrumVerification> verifyResourceTimeSpectrum(
    const ::loom::pnr::ResourceTimeScheduleWitness &witness,
    llvm::ArrayRef<ResourceTimeRegionMapping> regions,
    const ArtifactStore &store, ExecutionControlView executionControl) {
  if (llvm::Error error =
          ::loom::pnr::validateResourceTimeScheduleWitness(witness))
    return std::move(error);
  if (regions.size() != witness.regions.size())
    return invalid("region correspondence does not cover the witness");

  std::map<::dataflow::RootThreadLaunchRef, const ResourceTimeRegionMapping *,
           decltype(&rootLess)>
      regionByIdentity(&rootLess);
  std::optional<ArtifactIdentity> dataflowIdentity;
  for (const ResourceTimeRegionMapping &region : regions) {
    if (!llvm::is_contained(witness.regions, region.root) ||
        !regionByIdentity.emplace(region.root, &region).second)
      return invalid("region correspondence is foreign or duplicated");
    if (region.minimumFeasibleAccCoreCount == 0 ||
        region.minimumFeasibleAccCoreCount > region.maximumUsefulAccCoreCount)
      return invalid("region allocation bounds are invalid");
    if (dataflowIdentity && *dataflowIdentity != region.root.artifact)
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::Unsupported,
          "resource-time regions span multiple Dataflow identities without a "
          "typed transformation correspondence",
          0)};
    dataflowIdentity = region.root.artifact;
  }

  std::vector<ArtifactRootReference> mappingReferences;
  for (const auto &scenario : witness.scenarios)
    for (const auto &state : scenario.states)
      mappingReferences.push_back(state.mapping);
  llvm::sort(mappingReferences, referenceLess);
  mappingReferences.erase(
      std::unique(mappingReferences.begin(), mappingReferences.end()),
      mappingReferences.end());

  ::loom::mapping::SystemMappingImportSession importSession(
      store, std::max<std::size_t>(1, mappingReferences.size()),
      ::loom::mapping::SystemMappingImportSessionMode::ReuseEnclosing);
  std::map<ArtifactRootReference, ImportedMappingProjection,
           decltype(&artifactRootReferenceLess)>
      imported(&artifactRootReferenceLess);
  std::optional<ArtifactRootReference> fabric;
  for (const ArtifactRootReference &reference : mappingReferences) {
    if (executionControl.stopRequested())
      return ResourceTimeSpectrumVerification{
          incomplete(ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout,
                     "resource-time Mapping import was cancelled or timed out",
                     imported.size())};
    auto projection = importProjection(reference, regions, store);
    if (!projection)
      return projection.takeError();
    if (projection->dataflow.artifact != *dataflowIdentity)
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::Unsupported,
          "resource-time Mapping changes the Dataflow without a typed state "
          "correspondence",
          imported.size() + 1)};
    if (fabric && *fabric != projection->fabric)
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::Unsupported,
          "runtime resource-time transition changes the immutable Fabric",
          imported.size() + 1)};
    fabric = projection->fabric;
    imported.emplace(reference, std::move(*projection));
  }
  if (!fabric)
    return invalid("resource-time witness names no Mapping");

  const ArtifactRootReference verifiedDataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto verifiedDataflowArtifact =
      ::dataflow::importCanonicalDataflow(verifiedDataflowReference, store);
  if (!verifiedDataflowArtifact)
    return verifiedDataflowArtifact.takeError();
  auto verifiedDataflow = verifiedDataflowArtifact->view();
  if (!verifiedDataflow)
    return verifiedDataflow.takeError();

  auto fabricArtifact = ::loom::fabric::importEntireFabricRoot(*fabric, store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();

  VerifiedResourceTimeSpectrum verified{verifiedDataflowReference, *fabric, {}};
  verified.scenarios.reserve(witness.scenarios.size());
  for (auto indexedScenario : llvm::enumerate(witness.scenarios)) {
    if (executionControl.stopRequested())
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout,
          "resource-time spectrum verification was cancelled or timed out",
          imported.size())};
    const auto &scenario = indexedScenario.value();
    bool everyAllocationAtMinimum = true;
    bool everyAllocationAtMaximum = true;
    bool everyMinimumBoundExact = true;
    bool everyMaximumBoundExact = true;
    bool everyTemporalEpochSingle = true;
    std::uint64_t peakConcurrentRegions = 0;
    std::set<::dataflow::RootThreadLaunchRef, decltype(&rootLess)>
        observedRegions(&rootLess);
    std::vector<ArtifactRootReference> scenarioMappings;
    if (scenario.executions.size() != witness.regions.size())
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
          "resource-time scenario does not execute every covered region",
          imported.size())};
    for (const auto &execution : scenario.executions)
      observedRegions.insert(execution.region);
    if (observedRegions.size() != witness.regions.size())
      return ResourceTimeSpectrumVerification{
          incomplete(ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
                     "resource-time scenario execution coverage is incomplete",
                     imported.size())};
    observedRegions.clear();
    for (const auto &state : scenario.states) {
      const auto mapping = imported.find(state.mapping);
      if (mapping == imported.end())
        return invalid("resource-time state lost its imported Mapping");
      if (mapping->second.resourceTimeProgress.kind !=
          ::loom::mapping::MappingProgressClosureKind::
              ProvenNoClosedWaitSet)
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time Mapping progress is not established: " +
                ::loom::mapping::mappingProgressClosureReasonSpelling(
                    mapping->second.resourceTimeProgress.reason),
            imported.size())};
      if (llvm::Error error = verifiedDataflow->validate(state.event))
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time state event is not owned by its verified "
            "Mapping Dataflow",
            imported.size())};
      scenarioMappings.push_back(state.mapping);
      peakConcurrentRegions =
          std::max<std::uint64_t>(peakConcurrentRegions, state.active.size());
      for (const auto &allocation : state.active) {
        const auto correspondence = regionByIdentity.find(allocation.region);
        if (correspondence == regionByIdentity.end())
          return invalid("resource-time allocation has no region "
                         "correspondence");
        const ResourceTimeRegionMapping &region = *correspondence->second;
        everyTemporalEpochSingle &= region.logicalEpochCount == 1;
        observedRegions.insert(allocation.region);
        std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> observed =
            allocation.resources;
        for (const auto &resource : observed) {
          auto resolved = system->resolvePhysicalOwner(resource);
          if (!resolved)
            return ResourceTimeSpectrumVerification{incomplete(
                ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
                "resource-time allocation names a resource outside the "
                "verified Mapping Fabric",
                imported.size())};
        }
        canonicalizeResources(observed);
        const auto expected =
            mapping->second.resourcesByRoot.find(region.root.entity.value());
        if (expected == mapping->second.resourcesByRoot.end() ||
            observed != expected->second)
          return ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
              "resource-time allocation disagrees with independently imported "
              "SystemMapping execution bindings",
              imported.size())};
        const std::uint64_t count = observed.size();
        if (count < region.minimumFeasibleAccCoreCount ||
            count > region.maximumUsefulAccCoreCount)
          return ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
              "verified Mapping allocation is outside the region resource "
              "bounds",
              imported.size())};
        everyAllocationAtMinimum &= count == region.minimumFeasibleAccCoreCount;
        everyAllocationAtMaximum &= count == region.maximumUsefulAccCoreCount;
        everyMinimumBoundExact &=
            region.minimumBoundSupport == ResourceTimeEstimateSupport::Exact;
        everyMaximumBoundExact &=
            region.maximumBoundSupport == ResourceTimeEstimateSupport::Exact;
      }
    }
    if (observedRegions.size() != witness.regions.size())
      return ResourceTimeSpectrumVerification{
          incomplete(ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
                     "resource-time states never expose every covered region",
                     imported.size())};
    llvm::sort(scenarioMappings, referenceLess);
    scenarioMappings.erase(
        std::unique(scenarioMappings.begin(), scenarioMappings.end()),
        scenarioMappings.end());

    std::vector<std::pair<ArtifactRootReference, ArtifactRootReference>>
        mappingChanges;
    for (auto adjacent :
         llvm::zip(scenario.states, llvm::drop_begin(scenario.states))) {
      const auto &before = std::get<0>(adjacent);
      const auto &after = std::get<1>(adjacent);
      if (before.mapping != after.mapping)
        mappingChanges.emplace_back(before.mapping, after.mapping);
    }
    if (mappingChanges.size() != scenario.transitions.transitions.size())
      return ResourceTimeSpectrumVerification{incomplete(
          ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
          "resource-time Mapping changes do not match the transition chain",
          imported.size())};
    for (auto paired :
         llvm::zip(mappingChanges, scenario.transitions.transitions)) {
      const auto &change = std::get<0>(paired);
      const auto &transition = std::get<1>(paired);
      if (transition.beforeMapping != change.first ||
          transition.afterMapping != change.second)
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time transition does not match its adjacent states",
            imported.size())};
      switch (transition.status) {
      case ::loom::pnr::ResourceTimeTransitionStatus::Verified:
        break;
      case ::loom::pnr::ResourceTimeTransitionStatus::Unsupported:
        return ResourceTimeSpectrumVerification{
            incomplete(ResourceTimeSpectrumIncompleteReason::Unsupported,
                       "resource-time Mapping transition is unsupported",
                       imported.size())};
      case ::loom::pnr::ResourceTimeTransitionStatus::ProofNotEstablished:
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time Mapping transition is not proven", imported.size())};
      case ::loom::pnr::ResourceTimeTransitionStatus::CancelledOrTimeout:
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout,
            "resource-time Mapping transition was cancelled or timed out",
            imported.size())};
      }
    }

    PreMappingSpectrumClass spectrumClass =
        PreMappingSpectrumClass::Intermediate;
    const bool exactConcurrencyBounds =
        witness.concurrencyBoundStatus ==
        ::loom::pnr::ResourceTimeConcurrencyBoundStatus::Exact;
    if (witness.regions.size() > 1 && exactConcurrencyBounds &&
             everyMinimumBoundExact &&
             everyAllocationAtMinimum &&
             peakConcurrentRegions == witness.maximumConcurrentRegions)
      spectrumClass = PreMappingSpectrumClass::MaxSpatial;
    else if (witness.regions.size() > 1 && exactConcurrencyBounds &&
             everyTemporalEpochSingle &&
             everyMaximumBoundExact &&
             everyAllocationAtMaximum &&
             peakConcurrentRegions == witness.minimumConcurrentRegions)
      spectrumClass = PreMappingSpectrumClass::MaxTemporal;
    verified.scenarios.push_back(
        {indexedScenario.index(), spectrumClass, peakConcurrentRegions,
         scenario.makespanPicoseconds, std::move(scenarioMappings)});
  }
  return ResourceTimeSpectrumVerification{std::move(verified)};
}

llvm::Expected<ResourceTimeSpectrumFunnelResult>
verifyResourceTimeMappingFinalists(
    llvm::ArrayRef<ResourceTimeScheduleHint> hints,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    llvm::ArrayRef<ResourceTimeRegionResourceBound> bounds,
    llvm::ArrayRef<ArtifactRootReference> systemMappings,
    const ArtifactStore &store, ExecutionControlView executionControl,
    std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds) {
  const auto begin = std::chrono::steady_clock::now();
  ResourceTimeSpectrumFunnelAccounting accounting;
  accounting.hintCandidates = hints.size();
  if (hints.empty() || regions.empty() || bounds.size() != regions.size() ||
      systemMappings.empty())
    return invalid("resource-time spectrum funnel inputs must be nonempty and "
                   "aligned");
  for (std::size_t ordinal = 0; ordinal != regions.size(); ++ordinal)
    if (bounds[ordinal].region != regions[ordinal].region)
      return invalid("resource-time region bounds are not in canonical region "
                     "order");
  std::vector<ResourceTimeRegionMapping> correspondence;
  correspondence.reserve(regions.size());
  for (const ResourceTimeRegionFeature &region : regions) {
    const auto bound = llvm::find_if(bounds, [&](const auto &candidate) {
      return candidate.region == region.region;
    });
    if (bound == bounds.end() || bound->maximumUsefulResourceUnits == 0)
      return invalid("resource-time region has no resource bound");
    correspondence.push_back({region.region, 1,
                              bound->maximumUsefulResourceUnits,
                              ResourceTimeEstimateSupport::Unsupported,
                              ResourceTimeEstimateSupport::Unsupported,
                              region.logicalEpochCount});
  }

  std::vector<ArtifactRootReference> canonicalMappings(systemMappings.begin(),
                                                       systemMappings.end());
  llvm::sort(canonicalMappings, referenceLess);
  canonicalMappings.erase(
      std::unique(canonicalMappings.begin(), canonicalMappings.end()),
      canonicalMappings.end());
  // The finalist projection and the independent verifier share only the
  // existing invocation-local SystemMapping import session. A cache hit here
  // reuses an immutable import; it never bypasses the later verifier.
  ::loom::mapping::SystemMappingImportSession importSession(
      store, std::max<std::size_t>(1, canonicalMappings.size()),
      ::loom::mapping::SystemMappingImportSessionMode::ReuseEnclosing);
  std::vector<ImportedMappingProjection> imported;
  imported.reserve(canonicalMappings.size());
  for (const ArtifactRootReference &mapping : canonicalMappings) {
    if (executionControl.stopRequested()) {
      const auto importStats = importSession.statistics();
      accounting.mappingImportRequests = importStats.importRequests;
      accounting.mappingImportCacheHits = importStats.cacheHits;
      accounting.mappingImportCacheMisses = importStats.cacheMisses;
      accounting.mappingImportRetainedBytes = importStats.retainedBytes;
      accounting.elapsedNanoseconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - begin)
              .count();
      return ResourceTimeSpectrumFunnelResult{
          ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout,
              "resource-time spectrum materialization was cancelled or "
              "timed out",
              imported.size())},
          accounting};
    }
    auto projection = importProjection(mapping, correspondence, store);
    if (!projection)
      return projection.takeError();
    if (projection->resourceTimeProgress.kind ==
        ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
      ++accounting.mappingProgressQualified;
    else
      ++accounting.mappingProgressProofNotEstablished;
    imported.push_back(std::move(*projection));
  }
  const auto importStats = importSession.statistics();
  accounting.mappingImportRequests = importStats.importRequests;
  accounting.mappingImportCacheHits = importStats.cacheHits;
  accounting.mappingImportCacheMisses = importStats.cacheMisses;
  accounting.mappingImportRetainedBytes = importStats.retainedBytes;
  for (std::size_t ordinal = 0; ordinal != correspondence.size(); ++ordinal) {
    const auto &bound = bounds[ordinal];
    bool sawMinimum = false;
    bool sawMaximum = false;
    for (const ImportedMappingProjection &mapping : imported) {
      const auto resources = mapping.resourcesByRoot.find(
          correspondence[ordinal].root.entity.value());
      if (resources == mapping.resourcesByRoot.end())
        return invalid("resource-time Mapping projection lost a region");
      if (bound.support == ResourceTimeEstimateSupport::Exact &&
          resources->second.size() > bound.maximumUsefulResourceUnits) {
        accounting.elapsedNanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin)
                .count();
        return ResourceTimeSpectrumFunnelResult{
            ResourceTimeSpectrumVerification{incomplete(
                ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
                "verified Mapping allocation exceeds the workload-specific "
                "logical-domain resource bound",
                imported.size())},
            accounting};
      }
      sawMinimum |= resources->second.size() == 1;
      sawMaximum |=
          resources->second.size() == bound.maximumUsefulResourceUnits;
    }
    correspondence[ordinal].minimumBoundSupport =
        sawMinimum ? ResourceTimeEstimateSupport::Exact
                   : ResourceTimeEstimateSupport::Unsupported;
    correspondence[ordinal].maximumBoundSupport =
        sawMaximum && bound.support == ResourceTimeEstimateSupport::Exact
            ? ResourceTimeEstimateSupport::Exact
            : ResourceTimeEstimateSupport::Unsupported;
  }

  ::loom::pnr::ResourceTimeScheduleWitness witness;
  for (const ResourceTimeRegionFeature &region : regions)
    witness.regions.push_back(region.region);
  if (concurrencyBounds &&
      concurrencyBounds->support == ResourceTimeEstimateSupport::Exact) {
    if (concurrencyBounds->minimumPeakConcurrentRegions == 0 ||
        concurrencyBounds->maximumPeakConcurrentRegions == 0 ||
        concurrencyBounds->minimumPeakConcurrentRegions >
            concurrencyBounds->maximumPeakConcurrentRegions ||
        concurrencyBounds->maximumPeakConcurrentRegions > regions.size())
      return invalid("resource-time exact concurrency bounds are invalid");
    witness.minimumConcurrentRegions =
        concurrencyBounds->minimumPeakConcurrentRegions;
    witness.maximumConcurrentRegions =
        concurrencyBounds->maximumPeakConcurrentRegions;
    witness.concurrencyBoundStatus =
        ::loom::pnr::ResourceTimeConcurrencyBoundStatus::Exact;
  } else {
    witness.minimumConcurrentRegions = 1;
    witness.maximumConcurrentRegions = regions.size();
    witness.concurrencyBoundStatus =
        ::loom::pnr::ResourceTimeConcurrencyBoundStatus::ProofNotEstablished;
  }
  for (const ResourceTimeScheduleHint &hint : hints) {
    // Dataflow currently owns canonical root-start and root-completion event
    // families, but no token-publication or composite-completion family.
    // Never encode either shape as one arbitrary event; keep it as a typed
    // unsupported spectrum candidate until the event owner is extended.
    if (hasUnrepresentableCompositeEvent(hint)) {
      ++accounting.transitionUnsupportedHints;
      continue;
    }
    auto scenario = materializeHint(hint, regions, imported, accounting);
    if (!scenario)
      return scenario.takeError();
    if (*scenario) {
      witness.scenarios.push_back(std::move(**scenario));
      ++accounting.materializedScenarios;
    } else {
      const bool everyStateHasMapping =
          llvm::all_of(hint.states, [&](const ResourceTimeHintState &state) {
            return llvm::any_of(imported, [&](const auto &mapping) {
              return mappingMatchesState(mapping, state, accounting);
            });
          });
      if (everyStateHasMapping && imported.size() > 1)
        ++accounting.transitionUnsupportedHints;
      else
        ++accounting.unmatchedHints;
    }
  }
  if (witness.scenarios.empty()) {
    accounting.elapsedNanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    return ResourceTimeSpectrumFunnelResult{
        ResourceTimeSpectrumVerification{incomplete(
            accounting.transitionUnsupportedHints != 0
                ? ResourceTimeSpectrumIncompleteReason::Unsupported
                : ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            accounting.transitionUnsupportedHints != 0
                ? "bounded schedule requires a Mapping transition without a "
                  "verified safe-point migration contract"
                : "no bounded schedule hint matches an existing verified "
                  "SystemMapping allocation",
            imported.size())},
        accounting};
  }
  auto verified = verifyResourceTimeSpectrum(witness, correspondence, store,
                                             executionControl);
  if (!verified)
    return verified.takeError();
  // The independent verifier reuses the same invocation-owned immutable
  // SystemMapping session. Refresh the accounting after it runs so the
  // evidence covers both finalist projection and verifier imports.
  const auto finalImportStats = importSession.statistics();
  accounting.mappingImportRequests = finalImportStats.importRequests;
  accounting.mappingImportCacheHits = finalImportStats.cacheHits;
  accounting.mappingImportCacheMisses = finalImportStats.cacheMisses;
  accounting.mappingImportRetainedBytes = finalImportStats.retainedBytes;
  if (const auto *completed =
          std::get_if<VerifiedResourceTimeSpectrum>(&*verified)) {
    accounting.verifiedScenarios = completed->scenarios.size();
    std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
        verifiedMappings(&artifactRootReferenceLess);
    for (const auto &scenario : completed->scenarios)
      verifiedMappings.insert(scenario.systemMappings.begin(),
                              scenario.systemMappings.end());
    accounting.independentlyImportedMappings = verifiedMappings.size();
  } else {
    accounting.independentlyImportedMappings =
        std::get<IncompleteResourceTimeSpectrum>(*verified)
            .independentlyImportedMappingCount;
  }
  accounting.elapsedNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count();
  return ResourceTimeSpectrumFunnelResult{std::move(*verified), accounting};
}

} // namespace loom::dse
