#include "DSE/ResourceTimeSpectrum.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

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
  ImportedMappingProjection result{reference,
                                   dataflowReference,
                                   fabricReference,
                                   {},
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

bool sameHintAllocations(llvm::ArrayRef<ResourceTimeHintAllocation> lhs,
                         llvm::ArrayRef<ResourceTimeHintAllocation> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.region != right.region ||
        left.speedupPointOrdinal != right.speedupPointOrdinal ||
        left.resourceUnits != right.resourceUnits ||
        left.completionTimePicoseconds != right.completionTimePicoseconds)
      return false;
  return true;
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

llvm::Expected<std::optional<::loom::pnr::ResourceTimeScheduleScenario>>
materializeHint(
    const ResourceTimeScheduleHint &hint,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    llvm::ArrayRef<ImportedMappingProjection> mappings,
    llvm::ArrayRef<ResourceTimeMappingDeploymentEndpoint> mappingPath,
    const ArtifactStore &store, const BlobStore *blobs,
    ResourceTimeSpectrumFunnelAccounting &accounting,
    std::optional<std::string> &firstTransitionProofDiagnostic) {
  if (hint.states.size() != hint.actions.size() + 1)
    return invalid("resource-time hint action/state lineage is incomplete");
  // A resource-time schedule may change allocation at an event boundary.
  // Select the already imported Mapping independently for each state; a
  // single Mapping is only a valid shortcut when every state matches it.
  std::vector<const ImportedMappingProjection *> selectedStates(
      hint.states.size());
  std::vector<std::vector<const ImportedMappingProjection *>> matches;
  matches.reserve(hint.states.size());
  for (const ResourceTimeHintState &state : hint.states) {
    std::vector<const ImportedMappingProjection *> candidates;
    for (const ImportedMappingProjection &mapping : mappings)
      if (mappingMatchesState(mapping, state, accounting))
        candidates.push_back(&mapping);
    if (candidates.empty())
      return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
    matches.push_back(std::move(candidates));
  }
  const bool exactMappingPath =
      !mappingPath.empty() && mappingPath.size() == mappings.size();
  if (exactMappingPath) {
    std::size_t pathPosition = 0;
    for (std::size_t index = 0; index != hint.states.size(); ++index) {
      if (index != 0 && pathPosition + 1 != mappingPath.size()) {
        const ResourceTimeActionDelta &action = hint.actions[index - 1];
        if (action.kind == ResourceTimeActionKind::AdvanceEvent &&
            action.completedRegions.size() == 1)
          ++pathPosition;
      }
      const auto selected = llvm::find_if(mappings, [&](const auto &mapping) {
        return mapping.reference == mappingPath[pathPosition].mapping;
      });
      if (selected == mappings.end() ||
          !llvm::is_contained(matches[index], &*selected))
        return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
      selectedStates[index] = &*selected;
    }
    if (pathPosition + 1 != mappingPath.size())
      return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
  } else {
    for (std::size_t index = 0; index != hint.states.size(); ++index) {
      if (matches[index].size() == 1) {
        selectedStates[index] = matches[index].front();
        continue;
      }
      if (!hint.states[index].active.empty())
        return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
      // An empty boundary has no allocation fact of its own. Bind it to the
      // next uniquely matched nonempty state so a Mapping switch occurs at
      // the preceding completion. A terminal empty state retains the prior
      // Mapping.
      for (std::size_t next = index + 1; next != hint.states.size(); ++next) {
        if (hint.states[next].active.empty())
          continue;
        if (matches[next].size() != 1)
          return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
        selectedStates[index] = matches[next].front();
        break;
      }
      if (!selectedStates[index] && index != 0)
        selectedStates[index] = selectedStates[index - 1];
      if (!selectedStates[index])
        return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
    }
  }

  const bool mappingChanges = llvm::any_of(
      llvm::zip(llvm::drop_begin(selectedStates), selectedStates),
      [](auto pair) { return std::get<0>(pair) != std::get<1>(pair); });
  if (mappingChanges &&
      llvm::any_of(hint.actions, [](const ResourceTimeActionDelta &action) {
        return action.kind == ResourceTimeActionKind::AdvanceEvent &&
               action.completedRegions.size() > 1;
      })) {
    ++accounting.transitionUnsupportedHints;
    return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
  }

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

  auto materializeAllocations = [&](const ImportedMappingProjection &mapping,
                                    const ResourceTimeHintState &state)
      -> llvm::Expected<
          std::vector<::loom::pnr::ResourceTimeRegionAllocation>> {
    std::vector<::loom::pnr::ResourceTimeRegionAllocation> active;
    active.reserve(state.active.size());
    for (const ResourceTimeHintAllocation &allocation : state.active) {
      const auto resources =
          mapping.resourcesByRoot.find(allocation.region.entity.value());
      if (resources == mapping.resourcesByRoot.end())
        return invalid("matched Mapping lost a resource-time region");
      active.push_back({allocation.region, resources->second});
    }
    return active;
  };

  const auto appendState =
      [&](const ImportedMappingProjection &mapping,
          const ResourceTimeHintState &state, ::dataflow::EventFamilyKey event)
      -> llvm::Error {
    auto active = materializeAllocations(mapping, state);
    if (!active)
      return active.takeError();
    scenario.states.push_back({mapping.reference, std::move(event),
                               state.timePicoseconds, std::move(*active)});
    return llvm::Error::success();
  };

  for (std::size_t snapshot = 1; snapshot != hint.states.size(); ++snapshot) {
    const ResourceTimeActionDelta &action = hint.actions[snapshot - 1];
    if (action.kind != ResourceTimeActionKind::AdmitRegion &&
        action.completedRegions.empty())
      continue;
    const ResourceTimeHintState &state = hint.states[snapshot];
    if (action.kind == ResourceTimeActionKind::AdmitRegion) {
      if (llvm::Error error = appendState(
              *selectedStates[snapshot], state,
              ::dataflow::rootThreadStartEventFamily(*action.admittedRegion)))
        return std::move(error);
      continue;
    }

    if (action.completedRegions.size() == 1) {
      if (llvm::Error error = appendState(
              *selectedStates[snapshot], state,
              ::dataflow::rootThreadCompletionEventFamily(
                  action.completedRegions.front())))
        return std::move(error);
      continue;
    }

    if (selectedStates[snapshot - 1] != selectedStates[snapshot])
      return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};
    ResourceTimeHintState boundary = hint.states[snapshot - 1];
    boundary.timePicoseconds = state.timePicoseconds;
    for (const ::dataflow::RootThreadLaunchRef completed :
         action.completedRegions) {
      const auto active =
          llvm::find_if(boundary.active, [&](const auto &allocation) {
            return allocation.region == completed;
          });
      if (active == boundary.active.end())
        return invalid("resource-time event completes an inactive region");
      boundary.active.erase(active);
      if (llvm::Error error = appendState(
              *selectedStates[snapshot], boundary,
              ::dataflow::rootThreadCompletionEventFamily(completed)))
        return std::move(error);
    }
    if (!sameHintAllocations(boundary.active, state.active))
      return invalid("resource-time simultaneous completion does not match "
                     "the event frontier state");
  }
  if (scenario.states.empty())
    return std::optional<::loom::pnr::ResourceTimeScheduleScenario>{};

  // The schedule planner does not own Mapping legality. When it selects two
  // distinct imported states but no compiler-proven edge was attached, keep a
  // typed proof-not-established edge so the independent verifier reports the
  // causal gap instead of relabeling it as an unmatched application.
  for (std::size_t index = 1; index != scenario.states.size(); ++index) {
    const auto &before = scenario.states[index - 1];
    const auto &after = scenario.states[index];
    if (before.mapping == after.mapping)
      continue;
    ::loom::pnr::ResourceTimeTransition transition{
        after.event,
        std::nullopt,
        {before.mapping, std::nullopt},
        {after.mapping, std::nullopt},
        {},
        {},
        {},
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        ::loom::pnr::ResourceTimeTransitionStatus::ProofNotEstablished};
    transition.beforeActive = before.active;
    transition.afterActive = after.active;
    for (const auto &execution : scenario.executions)
      if (execution.completionPicoseconds < after.timePicoseconds)
        transition.completedBefore.push_back(execution.region);
    llvm::sort(transition.completedBefore, rootLess);
    if (blobs && exactMappingPath) {
      const auto parent = llvm::find_if(mappingPath, [&](const auto &endpoint) {
        return endpoint.mapping == before.mapping;
      });
      const auto child = llvm::find_if(mappingPath, [&](const auto &endpoint) {
        return endpoint.mapping == after.mapping;
      });
      const auto completing =
          llvm::find_if(before.active, [&](const auto &allocation) {
            return ::dataflow::rootThreadCompletionEventFamily(
                       allocation.region) == after.event;
          });
      if (parent != mappingPath.end() && child != mappingPath.end() &&
          completing != before.active.end()) {
        transition.safePoint = ::loom::pnr::ResourceTimeSafePointReference{
            {::dataflow::canonicalDataflowSchema.identity.str(),
             ::dataflow::canonicalDataflowSchema.version,
             completing->region.artifact},
            ::loom::pnr::ResourceTimeSafePointKind::Completion};
        transition.parent.deployment = parent->deployment;
        transition.child.deployment = child->deployment;
        auto finalized = ::loom::pnr::finalizeResourceTimeTransition(
            transition, store, *blobs);
        if (finalized)
          transition = std::move(*finalized);
        else {
          ++accounting.transitionProofFailures;
          std::string diagnostic = llvm::toString(finalized.takeError());
          if (!firstTransitionProofDiagnostic)
            firstTransitionProofDiagnostic = std::move(diagnostic);
        }
      }
    }
    scenario.transitions.transitions.push_back(std::move(transition));
  }
  return std::optional<::loom::pnr::ResourceTimeScheduleScenario>(
      std::move(scenario));
}

bool hasUnrepresentableEvent(const ResourceTimeScheduleHint &hint) {
  return llvm::any_of(hint.actions, [](const ResourceTimeActionDelta &action) {
    return action.kind == ResourceTimeActionKind::AdvanceEvent &&
           !action.tokenReadyProducers.empty();
  });
}

bool hasExplicitTemporalActiveSet(
    const ::loom::pnr::ResourceTimeScheduleScenario &scenario,
    std::size_t coveredRegionCount) {
  if (coveredRegionCount < 2 || scenario.states.size() < 2)
    return false;
  std::set<::dataflow::RootThreadLaunchRef, decltype(&rootLess)> first(
      &rootLess);
  bool haveFirst = false;
  bool sawDifferentActiveSet = false;
  for (const auto &state : scenario.states) {
    std::set<::dataflow::RootThreadLaunchRef, decltype(&rootLess)> active(
        &rootLess);
    for (const auto &allocation : state.active)
      active.insert(allocation.region);
    if (!haveFirst) {
      first = std::move(active);
      haveFirst = true;
      continue;
    }
    if (active != first)
      sawDifferentActiveSet = true;
  }
  return sawDifferentActiveSet;
}

} // namespace

llvm::StringRef resourceTimeSpectrumIncompleteReasonSpelling(
    ResourceTimeSpectrumIncompleteReason reason) {
  switch (reason) {
  case ResourceTimeSpectrumIncompleteReason::Unsupported:
    return "unsupported";
  case ResourceTimeSpectrumIncompleteReason::ProofNotEstablished:
    return "proof_not_established";
  case ResourceTimeSpectrumIncompleteReason::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown resource-time Spectrum incomplete reason");
}

llvm::Expected<ResourceTimeSpectrumVerification> verifyResourceTimeSpectrum(
    const ::loom::pnr::ResourceTimeScheduleWitness &witness,
    llvm::ArrayRef<ResourceTimeRegionMapping> regions,
    const ArtifactStore &store, ExecutionControlView executionControl,
    const BlobStore *blobs) {
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
    if (region.minimumFeasibleAccCoreCount &&
        (*region.minimumFeasibleAccCoreCount == 0 ||
         *region.minimumFeasibleAccCoreCount >
             region.maximumUsefulAccCoreCount))
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
          ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
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
        if ((region.minimumFeasibleAccCoreCount &&
             count < *region.minimumFeasibleAccCoreCount) ||
            count > region.maximumUsefulAccCoreCount)
          return ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
              "verified Mapping allocation is outside the region resource "
              "bounds",
              imported.size())};
        everyAllocationAtMinimum &=
            region.minimumFeasibleAccCoreCount &&
            count == *region.minimumFeasibleAccCoreCount;
        everyAllocationAtMaximum &= count == region.maximumUsefulAccCoreCount;
        everyMinimumBoundExact &=
            region.minimumFeasibleAccCoreCount &&
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
      if (transition.parent.mapping != change.first ||
          transition.child.mapping != change.second)
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time transition does not match its adjacent states",
            imported.size())};
      switch (transition.status) {
      case ::loom::pnr::ResourceTimeTransitionStatus::Verified:
        if (!blobs)
          return ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
              "resource-time Mapping transition has no independently "
              "verified Deployment closure",
              imported.size())};
        if (llvm::Error error =
                ::loom::pnr::verifyResourceTimeTransitionClosure(transition,
                                                                 store, *blobs))
          return ResourceTimeSpectrumVerification{incomplete(
              ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
              "resource-time transition closure is not established: " +
                  llvm::toString(std::move(error)),
              imported.size())};
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

    std::optional<::loom::pnr::ResourceTimeTransitionGraph> transitionGraph;
    if (!scenario.transitions.transitions.empty()) {
      transitionGraph.emplace(::loom::pnr::ResourceTimeTransitionGraph{
          scenario.transitions.transitions.front().parent,
          {},
          scenario.transitions.transitions});
      const auto appendEndpoint = [&](const auto &endpoint) {
        if (!llvm::is_contained(transitionGraph->endpoints, endpoint))
          transitionGraph->endpoints.push_back(endpoint);
      };
      for (const auto &edge : transitionGraph->transitions) {
        appendEndpoint(edge.parent);
        appendEndpoint(edge.child);
      }
      if (!blobs)
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time transition graph has no Deployment verifier",
            imported.size())};
      if (llvm::Error error = ::loom::pnr::verifyResourceTimeTransitionGraph(
              *transitionGraph, store, *blobs))
        return ResourceTimeSpectrumVerification{incomplete(
            ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
            "resource-time transition graph closure is not established: " +
                llvm::toString(std::move(error)),
            imported.size())};
    }

    PreMappingSpectrumClass spectrumClass =
        PreMappingSpectrumClass::Intermediate;
    const bool exactConcurrencyBounds =
        witness.concurrencyBoundStatus ==
        ::loom::pnr::ResourceTimeConcurrencyBoundStatus::Exact;
    if (witness.regions.size() > 1 && exactConcurrencyBounds &&
        everyMinimumBoundExact && everyAllocationAtMinimum &&
        peakConcurrentRegions == witness.maximumConcurrentRegions)
      spectrumClass = PreMappingSpectrumClass::MaxSpatial;
    else if (witness.regions.size() > 1 && exactConcurrencyBounds &&
             hasExplicitTemporalActiveSet(scenario, witness.regions.size()) &&
             everyMaximumBoundExact && everyAllocationAtMaximum &&
             peakConcurrentRegions == witness.minimumConcurrentRegions)
      spectrumClass = PreMappingSpectrumClass::MaxTemporal;
    verified.scenarios.push_back(
        {indexedScenario.index(), spectrumClass, peakConcurrentRegions,
         scenario.makespanPicoseconds, std::move(scenarioMappings),
         scenario.states, scenario.transitions, std::move(transitionGraph)});
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
    std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds,
    const BlobStore *blobs,
    llvm::ArrayRef<ResourceTimeMappingDeploymentEndpoint> mappingPath) {
  const auto begin = std::chrono::steady_clock::now();
  ResourceTimeSpectrumFunnelAccounting accounting;
  std::optional<std::string> firstTransitionProofDiagnostic;
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
    const std::optional<std::uint64_t> minimumFeasible =
        bound->minimumFeasibleResourceUnits == 0
            ? std::nullopt
            : std::optional<std::uint64_t>(bound->minimumFeasibleResourceUnits);
    if (minimumFeasible && *minimumFeasible > bound->maximumUsefulResourceUnits)
      return invalid("resource-time region minimum allocation exceeds its "
                     "maximum useful allocation");
    correspondence.push_back(
        {region.region, minimumFeasible, bound->maximumUsefulResourceUnits,
         bound->minimumSupport, ResourceTimeEstimateSupport::Unsupported,
         region.logicalEpochCount});
  }

  std::vector<ArtifactRootReference> canonicalMappings(systemMappings.begin(),
                                                       systemMappings.end());
  llvm::sort(canonicalMappings, referenceLess);
  canonicalMappings.erase(
      std::unique(canonicalMappings.begin(), canonicalMappings.end()),
      canonicalMappings.end());
  std::vector<ArtifactRootReference> endpointMappings;
  endpointMappings.reserve(mappingPath.size());
  if (!mappingPath.empty() && !blobs)
    return invalid(
        "resource-time Mapping path requires a Deployment BlobStore");
  if (!mappingPath.empty() && mappingPath.size() != canonicalMappings.size())
    return invalid("resource-time Mapping path must cover every Mapping "
                   "finalist exactly once");
  for (const ResourceTimeMappingDeploymentEndpoint &endpoint : mappingPath) {
    if (!llvm::is_contained(canonicalMappings, endpoint.mapping))
      return invalid("resource-time Mapping path names a foreign "
                     "Mapping finalist");
    if (endpoint.deployment.schemaIdentity !=
            ::loom::deployment::deploymentSchema.identity ||
        endpoint.deployment.schemaVersion !=
            ::loom::deployment::deploymentSchema.version)
      return invalid("resource-time Mapping path has a non-Deployment root");
    if (llvm::is_contained(endpointMappings, endpoint.mapping))
      return invalid("resource-time Mapping path repeats a Mapping");
    endpointMappings.push_back(endpoint.mapping);
  }
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
      accounting.independentlyImportedMappings = imported.size();
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
  if (!mappingPath.empty()) {
    const ArtifactRootReference &endpointDataflow = imported.front().dataflow;
    const ArtifactRootReference &endpointFabric = imported.front().fabric;
    for (const ImportedMappingProjection &mapping : imported) {
      if (mapping.dataflow != endpointDataflow)
        return invalid("resource-time Mapping path changes Canonical Dataflow "
                       "without a typed correspondence owner");
      if (mapping.fabric != endpointFabric)
        return invalid("resource-time Mapping path changes the immutable "
                       "Fabric");
    }
    for (const ResourceTimeMappingDeploymentEndpoint &endpoint : mappingPath) {
      auto deployment = ::loom::deployment::importDeployment(
          endpoint.deployment, store, *blobs);
      if (!deployment)
        return deployment.takeError();
      if (deployment->deployment().systemMapping() != endpoint.mapping)
        return invalid("resource-time Deployment endpoint does not select its "
                       "paired SystemMapping");
    }
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
        accounting.independentlyImportedMappings = imported.size();
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
      sawMinimum |= correspondence[ordinal].minimumFeasibleAccCoreCount &&
                    resources->second.size() ==
                        *correspondence[ordinal].minimumFeasibleAccCoreCount;
      sawMaximum |=
          resources->second.size() == bound.maximumUsefulResourceUnits;
    }
    correspondence[ordinal].minimumBoundSupport =
        sawMinimum && bound.minimumSupport == ResourceTimeEstimateSupport::Exact
            ? ResourceTimeEstimateSupport::Exact
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
    // Equal-time root completions are emitted as a deterministic sequence of
    // canonical completion families when no Mapping edge is crossed. Token
    // publication and a Mapping change across a composite frontier still lack
    // an owner event/correspondence and remain typed unsupported.
    if (hasUnrepresentableEvent(hint)) {
      ++accounting.transitionUnsupportedHints;
      continue;
    }
    auto scenario =
        materializeHint(hint, regions, imported, mappingPath, store, blobs,
                        accounting, firstTransitionProofDiagnostic);
    if (!scenario)
      return scenario.takeError();
    if (*scenario) {
      witness.scenarios.push_back(std::move(**scenario));
      ++accounting.materializedScenarios;
    } else {
      ++accounting.unmatchedHints;
    }
  }
  if (witness.scenarios.empty()) {
    accounting.independentlyImportedMappings = imported.size();
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
                                             executionControl, blobs);
  if (!verified)
    return verified.takeError();
  if (firstTransitionProofDiagnostic)
    if (auto *incomplete =
            std::get_if<IncompleteResourceTimeSpectrum>(&*verified))
      incomplete->diagnostic += ": transition finalization failed: " +
                                *firstTransitionProofDiagnostic;
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

bool resourceTimeSpectrumAdmitsMappingClass(
    const ResourceTimeSpectrumFunnelResult &result,
    const ArtifactRootReference &mapping,
    std::optional<PreMappingSpectrumClass> requestedClass) {
  const auto *verified =
      std::get_if<VerifiedResourceTimeSpectrum>(&result.verification);
  return verified &&
         llvm::any_of(verified->scenarios, [&](const auto &scenario) {
           return llvm::is_contained(scenario.systemMappings, mapping) &&
                  (!requestedClass ||
                   scenario.spectrumClass == *requestedClass);
         });
}

} // namespace loom::dse
