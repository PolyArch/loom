#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "ResourceTimeTransitionInternal.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <set>
#include <system_error>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_mapping_migration_invalid: " + message);
}

bool rootLess(::dataflow::RootThreadLaunchRef lhs,
              ::dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

std::vector<std::uint8_t> canonicalResourceBytes(
    const ::loom::fabric::FabricPhysicalOccurrenceOwnerRef &resource) {
  return ::loom::fabric::canonicalFabricBytes(resource);
}

bool allocationsEquivalent(llvm::ArrayRef<ResourceTimeRegionAllocation> lhs,
                           llvm::ArrayRef<ResourceTimeRegionAllocation> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const ResourceTimeRegionAllocation &left : lhs) {
    const auto right = llvm::find_if(rhs, [&](const auto &candidate) {
      return candidate.region == left.region;
    });
    if (right == rhs.end() || left.resources.size() != right->resources.size())
      return false;
    std::vector<std::vector<std::uint8_t>> leftResources;
    std::vector<std::vector<std::uint8_t>> rightResources;
    leftResources.reserve(left.resources.size());
    rightResources.reserve(right->resources.size());
    for (const auto &resource : left.resources)
      leftResources.push_back(canonicalResourceBytes(resource));
    for (const auto &resource : right->resources)
      rightResources.push_back(canonicalResourceBytes(resource));
    llvm::sort(leftResources);
    llvm::sort(rightResources);
    if (leftResources != rightResources)
      return false;
  }
  return true;
}

void canonicalizeResources(
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> &resources) {
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return canonicalResourceBytes(lhs) < canonicalResourceBytes(rhs);
  });
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
}

} // namespace

llvm::StringRef
resourceTimeTransitionStatusSpelling(ResourceTimeTransitionStatus status) {
  switch (status) {
  case ResourceTimeTransitionStatus::Verified:
    return "verified";
  case ResourceTimeTransitionStatus::Unsupported:
    return "unsupported";
  case ResourceTimeTransitionStatus::ProofNotEstablished:
    return "proof_not_established";
  case ResourceTimeTransitionStatus::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown resource-time transition status");
}

llvm::StringRef
resourceTimeLiveStateClassSpelling(ResourceTimeLiveStateClass stateClass) {
  switch (stateClass) {
  case ResourceTimeLiveStateClass::LogicalMemory:
    return "logical_memory";
  case ResourceTimeLiveStateClass::OrderedChannel:
    return "ordered_channel";
  case ResourceTimeLiveStateClass::DynamicWork:
    return "dynamic_work";
  }
  llvm_unreachable("unknown resource-time live-state class");
}

llvm::StringRef resourceTimeLiveStateMigrationSpelling(
    ResourceTimeLiveStateMigration migration) {
  switch (migration) {
  case ResourceTimeLiveStateMigration::RetainedInPlace:
    return "retained_in_place";
  case ResourceTimeLiveStateMigration::Copied:
    return "copied";
  }
  llvm_unreachable("unknown resource-time live-state migration");
}

llvm::StringRef resourceTimeTransitionRefusalReasonSpelling(
    ResourceTimeTransitionRefusalReason reason) {
  switch (reason) {
  case ResourceTimeTransitionRefusalReason::OrderedChannelState:
    return "ordered_channel_state";
  case ResourceTimeTransitionRefusalReason::DynamicWorkState:
    return "dynamic_work_state";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryUnbound:
    return "logical_memory_unbound";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryExtentUnknown:
    return "logical_memory_extent_unknown";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryCopyShapeUnsupported:
    return "logical_memory_copy_shape_unsupported";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryReinitialized:
    return "logical_memory_reinitialized";
  case ResourceTimeTransitionRefusalReason::HardwareBindingChanged:
    return "hardware_binding_changed";
  case ResourceTimeTransitionRefusalReason::
      RuntimeTransitionCapabilityUnavailable:
    return "runtime_transition_capability_unavailable";
  case ResourceTimeTransitionRefusalReason::CompletionFrontierInadmissible:
    return "completion_frontier_inadmissible";
  }
  llvm_unreachable("unknown resource-time transition refusal reason");
}

char ResourceTimeTransitionRefusal::ID = 0;

void ResourceTimeTransitionRefusal::log(llvm::raw_ostream &stream) const {
  stream << "resource_time_transition_refused("
         << resourceTimeTransitionRefusalReasonSpelling(reason_)
         << "): " << message_;
}

std::error_code ResourceTimeTransitionRefusal::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef
resourceTimeSafePointKindSpelling(ResourceTimeSafePointKind kind) {
  switch (kind) {
  case ResourceTimeSafePointKind::Completion:
    return "completion";
  case ResourceTimeSafePointKind::Explicit:
    return "explicit";
  }
  llvm_unreachable("unknown resource-time safe-point kind");
}

llvm::StringRef
resourceTimeReadinessKindSpelling(ResourceTimeReadinessKind kind) {
  switch (kind) {
  case ResourceTimeReadinessKind::Completion:
    return "completion";
  case ResourceTimeReadinessKind::FifoToken:
    return "fifo_token";
  }
  llvm_unreachable("unknown resource-time readiness kind");
}

llvm::StringRef resourceTimeConcurrencyBoundStatusSpelling(
    ResourceTimeConcurrencyBoundStatus status) {
  switch (status) {
  case ResourceTimeConcurrencyBoundStatus::Exact:
    return "exact";
  case ResourceTimeConcurrencyBoundStatus::ProofNotEstablished:
    return "proof_not_established";
  }
  llvm_unreachable("unknown resource-time concurrency bound status");
}

llvm::Error
validateResourceTimeTransition(const ResourceTimeTransition &transition) {
  const auto validateMappingReference =
      [](const ArtifactRootReference &root,
         llvm::StringRef name) -> llvm::Error {
    if (root.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        root.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
      return invalid(name + " is not a Mapping artifact reference");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validateMappingReference(transition.parent.mapping, "parent_mapping"))
    return error;
  if (llvm::Error error =
          validateMappingReference(transition.child.mapping, "child_mapping"))
    return error;
  const auto validateDeploymentReference =
      [](const std::optional<ArtifactRootReference> &root,
         llvm::StringRef name) -> llvm::Error {
    if (root &&
        (root->schemaIdentity !=
             ::loom::deployment::deploymentSchema.identity ||
         root->schemaVersion != ::loom::deployment::deploymentSchema.version))
      return invalid(name + " is not a Deployment artifact reference");
    return llvm::Error::success();
  };
  if (llvm::Error error = validateDeploymentReference(
          transition.parent.deployment, "parent_deployment"))
    return error;
  if (llvm::Error error = validateDeploymentReference(
          transition.child.deployment, "child_deployment"))
    return error;
  const auto validateAllocations =
      [](llvm::ArrayRef<ResourceTimeRegionAllocation> values,
         llvm::StringRef name) -> llvm::Error {
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> usedResources;
    for (std::size_t index = 0; index != values.size(); ++index) {
      const auto &allocation = values[index];
      if (allocation.resources.empty())
        return invalid(name + " contains a region with no resources");
      for (std::size_t prior = 0; prior != index; ++prior)
        if (values[prior].region == allocation.region)
          return invalid(name + " contains a duplicate region");
      for (std::size_t resource = 0; resource != allocation.resources.size();
           ++resource) {
        for (std::size_t prior = 0; prior != resource; ++prior)
          if (allocation.resources[prior] == allocation.resources[resource])
            return invalid(name + " contains a duplicate resource");
        if (llvm::is_contained(usedResources, allocation.resources[resource]))
          return invalid(name + " assigns one physical resource to "
                                "multiple active regions");
        usedResources.push_back(allocation.resources[resource]);
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validateAllocations(transition.beforeActive, "before_active"))
    return error;
  if (llvm::Error error =
          validateAllocations(transition.afterActive, "after_active"))
    return error;
  std::optional<ArtifactIdentity> activeDataflow;
  for (const auto *allocations :
       {&transition.beforeActive, &transition.afterActive})
    for (const ResourceTimeRegionAllocation &allocation : *allocations) {
      if (activeDataflow && *activeDataflow != allocation.region.artifact)
        return invalid("resource-time transition spans multiple Dataflow "
                       "identities without typed correspondence");
      activeDataflow = allocation.region.artifact;
    }
  for (std::size_t index = 0; index != transition.completedBefore.size();
       ++index) {
    const ::dataflow::RootThreadLaunchRef completed =
        transition.completedBefore[index];
    if (activeDataflow && *activeDataflow != completed.artifact)
      return invalid("resource-time completion frontier names a foreign "
                     "Dataflow root");
    activeDataflow = completed.artifact;
    const auto containsCompleted =
        [&](llvm::ArrayRef<ResourceTimeRegionAllocation> allocations) {
          return llvm::any_of(allocations, [&](const auto &allocation) {
            return allocation.region == completed;
          });
        };
    if (containsCompleted(transition.beforeActive) ||
        containsCompleted(transition.afterActive))
      return invalid("resource-time completion frontier contains an active "
                     "region");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (transition.completedBefore[prior] == completed)
        return invalid("resource-time completion frontier contains a duplicate "
                       "region");
  }
  for (std::size_t index = 0; index != transition.logicalMemories.size();
       ++index) {
    const ResourceTimeLogicalMemoryCorrespondence &memory =
        transition.logicalMemories[index];
    if (activeDataflow && *activeDataflow != memory.memory.artifact)
      return invalid("resource-time live-state correspondence names a "
                     "foreign Dataflow memory");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (transition.logicalMemories[prior].memory == memory.memory)
        return invalid("resource-time live-state correspondence repeats a "
                       "logical memory");
    if (memory.migration == ResourceTimeLiveStateMigration::RetainedInPlace) {
      if (memory.parentBinding != memory.childBinding ||
          memory.migrationTimePicoseconds != 0)
        return invalid("retained-in-place live state must keep its physical "
                       "binding at exact zero migration cost");
    } else if (memory.migration == ResourceTimeLiveStateMigration::Copied) {
      if (memory.parentBinding == memory.childBinding ||
          memory.migrationTimePicoseconds == 0)
        return invalid("copied live state must change physical binding at a "
                       "nonzero migration cost");
    } else {
      return invalid("resource-time live-state correspondence has an unknown "
                     "migration disposition");
    }
  }
  if (transition.status == ResourceTimeTransitionStatus::Verified) {
    if (!transition.safePoint)
      return invalid("verified resource-time transition has no compiler-known "
                     "safe point");
    if (!transition.parent.deployment || !transition.child.deployment)
      return invalid("verified resource-time transition has no exact parent "
                     "and child Deployment references");
    if (!transition.reprogrammingTimePicoseconds ||
        !transition.migrationTimePicoseconds)
      return invalid("verified resource-time transition has no exact "
                     "reprogramming and migration costs");
    if (!transition.resourceDeltaDigest ||
        !transition.configurationDeltaDigest || !transition.routeDeltaDigest)
      return invalid("verified resource-time transition lacks derived delta "
                     "or route digests");
  }
  if (transition.safePoint) {
    if (transition.safePoint->artifact.schemaIdentity.empty())
      return invalid("resource-time transition has an empty safe-point "
                     "artifact");
    if (transition.safePoint->kind == ResourceTimeSafePointKind::Completion) {
      if (transition.safePoint->artifact.schemaIdentity !=
              ::dataflow::canonicalDataflowSchema.identity ||
          transition.safePoint->artifact.schemaVersion !=
              ::dataflow::canonicalDataflowSchema.version)
        return invalid("completion safe point must be owned by Canonical "
                       "Dataflow");
      const auto completing =
          llvm::find_if(transition.beforeActive,
                        [&](const ResourceTimeRegionAllocation &allocation) {
                          return allocation.region.artifact ==
                                     transition.safePoint->artifact.artifact &&
                                 ::dataflow::rootThreadCompletionEventFamily(
                                     allocation.region) == transition.trigger;
                        });
      if (completing == transition.beforeActive.end())
        return invalid("completion safe point is not the completion event of "
                       "an active parent region (active region count " +
                       llvm::Twine(transition.beforeActive.size()) + ")");
    } else if (transition.safePoint->artifact.schemaIdentity ==
                   ::dataflow::canonicalDataflowSchema.identity &&
               transition.safePoint->artifact.schemaVersion ==
                   ::dataflow::canonicalDataflowSchema.version) {
      return invalid("explicit safe point requires a compiler proof artifact, "
                     "not only a Canonical Dataflow root");
    }
  }
  return llvm::Error::success();
}

llvm::Error
verifyResourceTimeTransitionClosure(const ResourceTimeTransition &transition,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs) {
  if (llvm::Error error = validateResourceTimeTransition(transition))
    return error;
  if (transition.status != ResourceTimeTransitionStatus::Verified)
    return invalid("resource-time transition closure requires a verified "
                   "edge status");
  if (!transition.safePoint || !transition.parent.deployment ||
      !transition.child.deployment)
    return invalid("verified resource-time transition lost required closure "
                   "references");

  auto parentDeployment = ::loom::deployment::importDeployment(
      *transition.parent.deployment, artifacts, blobs);
  if (!parentDeployment)
    return parentDeployment.takeError();
  auto childDeployment = ::loom::deployment::importDeployment(
      *transition.child.deployment, artifacts, blobs);
  if (!childDeployment)
    return childDeployment.takeError();
  if (parentDeployment->deployment().systemMapping() !=
      transition.parent.mapping)
    return invalid("parent Deployment does not select the parent "
                   "SystemMapping");
  if (childDeployment->deployment().systemMapping() != transition.child.mapping)
    return invalid("child Deployment does not select the child "
                   "SystemMapping");

  auto parentMapping = ::loom::mapping::importSystemMapping(
      transition.parent.mapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  auto childMapping =
      ::loom::mapping::importSystemMapping(transition.child.mapping, artifacts);
  if (!childMapping)
    return childMapping.takeError();
  if (parentMapping->view().dataflowIdentity() !=
      childMapping->view().dataflowIdentity())
    return invalid("resource-time transition changes Canonical Dataflow "
                   "without a typed live-state correspondence owner");
  if (parentMapping->view().fabricIdentity() !=
      childMapping->view().fabricIdentity())
    return invalid("resource-time transition changes the immutable Fabric");

  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      parentMapping->view().dataflowIdentity()};
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  if (llvm::Error error = dataflow->validate(transition.trigger))
    return invalid("resource-time transition trigger is not owned by the "
                   "endpoint "
                   "Dataflow: " +
                   llvm::toString(std::move(error)));
  if (transition.safePoint->kind == ResourceTimeSafePointKind::Explicit)
    return invalid("explicit resource-time safe-point closure is not "
                   "established by a typed compiler proof importer");

  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      parentMapping->view().fabricIdentity()};
  auto fabricArtifact =
      ::loom::fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();
  auto parentContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, parentMapping->view().executionBindings());
  if (!parentContexts)
    return parentContexts.takeError();
  auto childContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, childMapping->view().executionBindings());
  if (!childContexts)
    return childContexts.takeError();

  const auto verifyAllocations =
      [&](llvm::ArrayRef<ResourceTimeRegionAllocation> allocations,
          const ::loom::mapping::SystemExecutionContextProjection &contexts,
          llvm::StringRef name) -> llvm::Error {
    for (const ResourceTimeRegionAllocation &allocation : allocations) {
      if (allocation.region.artifact != dataflowReference.artifact)
        return invalid(name + " names a foreign Dataflow region");
      auto root = dataflow->resolve(allocation.region);
      if (!root)
        return root.takeError();
      auto expected =
          projectResourceTimeMappingResources(contexts, allocation.region);
      if (!expected)
        return expected.takeError();
      std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> observed =
          allocation.resources;
      canonicalizeResources(observed);
      if (observed != *expected)
        return invalid(name + " disagrees with the independently imported "
                              "SystemMapping execution binding");
      for (const auto &resource : observed) {
        auto resolved = system->resolvePhysicalOwner(resource);
        if (!resolved)
          return invalid(name + " names a resource outside the endpoint "
                                "Fabric");
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = verifyAllocations(
          transition.beforeActive, *parentContexts, "parent allocation"))
    return error;
  if (llvm::Error error = verifyAllocations(transition.afterActive,
                                            *childContexts, "child allocation"))
    return error;

  if (llvm::Error error = verifyResourceTimeTransitionDeltaDigests(
          transition, artifacts, blobs))
    return error;
  return llvm::Error::success();
}

llvm::Error validateResourceTimeTransitionSequence(
    const ResourceTimeTransitionSequence &sequence) {
  for (std::size_t index = 0; index != sequence.transitions.size(); ++index) {
    const ResourceTimeTransition &transition = sequence.transitions[index];
    if (llvm::Error error = validateResourceTimeTransition(transition))
      return error;
    if (index != 0 &&
        sequence.transitions[index - 1].child != transition.parent)
      return invalid("resource-time transition sequence is not chained by "
                     "Mapping and Deployment reference");
  }
  return llvm::Error::success();
}

llvm::Error validateResourceTimeScheduleWitness(
    const ResourceTimeScheduleWitness &witness) {
  if (witness.regions.empty())
    return invalid("resource-time schedule witness has no regions");
  if (witness.scenarios.empty())
    return invalid("resource-time schedule witness has no scenarios");
  if (witness.minimumConcurrentRegions == 0 ||
      witness.maximumConcurrentRegions == 0 ||
      witness.minimumConcurrentRegions > witness.maximumConcurrentRegions ||
      witness.maximumConcurrentRegions > witness.regions.size())
    return invalid("resource-time schedule witness has an invalid concurrency "
                   "bound");

  const auto hasRegion = [&](::dataflow::RootThreadLaunchRef reference) {
    return llvm::is_contained(witness.regions, reference);
  };
  for (std::size_t index = 0; index != witness.regions.size(); ++index) {
    if (index != 0 &&
        witness.regions[index].artifact != witness.regions.front().artifact)
      return invalid("resource-time schedule witness spans multiple Dataflow "
                     "identities without typed correspondence");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (witness.regions[prior] == witness.regions[index])
        return invalid("resource-time schedule witness contains a duplicate "
                       "region");
  }

  const auto mappingReference = [](const ArtifactRootReference &reference) {
    return reference.schemaIdentity ==
               ::loom::mapping::mappingArtifactSchema.identity &&
           reference.schemaVersion ==
               ::loom::mapping::mappingArtifactSchema.version;
  };
  for (auto indexedScenario : llvm::enumerate(witness.scenarios)) {
    const std::size_t scenarioOrdinal = indexedScenario.index();
    const ResourceTimeScheduleScenario &scenario = indexedScenario.value();
    if (scenario.executions.empty())
      return invalid("resource-time schedule scenario has no executions");
    if (scenario.states.empty())
      return invalid("resource-time schedule scenario has no event state");

    const auto findExecution = [&](::dataflow::RootThreadLaunchRef region)
        -> const ResourceTimeRegionExecution * {
      for (const ResourceTimeRegionExecution &execution : scenario.executions)
        if (execution.region == region)
          return &execution;
      return nullptr;
    };
    for (std::size_t index = 0; index != scenario.executions.size(); ++index) {
      const ResourceTimeRegionExecution &execution = scenario.executions[index];
      if (!hasRegion(execution.region))
        return invalid("resource-time execution references a foreign region");
      if (execution.readyPicoseconds > execution.startPicoseconds ||
          execution.startPicoseconds >= execution.completionPicoseconds)
        return invalid(
            "resource-time execution interval is not nonempty and ordered");
      for (const ResourceTimeRegionPrerequisite &prerequisite :
           execution.prerequisites) {
        if (!hasRegion(prerequisite.region) ||
            prerequisite.region == execution.region)
          return invalid("resource-time execution has an invalid prerequisite");
        const ResourceTimeRegionExecution *producer =
            findExecution(prerequisite.region);
        if (!producer)
          return invalid("resource-time prerequisite has no execution");
        if (prerequisite.readiness == ResourceTimeReadinessKind::Completion &&
            producer->completionPicoseconds > execution.readyPicoseconds)
          return invalid("completion-gated resource-time execution starts "
                         "before its prerequisite completes");
      }
      for (std::size_t prior = 0; prior != index; ++prior)
        if (scenario.executions[prior].region == execution.region)
          return invalid("resource-time scenario contains a duplicate "
                         "execution region");
    }

    std::vector<ArtifactRootReference> mappingReferences;
    mappingReferences.reserve(scenario.states.size() * 2);
    std::uint64_t maximumCompletion = 0;
    for (const ResourceTimeRegionExecution &execution : scenario.executions)
      maximumCompletion =
          std::max(maximumCompletion, execution.completionPicoseconds);

    std::optional<std::uint64_t> previousTime;
    std::vector<::dataflow::RootThreadLaunchRef> orderedActive;
    bool orderedSawAdmission = false;
    for (std::size_t stateOrdinal = 0; stateOrdinal != scenario.states.size();
         ++stateOrdinal) {
      const ResourceTimeScheduleState &state = scenario.states[stateOrdinal];
      if (!mappingReference(state.mapping))
        return invalid("resource-time state has a non-Mapping reference");
      if (previousTime && *previousTime > state.timePicoseconds)
        return invalid("resource-time state times are not monotonic");
      previousTime = state.timePicoseconds;
      if (!llvm::is_contained(mappingReferences, state.mapping))
        mappingReferences.push_back(state.mapping);
      if (state.active.size() > witness.maximumConcurrentRegions)
        return invalid("resource-time state exceeds the concurrency bound");

      const ResourceTimeRegionExecution *boundaryExecution = nullptr;
      bool boundaryIsStart = false;
      for (const ResourceTimeRegionExecution &execution : scenario.executions) {
        const bool isStart =
            execution.startPicoseconds == state.timePicoseconds &&
            state.event ==
                ::dataflow::rootThreadStartEventFamily(execution.region);
        const bool isCompletion =
            execution.completionPicoseconds == state.timePicoseconds &&
            state.event ==
                ::dataflow::rootThreadCompletionEventFamily(execution.region);
        if (!isStart && !isCompletion)
          continue;
        if (boundaryExecution)
          return invalid("resource-time state event matches multiple "
                         "execution boundaries");
        boundaryExecution = &execution;
        boundaryIsStart = isStart;
      }
      if (!boundaryExecution)
        return invalid("resource-time state event is not a start or completion "
                       "boundary at its timestamp");

      std::vector<::dataflow::RootThreadLaunchRef> expectedActive;
      const bool orderedTimestamp =
          (stateOrdinal != 0 &&
           scenario.states[stateOrdinal - 1].timePicoseconds ==
               state.timePicoseconds) ||
          (stateOrdinal + 1 != scenario.states.size() &&
           scenario.states[stateOrdinal + 1].timePicoseconds ==
               state.timePicoseconds);
      const bool firstAtTimestamp =
          stateOrdinal == 0 ||
          scenario.states[stateOrdinal - 1].timePicoseconds !=
              state.timePicoseconds;
      if (orderedTimestamp) {
        if (firstAtTimestamp) {
          orderedActive.clear();
          orderedSawAdmission = false;
          for (const ResourceTimeRegionExecution &execution :
               scenario.executions)
            if (execution.startPicoseconds < state.timePicoseconds &&
                state.timePicoseconds <= execution.completionPicoseconds)
              orderedActive.push_back(execution.region);
        }
        if (boundaryIsStart) {
          if (llvm::is_contained(orderedActive, boundaryExecution->region))
            return invalid("resource-time ordered boundary starts an active "
                           "region");
          orderedActive.push_back(boundaryExecution->region);
          orderedSawAdmission = true;
        } else {
          if (orderedSawAdmission)
            return invalid("resource-time ordered timestamp completes a "
                           "region after same-time admission");
          auto active = llvm::find(orderedActive, boundaryExecution->region);
          if (active == orderedActive.end())
            return invalid("resource-time ordered boundary completes an "
                           "inactive region");
          orderedActive.erase(active);
        }
        expectedActive = orderedActive;
      } else {
        for (const ResourceTimeRegionExecution &execution :
             scenario.executions) {
          const bool active =
              execution.startPicoseconds <= state.timePicoseconds &&
              state.timePicoseconds < execution.completionPicoseconds;
          if (active)
            expectedActive.push_back(execution.region);
        }
      }
      std::vector<::dataflow::RootThreadLaunchRef> observedActive;
      std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>
          usedResources;
      for (const ResourceTimeRegionAllocation &allocation : state.active) {
        if (!hasRegion(allocation.region))
          return invalid("resource-time state contains a foreign region");
        if (!findExecution(allocation.region))
          return invalid("resource-time state has no matching execution");
        if (allocation.resources.empty())
          return invalid("resource-time state has an unallocated region");
        if (llvm::is_contained(observedActive, allocation.region))
          return invalid("resource-time state contains a duplicate region");
        observedActive.push_back(allocation.region);
        for (const auto &resource : allocation.resources) {
          if (llvm::is_contained(usedResources, resource))
            return invalid("resource-time state assigns one physical resource "
                           "to multiple regions");
          usedResources.push_back(resource);
        }
      }
      llvm::sort(expectedActive, rootLess);
      llvm::sort(observedActive, rootLess);
      if (expectedActive != observedActive)
        return invalid("resource-time state active set disagrees with "
                       "execution intervals in scenario " +
                       llvm::Twine(scenarioOrdinal) + " at time " +
                       llvm::Twine(state.timePicoseconds) + " (expected " +
                       llvm::Twine(expectedActive.size()) + ", observed " +
                       llvm::Twine(observedActive.size()) + ")");
      const bool lastAtTimestamp =
          stateOrdinal + 1 == scenario.states.size() ||
          scenario.states[stateOrdinal + 1].timePicoseconds !=
              state.timePicoseconds;
      if (lastAtTimestamp && orderedTimestamp) {
        std::vector<::dataflow::RootThreadLaunchRef> rightOpenActive;
        for (const ResourceTimeRegionExecution &execution : scenario.executions)
          if (execution.startPicoseconds <= state.timePicoseconds &&
              state.timePicoseconds < execution.completionPicoseconds)
            rightOpenActive.push_back(execution.region);
        auto orderedFinal = orderedActive;
        llvm::sort(orderedFinal, rootLess);
        llvm::sort(rightOpenActive, rootLess);
        if (orderedFinal != rightOpenActive)
          return invalid("resource-time ordered timestamp does not close to "
                         "the right-open execution state");
      }
    }
    for (const ResourceTimeRegionExecution &execution : scenario.executions) {
      const std::uint64_t starts = llvm::count_if(
          scenario.states, [&](const ResourceTimeScheduleState &state) {
            return state.timePicoseconds == execution.startPicoseconds &&
                   state.event ==
                       ::dataflow::rootThreadStartEventFamily(execution.region);
          });
      const std::uint64_t completions = llvm::count_if(
          scenario.states, [&](const ResourceTimeScheduleState &state) {
            return state.timePicoseconds == execution.completionPicoseconds &&
                   state.event == ::dataflow::rootThreadCompletionEventFamily(
                                      execution.region);
          });
      if (starts != 1 || completions != 1)
        return invalid("resource-time schedule omits or repeats an execution "
                       "boundary");
    }
    if (scenario.makespanPicoseconds < maximumCompletion)
      return invalid("resource-time makespan precedes execution completion");

    if (mappingReferences.size() > 1 &&
        scenario.transitions.transitions.empty())
      return invalid("resource-time mapping change has no transition sequence");
    if (llvm::Error error =
            validateResourceTimeTransitionSequence(scenario.transitions))
      return error;
    std::vector<std::pair<const ResourceTimeScheduleState *,
                          const ResourceTimeScheduleState *>>
        mappingChanges;
    for (std::size_t index = 1; index != scenario.states.size(); ++index)
      if (scenario.states[index - 1].mapping != scenario.states[index].mapping)
        mappingChanges.emplace_back(&scenario.states[index - 1],
                                    &scenario.states[index]);
    if (mappingChanges.size() != scenario.transitions.transitions.size())
      return invalid("resource-time Mapping changes do not match the finite "
                     "transition sequence");
    for (auto paired :
         llvm::zip(mappingChanges, scenario.transitions.transitions)) {
      const auto &change = std::get<0>(paired);
      const ResourceTimeTransition &transition = std::get<1>(paired);
      if (!llvm::is_contained(mappingReferences, transition.parent.mapping) ||
          !llvm::is_contained(mappingReferences, transition.child.mapping))
        return invalid("resource-time transition is absent from its schedule "
                       "states");
      if (transition.parent.mapping != change.first->mapping ||
          transition.child.mapping != change.second->mapping)
        return invalid("resource-time transition endpoints disagree with "
                       "their adjacent schedule states");
      if (transition.trigger != change.second->event)
        return invalid("resource-time transition trigger disagrees with its "
                       "child schedule event");
      if (!allocationsEquivalent(transition.beforeActive,
                                 change.first->active) ||
          !allocationsEquivalent(transition.afterActive, change.second->active))
        return invalid("resource-time transition allocation evidence "
                       "disagrees with its adjacent schedule states");
    }
  }
  return llvm::Error::success();
}

} // namespace loom::pnr
