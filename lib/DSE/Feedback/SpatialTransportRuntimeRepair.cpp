#include "DSE/JointHardwareReopen.h"

#include "../JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
#include <system_error>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_transport_runtime_repair_invalid: " + message);
}

bool deadlineReached(const PlanExecutionPolicy &policy) {
  const auto deadline = policy.dispatchNotAfterUnixNanoseconds();
  if (!deadline)
    return false;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  if (now.count() < 0)
    return false;
  return static_cast<std::uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                 .count()) >= *deadline;
}

} // namespace

llvm::Expected<JointSpatialTransportMappingRepair>
executeSpatialTransportRuntimeRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    const SpatialTransportRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  JointSpatialTransportMappingRepair result;
  result.feedback = feedback;
  if (feedback.disposition != SpatialTransportRuntimeFeedbackDisposition::Exact)
    return result;
  if (!feedback.parentMapping || !feedback.owners)
    return invalid("exact feedback has no immutable Mapping owners");
  if (!parentExecution.summary.selectedMapping ||
      *parentExecution.summary.selectedMapping != *feedback.parentMapping)
    return invalid("feedback does not name the selected parent Mapping");
  if (feedback.alternatives.empty())
    return invalid("exact feedback has no repair alternative");

  auto parentMapping =
      mapping::importSystemMapping(*feedback.parentMapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  const JointDesignPair *parentPair = nullptr;
  for (const JointDesignPlanPair &candidate : parentPlan.pairOutputs) {
    if (candidate.pair.software.dataflow.artifact !=
            parentMapping->view().dataflowIdentity() ||
        candidate.pair.system.artifact !=
            parentMapping->view().fabricIdentity())
      continue;
    if (parentPair)
      return invalid("selected Mapping matches more than one application pair");
    parentPair = &candidate.pair;
  }
  if (!parentPair)
    return invalid(
        "selected Mapping does not match an authored application pair");
  const auto mappedPair = llvm::find_if(
      parentExecution.mappedPairs, [&](const JointMappedPair &candidate) {
        return candidate.pair == *parentPair &&
               llvm::is_contained(candidate.systemMappings,
                                  *feedback.parentMapping);
      });
  if (mappedPair == parentExecution.mappedPairs.end())
    return invalid("selected Mapping is absent from its parent pair outcome");

  auto dataflow =
      ::dataflow::importCanonicalDataflow(feedback.owners->dataflow, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto tech =
      mapping::importTechMapping(feedback.owners->techMapping, artifacts);
  if (!tech)
    return tech.takeError();
  auto fabric = ::loom::fabric::importEntireFabricRoot(feedback.owners->fabric,
                                                       artifacts);
  if (!fabric)
    return fabric.takeError();
  auto timing = normalizedTimingProfiles(parentPair->system, artifacts);
  if (!timing)
    return timing.takeError();

  auto repairPolicy = JointDesignPolicy::get(1, 1, 1, 1, 1);
  if (!repairPolicy)
    return repairPolicy.takeError();
  const std::uint64_t feedbackProbeLimit =
      request.boundedQuality
          ? request.boundedQuality->maximumHardwareRepairProbes
          : 1;
  result.candidateLimit = std::min<std::uint64_t>(
      feedback.alternatives.size(),
      std::min(policy.maximumSpatialMappingsPerPair(), feedbackProbeLimit));
  result.candidatesPlanned = result.candidateLimit;
  result.candidatesReserved = result.candidatesPlanned;
  auto scheduler = SiteScheduler::create(request.siteCapacity);
  if (!scheduler)
    return scheduler.takeError();
  for (std::size_t ordinal = 0; ordinal != result.candidateLimit; ++ordinal) {
    if (deadlineReached(request.executionPolicy)) {
      result.candidatesCancelled += result.candidateLimit - ordinal;
      break;
    }
    const SpatialTransportRepairAlternative &alternative =
        feedback.alternatives[ordinal];
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> domain;
    domain.reserve(fabric->view().admittedTraversals().size());
    for (const auto &traversal : fabric->view().admittedTraversals())
      if (traversal != alternative.forbiddenTraversal)
        domain.push_back(traversal);
    if (domain.size() == fabric->view().admittedTraversals().size()) {
      ++result.candidatesRejected;
      continue;
    }
    auto constraints = mapping::finalizeSpatialNetTraversalDomainConstraintSet(
        *dataflowView, tech->view(), fabric->view(), alternative.producer,
        domain, artifacts);
    if (!constraints)
      return constraints.takeError();
    result.constraintSets.push_back(constraints->reference());

    ResolvedConfig config = parentPlan.resolvedConfig;
    config.dse.planNodes.clear();
    JointDesignMappingSeed seed;
    seed.techMappings.push_back(feedback.owners->techMapping);
    seed.spatialRepairConstraints.push_back(
        {feedback.owners->techMapping, constraints->reference()});
    auto plan = buildJointDesignExplorationPlan(
        {{parentPair->software.workloads}, {parentPair->system}}, *timing,
        *repairPolicy, config, artifacts, &seed,
        parentPlan.systemBindingPartitions);
    if (!plan)
      return plan.takeError();
    llvm::SmallString<256> journal(request.journalRoot);
    llvm::sys::path::append(journal,
                            "spatial-transport-" + std::to_string(ordinal));
    JointHardwareReopenRequest childRequest = request;
    childRequest.journalRoot = journal.str().str();
    auto execution = executeJointPlan(*plan, request.evidence, childRequest,
                                      *scheduler, artifacts, blobs);
    if (!execution)
      return execution.takeError();
    ++result.candidatesConsumed;
    result.childSystems.push_back(parentPair->system);
    result.reuseDispositions.push_back(
        JointMappingReuseDisposition::ColdFallback);
    result.executions.push_back(std::move(*execution));
  }
  if (result.candidatesReserved != result.candidatesConsumed +
                                       result.candidatesRejected +
                                       result.candidatesCancelled)
    return invalid("transport repair work ledger did not close");
  return result;
}

} // namespace loom::dse
