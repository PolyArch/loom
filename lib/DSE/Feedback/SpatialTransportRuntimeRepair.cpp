#include "DSE/JointHardwareReopen.h"

#include "../JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialMappingWarmSeed.h"
#include "PnR/SpatialPnrProblem.h"
#include "Runtime/Gem5SystemExecution.h"

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

std::optional<std::chrono::steady_clock::time_point>
steadyDeadline(const PlanExecutionPolicy &policy) {
  const auto deadline = policy.dispatchNotAfterUnixNanoseconds();
  if (!deadline)
    return std::nullopt;
  const auto systemNow = std::chrono::system_clock::now().time_since_epoch();
  const auto nowNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(systemNow).count();
  const auto steadyNow = std::chrono::steady_clock::now();
  if (nowNanoseconds < 0 ||
      static_cast<std::uint64_t>(nowNanoseconds) >= *deadline)
    return steadyNow;
  return steadyNow + std::chrono::nanoseconds(
                         *deadline - static_cast<std::uint64_t>(nowNanoseconds));
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
  if (!feedback.parentMapping || !feedback.parentSpatialMapping ||
      !feedback.parentConstraints || !feedback.constraintSet ||
      !feedback.owners)
    return invalid("exact feedback has no immutable Mapping owners");
  const std::optional<ArtifactRootReference> &parentCandidate =
      parentExecution.summary.selectedMapping
          ? parentExecution.summary.selectedMapping
          : parentExecution.summary.qualityIncompleteCandidate;
  if (!parentCandidate || *parentCandidate != *feedback.parentMapping)
    return invalid("feedback does not name the parent quality candidate");
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
  const std::uint64_t feedbackProbeLimit =
      request.boundedQuality
          ? request.boundedQuality->maximumHardwareRepairProbes
          : 8;
  result.candidateLimit = std::min<std::uint64_t>(
      feedbackProbeLimit, policy.maximumSpatialMappingsPerPair());
  if (deadlineReached(request.executionPolicy)) {
    result.candidatesCancelled = result.candidateLimit;
    result.exactRepair = pnr::SpatialExactRepairResult{
        pnr::SpatialExactRepairResultKind::TimedOut, 0, 0, 0, 0, 0,
        "local runtime-counterexample repair started after its deadline"};
    return result;
  }
  auto spatialConfig =
      pnr::projectResolvedSpatialPnrConfigView(parentPlan.resolvedConfig);
  if (!spatialConfig)
    return spatialConfig.takeError();
  auto physicalTiming =
      fabric::projectNormalizedFabricPhysicalTimingProfile(fabric->view());
  if (!physicalTiming)
    return physicalTiming.takeError();
  if (!feedback.runtimeEvidence)
    return invalid("exact feedback has no replay-verified Evidence root");
  auto verified =
      evaluation::models::importVerifiedCgraClosedWaitEvidence(
          *feedback.runtimeEvidence, artifacts, blobs);
  if (!verified)
    return verified.takeError();
  const SpatialTransportCegarPolicy cegarPolicy{
      result.candidateLimit,
      result.candidateLimit,
      spatialConfig->policy().search.exactRepair.maxSolverCalls,
      runtime::gem5MaximumSpatialWork,
      steadyDeadline(request.executionPolicy)};
  auto cegar = executeSpatialTransportCegar(
      *feedback.parentSpatialMapping, *feedback.parentConstraints, *verified,
      parentPlan.resolvedConfig, *physicalTiming, cegarPolicy, artifacts,
      blobs);
  if (!cegar)
    return cegar.takeError();
  if (cegar->iterations.empty()) {
    if (cegar->termination == SpatialTransportCegarTermination::TimedOut) {
      result.candidatesCancelled = result.candidateLimit;
      result.cegar = std::move(*cegar);
      return result;
    }
    return invalid("CEGAR produced no iteration or typed timeout");
  }
  if (cegar->iterations.front().accumulatedConstraints !=
      *feedback.constraintSet)
    return invalid("CEGAR replay did not reproduce the supplied promotion");
  result.candidatesPlanned = cegar->iterations.size();
  result.candidatesReserved = result.candidatesPlanned;
  for (const SpatialTransportCegarIteration &iteration : cegar->iterations) {
    result.constraintSets.push_back(iteration.accumulatedConstraints);
    result.warmSeedAccounting = iteration.warmSeed;
    result.exactRepair = iteration.repair;
    if (iteration.childMapping) {
      result.repairedSpatialMappings.push_back(*iteration.childMapping);
      ++result.candidatesConsumed;
    } else {
      ++result.candidatesRejected;
    }
  }
  result.cegar = std::move(*cegar);
  if (result.cegar->termination != SpatialTransportCegarTermination::Retired ||
      !result.cegar->finalMapping || !result.cegar->finalConstraints)
    return result;
  const ArtifactRootReference childSpatial = *result.cegar->finalMapping;
  const ArtifactRootReference childConstraints =
      *result.cegar->finalConstraints;

  auto repairPolicy = JointDesignPolicy::get(1, 1, 1, 1, 1);
  if (!repairPolicy)
    return repairPolicy.takeError();
  ResolvedConfig config = parentPlan.resolvedConfig;
  config.dse.planNodes.clear();
  JointDesignMappingSeed seed;
  seed.techMappings.push_back(feedback.owners->techMapping);
  seed.spatialMappings.push_back(childSpatial);
  seed.spatialRepairConstraints.push_back(
      {feedback.owners->techMapping, childConstraints});
  auto plan = buildJointDesignExplorationPlan(
      {{parentPair->software.workloads}, {parentPair->system}}, *timing,
      *repairPolicy, config, artifacts, &seed,
      parentPlan.systemBindingPartitions);
  if (!plan)
    return plan.takeError();
  result.preparedSeedHandoff = true;
  llvm::SmallString<256> journal(request.journalRoot);
  llvm::sys::path::append(journal, "spatial-transport-local");
  request.journalRoot = journal.str().str();
  auto execution = executeJointRepairPlan(*plan, *repairPolicy,
                                          std::move(request), artifacts, blobs);
  if (!execution)
    return execution.takeError();
  result.childSystems.push_back(parentPair->system);
  result.reuseDispositions.push_back(JointMappingReuseDisposition::LocalRepair);
  result.executions.push_back(std::move(*execution));
  if (result.candidatesReserved != result.candidatesConsumed +
                                       result.candidatesRejected +
                                       result.candidatesCancelled)
    return invalid("transport repair work ledger did not close");
  return result;
}

} // namespace loom::dse
