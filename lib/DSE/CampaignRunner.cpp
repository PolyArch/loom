#include "DSE/CampaignRunner.h"

#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Evidence.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral kEvaluationRegistryIdentity =
    "loom.evaluation.registry";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_campaign_invalid: " + message);
}

bool terminal(JournalWorkUnitStatus status) {
  return status == JournalWorkUnitStatus::Completed ||
         status == JournalWorkUnitStatus::Failed ||
         status == JournalWorkUnitStatus::TimedOut ||
         status == JournalWorkUnitStatus::Unsupported;
}

llvm::Expected<std::uint64_t> unixNanosecondsNow() {
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (nanoseconds <= 0)
    return invalid(
        "system clock cannot represent a positive dispatch deadline");
  return static_cast<std::uint64_t>(nanoseconds);
}

llvm::Error add(std::uint64_t &value, std::uint64_t amount,
                llvm::StringRef field) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return invalid(field + " overflows uint64");
  value += amount;
  return llvm::Error::success();
}

using Interval = std::pair<std::uint64_t, std::uint64_t>;

llvm::Expected<std::uint64_t>
intervalUnionNanoseconds(std::vector<Interval> intervals) {
  if (intervals.empty())
    return 0;
  llvm::sort(intervals);
  std::uint64_t total = 0;
  std::uint64_t begin = intervals.front().first;
  std::uint64_t end = intervals.front().second;
  for (std::size_t index = 1; index != intervals.size(); ++index) {
    const Interval &interval = intervals[index];
    if (interval.first <= end) {
      end = std::max(end, interval.second);
      continue;
    }
    if (llvm::Error error = add(total, end - begin, "active interval union"))
      return std::move(error);
    begin = interval.first;
    end = interval.second;
  }
  if (llvm::Error error = add(total, end - begin, "active interval union"))
    return std::move(error);
  return total;
}

llvm::ArrayRef<PlanInputBinding>
inputBindings(const ResolvedDsePlanNode &node) {
  return std::visit(
      [](const auto &resolved) -> llvm::ArrayRef<PlanInputBinding> {
        return resolved.inputBindings();
      },
      node);
}

llvm::Expected<std::vector<std::vector<std::uint64_t>>>
ancestorNodes(const ResolvedDsePlan &plan) {
  std::vector<std::vector<std::uint64_t>> ancestors(plan.nodes().size());
  for (std::size_t nodeOrdinal = 0; nodeOrdinal != plan.nodes().size();
       ++nodeOrdinal) {
    std::vector<std::uint64_t> &nodeAncestors = ancestors[nodeOrdinal];
    for (const PlanInputBinding &binding :
         inputBindings(plan.nodes()[nodeOrdinal])) {
      std::vector<PlanOutputRef> outputs;
      if (const auto *output = std::get_if<PlanOutputRef>(&binding))
        outputs.push_back(*output);
      else if (const auto *join = std::get_if<BoundedPlanOutputJoin>(&binding))
        outputs = join->outputs;
      for (PlanOutputRef output : outputs) {
        if (output.producerNodeOrdinal >= nodeOrdinal)
          return invalid("resolved plan contains a non-prior use-def edge");
        nodeAncestors.push_back(output.producerNodeOrdinal);
        const std::vector<std::uint64_t> &transitive =
            ancestors[output.producerNodeOrdinal];
        nodeAncestors.insert(nodeAncestors.end(), transitive.begin(),
                             transitive.end());
      }
    }
    llvm::sort(nodeAncestors);
    nodeAncestors.erase(std::unique(nodeAncestors.begin(), nodeAncestors.end()),
                        nodeAncestors.end());
  }
  return ancestors;
}

bool containsNode(llvm::ArrayRef<std::uint64_t> nodes,
                  std::uint64_t candidate) {
  return llvm::binary_search(nodes, candidate);
}

llvm::Expected<std::uint64_t>
campaignActiveNanoseconds(llvm::ArrayRef<JournalWorkUnitRecord> records) {
  std::vector<Interval> intervals;
  for (const JournalWorkUnitRecord &record : records)
    for (const JournalActiveWallInterval &interval : record.activeWallIntervals)
      intervals.push_back(
          {interval.beginUnixTimeNanoseconds, interval.endUnixTimeNanoseconds});
  return intervalUnionNanoseconds(std::move(intervals));
}

llvm::Expected<std::uint64_t>
maximumSampleActiveNanoseconds(const ResolvedDsePlan &plan,
                               llvm::ArrayRef<JournalWorkUnitRecord> records) {
  auto ancestors = ancestorNodes(plan);
  if (!ancestors)
    return ancestors.takeError();
  std::uint64_t maximum = 0;
  for (const JournalWorkUnitRecord &sample : records) {
    if (sample.status != JournalWorkUnitStatus::Completed ||
        sample.key.descriptor().ownerRegistryIdentity() !=
            kEvaluationRegistryIdentity)
      continue;
    if (sample.key.planNodeOrdinal() >= plan.nodes().size())
      return invalid("Evaluation work references an unknown plan node");
    if (sample.finalizedOutputs.size() != 1 ||
        sample.finalizedOutputs.front().schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        sample.finalizedOutputs.front().schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      continue;

    std::vector<Interval> intervals;
    for (const JournalWorkUnitRecord &record : records) {
      const bool sameWork = record.key == sample.key;
      const bool ancestor =
          record.key.planNodeOrdinal() < sample.key.planNodeOrdinal() &&
          containsNode((*ancestors)[sample.key.planNodeOrdinal()],
                       record.key.planNodeOrdinal());
      if (!sameWork && !ancestor)
        continue;
      for (const JournalActiveWallInterval &interval :
           record.activeWallIntervals)
        intervals.push_back({interval.beginUnixTimeNanoseconds,
                             interval.endUnixTimeNanoseconds});
    }
    auto active = intervalUnionNanoseconds(std::move(intervals));
    if (!active)
      return active.takeError();
    maximum = std::max(maximum, *active);
  }
  return maximum;
}

std::uint64_t
terminalObservationCount(llvm::ArrayRef<JournalWorkUnitRecord> records) {
  return static_cast<std::uint64_t>(
      llvm::count_if(records, [](const JournalWorkUnitRecord &record) {
        return terminal(record.status);
      }));
}

bool hasPreparedAttempt(llvm::ArrayRef<JournalWorkUnitRecord> records) {
  return llvm::any_of(records, [](const JournalWorkUnitRecord &record) {
    return record.status == JournalWorkUnitStatus::Prepared;
  });
}

llvm::Expected<std::optional<std::uint64_t>>
exactCandidateCount(const PlanInputBinding &binding,
                    const CompletedDsePlanExecution &completedPrefix) {
  if (const auto *exact = std::get_if<ExactPlanArtifacts>(&binding))
    return static_cast<std::uint64_t>(exact->artifacts.size());
  if (const auto *output = std::get_if<PlanOutputRef>(&binding)) {
    if (!completedPrefix.hasOutput(*output))
      return std::optional<std::uint64_t>{};
    return static_cast<std::uint64_t>(completedPrefix.resolve(*output).size());
  }
  const auto &join = std::get<BoundedPlanOutputJoin>(binding);
  std::vector<ArtifactRootReference> artifacts = join.exactArtifacts;
  for (PlanOutputRef output : join.outputs) {
    if (!completedPrefix.hasOutput(output))
      return std::optional<std::uint64_t>{};
    llvm::ArrayRef<ArtifactRootReference> source =
        completedPrefix.resolve(output);
    artifacts.insert(artifacts.end(), source.begin(), source.end());
  }
  llvm::sort(artifacts, artifactRootReferenceLess);
  artifacts.erase(std::unique(artifacts.begin(), artifacts.end()),
                  artifacts.end());
  return std::min<std::uint64_t>(artifacts.size(), join.maximumArtifacts);
}

llvm::Expected<std::optional<std::uint64_t>>
remainingPlanWorkUnits(const ResolvedDsePlan &plan,
                       const IncompleteDsePlanExecution &incomplete,
                       llvm::ArrayRef<JournalWorkUnitRecord> records) {
  if (!incomplete.executionStopped())
    return std::uint64_t{0};
  std::uint64_t total = 0;
  for (std::uint64_t ordinal = incomplete.nodeOrdinal();
       ordinal < plan.nodes().size(); ++ordinal) {
    const ResolvedDsePlanNode &node = plan.nodes()[ordinal];
    if (std::holds_alternative<ResolvedGeneratePlanNode>(node)) {
      if (llvm::Error error = add(total, 1, "remaining Generate work"))
        return std::move(error);
      continue;
    }
    const ResolvedPromotePlanNode &promote =
        std::get<ResolvedPromotePlanNode>(node);
    const PromotionAcquisitionDescriptor *descriptor =
        promote.acquisitionRef().descriptor();
    if (!descriptor || descriptor->candidateInputSlot.ordinal() >=
                           promote.inputBindings().size())
      return invalid("remaining Promote node lost its candidate input");
    auto candidates = exactCandidateCount(
        promote.inputBindings()[descriptor->candidateInputSlot.ordinal()],
        incomplete.availableExecution());
    if (!candidates)
      return candidates.takeError();
    if (!*candidates)
      return std::optional<std::uint64_t>{};
    const std::uint64_t obligations =
        promote.acquisitionBinding().evidenceObligations().size();
    if (obligations != 0 &&
        **candidates > std::numeric_limits<std::uint64_t>::max() / obligations)
      return invalid("remaining Promote work overflows uint64");
    if (llvm::Error error =
            add(total, **candidates * obligations, "remaining Promote work"))
      return std::move(error);
  }

  std::uint64_t terminalRemaining = 0;
  for (const JournalWorkUnitRecord &record : records)
    if (record.key.planNodeOrdinal() >= incomplete.nodeOrdinal() &&
        terminal(record.status))
      if (llvm::Error error =
              add(terminalRemaining, 1, "remaining terminal work"))
        return std::move(error);
  if (terminalRemaining > total)
    return invalid("Journal terminal work exceeds the remaining plan bound");
  return total - terminalRemaining;
}

llvm::Expected<std::optional<std::uint64_t>>
conservativeRemainingEstimate(const ResolvedDsePlan &plan,
                              const DsePlanExecutionOutcome &outcome,
                              llvm::ArrayRef<JournalWorkUnitRecord> records,
                              const DseOperationalProjection &projection) {
  const auto *incomplete = std::get_if<IncompleteDsePlanExecution>(&outcome);
  if (!incomplete)
    return std::uint64_t{0};
  auto remaining = remainingPlanWorkUnits(plan, *incomplete, records);
  if (!remaining)
    return remaining.takeError();
  if (!*remaining)
    return std::optional<std::uint64_t>{};
  if (**remaining == 0)
    return std::uint64_t{0};
  std::uint64_t maximumP90 = 0;
  for (const WorkUnitDurationProjection &duration : projection.durations)
    maximumP90 = std::max(maximumP90, duration.p90Nanoseconds);
  if (maximumP90 == 0)
    return std::optional<std::uint64_t>{};
  if (**remaining > std::numeric_limits<std::uint64_t>::max() / maximumP90)
    return invalid("remaining campaign estimate overflows uint64");
  return **remaining * maximumP90;
}

llvm::Expected<PlanExecutionPolicy>
pilotPolicy(const PlanExecutionPolicy &base,
            const CampaignExecutionPolicy &campaign) {
  std::uint64_t dispatches = campaign.pilotDispatchCount();
  if (base.maximumDispatches())
    dispatches = std::min(dispatches, *base.maximumDispatches());
  return PlanExecutionPolicy::get(base.workerCount(), base.inProcessClaim(),
                                  base.externalSite(), base.resourceBindings(),
                                  dispatches,
                                  base.dispatchNotAfterUnixNanoseconds());
}

llvm::Expected<PlanExecutionPolicy>
admittedPolicy(const PlanExecutionPolicy &base, std::uint64_t campaignActive,
               const CampaignExecutionPolicy &campaign) {
  if (campaignActive >= campaign.campaignActiveWallTimeLimitNanoseconds())
    return invalid("campaign has no remaining active-time budget");
  const std::uint64_t remaining =
      campaign.campaignActiveWallTimeLimitNanoseconds() - campaignActive;
  auto now = unixNanosecondsNow();
  if (!now)
    return now.takeError();
  if (remaining > std::numeric_limits<std::uint64_t>::max() - *now)
    return invalid("campaign dispatch deadline overflows uint64");
  std::uint64_t deadline = *now + remaining;
  if (base.dispatchNotAfterUnixNanoseconds())
    deadline = std::min(deadline, *base.dispatchNotAfterUnixNanoseconds());
  return PlanExecutionPolicy::get(base.workerCount(), base.inProcessClaim(),
                                  base.externalSite(), base.resourceBindings(),
                                  base.maximumDispatches(), deadline);
}

CampaignExecutionResult refuse(CampaignAdmissionFailureReason reason,
                               DsePlanExecutionOutcome outcome,
                               DseOperationalProjection projection) {
  return CampaignExecutionResult{CampaignAdmissionRefusal{
      reason, std::move(outcome), std::move(projection)}};
}

llvm::Expected<std::optional<CampaignAdmissionFailureReason>>
validateObservedLimits(const ResolvedDsePlan &plan,
                       llvm::ArrayRef<JournalWorkUnitRecord> records,
                       const CampaignExecutionPolicy &policy) {
  if (hasPreparedAttempt(records))
    return std::optional<CampaignAdmissionFailureReason>(
        CampaignAdmissionFailureReason::PreparedAttemptIncomplete);
  auto sampleActive = maximumSampleActiveNanoseconds(plan, records);
  if (!sampleActive)
    return sampleActive.takeError();
  if (*sampleActive > policy.sampleActiveWallTimeLimitNanoseconds())
    return std::optional<CampaignAdmissionFailureReason>(
        CampaignAdmissionFailureReason::SampleActiveWallTimeLimit);
  auto campaignActive = campaignActiveNanoseconds(records);
  if (!campaignActive)
    return campaignActive.takeError();
  if (*campaignActive > policy.campaignActiveWallTimeLimitNanoseconds())
    return std::optional<CampaignAdmissionFailureReason>(
        CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit);
  return std::optional<CampaignAdmissionFailureReason>{};
}

} // namespace

llvm::Expected<CampaignExecutionPolicy> CampaignExecutionPolicy::get(
    std::uint64_t pilotDispatchCount,
    std::uint64_t minimumObservedPilotWorkUnits,
    std::uint64_t sampleActiveWallTimeLimitNanoseconds,
    std::uint64_t campaignActiveWallTimeLimitNanoseconds) {
  if (pilotDispatchCount == 0)
    return invalid("pilot dispatch count must be positive");
  if (minimumObservedPilotWorkUnits == 0 ||
      minimumObservedPilotWorkUnits > pilotDispatchCount)
    return invalid(
        "minimum pilot observations must be positive and no larger than "
        "the pilot dispatch count");
  if (sampleActiveWallTimeLimitNanoseconds == 0 ||
      sampleActiveWallTimeLimitNanoseconds >
          maximumSampleActiveWallTimeNanoseconds)
    return invalid("sample active-time limit exceeds its configured bound");
  if (campaignActiveWallTimeLimitNanoseconds == 0 ||
      campaignActiveWallTimeLimitNanoseconds >
          maximumCampaignActiveWallTimeNanoseconds)
    return invalid("campaign active-time limit exceeds its configured bound");
  return CampaignExecutionPolicy(pilotDispatchCount,
                                 minimumObservedPilotWorkUnits,
                                 sampleActiveWallTimeLimitNanoseconds,
                                 campaignActiveWallTimeLimitNanoseconds);
}

llvm::Expected<CampaignExecutionResult>
runGroundTruthCampaign(const ResolvedDseConfigView &view,
                       const DseRunClosure &closure,
                       const CampaignExecutionPolicy &campaignPolicy,
                       const PlanExecutionPolicy &executionPolicy,
                       SiteScheduler &scheduler, ExecutionJournal &journal,
                       const ArtifactStore &store, const BlobStore &blobs) {
  auto pilotExecutionPolicy = pilotPolicy(executionPolicy, campaignPolicy);
  if (!pilotExecutionPolicy)
    return pilotExecutionPolicy.takeError();
  auto pilot =
      resumeDsePlan(view, closure, journal, scheduler, *pilotExecutionPolicy,
                    store, blobs, InvocationManifestRetention::Retain);
  if (!pilot)
    return pilot.takeError();
  const auto releaseWith = [&](llvm::Error error) {
    return llvm::joinErrors(std::move(error),
                            journal.releaseInvocationOccurrence());
  };
  auto records = journal.workUnits();
  if (!records)
    return releaseWith(records.takeError());
  auto projection = projectDseOperationalState(journal, scheduler,
                                               executionPolicy.workerCount());
  if (!projection)
    return releaseWith(projection.takeError());

  auto limitFailure =
      validateObservedLimits(view.plan(), *records, campaignPolicy);
  if (!limitFailure)
    return releaseWith(limitFailure.takeError());
  if (*limitFailure)
    return refuse(**limitFailure, std::move(*pilot), std::move(*projection));

  if (std::holds_alternative<CompletedDsePlanExecution>(*pilot))
    return CampaignExecutionResult{
        CampaignExecution{std::move(*pilot), std::move(*projection)}};

  if (terminalObservationCount(*records) <
      campaignPolicy.minimumObservedPilotWorkUnits())
    return refuse(CampaignAdmissionFailureReason::InsufficientPilotObservations,
                  std::move(*pilot), std::move(*projection));

  if (!projection->estimatedRemainingNanoseconds) {
    auto estimate = conservativeRemainingEstimate(view.plan(), *pilot, *records,
                                                  *projection);
    if (!estimate)
      return releaseWith(estimate.takeError());
    projection->estimatedRemainingNanoseconds = *estimate;
  }

  auto active = campaignActiveNanoseconds(*records);
  if (!active)
    return releaseWith(active.takeError());
  if (!projection->estimatedRemainingNanoseconds) {
    return refuse(CampaignAdmissionFailureReason::ThroughputUnavailable,
                  std::move(*pilot), std::move(*projection));
  }
  if (*projection->estimatedRemainingNanoseconds >
      campaignPolicy.campaignActiveWallTimeLimitNanoseconds() - *active)
    return refuse(CampaignAdmissionFailureReason::EstimatedCompletionLimit,
                  std::move(*pilot), std::move(*projection));

  auto fullExecutionPolicy =
      admittedPolicy(executionPolicy, *active, campaignPolicy);
  if (!fullExecutionPolicy)
    return releaseWith(fullExecutionPolicy.takeError());
  auto outcome = executeDsePlan(view, closure, journal, scheduler,
                                *fullExecutionPolicy, store, blobs);
  if (!outcome)
    return releaseWith(outcome.takeError());
  records = journal.workUnits();
  if (!records)
    return releaseWith(records.takeError());
  projection = projectDseOperationalState(journal, scheduler,
                                          executionPolicy.workerCount());
  if (!projection)
    return releaseWith(projection.takeError());
  limitFailure = validateObservedLimits(view.plan(), *records, campaignPolicy);
  if (!limitFailure)
    return releaseWith(limitFailure.takeError());
  if (*limitFailure)
    return refuse(**limitFailure, std::move(*outcome), std::move(*projection));
  return CampaignExecutionResult{
      CampaignExecution{std::move(*outcome), std::move(*projection)}};
}

} // namespace loom::dse
