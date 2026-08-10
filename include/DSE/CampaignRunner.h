#ifndef LOOM_DSE_CAMPAIGNRUNNER_H
#define LOOM_DSE_CAMPAIGNRUNNER_H

#include "DSE/PlanExecutor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::dse {

struct OperationalStatusCounts final {
  std::uint64_t completed = 0;
  std::uint64_t running = 0;
  std::uint64_t prepared = 0;
  std::uint64_t queued = 0;
  std::uint64_t failed = 0;
  std::uint64_t timedOut = 0;
  std::uint64_t unsupported = 0;
};

struct WorkUnitDurationProjection final {
  WorkUnitDescriptorRef descriptor;
  std::uint64_t terminalCount = 0;
  std::uint64_t p50Nanoseconds = 0;
  std::uint64_t p90Nanoseconds = 0;
};

struct LimitingSiteResource final {
  SiteResourceKind kind;
  std::optional<SiteResourceKey> key;
  std::uint64_t allocated = 0;
  std::uint64_t queuedDemand = 0;
  std::uint64_t capacity = 0;
};

struct DseOperationalProjection final {
  std::uint64_t observedUnixTimeNanoseconds = 0;
  OperationalStatusCounts status;
  double recentThroughputPerSecond = 0.0;
  std::vector<WorkUnitDurationProjection> durations;
  std::optional<std::uint64_t> estimatedRemainingNanoseconds;
  std::optional<LimitingSiteResource> limitingResource;
};

llvm::Expected<DseOperationalProjection> projectDseOperationalState(
    const ExecutionJournal &journal, const SiteScheduler &scheduler,
    std::uint64_t requestedWorkerCount,
    std::uint64_t recentWindowNanoseconds = 60ULL * 1000ULL * 1000ULL *
                                             1000ULL);

llvm::Error writeDseOperationalProjectionJsonLine(
    const DseOperationalProjection &projection, llvm::raw_ostream &output);

class CampaignExecutionPolicy final {
public:
  static constexpr std::uint64_t maximumSampleActiveWallTimeNanoseconds =
      600ULL * 1000ULL * 1000ULL * 1000ULL;
  static constexpr std::uint64_t maximumCampaignActiveWallTimeNanoseconds =
      23ULL * 60ULL * 60ULL * 1000ULL * 1000ULL * 1000ULL;

  static llvm::Expected<CampaignExecutionPolicy>
  get(std::uint64_t pilotDispatchCount,
      std::uint64_t minimumObservedPilotWorkUnits,
      std::uint64_t sampleActiveWallTimeLimitNanoseconds =
          maximumSampleActiveWallTimeNanoseconds,
      std::uint64_t campaignActiveWallTimeLimitNanoseconds =
          maximumCampaignActiveWallTimeNanoseconds);

  std::uint64_t pilotDispatchCount() const { return pilotDispatchCount_; }
  std::uint64_t minimumObservedPilotWorkUnits() const {
    return minimumObservedPilotWorkUnits_;
  }
  std::uint64_t sampleActiveWallTimeLimitNanoseconds() const {
    return sampleActiveWallTimeLimitNanoseconds_;
  }
  std::uint64_t campaignActiveWallTimeLimitNanoseconds() const {
    return campaignActiveWallTimeLimitNanoseconds_;
  }

private:
  CampaignExecutionPolicy(
      std::uint64_t pilotDispatchCount,
      std::uint64_t minimumObservedPilotWorkUnits,
      std::uint64_t sampleActiveWallTimeLimitNanoseconds,
      std::uint64_t campaignActiveWallTimeLimitNanoseconds)
      : pilotDispatchCount_(pilotDispatchCount),
        minimumObservedPilotWorkUnits_(minimumObservedPilotWorkUnits),
        sampleActiveWallTimeLimitNanoseconds_(
            sampleActiveWallTimeLimitNanoseconds),
        campaignActiveWallTimeLimitNanoseconds_(
            campaignActiveWallTimeLimitNanoseconds) {}

  std::uint64_t pilotDispatchCount_ = 0;
  std::uint64_t minimumObservedPilotWorkUnits_ = 0;
  std::uint64_t sampleActiveWallTimeLimitNanoseconds_ = 0;
  std::uint64_t campaignActiveWallTimeLimitNanoseconds_ = 0;
};

enum class CampaignAdmissionFailureReason : std::uint32_t {
  InsufficientPilotObservations = 0,
  PreparedAttemptIncomplete = 1,
  SampleActiveWallTimeLimit = 2,
  CampaignActiveWallTimeLimit = 3,
  EstimatedCompletionLimit = 4,
  ThroughputUnavailable = 5,
};

struct CampaignExecution final {
  DsePlanExecutionOutcome outcome;
  DseOperationalProjection projection;
};

struct CampaignAdmissionRefusal final {
  CampaignAdmissionFailureReason reason;
  DsePlanExecutionOutcome outcome;
  DseOperationalProjection projection;
};

using CampaignExecutionResult =
    std::variant<CampaignExecution, CampaignAdmissionRefusal>;

llvm::Expected<CampaignExecutionResult> runGroundTruthCampaign(
    const ResolvedDseConfigView &view, const DseRunClosure &closure,
    const CampaignExecutionPolicy &campaignPolicy,
    const PlanExecutionPolicy &executionPolicy, SiteScheduler &scheduler,
    ExecutionJournal &journal, const ArtifactStore &store,
    const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_CAMPAIGNRUNNER_H
