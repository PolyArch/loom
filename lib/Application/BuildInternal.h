#ifndef LOOM_APPLICATION_BUILDINTERNAL_H
#define LOOM_APPLICATION_BUILDINTERNAL_H

#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
class Twine;
} // namespace llvm

namespace loom::application::build_detail {

using MonotonicClock = std::chrono::steady_clock;

void emitElapsed(ApplicationBuildOperation operation,
                 MonotonicClock::time_point begin,
                 std::uint64_t deterministicWork = 1);

class ApplicationBuildOperationTimer final {
public:
  explicit ApplicationBuildOperationTimer(ApplicationBuildOperation operation)
      : operation_(operation), begin_(MonotonicClock::now()) {}

  ~ApplicationBuildOperationTimer() { emitElapsed(operation_, begin_); }

  ApplicationBuildOperationTimer(const ApplicationBuildOperationTimer &) =
      delete;
  ApplicationBuildOperationTimer &
  operator=(const ApplicationBuildOperationTimer &) = delete;

private:
  ApplicationBuildOperation operation_;
  MonotonicClock::time_point begin_;
};

llvm::Error invalid(const llvm::Twine &message);

ApplicationPairDecisionDisposition mapIncompleteReasonToPairDisposition(
    const dse::DsePlanIncompleteReason &reason);

ApplicationPairDecisionDisposition
mapResourceTimeFrontierReasonToPairDisposition(
    dse::ResourceTimeFrontierIncompleteReason reason);

std::optional<ApplicationPairDecisionDisposition>
mapRuntimeDispositionToPairDisposition(
    ApplicationMappingRuntimeDisposition disposition);

std::optional<dse::PreMappingSpectrumClass>
requestedResourceTimeSpectrumClass(dse::PreMappingSpectrumEndpoint endpoint);

llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
deriveApplicationPartitionDelta(const dse::JointDesignExplorationPlan &parent,
                                const dse::JointDesignExplorationPlan &child);

std::optional<ApplicationPairDecisionDisposition>
classifyResourceTimeSelectionOutcome(
    const std::optional<dse::ResourceTimeSpectrumFunnelResult> &spectrum,
    std::optional<dse::PreMappingSpectrumClass> requestedClass);

ApplicationPairDecisionDisposition prioritizeIncompletePairDisposition(
    llvm::ArrayRef<ApplicationPairDecisionDisposition> causes,
    bool declaredWorkExhausted);

ApplicationPairDecisionRecord deriveApplicationPairDecision(
    const PreparedApplicationBuild &prepared,
    const std::vector<ApplicationMappingCandidateOutcome> &outcomes,
    const dse::JointDesignExecution &execution,
    llvm::ArrayRef<ApplicationPairQualityInvocationRecord> qualityInvocations);

ApplicationPairDecisionRecord makePreparationPairDecision(
    const std::optional<ArtifactRootReference> &sourceProgram,
    const std::optional<ArtifactRootReference> &fabric,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    llvm::ArrayRef<dse::PreMappingCandidatePlanningRecord> inventory,
    ApplicationPairDecisionDisposition disposition, llvm::StringRef detail,
    std::optional<std::uint64_t> sourceHostOnlyWork = std::nullopt,
    std::optional<std::array<std::uint8_t, 32>> invocationRunKey = std::nullopt,
    bool ownerVerifiedPreAdmission = false,
    std::optional<SelectedApplicationInput> portfolioInput = std::nullopt);

ApplicationPairDecisionRecord makePreAdmissionFailurePairDecision(
    std::optional<SelectedApplicationInput> portfolioInput,
    const ArtifactRootReference &requestedSystem,
    ApplicationPairDecisionDisposition disposition, llvm::StringRef detail);

llvm::Expected<const dse::ResourceTimeScheduleHint *>
findResourceTimeScheduleHint(
    const dse::ResourceTimeCandidateFunnelEvaluation &evaluation,
    const ComponentViewDigest &digest);

llvm::Expected<std::optional<dse::ResourceTimeSpectrumFunnelResult>>
verifyResourceTimeAlternative(
    const dse::ResourceTimeMappingFunnel &funnel,
    const PreparedApplicationMappingAlternative &alternative,
    llvm::ArrayRef<ArtifactRootReference> systemMappings,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ComponentViewDigest &scheduleHintDigest,
    llvm::ArrayRef<dse::ResourceTimeMappingDeploymentEndpoint> endpoints = {},
    ExecutionControlView executionControl = {});

} // namespace loom::application::build_detail

#endif // LOOM_APPLICATION_BUILDINTERNAL_H
