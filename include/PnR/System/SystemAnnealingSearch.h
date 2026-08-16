#ifndef LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H
#define LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H

#include "Common/ExecutionControl.h"
#include "PnR/System/SystemActionDomain.h"
#include "PnR/System/SystemActionExecutor.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr {

enum class SystemActionTransitionFailureKind : std::uint8_t {
  IntrinsicInvalid,
  WorkLimit,
};

class SystemActionTransitionFailure final
    : public llvm::ErrorInfo<SystemActionTransitionFailure> {
public:
  static char ID;

  SystemActionTransitionFailure(
      SystemActionTransitionFailureKind kind, std::string message,
      std::optional<SystemUpstreamReopenWitness> reopenWitness = std::nullopt)
      : kind_(kind), message_(std::move(message)),
        reopenWitness_(std::move(reopenWitness)) {}

  SystemActionTransitionFailureKind kind() const { return kind_; }
  const std::optional<SystemUpstreamReopenWitness> &reopenWitness() const {
    return reopenWitness_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemActionTransitionFailureKind kind_;
  std::string message_;
  std::optional<SystemUpstreamReopenWitness> reopenWitness_;
};

struct SystemAnnealingStatistics final {
  bool interrupted = false;
  bool completionGoalReached = false;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t acceptedActionCount = 0;
  std::uint64_t upstreamReopenWitnessCount = 0;
  std::uint64_t upstreamReopenActionProposalCount = 0;
  std::uint64_t upstreamReopenAcceptedActionCount = 0;
  std::uint64_t mutationOracleVerificationCount = 0;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;

  friend bool operator==(const SystemAnnealingStatistics &lhs,
                         const SystemAnnealingStatistics &rhs) {
    return lhs.interrupted == rhs.interrupted &&
           lhs.completionGoalReached == rhs.completionGoalReached &&
           lhs.calibrationProposalSlots == rhs.calibrationProposalSlots &&
           lhs.annealingBaseProposalSlots == rhs.annealingBaseProposalSlots &&
           lhs.annealingMovableProposalSlots ==
               rhs.annealingMovableProposalSlots &&
           lhs.acceptedActionCount == rhs.acceptedActionCount &&
           lhs.upstreamReopenWitnessCount == rhs.upstreamReopenWitnessCount &&
           lhs.upstreamReopenActionProposalCount ==
               rhs.upstreamReopenActionProposalCount &&
           lhs.upstreamReopenAcceptedActionCount ==
               rhs.upstreamReopenAcceptedActionCount &&
           lhs.mutationOracleVerificationCount ==
               rhs.mutationOracleVerificationCount &&
           lhs.assignmentAttempts == rhs.assignmentAttempts &&
           lhs.endpointExpansions == rhs.endpointExpansions &&
           lhs.negotiationIterations == rhs.negotiationIterations;
  }
};

class SystemAnnealingSearchScratch final {
public:
  llvm::Expected<SystemAnnealingStatistics>
  run(SystemCandidateStateHandle &candidate, std::uint64_t seedAttemptOrdinal,
      ExecutionControlView executionControl = {});

private:
  SystemActionDomainScratch actionDomain_;
  std::vector<dse::ObjectiveWideValue> positiveCalibrationDeltas_;
  std::vector<SystemExecutionBindingReopenAction> pendingReopenActions_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H
