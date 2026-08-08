#ifndef LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H
#define LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H

#include "PnR/System/SystemActionDomain.h"
#include "PnR/System/SystemActionExecutor.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstdint>
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

  SystemActionTransitionFailure(SystemActionTransitionFailureKind kind,
                                std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SystemActionTransitionFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemActionTransitionFailureKind kind_;
  std::string message_;
};

struct SystemAnnealingStatistics final {
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t acceptedActionCount = 0;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;

  friend bool operator==(const SystemAnnealingStatistics &lhs,
                         const SystemAnnealingStatistics &rhs) {
    return lhs.calibrationProposalSlots == rhs.calibrationProposalSlots &&
           lhs.annealingBaseProposalSlots == rhs.annealingBaseProposalSlots &&
           lhs.annealingMovableProposalSlots ==
               rhs.annealingMovableProposalSlots &&
           lhs.acceptedActionCount == rhs.acceptedActionCount &&
           lhs.assignmentAttempts == rhs.assignmentAttempts &&
           lhs.endpointExpansions == rhs.endpointExpansions;
  }
};

class SystemAnnealingSearchScratch final {
public:
  llvm::Expected<SystemAnnealingStatistics>
  run(SystemCandidateStateHandle &candidate, std::uint64_t seedAttemptOrdinal);

private:
  SystemActionDomainScratch actionDomain_;
  std::vector<dse::ObjectiveWideValue> positiveCalibrationDeltas_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMANNEALINGSEARCH_H
