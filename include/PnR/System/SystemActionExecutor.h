#ifndef LOOM_PNR_SYSTEM_SYSTEMACTIONEXECUTOR_H
#define LOOM_PNR_SYSTEM_SYSTEMACTIONEXECUTOR_H

#include "PnR/System/SystemAction.h"
#include "PnR/System/SystemCandidateState.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

struct SystemActionProbeResult final {
  SystemCandidateStateHandle candidate;
  dse::ObjectiveVector objective;
  dse::ObjectiveSignedDifference energyDifference;
};

struct SystemActionProbeAccounting final {
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
};

enum class SystemActionExecutionContext : std::uint8_t {
  NonFinal,
  FinalClosure,
};

llvm::Expected<SystemActionProbeResult>
probeSystemAction(const SystemCandidateStateHandle &current,
                  const dse::ObjectiveVector &currentObjective,
                  const SystemMappingAction &action,
                  SystemActionProbeAccounting &accounting,
                  SystemActionExecutionContext context =
                      SystemActionExecutionContext::NonFinal);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMACTIONEXECUTOR_H
