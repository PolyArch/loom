#ifndef LOOM_PNR_SYSTEM_SYSTEMACTIONEXECUTOR_H
#define LOOM_PNR_SYSTEM_SYSTEMACTIONEXECUTOR_H

#include "PnR/System/SystemAction.h"
#include "PnR/System/SystemCandidateState.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

/// Invocation-local projection of a physical routing witness into mutable
/// graph-binding decisions. It refers only to the frozen candidate domain and
/// never enters a Mapping artifact or persistent no-good set.
struct SystemUpstreamReopenWitness final {
  PnrIndex capacityCell = getInvalidPnrIndex();
  std::vector<PnrIndex> graphDecisions;
};

enum class SystemActionMutationDomain : std::uint8_t {
  ExecutionBinding,
  TransportRouting,
  ResourceAllocation,
};

/// Exact invocation-local dependency cone and derived before/after values for
/// one immutable Action transaction. Objective delta remains the separately
/// typed energyDifference below.
struct SystemActionMutationRecord final {
  SystemActionMutationDomain domain =
      SystemActionMutationDomain::ExecutionBinding;
  std::vector<PnrIndex> threadDecisions;
  std::vector<PnrIndex> graphDecisions;
  std::vector<PnrIndex> serviceLegs;
  std::vector<PnrIndex> serviceTargets;
  std::vector<PnrIndex> instructionResourceUses;
  std::vector<PnrIndex> serviceResourceUses;
  std::uint64_t capacityOveruseBefore = 0;
  std::uint64_t capacityOveruseAfter = 0;
  std::uint64_t recurrenceMinimumInitiationIntervalBefore = 1;
  std::uint64_t recurrenceMinimumInitiationIntervalAfter = 1;
  std::uint64_t resourceMinimumInitiationIntervalBefore = 1;
  std::uint64_t resourceMinimumInitiationIntervalAfter = 1;
  std::uint64_t transportBitCycleDemandBefore = 0;
  std::uint64_t transportBitCycleDemandAfter = 0;
  ::loom::mapping::MappingProgressClosureKind progressBefore =
      ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished;
  ::loom::mapping::MappingProgressClosureKind progressAfter =
      ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished;
};

struct SystemActionProbeResult final {
  SystemCandidateStateHandle candidate;
  dse::ObjectiveVector objective;
  dse::ObjectiveSignedDifference energyDifference;
  SystemActionMutationRecord mutation;
  std::optional<SystemUpstreamReopenWitness> reopenWitness;
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
