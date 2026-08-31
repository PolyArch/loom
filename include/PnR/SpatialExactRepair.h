#ifndef LOOM_PNR_SPATIALEXACTREPAIR_H
#define LOOM_PNR_SPATIALEXACTREPAIR_H

#include "Common/ExecutionControl.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialPnrWorkLedger.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::pnr {

namespace detail {
struct SpatialRuntimeCounterexampleBreaker;
}

enum class SpatialExactRepairResultKind : std::uint8_t {
  Repaired,
  RegionInfeasibleUnderFixedBoundary,
  UnknownBudgetExhausted,
  TimedOut,
  RoutingIncomplete,
  ProofNotEstablished,
  RegionTooLarge,
  UnsupportedEncoding,
  InternalError,
};

struct SpatialExactRepairResult final {
  SpatialExactRepairResultKind kind;
  std::uint64_t regionDecisions = 0;
  std::uint64_t solverCalls = 0;
  std::uint64_t actionCount = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
  std::string detail;
  /// Deterministic canonical solve work, including exact memo hits. Actual
  /// solver invocations remain in `solverCalls`.
  std::uint64_t logicalSolverCalls = 0;
};

/// Returns the exact reason that the selected repair provider cannot encode
/// the frozen search domain. This check is run before mutable Candidate state
/// exists; an empty result only excludes statically recognizable domain-level
/// capability mismatches. Candidate-local witnesses still undergo typed
/// runtime capability checks.
std::optional<std::string>
unsupportedSpatialExactRepairDomain(const FrozenSpatialPnrProblem &problem);

/// Worker-local bounded exact repair scratch for one dependency-closed
/// Spatial Mapping violation region. The transport-closure profile encodes
/// compute and memory placement, terminal attachment, local transfer
/// disposition, routes, tags, and their closed constraints. The
/// atomic-capacity profile encodes compute binding only. A witness outside the
/// selected profile's total encoding returns UnsupportedEncoding and can never
/// prove the invocation infeasible.
class SpatialExactRepairScratch final {
public:
  llvm::Expected<SpatialExactRepairResult>
  repair(SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
         std::uint64_t solverCallLimit,
         DeterministicPnrRandomStream &exactRepairStream,
         SpatialPnrWorkLedgerView workLedger = {},
         /// Engaged only by a runtime CEGAR owner. It selects one exact live
         /// frozen clause instead of allowing an unrelated static witness to
         /// take precedence.
         std::optional<PnrIndex> runtimeCounterexampleClause = std::nullopt,
         ExecutionControlView executionControl = {});

  std::size_t retainedStorageBytes() const;

private:
  llvm::Error planRegionDecision();
  llvm::Error consumePendingRegionDecisions();

  llvm::Expected<SpatialExactRepairResult>
  repairTransportClosure(SpatialCandidateState &candidate,
                         std::uint64_t restartOrdinal,
                         std::uint64_t solverCallLimit,
                         DeterministicPnrRandomStream &exactRepairStream,
                         std::optional<PnrIndex> runtimeCounterexampleClause);

  llvm::Expected<SpatialExactRepairResult> repairTransportClosureRegion(
      SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
      std::uint64_t solverCallLimit, std::int32_t solverSeed,
      llvm::ArrayRef<SpatialFixedTerminalCutCertificate> certificates,
      bool &requiresRegionExpansion,
      const detail::SpatialRuntimeCounterexampleBreaker *runtimeBreaker =
          nullptr);

  SpatialActionExecutorScratch actionExecutor_;
  std::vector<std::uint8_t> decisionIncluded_;
  std::vector<std::uint8_t> relationIncluded_;
  std::vector<std::uint8_t> netIncluded_;
  std::vector<PnrIndex> decisionQueue_;
  std::vector<PnrIndex> decisions_;
  std::vector<PnrIndex> relations_;
  std::vector<PnrIndex> affectedNets_;
  SpatialFixedTerminalCutCertificate routeCutCertificate_;
  std::vector<SpatialFixedTerminalCutCertificate> learnedCutCertificates_;
  std::vector<std::uint8_t> routeCutBlockedTraversals_;
  std::vector<std::uint8_t> routeCutReachableEndpoints_;
  std::vector<PnrIndex> routeCutWorklist_;
  std::vector<int> decisionVariables_;
  std::vector<PnrIndex> legalValueOffsets_;
  std::vector<std::int64_t> legalValues_;
  std::vector<std::int64_t> elementValues_;
  std::vector<SpatialMappingAction> actions_;
  SpatialPnrWorkLedgerView workLedger_;
  ExecutionControlView executionControl_;
  std::uint64_t accountedRegionDecisionCount_ = 0;
  std::uint64_t pendingRegionDecisionCount_ = 0;
  std::vector<std::uint8_t> accountedRegionDecisions_;
  std::vector<std::uint8_t> accountedRegionNets_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALEXACTREPAIR_H
