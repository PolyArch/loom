#ifndef LOOM_PNR_SPATIALEXACTREPAIR_H
#define LOOM_PNR_SPATIALEXACTREPAIR_H

#include "PnR/SpatialActionExecutor.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::pnr {

enum class SpatialExactRepairResultKind : std::uint8_t {
  Repaired,
  RegionInfeasibleUnderFixedBoundary,
  UnknownBudgetExhausted,
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
};

/// Worker-local bounded exact repair scratch for the complete
/// relation-connected compute-binding region of one canonical CapacityOveruse
/// witness. Witness domains without a total exact encoding fail closed.
class SpatialExactRepairScratch final {
public:
  llvm::Expected<SpatialExactRepairResult>
  repairCapacityOveruse(SpatialCandidateState &candidate,
                        std::uint64_t restartOrdinal);

  std::size_t retainedStorageBytes() const;

private:
  llvm::Expected<SpatialExactRepairResult>
  repairRouteCapacityOveruse(SpatialCandidateState &candidate,
                             std::uint64_t restartOrdinal);

  SpatialActionExecutorScratch actionExecutor_;
  std::vector<std::uint8_t> decisionIncluded_;
  std::vector<std::uint8_t> relationIncluded_;
  std::vector<std::uint8_t> netIncluded_;
  std::vector<PnrIndex> decisionQueue_;
  std::vector<PnrIndex> decisions_;
  std::vector<PnrIndex> relations_;
  std::vector<PnrIndex> affectedNets_;
  std::vector<PnrIndex> routeCapacityWitnesses_;
  std::vector<PnrIndex> routeCutLogicalNets_;
  std::vector<PnrIndex> routeCutDecisionLocals_;
  std::vector<int> decisionVariables_;
  std::vector<PnrIndex> legalValueOffsets_;
  std::vector<std::int64_t> legalValues_;
  std::vector<std::int64_t> elementValues_;
  std::vector<SpatialMappingAction> actions_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALEXACTREPAIR_H
