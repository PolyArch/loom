#ifndef LOOM_PNR_CPSAT_SPATIALEXACTREPAIRINTERNAL_H
#define LOOM_PNR_CPSAT_SPATIALEXACTREPAIRINTERNAL_H

#include "PnR/SpatialExactRepair.h"

namespace loom::pnr::detail {

llvm::Error repairError(const llvm::Twine &detail);
SpatialExactRepairResult
repairResult(SpatialExactRepairResultKind kind, std::uint64_t regionDecisions,
             std::uint64_t solverCalls = 0, std::uint64_t actionCount = 0,
             std::string detail = {}, std::uint64_t endpointExpansions = 0,
             std::uint64_t negotiationIterations = 0,
             std::uint64_t logicalSolverCalls = 0);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_CPSAT_SPATIALEXACTREPAIRINTERNAL_H
