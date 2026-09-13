#ifndef LOOM_LIB_PNR_GENERATOR_SPATIALPNRWORKACCOUNTING_H
#define LOOM_LIB_PNR_GENERATOR_SPATIALPNRWORKACCOUNTING_H

#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrWorkLedger.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr::detail {

/// Adds one owner's work to a ledger counter, refusing the overflow that would
/// silently reopen a bounded schedule.
llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &target,
                       llvm::StringRef subject);

/// The one view every Spatial owner plans and consumes through, so a restart's
/// planned and consumed work remain one ledger rather than several.
SpatialPnrWorkLedgerView
canonicalWorkLedger(SpatialPnrGenerationAccounting &accounting);

/// Folds one restart's ledger into the invocation's, refusing any overflow.
llvm::Error
accumulateRestartAccounting(const SpatialPnrGenerationAccounting &source,
                            SpatialPnrGenerationAccounting &target);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_GENERATOR_SPATIALPNRWORKACCOUNTING_H
