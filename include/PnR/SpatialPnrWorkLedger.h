#ifndef LOOM_PNR_SPATIALPNRWORKLEDGER_H
#define LOOM_PNR_SPATIALPNRWORKLEDGER_H

#include "PnR/PnrWorkLedger.h"

namespace loom::pnr {

using SpatialPnrWorkKind = PnrWorkKind;
inline constexpr std::size_t spatialPnrWorkKindCount = pnrWorkKindCount;
using SpatialPnrWorkCounterRef = PnrWorkCounterRef;
using SpatialPnrWorkLedgerView = PnrWorkLedgerView;

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPNRWORKLEDGER_H
