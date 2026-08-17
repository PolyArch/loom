#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMRECURRENCETIMING_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMRECURRENCETIMING_H

#include "PnR/System/SystemCandidateState.h"

namespace loom::pnr::detail {

llvm::Expected<SpatialRecurrenceTimingProjection> projectSystemRecurrenceTiming(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMRECURRENCETIMING_H
