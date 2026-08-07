#ifndef LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H
#define LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H

#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::pnr::detail {

struct CanonicalSystemServiceRoutes final {
  std::vector<SystemServiceRouteSelection> routes;
  std::vector<SystemServiceRouteNodeSelection> nodes;
  std::vector<SystemServiceRouteSinkSelection> sinks;
};

llvm::Expected<CanonicalSystemServiceRoutes>
buildCanonicalSystemServiceRoutes(const FrozenSystemPnrProblem &problem,
                                  llvm::ArrayRef<PnrIndex> threadChoices,
                                  llvm::ArrayRef<PnrIndex> graphChoices);

llvm::Error verifySystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> routes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H
