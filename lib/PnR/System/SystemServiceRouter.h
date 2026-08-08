#ifndef LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H
#define LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H

#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace loom::pnr::detail {

struct CanonicalSystemServiceRoutes final {
  std::vector<SystemServiceRouteSelection> routes;
  std::vector<SystemServiceRouteNodeSelection> nodes;
  std::vector<SystemServiceRouteSinkSelection> sinks;
};

struct SystemServiceRouteTraversalExclusion final {
  PnrIndex leg = getInvalidPnrIndex();
  PnrIndex traversal = getInvalidPnrIndex();
};

llvm::Expected<CanonicalSystemServiceRoutes>
buildCanonicalSystemServiceRoutes(const FrozenSystemPnrProblem &problem,
                                  llvm::ArrayRef<PnrIndex> threadChoices,
                                  llvm::ArrayRef<PnrIndex> graphChoices,
                                  std::uint64_t &endpointExpansions);

llvm::Expected<CanonicalSystemServiceRoutes> buildSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    std::optional<SystemServiceRouteTraversalExclusion> exclusion,
    std::uint64_t &endpointExpansions);

llvm::Error verifySystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> routes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H
