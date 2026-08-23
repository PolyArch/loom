#ifndef LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H
#define LOOM_PNR_SYSTEM_SYSTEMSERVICEROUTER_H

#include "PnR/RoutingNegotiation.h"
#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
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

enum class SystemServiceRouteRepairRegionKind : std::uint8_t {
  SingleSink,
  RootedSubtree,
};

struct SystemServiceRouteRepairRegion final {
  SystemServiceRouteRepairRegionKind kind;
  PnrIndex leg = getInvalidPnrIndex();
  PnrIndex anchor = getInvalidPnrIndex();
};

struct SystemServiceRoutesView final {
  llvm::ArrayRef<SystemServiceRouteSelection> routes;
  llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes;
  llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks;
};

using SystemRouteArcCostProjection =
    llvm::function_ref<llvm::Expected<llvm::ArrayRef<RouteCost>>(
        llvm::ArrayRef<std::uint64_t> workingCapacityUsage)>;

struct SystemServiceRouteBuildRequest final {
  llvm::ArrayRef<PnrIndex> legOrder;
  llvm::ArrayRef<RouteCost> lowerBoundArcCosts;
  SystemRouteArcCostProjection currentArcCosts;
  std::optional<SystemServiceRoutesView> priorRoutes;
  std::optional<SystemServiceRouteTraversalExclusion> exclusion;
  std::optional<SystemServiceRouteRepairRegion> repairRegion;
  llvm::ArrayRef<PnrIndex> reroutedLegs;
  bool enforceCapacity = true;
};

struct BuiltSystemServiceRoutes final {
  CanonicalSystemServiceRoutes selections;
  std::vector<std::uint64_t> capacityUsage;
};

struct SystemFixedTerminalCapacityLegEvidence final {
  PnrIndex leg = getInvalidPnrIndex();
  PnrIndex sourceEndpoint = getInvalidPnrIndex();
  std::vector<PnrIndex> sinkEndpoints;
  std::vector<PnrIndex> claimingTraversals;
  std::uint64_t minimumClaim = 0;
  std::uint64_t reachableEndpointCount = 0;
  std::vector<PnrIndex> unreachableSinkEndpoints;

  bool isForced() const { return !unreachableSinkEndpoints.empty(); }
};

struct SystemFixedTerminalCapacityConflict final {
  PnrIndex capacityCell = getInvalidPnrIndex();
  std::uint64_t usage = 0;
  std::uint64_t capacity = 0;
  std::uint64_t mandatoryUsage = 0;
  std::vector<SystemFixedTerminalCapacityLegEvidence> logicalNets;

  bool hasCertificate() const { return mandatoryUsage > capacity; }
};

llvm::Expected<std::vector<RouteCost>>
buildSystemServiceRouteLowerBoundArcCosts(
    const FrozenEndpointRoutingTopology &topology);

llvm::Expected<std::vector<PnrIndex>>
buildSystemServiceRouteLegOrder(const FrozenEndpointRoutingTopology &topology,
                                SystemServiceRoutesView routes,
                                llvm::ArrayRef<std::uint64_t> capacityUsage);

llvm::Expected<std::vector<std::uint64_t>>
measureSystemServiceRouteCapacityUsage(
    const FrozenEndpointRoutingTopology &topology,
    SystemServiceRoutesView routes, bool enforceCapacity);

llvm::Expected<std::uint64_t> measureSystemServiceRouteTraversalClaim(
    const FrozenEndpointRoutingTopology &topology,
    SystemServiceRoutesView routes);

llvm::Expected<std::vector<SystemFixedTerminalCapacityConflict>>
analyzeSystemFixedTerminalCapacityConflicts(
    const FrozenSystemPnrProblem &problem, SystemServiceRoutesView routes,
    llvm::ArrayRef<std::uint64_t> capacityUsage);

llvm::Expected<BuiltSystemServiceRoutes>
buildSystemServiceRoutes(const FrozenSystemPnrProblem &problem,
                         llvm::ArrayRef<PnrIndex> threadChoices,
                         llvm::ArrayRef<PnrIndex> graphChoices,
                         const SystemServiceRouteBuildRequest &request,
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
