#ifndef LOOM_LIB_PNR_SPATIALPHYSICALTIMING_H
#define LOOM_LIB_PNR_SPATIALPHYSICALTIMING_H

#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/PnrIndex.h"
#include "PnR/RoutingNegotiation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class RouteTreeState;
struct FrozenSpatialTerminalBinding;

namespace detail {

struct SpatialLogicalNetPhysicalTiming final {
  std::uint64_t worstArrivalDelayQuanta = 0;
  std::uint64_t totalNegativeSlackQuanta = 0;
  std::uint64_t structuralCriticality = 0;
};

struct SpatialPhysicalTimingProjection final {
  std::uint64_t worstArrivalDelayQuanta = 0;
  std::uint64_t totalNegativeSlackQuanta = 0;
};

llvm::Expected<std::uint64_t> advanceSpatialPhysicalTiming(
    std::uint64_t delayQuanta,
    ::loom::fabric::FabricPhysicalTimingBoundaryKind boundary,
    std::uint64_t arrivalQuanta, std::uint64_t requiredQuanta,
    SpatialLogicalNetPhysicalTiming &timing);

llvm::Error
observeSpatialPhysicalTimingEndpoint(std::uint64_t arrivalQuanta,
                                     std::uint64_t requiredQuanta,
                                     SpatialLogicalNetPhysicalTiming &timing);

llvm::Expected<RouteCost>
physicalTimingDrivenTraversalCost(std::uint64_t delayQuanta,
                                  std::uint64_t requiredQuanta,
                                  std::uint64_t structuralCriticality);

llvm::Expected<RouteCost>
physicalTimingDrivenNegativeSlackCost(std::uint64_t excessDeltaQuanta,
                                      std::uint64_t requiredQuanta,
                                      std::uint64_t structuralCriticality);

llvm::Expected<std::optional<PnrIndex>> projectSelectedSpatialTerminalTraversal(
    const FrozenSpatialPnrProblem &problem,
    FrozenSpatialTerminalBinding binding,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments);

llvm::Expected<std::vector<std::uint64_t>>
projectSpatialLogicalNetRouteNodeArrivals(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route, llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments);

llvm::Expected<std::uint64_t> projectSpatialLogicalNetSourceArrival(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments);

llvm::Expected<SpatialLogicalNetPhysicalTiming>
projectSpatialLogicalNetPhysicalTiming(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route, PnrIndex registerFifoTransfer,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
    std::vector<std::uint64_t> *routeNodeArrivals = nullptr,
    std::vector<std::pair<PnrIndex, std::uint64_t>> *routeNodeWorklist =
        nullptr);

llvm::Expected<SpatialPhysicalTimingProjection> projectSpatialPhysicalTiming(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> graphBoundaryAttachments,
    std::vector<std::uint64_t> *netWorstArrivals = nullptr,
    std::vector<std::uint64_t> *netNegativeSlacks = nullptr);

llvm::Expected<SpatialPhysicalTimingProjection>
projectSpatialMappingPhysicalTiming(
    const ::loom::mapping::SpatialMappingView &mapping,
    const ::loom::fabric::FabricPhysicalTimingProfileView &profile);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALPHYSICALTIMING_H
