#ifndef LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H
#define LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H

#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialProgressState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

class FrozenSpatialPortIndex;
class FrozenSpatialRoutingGraph;
class SpatialCandidateState;
struct FrozenSpatialTerminalBinding;

llvm::Expected<bool> spatialAttachmentProvidesLocalProgressBoundary(
    const FrozenSpatialPortIndex &ports, PnrIndex attachmentOption);

llvm::Expected<bool> spatialTerminalProvidesLocalProgressBoundary(
    const SpatialCandidateState &candidate,
    FrozenSpatialTerminalBinding terminal);

llvm::Expected<llvm::ArrayRef<FrozenSpatialProgressPrerequisite>>
spatialSinkProgressDependencies(const FrozenSpatialPnrProblem &problem,
                                PnrIndex logicalNet, PnrIndex dependentSink);

llvm::Expected<bool> spatialRouteProgressDependencySatisfied(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    const FrozenSpatialProgressPrerequisite &prerequisite,
    PnrIndex dependentSink);

llvm::Expected<std::uint64_t>
spatialCandidateProgressWitnessCount(const SpatialCandidateState &candidate);

llvm::Expected<SpatialProgressNetCapacityProjection>
projectSpatialNetCapacityProofInputs(const SpatialCandidateState &candidate,
                                     PnrIndex logicalNet,
                                     const RouteTreeState *route = nullptr);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H
