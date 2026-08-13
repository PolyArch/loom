#ifndef LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H
#define LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H

#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class FrozenSpatialPortIndex;
class FrozenSpatialRoutingGraph;
class SpatialCandidateState;
struct FrozenSpatialTerminalBinding;

llvm::Expected<bool> spatialAttachmentProvidesLocalProgressBoundary(
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialRoutingGraph &routing, PnrIndex attachmentOption);

llvm::Expected<bool> spatialTerminalProvidesLocalProgressBoundary(
    const SpatialCandidateState &candidate,
    FrozenSpatialTerminalBinding terminal);

llvm::Expected<llvm::ArrayRef<PnrIndex>>
spatialSinkProgressDependencies(const FrozenSpatialPnrProblem &problem,
                                PnrIndex logicalNet, PnrIndex dependentSink);

llvm::Expected<bool> spatialRouteProgressDependencySatisfied(
    const SpatialCandidateState &candidate, PnrIndex logicalNet,
    PnrIndex prerequisiteSink, PnrIndex dependentSink);

llvm::Expected<std::uint64_t>
spatialCandidateClosedWaitCount(const SpatialCandidateState &candidate);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALPROGRESSANALYSIS_H
