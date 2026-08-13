#ifndef LOOM_LIB_PNR_SPATIALTAGPRESSUREDIAGNOSTIC_H
#define LOOM_LIB_PNR_SPATIALTAGPRESSUREDIAGNOSTIC_H

#include <cstdint>

namespace loom::pnr {

class SpatialCandidateState;
class SpatialRouteCostState;
struct SpatialTagAssignmentSummary;

std::uint64_t reportSpatialTagDomainPressure(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    const SpatialTagAssignmentSummary &summary, std::uint64_t iteration,
    std::uint64_t sessionIteration);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALTAGPRESSUREDIAGNOSTIC_H
