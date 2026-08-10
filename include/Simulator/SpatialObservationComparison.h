#ifndef LOOM_SIMULATOR_SPATIALOBSERVATIONCOMPARISON_H
#define LOOM_SIMULATOR_SPATIALOBSERVATIONCOMPARISON_H

#include "Simulator/SimulationExecution.h"

namespace loom::sim {

/// Exact comparison of two observations aligned by one canonical Spatial
/// observable contract. This relation includes semantic state, defined bits,
/// pointer provenance, stream termination, and memory payload framing.
bool haveExactlyEqualSpatialFunctionalObservations(
    const SpatialFunctionalObservations &reference,
    const SpatialFunctionalObservations &candidate);

/// Exact comparison of two observations aligned by one canonical System
/// observable contract, including every external boundary category.
bool haveExactlyEqualSystemFunctionalObservations(
    const SystemFunctionalObservations &reference,
    const SystemFunctionalObservations &candidate);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALOBSERVATIONCOMPARISON_H
