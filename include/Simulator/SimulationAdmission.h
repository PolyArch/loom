#ifndef LOOM_SIMULATOR_SIMULATIONADMISSION_H
#define LOOM_SIMULATOR_SIMULATIONADMISSION_H

#include "Simulator/SimulationArtifacts.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"

#include "llvm/Support/Error.h"

namespace dataflow {
class CanonicalDataflowProgramView;
} // namespace dataflow

// Cold-path DFG-sim admission of the shared spatial workload and runtime
// input, owned by docs/spec-sim-dfg.md `Admission`. The adapter validates
// both artifacts through the same rooted launch against the exact Dataflow
// owner view and keeps no persistent simulator-local catalog; it runs before
// any execution state exists. CGRA-sim shares this workload admission stage
// after its exact D/F/SpatialMapping closure has been strictly prepared.
namespace loom::sim {

/// DFG-sim admission: the workload's exact RootedGraphLaunchRef resolves
/// against this Dataflow owner (missing, stale, foreign-artifact, or
/// wrong-kind references fail), the runtime input names the exact workload,
/// and every workload/runtime-input rule holds. Returns the called graph.
llvm::Expected<dataflow::GraphRef> admitDfgSpatialSimulation(
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const dataflow::CanonicalDataflowProgramView &program);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONADMISSION_H
