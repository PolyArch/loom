//===- SimulationAdmission.cpp - DFG workload admission ------------------===//
//
// Cold-path DFG-sim admission of the shared spatial workload and runtime
// input. Validation reuses the one semantic validator of each family against
// the exact Dataflow owner view; no simulator-local catalog is created.
//
//===----------------------------------------------------------------------===//

#include "Simulator/SimulationAdmission.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "SimulationWireInternal.h"

#include <utility>

namespace loom::sim {

llvm::Expected<dataflow::GraphRef> admitDfgSpatialSimulation(
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const dataflow::CanonicalDataflowProgramView &program) {
  const SpatialSimulationWorkload *spatialWorkload = workload.spatial();
  const SpatialSimulationRuntimeInput *spatialInput = runtimeInput.spatial();
  if (!spatialWorkload || !spatialInput)
    return detail::invalid(
        "DFG-sim admission requires Spatial workload and runtime-input roots");
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(program, spatialWorkload->launchRef);
  if (!context)
    return context.takeError();
  if (llvm::Error error =
          detail::validateSpatialWorkload(*spatialWorkload, *context, program))
    return std::move(error);
  if (llvm::Error error = detail::validateSpatialRuntimeInput(
          *spatialInput, *spatialWorkload, workload.identity(), *context,
          program))
    return std::move(error);
  return context->graph;
}

} // namespace loom::sim
