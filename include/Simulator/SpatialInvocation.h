#ifndef LOOM_SIMULATOR_SPATIALINVOCATION_H
#define LOOM_SIMULATOR_SPATIALINVOCATION_H

#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::sim {

struct SpatialInvocationMemoryWrite final {
  std::uint64_t address = 0;
  std::vector<std::uint8_t> bytes;
};

/// Derives the sorted logical roots whose canonical actors can modify memory
/// during one exact rooted graph invocation. Dataflow actor semantics remain
/// the owner; invocation producers and consumers share this projection.
llvm::Expected<std::vector<dataflow::LogicalMemoryRootRef>>
projectSpatialInvocationWritableMemoryRoots(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch);

llvm::Expected<CanonicalSimulationRuntimeInput>
materializeSpatialInvocationRuntimeInput(
    const ImportedSpatialSimulationWorkload &workload,
    const runtime::SpatialInvocationWire &wire);

llvm::Expected<ImportedSpatialSimulationInputs>
materializeSpatialInvocationInputs(ImportedSpatialSimulationWorkload workload,
                                   const runtime::SpatialInvocationWire &wire);

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationInputs &inputs,
    const SpatialFunctionalObservations &observations);

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationWorkload &workload,
    const SpatialFunctionalObservations &observations);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALINVOCATION_H
