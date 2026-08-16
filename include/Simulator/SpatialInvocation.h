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

llvm::Expected<ImportedSpatialSimulationInputs>
materializeSpatialInvocationInputs(ImportedSpatialSimulationWorkload workload,
                                   const runtime::SpatialInvocationWire &wire);

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationInputs &inputs,
    const SpatialFunctionalObservations &observations);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALINVOCATION_H
