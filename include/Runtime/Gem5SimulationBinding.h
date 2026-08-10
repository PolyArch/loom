#ifndef LOOM_RUNTIME_GEM5SIMULATIONBINDING_H
#define LOOM_RUNTIME_GEM5SIMULATIONBINDING_H

#include "Common/Artifact.h"

namespace loom::runtime {

inline constexpr ArtifactSchemaDescriptor gem5SimulationBindingSchema{
    "loom.gem5_simulation_binding", SchemaVersion{2, 0}};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SIMULATIONBINDING_H
