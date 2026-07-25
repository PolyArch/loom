#ifndef LOOM_SIMULATOR_ARTIFACTSCHEMA_H
#define LOOM_SIMULATOR_ARTIFACTSCHEMA_H

#include "Common/Artifact.h"

namespace loom::sim {

/// The simulation families' own persistent schema descriptors. The Simulation
/// Artifact authority owns these identities and versions; a consumer
/// references these descriptors instead of constructing a schema string or
/// keeping a parallel version fact.
inline constexpr ArtifactSchemaDescriptor workloadSchema{
    "loom.simulation_workload", SchemaVersion{1, 0}};
inline constexpr ArtifactSchemaDescriptor runtimeInputSchema{
    "loom.simulation_runtime_input", SchemaVersion{1, 0}};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_ARTIFACTSCHEMA_H
