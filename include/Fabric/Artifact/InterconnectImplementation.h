#ifndef LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H
#define LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"

namespace loom::fabric {

/// Materializes the complete event-transport implementation used by the gem5
/// System provider. The exact System remains the semantic architecture owner;
/// this sibling root owns only protocol-local objects and their total typed
/// refinement relation.
llvm::Expected<FinalizedFabricRoot>
finalizeGem5EventInterconnectImplementation(
    const ArtifactRootReference &refinedSystem, const ArtifactStore &store);

/// Returns the closed protocol schema selected by one strictly imported
/// InterconnectImplementation root. Other Fabric root kinds are rejected.
llvm::Expected<::fabric::InterconnectProtocolSchema>
interconnectProtocolSchema(const FinalizedFabricRoot &implementation);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H
