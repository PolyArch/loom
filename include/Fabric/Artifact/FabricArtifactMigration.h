#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACTMIGRATION_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACTMIGRATION_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/Support/Error.h"

namespace loom::fabric {

/// The exact previous-family descriptor whose objects the current owner
/// re-finalizes. This migration boundary is its only semantic owner: the
/// ordinary strict importer accepts and emits exactly `fabricArtifactSchema`
/// and never reads, defaults, or upgrades a 7.0 object.
inline constexpr ArtifactSchemaDescriptor fabricArtifactSchemaV7_0{
    "loom.fabric", SchemaVersion{7, 0}};

/// Re-finalizes one exact loom.fabric 7.0 root under the current 7.2
/// descriptor and returns the new ArtifactRootReference. This is an
/// envelope-level rewrite: it rewrites only the dependency rows to the
/// migrated references, recursively across the direct dependency closure,
/// publishes the result under the current descriptor, and independently
/// reverifies it through the complete strict import before returning. It
/// therefore succeeds exactly when the 7.0 canonical MLIR payload already
/// satisfies the current grammar. A 7.0 System payload predates the required
/// 7.2 private-cache and SpatialCore memory-path realizations and must be
/// re-finalized from its authoring source instead. The new identity differs
/// from the 7.0 identity, so every Mapping, ResolvedConfig, and evaluation
/// provenance that names the 7.0 root no longer resolves and must be
/// regenerated against the migrated root.
llvm::Expected<ArtifactRootReference>
migrateFabricRootV7_0ToCurrent(const ArtifactRootReference &reference,
                               const ArtifactStore &store);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACTMIGRATION_H
