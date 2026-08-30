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

/// Re-finalizes one exact loom.fabric 7.0 root under the current 7.1
/// descriptor and returns the new ArtifactRootReference. The queue discipline
/// extension is non-breaking: a 7.0 FIFO carries no queue_discipline
/// attribute, which 7.1 reads as the canonical StrictFifo default, so the
/// canonical MLIR payload and its embedded resource contracts are unchanged.
/// Migration rewrites only the envelope dependency rows to the migrated 7.1
/// references, recursively across the direct dependency closure, publishes
/// the result under the 7.1 descriptor, and independently reverifies it
/// through the complete strict 7.1 import before returning. The new identity
/// differs from the 7.0 identity, so every Mapping, ResolvedConfig, and
/// evaluation provenance that names the 7.0 root no longer resolves and must
/// be regenerated against the migrated root.
llvm::Expected<ArtifactRootReference>
migrateFabricRootV7_0ToV7_1(const ArtifactRootReference &reference,
                            const ArtifactStore &store);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACTMIGRATION_H
