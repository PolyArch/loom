#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSETMIGRATION_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSETMIGRATION_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Mapping/Artifact/MappingConstraintSet.h"

#include "llvm/Support/Error.h"

namespace loom::mapping {

/// The exact previous-family descriptor whose objects the current owner
/// re-finalizes. `loom.mapping_constraints` is one family with a Spatial and a
/// System root, so this descriptor covers both; it is not Spatial-only. This
/// migration boundary is its only semantic owner: the ordinary strict
/// importers accept and emit exactly `mappingConstraintSetSchema` and never
/// read, default, or upgrade a 1.0 object.
inline constexpr ArtifactSchemaDescriptor mappingConstraintSetSchemaV1_0{
    "loom.mapping_constraints", SchemaVersion{1, 0}};

/// Re-finalizes one exact loom.mapping_constraints 1.0 Spatial root under the
/// current 1.1 descriptor and returns the new ArtifactRootReference. The
/// runtime-counterexample no-good clause is an optional semantic extension: a
/// 1.0 payload carries none, so its canonical clause sequence is unchanged and
/// migration rewrites nothing but the schema version it is finalized under.
/// The new identity nevertheless differs from the 1.0 identity, because the
/// version is hashed into Artifact identity, so every plan, provenance record,
/// and frozen-model cache entry that names the 1.0 root must be regenerated
/// against the migrated root. A payload that already carries the 1.1-only
/// clause kind is rejected as mislabelled rather than migrated. There is no
/// automatic fallback: a caller must invoke this owner deliberately.
llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                       const ArtifactStore &store);

/// The System twin of `migrateSpatialConstraintRootV1_0ToV1_1`. The System
/// root's clause catalog is unchanged between 1.0 and 1.1 — the no-good kind
/// is Spatial-only — so a System migration only re-finalizes the identical
/// clause sequence under the 1.1 descriptor for a new identity.
llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                      const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSETMIGRATION_H
