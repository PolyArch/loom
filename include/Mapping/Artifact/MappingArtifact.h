#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H

#include "Common/Artifact.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/Support/Error.h"

namespace loom::mapping {

inline constexpr ArtifactSchemaDescriptor mappingArtifactSchema{
    "loom.mapping", SchemaVersion{2, 0}};

/// Canonicalizes one complete in-memory Mapping root for final verification.
/// This syntax layer normalizes schema-owned record order and Mapping-local
/// IDs. Exact upstream import and profile completeness are enforced by the
/// finalizer that publishes a MappingArtifact.
llvm::Expected<CanonicalSemanticBytes>
writeCanonicalMappingAssembly(::mapping::TechOp root);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
