#ifndef LOOM_COMMON_ARTIFACTLOCALREFERENCEREGISTRY_H
#define LOOM_COMMON_ARTIFACTLOCALREFERENCEREGISTRY_H

#include "Common/ArtifactLocalReference.h"

namespace loom {

/// Owner-library registration boundary for one exact local-reference kind.
/// This declaration is intentionally private to library implementations;
/// consumers can frame, resolve, and validate references but cannot author an
/// Artifact family's schema or local-kind catalog.
llvm::Error
registerArtifactLocalReferenceKind(const ArtifactSchemaDescriptor &ownerSchema,
                                   std::uint32_t ownerLocalKind,
                                   ArtifactLocalReferenceCodec codec);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTLOCALREFERENCEREGISTRY_H
