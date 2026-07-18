#ifndef LOOM_COMMON_ARTIFACTFINALIZER_H
#define LOOM_COMMON_ARTIFACTFINALIZER_H

#include "Common/Artifact.h"

namespace loom {

ArtifactIdentity
finalizeArtifactIdentity(const ArtifactSchemaDescriptor &schema,
                         const CanonicalSemanticBytes &canonicalBytes);

} // namespace loom

#endif // LOOM_COMMON_ARTIFACTFINALIZER_H
