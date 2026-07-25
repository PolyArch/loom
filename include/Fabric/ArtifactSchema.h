#ifndef LOOM_FABRIC_ARTIFACTSCHEMA_H
#define LOOM_FABRIC_ARTIFACTSCHEMA_H

#include "Common/Artifact.h"

namespace fabric {

/// The Fabric family's own persistent schema descriptor. The Fabric artifact
/// authority owns this identity and version; a consumer references this
/// descriptor instead of constructing a schema string or keeping a parallel
/// version fact.
inline constexpr ::loom::ArtifactSchemaDescriptor artifactSchema{
    "loom.fabric", ::loom::SchemaVersion{1, 0}};

} // namespace fabric

#endif // LOOM_FABRIC_ARTIFACTSCHEMA_H
