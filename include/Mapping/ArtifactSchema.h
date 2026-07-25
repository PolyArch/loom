#ifndef LOOM_MAPPING_ARTIFACTSCHEMA_H
#define LOOM_MAPPING_ARTIFACTSCHEMA_H

#include "Common/Artifact.h"

namespace loom::mapping {

/// The Mapping family's own persistent schema descriptor. The Mapping schema
/// declaration owns this identity and version; a consumer references this
/// descriptor instead of constructing a schema string or keeping a parallel
/// version fact.
inline constexpr ArtifactSchemaDescriptor artifactSchema{"loom.mapping",
                                                         SchemaVersion{2, 0}};

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACTSCHEMA_H
