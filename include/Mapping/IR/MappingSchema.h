#ifndef LOOM_MAPPING_IR_MAPPINGSCHEMA_H
#define LOOM_MAPPING_IR_MAPPINGSCHEMA_H

#include "Common/Artifact.h"

namespace loom::mapping {

inline constexpr ArtifactSchemaDescriptor mappingArtifactSchema{
    "loom.mapping", SchemaVersion{5, 0}};

} // namespace loom::mapping

#endif // LOOM_MAPPING_IR_MAPPINGSCHEMA_H
