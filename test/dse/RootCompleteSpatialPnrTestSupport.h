#pragma once

#include "Fabric/Artifact/FabricArtifact.h"

#include <cstdint>

namespace loom {

class ArtifactStore;

namespace test {

fabric::FinalizedFabricRoot buildSpatialCore(ArtifactStore &store,
                                             std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildLineageSpatialCore(ArtifactStore &store, std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildFeedbackPruningSpatialCore(ArtifactStore &store);

} // namespace test
} // namespace loom
