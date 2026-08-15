#ifndef LOOM_TEST_INTERNALMEMORYEDGETESTSUPPORT_H
#define LOOM_TEST_INTERNALMEMORYEDGETESTSUPPORT_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"

namespace loom {
class ArtifactStore;
}

namespace loom::test {

fabric::FinalizedFabricRoot
buildInternalMemoryEdgeFabric(ArtifactStore &store,
                              ::fabric::Schedule schedule);

} // namespace loom::test

#endif // LOOM_TEST_INTERNALMEMORYEDGETESTSUPPORT_H
