#ifndef LOOM_TEST_MAPPING_CGRAADMISSIONTESTSUPPORT_H
#define LOOM_TEST_MAPPING_CGRAADMISSIONTESTSUPPORT_H

#include "Common/Artifact.h"

namespace loom {
class ArtifactStore;
class BlobStore;
}

namespace loom::test {

void exerciseCgraAdmission(const ArtifactRootReference &dataflow,
                           const ArtifactRootReference &fabric,
                           const ArtifactRootReference &spatialMapping,
                           const ArtifactRootReference &foreignFabric,
                           const ArtifactStore &store, const BlobStore &blobs,
                           bool expectPhysicalTags = false,
                           bool expectCausalComputeRelease = false);

void exerciseCgraMemoryAdmission(const ArtifactRootReference &dataflow,
                                 const ArtifactRootReference &fabric,
                                 const ArtifactRootReference &spatialMapping,
                                 const ArtifactStore &store);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_CGRAADMISSIONTESTSUPPORT_H
