#ifndef LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
#define LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H

#include "Deployment/Deployment.h"
#include "Evaluation/Request.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "DeploymentTestSupport.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::eda::test {

enum class MappedRtlFixtureTopology : std::uint8_t {
  Minimal,
  HeterogeneousPortable,
};

struct MappedRtlRequestFixture final {
  evaluation::EvaluationRequest request;
  evaluation::CaseArtifactResolution resolution;
  hardware::FinalizedHardwareImplementation implementation;
  deployment::FinalizedDeployment deployment;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ArtifactRootReference module;
  ArtifactRootReference techMapping;
  ArtifactRootReference spatialMapping;
};

MappedRtlRequestFixture buildMappedRtlRequestFixture(
    llvm::StringRef test, llvm::StringRef stableSimulatorBuildIdentity,
    ArtifactStore &artifacts, BlobStore &blobs,
    const deployment::test::TemporaryTree &tree,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable);

} // namespace loom::eda::test

#endif // LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
