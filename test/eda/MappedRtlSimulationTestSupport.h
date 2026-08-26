#ifndef LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
#define LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H

#include "Deployment/Deployment.h"
#include "Evaluation/Request.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "DeploymentTestSupport.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace mlir {
class MLIRContext;
}

namespace loom::eda::test {

enum class MappedRtlFixtureTopology : std::uint8_t {
  Minimal,
  HeterogeneousPortable,
};

enum class MappedRtlRouteCoverage : std::uint8_t {
  AnyLegal,
  BypassFifo,
};

enum class MappedSystemInterconnect : std::uint8_t {
  None,
  Gem5EventTransport,
};

enum class MappedSpatialHardwareFixtureOperation : std::uint8_t {
  DataflowPublication,
  FabricModuleConstructionAndFinalization,
  TechMapping,
  SpatialPnr,
  SystemFabricAndInterconnectConstruction,
  ConfigurationAbiAndHardwareImplementationGeneration,
};

enum class MappedSpatialHardwareFixtureBoundary : std::uint8_t {
  Begin,
  End,
};

/// Synchronously brackets each operation. End is emitted on normal return and
/// C++ stack unwinding; observers must not throw.
using MappedSpatialHardwareFixtureObserver =
    llvm::function_ref<void(MappedSpatialHardwareFixtureOperation,
                            MappedSpatialHardwareFixtureBoundary)>;

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

struct MappedSpatialHardwareFixture final {
  fabric::FinalizedFabricRoot module;
  ArtifactRootReference techMapping;
  mapping::FinalizedSpatialMapping spatialMapping;
  fabric::FinalizedFabricRoot system;
  std::optional<ArtifactRootReference> interconnect;
  std::vector<hardware::FinalizedHardwareImplementation> implementations;
};

MappedSpatialHardwareFixture buildMappedSpatialHardwareFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    mlir::MLIRContext &context, ArtifactStore &artifacts, BlobStore &blobs,
    deployment::test::MappedSpatialSystemSpec systemSpec,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable,
    MappedRtlRouteCoverage routeCoverage = MappedRtlRouteCoverage::BypassFifo,
    MappedSystemInterconnect interconnect = MappedSystemInterconnect::None,
    MappedSpatialHardwareFixtureObserver observer = {});

MappedRtlRequestFixture buildMappedRtlRequestFixture(
    llvm::StringRef test, llvm::StringRef stableSimulatorBuildIdentity,
    ArtifactStore &artifacts, BlobStore &blobs,
    const deployment::test::TemporaryTree &tree,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable);

} // namespace loom::eda::test

#endif // LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
