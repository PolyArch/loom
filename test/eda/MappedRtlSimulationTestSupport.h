#ifndef LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
#define LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H

#include "ADG/BuiltinDescriptor.h"
#include "Deployment/Deployment.h"
#include "Evaluation/Request.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "DeploymentTestSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
class ExecutionControlView;
namespace dse {
struct SpatialTransportRepairAlternative;
}
namespace pnr {
class ResolvedPnrConfigView;
}
} // namespace loom

namespace mlir {
class MLIRContext;
}

namespace loom::eda::test {

enum class MappedRtlFixtureTopology : std::uint8_t {
  Minimal,
  HeterogeneousPortable,
  BuiltinCoverage,
};

enum class MappedRtlRouteCoverage : std::uint8_t {
  AnyLegal,
  BypassFifo,
};

enum class MappedSystemInterconnect : std::uint8_t {
  None,
  Gem5EventTransport,
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

struct MappedSpatialHardwareFixture final {
  fabric::FinalizedFabricRoot module;
  ArtifactRootReference techMapping;
  mapping::FinalizedSpatialMapping spatialMapping;
  fabric::FinalizedFabricRoot system;
  std::optional<ArtifactRootReference> interconnect;
  std::vector<hardware::FinalizedHardwareImplementation> implementations;
};

struct MappedSpatialMappingFixture final {
  fabric::FinalizedFabricRoot module;
  ArtifactRootReference techMapping;
  mapping::FinalizedSpatialMapping spatialMapping;
};

struct MappedSpatialMappingRepairFixture final {
  std::optional<mapping::FinalizedSpatialMapping> spatialMapping;
  ArtifactRootReference constraintSet;
};

MappedSpatialMappingFixture buildMappedSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    mlir::MLIRContext &context, ArtifactStore &artifacts, BlobStore &blobs,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable,
    MappedRtlRouteCoverage routeCoverage = MappedRtlRouteCoverage::BypassFifo,
    std::size_t spatialMemoryOccurrenceCount = 1);

MappedSpatialMappingFixture buildMappedBuiltinSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const adg::BuiltinTargetScale &scale, mlir::MLIRContext &context,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    BlobStore &blobs,
    MappedRtlRouteCoverage routeCoverage = MappedRtlRouteCoverage::AnyLegal);

llvm::Expected<MappedSpatialMappingRepairFixture>
rerouteMappedSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const MappedSpatialMappingFixture &parent,
    const dse::SpatialTransportRepairAlternative &alternative,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    BlobStore &blobs);

fabric::FinalizedFabricRoot
buildMappedBuiltinSystemFixture(llvm::StringRef test,
                                const fabric::FinalizedFabricRoot &module,
                                ArtifactStore &artifacts);
fabric::FinalizedFabricRoot buildMappedBuiltinSystemFixture(
    llvm::StringRef test, const adg::BuiltinTargetScale &scale,
    const fabric::FinalizedFabricRoot &module, ArtifactStore &artifacts);

MappedSpatialHardwareFixture buildMappedSpatialHardwareFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    mlir::MLIRContext &context, ArtifactStore &artifacts, BlobStore &blobs,
    deployment::test::MappedSpatialSystemSpec systemSpec,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable,
    MappedRtlRouteCoverage routeCoverage = MappedRtlRouteCoverage::BypassFifo,
    MappedSystemInterconnect interconnect = MappedSystemInterconnect::None,
    std::size_t spatialMemoryOccurrenceCount = 1);

MappedRtlRequestFixture buildMappedRtlRequestFixture(
    llvm::StringRef test, llvm::StringRef stableSimulatorBuildIdentity,
    ArtifactStore &artifacts, BlobStore &blobs,
    const deployment::test::TemporaryTree &tree,
    MappedRtlFixtureTopology topology =
        MappedRtlFixtureTopology::HeterogeneousPortable);

} // namespace loom::eda::test

#endif // LOOM_TEST_EDA_MAPPEDRTLSIMULATIONTESTSUPPORT_H
