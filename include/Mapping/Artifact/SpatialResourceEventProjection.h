#ifndef LOOM_MAPPING_ARTIFACT_SPATIALRESOURCEEVENTPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SPATIALRESOURCEEVENTPROJECTION_H

#include "Mapping/Artifact/MappingProgressProjection.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

namespace loom::mapping {

/// A physical result handoff retains the selected Spatial source. It is not
/// the producing actor's logical transition. Destinations are a removable
/// projection of the same immutable Mapping, never a second event catalog.
struct RootedSpatialResultHandoffProjection final {
  ::dataflow::RootedGraphLaunchRef graph;
  SpatialComputeResultHandoffView result;
};

using MappingCausalReleaseEventProjection =
    std::variant<std::vector<::dataflow::EventFamilyKey>,
                 RootedSpatialResultHandoffProjection>;

struct MappingCausalReleasePointProjection final {
  MappingCausalReleaseEventProjection event;
  std::optional<std::vector<std::uint8_t>> guaranteedOffset;
};

llvm::Expected<std::vector<::dataflow::EventFamilyKey>>
projectRootedSpatialActivityEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    const SpatialActivityEventRef &event);

llvm::Expected<::dataflow::GraphRef> resolveSpatialActivityEventGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SpatialActivityEventRef &event);

llvm::Expected<std::vector<MappingCausalReleasePointProjection>>
projectRootedSpatialCausalRelease(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<SpatialEventPointView> release,
    llvm::ArrayRef<SpatialComputeResultHandoffView> resultHandoffs);

/// Logical events required before a release can occur. A selected durable
/// handoff does not require later consumer firing; its capacity and progress
/// obligations remain owned by the selected storage projection.
llvm::Expected<std::vector<MappingProgressCausalReleaseProjection>>
projectMappingCausalReleasePrerequisites(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<MappingCausalReleasePointProjection> release);

/// Canonical source key shared by closure ordering and Runtime image
/// derivation. The selected Mapping owns the removable handoff destinations.
llvm::Expected<std::vector<std::uint8_t>> encodeMappingCausalReleaseEventKey(
    const ArtifactIdentity &dataflowIdentity,
    const MappingCausalReleaseEventProjection &event);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALRESOURCEEVENTPROJECTION_H
