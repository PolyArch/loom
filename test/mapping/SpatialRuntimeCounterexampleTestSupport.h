#pragma once

namespace dataflow {
class CanonicalDataflowProgramView;
} // namespace dataflow

namespace loom {
class ArtifactStore;

namespace fabric {
class FabricArtifactView;
class FabricPhysicalTimingProfileView;
} // namespace fabric

namespace mapping {
class FinalizedSpatialMapping;
class FinalizedSpatialMappingConstraintSet;
class TechMappingView;
} // namespace mapping

namespace pnr {
class ResolvedPnrConfigView;
class SpatialCandidateState;
} // namespace pnr

namespace test {

void exerciseSpatialRuntimeCounterexampleNoGood(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const fabric::FabricArtifactView &foreignFabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parent,
    const mapping::FinalizedSpatialMapping &mapping,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store);

void exerciseSpatialPhysicalTagRuntimeCounterexampleNoGood(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parent,
    const mapping::FinalizedSpatialMapping &mapping,
    const pnr::ResolvedPnrConfigView &pnrConfig,
    pnr::SpatialCandidateState &candidate, const ArtifactStore &store);

} // namespace test
} // namespace loom
