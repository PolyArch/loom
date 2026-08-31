#ifndef LOOM_TEST_MAPPING_SPATIALRUNTIMECOUNTEREXAMPLEEXACTREPAIRTESTSUPPORT_H
#define LOOM_TEST_MAPPING_SPATIALRUNTIMECOUNTEREXAMPLEEXACTREPAIRTESTSUPPORT_H

namespace dataflow {
class CanonicalDataflowProgramView;
}

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
} // namespace pnr

namespace test {

/// Exercises literal-breaking exact repair from a finalized non-tagged parent
/// Mapping. Every child is reconstructed from the exact parent Mapping, not a
/// canonical cold seed, and is independently finalized and admitted.
void exerciseSpatialRuntimeCounterexampleExactRepair(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints,
    const mapping::FinalizedSpatialMapping &parentMapping,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store);

/// Exercises exact route-local Physical Tag repair and the source-local,
/// sink-local, and sink-qualified traversal positions of a tagged parent.
void exerciseSpatialTaggedRuntimeCounterexampleExactRepair(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints,
    const mapping::FinalizedSpatialMapping &parentMapping,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store);

/// Builds a two-actor Temporal PE fixture with a real register FIFO, publishes
/// an external parent route, then proves that exact repair can break the
/// no-good by committing the ordinary external-to-register-FIFO action.
void exerciseSpatialRegisterFifoRuntimeCounterexampleExactRepair();

} // namespace test
} // namespace loom

#endif // LOOM_TEST_MAPPING_SPATIALRUNTIMECOUNTEREXAMPLEEXACTREPAIRTESTSUPPORT_H
