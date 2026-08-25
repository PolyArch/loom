#ifndef LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
#define LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialPnrProblem.h"

#include <cstdint>

namespace mlir {
class MLIRContext;
}

namespace loom::test {

void exerciseHandshakeCandidateRefcounts(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

adg::FinalizedFabricDesign
buildTemporalCapacityFabric(const ArtifactStore &store);

adg::FinalizedFabricDesign
buildTemporalSwitchPackingFabric(const ArtifactStore &store,
                                 std::uint64_t residentRows = 2);

ResolvedConfig buildSpatialPnrTestResolvedConfig();

mapping::FinalizedSpatialMappingConstraintSet buildSpatialMappingConstraints(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store,
    bool restrictTagsToZero = false, bool rejectComputePlacement = false);

void exerciseCapacityOveruseCandidate(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseCombinedCapacityProjection(
    const fabric::FabricArtifactView &fabric);

void exerciseCapacityExactRepairNoMutation(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialExactRepairResultKind expected);

void exerciseTemporalComputeUseProjection(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseCanonicalCandidateInitialization(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseSpatialInitializerDiversification(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseSpatialActionDomainAndObjective(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseSpatialProgressWitnessClosure(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseSpatialAnnealingReplay(
    const pnr::FrozenSpatialPnrProblemHandle &problem, bool warmScratch);

void exercisePathFinderFixedTerminalCutRejection(
    pnr::SpatialCandidateState &candidate,
    pnr::SpatialCandidateScratch &candidateScratch);

void exerciseSpatialActionSequence(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate, std::uint64_t proposalCount);

void exerciseSpatialMemoryActionDomain(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate);

void exerciseSpatialAttachmentConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store);

void exerciseSpatialRouteConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store);

void exerciseSpatialTagConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
