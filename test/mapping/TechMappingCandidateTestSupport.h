#ifndef LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
#define LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "PnR/SpatialPnrProblem.h"

namespace loom::test {

void exerciseHandshakeCandidateRefcounts(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

adg::FinalizedFabricDesign
buildTemporalCapacityFabric(const ArtifactStore &store);

void exerciseCapacityOveruseCandidate(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
