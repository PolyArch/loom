#ifndef LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
#define LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::test {

void exerciseHandshakeCandidateRefcounts(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_TECHMAPPINGCANDIDATETESTSUPPORT_H
