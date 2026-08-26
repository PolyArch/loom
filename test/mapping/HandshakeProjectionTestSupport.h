#ifndef LOOM_TEST_MAPPING_HANDSHAKEPROJECTIONTESTSUPPORT_H
#define LOOM_TEST_MAPPING_HANDSHAKEPROJECTIONTESTSUPPORT_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::test {

void exerciseDenseHandshakeProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseDenseHandshakeFixedArcProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

void exerciseDenseHandshakeCycleProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_HANDSHAKEPROJECTIONTESTSUPPORT_H
