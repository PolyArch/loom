#ifndef LOOM_LIB_PNR_SPATIALCANDIDATEOPERANDPAIRINGPREFERENCE_H
#define LOOM_LIB_PNR_SPATIALCANDIDATEOPERANDPAIRINGPREFERENCE_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr::detail {

/// Scores one Temporal operand attachment against already selected members of
/// its Dataflow-owned pairing groups. The result is analytic pressure only;
/// structural relations and final Mapping verification remain authoritative.
llvm::Expected<std::uint64_t> scoreSpatialOperandPairingAttachment(
    const FrozenSpatialPnrProblem &problem, PnrIndex demand,
    PnrIndex attachmentOption,
    llvm::ArrayRef<PnrIndex> selectedAttachmentOptions);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALCANDIDATEOPERANDPAIRINGPREFERENCE_H
