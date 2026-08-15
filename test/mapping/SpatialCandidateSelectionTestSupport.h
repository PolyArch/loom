#pragma once

#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace loom::pnr {
class SpatialCandidateState;
class SpatialMoveTransaction;
} // namespace loom::pnr

namespace loom::test {

llvm::Error selectReachableGraphBoundaries(
    pnr::SpatialCandidateState &candidate, pnr::SpatialMoveTransaction &move,
    llvm::ArrayRef<pnr::PnrIndex> selectedPortAttachments = {},
    bool requireDistinctEndpoints = false);

} // namespace loom::test
