#ifndef FABRIC_IR_RESULTPRESENTATION_H
#define FABRIC_IR_RESULTPRESENTATION_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <vector>

namespace fabric {

struct ResultPresentationPosition final {
  std::uint32_t requester;
  std::uint32_t evaluation;
};

/// Direct tuples have one priority position per producing context. A held lane
/// has one immediately evaluated position. This is presentation state only;
/// it neither acquires operation capacity nor commits a token.
std::vector<ResultPresentationPosition>
resultPresentationPositions(llvm::ArrayRef<std::uint32_t> evaluationCounts);

std::vector<std::uint32_t> resultPresentationOrder(std::uint32_t requesterCount,
                                                   std::uint32_t cursor);

struct ResultPresentationEvaluation final {
  bool eligible;
  bool evaluated;
};

struct ResultPresentationRequest final {
  /// A lane may fan out, but two lanes may not claim the same destination.
  llvm::SmallVector<llvm::APInt> laneDestinations;
  llvm::SmallVector<ResultPresentationEvaluation> evaluations;
};

struct ResultPresentation final {
  llvm::SmallBitVector selected;
  std::uint32_t nextCursor = 0;
};

/// Focus the next configured-eligible priority position, and present complete
/// disjoint tuples in cyclic physical-requester order from that focus. Hold
/// priority until its context is evaluated, even if other requesters present
/// opportunistically. Advance on that evaluation, including an empty/refused
/// offer. This prevents periodic dispatch/presentation phase starvation.
ResultPresentation
selectResultPresentation(llvm::ArrayRef<ResultPresentationRequest> requests,
                         std::uint32_t cursor);

} // namespace fabric

#endif // FABRIC_IR_RESULTPRESENTATION_H
