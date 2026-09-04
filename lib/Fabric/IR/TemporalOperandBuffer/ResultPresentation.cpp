#include "Fabric/IR/ResultPresentation.h"

#include <cassert>
#include <limits>

namespace fabric {

std::vector<ResultPresentationPosition>
resultPresentationPositions(llvm::ArrayRef<std::uint32_t> evaluationCounts) {
  std::vector<ResultPresentationPosition> positions;
  for (std::uint32_t requester = 0; requester != evaluationCounts.size();
       ++requester) {
    assert(evaluationCounts[requester] != 0);
    for (std::uint32_t evaluation = 0;
         evaluation != evaluationCounts[requester]; ++evaluation)
      positions.push_back({requester, evaluation});
  }
  assert(!positions.empty() &&
         positions.size() <= std::numeric_limits<std::uint32_t>::max());
  return positions;
}

std::vector<std::uint32_t> resultPresentationOrder(std::uint32_t requesterCount,
                                                   std::uint32_t cursor) {
  assert(requesterCount != 0 && cursor < requesterCount);
  std::vector<std::uint32_t> order;
  order.reserve(requesterCount);
  for (std::uint32_t offset = 0; offset != requesterCount; ++offset)
    order.push_back(static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(cursor) + offset) % requesterCount));
  return order;
}

ResultPresentation
selectResultPresentation(llvm::ArrayRef<ResultPresentationRequest> requests,
                         std::uint32_t cursor) {
  assert(!requests.empty() && !requests.front().laneDestinations.empty());
  llvm::SmallVector<std::uint32_t> evaluationCounts;
  for (const auto &request : requests)
    evaluationCounts.push_back(request.evaluations.size());
  const auto positions = resultPresentationPositions(evaluationCounts);
  ResultPresentation result{llvm::SmallBitVector(requests.size()), cursor};
  for (std::uint32_t focus :
       resultPresentationOrder(positions.size(), cursor)) {
    const auto position = positions[focus];
    const auto evaluation =
        requests[position.requester].evaluations[position.evaluation];
    if (!evaluation.eligible)
      continue;
    if (evaluation.evaluated)
      result.nextCursor = (focus + 1) % positions.size();
    llvm::APInt occupied(
        requests.front().laneDestinations.front().getBitWidth(), 0);
    for (std::uint32_t requester :
         resultPresentationOrder(requests.size(), position.requester)) {
      llvm::APInt claims(occupied.getBitWidth(), 0);
      bool distinctLanes = true;
      for (const llvm::APInt &lane : requests[requester].laneDestinations) {
        assert(lane.getBitWidth() == occupied.getBitWidth());
        distinctLanes &= (claims & lane).isZero();
        claims |= lane;
      }
      if (!distinctLanes || claims.isZero() || !(claims & occupied).isZero())
        continue;
      result.selected.set(requester);
      occupied |= claims;
    }
    return result;
  }
  return result;
}

} // namespace fabric
