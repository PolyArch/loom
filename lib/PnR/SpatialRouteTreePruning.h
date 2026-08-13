#ifndef LOOM_LIB_PNR_SPATIALROUTETREEPRUNING_H
#define LOOM_LIB_PNR_SPATIALROUTETREEPRUNING_H

#include "PnR/PnrIndex.h"
#include "PnR/SpatialNetRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <vector>

namespace loom::pnr {

class FrozenSpatialPnrProblem;
class SpatialCandidateState;
class SpatialRouteCostState;

namespace detail {

class SpatialRouteTreePruningScratch final {
public:
  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<SpatialNegotiatedRoutePlan>
  project(const SpatialCandidateState &candidate,
          const SpatialRouteCostState &costs, PnrIndex logicalNet);

  std::size_t retainedStorageBytes() const;

private:
  bool traversalOverused(const SpatialCandidateState &candidate,
                         const SpatialRouteCostState &costs,
                         PnrIndex traversal);

  std::vector<std::uint64_t> traversalEpochs_;
  std::vector<std::uint8_t> traversalOveruse_;
  std::vector<std::uint64_t> nodeEpochs_;
  std::vector<std::uint8_t> nodeAffected_;
  std::vector<std::uint8_t> sinkAffected_;
  std::vector<PnrIndex> selectedSinks_;
  std::vector<PnrIndex> pathSlots_;
  std::uint64_t projectionEpoch_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALROUTETREEPRUNING_H
