#ifndef LOOM_LIB_PNR_SPATIALTAGASSIGNMENTSTATE_H
#define LOOM_LIB_PNR_SPATIALTAGASSIGNMENTSTATE_H

#include "PnR/SpatialTagAssignment.h"

#include "SpatialSwitchRowPacking.h"
#include "SpatialTagColoring.h"

#include "llvm/ADT/DenseMap.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

class SpatialTagConstraintModel;

using SpatialTagDomainOccupancy =
    llvm::DenseMap<llvm::APInt, std::vector<SpatialTagVertexRef>>;

struct SpatialTagNetState final {
  SpatialTagContinuityProjection continuity;
  std::vector<std::optional<llvm::APInt>> values;
};

struct SpatialTagAssignmentScratchStorage final {
  const FrozenSpatialPnrProblem *problem = nullptr;
  std::vector<SpatialTagNetState> stagedNets;
  std::vector<std::vector<std::optional<llvm::APInt>>> stagedValues;
  std::vector<PnrIndex> touchedRoutes;
  std::vector<PnrIndex> routedNets;
  std::vector<PnrIndex> valueOnlyNets;
  std::vector<PnrIndex> rebuiltNets;
  std::vector<PnrIndex> synchronizedNets;
  std::vector<PnrIndex> changedDomains;
  SpatialTagContinuityScratch continuityScratch;
  SpatialTagInterferenceUpdateScratch interferenceScratch;
  SpatialTagColoringCache stagedColoringCache;
  bool coloringCacheActive = false;
  bool active = false;
};

struct SpatialTagAssignmentStateStorage final {
  const FrozenSpatialPnrProblem *problem = nullptr;
  const SpatialTagConstraintModel *constraints = nullptr;
  bool hasTaggedTransport = false;
  std::vector<SpatialTagNetState> nets;
  std::vector<SpatialTagDomainOccupancy> occupancy;
  SpatialTagInterferenceProjection interference;
  SpatialTagColoringCache coloringCache;
  std::vector<PnrIndex> residentCounts;
  std::vector<std::uint8_t> classBuilt;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
  std::uint64_t residentCapacityOveruse = 0;
};

inline llvm::ArrayRef<PnrIndex> tagSegmentDomains(const SpatialTagNetState &net,
                                                  PnrIndex segment) {
  const auto offsets = net.continuity.segmentDomainOffsets();
  assert(segment + 1 < offsets.size());
  return net.continuity.segmentDomains().slice(
      offsets[segment], offsets[segment + 1] - offsets[segment]);
}
std::uint64_t
tagDomainConflictCount(llvm::ArrayRef<SpatialTagDomainOccupancy> occupancy,
                       const SpatialTagInterferenceProjection &interference,
                       PnrIndex domain);

llvm::Expected<SpatialTagAssignmentSummary>
summarizeTagAssignmentState(const SpatialTagAssignmentStateStorage &storage,
                            bool includeDomainDetails);
llvm::Expected<SpatialTagAssignmentDelta>
summarizeTagAssignmentDelta(const SpatialTagAssignmentStateStorage &storage,
                            llvm::ArrayRef<PnrIndex> logicalNets,
                            llvm::ArrayRef<PnrIndex> changedDomains);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALTAGASSIGNMENTSTATE_H
