#ifndef LOOM_LIB_PNR_SPATIALTAGASSIGNMENTSTATE_H
#define LOOM_LIB_PNR_SPATIALTAGASSIGNMENTSTATE_H

#include "Fabric/IR/PhysicalTag.h"
#include "PnR/SpatialTagAssignment.h"

#include "SpatialSwitchRowPacking.h"
#include "SpatialTagColoring.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::pnr::detail {

class SpatialTagConstraintModel;

/// Sorted-by-value occupancy of one match domain. Domains hold few distinct
/// values, so ordered flat storage beats hashing and keeps iteration
/// canonical. The entry shape mirrors the map interface its consumers use.
class SpatialTagDomainOccupancy final {
public:
  using Entry = std::pair<llvm::APInt, std::vector<SpatialTagVertexRef>>;
  using iterator = std::vector<Entry>::iterator;
  using const_iterator = std::vector<Entry>::const_iterator;

  iterator begin() { return entries_.begin(); }
  iterator end() { return entries_.end(); }
  const_iterator begin() const { return entries_.begin(); }
  const_iterator end() const { return entries_.end(); }
  std::size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

  iterator find(const llvm::APInt &value) {
    const auto found = lowerBound(value);
    if (found != entries_.end() &&
        ::fabric::comparePhysicalTagValues(found->first, value) == 0)
      return found;
    return entries_.end();
  }
  const_iterator find(const llvm::APInt &value) const {
    return const_cast<SpatialTagDomainOccupancy *>(this)->find(value);
  }
  std::vector<SpatialTagVertexRef> &operator[](const llvm::APInt &value) {
    auto found = lowerBound(value);
    if (found == entries_.end() ||
        ::fabric::comparePhysicalTagValues(found->first, value) != 0)
      found = entries_.insert(found, {value, {}});
    return found->second;
  }
  void erase(iterator entry) { entries_.erase(entry); }

private:
  iterator lowerBound(const llvm::APInt &value) {
    return std::lower_bound(entries_.begin(), entries_.end(), value,
                            [](const Entry &entry, const llvm::APInt &target) {
                              return ::fabric::comparePhysicalTagValues(
                                         entry.first, target) < 0;
                            });
  }

  std::vector<Entry> entries_;
};

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
