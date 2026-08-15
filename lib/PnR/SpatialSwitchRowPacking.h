#ifndef LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H
#define LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

namespace loom::pnr::detail {

struct SpatialTemporalSwitchInputSignature final {
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  ::loom::fabric::FabricOrdinal input = 0;
  std::vector<::loom::fabric::FabricOrdinal> outputs;
  std::vector<PnrIndex> traversals;
};

/// One continuity segment's complete row demand in one Temporal switch table.
/// Multiple input signatures are retained so re-entry cannot be mistaken for
/// one widened crossbar selection.
struct SpatialTemporalSwitchSegmentDemand final {
  PnrIndex domain = 0;
  PnrIndex logicalNet = 0;
  PnrIndex segment = 0;
  std::vector<SpatialTemporalSwitchInputSignature> signatures;
};

/// Exact segment interference after applying Temporal switch row compatibility.
/// Non-switch match domains remain cliques. The CSR is symmetric and excludes
/// self edges.
class SpatialTagInterferenceProjection final {
public:
  llvm::ArrayRef<PnrIndex> netSegmentOffsets() const {
    return netSegmentOffsets_;
  }
  llvm::ArrayRef<PnrIndex> conflictOffsets() const { return conflictOffsets_; }
  llvm::ArrayRef<PnrIndex> conflicts() const { return conflicts_; }
  llvm::ArrayRef<PnrIndex> conflicts(PnrIndex vertex) const;
  bool interferes(PnrIndex lhs, PnrIndex rhs) const;
  bool interferes(PnrIndex domain, PnrIndex lhs, PnrIndex rhs) const;
  std::size_t retainedStorageBytes() const;

private:
  std::vector<PnrIndex> netSegmentOffsets_;
  std::vector<PnrIndex> conflictOffsets_;
  std::vector<PnrIndex> conflicts_;
  std::vector<std::uint8_t> temporalSwitchDomains_;
  std::set<std::tuple<PnrIndex, PnrIndex, PnrIndex>> compatibleSwitchPairs_;

  friend llvm::Expected<SpatialTagInterferenceProjection>
  deriveSpatialTagInterference(
      const FrozenSpatialPnrProblem &problem,
      llvm::ArrayRef<const RouteTreeState *> routes,
      llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);
};

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);

llvm::Expected<std::vector<SpatialTemporalSwitchSegmentDemand>>
deriveSpatialTemporalSwitchSegmentDemands(
    const FrozenSpatialPnrProblem &problem, PnrIndex logicalNet,
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity);

bool compatibleSpatialTemporalSwitchDemands(
    const SpatialTemporalSwitchSegmentDemand &lhs,
    const SpatialTemporalSwitchSegmentDemand &rhs);

llvm::Expected<SpatialTagInterferenceProjection> deriveSpatialTagInterference(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<const SpatialTagContinuityProjection *> continuity);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALSWITCHROWPACKING_H
