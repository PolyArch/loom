#ifndef LOOM_LIB_PNR_SPATIALROUTECOSTSTATEINTERNAL_H
#define LOOM_LIB_PNR_SPATIALROUTECOSTSTATEINTERNAL_H

#include "PnR/SpatialRouteCostState.h"

#include "SpatialSwitchRowPacking.h"

#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <vector>

namespace loom::pnr::detail {

struct SpatialRouteCostSwitchRowState final {
  struct DemandJournal final {
    PnrIndex logicalNet = 0;
    std::vector<SpatialTemporalSwitchSegmentDemand> demands;
    std::uint8_t settled = 0;
  };

  bool enabled = false;
  std::vector<std::vector<SpatialTemporalSwitchSegmentDemand>> netDemands;
  std::vector<std::uint8_t> netDemandsSettled;
  std::vector<SpatialTemporalSwitchSegmentDemand> selectedNetDemands;
  std::vector<DemandJournal> demandJournal;

  /// Reused per-update scratch for the selected-net marginal row projection.
  /// The demand references and signature views are rebuilt on every update;
  /// the storage keeps its capacity across proposals.
  struct SelectedDemandRef final {
    const SpatialTemporalSwitchSegmentDemand *route = nullptr;
    const llvm::APInt *tag = nullptr;
  };
  std::vector<std::vector<SelectedDemandRef>> updateDomainDemands;
  std::vector<PnrIndex> updateTouchedDomains;
  std::vector<std::uint64_t> updateDomainMarks;
  std::uint64_t updateEpoch = 0;
  std::vector<std::uint64_t> updateMarginalRows;
  std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>
      updateSignatureViews;
  std::vector<::loom::fabric::FabricTemporalSwitchCandidateRouteDemandView>
      updateDemandViews;
  std::vector<SpatialTagDomainUse> updateUses;
  SpatialTemporalSwitchDemandScratch demandScratch;

  std::size_t retainedStorageBytes() const;
};

inline llvm::Error routeCostStateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial route cost state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

inline std::uint64_t saturatedAdd(std::uint64_t lhs, std::uint64_t rhs) {
  return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
             ? std::numeric_limits<std::uint64_t>::max()
             : lhs + rhs;
}

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALROUTECOSTSTATEINTERNAL_H
