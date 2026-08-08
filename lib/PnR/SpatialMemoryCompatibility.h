#ifndef LOOM_LIB_PNR_SPATIALMEMORYCOMPATIBILITY_H
#define LOOM_LIB_PNR_SPATIALMEMORYCOMPATIBILITY_H

#include "PnR/SpatialPnrProblem.h"

#include <algorithm>
#include <variant>

namespace loom::pnr::detail {

inline bool memoryDispatchMatchesTarget(
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialMemoryDispatchOption &option,
    const FrozenSpatialMemoryBindingTargetOption &target) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *local =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
    if (!local || local->underlying() != region->service)
      return false;
    const auto regions = memory.dispatchServiceRegionOrdinals().slice(
        option.serviceRegionOffset, option.serviceRegionCount);
    return std::binary_search(regions.begin(), regions.end(), region->ordinal);
  }
  return std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
      option.target);
}

inline bool memoryExposureMatchesTarget(
    const FrozenSpatialMemoryBindingTargetOption &target,
    const FrozenSpatialMemoryExposureOption &option) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *local =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
    return local && local->underlying() == region->service;
  }
  return std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
      option.target);
}

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALMEMORYCOMPATIBILITY_H
