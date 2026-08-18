#ifndef LOOM_LIB_PNR_SPATIALPNRPROBLEMIDENTITY_H
#define LOOM_LIB_PNR_SPATIALPNRPROBLEMIDENTITY_H

#include "Common/ComponentViewDigest.h"
#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

class SpatialPnrProblemIdentity final {
public:
  static llvm::Error validateInputs(
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const ::loom::mapping::TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const ResolvedPnrConfigView &config,
      const ::loom::mapping::SpatialMappingConstraintSetView &constraintSet);

  static FrozenSpatialPnrCacheKey deriveCacheKey(
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const ::loom::mapping::TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const ResolvedPnrConfigView &config,
      const ::loom::mapping::SpatialMappingConstraintSetView &constraintSet,
      const ComponentViewDigest &physicalTimingDigest);

  static llvm::Error revalidateCacheHit(
      const FrozenSpatialPnrProblem &problem,
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const ::loom::mapping::TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
      const ResolvedPnrConfigView &config,
      const ::loom::mapping::SpatialMappingConstraintSetView &constraintSet);
};

} // namespace loom::pnr::detail

#endif
