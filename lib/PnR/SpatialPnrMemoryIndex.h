#ifndef LOOM_LIB_PNR_SPATIALPNRMEMORYINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRMEMORYINDEX_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr {

class FrozenSpatialMemoryIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialMemoryIndex>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::mapping::TechMappingView &techMapping,
        const ::loom::fabric::FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations);

  static llvm::Error verify(const FrozenSpatialMemoryIndex &memory,
                            const FrozenSpatialRealizationIndex &realizations);
};

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALPNRMEMORYINDEX_H
