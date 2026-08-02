#ifndef LOOM_LIB_PNR_FROZENMODELINTERNAL_H
#define LOOM_LIB_PNR_FROZENMODELINTERNAL_H

#include "PnR/FrozenModel.h"

namespace loom::pnr::detail {

class FrozenModelBuilder final {
public:
  static llvm::Expected<FrozenModelCacheKey>
  deriveCacheKey(const PnrProblemInputs &inputs);

  static llvm::Expected<FrozenModelHandle>
  build(const PnrProblemInputs &inputs);

  static llvm::Error revalidateCacheHit(const FrozenModel &model,
                                        const PnrProblemInputs &inputs);

  static llvm::Expected<FrozenRealizationGraph>
  buildRealizations(const PnrProblemInputs &inputs);

  static llvm::Expected<FrozenRoutingGraph>
  buildRouting(const PnrProblemInputs &inputs);

private:
  static FrozenModelCacheKey
  deriveValidatedCacheKey(const PnrProblemInputs &inputs);
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_FROZENMODELINTERNAL_H
