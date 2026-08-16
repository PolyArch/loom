#ifndef LOOM_DSE_PRODUCTIONOWNERS_H
#define LOOM_DSE_PRODUCTIONOWNERS_H

#include "llvm/Support/Error.h"

namespace loom::dse {

/// Registers the complete production Evaluation, candidate-generator, and
/// promotion-acquisition owner set. Exact re-registration is idempotent.
llvm::Error registerProductionDseOwners();

} // namespace loom::dse

#endif // LOOM_DSE_PRODUCTIONOWNERS_H
