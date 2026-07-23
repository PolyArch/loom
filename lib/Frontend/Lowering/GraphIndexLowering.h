#ifndef LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H

#include "Dataflow/IR/DataflowOps.h"

namespace loom {
namespace lowering {

// `indexBits` is the canonical index width the caller's pass boundary already
// resolved; index-domain rewriting never resolves it again.
void lowerGraphIndexDomains(::dataflow::GraphOp graph, unsigned indexBits);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H
