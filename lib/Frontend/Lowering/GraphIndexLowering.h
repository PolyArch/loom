#ifndef LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H

#include "Dataflow/IR/DataflowOps.h"

namespace loom {
namespace lowering {

void lowerGraphIndexDomains(::dataflow::GraphFuncOp graph);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPHINDEXLOWERING_H
