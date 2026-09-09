#ifndef LOOM_FRONTEND_LOWERING_GRAPHPOINTERADDRESSING_H
#define LOOM_FRONTEND_LOWERING_GRAPHPOINTERADDRESSING_H

#include "Dataflow/IR/DataflowOps.h"

namespace loom::lowering {

// Expand typed GEP paths before region lowering assigns execution controls to
// their constants. Every remaining GEP adds one A(AS)-bit byte offset.
mlir::LogicalResult normalizeGraphPointerAddresses(dataflow::GraphOp graph);

} // namespace loom::lowering

#endif
