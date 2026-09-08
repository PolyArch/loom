#ifndef LOOM_FRONTEND_LOWERING_GRAPH_EVENT_DEPENDENCIES_H
#define LOOM_FRONTEND_LOWERING_GRAPH_EVENT_DEPENDENCIES_H

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace loom::lowering {

// Keep the frontier events not already implied by another event's causal chain.
::llvm::SmallVector<::mlir::Value, 4> reduceEvents(::mlir::ValueRange inputs);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPH_EVENT_DEPENDENCIES_H
