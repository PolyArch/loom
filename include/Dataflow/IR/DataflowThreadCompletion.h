#ifndef DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
#define DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace dataflow {

bool completionEventCovers(mlir::Value terminal, mlir::Value event);

llvm::SmallVector<mlir::Value, 4>
computeMinimalThreadCompletionFrontier(mlir::ValueRange candidates);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
