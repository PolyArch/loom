#ifndef DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
#define DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace dataflow {

bool completionEventCovers(mlir::Value terminal, mlir::Value event);

bool isThreadCompletionFrontierMemberNecessary(
    mlir::ValueRange frontier, unsigned memberIndex,
    mlir::ValueRange graphLaunchCompletions);

llvm::SmallVector<mlir::Value, 4>
computeMinimalThreadCompletionFrontier(mlir::ValueRange candidates,
                                       mlir::ValueRange graphLaunchCompletions);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
