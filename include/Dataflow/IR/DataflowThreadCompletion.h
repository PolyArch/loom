#ifndef DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
#define DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace dataflow {

class ThreadCompletionCoverageAnalysis {
public:
  bool covers(mlir::Value terminal, mlir::Value event);

  bool isFrontierMemberNecessary(mlir::ValueRange frontier,
                                 unsigned memberIndex,
                                 mlir::ValueRange graphLaunchCompletions);

  llvm::SmallVector<mlir::Value, 4>
  computeMinimalFrontier(mlir::ValueRange candidates,
                         mlir::ValueRange graphLaunchCompletions);

private:
  llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, bool> coverage;
};

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWTHREADCOMPLETION_H
