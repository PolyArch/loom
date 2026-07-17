#ifndef DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
#define DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H

#include "Dataflow/IR/DataflowOps.h"

#include "llvm/Support/Error.h"

namespace dataflow {

llvm::Error validateFinalizedGraph(GraphFuncOp graph);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
