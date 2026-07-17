#ifndef DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
#define DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace dataflow {

llvm::Error validateFinalizedGraph(GraphOp graph);
llvm::Error validateFinalizedProgram(::mlir::ModuleOp module);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
