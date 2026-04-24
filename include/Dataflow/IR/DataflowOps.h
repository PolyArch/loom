#ifndef DATAFLOW_IR_DATAFLOWOPS_H
#define DATAFLOW_IR_DATAFLOWOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.h.inc"

#endif // DATAFLOW_IR_DATAFLOWOPS_H
