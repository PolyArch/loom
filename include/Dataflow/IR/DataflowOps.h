#ifndef DATAFLOW_IR_DATAFLOWOPS_H
#define DATAFLOW_IR_DATAFLOWOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEnums.h"

namespace dataflow {

enum class GraphPortKind { Value, Stream, Memory };

} // namespace dataflow

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.h.inc"

#endif // DATAFLOW_IR_DATAFLOWOPS_H
