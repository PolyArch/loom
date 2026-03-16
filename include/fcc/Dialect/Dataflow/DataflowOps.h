#ifndef FCC_DIALECT_DATAFLOW_DATAFLOWOPS_H
#define FCC_DIALECT_DATAFLOW_DATAFLOWOPS_H

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "fcc/Dialect/Dataflow/DataflowOps.h.inc"

#endif
