#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_H
#define LOOM_DIALECT_DATAFLOW_DATAFLOWOPS_H

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "loom/Dialect/Dataflow/DataflowOps.h.inc"

#endif
