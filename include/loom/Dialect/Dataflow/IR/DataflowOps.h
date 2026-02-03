//===- DataflowOps.h - Dataflow operations --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_DATAFLOW_IR_DATAFLOWOPS_H
#define LOOM_DIALECT_DATAFLOW_IR_DATAFLOWOPS_H

#include "loom/Dialect/Dataflow/IR/DataflowDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

#endif // LOOM_DIALECT_DATAFLOW_IR_DATAFLOWOPS_H
