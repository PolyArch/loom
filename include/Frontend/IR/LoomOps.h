#ifndef LOOM_FRONTEND_IR_LOOMOPS_H
#define LOOM_FRONTEND_IR_LOOMOPS_H

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Frontend/IR/LoomDialect.h"

#define GET_OP_CLASSES
#include "Frontend/IR/LoomOps.h.inc"

#endif // LOOM_FRONTEND_IR_LOOMOPS_H
