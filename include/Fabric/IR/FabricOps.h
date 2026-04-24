#ifndef FABRIC_IR_FABRICOPS_H
#define FABRIC_IR_FABRICOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Fabric/IR/FabricTypes.h"

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.h.inc"

#endif // FABRIC_IR_FABRICOPS_H
