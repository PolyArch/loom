#ifndef LOOM_DIALECT_FABRIC_FABRICOPS_H
#define LOOM_DIALECT_FABRIC_FABRICOPS_H

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "loom/Dialect/Fabric/FabricOps.h.inc"

#endif
