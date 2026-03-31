#ifndef FCC_DIALECT_FABRIC_FABRICOPS_H
#define FCC_DIALECT_FABRIC_FABRICOPS_H

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "fcc/Dialect/Fabric/FabricOps.h.inc"

#endif
