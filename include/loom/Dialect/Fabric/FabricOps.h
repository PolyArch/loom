//===-- FabricOps.h - Fabric dialect operations ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICOPS_H
#define LOOM_DIALECT_FABRIC_FABRICOPS_H

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "FabricOps.h.inc"

#endif // LOOM_DIALECT_FABRIC_FABRICOPS_H
