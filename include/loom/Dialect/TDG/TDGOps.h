#ifndef LOOM_DIALECT_TDG_TDGOPS_H
#define LOOM_DIALECT_TDG_TDGOPS_H

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "loom/Dialect/TDG/TDGOps.h.inc"

#endif
