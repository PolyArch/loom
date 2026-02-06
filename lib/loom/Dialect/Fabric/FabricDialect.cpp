//===-- FabricDialect.cpp - Fabric dialect registration ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::fabric;

#include "FabricDialect.cpp.inc"

#define GET_OP_CLASSES
#include "FabricOps.cpp.inc"

void FabricDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "FabricOps.cpp.inc"
      >();
}
