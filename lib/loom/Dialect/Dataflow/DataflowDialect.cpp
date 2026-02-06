//===-- DataflowDialect.cpp - Dataflow dialect registration -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dataflow dialect registration and initialization,
// adding all dataflow operations to the dialect.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace loom::dataflow;

#include "DataflowDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DataflowOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "DataflowTypes.cpp.inc"
      >();
}
