//===- DataflowDialect.cpp - Dataflow dialect -----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Dataflow/IR/DataflowDialect.h"
#include "loom/Dialect/Dataflow/IR/DataflowOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace loom::dataflow;

#include "DataflowDialect.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DataflowOps.cpp.inc"
      >();
}
