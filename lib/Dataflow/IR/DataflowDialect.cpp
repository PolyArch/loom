#include "Dataflow/IR/DataflowDialect.h"

#include "Dataflow/IR/DataflowOps.h"

using namespace mlir;
using namespace dataflow;

#include "Dataflow/IR/DataflowDialect.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dataflow/IR/DataflowOps.cpp.inc"
      >();
}
