#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"

using namespace loom::dataflow;

#include "loom/Dialect/Dataflow/DataflowDialect.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "loom/Dialect/Dataflow/DataflowOps.cpp.inc"
      >();
}
