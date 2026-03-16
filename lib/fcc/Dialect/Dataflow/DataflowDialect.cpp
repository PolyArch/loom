#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Dataflow/DataflowOps.h"

using namespace fcc::dataflow;

#include "fcc/Dialect/Dataflow/DataflowDialect.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "fcc/Dialect/Dataflow/DataflowOps.cpp.inc"
      >();
}
