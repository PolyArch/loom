#include "fcc/Dialect/Dataflow/DataflowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace fcc::dataflow;

#define GET_OP_CLASSES
#include "fcc/Dialect/Dataflow/DataflowOps.cpp.inc"
