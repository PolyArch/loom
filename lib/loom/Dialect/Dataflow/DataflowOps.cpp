#include "loom/Dialect/Dataflow/DataflowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace loom::dataflow;

#define GET_OP_CLASSES
#include "loom/Dialect/Dataflow/DataflowOps.cpp.inc"
