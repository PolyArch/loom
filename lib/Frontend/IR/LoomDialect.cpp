#include "Frontend/IR/LoomDialect.h"

#include "Frontend/IR/LoomOps.h"

using namespace mlir;
using namespace loom;

#include "Frontend/IR/LoomDialect.cpp.inc"

void LoomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Frontend/IR/LoomOps.cpp.inc"
      >();
}
