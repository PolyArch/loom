#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/Dialect/TDG/TDGTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace loom::tdg;

#include "loom/Dialect/TDG/TDGDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "loom/Dialect/TDG/TDGTypes.cpp.inc"

void TDGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "loom/Dialect/TDG/TDGOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "loom/Dialect/TDG/TDGTypes.cpp.inc"
      >();
}
