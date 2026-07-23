#include "Dataflow/IR/DataflowDialect.h"

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dataflow;

#include "Dataflow/IR/DataflowDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dataflow/IR/DataflowTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dataflow/IR/DataflowAttrs.cpp.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dataflow/IR/DataflowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dataflow/IR/DataflowTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dataflow/IR/DataflowAttrs.cpp.inc"
      >();

  attachCanonicalDataflowActorInterfaces(*getContext());
}
