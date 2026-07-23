#ifndef DATAFLOW_IR_DATAFLOWATTRS_H
#define DATAFLOW_IR_DATAFLOWATTRS_H

#include "Dataflow/IR/DataflowEnums.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Dataflow/IR/DataflowAttrs.h.inc"

#endif // DATAFLOW_IR_DATAFLOWATTRS_H
