#ifndef DATAFLOW_IR_DATAFLOWINTERFACES_H
#define DATAFLOW_IR_DATAFLOWINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/MLIRContext.h"

#include "Dataflow/IR/DataflowInterfaces.h.inc"

namespace dataflow {

void attachCanonicalDataflowActorInterfaces(::mlir::MLIRContext &context);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWINTERFACES_H
