#ifndef DATAFLOW_IR_DATAFLOWINTERFACES_H
#define DATAFLOW_IR_DATAFLOWINTERFACES_H

#include "Dataflow/IR/OperationSchema.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "Dataflow/IR/DataflowInterfaces.h.inc"

namespace dataflow {

/// Attaches the canonical actor projection to every registered operation that
/// does not already implement it directly. The attachment is expanded from the
/// one generated registry, so it can never diverge from the registration and
/// classification authority.
void attachCanonicalDataflowActorInterfaces(::mlir::MLIRContext &context);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWINTERFACES_H
