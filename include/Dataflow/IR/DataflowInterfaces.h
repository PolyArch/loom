#ifndef DATAFLOW_IR_DATAFLOWINTERFACES_H
#define DATAFLOW_IR_DATAFLOWINTERFACES_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include <optional>

#include "Dataflow/IR/DataflowInterfaces.h.inc"

namespace dataflow {

enum class CanonicalDataflowActorKind { Compute, Control, Memory };

void attachCanonicalDataflowActorInterfaces(::mlir::MLIRContext &context);

std::optional<CanonicalDataflowActorKind>
classifyCanonicalDataflowActor(::mlir::Operation *op);

bool isCanonicalDataflowActor(::mlir::Operation *op);
bool isCanonicalDataflowActor(::mlir::Operation *op,
                              CanonicalDataflowActorKind kind);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWINTERFACES_H
