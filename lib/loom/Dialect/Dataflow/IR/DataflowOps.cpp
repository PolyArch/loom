//===-- DataflowOps.cpp - Dataflow dialect operation verifiers --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements verification logic for Dataflow dialect operations.
// Each operation verifies type constraints: CarryOp checks matching a/b types,
// InvariantOp checks result matches input, StreamOp requires index-typed
// operands, and GateOp validates condition and value types.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::dataflow;

#define GET_OP_CLASSES
#include "DataflowOps.cpp.inc"

LogicalResult CarryOp::verify() {
  if (!getD().getType().isInteger(1))
    return emitOpError("expects i1 control input");
  if (getA().getType() != getB().getType())
    return emitOpError("expects matching a and b types");
  if (getO().getType() != getA().getType())
    return emitOpError("expects result type to match a");
  return success();
}

LogicalResult InvariantOp::verify() {
  if (!getD().getType().isInteger(1))
    return emitOpError("expects i1 control input");
  if (getO().getType() != getA().getType())
    return emitOpError("expects result type to match a");
  return success();
}

LogicalResult StreamOp::verify() {
  auto indexType = IndexType::get(getContext());
  if (getStart().getType() != indexType || getStep().getType() != indexType ||
      getBound().getType() != indexType)
    return emitOpError("expects index typed operands");
  return success();
}

LogicalResult GateOp::verify() {
  if (!getBeforeCond().getType().isInteger(1))
    return emitOpError("expects i1 condition input");
  if (getIndex().getType() != getBeforeValue().getType())
    return emitOpError("expects result type to match value");
  if (!getCond().getType().isInteger(1))
    return emitOpError("expects i1 condition result");
  return success();
}
