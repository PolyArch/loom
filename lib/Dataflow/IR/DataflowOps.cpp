#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace dataflow;

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// dataflow.stream
//===----------------------------------------------------------------------===//

static bool isValidStepOp(llvm::StringRef s) {
  return llvm::StringSwitch<bool>(s)
      .Case("+=", true)
      .Case("*=", true)
      .Case("-=", true)
      .Case("/=", true)
      .Case("<<=", true)
      .Case(">>=", true)
      .Default(false);
}

static bool isValidContCond(llvm::StringRef s) {
  return llvm::StringSwitch<bool>(s)
      .Case("<", true)
      .Case("<=", true)
      .Case(">", true)
      .Case(">=", true)
      .Case("!=", true)
      .Default(false);
}

LogicalResult StreamOp::verify() {
  if (!isValidStepOp(getStepOp()))
    return emitOpError("'step_op' must be one of '+=', '*=', '-=', '/=', "
                       "'<<=', '>>='; got \"")
           << getStepOp() << "\"";
  if (!isValidContCond(getContCond()))
    return emitOpError(
               "'cont_cond' must be one of '<', '<=', '>', '>=', '!='; got \"")
           << getContCond() << "\"";
  return success();
}
