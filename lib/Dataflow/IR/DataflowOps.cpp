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

//===----------------------------------------------------------------------===//
// dataflow.constant
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto typed = llvm::dyn_cast<TypedAttr>(getConstValue());
  if (!typed)
    return emitOpError("'const_value' must be a typed attribute");
  if (typed.getType() != getValue().getType())
    return emitOpError("'const_value' type ")
           << typed.getType() << " must match result type "
           << getValue().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.sync
//===----------------------------------------------------------------------===//

LogicalResult SyncOp::verify() {
  auto ins = getInputs();
  auto outs = getOutputs();
  if (ins.size() != outs.size())
    return emitOpError("number of inputs (")
           << ins.size() << ") must equal number of outputs ("
           << outs.size() << ")";
  for (unsigned i = 0, e = ins.size(); i < e; ++i) {
    if (ins[i].getType() != outs[i].getType())
      return emitOpError("input #")
             << i << " type " << ins[i].getType() << " must match output #"
             << i << " type " << outs[i].getType();
  }
  return success();
}
