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
// Streaming Ops
//===----------------------------------------------------------------------===//

// dataflow.stream

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
// Control Ops
//===----------------------------------------------------------------------===//

// dataflow.constant

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

//===----------------------------------------------------------------------===//
// dataflow.mux / dataflow.demux
//===----------------------------------------------------------------------===//

static LogicalResult verifySelAgainstArity(Operation *op, Type selType,
                                           size_t n, StringRef fanName) {
  if (n < 2)
    return op->emitOpError()
           << "requires at least 2 " << fanName << ", got " << n;
  bool isI1 = selType.isInteger(1);
  bool isIndex = isa<IndexType>(selType);
  if (n == 2) {
    if (!isI1)
      return op->emitOpError()
             << "with 2 " << fanName << ", 'sel' must be 'i1', got "
             << selType;
  } else {
    if (!isIndex)
      return op->emitOpError()
             << "with more than 2 " << fanName
             << ", 'sel' must be 'index', got " << selType;
  }
  return success();
}

LogicalResult MuxOp::verify() {
  if (failed(verifySelAgainstArity(getOperation(), getSel().getType(),
                                   getInputs().size(), "inputs")))
    return failure();
  Type outTy = getOutput().getType();
  for (auto [i, in] : llvm::enumerate(getInputs())) {
    if (in.getType() != outTy)
      return emitOpError("input #")
             << i << " type " << in.getType() << " must match output type "
             << outTy;
  }
  return success();
}

LogicalResult DemuxOp::verify() {
  if (failed(verifySelAgainstArity(getOperation(), getSel().getType(),
                                   getOutputs().size(), "outputs")))
    return failure();
  Type inTy = getInput().getType();
  for (auto [i, out] : llvm::enumerate(getOutputs())) {
    if (out.getType() != inTy)
      return emitOpError("output #")
             << i << " type " << out.getType() << " must match input type "
             << inTy;
  }
  return success();
}

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
