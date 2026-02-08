//===-- FabricFifoOps.cpp - Fabric FIFO operation impls ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::fabric;

//===----------------------------------------------------------------------===//
// FifoOp parse/print
//
// Named:  fabric.fifo @name [depth = N] : (T) -> (T)
//         fabric.fifo @name [depth = N, bypassable] {bypassed = false} : (T) -> (T)
// Inline: %out = fabric.fifo [depth = N] %in : T
//         %out = fabric.fifo [depth = N, bypassable] {bypassed = false} %in : T
//===----------------------------------------------------------------------===//

ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse required [depth = N].
  if (parser.parseLSquare())
    return failure();

  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  if (keyword != "depth")
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'depth' keyword");
  if (parser.parseEqual())
    return failure();

  IntegerAttr depthAttr;
  if (parser.parseAttribute(depthAttr, parser.getBuilder().getIntegerType(64)))
    return failure();
  result.addAttribute(getDepthAttrName(result.name), depthAttr);

  // Parse optional `, bypassable`.
  if (succeeded(parser.parseOptionalComma())) {
    StringRef bypassKw;
    if (parser.parseKeyword(&bypassKw))
      return failure();
    if (bypassKw != "bypassable")
      return parser.emitError(parser.getCurrentLocation(),
                              "expected 'bypassable' keyword");
    result.addAttribute(getBypassableAttrName(result.name),
                        parser.getBuilder().getUnitAttr());
  }

  if (parser.parseRSquare())
    return failure();

  // Parse optional {bypassed = true/false}.
  if (succeeded(parser.parseOptionalLBrace())) {
    StringRef bypassedKw;
    if (parser.parseKeyword(&bypassedKw))
      return failure();
    if (bypassedKw != "bypassed")
      return parser.emitError(parser.getCurrentLocation(),
                              "expected 'bypassed' keyword");
    if (parser.parseEqual())
      return failure();

    BoolAttr bypassedAttr;
    if (parser.parseAttribute(bypassedAttr))
      return failure();
    result.addAttribute(getBypassedAttrName(result.name), bypassedAttr);

    if (parser.parseRBrace())
      return failure();
  }

  if (isNamed) {
    // Named form: parse `: (T) -> (T)` and store as function_type.
    FunctionType fnType;
    if (parser.parseColonType(fnType))
      return failure();
    result.addAttribute(getFunctionTypeAttrName(result.name),
                        TypeAttr::get(fnType));
  } else {
    // Inline form: parse operand and type.
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();

    if (parser.parseColon())
      return failure();

    Type elemType;
    if (parser.parseType(elemType))
      return failure();

    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();
    result.addTypes(elemType);
  }

  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  p << " [depth = " << getDepth();
  if (getBypassable())
    p << ", bypassable";
  p << "]";

  if (getBypassed().has_value()) {
    p << " {bypassed = ";
    p << (*getBypassed() ? "true" : "false");
    p << "}";
  }

  if (isNamed) {
    p << " : ";
    p.printAttribute(getFunctionTypeAttr());
  } else {
    p << " ";
    p.printOperand(getInputs().front());
    p << " : ";
    p.printType(getInputs().front().getType());
  }
}

//===----------------------------------------------------------------------===//
// FifoOp verify
//===----------------------------------------------------------------------===//

/// Check if a type is a valid native type for fabric.fifo.
static bool isValidFifoType(Type type) {
  // Native types: i1, i8, i16, i32, i64, f16, bf16, f32, f64, index, none.
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    unsigned w = intTy.getWidth();
    return w == 1 || w == 8 || w == 16 || w == 32 || w == 64;
  }
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return true;
  if (isa<IndexType>(type))
    return true;
  if (isa<NoneType>(type))
    return true;
  // Tagged types.
  if (isa<loom::dataflow::TaggedType>(type))
    return true;
  return false;
}

LogicalResult FifoOp::verify() {
  bool isNamed = getSymName().has_value();

  if (isNamed) {
    // Named form: validate function_type.
    if (!getFunctionType())
      return emitOpError("named fifo requires function_type attribute");
    auto fnType = *getFunctionType();
    if (fnType.getNumInputs() != 1 || fnType.getNumResults() != 1)
      return emitOpError("named fifo must have exactly 1 input and 1 output");
    if (fnType.getInput(0) != fnType.getResult(0))
      return emitOpError("[COMP_FIFO_TYPE_MISMATCH] "
                         "input type must match output type; got ")
             << fnType.getInput(0) << " vs " << fnType.getResult(0);
    if (!isValidFifoType(fnType.getInput(0)))
      return emitOpError("[COMP_FIFO_INVALID_TYPE] "
                         "type must be a native type or !dataflow.tagged; got ")
             << fnType.getInput(0);
    // Named form should have no SSA operands/results.
    if (!getInputs().empty() || !getOutputs().empty())
      return emitOpError(
          "named fifo must not have SSA operands or results");
  } else {
    // Inline form: must have exactly 1 input and 1 output.
    if (getInputs().size() != 1)
      return emitOpError("inline fifo must have exactly 1 input; got ")
             << getInputs().size();
    if (getOutputs().size() != 1)
      return emitOpError("inline fifo must have exactly 1 output; got ")
             << getOutputs().size();
    if (getInputs().front().getType() != getOutputs().front().getType())
      return emitOpError("[COMP_FIFO_TYPE_MISMATCH] "
                         "input type must match output type; got ")
             << getInputs().front().getType() << " vs "
             << getOutputs().front().getType();
    if (!isValidFifoType(getInputs().front().getType()))
      return emitOpError("[COMP_FIFO_INVALID_TYPE] "
                         "type must be a native type or !dataflow.tagged; got ")
             << getInputs().front().getType();
  }

  // Depth must be >= 1.
  if (getDepth() < 1)
    return emitOpError("[COMP_FIFO_DEPTH_ZERO] depth must be >= 1; got ")
           << getDepth();

  // Bypassable/bypassed consistency.
  if (!getBypassable() && getBypassed().has_value())
    return emitOpError("[COMP_FIFO_BYPASSED_NOT_BYPASSABLE] "
                       "'bypassed' attribute present without 'bypassable'");
  if (getBypassable() && !getBypassed().has_value())
    return emitOpError("[COMP_FIFO_BYPASSED_MISSING] "
                       "'bypassable' is set but 'bypassed' attribute is missing");

  return success();
}
