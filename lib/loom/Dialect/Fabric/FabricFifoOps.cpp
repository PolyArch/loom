//===-- FabricFifoOps.cpp - Fabric FIFO operation impls ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Fabric/FabricTypeUtils.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricError.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom;
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

  // Parse remaining optional params: bypassable, viz_row, viz_col.
  while (succeeded(parser.parseOptionalComma())) {
    StringRef kw;
    if (parser.parseKeyword(&kw))
      return failure();
    if (kw == "bypassable") {
      result.addAttribute(getBypassableAttrName(result.name),
                          parser.getBuilder().getUnitAttr());
    } else if (kw == "viz_row" || kw == "viz_col") {
      if (parser.parseEqual())
        return failure();
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(kw, attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << kw << "' in fifo hardware parameters";
    }
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
  if (auto vizRow = getOperation()->getAttr("viz_row"))
    p << ", viz_row = " << cast<IntegerAttr>(vizRow).getInt();
  if (auto vizCol = getOperation()->getAttr("viz_col"))
    p << ", viz_col = " << cast<IntegerAttr>(vizCol).getInt();
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

/// Check whether a type is a valid scalar routing payload type.
/// Routing nodes only accept: BitsType, NoneType.
static bool isValidRoutingPayloadType(Type t) {
  return isa<loom::dataflow::BitsType>(t) || isa<NoneType>(t);
}

/// Check if a type is a valid type for fabric.fifo.
/// Only routing payload types (bits, none) and tagged types with
/// routing payload value types are allowed.
static bool isValidFifoType(Type type) {
  // Routing payload types: bits<N>, none.
  if (isValidRoutingPayloadType(type))
    return true;
  // Tagged types with valid routing payload value type.
  if (auto tagged = dyn_cast<loom::dataflow::TaggedType>(type))
    return isValidRoutingPayloadType(tagged.getValueType());
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
    if (!isRoutingTypeCompatible(fnType.getInput(0), fnType.getResult(0)))
      return emitOpError(cplErrMsg(CplError::FIFO_TYPE_MISMATCH,
                         "input and output types must be bit-width compatible; got "))
             << fnType.getInput(0) << " vs " << fnType.getResult(0);
    if (!isValidFifoType(fnType.getInput(0)))
      return emitOpError(cplErrMsg(CplError::FIFO_INVALID_TYPE,
                         "type must be !dataflow.bits<N>, none, or !dataflow.tagged; got "))
             << fnType.getInput(0);
    if (!isValidFifoType(fnType.getResult(0)))
      return emitOpError(cplErrMsg(CplError::FIFO_INVALID_TYPE,
                         "type must be !dataflow.bits<N>, none, or !dataflow.tagged; got "))
             << fnType.getResult(0);
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
    if (!isRoutingTypeCompatible(getInputs().front().getType(),
                                getOutputs().front().getType()))
      return emitOpError(cplErrMsg(CplError::FIFO_TYPE_MISMATCH,
                         "input and output types must be bit-width compatible; got "))
             << getInputs().front().getType() << " vs "
             << getOutputs().front().getType();
    if (!isValidFifoType(getInputs().front().getType()))
      return emitOpError(cplErrMsg(CplError::FIFO_INVALID_TYPE,
                         "type must be !dataflow.bits<N>, none, or !dataflow.tagged; got "))
             << getInputs().front().getType();
  }

  // Depth must be >= 1.
  if (getDepth() < 1)
    return emitOpError(cplErrMsg(CplError::FIFO_DEPTH_ZERO,
                       "depth must be >= 1; got "))
           << getDepth();

  // Bypassable/bypassed consistency.
  if (!getBypassable() && getBypassed().has_value())
    return emitOpError(cplErrMsg(CplError::FIFO_BYPASSED_NOT_BYPASSABLE,
                       "'bypassed' attribute present without 'bypassable'"));
  if (getBypassable() && !getBypassed().has_value())
    return emitOpError(cplErrMsg(CplError::FIFO_BYPASSED_MISSING,
                       "'bypassable' is set but 'bypassed' attribute is missing"));

  return success();
}
