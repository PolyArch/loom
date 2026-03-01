//===-- FabricFifoOps.cpp - Fabric FIFO operation impls ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
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

/// Get the bit width of a native type for routing compatibility checks.
/// Returns std::nullopt for types without a well-defined bit width (index,
/// none), which must match exactly.
static std::optional<unsigned> getNativeBitWidth(Type t) {
  if (auto bitsType = dyn_cast<dataflow::BitsType>(t))
    return bitsType.getWidth();
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  if (isa<Float16Type, BFloat16Type>(t))
    return 16u;
  if (isa<Float32Type>(t))
    return 32u;
  if (isa<Float64Type>(t))
    return 64u;
  return std::nullopt;
}

/// Check whether two types are compatible for routing through pass-through
/// nodes (switch, temporal_sw, fifo). Rules:
///   - Exact match: always compatible.
///   - Both native with known bit width: compatible if widths match.
///   - Both tagged: compatible if value bit widths AND tag bit widths match.
///   - One native, one tagged (category mismatch): never compatible.
static bool isRoutingTypeCompatible(Type a, Type b) {
  if (a == b)
    return true;

  bool isTaggedA = isa<dataflow::TaggedType>(a);
  bool isTaggedB = isa<dataflow::TaggedType>(b);

  // Category mismatch: native-to-tagged is never allowed.
  if (isTaggedA != isTaggedB)
    return false;

  if (isTaggedA) {
    // Both tagged: value bit widths and tag bit widths must each match.
    auto tagA = cast<dataflow::TaggedType>(a);
    auto tagB = cast<dataflow::TaggedType>(b);
    auto valWidthA = getNativeBitWidth(tagA.getValueType());
    auto valWidthB = getNativeBitWidth(tagB.getValueType());
    if (!valWidthA || !valWidthB)
      return false;
    return *valWidthA == *valWidthB &&
           tagA.getTagType().getWidth() == tagB.getTagType().getWidth();
  }

  // Both native: check bit width equality.
  auto widthA = getNativeBitWidth(a);
  auto widthB = getNativeBitWidth(b);
  if (!widthA || !widthB)
    return false;
  return *widthA == *widthB;
}

/// Check whether a type is a valid routing payload type.
/// Routing nodes only accept: BitsType, NoneType, IndexType.
static bool isValidRoutingPayloadType(Type t) {
  return isa<loom::dataflow::BitsType>(t) || isa<NoneType>(t) ||
         isa<IndexType>(t);
}

/// Check if a type is a valid type for fabric.fifo.
/// Only routing payload types (bits, none, index) and tagged types with
/// routing payload value types are allowed.
static bool isValidFifoType(Type type) {
  // Routing payload types: bits<N>, none, index.
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
                         "type must be a native type or !dataflow.tagged; got "))
             << fnType.getInput(0);
    if (!isValidFifoType(fnType.getResult(0)))
      return emitOpError(cplErrMsg(CplError::FIFO_INVALID_TYPE,
                         "type must be a native type or !dataflow.tagged; got "))
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
                         "type must be a native type or !dataflow.tagged; got "))
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
