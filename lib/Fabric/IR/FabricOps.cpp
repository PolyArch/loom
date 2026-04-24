#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace fabric;

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.cpp.inc"

//===----------------------------------------------------------------------===//
// fabric.mux
//===----------------------------------------------------------------------===//

// Format:
//   fabric.mux %a, %b[, ...] [ {sel = N : i32, discard = B, disconnect = B} ]
//     : <fabric-type>
//
// Software parameters live in `{...}`. Fabric mux has no hardware parameters,
// so `[...]` never appears here.

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None))
    return failure();

  // Optional software-parameter dictionary in `{}`.
  if (succeeded(parser.parseOptionalLBrace())) {
    // Re-present the opening brace to the attribute-dict parser.
    // We parsed `{`, so rewind by manually parsing attributes until `}`.
    NamedAttrList attrs;
    // Parse attribute entries: name `=` attribute, comma-separated.
    if (failed(parser.parseOptionalRBrace())) {
      do {
        StringRef name;
        Attribute value;
        if (parser.parseKeyword(&name) || parser.parseEqual() ||
            parser.parseAttribute(value))
          return failure();
        attrs.append(name, value);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRBrace())
        return failure();
    }
    result.attributes.append(attrs);
  }

  Type elemType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(elemType))
    return failure();

  SmallVector<Type, 4> operandTypes(operands.size(), elemType);
  if (parser.resolveOperands(operands, operandTypes, typeLoc, result.operands))
    return failure();
  result.addTypes(elemType);
  return success();
}

void MuxOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getInputs());

  // Print software parameters in `{}`, if any is set.
  SmallVector<NamedAttribute, 3> swAttrs;
  if (auto a = getSelAttr())
    swAttrs.push_back(NamedAttribute(getSelAttrName(), a));
  if (auto a = getDiscardAttr())
    swAttrs.push_back(NamedAttribute(getDiscardAttrName(), a));
  if (auto a = getDisconnectAttr())
    swAttrs.push_back(NamedAttribute(getDisconnectAttrName(), a));
  if (!swAttrs.empty()) {
    p << " {";
    llvm::interleaveComma(swAttrs, p, [&](const NamedAttribute &na) {
      p << na.getName().getValue() << " = ";
      p.printAttribute(na.getValue());
    });
    p << "}";
  }

  p << " : " << getOutput().getType();
}

LogicalResult MuxOp::verify() {
  auto operands = getInputs();
  if (operands.size() < 2)
    return emitOpError("requires at least 2 inputs, got ")
           << operands.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError(
        "software parameters must be all set or all unset "
        "(sel, discard, disconnect)");

  if (present == 0)
    return success();

  bool discard = discardAttr.getValue();
  bool disconnect = disconnectAttr.getValue();
  int32_t sel = selAttr.getInt();
  int64_t n = static_cast<int64_t>(operands.size());

  if (discard && disconnect)
    return emitOpError("'discard' and 'disconnect' cannot both be true");

  if (disconnect) {
    if (sel != 0)
      return emitOpError("when 'disconnect' is true, 'sel' must be 0");
  } else {
    if (sel < 0 || sel >= n)
      return emitOpError("'sel' (")
             << sel << ") must be in [0, " << n << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.fifo
//===----------------------------------------------------------------------===//
//
// Format:
//   fabric.fifo %in [max_depth = N, bypassable = true|false]
//                   [ {bypassed = true|false} ]
//                   : <fabric-type>

static ParseResult parseBoolKeyword(OpAsmParser &parser, bool &out) {
  StringRef kw;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&kw))
    return failure();
  if (kw == "true") {
    out = true;
    return success();
  }
  if (kw == "false") {
    out = false;
    return success();
  }
  return parser.emitError(loc, "expected 'true' or 'false'");
}

ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();

  // Hardware params: [max_depth = N, bypassable = true|false]
  int32_t maxDepth = 0;
  bool bypassable = false;
  if (parser.parseLSquare() ||
      parser.parseKeyword("max_depth") || parser.parseEqual() ||
      parser.parseInteger(maxDepth) ||
      parser.parseComma() ||
      parser.parseKeyword("bypassable") || parser.parseEqual() ||
      parseBoolKeyword(parser, bypassable) ||
      parser.parseRSquare())
    return failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("max_depth", builder.getI32IntegerAttr(maxDepth));
  result.addAttribute("bypassable", builder.getBoolAttr(bypassable));

  // Optional software param: {bypassed = true|false}
  if (succeeded(parser.parseOptionalLBrace())) {
    bool bypassed = false;
    if (parser.parseKeyword("bypassed") || parser.parseEqual() ||
        parseBoolKeyword(parser, bypassed) ||
        parser.parseRBrace())
      return failure();
    result.addAttribute("bypassed", builder.getBoolAttr(bypassed));
  }

  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(type))
    return failure();
  if (parser.resolveOperand(input, type, result.operands))
    return failure();
  (void)typeLoc;
  result.addTypes(type);
  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p << " [max_depth = " << getMaxDepth()
    << ", bypassable = " << (getBypassable() ? "true" : "false") << "]";
  if (auto a = getBypassedAttr())
    p << " {bypassed = " << (a.getValue() ? "true" : "false") << "}";
  p << " : " << getOutput().getType();
}

LogicalResult FifoOp::verify() {
  if (getMaxDepth() <= 0)
    return emitOpError("'max_depth' must be > 0, got ") << getMaxDepth();
  if (!getBypassable() && getBypassedAttr())
    return emitOpError(
        "'bypassed' software parameter is only allowed when 'bypassable' is "
        "true");
  return success();
}
