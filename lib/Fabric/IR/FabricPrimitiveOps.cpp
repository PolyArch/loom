#include "Fabric/IR/FabricOps.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace mlir;
using namespace fabric;

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
    return emitOpError("requires at least 2 inputs, got ") << operands.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError("software parameters must be all set or all unset "
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
      return emitOpError("'sel' (") << sel << ") must be in [0, " << n << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.demux
//===----------------------------------------------------------------------===//
//
// Format:
//   fabric.demux %in [ {sel = N : i32, discard = B, disconnect = B} ]
//                : <fabric-type> -> N
//
// Software parameters live in `{...}`. Fabric demux has no hardware
// parameters, so `[...]` never appears. The trailing `-> N` records the
// number of output ports so the parser can recreate them all (they share
// the input's type via SameOperandsAndResultType).

ParseResult DemuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();

  if (succeeded(parser.parseOptionalLBrace())) {
    NamedAttrList attrs;
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
  unsigned numOuts = 0;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(elemType) ||
      parser.parseArrow() || parser.parseInteger(numOuts))
    return failure();
  if (parser.resolveOperand(input, elemType, result.operands))
    return failure();
  (void)typeLoc;
  SmallVector<Type, 4> outputTypes(numOuts, elemType);
  result.addTypes(outputTypes);
  return success();
}

void DemuxOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();

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

  p << " : " << getInput().getType() << " -> " << getOutputs().size();
}

LogicalResult DemuxOp::verify() {
  auto outputs = getOutputs();
  if (outputs.size() < 2)
    return emitOpError("requires at least 2 outputs, got ") << outputs.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError("software parameters must be all set or all unset "
                       "(sel, discard, disconnect)");
  if (present == 0)
    return success();

  bool discard = discardAttr.getValue();
  bool disconnect = disconnectAttr.getValue();
  int32_t sel = selAttr.getInt();
  int64_t n = static_cast<int64_t>(outputs.size());

  if (discard && disconnect)
    return emitOpError("'discard' and 'disconnect' cannot both be true");

  if (disconnect) {
    if (sel != 0)
      return emitOpError("when 'disconnect' is true, 'sel' must be 0");
  } else {
    if (sel < 0 || sel >= n)
      return emitOpError("'sel' (") << sel << ") must be in [0, " << n << ")";
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

// Canonical FIFO assembly:
//   fabric.fifo %src [max_depth = N, bypassable = B] [{bypassed = B}]
//       : <source-type> [to <storage-type>]
// The storage type defaults to the source type. A differing same-kind type
// uses the enclosing fabric.module connection semantics.
ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SMLoc operandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input))
    return failure();

  // Hardware params: [max_depth = N, bypassable = true|false]
  int32_t maxDepth = 0;
  bool bypassable = false;
  if (parser.parseLSquare() || parser.parseKeyword("max_depth") ||
      parser.parseEqual() || parser.parseInteger(maxDepth) ||
      parser.parseComma() || parser.parseKeyword("bypassable") ||
      parser.parseEqual() || parseBoolKeyword(parser, bypassable) ||
      parser.parseRSquare())
    return failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("max_depth", builder.getI32IntegerAttr(maxDepth));
  result.addAttribute("bypassable", builder.getBoolAttr(bypassable));

  // Optional software param: {bypassed = true|false}
  if (succeeded(parser.parseOptionalLBrace())) {
    bool bypassed = false;
    if (parser.parseKeyword("bypassed") || parser.parseEqual() ||
        parseBoolKeyword(parser, bypassed) || parser.parseRBrace())
      return failure();
    result.addAttribute("bypassed", builder.getBoolAttr(bypassed));
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Type outerType;
  if (parser.parseColon() || parser.parseType(outerType))
    return failure();
  // Optional `to <inner-type>` after the FIFO's own type. When absent,
  // inner == outer.
  Type innerType = outerType;
  if (succeeded(parser.parseOptionalKeyword("to"))) {
    if (parser.parseType(innerType))
      return failure();
  }
  if (parser.resolveOperand(input, outerType, result.operands))
    return failure();
  (void)operandLoc;
  result.addTypes(innerType);
  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p << " [max_depth = " << getMaxDepth()
    << ", bypassable = " << (getBypassable() ? "true" : "false") << "]";
  if (auto a = getBypassedAttr())
    p << " {bypassed = " << (a.getValue() ? "true" : "false") << "}";
  SmallVector<StringRef, 3> elided{"max_depth", "bypassable", "bypassed"};
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  Type outerTy = getInput().getType();
  Type innerTy = getOutput().getType();
  p << " : " << outerTy;
  if (outerTy != innerTy)
    p << " to " << innerTy;
}

LogicalResult FifoOp::verify() {
  if (getMaxDepth() <= 0)
    return emitOpError("'max_depth' must be > 0, got ") << getMaxDepth();
  if (!getBypassable() && getBypassedAttr())
    return emitOpError(
        "'bypassed' software parameter is only allowed when 'bypassable' is "
        "true");
  // Width-relaxation rule at the FIFO operand boundary. The outer SSA
  // source type may differ from the FIFO's inner type only in width, and
  // only for the same fabric kind (bits / bits_tag). memref operands
  // (not currently part of the SameOperandsAndResultType-constrained type)
  // are not legal here; the type constraint already rejects them, but the
  // explicit `to <T_inner>` clause is still rejected when the kinds
  // disagree. Emit a clear diagnostic when the kinds disagree.
  Type outerTy = getInput().getType();
  Type innerTy = getOutput().getType();
  if (outerTy != innerTy) {
    if (!haveSameFabricModulePortKind(outerTy, innerTy))
      return emitOpError(
                 "operand outer type and inner type must share the same "
                 "fabric kind (bits, bits_tag); got outer ")
             << outerTy << " and inner " << innerTy;
    if (isa<MemRefType>(outerTy))
      return emitOpError(
          "memref operands cannot use the 'to <inner-type>' clause: memref "
          "types must match exactly");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.op
//===----------------------------------------------------------------------===//

LogicalResult OpOp::verify() {
  // 1. Operand and result types: all must be fabric.bits<N>. (Already enforced
  //    by the type constraint, but emit a clearer error if something slipped
  //    through, e.g. via the generic op syntax.)
  for (auto [i, t] : llvm::enumerate(getInputs().getTypes()))
    if (!getFabricBitsWidth(t))
      return emitOpError("input #") << i << " must be fabric.bits<N>";
  for (auto [i, t] : llvm::enumerate(getOutputs().getTypes()))
    if (!getFabricBitsWidth(t))
      return emitOpError("output #") << i << " must be fabric.bits<N>";

  // 2. The physical resource owns one explicit generated implementation
  //    family. op_list is an enabled subset of that family's registered
  //    operation schemas.
  std::optional<ImplementationFamilyId> family = getImplementationFamily();
  if (!family)
    return emitOpError("requires an explicit implementation_family");

  ArrayAttr opList = getOpList();
  if (opList.empty())
    return emitOpError("'op_list' must be non-empty");
  llvm::StringSet<> uniqueNames;
  for (auto [i, attr] : llvm::enumerate(opList)) {
    auto sym = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!sym)
      return emitOpError("'op_list' entry #")
             << i << " must be a flat symbol reference";
    StringRef n = sym.getValue();
    std::optional<dataflow::OperationSchemaId> schema =
        dataflow::findOperationSchema(n);
    if (!schema)
      return emitOpError("op_list member @")
             << n << " is not a registered canonical operation schema";
    if (!uniqueNames.insert(n).second)
      return emitOpError("op_list contains duplicate member @") << n;
    if (!admitsOperationSchema(*family, *schema))
      return emitOpError("op_list member @")
             << n << " is not admitted by implementation family "
             << implementationFamilyKeyword(*family);
  }

  auto params = parseFamilyCapabilityParams(*family, getHwParams());
  if (!params)
    return emitOpError(llvm::toString(params.takeError()));
  if (auto record = (*this)->getAttrOfType<DenseI8ArrayAttr>(
          kResourceContractRecordAttrName)) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(record.size());
    for (std::int8_t byte : record.asArrayRef())
      bytes.push_back(static_cast<std::uint8_t>(byte));
    auto contract = decodeResourceContractRecord(bytes);
    if (!contract)
      return emitOpError(llvm::toString(contract.takeError()));
    auto canonical = encodeResourceContractRecord(*contract);
    if (!canonical)
      return emitOpError(llvm::toString(canonical.takeError()));
    if (*canonical != bytes)
      return emitOpError("resource contract is not canonical");
    if (contract->usePatternCount() == 0)
      return emitOpError("resource contract has no use pattern");
  }
  return success();
}
