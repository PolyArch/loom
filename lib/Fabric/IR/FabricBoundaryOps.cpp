#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

//===----------------------------------------------------------------------===//
// fabric.s2t
//===----------------------------------------------------------------------===//
//
// Assembly forms (two disjoint variants by operand count):
//
//   Constant-tag form (1 SSA operand + sw_configs.tag):
//     fabric.s2t %data {sw_configs = {tag = K : i<TW>}}
//                : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
//
//   General form (2 SSA operands -- data, tag):
//     fabric.s2t %data, %tag
//                : (!fabric.bits<BW>, !fabric.bits<TW>)
//                  -> !fabric.bits_tag<BW, TW>

ParseResult S2tOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  OpAsmParser::UnresolvedOperand first;
  if (parser.parseOperand(first))
    return failure();
  operands.push_back(first);
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand second;
    if (parser.parseOperand(second))
      return failure();
    operands.push_back(second);
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  SMLoc typeLoc = parser.getCurrentLocation();
  SmallVector<Type, 2> operandTypes;
  Type resultType;
  if (operands.size() == 2) {
    // Functional form: `(T0, T1) -> Tres`.
    if (parser.parseLParen())
      return failure();
    Type t0, t1;
    if (parser.parseType(t0) || parser.parseComma() || parser.parseType(t1) ||
        parser.parseRParen())
      return failure();
    operandTypes.push_back(t0);
    operandTypes.push_back(t1);
    if (parser.parseArrow() || parser.parseType(resultType))
      return failure();
  } else {
    // Single-operand form: `Tin -> Tres`.
    Type t0;
    if (parser.parseType(t0))
      return failure();
    operandTypes.push_back(t0);
    if (parser.parseArrow() || parser.parseType(resultType))
      return failure();
  }

  if (parser.resolveOperands(operands, operandTypes, typeLoc, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void S2tOp::print(OpAsmPrinter &p) {
  p << ' ';
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p << v; });
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : ";
  if (getInputs().size() == 2) {
    p << '(' << getInputs()[0].getType() << ", " << getInputs()[1].getType()
      << ") -> " << getOutput().getType();
  } else {
    p << getInputs()[0].getType() << " -> " << getOutput().getType();
  }
}

LogicalResult S2tOp::verify() {
  auto operands = getInputs();
  if (operands.size() < 1 || operands.size() > 2)
    return emitOpError("expects 1 or 2 SSA operands, got ") << operands.size();

  auto resultTagTy = dyn_cast<BitsTagType>(getOutput().getType());
  if (!resultTagTy)
    return emitOpError("result must be a !fabric.bits_tag<BW, TW> type");
  unsigned resBW = resultTagTy.getWidth();
  unsigned resTW = resultTagTy.getTagWidth();

  // Operand 0 must be !fabric.bits<resBW>.
  auto in0 = dyn_cast<BitsType>(operands[0].getType());
  if (!in0)
    return emitOpError("operand #0 must be !fabric.bits<BW>, got ")
           << operands[0].getType();
  if (in0.getWidth() != resBW)
    return emitOpError("operand #0 bits-width ")
           << in0.getWidth() << " must equal result data-width " << resBW;

  if (operands.size() == 2) {
    // General form: operand[1] must be !fabric.bits<resTW>.
    auto in1 = dyn_cast<BitsType>(operands[1].getType());
    if (!in1)
      return emitOpError("operand #1 must be !fabric.bits<TW>, got ")
             << operands[1].getType();
    if (in1.getWidth() != resTW)
      return emitOpError("operand #1 bits-width ")
             << in1.getWidth() << " must equal result tag-width " << resTW;

    // sw_configs.tag must NOT be present in the general form (tag arrives
    // through SSA, not as a runtime config).
    if (auto sw = getSwConfigsAttr())
      if (sw.get("tag"))
        return emitOpError(
            "two-operand form must not carry 'sw_configs.tag'; the tag is "
            "supplied as the second SSA operand");
    return success();
  }

  // Constant-tag form: requires sw_configs.tag IntegerAttr matching TW.
  auto sw = getSwConfigsAttr();
  if (!sw)
    return emitOpError(
        "constant-tag form requires 'sw_configs.tag' integer attribute");
  auto tagAttr = sw.get("tag");
  if (!tagAttr)
    return emitOpError(
        "constant-tag form requires 'sw_configs.tag' integer attribute");
  auto tagInt = dyn_cast<IntegerAttr>(tagAttr);
  if (!tagInt)
    return emitOpError("'sw_configs.tag' must be an IntegerAttr");
  auto tagTy = dyn_cast<IntegerType>(tagInt.getType());
  if (!tagTy)
    return emitOpError("'sw_configs.tag' must have an IntegerType");
  unsigned tagWidth = tagTy.getWidth();
  if (tagWidth != resTW)
    return emitOpError("'sw_configs.tag' integer attribute width ")
           << tagWidth << " must equal result tag-width " << resTW;
  // Range check: an IntegerAttr whose bit-width equals TW automatically
  // represents an element of [0, 2^TW) when interpreted as an unsigned
  // bit-pattern. Negative-looking literals (e.g. `-1 : i4`) share storage
  // with their unsigned complement (`15 : i4`); the tag is treated as an
  // unsigned bit pattern so no further check is needed here.
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.t2t
//===----------------------------------------------------------------------===//
//
// Assembly form:
//
//   fabric.t2t %in
//              {hw_params = [{lookup_table = [...]}]}
//              : !fabric.bits_tag<BW, TW1> -> !fabric.bits_tag<BW, TW2>

ParseResult T2tOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inTy, outTy;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(inTy) || parser.parseArrow() ||
      parser.parseType(outTy))
    return failure();
  if (parser.resolveOperand(input, inTy, result.operands))
    return failure();
  (void)typeLoc;
  result.addTypes(outTy);
  return success();
}

void T2tOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : " << getInput().getType() << " -> " << getOutput().getType();
}

LogicalResult T2tOp::verify() {
  auto inTy = dyn_cast<BitsTagType>(getInput().getType());
  auto outTy = dyn_cast<BitsTagType>(getOutput().getType());
  if (!inTy || !outTy)
    return emitOpError(
        "operand and result must be !fabric.bits_tag<BW, TW> types");

  unsigned inBW = inTy.getWidth();
  unsigned outBW = outTy.getWidth();
  unsigned inTW = inTy.getTagWidth();
  unsigned outTW = outTy.getTagWidth();
  if (inBW != outBW)
    return emitOpError("operand data-width ")
           << inBW << " must equal result data-width " << outBW
           << " (fabric.t2t only remaps the tag)";

  // hw_params: required, length-1 array wrapping a dictionary with key
  // `lookup_table`.
  auto hp = getHwParamsAttr();
  if (!hp)
    return emitOpError("requires 'hw_params' attribute carrying the "
                       "lookup table");
  if (hp.size() != 1)
    return emitOpError("'hw_params' must be a length-1 array wrapping a "
                       "dictionary, got length ")
           << hp.size();
  auto hwDict = dyn_cast<DictionaryAttr>(hp[0]);
  if (!hwDict)
    return emitOpError(
        "'hw_params' inner element must be a dictionary attribute");
  auto lutAttr = hwDict.get("lookup_table");
  if (!lutAttr)
    return emitOpError("'hw_params[0]' must contain key 'lookup_table'");
  auto lut = dyn_cast<ArrayAttr>(lutAttr);
  if (!lut)
    return emitOpError("'lookup_table' must be an array of dictionaries");
  if (lut.empty())
    return emitOpError("'lookup_table' must be non-empty");

  llvm::DenseSet<uint64_t> seenInputTags;
  for (size_t i = 0; i < lut.size(); ++i) {
    Attribute entryAttr = lut[i];
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    if (!entry)
      return emitOpError("'lookup_table' entry #")
             << i << " must be a dictionary attribute";
    auto inTagAttr = entry.get("input_tag");
    auto outTagAttr = entry.get("output_tag");
    if (!inTagAttr || !outTagAttr)
      return emitOpError("'lookup_table' entry #")
             << i << " must have keys 'input_tag' and 'output_tag'";
    auto inTagInt = dyn_cast<IntegerAttr>(inTagAttr);
    auto outTagInt = dyn_cast<IntegerAttr>(outTagAttr);
    if (!inTagInt || !outTagInt)
      return emitOpError("'lookup_table' entry #")
             << i
             << " 'input_tag'/'output_tag' must be IntegerAttr";
    auto inTagTy = dyn_cast<IntegerType>(inTagInt.getType());
    auto outTagTy = dyn_cast<IntegerType>(outTagInt.getType());
    if (!inTagTy || !outTagTy)
      return emitOpError("'lookup_table' entry #")
             << i
             << " 'input_tag'/'output_tag' must have IntegerType";
    if (inTagTy.getWidth() != inTW)
      return emitOpError("'lookup_table' entry #")
             << i << " 'input_tag' integer width " << inTagTy.getWidth()
             << " must equal operand tag-width " << inTW;
    if (outTagTy.getWidth() != outTW)
      return emitOpError("'lookup_table' entry #")
             << i << " 'output_tag' integer width " << outTagTy.getWidth()
             << " must equal result tag-width " << outTW;

    // Range check is automatic: an IntegerAttr whose bit-width equals TW
    // represents an element of [0, 2^TW) when interpreted as an unsigned
    // bit-pattern (we treat tags as unsigned bit patterns).
    uint64_t key = inTagInt.getValue().getZExtValue();
    if (!seenInputTags.insert(key).second)
      return emitOpError("'lookup_table' has duplicate 'input_tag' value ")
             << key;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.t2s
//===----------------------------------------------------------------------===//
//
// Assembly forms:
//
//   Split form (2 results -- data + tag):
//     %data, %tag = fabric.t2s %in
//                   : !fabric.bits_tag<BW, TW>
//                     -> (!fabric.bits<BW>, !fabric.bits<TW>)
//
//   Drop-tag form (1 result -- data only):
//     %data = fabric.t2s %in : !fabric.bits_tag<BW, TW> -> !fabric.bits<BW>

ParseResult T2sOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  Type inTy;
  if (parser.parseType(inTy) || parser.parseArrow())
    return failure();
  SmallVector<Type, 2> outTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(outTypes) || parser.parseRParen())
      return failure();
  } else {
    Type t;
    if (parser.parseType(t))
      return failure();
    outTypes.push_back(t);
  }
  if (parser.resolveOperand(input, inTy, result.operands))
    return failure();
  result.addTypes(outTypes);
  return success();
}

void T2sOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : " << getInput().getType() << " -> ";
  auto rTypes = getResultTypes();
  if (rTypes.size() == 1) {
    p << rTypes.front();
  } else {
    p << '(';
    llvm::interleaveComma(rTypes, p);
    p << ')';
  }
}

LogicalResult T2sOp::verify() {
  auto inTy = dyn_cast<BitsTagType>(getInput().getType());
  if (!inTy)
    return emitOpError("operand must be a !fabric.bits_tag<BW, TW> type");
  unsigned inBW = inTy.getWidth();
  unsigned inTW = inTy.getTagWidth();

  auto results = getOutputs();
  if (results.size() < 1 || results.size() > 2)
    return emitOpError("expects 1 or 2 results, got ") << results.size();

  auto r0 = dyn_cast<BitsType>(results[0].getType());
  if (!r0)
    return emitOpError("result #0 must be !fabric.bits<BW>, got ")
           << results[0].getType();
  if (r0.getWidth() != inBW)
    return emitOpError("result #0 bits-width ")
           << r0.getWidth() << " must equal operand data-width " << inBW;

  if (results.size() == 2) {
    auto r1 = dyn_cast<BitsType>(results[1].getType());
    if (!r1)
      return emitOpError("result #1 must be !fabric.bits<TW>, got ")
             << results[1].getType();
    if (r1.getWidth() != inTW)
      return emitOpError("result #1 bits-width ")
             << r1.getWidth() << " must equal operand tag-width " << inTW;
  }
  return success();
}
