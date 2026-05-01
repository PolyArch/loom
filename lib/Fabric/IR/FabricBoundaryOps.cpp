#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

//===----------------------------------------------------------------------===//
// fabric.boundary
//===----------------------------------------------------------------------===//
//
// Unified boundary op carrying a mandatory `[<direction>]` predicate
// (`s2t` | `t2t` | `t2s`). Per-direction shapes:
//
//   [s2t] general form (2 SSA operands -- data, tag):
//     fabric.boundary [s2t] %data, %tag
//                : (!fabric.bits<BW>, !fabric.bits<TW>)
//                  -> !fabric.bits_tag<BW, TW>
//
//   [s2t] constant-tag form (1 SSA operand + sw_configs.tag):
//     fabric.boundary [s2t] %data {sw_configs = {tag = K : i<TW>}}
//                : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
//
//   [t2t] tag-remap form (1 SSA operand + hw_params + sw_configs):
//     fabric.boundary [t2t] %in
//                {hw_params = [{lut_size = N : i32}],
//                 sw_configs = {lookup_table = [{src_tag = ..., dst_tag = ...}, ...]}}
//                : !fabric.bits_tag<BW, TW1> -> !fabric.bits_tag<BW, TW2>
//
//   [t2s] split form (2 results -- data, tag):
//     %data, %tag = fabric.boundary [t2s] %in
//                : !fabric.bits_tag<BW, TW>
//                  -> (!fabric.bits<BW>, !fabric.bits<TW>)
//
//   [t2s] drop-tag form (1 result -- data only):
//     %data = fabric.boundary [t2s] %in
//                : !fabric.bits_tag<BW, TW> -> !fabric.bits<BW>

ParseResult BoundaryOp::parse(OpAsmParser &parser, OperationState &result) {
  // Mandatory `[<direction>]` predicate.
  StringRef directionKw;
  SMLoc directionLoc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseKeyword(&directionKw) ||
      parser.parseRSquare())
    return failure();
  auto sym = symbolizeBoundaryDirection(directionKw);
  if (!sym)
    return parser.emitError(directionLoc,
                            "expected fabric boundary direction keyword "
                            "'s2t', 't2t' or 't2s', got '")
           << directionKw << "'";
  result.addAttribute(
      "direction",
      BoundaryDirectionAttr::get(parser.getContext(), *sym));

  // SSA operand list (1 or 2 operands).
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
  SmallVector<Type, 2> resultTypes;

  // The functional `(...) -> ...` form is used iff there are 2 operands.
  // For 1-operand cases the input type is a bare type. The result side may
  // be either a bare type or a parenthesized list of types.
  if (operands.size() == 2) {
    if (parser.parseLParen())
      return failure();
    Type t0, t1;
    if (parser.parseType(t0) || parser.parseComma() || parser.parseType(t1) ||
        parser.parseRParen())
      return failure();
    operandTypes.push_back(t0);
    operandTypes.push_back(t1);
    if (parser.parseArrow())
      return failure();
  } else {
    Type t0;
    if (parser.parseType(t0) || parser.parseArrow())
      return failure();
    operandTypes.push_back(t0);
  }

  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  } else {
    Type t;
    if (parser.parseType(t))
      return failure();
    resultTypes.push_back(t);
  }

  if (parser.resolveOperands(operands, operandTypes, typeLoc, result.operands))
    return failure();
  result.addTypes(resultTypes);
  return success();
}

void BoundaryOp::print(OpAsmPrinter &p) {
  p << " [" << stringifyBoundaryDirection(getDirection()) << "] ";
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p << v; });
  // Elide the `direction` attribute since it is rendered as the bracket
  // predicate above.
  SmallVector<StringRef, 1> elided{"direction"};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elided);
  p << " : ";
  if (getInputs().size() == 2) {
    p << '(' << getInputs()[0].getType() << ", " << getInputs()[1].getType()
      << ") -> ";
  } else {
    p << getInputs()[0].getType() << " -> ";
  }
  auto rTypes = getResultTypes();
  if (rTypes.size() == 1) {
    p << rTypes.front();
  } else {
    p << '(';
    llvm::interleaveComma(rTypes, p);
    p << ')';
  }
}

namespace {

// Reject negative integer-attribute literals. For signed/unsigned
// IntegerType we use `getAPSInt().isNegative()` which is exact. For
// MLIR's signless integer type (`iN`) the parser stores the literal as
// a normalized bit-pattern without preserving the original sign, so a
// source literal `-K : iN` is indistinguishable from its unsigned
// twos-complement twin `(2^N - K) : iN` after parse. The boundary op's
// "non-negative literal" rule therefore only fires on typed (signed or
// unsigned) IntegerAttrs; signless tag literals are always treated as
// the unsigned interpretation of their bit-pattern.
static bool isNegativeIntLiteral(IntegerAttr attr) {
  auto ty = dyn_cast<IntegerType>(attr.getType());
  if (!ty)
    return false;
  if (ty.isSignless())
    return false;
  return attr.getAPSInt().isNegative();
}

static LogicalResult verifyS2t(BoundaryOp op) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() < 1 || operands.size() > 2)
    return op.emitOpError("[s2t] expects 1 or 2 SSA operands, got ")
           << operands.size();
  if (results.size() != 1)
    return op.emitOpError("[s2t] expects exactly 1 result, got ")
           << results.size();

  auto resultTagTy = dyn_cast<BitsTagType>(results[0].getType());
  if (!resultTagTy)
    return op.emitOpError(
        "[s2t] result must be a !fabric.bits_tag<BW, TW> type");
  unsigned resBW = resultTagTy.getWidth();
  unsigned resTW = resultTagTy.getTagWidth();

  auto in0 = dyn_cast<BitsType>(operands[0].getType());
  if (!in0)
    return op.emitOpError("[s2t] operand #0 must be !fabric.bits<BW>, got ")
           << operands[0].getType();
  if (in0.getWidth() != resBW)
    return op.emitOpError("[s2t] operand #0 bits-width ")
           << in0.getWidth() << " must equal result data-width " << resBW;

  if (op.getHwParamsAttr())
    return op.emitOpError("[s2t] must not carry 'hw_params'");

  if (operands.size() == 2) {
    auto in1 = dyn_cast<BitsType>(operands[1].getType());
    if (!in1)
      return op.emitOpError("[s2t] operand #1 must be !fabric.bits<TW>, got ")
             << operands[1].getType();
    if (in1.getWidth() != resTW)
      return op.emitOpError("[s2t] operand #1 bits-width ")
             << in1.getWidth() << " must equal result tag-width " << resTW;

    if (auto sw = op.getSwConfigsAttr())
      if (sw.get("tag"))
        return op.emitOpError(
            "[s2t] two-operand form must not carry 'sw_configs.tag'; the tag "
            "is supplied as the second SSA operand");
    return success();
  }

  // Constant-tag form (1 operand): require sw_configs.tag.
  auto sw = op.getSwConfigsAttr();
  if (!sw)
    return op.emitOpError(
        "[s2t] constant-tag form requires 'sw_configs.tag' integer attribute");
  auto tagAttr = sw.get("tag");
  if (!tagAttr)
    return op.emitOpError(
        "[s2t] constant-tag form requires 'sw_configs.tag' integer attribute");
  auto tagInt = dyn_cast<IntegerAttr>(tagAttr);
  if (!tagInt)
    return op.emitOpError("[s2t] 'sw_configs.tag' must be an IntegerAttr");
  auto tagTy = dyn_cast<IntegerType>(tagInt.getType());
  if (!tagTy)
    return op.emitOpError("[s2t] 'sw_configs.tag' must have an IntegerType");
  if (isNegativeIntLiteral(tagInt))
    return op.emitOpError(
        "'sw_configs.tag' must be a non-negative integer literal");
  unsigned tagWidth = tagTy.getWidth();
  if (tagWidth != resTW)
    return op.emitOpError("[s2t] 'sw_configs.tag' integer attribute width ")
           << tagWidth << " must equal result tag-width " << resTW;
  return success();
}

static LogicalResult verifyT2t(BoundaryOp op) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() != 1)
    return op.emitOpError("[t2t] expects exactly 1 SSA operand, got ")
           << operands.size();
  if (results.size() != 1)
    return op.emitOpError("[t2t] expects exactly 1 result, got ")
           << results.size();

  auto inTy = dyn_cast<BitsTagType>(operands[0].getType());
  auto outTy = dyn_cast<BitsTagType>(results[0].getType());
  if (!inTy || !outTy)
    return op.emitOpError(
        "[t2t] operand and result must be !fabric.bits_tag<BW, TW> types");

  unsigned inBW = inTy.getWidth();
  unsigned outBW = outTy.getWidth();
  unsigned inTW = inTy.getTagWidth();
  unsigned outTW = outTy.getTagWidth();
  if (inBW != outBW)
    return op.emitOpError("[t2t] operand data-width ")
           << inBW << " must equal result data-width " << outBW
           << " (only the tag is remapped)";

  // hw_params: required, length-1 array wrapping a dictionary with key
  // `lut_size` (positive integer).
  auto hp = op.getHwParamsAttr();
  if (!hp)
    return op.emitOpError(
        "[t2t] requires 'hw_params' attribute carrying 'lut_size'");
  if (hp.size() != 1)
    return op.emitOpError(
               "[t2t] 'hw_params' must be a length-1 array wrapping a "
               "dictionary, got length ")
           << hp.size();
  auto hwDict = dyn_cast<DictionaryAttr>(hp[0]);
  if (!hwDict)
    return op.emitOpError(
        "[t2t] 'hw_params' inner element must be a dictionary attribute");
  auto lutSizeAttr = hwDict.get("lut_size");
  if (!lutSizeAttr)
    return op.emitOpError("[t2t] 'hw_params[0]' must contain key 'lut_size'");
  auto lutSizeInt = dyn_cast<IntegerAttr>(lutSizeAttr);
  if (!lutSizeInt)
    return op.emitOpError("[t2t] 'lut_size' must be an IntegerAttr");
  if (isNegativeIntLiteral(lutSizeInt) ||
      lutSizeInt.getValue().getZExtValue() < 1)
    return op.emitOpError(
        "[t2t] 'lut_size' must be a positive integer (>= 1)");
  uint64_t lutSize = lutSizeInt.getValue().getZExtValue();

  // sw_configs: required, must contain `lookup_table`.
  auto sw = op.getSwConfigsAttr();
  if (!sw)
    return op.emitOpError(
        "[t2t] requires 'sw_configs' attribute carrying 'lookup_table'");
  auto lutAttr = sw.get("lookup_table");
  if (!lutAttr)
    return op.emitOpError("[t2t] 'sw_configs' must contain key 'lookup_table'");
  auto lut = dyn_cast<ArrayAttr>(lutAttr);
  if (!lut)
    return op.emitOpError(
        "[t2t] 'lookup_table' must be an array of dictionaries");

  if (lut.size() > lutSize)
    return op.emitOpError(
               "[t2t] 'lookup_table' has more LUT entries than declared "
               "lut_size: ")
           << lut.size() << " > " << lutSize;

  llvm::DenseSet<uint64_t> seenSrcTags;
  for (size_t i = 0; i < lut.size(); ++i) {
    Attribute entryAttr = lut[i];
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    if (!entry)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " must be a dictionary attribute";
    auto srcTagAttr = entry.get("src_tag");
    auto dstTagAttr = entry.get("dst_tag");
    if (!srcTagAttr || !dstTagAttr)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " must have keys 'src_tag' and 'dst_tag'";
    auto srcTagInt = dyn_cast<IntegerAttr>(srcTagAttr);
    auto dstTagInt = dyn_cast<IntegerAttr>(dstTagAttr);
    if (!srcTagInt || !dstTagInt)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " 'src_tag'/'dst_tag' must be IntegerAttr";
    auto srcTagTy = dyn_cast<IntegerType>(srcTagInt.getType());
    auto dstTagTy = dyn_cast<IntegerType>(dstTagInt.getType());
    if (!srcTagTy || !dstTagTy)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " 'src_tag'/'dst_tag' must have IntegerType";
    if (srcTagTy.getWidth() != inTW)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " 'src_tag' integer width " << srcTagTy.getWidth()
             << " must equal operand tag-width " << inTW;
    if (dstTagTy.getWidth() != outTW)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " 'dst_tag' integer width " << dstTagTy.getWidth()
             << " must equal result tag-width " << outTW;
    if (isNegativeIntLiteral(srcTagInt))
      return op.emitOpError("'lookup_table' entry #")
             << i << " has negative src_tag literal";
    if (isNegativeIntLiteral(dstTagInt))
      return op.emitOpError("'lookup_table' entry #")
             << i << " has negative dst_tag literal";

    uint64_t key = srcTagInt.getValue().getZExtValue();
    if (!seenSrcTags.insert(key).second)
      return op.emitOpError("[t2t] duplicate src_tag value ") << key;
  }
  return success();
}

static LogicalResult verifyT2s(BoundaryOp op) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() != 1)
    return op.emitOpError("[t2s] expects exactly 1 SSA operand, got ")
           << operands.size();
  if (results.size() < 1 || results.size() > 2)
    return op.emitOpError("[t2s] expects 1 or 2 results, got ")
           << results.size();

  auto inTy = dyn_cast<BitsTagType>(operands[0].getType());
  if (!inTy)
    return op.emitOpError(
        "[t2s] operand must be a !fabric.bits_tag<BW, TW> type");
  unsigned inBW = inTy.getWidth();
  unsigned inTW = inTy.getTagWidth();

  auto r0 = dyn_cast<BitsType>(results[0].getType());
  if (!r0)
    return op.emitOpError("[t2s] result #0 must be !fabric.bits<BW>, got ")
           << results[0].getType();
  if (r0.getWidth() != inBW)
    return op.emitOpError("[t2s] result #0 bits-width ")
           << r0.getWidth() << " must equal operand data-width " << inBW;

  if (results.size() == 2) {
    auto r1 = dyn_cast<BitsType>(results[1].getType());
    if (!r1)
      return op.emitOpError("[t2s] result #1 must be !fabric.bits<TW>, got ")
             << results[1].getType();
    if (r1.getWidth() != inTW)
      return op.emitOpError("[t2s] result #1 bits-width ")
             << r1.getWidth() << " must equal operand tag-width " << inTW;
  }

  if (op.getHwParamsAttr())
    return op.emitOpError("[t2s] must not carry 'hw_params'");
  if (op.getSwConfigsAttr())
    return op.emitOpError("[t2s] must not carry 'sw_configs'");
  return success();
}

} // namespace

LogicalResult BoundaryOp::verify() {
  switch (getDirection()) {
  case BoundaryDirection::S2t:
    return verifyS2t(*this);
  case BoundaryDirection::T2t:
    return verifyT2t(*this);
  case BoundaryDirection::T2s:
    return verifyT2s(*this);
  }
  return emitOpError("unknown boundary direction");
}
