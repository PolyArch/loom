#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/BoundaryDataPath.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

namespace mlir {

template <>
void RegisteredOperationName::Model<::fabric::BoundaryOp>::setInherentAttr(
    Operation *op, StringAttr name, Attribute value) {
  auto boundary = cast<::fabric::BoundaryOp>(op);
  if (!value ||
      name != ::fabric::BoundaryOp::getSwConfigsAttrName(op->getName())) {
    ::fabric::BoundaryOp::setInherentAttr(boundary.getProperties(), name,
                                          value);
    return;
  }

  ::fabric::BoundaryOp::Properties converted;
  NamedAttrList attributes;
  attributes.set(name, value);
  if (failed(::fabric::BoundaryOp::setPropertiesFromAttr(
          converted, attributes.getDictionary(op->getContext()),
          [&]() { return mlir::emitError(op->getLoc()); }))) {
    // Keep malformed software configuration present for op verification.
    ::fabric::BoundaryOp::setInherentAttr(
        boundary.getProperties(), name, DictionaryAttr::get(op->getContext()));
    return;
  }
  ::fabric::BoundaryOp::setInherentAttr(boundary.getProperties(), name, value);
}

template <>
LogicalResult
RegisteredOperationName::Model<::fabric::BoundaryOp>::setPropertiesFromAttr(
    OperationName, PropertyRef properties, Attribute attr,
    function_ref<InFlightDiagnostic()> emitError) {
  auto *boundaryProperties =
      properties.as<::fabric::BoundaryOp::Properties *>();
  return ::fabric::BoundaryOp::setPropertiesFromParsedAttr(*boundaryProperties,
                                                           attr, emitError);
}

} // namespace mlir

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
//   [s2t] configurable-tag form (1 SSA operand, optional sw_configs.tag):
//     fabric.boundary [s2t] %data {sw_configs = {tag = K : i<TW>}}
//                : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
//
//   [t2t] tag-remap form (1 SSA operand + hw_params, optional sw_configs):
//     fabric.boundary [t2t] %in
//                {hw_params = [{lut_size = N : i32}],
//                 sw_configs = {lookup_table = [{src_tag = ..., dst_tag = ...},
//                 ...]}}
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
  result.addAttribute("direction",
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
  SmallVector<Type, 2> sourceTypes;
  SmallVector<Type, 2> inputPortTypes;
  SmallVector<Type, 2> resultTypes;

  auto parseInputType = [&]() -> ParseResult {
    Type sourceType;
    if (parser.parseType(sourceType))
      return failure();
    Type inputPortType = sourceType;
    if (succeeded(parser.parseOptionalKeyword("to")))
      if (parser.parseType(inputPortType))
        return failure();
    sourceTypes.push_back(sourceType);
    inputPortTypes.push_back(inputPortType);
    return success();
  };

  // The functional `(...) -> ...` form is used iff there are 2 operands.
  // For 1-operand cases the input type is a bare type. The result side may
  // be either a bare type or a parenthesized list of types.
  if (operands.size() == 2) {
    if (parser.parseLParen() || parseInputType() || parser.parseComma() ||
        parseInputType() || parser.parseRParen())
      return failure();
    if (parser.parseArrow())
      return failure();
  } else {
    if (parseInputType() || parser.parseArrow())
      return failure();
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

  if (parser.resolveOperands(operands, sourceTypes, typeLoc, result.operands))
    return failure();
  bool anyDiffer = false;
  for (auto [sourceType, inputPortType] :
       llvm::zip(sourceTypes, inputPortTypes))
    if (sourceType != inputPortType) {
      anyDiffer = true;
      break;
    }
  if (anyDiffer) {
    result.getOrAddProperties<Properties>().setInnerInputTypes(inputPortTypes);
  }
  result.addTypes(resultTypes);
  return success();
}

void BoundaryOp::print(OpAsmPrinter &p) {
  p << " [" << stringifyBoundaryDirection(getDirection()) << "] ";
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p << v; });
  ArrayRef<Type> innerTypes = getInnerInputTypes();
  SmallVector<Type, 2> inputPortTypes;
  inputPortTypes.reserve(getInputs().size());
  if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
    inputPortTypes.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value input : getInputs())
      inputPortTypes.push_back(input.getType());
  }
  // Elide the `direction` attribute since it is rendered as the bracket
  // predicate above, and the destination input types rendered below.
  SmallVector<StringRef, 1> elided{getDirectionAttrName().getValue()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elided);
  p << " : ";
  auto printInputType = [&](unsigned index) {
    Type sourceType = getInputs()[index].getType();
    Type inputPortType = inputPortTypes[index];
    p << sourceType;
    if (inputPortType && inputPortType != sourceType)
      p << " to " << inputPortType;
  };
  if (getInputs().size() == 2) {
    p << '(';
    printInputType(0);
    p << ", ";
    printInputType(1);
    p << ") -> ";
  } else {
    printInputType(0);
    p << " -> ";
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

static LogicalResult
collectInputPortTypes(BoundaryOp op, SmallVectorImpl<Type> &inputPortTypes) {
  ArrayRef<Type> innerTypes = op.getInnerInputTypes();
  if (!innerTypes.empty()) {
    inputPortTypes.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value input : op.getInputs())
      inputPortTypes.push_back(input.getType());
  }

  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(op.getInputs(), inputPortTypes))) {
    Value input;
    Type inputPortType;
    std::tie(input, inputPortType) = pair;
    Type sourceType = input.getType();
    if (isa<MemRefType>(sourceType) || isa<MemRefType>(inputPortType))
      return op.emitOpError("incoming connection operand #")
             << i
             << ": memref capabilities cannot use the 'to "
                "<destination-type>' clause or serve as boundary transport "
                "ports";
    if (!haveSameFabricModulePortKind(sourceType, inputPortType))
      return op.emitOpError("incoming connection operand #")
             << i << " source type " << sourceType
             << " and destination port type " << inputPortType
             << " must share the same fabric kind (bits or bits_tag)";
  }
  return success();
}

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

static std::optional<DataPathType> getDataPathType(Type type) {
  if (auto bits = dyn_cast<BitsType>(type))
    return DataPathType{DataPathKind::Bits, bits.getWidth(), 0};
  if (auto tagged = dyn_cast<BitsTagType>(type))
    return DataPathType{DataPathKind::BitsTag, tagged.getWidth(),
                        tagged.getTagWidth()};
  return std::nullopt;
}

static LogicalResult verifyBoundaryDataPath(BoundaryOp op, Type sourceType,
                                            Type targetType) {
  const std::optional<DataPathType> source = getDataPathType(sourceType);
  const std::optional<DataPathType> target = getDataPathType(targetType);
  BoundaryDataPathError error = BoundaryDataPathError::None;
  if (!source)
    error = BoundaryDataPathError::InvalidSource;
  else if (!target)
    error = BoundaryDataPathError::InvalidTarget;
  else
    error = checkBoundaryDataPath(op.getDirection(), *source, *target);
  if (error == BoundaryDataPathError::None)
    return success();

  switch (op.getDirection()) {
  case BoundaryDirection::S2t:
    if (error == BoundaryDataPathError::InvalidTarget)
      return op.emitOpError(
          "[s2t] result must be a !fabric.bits_tag<BW, TW> type");
    if (error == BoundaryDataPathError::PayloadWidthMismatch)
      return op.emitOpError("[s2t] operand #0 bits-width ")
             << source->payloadWidthBits << " must equal result data-width "
             << target->payloadWidthBits;
    return op.emitOpError("[s2t] operand #0 must be !fabric.bits<BW>, got ")
           << sourceType;
  case BoundaryDirection::T2t:
    if (error == BoundaryDataPathError::PayloadWidthMismatch)
      return op.emitOpError("[t2t] operand data-width ")
             << source->payloadWidthBits << " must equal result data-width "
             << target->payloadWidthBits << " (only the tag is remapped)";
    return op.emitOpError(
        "[t2t] operand and result must be !fabric.bits_tag<BW, TW> types");
  case BoundaryDirection::T2s:
    if (error == BoundaryDataPathError::InvalidSource)
      return op.emitOpError(
          "[t2s] operand must be a !fabric.bits_tag<BW, TW> type");
    if (error == BoundaryDataPathError::PayloadWidthMismatch)
      return op.emitOpError("[t2s] result #0 bits-width ")
             << target->payloadWidthBits << " must equal operand data-width "
             << source->payloadWidthBits;
    return op.emitOpError("[t2s] result #0 must be !fabric.bits<BW>, got ")
           << targetType;
  }
  return op.emitOpError("unknown boundary direction");
}

static LogicalResult verifyS2t(BoundaryOp op, ArrayRef<Type> inputPortTypes) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() < 1 || operands.size() > 2)
    return op.emitOpError("[s2t] expects 1 or 2 SSA operands, got ")
           << operands.size();
  if (results.size() != 1)
    return op.emitOpError("[s2t] expects exactly 1 result, got ")
           << results.size();

  if (failed(
          verifyBoundaryDataPath(op, inputPortTypes[0], results[0].getType())))
    return failure();
  auto resultTagTy = cast<BitsTagType>(results[0].getType());
  unsigned resTW = resultTagTy.getTagWidth();

  if (op.getHwParamsAttr())
    return op.emitOpError("[s2t] must not carry 'hw_params'");

  if (operands.size() == 2) {
    auto in1 = dyn_cast<BitsType>(inputPortTypes[1]);
    if (!in1)
      return op.emitOpError("[s2t] operand #1 must be !fabric.bits<TW>, got ")
             << inputPortTypes[1];
    if (in1.getWidth() != resTW)
      return op.emitOpError("[s2t] operand #1 bits-width ")
             << in1.getWidth() << " must equal result tag-width " << resTW;

    if (op.getSwConfigsAttr())
      return op.emitOpError(
          "[s2t] two-operand form must not carry 'sw_configs'");
    return success();
  }

  // The one-operand form is an unconfigured capability when sw_configs is
  // absent. A present configured projection contains only `tag`.
  auto sw = op.getSwConfigsAttr();
  if (!sw)
    return success();
  auto tagAttr = sw.get("tag");
  if (sw.size() != 1 || !tagAttr)
    return op.emitOpError(
        "[s2t] present 'sw_configs' must contain exactly the 'tag' field");
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

static LogicalResult verifyT2t(BoundaryOp op, ArrayRef<Type> inputPortTypes) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() != 1)
    return op.emitOpError("[t2t] expects exactly 1 SSA operand, got ")
           << operands.size();
  if (results.size() != 1)
    return op.emitOpError("[t2t] expects exactly 1 result, got ")
           << results.size();

  if (failed(
          verifyBoundaryDataPath(op, inputPortTypes[0], results[0].getType())))
    return failure();
  auto inTy = cast<BitsTagType>(inputPortTypes[0]);
  auto outTy = cast<BitsTagType>(results[0].getType());
  unsigned inTW = inTy.getTagWidth();
  unsigned outTW = outTy.getTagWidth();

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
  if (isNegativeIntLiteral(lutSizeInt) || lutSizeInt.getValue().isZero())
    return op.emitOpError("[t2t] 'lut_size' must be a positive integer (>= 1)");
  uint64_t lutSize = lutSizeInt.getValue().getLimitedValue();

  // Absence denotes the canonical unconfigured capability. A present
  // configured projection contains only a nonempty `lookup_table`.
  auto sw = op.getSwConfigsAttr();
  if (!sw)
    return success();
  auto lutAttr = sw.get("lookup_table");
  if (sw.size() != 1 || !lutAttr)
    return op.emitOpError(
        "[t2t] present 'sw_configs' must contain exactly the 'lookup_table' "
        "field");
  auto lut = dyn_cast<ArrayAttr>(lutAttr);
  if (!lut)
    return op.emitOpError(
        "[t2t] 'lookup_table' must be an array of dictionaries");
  if (lut.empty())
    return op.emitOpError("[t2t] a present 'lookup_table' must be nonempty");

  if (lut.size() > lutSize)
    return op.emitOpError(
               "[t2t] 'lookup_table' has more LUT entries than declared "
               "lut_size: ")
           << lut.size() << " > " << lutSize;

  llvm::DenseSet<llvm::APInt> seenSrcTags;
  for (size_t i = 0; i < lut.size(); ++i) {
    Attribute entryAttr = lut[i];
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    if (!entry)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " must be a dictionary attribute";
    auto srcTagAttr = entry.get("src_tag");
    auto dstTagAttr = entry.get("dst_tag");
    if (entry.size() != 2 || !srcTagAttr || !dstTagAttr)
      return op.emitOpError("[t2t] 'lookup_table' entry #")
             << i << " must contain exactly 'src_tag' and 'dst_tag'";
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

    llvm::APInt key = srcTagInt.getValue();
    if (!seenSrcTags.insert(key).second)
      return op.emitOpError("[t2t] duplicate src_tag value");
  }
  return success();
}

static LogicalResult verifyT2s(BoundaryOp op, ArrayRef<Type> inputPortTypes) {
  auto operands = op.getInputs();
  auto results = op.getOutputs();
  if (operands.size() != 1)
    return op.emitOpError("[t2s] expects exactly 1 SSA operand, got ")
           << operands.size();
  if (results.size() < 1 || results.size() > 2)
    return op.emitOpError("[t2s] expects 1 or 2 results, got ")
           << results.size();

  if (failed(
          verifyBoundaryDataPath(op, inputPortTypes[0], results[0].getType())))
    return failure();
  auto inTy = cast<BitsTagType>(inputPortTypes[0]);
  unsigned inTW = inTy.getTagWidth();

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
  if (failed(verifyInnerInputTypesProperty(getOperation(), getInputs(),
                                           getInnerInputTypes())))
    return failure();

  SmallVector<Type, 2> inputPortTypes;
  if (failed(collectInputPortTypes(*this, inputPortTypes)))
    return failure();
  switch (getDirection()) {
  case BoundaryDirection::S2t:
    return verifyS2t(*this, inputPortTypes);
  case BoundaryDirection::T2t:
    return verifyT2t(*this, inputPortTypes);
  case BoundaryDirection::T2s:
    return verifyT2s(*this, inputPortTypes);
  }
  return emitOpError("unknown boundary direction");
}
