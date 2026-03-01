//===-- FabricMemOps.cpp - Fabric memory operation impls ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "loom/Hardware/Common/FabricError.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <cmath>

using namespace mlir;
using namespace loom;
using namespace loom::fabric;
using namespace loom::dataflow;

//===----------------------------------------------------------------------===//
// Shared memory parse/print helpers
//===----------------------------------------------------------------------===//

/// Parse [hw_params] for memory operations.
/// Handles: ldCount, stCount, lsqDepth, is_private (memory only).
static ParseResult parseMemHwParams(OpAsmParser &parser,
                                    OperationState &result,
                                    bool allowPrivate) {
  if (parser.parseLSquare())
    return failure();

  bool first = true;
  while (true) {
    if (!first && failed(parser.parseOptionalComma()))
      break;
    first = false;

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();
    if (parser.parseEqual())
      return failure();

    if (keyword == "ldCount") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("ldCount", attr);
    } else if (keyword == "stCount") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("stCount", attr);
    } else if (keyword == "lsqDepth") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("lsqDepth", attr);
    } else if (keyword == "is_private" && allowPrivate) {
      BoolAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute("is_private", attr);
    } else if (keyword == "numRegion") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("numRegion", attr);
    } else if (keyword == "addr_offset_table") {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return failure();
      if (!isa<DenseI64ArrayAttr>(attr))
        return parser.emitError(parser.getCurrentLocation(),
                                "expected array<i64: ...> for "
                                "addr_offset_table");
      result.addAttribute("addrOffsetTable", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in memory hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

/// Parse the common memory format after [hw_params]:
///   (%operands) : memref<T>, (input_types) -> (output_types)  [inline]
///   : memref<T>, (input_types) -> (output_types)              [named]
static ParseResult parseMemTypeSignature(OpAsmParser &parser,
                                         OperationState &result,
                                         bool isNamed) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (!isNamed) {
    if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
      return failure();
  }

  if (parser.parseColon())
    return failure();

  // Parse memref type.
  Type memrefType;
  if (parser.parseType(memrefType))
    return failure();
  if (!isa<MemRefType>(memrefType))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected memref type");
  result.addAttribute("memref_type", TypeAttr::get(memrefType));

  if (parser.parseComma())
    return failure();

  // Parse (input_types) -> (output_types) as FunctionType.
  FunctionType fnType;
  if (parser.parseType(fnType))
    return failure();

  if (isNamed) {
    result.addAttribute("function_type", TypeAttr::get(fnType));
  } else {
    auto inputTypes = fnType.getInputs();
    if (parser.resolveOperands(operands, inputTypes, parser.getNameLoc(),
                               result.operands))
      return failure();
    result.addTypes(fnType.getResults());
  }

  return success();
}

/// Print [hw_params] for memory ops.
static void printMemHwParams(OpAsmPrinter &p, Operation *op,
                             int64_t ldCount, int64_t stCount,
                             int64_t lsqDepth,
                             std::optional<bool> isPrivate,
                             int64_t numRegion = 1) {
  p << " [ldCount = " << ldCount << ", stCount = " << stCount;
  if (lsqDepth != 0)
    p << ", lsqDepth = " << lsqDepth;
  if (isPrivate)
    p << ", is_private = " << (*isPrivate ? "true" : "false");
  if (numRegion != 1)
    p << ", numRegion = " << numRegion;
  if (auto aot = op->getAttr("addrOffsetTable")) {
    p << ", addr_offset_table = ";
    p.printAttribute(aot);
  }
  p << "]";
}

/// Print the memory type signature.
static void printMemTypeSignature(OpAsmPrinter &p, Type memrefType,
                                  TypeRange inputTypes, TypeRange outputTypes,
                                  ValueRange operands, bool isNamed) {
  if (!isNamed) {
    p << " (";
    p.printOperands(operands);
    p << ")";
  }

  p << " : ";
  p.printType(memrefType);
  p << ", (";
  llvm::interleaveComma(inputTypes, p, [&](Type t) { p.printType(t); });
  p << ") -> (";
  llvm::interleaveComma(outputTypes, p, [&](Type t) { p.printType(t); });
  p << ")";
}

/// Shared memory verification logic.
static LogicalResult verifyMemCommon(Operation *op, int64_t ldCount,
                                     int64_t stCount, int64_t lsqDepth,
                                     int64_t numRegion, Type memrefType) {
  if (ldCount == 0 && stCount == 0)
    return op->emitOpError(cplErrMsg(CplError::MEMORY_PORTS_EMPTY,
                           "ldCount and stCount cannot both be 0"));

  if (stCount == 0 && lsqDepth != 0)
    return op->emitOpError(cplErrMsg(CplError::MEMORY_LSQ_WITHOUT_STORE,
                           "lsqDepth must be 0 when stCount is 0"));

  if (stCount > 0 && lsqDepth < 1)
    return op->emitOpError(cplErrMsg(CplError::MEMORY_LSQ_MIN,
                           "lsqDepth must be >= 1 when stCount > 0"));

  if (numRegion < 1)
    return op->emitOpError(cplErrMsg(CplError::MEMORY_INVALID_REGION,
                           "numRegion must be >= 1"));

  if (!isa<MemRefType>(memrefType))
    return op->emitOpError("memref_type must be a memref type");

  return success();
}

/// Verify addr_offset_table entries against numRegion, tag ranges, and overlap.
static LogicalResult
verifyAddrOffsetTable(Operation *op, int64_t numRegion,
                      std::optional<ArrayRef<int64_t>> aotOpt,
                      const char *emptyRangeCode, const char *overlapCode) {
  if (!aotOpt)
    return success();

  ArrayRef<int64_t> table = *aotOpt;

  // Each entry has 4 fields: [valid, start_tag, end_tag, base_addr].
  // The flat array must have exactly numRegion * 4 values.
  if (table.size() % 4 != 0)
    return op->emitOpError(
        "addr_offset_table must have 4 fields per entry "
        "(valid, start_tag, end_tag, base_addr); got ")
           << table.size() << " values";

  int64_t actualEntries = static_cast<int64_t>(table.size()) / 4;
  if (actualEntries != numRegion)
    return op->emitOpError("addr_offset_table has ")
           << actualEntries << " entries but numRegion is " << numRegion;

  // Per-entry validation.
  for (int64_t i = 0; i < numRegion; ++i) {
    int64_t valid = table[i * 4];
    int64_t startTag = table[i * 4 + 1];
    int64_t endTag = table[i * 4 + 2];

    if (valid != 0 && valid != 1)
      return op->emitOpError("addr_offset_table entry ")
             << i << " has invalid valid field " << valid;

    if (valid) {
      // Half-open interval: start_tag < end_tag required.
      if (endTag <= startTag)
        return op->emitOpError(cplErrMsg(emptyRangeCode,
                               "addr_offset_table entry "))
               << i << " has end_tag " << endTag
               << " <= start_tag " << startTag;
    }
  }

  // Check pairwise overlap among valid entries using half-open interval test.
  for (int64_t i = 0; i < numRegion; ++i) {
    if (table[i * 4] == 0) continue;
    int64_t aStart = table[i * 4 + 1];
    int64_t aEnd = table[i * 4 + 2];
    for (int64_t j = i + 1; j < numRegion; ++j) {
      if (table[j * 4] == 0) continue;
      int64_t bStart = table[j * 4 + 1];
      int64_t bEnd = table[j * 4 + 2];
      if (aStart < bEnd && bStart < aEnd)
        return op->emitOpError(cplErrMsg(overlapCode,
                               "addr_offset_table entries "))
               << i << " and " << j << " have overlapping tag ranges";
    }
  }

  return success();
}

/// Validate memory port types against the function_type if present.
static LogicalResult verifyMemPortTypes(Operation *op, int64_t ldCount,
                                        int64_t stCount, Type memrefType,
                                        bool isPrivate,
                                        std::optional<FunctionType> fnType) {
  if (!fnType)
    return success();

  auto memref = cast<MemRefType>(memrefType);
  Type elemType = memref.getElementType();
  auto inputs = fnType->getInputs();
  auto outputs = fnType->getResults();

  // Helper to check if a type is an address type:
  // bits<ADDR_BIT_WIDTH>, tagged<bits<ADDR_BIT_WIDTH>, iK>,
  // or legacy index / tagged<index, iK>.
  auto isAddrType = [](Type t) -> bool {
    Type v = t;
    if (auto tagged = dyn_cast<TaggedType>(t))
      v = tagged.getValueType();
    if (v.isIndex())
      return true;
    if (auto bits = dyn_cast<BitsType>(v))
      return bits.getWidth() == loom::ADDR_BIT_WIDTH;
    return false;
  };

  // Helper to check if a type is tagged.
  auto isTagged = [](Type t) -> bool {
    return isa<TaggedType>(t);
  };

  // Helper to get value type from potentially tagged type.
  auto getValType = [](Type t) -> Type {
    if (auto tagged = dyn_cast<TaggedType>(t))
      return tagged.getValueType();
    return t;
  };

  // Helper to get tag width from a tagged type (0 if not tagged).
  auto getTagWidth = [](Type t) -> unsigned {
    if (auto tagged = dyn_cast<TaggedType>(t))
      return tagged.getTagType().getWidth();
    return 0;
  };

  // Compute minimum tag width for a count.
  auto minTagWidth = [](int64_t count) -> unsigned {
    if (count <= 1)
      return 0;
    return static_cast<unsigned>(std::ceil(std::log2(count)));
  };

  // Presence-based input layout: [ld_addr?] [st_addr? st_data?]
  unsigned expectedInputs = 0;
  if (ldCount > 0) expectedInputs++;       // ld_addr
  if (stCount > 0) expectedInputs += 2;    // st_addr + st_data
  if (inputs.size() != expectedInputs)
    return success(); // Port count mismatch handled elsewhere.

  // Uniform tagging rule: if ldCount > 1 OR stCount > 1, all ports tagged.
  bool needTag = (ldCount > 1 || stCount > 1);
  unsigned maxCount = std::max(ldCount, stCount);

  // Track current input port index.
  unsigned inIdx = 0;

  // Validate ld_addr (singular) if ldCount > 0.
  if (ldCount > 0 && inIdx < inputs.size()) {
    Type t = inputs[inIdx++];
    if (!isAddrType(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_ADDR_TYPE,
                             "load address port must be index or "
                             "!dataflow.tagged<index, iK>; got "))
             << t;
    if (needTag && !isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_REQUIRED,
                             "load address port must be tagged when "
                             "ldCount > 1 or stCount > 1"));
    if (!needTag && isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_FOR_SINGLE,
                             "load address port must not be tagged "
                             "when both ldCount <= 1 and stCount <= 1"));
    if (needTag && isTagged(t)) {
      unsigned tw = getTagWidth(t);
      unsigned minW = minTagWidth(maxCount);
      if (tw < minW)
        return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_WIDTH,
                               "load address port tag width "))
               << tw << " is smaller than log2Ceil(max(ldCount,stCount)) = "
               << minW;
    }
  }

  // Validate st_addr (singular) if stCount > 0.
  if (stCount > 0 && inIdx < inputs.size()) {
    Type t = inputs[inIdx++];
    if (!isAddrType(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_ADDR_TYPE,
                             "store address port must be index or "
                             "!dataflow.tagged<index, iK>; got "))
             << t;
    if (needTag && !isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_REQUIRED,
                             "store address port must be tagged when "
                             "ldCount > 1 or stCount > 1"));
    if (!needTag && isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_FOR_SINGLE,
                             "store address port must not be tagged "
                             "when both ldCount <= 1 and stCount <= 1"));
    if (needTag && isTagged(t)) {
      unsigned tw = getTagWidth(t);
      unsigned minW = minTagWidth(maxCount);
      if (tw < minW)
        return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_WIDTH,
                               "store address port tag width "))
               << tw << " is smaller than log2Ceil(max(ldCount,stCount)) = "
               << minW;
    }
  }

  // Helper to get bit width from a type for width-compatible matching.
  auto getBitWidth = [](Type t) -> std::optional<unsigned> {
    if (auto bits = dyn_cast<BitsType>(t)) return bits.getWidth();
    if (auto intTy = dyn_cast<IntegerType>(t)) return intTy.getWidth();
    if (isa<Float32Type>(t)) return 32u;
    if (isa<Float64Type>(t)) return 64u;
    if (isa<Float16Type, BFloat16Type>(t)) return 16u;
    if (t.isIndex()) return (unsigned)loom::ADDR_BIT_WIDTH;
    return std::nullopt;
  };

  // Helper to check data port value type matches memref element type.
  // Allows exact match OR width-compatible match (bits<N> vs native type).
  auto isDataTypeCompatible = [&](Type valT, Type elemT) -> bool {
    if (valT == elemT) return true;
    auto valW = getBitWidth(valT);
    auto elemW = getBitWidth(elemT);
    return valW && elemW && *valW == *elemW;
  };

  // Validate st_data (singular) if stCount > 0.
  if (stCount > 0 && inIdx < inputs.size()) {
    Type t = inputs[inIdx++];
    Type valT = getValType(t);
    if (!isDataTypeCompatible(valT, elemType))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_DATA_TYPE,
                             "store data port value type "))
             << valT << " does not match memref element type " << elemType;
    if (needTag && !isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_REQUIRED,
                             "store data port must be tagged when "
                             "ldCount > 1 or stCount > 1"));
    if (!needTag && isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_FOR_SINGLE,
                             "store data port must not be tagged "
                             "when both ldCount <= 1 and stCount <= 1"));
  }

  // Presence-based output layout: [memref?] [ld_data? ld_done?] [st_done?]
  unsigned outIdx = 0;
  if (!isPrivate && outputs.size() > 0 && isa<MemRefType>(outputs[0]))
    outIdx++; // Skip memref output.

  // Validate ld_data (singular) if ldCount > 0.
  if (ldCount > 0 && outIdx < outputs.size()) {
    Type t = outputs[outIdx++];
    Type valT = getValType(t);
    if (!isDataTypeCompatible(valT, elemType))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_DATA_TYPE,
                             "load data port value type "))
             << valT << " does not match memref element type " << elemType;
    if (needTag && !isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_REQUIRED,
                             "load data port must be tagged when "
                             "ldCount > 1 or stCount > 1"));
    if (!needTag && isTagged(t))
      return op->emitOpError(cplErrMsg(CplError::MEMORY_TAG_FOR_SINGLE,
                             "load data port must not be tagged "
                             "when both ldCount <= 1 and stCount <= 1"));
  }

  // ld_done and st_done: skip detailed type checking for now
  // (they carry tag or none, validated structurally by port count)

  return success();
}

//===----------------------------------------------------------------------===//
// MemoryOp parse/print/verify
//===----------------------------------------------------------------------===//

ParseResult MemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse [hw_params].
  if (parseMemHwParams(parser, result, /*allowPrivate=*/true))
    return failure();

  // Parse type signature.
  if (parseMemTypeSignature(parser, result, isNamed))
    return failure();

  return success();
}

void MemoryOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  printMemHwParams(p, getOperation(), getLdCount(), getStCount(), getLsqDepth(),
                   getIsPrivate(), getNumRegion());

  SmallVector<Type> inputTypes, outputTypes;
  if (isNamed) {
    auto fnType = *getFunctionType();
    inputTypes.assign(fnType.getInputs().begin(), fnType.getInputs().end());
    outputTypes.assign(fnType.getResults().begin(), fnType.getResults().end());
  } else {
    for (auto v : getInputs())
      inputTypes.push_back(v.getType());
    for (auto v : getOutputs())
      outputTypes.push_back(v.getType());
  }

  printMemTypeSignature(p, getMemrefType(), inputTypes, outputTypes,
                        getInputs(), isNamed);
}

LogicalResult MemoryOp::verify() {
  if (failed(verifyMemCommon(getOperation(), getLdCount(), getStCount(),
                             getLsqDepth(), getNumRegion(), getMemrefType())))
    return failure();

  if (failed(verifyAddrOffsetTable(
          getOperation(), getNumRegion(), getAddrOffsetTable(),
          "CFG_MEMORY_EMPTY_TAG_RANGE", "CFG_MEMORY_OVERLAP_TAG_REGION")))
    return failure();

  // Static shape required for on-chip memory.
  auto memref = cast<MemRefType>(getMemrefType());
  if (!memref.hasStaticShape())
    return emitOpError(cplErrMsg(CplError::MEMORY_STATIC_REQUIRED,
                       "on-chip memory requires static memref shape"));

  // Validate port types.
  bool isPriv = getIsPrivate();
  std::optional<FunctionType> fnType = getFunctionType();
  if (!fnType) {
    // Inline form: derive function type from SSA operands/results.
    SmallVector<Type> inTypes, outTypes;
    for (auto v : getInputs())
      inTypes.push_back(v.getType());
    for (auto v : getOutputs())
      outTypes.push_back(v.getType());
    fnType = FunctionType::get(getContext(), inTypes, outTypes);
  }
  if (failed(verifyMemPortTypes(getOperation(), getLdCount(), getStCount(),
                                getMemrefType(), isPriv, fnType)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ExtMemoryOp parse/print/verify
//===----------------------------------------------------------------------===//

ParseResult ExtMemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse [hw_params].
  if (parseMemHwParams(parser, result, /*allowPrivate=*/false))
    return failure();

  // Parse type signature.
  if (parseMemTypeSignature(parser, result, isNamed))
    return failure();

  return success();
}

void ExtMemoryOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  printMemHwParams(p, getOperation(), getLdCount(), getStCount(), getLsqDepth(),
                   /*isPrivate=*/std::nullopt, getNumRegion());

  SmallVector<Type> inputTypes, outputTypes;
  if (isNamed) {
    auto fnType = *getFunctionType();
    inputTypes.assign(fnType.getInputs().begin(), fnType.getInputs().end());
    outputTypes.assign(fnType.getResults().begin(), fnType.getResults().end());
  } else {
    for (auto v : getInputs())
      inputTypes.push_back(v.getType());
    for (auto v : getOutputs())
      outputTypes.push_back(v.getType());
  }

  printMemTypeSignature(p, getMemrefType(), inputTypes, outputTypes,
                        getInputs(), isNamed);
}

LogicalResult ExtMemoryOp::verify() {
  if (failed(verifyMemCommon(getOperation(), getLdCount(), getStCount(),
                             getLsqDepth(), getNumRegion(), getMemrefType())))
    return failure();

  if (failed(verifyAddrOffsetTable(
          getOperation(), getNumRegion(), getAddrOffsetTable(),
          "CFG_EXTMEMORY_EMPTY_TAG_RANGE",
          "CFG_EXTMEMORY_OVERLAP_TAG_REGION")))
    return failure();

  // CPL_MEMORY_EXTMEM_PRIVATE: is_private must not be present.
  if (getOperation()->getAttr("is_private"))
    return emitOpError(cplErrMsg(CplError::MEMORY_EXTMEM_PRIVATE,
                       "is_private must not be set on fabric.extmemory"));

  // CPL_MEMORY_EXTMEM_BINDING: validate memref operand is a block argument
  // of the parent module (checked for inline form only).
  if (!getInputs().empty()) {
    Value firstInput = getInputs().front();
    if (isa<MemRefType>(firstInput.getType())) {
      if (!isa<BlockArgument>(firstInput))
        return emitOpError(cplErrMsg(CplError::MEMORY_EXTMEM_BINDING,
                           "first memref operand must be a block argument "
                           "of the parent fabric.module"));
    }
  }

  // Validate port types.
  std::optional<FunctionType> fnType = getFunctionType();
  if (!fnType) {
    // Inline form: derive function type from SSA operands/results.
    SmallVector<Type> inTypes, outTypes;
    for (auto v : getInputs())
      inTypes.push_back(v.getType());
    for (auto v : getOutputs())
      outTypes.push_back(v.getType());
    fnType = FunctionType::get(getContext(), inTypes, outTypes);
  }
  if (failed(verifyMemPortTypes(getOperation(), getLdCount(), getStCount(),
                                getMemrefType(), /*isPrivate=*/false, fnType)))
    return failure();

  return success();
}
