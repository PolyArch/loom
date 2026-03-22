//===- FabricOpsMemory.cpp - Switch / Memory / Tag op impls -----*- C++ -*-===//
//
// Implementations for SpatialSwOp, TemporalSwOp, FifoOp, ExtMemoryOp,
// MemoryOp, AddTagOp, DelTagOp, and MapTagOp.
//
//===----------------------------------------------------------------------===//

#include "FabricOpsInternal.h"

#include "loom/Dialect/Fabric/FabricTypes.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace loom::fabric;
using loom::fabric::detail::getFabricScalarWidth;
using loom::fabric::detail::getSpatialSwitchPayloadWidth;
using loom::fabric::detail::getDirectRegionParent;
using loom::fabric::detail::hasDirectRegionParentOfType;
using loom::fabric::detail::kMaxSwitchPorts;
using loom::fabric::detail::normalizeMemoryConfigAttrs;
using loom::fabric::detail::parseOptionalOperandListInParens;
using loom::fabric::detail::printNamedAttrsWithAliases;
using loom::fabric::detail::verifyBinaryRowTable;
using loom::fabric::detail::verifyModuleLevelComponentPlacement;

namespace {

enum class MemoryFamilyKind { Addr, Data, Done };

static unsigned requiredMemoryTagWidth(int64_t laneCount) {
  if (laneCount <= 1)
    return 0;
  return llvm::Log2_64_Ceil(static_cast<uint64_t>(laneCount));
}

static LogicalResult
getMemoryElementWidth(Operation *op, Type memrefType, unsigned &elementWidth) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(memrefType);
  if (!memref)
    return op->emitOpError("memrefType must be a memref");
  auto width = getFabricScalarWidth(memref.getElementType());
  if (!width) {
    return op->emitOpError("memrefType element type is unsupported for memory families: ")
           << memref.getElementType();
  }
  elementWidth = *width;
  return success();
}

static LogicalResult verifyMemoryLaneCount(Operation *op, StringRef name,
                                           int64_t count) {
  if (count < 0)
    return op->emitOpError() << name << " must be >= 0";
  return success();
}

static LogicalResult verifyMemoryFamilyType(Operation *op, Type portType,
                                            MemoryFamilyKind kind,
                                            int64_t laneCount,
                                            unsigned elementWidth) {
  bool requireTagged = laneCount > 1;
  Type payloadType = portType;
  if (auto tagged = mlir::dyn_cast<TaggedType>(portType)) {
    auto tagTy = mlir::dyn_cast<mlir::IntegerType>(tagged.getTagType());
    if (!tagTy) {
      return op->emitOpError("tagged memory family must use an integer tag type: ")
             << portType;
    }
    unsigned minWidth = std::max(1u, requiredMemoryTagWidth(laneCount));
    if (tagTy.getWidth() < minWidth) {
      return op->emitOpError("tagged memory family tag width is too small: ")
             << tagTy.getWidth() << " < " << minWidth;
    }
    payloadType = tagged.getValueType();
  } else if (requireTagged) {
    return op->emitOpError("multi-lane memory family must be tagged: ")
           << portType;
  }

  if (mlir::isa<mlir::MemRefType>(payloadType)) {
    return op->emitOpError("memory request/response family must not use memref payloads: ")
           << payloadType;
  }
  if (!getFabricScalarWidth(payloadType)) {
    return op->emitOpError("memory family payload must be scalar or none, got ")
           << payloadType;
  }
  (void)kind;
  (void)elementWidth;
  return success();
}

static LogicalResult verifyExtMemoryFunctionType(ExtMemoryOp op) {
  if (failed(verifyMemoryLaneCount(op, "ldCount", op.getLdCount())) ||
      failed(verifyMemoryLaneCount(op, "stCount", op.getStCount())))
    return failure();
  if (op.getLsqDepth() < 0)
    return op.emitOpError("lsqDepth must be >= 0");

  unsigned elementWidth = 0;
  if (failed(getMemoryElementWidth(op, op.getMemrefType(), elementWidth)))
    return failure();

  auto fnType = op.getFunctionType();
  unsigned expectedInputs = 1;
  if (op.getLdCount() > 0)
    ++expectedInputs;
  if (op.getStCount() > 0)
    expectedInputs += 2;
  unsigned expectedOutputs = 0;
  if (op.getLdCount() > 0)
    expectedOutputs += 2;
  if (op.getStCount() > 0)
    ++expectedOutputs;

  if (fnType.getNumInputs() != expectedInputs) {
    return op.emitOpError("function_type must have ")
           << expectedInputs << " input(s), got " << fnType.getNumInputs();
  }
  if (fnType.getNumResults() != expectedOutputs) {
    return op.emitOpError("function_type must have ")
           << expectedOutputs << " result(s), got " << fnType.getNumResults();
  }
  if (fnType.getInput(0) != op.getMemrefType())
    return op.emitOpError("function type input 0 must match memrefType");

  unsigned inputIdx = 1;
  if (op.getLdCount() > 0 &&
      failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                    MemoryFamilyKind::Addr, op.getLdCount(),
                                    elementWidth)))
    return failure();
  if (op.getStCount() > 0) {
    if (failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                      MemoryFamilyKind::Addr, op.getStCount(),
                                      elementWidth)))
      return failure();
    if (failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                      MemoryFamilyKind::Data, op.getStCount(),
                                      elementWidth)))
      return failure();
  }

  unsigned resultIdx = 0;
  if (op.getLdCount() > 0) {
    if (failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                      MemoryFamilyKind::Data, op.getLdCount(),
                                      elementWidth)))
      return failure();
    if (failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                      MemoryFamilyKind::Done, op.getLdCount(),
                                      elementWidth)))
      return failure();
  }
  if (op.getStCount() > 0 &&
      failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                    MemoryFamilyKind::Done, op.getStCount(),
                                    elementWidth)))
    return failure();

  return success();
}

static LogicalResult verifyMemoryFunctionType(MemoryOp op) {
  if (failed(verifyMemoryLaneCount(op, "ldCount", op.getLdCount())) ||
      failed(verifyMemoryLaneCount(op, "stCount", op.getStCount())))
    return failure();
  if (op.getLsqDepth() < 0)
    return op.emitOpError("lsqDepth must be >= 0");

  unsigned elementWidth = 0;
  if (failed(getMemoryElementWidth(op, op.getMemrefType(), elementWidth)))
    return failure();

  auto fnType = op.getFunctionType();
  unsigned expectedInputs = 0;
  if (op.getLdCount() > 0)
    ++expectedInputs;
  if (op.getStCount() > 0)
    expectedInputs += 2;
  unsigned expectedOutputs = 0;
  if (op.getLdCount() > 0)
    expectedOutputs += 2;
  if (op.getStCount() > 0)
    ++expectedOutputs;
  if (!op.getIsPrivate())
    ++expectedOutputs;

  if (fnType.getNumInputs() != expectedInputs) {
    return op.emitOpError("function_type must have ")
           << expectedInputs << " input(s), got " << fnType.getNumInputs();
  }
  if (fnType.getNumResults() != expectedOutputs) {
    return op.emitOpError("function_type must have ")
           << expectedOutputs << " result(s), got " << fnType.getNumResults();
  }

  unsigned inputIdx = 0;
  if (op.getLdCount() > 0 &&
      failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                    MemoryFamilyKind::Addr, op.getLdCount(),
                                    elementWidth)))
    return failure();
  if (op.getStCount() > 0) {
    if (failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                      MemoryFamilyKind::Addr, op.getStCount(),
                                      elementWidth)))
      return failure();
    if (failed(verifyMemoryFamilyType(op, fnType.getInput(inputIdx++),
                                      MemoryFamilyKind::Data, op.getStCount(),
                                      elementWidth)))
      return failure();
  }

  unsigned resultIdx = 0;
  if (op.getLdCount() > 0) {
    if (failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                      MemoryFamilyKind::Data, op.getLdCount(),
                                      elementWidth)))
      return failure();
    if (failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                      MemoryFamilyKind::Done, op.getLdCount(),
                                      elementWidth)))
      return failure();
  }
  if (op.getStCount() > 0 &&
      failed(verifyMemoryFamilyType(op, fnType.getResult(resultIdx++),
                                    MemoryFamilyKind::Done, op.getStCount(),
                                    elementWidth)))
    return failure();
  if (!op.getIsPrivate() && fnType.getResult(resultIdx) != op.getMemrefType())
    return op.emitOpError("public memory memref result must match memrefType");

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// SpatialSwOp
//===----------------------------------------------------------------------===//

/// Parse [key = value, ...] hw params block for spatial_sw operations.
/// Handles connectivity_table and route_table as arrays of binary strings,
/// and decomposable_bits as integer.
static ParseResult parseSWHwParams(OpAsmParser &parser,
                                   OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success(); // No brackets.

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

    if (keyword == "connectivity_table" || keyword == "route_table") {
      // Parse array of binary strings: ["110100", "011010", ...]
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(keyword, attr);
    } else if (keyword == "decomposable_bits") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("decomposable_bits", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in SW hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult SpatialSwOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
    // Named form
  }

  // Parse [hw_params] in square brackets
  if (parseSWHwParams(parser, result))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (hasOperands)
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());

  // Parse optional {runtime_config} attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  // Parse function type signature
  FunctionType funcType;
  if (parser.parseType(funcType))
    return failure();
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(funcType.getResults());
  if (hasOperands &&
      parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void SpatialSwOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  // Print hw params in square brackets
  bool hasHwParams = false;
  auto startBracket = [&]() {
    if (!hasHwParams)
      p << " [";
    else
      p << ", ";
    hasHwParams = true;
  };

  if (auto ct = getConnectivityTable()) {
    startBracket();
    p << "connectivity_table = ";
    p.printAttribute(*ct);
  }
  if (auto rt = getRouteTable()) {
    startBracket();
    p << "route_table = ";
    p.printAttribute(*rt);
  }
  if (getDecomposableBits() != -1) {
    startBracket();
    p << "decomposable_bits = " << getDecomposableBits();
  }
  if (hasHwParams)
    p << "]";

  if (getNumOperands() > 0) {
    p << " (";
    p.printOperands(getInputs());
    p << ")";
  }

  // Print remaining attrs as {key = val}
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "connectivity_table", "route_table",
       "decomposable_bits", "inline_instantiation"});
  p << " : " << getFunctionType();
}

LogicalResult SpatialSwOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

  auto funcType = getFunctionType();
  if (funcType.getNumInputs() == 0 || funcType.getNumResults() == 0)
    return emitOpError("must have at least one input and one output");
  if (funcType.getNumInputs() > kMaxSwitchPorts) {
    return emitOpError("may have at most ")
           << kMaxSwitchPorts << " input ports";
  }
  if (funcType.getNumResults() > kMaxSwitchPorts) {
    return emitOpError("may have at most ")
           << kMaxSwitchPorts << " output ports";
  }

  bool sawTagged = false;
  bool sawNonTagged = false;
  auto classifyPort = [&](mlir::Type type) {
    if (mlir::isa<loom::fabric::TaggedType>(type))
      sawTagged = true;
    else
      sawNonTagged = true;
  };

  for (mlir::Type type : funcType.getInputs())
    classifyPort(type);
  for (mlir::Type type : funcType.getResults())
    classifyPort(type);

  if (sawTagged && sawNonTagged) {
    return emitOpError(
        "all ports must share the same tag-kind; mixing tagged and non-tagged ports is not allowed");
  }

  int64_t decomposableBits = getDecomposableBits();
  if (decomposableBits == 0 || decomposableBits < -1) {
    return emitOpError(
        "decomposable_bits must be -1 or a positive integer");
  }

  if (sawTagged && decomposableBits != -1) {
    return emitOpError(
        "tagged spatial_sw cannot be decomposable");
  }

  if (decomposableBits > 0) {
    for (mlir::Type type : funcType.getInputs()) {
      auto width = getSpatialSwitchPayloadWidth(type);
      if (!width) {
        return emitOpError("unsupported port type for decomposable spatial_sw: ")
               << type;
      }
      if ((*width % static_cast<unsigned>(decomposableBits)) != 0) {
        return emitOpError("input payload width must be divisible by decomposable_bits");
      }
    }
    for (mlir::Type type : funcType.getResults()) {
      auto width = getSpatialSwitchPayloadWidth(type);
      if (!width) {
        return emitOpError("unsupported port type for decomposable spatial_sw: ")
               << type;
      }
      if ((*width % static_cast<unsigned>(decomposableBits)) != 0) {
        return emitOpError("output payload width must be divisible by decomposable_bits");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TemporalSwOp
//===----------------------------------------------------------------------===//

/// Parse [hw_params] for temporal_sw: num_route_table, connectivity_table.
static ParseResult parseTemporalSWHwParams(OpAsmParser &parser,
                                           OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success();

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

    if (keyword == "num_route_table") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("num_route_table", attr);
    } else if (keyword == "connectivity_table") {
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute("connectivity_table", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in temporal_sw hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult TemporalSwOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
  }

  // Parse [hw_params]
  if (parseTemporalSWHwParams(parser, result))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (hasOperands)
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());

  // Parse {runtime_config} (route_table, etc.)
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  FunctionType funcType;
  if (parser.parseType(funcType))
    return failure();
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(funcType.getResults());
  if (hasOperands &&
      parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void TemporalSwOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  // Print [hw_params]
  bool hasHw = false;
  auto startHw = [&]() {
    if (!hasHw) p << " [";
    else p << ", ";
    hasHw = true;
  };

  p << " [num_route_table = " << getNumRouteTable();
  hasHw = true;

  if (auto ct = getConnectivityTable()) {
    startHw();
    p << "connectivity_table = ";
    p.printAttribute(*ct);
  }
  if (hasHw)
    p << "]";

  if (getNumOperands() > 0) {
    p << " (";
    p.printOperands(getInputs());
    p << ")";
  }

  // Print {runtime_config} (route_table, etc.)
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "num_route_table", "connectivity_table",
       "inline_instantiation"});
  p << " : " << getFunctionType();
}

LogicalResult TemporalSwOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

  auto funcType = getFunctionType();
  if (funcType.getNumInputs() == 0 || funcType.getNumResults() == 0)
    return emitOpError("must have at least one input and one output");
  if (funcType.getNumInputs() > kMaxSwitchPorts) {
    return emitOpError("may have at most ")
           << kMaxSwitchPorts << " input ports";
  }
  if (funcType.getNumResults() > kMaxSwitchPorts) {
    return emitOpError("may have at most ")
           << kMaxSwitchPorts << " output ports";
  }

  if (getNumRouteTable() < 1)
    return emitOpError("num_route_table must be >= 1");

  mlir::Type canonicalType;
  auto verifyPortType = [&](mlir::Type type, llvm::StringRef role,
                            unsigned idx) -> LogicalResult {
    auto tagged = mlir::dyn_cast<loom::fabric::TaggedType>(type);
    if (!tagged) {
      return emitOpError() << role << " " << idx
                           << " must be !fabric.tagged";
    }
    if (!canonicalType) {
      canonicalType = type;
      return success();
    }
    if (type != canonicalType) {
      return emitOpError() << "all ports must have the same tagged type; "
                           << role << " " << idx << " has type " << type
                           << " but expected " << canonicalType;
    }
    return success();
  };

  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    if (failed(verifyPortType(funcType.getInput(i), "input", i)))
      return failure();
  }
  for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
    if (failed(verifyPortType(funcType.getResult(i), "output", i)))
      return failure();
  }

  if (failed(verifyBinaryRowTable(getConnectivityTable().value_or(ArrayAttr()),
                                  funcType.getNumResults(),
                                  funcType.getNumInputs(), getOperation(),
                                  "connectivity_table"))) {
    return failure();
  }

  if (auto routeTable = getRouteTable()) {
    if (routeTable->size() > static_cast<unsigned>(getNumRouteTable()))
      return emitOpError("route_table cannot contain more rows than num_route_table");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FifoOp
//===----------------------------------------------------------------------===//

/// Parse [hw_params] for fifo: depth, bypassable.
static ParseResult parseFifoHwParams(OpAsmParser &parser,
                                     OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success();

  bool first = true;
  while (true) {
    if (!first && failed(parser.parseOptionalComma()))
      break;
    first = false;

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();

    if (keyword == "depth") {
      if (parser.parseEqual())
        return failure();
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("depth", attr);
    } else if (keyword == "bypassable") {
      // Unit attribute: just parse presence
      result.addAttribute("bypassable", parser.getBuilder().getUnitAttr());
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in fifo hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
  }

  // Parse [hw_params]
  if (parseFifoHwParams(parser, result))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (hasOperands)
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());

  // Parse {runtime_config} (bypassed, etc.)
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  FunctionType funcType;
  if (parser.parseType(funcType))
    return failure();
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(funcType.getResults());
  if (hasOperands &&
      parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  // Print [hw_params]: depth, bypassable
  p << " [depth = " << getDepth();
  if (getBypassable())
    p << ", bypassable";
  p << "]";

  if (getNumOperands() > 0) {
    p << " (";
    p.printOperands(getInputs());
    p << ")";
  }

  // Print {runtime_config} (bypassed, etc.)
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "depth", "bypassable",
       "inline_instantiation"});
  p << " : " << getFunctionType();
}

LogicalResult FifoOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

  auto fnType = getFunctionType();
  if (fnType.getNumInputs() != 1 || fnType.getNumResults() != 1)
    return emitOpError("must have exactly one input and one output");
  if (fnType.getInput(0) != fnType.getResult(0))
    return emitOpError("input and output types must match");
  if (getDepth() < 1)
    return emitOpError("depth must be at least 1");

  bool bypassable = static_cast<bool>(getBypassable());
  bool bypassed = false;
  if (auto attr = mlir::dyn_cast_or_null<mlir::BoolAttr>((*this)->getAttr("bypassed")))
    bypassed = attr.getValue();
  if (bypassed && !bypassable)
    return emitOpError("bypassed requires bypassable");

  return success();
}

//===----------------------------------------------------------------------===//
// ExtMemoryOp
//===----------------------------------------------------------------------===//

/// Parse [hw_params] for extmemory: ldCount, stCount, lsqDepth, memrefType.
static ParseResult parseExtMemHwParams(OpAsmParser &parser,
                                       OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success();

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

    if (keyword == "ldCount" || keyword == "stCount" ||
        keyword == "lsqDepth" || keyword == "numRegion") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(keyword, attr);
    } else if (keyword == "memrefType") {
      TypeAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute("memref_type", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in extmemory hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult ExtMemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
  }

  // Parse [hw_params]
  if (parseExtMemHwParams(parser, result))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (hasOperands)
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());

  // Parse {runtime_config} attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (normalizeMemoryConfigAttrs(parser, result))
    return failure();

  if (parser.parseColon())
    return failure();

  FunctionType funcType;
  if (parser.parseType(funcType))
    return failure();
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(funcType.getResults());

  if (hasOperands &&
      parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void ExtMemoryOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  // Print [hw_params]
  bool hasHw = false;
  auto startHw = [&]() {
    if (!hasHw) p << " [";
    else p << ", ";
    hasHw = true;
  };

  if (auto attr = (*this)->getAttr("ldCount")) {
    startHw();
    p << "ldCount = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("stCount")) {
    startHw();
    p << "stCount = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("lsqDepth")) {
    startHw();
    p << "lsqDepth = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("memref_type")) {
    startHw();
    p << "memrefType = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("numRegion")) {
    startHw();
    p << "numRegion = ";
    p.printAttribute(attr);
  }
  if (hasHw)
    p << "]";

  if (getNumOperands() > 0) {
    p << " (";
    p.printOperands(getInputs());
    p << ")";
  }

  SmallVector<StringRef> excludes = {"sym_name", "function_type"};
  excludes.append({"ldCount", "stCount", "lsqDepth", "memref_type",
                   "numRegion", "inline_instantiation"});
  printNamedAttrsWithAliases(p, getOperation(), excludes);
  p << " : " << getFunctionType();
}

LogicalResult ExtMemoryOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

  if (failed(verifyExtMemoryFunctionType(*this)))
    return failure();

  constexpr int64_t kRegionFieldCount = 5;
  int64_t numRegion = getNumRegion();
  if (numRegion < 1)
    return emitOpError("numRegion must be >= 1");

  if (getNumOperands() > 0) {
    auto fnType = getFunctionType();
    if (getNumOperands() != fnType.getNumInputs())
      return emitOpError("operand count must match function_type inputs");
    auto firstInput = getInputs().front();
    if (firstInput.getType() != getMemrefType())
      return emitOpError("operand 0 must match memrefType");
  }

  auto table = getAddrOffsetTable();
  if (!table)
    return success();
  if (static_cast<int64_t>(table->size()) != numRegion * kRegionFieldCount) {
    return emitOpError("addr_offset_table length must be numRegion * 5");
  }

  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> ranges;
  auto vals = *table;
  for (int64_t i = 0; i < numRegion; ++i) {
    int64_t valid = vals[i * kRegionFieldCount + 0];
    int64_t start = vals[i * kRegionFieldCount + 1];
    int64_t end = vals[i * kRegionFieldCount + 2];
    if (valid == 0)
      continue;
    if (valid != 1)
      return emitOpError("addr_offset_table valid flag must be 0 or 1");
    if (start >= end)
      return emitOpError("addr_offset_table requires start_tag < end_tag");
    if (vals[i * kRegionFieldCount + 4] < 0)
      return emitOpError("addr_offset_table elem_size_log2 must be >= 0");
    for (auto [otherStart, otherEnd] : ranges) {
      if (!(end <= otherStart || start >= otherEnd))
        return emitOpError("addr_offset_table tag ranges must not overlap");
    }
    ranges.push_back({start, end});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

static ParseResult parseMemoryHwParams(OpAsmParser &parser,
                                       OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success();

  bool first = true;
  while (true) {
    if (!first && failed(parser.parseOptionalComma()))
      break;
    first = false;

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();

    if (keyword == "is_private") {
      if (failed(parser.parseOptionalEqual())) {
        result.addAttribute("is_private", parser.getBuilder().getBoolAttr(true));
        continue;
      }
      BoolAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute("is_private", attr);
      continue;
    }

    if (parser.parseEqual())
      return failure();

    if (keyword == "ldCount" || keyword == "stCount" ||
        keyword == "lsqDepth" || keyword == "numRegion") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(keyword, attr);
    } else if (keyword == "memrefType") {
      TypeAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute("memref_type", attr);
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

ParseResult MemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
  }

  if (parseMemoryHwParams(parser, result))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (hasOperands)
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (normalizeMemoryConfigAttrs(parser, result))
    return failure();

  if (parser.parseColon())
    return failure();

  FunctionType funcType;
  if (parser.parseType(funcType))
    return failure();
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(funcType.getResults());

  if (hasOperands &&
      parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void MemoryOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  bool hasHw = false;
  auto startHw = [&]() {
    if (!hasHw) p << " [";
    else p << ", ";
    hasHw = true;
  };

  if (auto attr = (*this)->getAttr("ldCount")) {
    startHw();
    p << "ldCount = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("stCount")) {
    startHw();
    p << "stCount = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("lsqDepth")) {
    startHw();
    p << "lsqDepth = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttr("memref_type")) {
    startHw();
    p << "memrefType = ";
    p.printAttribute(attr);
  }
  if (auto attr = (*this)->getAttrOfType<mlir::BoolAttr>("is_private")) {
    startHw();
    if (attr.getValue()) {
      p << "is_private";
    } else {
      p << "is_private = ";
      p.printAttribute(attr);
    }
  }
  if (auto attr = (*this)->getAttr("numRegion")) {
    startHw();
    p << "numRegion = ";
    p.printAttribute(attr);
  }
  if (hasHw)
    p << "]";

  if (getNumOperands() > 0) {
    p << " (";
    p.printOperands(getInputs());
    p << ")";
  }

  printNamedAttrsWithAliases(p, getOperation(),
                             {"sym_name", "function_type", "ldCount",
                              "stCount", "lsqDepth", "memref_type",
                              "is_private", "numRegion",
                              "inline_instantiation"});
  p << " : " << getFunctionType();
}

LogicalResult MemoryOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

  if (failed(verifyMemoryFunctionType(*this)))
    return failure();

  constexpr int64_t kRegionFieldCount = 5;
  int64_t numRegion = getNumRegion();
  if (numRegion < 1)
    return emitOpError("numRegion must be >= 1");

  auto fnType = getFunctionType();
  if (getNumOperands() > 0 && getNumOperands() != fnType.getNumInputs())
    return emitOpError("operand count must match function_type inputs");

  auto table = getAddrOffsetTable();
  if (!table)
    return success();
  if (static_cast<int64_t>(table->size()) != numRegion * kRegionFieldCount) {
    return emitOpError("addr_offset_table length must be numRegion * 5");
  }

  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> ranges;
  auto vals = *table;
  for (int64_t i = 0; i < numRegion; ++i) {
    int64_t valid = vals[i * kRegionFieldCount + 0];
    int64_t start = vals[i * kRegionFieldCount + 1];
    int64_t end = vals[i * kRegionFieldCount + 2];
    if (valid == 0)
      continue;
    if (valid != 1)
      return emitOpError("addr_offset_table valid flag must be 0 or 1");
    if (start >= end)
      return emitOpError("addr_offset_table requires start_tag < end_tag");
    if (vals[i * kRegionFieldCount + 4] < 0)
      return emitOpError("addr_offset_table elem_size_log2 must be >= 0");
    for (auto [otherStart, otherEnd] : ranges) {
      if (!(end <= otherStart || start >= otherEnd))
        return emitOpError("addr_offset_table tag ranges must not overlap");
    }
    ranges.push_back({start, end});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AddTagOp / DelTagOp / MapTagOp
//===----------------------------------------------------------------------===//

LogicalResult AddTagOp::verify() {
  if (!hasDirectRegionParentOfType<loom::fabric::ModuleOp>(getOperation()))
    return emitOpError("must appear directly inside fabric.module");
  return success();
}

LogicalResult DelTagOp::verify() {
  if (!hasDirectRegionParentOfType<loom::fabric::ModuleOp>(getOperation()))
    return emitOpError("must appear directly inside fabric.module");
  return success();
}

static ParseResult parseMapTagHwParams(OpAsmParser &parser,
                                       OperationState &result) {
  if (failed(parser.parseOptionalLSquare())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected map_tag hardware parameters in []");
  }

  StringRef keyword;
  if (parser.parseKeyword(&keyword) || keyword != "table_size")
    return parser.emitError(parser.getCurrentLocation(),
                            "expected table_size in map_tag hardware parameters");
  if (parser.parseEqual())
    return failure();

  Attribute tableSizeAttr;
  if (parser.parseAttribute(tableSizeAttr)) {
    return failure();
  }
  auto intAttr = mlir::dyn_cast<IntegerAttr>(tableSizeAttr);
  if (!intAttr) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected integer attribute for map_tag table_size");
  }
  result.addAttribute("table_size", intAttr);

  if (failed(parser.parseOptionalComma()) == false) {
    return parser.emitError(parser.getCurrentLocation(),
                            "unexpected extra map_tag hardware parameter");
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult MapTagOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tagged;
  Type inputType, outputType;

  if (parser.parseOperand(tagged))
    return failure();
  if (parseMapTagHwParams(parser, result))
    return failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (parser.parseColon() || parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(outputType)) {
    return failure();
  }

  if (parser.resolveOperand(tagged, inputType, result.operands))
    return failure();
  result.addTypes(outputType);
  return success();
}

void MapTagOp::print(OpAsmPrinter &p) {
  p << " " << getTagged();
  p << " [table_size = ";
  p.printAttribute((*this)->getAttr("table_size"));
  p << "]";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"table_size"});
  p << " : " << getTagged().getType() << " -> " << getResult().getType();
}

LogicalResult MapTagOp::verify() {
  if (!hasDirectRegionParentOfType<loom::fabric::ModuleOp>(getOperation()))
    return emitOpError("must appear directly inside fabric.module");

  auto inputType = mlir::dyn_cast<loom::fabric::TaggedType>(getTagged().getType());
  auto outputType = mlir::dyn_cast<loom::fabric::TaggedType>(getResult().getType());
  if (!inputType || !outputType)
    return emitOpError("requires tagged input and tagged result types");

  if (inputType.getValueType() != outputType.getValueType())
    return emitOpError("requires identical tagged value types");

  auto inputTagType = mlir::dyn_cast<mlir::IntegerType>(inputType.getTagType());
  auto outputTagType = mlir::dyn_cast<mlir::IntegerType>(outputType.getTagType());
  if (!inputTagType || !outputTagType)
    return emitOpError("requires integer tag types");

  auto tableSizeAttr = getTableSizeAttr();
  if (!tableSizeAttr)
    return emitOpError("requires table_size hardware parameter");
  int64_t tableSize = tableSizeAttr.getInt();
  if (tableSize <= 0)
    return emitOpError("table_size must be positive");
  if (tableSize > 256)
    return emitOpError("table_size must be <= 256");

  if (inputTagType.getWidth() < 63) {
    uint64_t maxEntries = uint64_t{1} << inputTagType.getWidth();
    if (static_cast<uint64_t>(tableSize) > maxEntries) {
      return emitOpError("table_size exceeds input tag domain");
    }
  }

  auto tableAttr = getTable();
  if (tableAttr && static_cast<int64_t>(tableAttr->size()) != tableSize) {
    return emitOpError("table length must match table_size");
  }

  uint64_t maxInput =
      inputTagType.getWidth() < 63 ? (uint64_t{1} << inputTagType.getWidth()) : 0;
  uint64_t maxOutput = outputTagType.getWidth() < 63
                           ? (uint64_t{1} << outputTagType.getWidth())
                           : 0;

  if (tableAttr) {
    for (int64_t index = 0, e = static_cast<int64_t>(tableAttr->size()); index < e;
         ++index) {
      Attribute attr = (*tableAttr)[static_cast<size_t>(index)];
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        int64_t value = intAttr.getInt();
        if (value < 0)
          return emitOpError("legacy table entries must be non-negative");
        if (outputTagType.getWidth() < 63 &&
            static_cast<uint64_t>(value) >= maxOutput) {
          return emitOpError("legacy table entry exceeds output tag domain");
        }
        continue;
      }

      auto entry = mlir::dyn_cast<mlir::ArrayAttr>(attr);
      if (!entry || entry.size() != 3)
        return emitOpError("table entries must be integers or [valid, src_tag, dst_tag] triples");

      auto validAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry[0]);
      auto srcAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry[1]);
      auto dstAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry[2]);
      if (!validAttr || !srcAttr || !dstAttr)
        return emitOpError("map_tag table triple entries must contain integers");

      int64_t valid = validAttr.getInt();
      int64_t src = srcAttr.getInt();
      int64_t dst = dstAttr.getInt();
      if (valid != 0 && valid != 1)
        return emitOpError("map_tag table valid bit must be 0 or 1");
      if (src < 0 || dst < 0)
        return emitOpError("map_tag table src_tag and dst_tag must be non-negative");
      if (inputTagType.getWidth() < 63 &&
          static_cast<uint64_t>(src) >= maxInput) {
        return emitOpError("map_tag table src_tag exceeds input tag domain");
      }
      if (outputTagType.getWidth() < 63 &&
          static_cast<uint64_t>(dst) >= maxOutput) {
        return emitOpError("map_tag table dst_tag exceeds output tag domain");
      }
    }
  }

  return success();
}
