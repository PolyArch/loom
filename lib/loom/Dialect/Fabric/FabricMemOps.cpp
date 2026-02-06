//===-- FabricMemOps.cpp - Fabric memory operation impls ---------*- C++ -*-===//
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
static void printMemHwParams(OpAsmPrinter &p, int64_t ldCount, int64_t stCount,
                             int64_t lsqDepth,
                             std::optional<bool> isPrivate) {
  p << " [ldCount = " << ldCount << ", stCount = " << stCount;
  if (lsqDepth != 0)
    p << ", lsqDepth = " << lsqDepth;
  if (isPrivate)
    p << ", is_private = " << (*isPrivate ? "true" : "false");
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
                                     Type memrefType) {
  if (ldCount == 0 && stCount == 0)
    return op->emitOpError("ldCount and stCount cannot both be 0");

  if (stCount == 0 && lsqDepth != 0)
    return op->emitOpError("lsqDepth must be 0 when stCount is 0");

  if (stCount > 0 && lsqDepth < 1)
    return op->emitOpError("lsqDepth must be >= 1 when stCount > 0");

  if (!isa<MemRefType>(memrefType))
    return op->emitOpError("memref_type must be a memref type");

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

  printMemHwParams(p, getLdCount(), getStCount(), getLsqDepth(), getIsPrivate());

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
                             getLsqDepth(), getMemrefType())))
    return failure();

  // Static shape required for on-chip memory.
  auto memref = cast<MemRefType>(getMemrefType());
  if (!memref.hasStaticShape())
    return emitOpError("on-chip memory requires static memref shape");

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

  printMemHwParams(p, getLdCount(), getStCount(), getLsqDepth(),
                   /*isPrivate=*/std::nullopt);

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
                             getLsqDepth(), getMemrefType())))
    return failure();

  return success();
}
