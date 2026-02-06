//===-- FabricMemOps.cpp - Fabric memory operation impls ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <cmath>

using namespace mlir;
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
    return op->emitOpError("[COMP_MEMORY_PORTS_EMPTY] "
                           "ldCount and stCount cannot both be 0");

  if (stCount == 0 && lsqDepth != 0)
    return op->emitOpError("[COMP_MEMORY_LSQ_WITHOUT_STORE] "
                           "lsqDepth must be 0 when stCount is 0");

  if (stCount > 0 && lsqDepth < 1)
    return op->emitOpError("[COMP_MEMORY_LSQ_MIN] "
                           "lsqDepth must be >= 1 when stCount > 0");

  if (!isa<MemRefType>(memrefType))
    return op->emitOpError("memref_type must be a memref type");

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

  // Helper to check if a type is index or tagged<index, iK>.
  auto isAddrType = [](Type t) -> bool {
    if (t.isIndex())
      return true;
    if (auto tagged = dyn_cast<TaggedType>(t))
      return tagged.getValueType().isIndex();
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

  // Validate input port types:
  // Layout: [ldaddr * ldCount] [staddr * stCount] [stdata * stCount]
  unsigned expectedInputs = ldCount + 2 * stCount;
  if (inputs.size() != expectedInputs)
    return success(); // Port count mismatch handled elsewhere.

  // Validate load address ports.
  for (int64_t i = 0; i < ldCount; ++i) {
    Type t = inputs[i];
    if (!isAddrType(t))
      return op->emitOpError("[COMP_MEMORY_ADDR_TYPE] "
                             "load address port ")
             << i << " must be index or !dataflow.tagged<index, iK>; got "
             << t;

    // COMP_MEMORY_TAG_REQUIRED / COMP_MEMORY_TAG_FOR_SINGLE
    if (ldCount > 1 && !isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_REQUIRED] "
                             "load address port ")
             << i << " must be tagged when ldCount > 1";
    if (ldCount == 1 && isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_FOR_SINGLE] "
                             "load address port must not be tagged "
                             "when ldCount == 1");

    // COMP_MEMORY_TAG_WIDTH
    if (ldCount > 1 && isTagged(t)) {
      unsigned tw = getTagWidth(t);
      unsigned minW = minTagWidth(ldCount);
      if (tw < minW)
        return op->emitOpError("[COMP_MEMORY_TAG_WIDTH] "
                               "load address port ")
               << i << " tag width " << tw
               << " is smaller than log2Ceil(ldCount) = " << minW;
    }
  }

  // Validate store address ports.
  for (int64_t i = 0; i < stCount; ++i) {
    unsigned idx = ldCount + i;
    Type t = inputs[idx];
    if (!isAddrType(t))
      return op->emitOpError("[COMP_MEMORY_ADDR_TYPE] "
                             "store address port ")
             << i << " must be index or !dataflow.tagged<index, iK>; got "
             << t;

    if (stCount > 1 && !isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_REQUIRED] "
                             "store address port ")
             << i << " must be tagged when stCount > 1";
    if (stCount == 1 && isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_FOR_SINGLE] "
                             "store address port must not be tagged "
                             "when stCount == 1");

    if (stCount > 1 && isTagged(t)) {
      unsigned tw = getTagWidth(t);
      unsigned minW = minTagWidth(stCount);
      if (tw < minW)
        return op->emitOpError("[COMP_MEMORY_TAG_WIDTH] "
                               "store address port ")
               << i << " tag width " << tw
               << " is smaller than log2Ceil(stCount) = " << minW;
    }
  }

  // Validate store data ports.
  for (int64_t i = 0; i < stCount; ++i) {
    unsigned idx = ldCount + stCount + i;
    Type t = inputs[idx];
    Type valT = getValType(t);
    if (valT != elemType)
      return op->emitOpError("[COMP_MEMORY_DATA_TYPE] "
                             "store data port ")
             << i << " value type " << valT
             << " does not match memref element type " << elemType;

    if (stCount > 1 && !isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_REQUIRED] "
                             "store data port ")
             << i << " must be tagged when stCount > 1";
    if (stCount == 1 && isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_FOR_SINGLE] "
                             "store data port must not be tagged "
                             "when stCount == 1");
  }

  // Validate output port types:
  // Layout: [memref?] [lddata * ldCount] [lddone] [stdone]
  unsigned outIdx = 0;
  if (!isPrivate && outputs.size() > 0 && isa<MemRefType>(outputs[0]))
    outIdx++; // Skip memref output.

  // Validate load data ports.
  for (int64_t i = 0; i < ldCount && outIdx < outputs.size(); ++i, ++outIdx) {
    Type t = outputs[outIdx];
    Type valT = getValType(t);
    if (valT != elemType)
      return op->emitOpError("[COMP_MEMORY_DATA_TYPE] "
                             "load data port ")
             << i << " value type " << valT
             << " does not match memref element type " << elemType;

    if (ldCount > 1 && !isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_REQUIRED] "
                             "load data port ")
             << i << " must be tagged when ldCount > 1";
    if (ldCount == 1 && isTagged(t))
      return op->emitOpError("[COMP_MEMORY_TAG_FOR_SINGLE] "
                             "load data port must not be tagged "
                             "when ldCount == 1");
  }

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
    return emitOpError("[COMP_MEMORY_STATIC_REQUIRED] "
                       "on-chip memory requires static memref shape");

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

  // COMP_MEMORY_EXTMEM_PRIVATE: is_private must not be present.
  if (getOperation()->getAttr("is_private"))
    return emitOpError("[COMP_MEMORY_EXTMEM_PRIVATE] "
                       "is_private must not be set on fabric.extmemory");

  // COMP_MEMORY_EXTMEM_BINDING: validate memref operand is a block argument
  // of the parent module (checked for inline form only).
  if (!getInputs().empty()) {
    Value firstInput = getInputs().front();
    if (isa<MemRefType>(firstInput.getType())) {
      if (!isa<BlockArgument>(firstInput))
        return emitOpError("[COMP_MEMORY_EXTMEM_BINDING] "
                           "first memref operand must be a block argument "
                           "of the parent fabric.module");
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
