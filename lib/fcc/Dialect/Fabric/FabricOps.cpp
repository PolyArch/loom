#include "fcc/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace fcc::fabric;

namespace {

static std::optional<unsigned> getFabricScalarWidth(mlir::Type type) {
  if (auto bits = mlir::dyn_cast<fcc::fabric::BitsType>(type))
    return bits.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth();
  if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(type))
    return 16u;
  if (mlir::isa<mlir::Float32Type>(type))
    return 32u;
  if (mlir::isa<mlir::Float64Type>(type))
    return 64u;
  if (type.isIndex())
    return static_cast<unsigned>(fcc::fabric::ADDR_BIT_WIDTH);
  if (mlir::isa<mlir::NoneType>(type))
    return 0u;
  return std::nullopt;
}

static std::optional<unsigned> getSpatialSwitchPayloadWidth(mlir::Type type) {
  if (auto tagged = mlir::dyn_cast<fcc::fabric::TaggedType>(type))
    return getFabricScalarWidth(tagged.getValueType());
  return getFabricScalarWidth(type);
}

static LogicalResult verifyBinaryRowTable(ArrayAttr tableAttr, unsigned expectedRows,
                                         unsigned expectedCols,
                                         Operation *op,
                                         llvm::StringRef name) {
  if (!tableAttr)
    return success();
  if (tableAttr.size() != expectedRows) {
    return op->emitOpError() << name << " must have " << expectedRows
                             << " row(s), got " << tableAttr.size();
  }
  for (unsigned rowIdx = 0; rowIdx < tableAttr.size(); ++rowIdx) {
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(tableAttr[rowIdx]);
    if (!strAttr) {
      return op->emitOpError() << name << " row " << rowIdx
                               << " must be a binary string";
    }
    llvm::StringRef row = strAttr.getValue();
    if (row.size() != expectedCols) {
      return op->emitOpError() << name << " row " << rowIdx << " must have "
                               << expectedCols << " column(s), got "
                               << row.size();
    }
    for (char ch : row) {
      if (ch != '0' && ch != '1')
        return op->emitOpError() << name << " row " << rowIdx
                                 << " must contain only '0' or '1'";
    }
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes, resultTypes;

  if (parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      Type argType;
      if (parser.parseArgument(arg) || parser.parseColonType(argType))
        return failure();
      arg.type = argType;
      args.push_back(arg);
      argTypes.push_back(argType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      do {
        Type resultType;
        if (parser.parseType(resultType))
          return failure();
        resultTypes.push_back(resultType);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  auto funcType =
      FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(funcType));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  return success();
}

void ModuleOp::print(OpAsmPrinter &p) {
  p << " @" << getSymName() << "(";
  auto funcType = getFunctionType();
  auto &body = getBody().front();
  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    if (i > 0)
      p << ", ";
    p.printRegionArgument(body.getArgument(i));
  }
  p << ")";
  if (funcType.getNumResults() > 0) {
    p << " -> (";
    llvm::interleaveComma(funcType.getResults(), p);
    p << ")";
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type"});
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult ModuleOp::verify() {
  // Basic verification - yield operands match results
  auto &body = getBody().front();
  auto yieldOp = dyn_cast<YieldOp>(body.getTerminator());
  if (!yieldOp)
    return emitOpError("expected fabric.yield terminator");

  auto funcType = getFunctionType();
  if (yieldOp.getNumOperands() != funcType.getNumResults())
    return emitOpError("yield operand count mismatch with function results");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

LogicalResult MuxOp::verify() {
  auto *parent = (*this)->getParentOp();
  bool insideFU = false;
  while (parent) {
    if (mlir::isa<FunctionUnitOp>(parent)) {
      insideFU = true;
      break;
    }
    parent = parent->getParentOp();
  }
  if (!insideFU)
    return emitOpError("must appear inside fabric.function_unit");

  unsigned numInputs = getInputs().size();
  unsigned numResults = getResults().size();
  if (numInputs == 0 || numResults == 0)
    return emitOpError("must have at least one input and one result");
  if (numInputs > 1 && numResults > 1)
    return emitOpError("must be either M:1 or 1:M, not M:N");

  mlir::Type expectedType = numInputs > 0 ? getInputs().front().getType()
                                          : getResults().front().getType();
  for (mlir::Value input : getInputs()) {
    if (input.getType() != expectedType)
      return emitOpError("requires all input types to match");
  }
  for (mlir::Value result : getResults()) {
    if (result.getType() != expectedType)
      return emitOpError("requires all result types to match input types");
  }

  if (numInputs == 1 && numResults == 1 &&
      (getDiscard() || getDisconnect())) {
    return emitOpError("1:1 mux cannot set discard or disconnect");
  }

  if (!getDisconnect()) {
    unsigned fanout = std::max(numInputs, numResults);
    if (getSel() < 0 || static_cast<uint64_t>(getSel()) >= fanout) {
      return emitOpError("sel out of range for ")
             << fanout << "-way mux";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SpatialPEOp
//===----------------------------------------------------------------------===//

ParseResult SpatialPEOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional symbol name
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
    // Named form
  }

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes, resultTypes;

  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      Type argType;
      if (parser.parseArgument(arg) || parser.parseColonType(argType))
        return failure();
      arg.type = argType;
      args.push_back(arg);
      argTypes.push_back(argType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      do {
        Type resultType;
        if (parser.parseType(resultType))
          return failure();
        resultTypes.push_back(resultType);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  auto funcType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(resultTypes);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void SpatialPEOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  auto funcType = getFunctionType();
  auto &body = getBody().front();
  p << "(";
  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    if (i > 0)
      p << ", ";
    p.printRegionArgument(body.getArgument(i));
  }
  p << ")";
  if (funcType.getNumResults() > 0) {
    p << " -> (";
    llvm::interleaveComma(funcType.getResults(), p);
    p << ")";
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type"});
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult SpatialPEOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// TemporalPEOp
//===----------------------------------------------------------------------===//

static ParseResult parseTemporalPEHwParams(OpAsmParser &parser,
                                           OperationState &result) {
  if (failed(parser.parseOptionalLSquare())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected temporal_pe hardware parameters in []");
  }

  bool sawNumRegister = false;
  bool sawNumInstruction = false;
  bool sawRegFifoDepth = false;
  bool sawEnableShareOperandBuffer = false;

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

    if (keyword == "num_register" || keyword == "num_instruction" ||
        keyword == "reg_fifo_depth" || keyword == "operand_buffer_size") {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return failure();
      auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
      if (!intAttr) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected integer attribute for temporal_pe hardware parameter '")
               << keyword << "'";
      }
      result.addAttribute(keyword, intAttr);
      sawNumRegister |= (keyword == "num_register");
      sawNumInstruction |= (keyword == "num_instruction");
      sawRegFifoDepth |= (keyword == "reg_fifo_depth");
    } else if (keyword == "enable_share_operand_buffer") {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return failure();
      auto boolAttr = mlir::dyn_cast<BoolAttr>(attr);
      if (!boolAttr) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected bool attribute for temporal_pe hardware parameter '")
               << keyword << "'";
      }
      result.addAttribute(keyword, boolAttr);
      sawEnableShareOperandBuffer = true;
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in temporal_pe hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();

  if (!sawNumRegister || !sawNumInstruction || !sawRegFifoDepth) {
    return parser.emitError(parser.getCurrentLocation(),
                            "temporal_pe hardware parameters must include "
                            "num_register, num_instruction, and reg_fifo_depth");
  }
  if (!sawEnableShareOperandBuffer) {
    result.addAttribute("enable_share_operand_buffer",
                        parser.getBuilder().getBoolAttr(false));
  }

  return success();
}

ParseResult TemporalPEOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
    // Named form
  }

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes, resultTypes;

  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      Type argType;
      if (parser.parseArgument(arg) || parser.parseColonType(argType))
        return failure();
      arg.type = argType;
      args.push_back(arg);
      argTypes.push_back(argType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      do {
        Type t;
        if (parser.parseType(t))
          return failure();
        resultTypes.push_back(t);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  auto funcType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(funcType));
  result.addTypes(resultTypes);

  if (parseTemporalPEHwParams(parser, result))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void TemporalPEOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  auto funcType = getFunctionType();
  auto &body = getBody().front();
  p << "(";
  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    if (i > 0)
      p << ", ";
    p.printRegionArgument(body.getArgument(i));
  }
  p << ")";
  if (funcType.getNumResults() > 0) {
    p << " -> (";
    llvm::interleaveComma(funcType.getResults(), p);
    p << ")";
  }

  p << " [num_register = ";
  p.printAttribute((*this)->getAttr("num_register"));
  p << ", num_instruction = ";
  p.printAttribute((*this)->getAttr("num_instruction"));
  p << ", reg_fifo_depth = ";
  p.printAttribute((*this)->getAttr("reg_fifo_depth"));
  p << ", enable_share_operand_buffer = ";
  p.printAttribute((*this)->getAttr("enable_share_operand_buffer"));
  if (auto operandBufferSize = (*this)->getAttr("operand_buffer_size")) {
    p << ", operand_buffer_size = ";
    p.printAttribute(operandBufferSize);
  }
  p << "]";

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "num_register", "num_instruction",
       "reg_fifo_depth", "enable_share_operand_buffer",
       "operand_buffer_size"});
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult TemporalPEOp::verify() {
  if (getNumInstruction() <= 0)
    return emitOpError("num_instruction must be greater than 0");

  if (getNumRegister() < 0)
    return emitOpError("num_register must be greater than or equal to 0");

  if (getRegFifoDepth() < 0)
    return emitOpError("reg_fifo_depth must be greater than or equal to 0");

  if (getNumRegister() == 0) {
    if (getRegFifoDepth() != 0) {
      return emitOpError(
          "reg_fifo_depth must be 0 when num_register is 0");
    }
  } else if (getRegFifoDepth() < 1) {
    return emitOpError(
        "reg_fifo_depth must be at least 1 when num_register is greater than 0");
  }

  if (getEnableShareOperandBuffer()) {
    if (!getOperandBufferSize()) {
      return emitOpError(
          "operand_buffer_size must be present when enable_share_operand_buffer is true");
    }
    int64_t size = *getOperandBufferSize();
    if (size < 1 || size > 8192) {
      return emitOpError(
          "operand_buffer_size must be in range [1, 8192]");
    }
  } else if (getOperandBufferSize()) {
    return emitOpError(
        "operand_buffer_size must be absent when enable_share_operand_buffer is false");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FunctionUnitOp
//===----------------------------------------------------------------------===//

/// Parse [key = value, ...] hw params block for FunctionUnit operations.
/// Populates latency, interval.
static ParseResult parseFUHwParams(OpAsmParser &parser,
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

    if (keyword == "latency") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("latency", attr);
    } else if (keyword == "interval") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute("interval", attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in FU hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult FunctionUnitOp::parse(OpAsmParser &parser,
                                  OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes, resultTypes;

  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      Type argType;
      if (parser.parseArgument(arg) || parser.parseColonType(argType))
        return failure();
      arg.type = argType;
      args.push_back(arg);
      argTypes.push_back(argType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      do {
        Type t;
        if (parser.parseType(t))
          return failure();
        resultTypes.push_back(t);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  auto funcType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(funcType));

  // Parse [latency = N, interval = N] hw params
  if (parseFUHwParams(parser, result))
    return failure();

  // Parse optional {runtime_config} attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void FunctionUnitOp::print(OpAsmPrinter &p) {
  p << " @" << getSymName();
  auto funcType = getFunctionType();
  auto &body = getBody().front();
  p << "(";
  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    if (i > 0)
      p << ", ";
    p.printRegionArgument(body.getArgument(i));
  }
  p << ")";
  if (funcType.getNumResults() > 0) {
    p << " -> (";
    llvm::interleaveComma(funcType.getResults(), p);
    p << ")";
  }

  // Print [latency = N, interval = N] hw params in square brackets
  bool hasHwParams = false;
  if (auto lat = getLatency()) {
    p << " [latency = " << *lat;
    hasHwParams = true;
  }
  if (auto intv = getInterval()) {
    if (hasHwParams)
      p << ", ";
    else
      p << " [";
    p << "interval = " << *intv;
    hasHwParams = true;
  }
  if (hasHwParams)
    p << "]";

  // Print remaining attrs (excluding known ones) as {key = val}
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "latency", "interval"});
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult FunctionUnitOp::verify() {
  return success();
}

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

  // Print remaining attrs as {key = val}
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "connectivity_table", "route_table",
       "decomposable_bits"});
  p << " : " << getFunctionType();
}

LogicalResult SpatialSwOp::verify() {
  auto funcType = getFunctionType();
  if (funcType.getNumInputs() == 0 || funcType.getNumResults() == 0)
    return emitOpError("must have at least one input and one output");

  bool sawTagged = false;
  bool sawNonTagged = false;
  auto classifyPort = [&](mlir::Type type) {
    if (mlir::isa<fcc::fabric::TaggedType>(type))
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

  // Print {runtime_config} (route_table, etc.)
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "num_route_table", "connectivity_table"});
  p << " : " << getFunctionType();
}

LogicalResult TemporalSwOp::verify() {
  auto funcType = getFunctionType();
  if (funcType.getNumInputs() == 0 || funcType.getNumResults() == 0)
    return emitOpError("must have at least one input and one output");

  if (getNumRouteTable() < 1)
    return emitOpError("num_route_table must be >= 1");

  mlir::Type canonicalType;
  auto verifyPortType = [&](mlir::Type type, llvm::StringRef role,
                            unsigned idx) -> LogicalResult {
    auto tagged = mlir::dyn_cast<fcc::fabric::TaggedType>(type);
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
    if (parser.parseEqual())
      return failure();

    if (keyword == "depth") {
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

  // Print {runtime_config} (bypassed, etc.)
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type", "depth", "bypassable"});
  p << " : " << getFunctionType();
}

LogicalResult FifoOp::verify() {
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

static ParseResult parseOptionalOperandListInParens(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
    bool &hasOperands) {
  hasOperands = succeeded(parser.parseOptionalLParen());
  if (!hasOperands)
    return success();

  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand))
        return failure();
      ops.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  return success();
}

static ParseResult normalizeMemoryConfigAttrs(OpAsmParser &parser,
                                              OperationState &result) {
  auto legacyAttr = result.attributes.get("addr_offset_table");
  if (!legacyAttr)
    return success();
  if (result.attributes.get("addrOffsetTable")) {
    return parser.emitError(parser.getCurrentLocation(),
                            "addr_offset_table specified multiple times");
  }
  result.attributes.set("addrOffsetTable", legacyAttr);
  result.attributes.erase("addr_offset_table");
  return success();
}

static void printNamedAttrsWithAliases(OpAsmPrinter &p, Operation *op,
                                       ArrayRef<StringRef> excludes) {
  SmallVector<NamedAttribute> attrs;
  for (NamedAttribute attr : op->getAttrs()) {
    if (llvm::is_contained(excludes, attr.getName().getValue()))
      continue;
    attrs.push_back(attr);
  }
  if (attrs.empty())
    return;

  p << " attributes {";
  bool first = true;
  for (NamedAttribute attr : attrs) {
    if (!first)
      p << ", ";
    first = false;

    StringRef name = attr.getName().getValue();
    if (name == "addrOffsetTable")
      p << "addr_offset_table";
    else
      p << name;

    if (mlir::isa<mlir::UnitAttr>(attr.getValue()))
      continue;
    p << " = ";
    p.printAttribute(attr.getValue());
  }
  p << "}";
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
                   "numRegion"});
  printNamedAttrsWithAliases(p, getOperation(), excludes);
  p << " : " << getFunctionType();
}

LogicalResult ExtMemoryOp::verify() {
  constexpr int64_t kRegionFieldCount = 5;
  int64_t numRegion = getNumRegion();
  if (numRegion < 1)
    return emitOpError("numRegion must be >= 1");

  auto fnType = getFunctionType();
  if (fnType.getNumInputs() < 1)
    return emitOpError("requires a memref input in the function type");
  if (fnType.getInput(0) != getMemrefType())
    return emitOpError("function type input 0 must match memrefType");

  if (getNumOperands() > 0) {
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
      result.addAttribute("is_private", parser.getBuilder().getBoolAttr(true));
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
  if (getIsPrivate()) {
    startHw();
    p << "is_private";
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
                              "is_private", "numRegion"});
  p << " : " << getFunctionType();
}

LogicalResult MemoryOp::verify() {
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
  return success();
}

LogicalResult DelTagOp::verify() {
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
  auto inputType = mlir::dyn_cast<fcc::fabric::TaggedType>(getTagged().getType());
  auto outputType = mlir::dyn_cast<fcc::fabric::TaggedType>(getResult().getType());
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

  if (tableAttr && outputTagType.getWidth() < 63) {
    uint64_t maxOutput = uint64_t{1} << outputTagType.getWidth();
    for (Attribute attr : *tableAttr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (!intAttr)
        return emitOpError("table entries must be integers");
      int64_t value = intAttr.getInt();
      if (value < 0)
        return emitOpError("table entries must be non-negative");
      if (static_cast<uint64_t>(value) >= maxOutput)
        return emitOpError("table entry exceeds output tag domain");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "fcc/Dialect/Fabric/FabricOps.cpp.inc"
