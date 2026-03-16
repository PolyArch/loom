#include "fcc/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace fcc::fabric;

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
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type"});
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult TemporalPEOp::verify() {
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

/// Parse [hw_params] for extmemory: ldCount, stCount, lsqDepth, memref_type.
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
    } else if (keyword == "memref_type") {
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

  // Parse {runtime_config} attributes
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

void ExtMemoryOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  // Collect known hw param attr names
  SmallVector<StringRef> hwParams = {"ldCount", "stCount", "lsqDepth",
                                     "memref_type", "numRegion"};

  // Print [hw_params]
  bool hasHw = false;
  auto startHw = [&]() {
    if (!hasHw) p << " [";
    else p << ", ";
    hasHw = true;
  };

  for (auto name : hwParams) {
    if (auto attr = (*this)->getAttr(name)) {
      startHw();
      p << name << " = ";
      p.printAttribute(attr);
    }
  }
  if (hasHw)
    p << "]";

  // Print remaining attrs as {key = val}
  SmallVector<StringRef> excludes = {"sym_name", "function_type"};
  excludes.append(hwParams.begin(), hwParams.end());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), excludes);
  p << " : " << getFunctionType();
}

LogicalResult ExtMemoryOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

ParseResult MemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalSymbolName(nameAttr,
                                               SymbolTable::getSymbolAttrName(),
                                               result.attributes))) {
  }

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

void MemoryOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"sym_name", "function_type"});
  p << " : " << getFunctionType();
}

LogicalResult MemoryOp::verify() {
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

LogicalResult MapTagOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "fcc/Dialect/Fabric/FabricOps.cpp.inc"
