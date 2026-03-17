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

LogicalResult MapTagOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "fcc/Dialect/Fabric/FabricOps.cpp.inc"
