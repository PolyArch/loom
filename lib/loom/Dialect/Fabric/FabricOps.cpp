//===- FabricOps.cpp - Core Fabric op implementations -----------*- C++ -*-===//
//
// Implementations for ModuleOp, InstanceOp, MuxOp, SpatialPEOp,
// TemporalPEOp, and FunctionUnitOp, plus shared helper definitions.
//
//===----------------------------------------------------------------------===//

#include "FabricOpsInternal.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace loom::fabric;
using loom::fabric::detail::hasDirectRegionParentOfType;
using loom::fabric::detail::parseOptionalOperandListInParens;
using loom::fabric::detail::verifyModuleLevelComponentPlacement;

namespace {

bool isNativeFunctionUnitType(mlir::Type type) {
  if (!type)
    return false;
  if (mlir::isa<mlir::NoneType>(type) || type.isIndex() ||
      mlir::isa<mlir::FloatType>(type)) {
    return true;
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.isSignless();
  return false;
}

bool isAllowedFunctionUnitBodyOp(llvm::StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Case("fabric.mux", true)
      .Case("arith.addf", true)
      .Case("arith.addi", true)
      .Case("arith.andi", true)
      .Case("arith.cmpf", true)
      .Case("arith.cmpi", true)
      .Case("arith.divf", true)
      .Case("arith.divsi", true)
      .Case("arith.divui", true)
      .Case("arith.extsi", true)
      .Case("arith.extui", true)
      .Case("arith.fptosi", true)
      .Case("arith.fptoui", true)
      .Case("arith.index_cast", true)
      .Case("arith.index_castui", true)
      .Case("arith.mulf", true)
      .Case("arith.muli", true)
      .Case("arith.negf", true)
      .Case("arith.ori", true)
      .Case("arith.remsi", true)
      .Case("arith.remui", true)
      .Case("arith.select", true)
      .Case("arith.shli", true)
      .Case("arith.shrsi", true)
      .Case("arith.shrui", true)
      .Case("arith.sitofp", true)
      .Case("arith.subf", true)
      .Case("arith.subi", true)
      .Case("arith.trunci", true)
      .Case("arith.uitofp", true)
      .Case("arith.xori", true)
      .Case("math.absf", true)
      .Case("math.cos", true)
      .Case("math.exp", true)
      .Case("math.fma", true)
      .Case("math.log2", true)
      .Case("math.sin", true)
      .Case("math.sqrt", true)
      .Case("llvm.intr.bitreverse", true)
      .Case("dataflow.carry", true)
      .Case("dataflow.gate", true)
      .Case("dataflow.invariant", true)
      .Case("dataflow.stream", true)
      .Case("handshake.cond_br", true)
      .Case("handshake.constant", true)
      .Case("handshake.join", true)
      .Case("handshake.load", true)
      .Case("handshake.mux", true)
      .Case("handshake.store", true)
      .Default(false);
}

bool isDedicatedDataflowFunctionUnitOp(llvm::StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Case("dataflow.carry", true)
      .Case("dataflow.gate", true)
      .Case("dataflow.invariant", true)
      .Case("dataflow.stream", true)
      .Default(false);
}

} // namespace

//===----------------------------------------------------------------------===//
// Shared helper function definitions (declared in FabricOpsInternal.h)
//===----------------------------------------------------------------------===//

std::optional<unsigned>
loom::fabric::detail::getFabricScalarWidth(mlir::Type type) {
  if (auto bits = mlir::dyn_cast<loom::fabric::BitsType>(type))
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
    return loom::fabric::getConfiguredIndexBitWidth();
  if (mlir::isa<mlir::NoneType>(type))
    return 0u;
  return std::nullopt;
}

std::optional<unsigned>
loom::fabric::detail::getSpatialSwitchPayloadWidth(mlir::Type type) {
  if (auto tagged = mlir::dyn_cast<loom::fabric::TaggedType>(type))
    return getFabricScalarWidth(tagged.getValueType());
  return getFabricScalarWidth(type);
}

mlir::LogicalResult loom::fabric::detail::verifyBinaryRowTable(
    mlir::ArrayAttr tableAttr, unsigned expectedRows, unsigned expectedCols,
    mlir::Operation *op, llvm::StringRef name) {
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

mlir::Operation *
loom::fabric::detail::getDirectRegionParent(mlir::Operation *op) {
  if (!op)
    return nullptr;
  mlir::Block *block = op->getBlock();
  if (!block)
    return nullptr;
  return block->getParentOp();
}

mlir::ParseResult loom::fabric::detail::parseOptionalOperandListInParens(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &ops,
    bool &hasOperands) {
  hasOperands = succeeded(parser.parseOptionalLParen());
  if (!hasOperands)
    return success();

  if (failed(parser.parseOptionalRParen())) {
    do {
      mlir::OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand))
        return failure();
      ops.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  return success();
}

mlir::ParseResult loom::fabric::detail::normalizeMemoryConfigAttrs(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
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

void loom::fabric::detail::printNamedAttrsWithAliases(
    mlir::OpAsmPrinter &p, mlir::Operation *op,
    mlir::ArrayRef<mlir::StringRef> excludes) {
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

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

ParseResult loom::fabric::ModuleOp::parse(OpAsmParser &parser,
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

void loom::fabric::ModuleOp::print(OpAsmPrinter &p) {
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

LogicalResult loom::fabric::ModuleOp::verify() {
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
  if (!hasDirectRegionParentOfType<loom::fabric::ModuleOp, SpatialPEOp,
                                   TemporalPEOp>(getOperation())) {
    return emitOpError(
        "must appear directly inside fabric.module, fabric.spatial_pe, or fabric.temporal_pe");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

LogicalResult MuxOp::verify() {
  if (!hasDirectRegionParentOfType<FunctionUnitOp>(getOperation()))
    return emitOpError("must appear directly inside fabric.function_unit");

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
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  bool inlineInstantiation = succeeded(parser.parseOptionalKeyword("inputs"));

  if (inlineInstantiation) {
    result.addAttribute("inline_instantiation",
                        parser.getBuilder().getUnitAttr());
    bool hasOperands = false;
    if (parseOptionalOperandListInParens(parser, operands, hasOperands))
      return failure();
    if (!hasOperands) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected operand list after 'inputs'");
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
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    auto *body = result.addRegion();
    if (parser.parseRegion(*body))
      return failure();
    if (body->empty())
      body->emplaceBlock();
    return success();
  }

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
  if (getNumOperands() > 0) {
    p << " inputs(";
    p.printOperands(getInputs());
    p << ")";
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                       {"sym_name", "function_type",
                                        "inline_instantiation"});
    p << " : " << funcType;
    p << " ";
    p.printRegion(getBody(), false);
    return;
  }

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
  return verifyModuleLevelComponentPlacement(*this);
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
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (succeeded(parser.parseOptionalLParen())) {
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

    auto funcType =
        FunctionType::get(parser.getContext(), argTypes, resultTypes);
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

  if (parseTemporalPEHwParams(parser, result))
    return failure();
  if (failed(parser.parseOptionalKeyword("inputs"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '(' for a temporal_pe definition or "
                            "'inputs' for inline instantiation");
  }

  result.addAttribute("inline_instantiation",
                      parser.getBuilder().getUnitAttr());
  bool hasOperands = false;
  if (parseOptionalOperandListInParens(parser, operands, hasOperands))
    return failure();
  if (!hasOperands) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected operand list after 'inputs'");
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
  if (parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  if (body->empty())
    body->emplaceBlock();
  return success();
}

void TemporalPEOp::print(OpAsmPrinter &p) {
  if (auto name = getSymName())
    p << " @" << *name;

  auto funcType = getFunctionType();

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

  SmallVector<StringRef> excludes = {
      "sym_name", "function_type", "num_register", "num_instruction",
      "reg_fifo_depth", "enable_share_operand_buffer", "operand_buffer_size",
      "inline_instantiation"};
  if (getNumOperands() > 0) {
    p << " inputs(";
    p.printOperands(getInputs());
    p << ")";
    p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), excludes);
    p << " : " << funcType;
    p << " ";
    p.printRegion(getBody(), false);
    return;
  }

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

  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), excludes);
  p << " ";
  p.printRegion(getBody(), false);
}

LogicalResult TemporalPEOp::verify() {
  if (failed(verifyModuleLevelComponentPlacement(*this)))
    return failure();

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
  if (!hasDirectRegionParentOfType<mlir::ModuleOp, loom::fabric::ModuleOp,
                                   SpatialPEOp, TemporalPEOp>(getOperation())) {
    return emitOpError(
        "must appear directly inside the top-level module, fabric.module, fabric.spatial_pe, or fabric.temporal_pe");
  }

  auto latencyAttr = getLatencyAttr();
  auto intervalAttr = getIntervalAttr();
  if (!latencyAttr || !intervalAttr) {
    return emitOpError(
        "must specify both latency and interval hardware parameters");
  }

  std::int64_t latency = latencyAttr.getInt();
  std::int64_t interval = intervalAttr.getInt();
  if (latency < -1) {
    return emitOpError("latency must be -1 or at least 0");
  }
  if (interval < -1 || interval == 0) {
    return emitOpError("interval must be -1 or at least 1");
  }

  auto &body = getBody().front();
  auto yieldOp = mlir::dyn_cast<loom::fabric::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return emitOpError("expected fabric.yield terminator");

  auto funcType = getFunctionType();
  for (mlir::Type inputType : funcType.getInputs()) {
    if (!isNativeFunctionUnitType(inputType)) {
      return emitOpError()
             << "function_unit inputs must use native semantic types, but got '"
             << inputType << "'";
    }
  }
  for (mlir::Type resultType : funcType.getResults()) {
    if (!isNativeFunctionUnitType(resultType)) {
      return emitOpError()
             << "function_unit results must use native semantic types, but got '"
             << resultType << "'";
    }
  }
  if (yieldOp.getNumOperands() != funcType.getNumResults()) {
    return emitOpError("yield operand count mismatch with function_unit results");
  }
  for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {
    mlir::Type yieldedType = yieldOp.getOperand(idx).getType();
    mlir::Type resultType = funcType.getResult(idx);
    if (yieldedType != resultType) {
      return emitOpError() << "yield operand " << idx
                           << " type mismatch with function_unit result: got '"
                           << yieldedType << "', expected '" << resultType
                           << "'";
    }
  }

  for (mlir::Value operand : yieldOp.getOperands()) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      (void)blockArg;
      return emitOpError(
          "passthrough function_unit is illegal; fabric.yield operands must not be direct block arguments");
    }
  }

  unsigned nonTerminatorCount = 0;
  unsigned dedicatedDataflowCount = 0;
  llvm::StringRef dedicatedDataflowOpName;
  for (mlir::Operation &bodyOp : getBody().front().getOperations()) {
    if (mlir::isa<loom::fabric::YieldOp>(bodyOp))
      continue;

    ++nonTerminatorCount;
    llvm::StringRef opName = bodyOp.getName().getStringRef();
    if (!isAllowedFunctionUnitBodyOp(opName)) {
      return emitOpError() << "operation '" << opName
                           << "' is not allowed inside fabric.function_unit";
    }

    for (mlir::Type operandType : bodyOp.getOperandTypes()) {
      if (!isNativeFunctionUnitType(operandType)) {
        return emitOpError()
               << "operation '" << opName
               << "' inside fabric.function_unit uses illegal operand type '"
               << operandType << "'";
      }
    }
    for (mlir::Type resultType : bodyOp.getResultTypes()) {
      if (!isNativeFunctionUnitType(resultType)) {
        return emitOpError()
               << "operation '" << opName
               << "' inside fabric.function_unit uses illegal result type '"
               << resultType << "'";
      }
    }

    if (opName == "handshake.join") {
      if (bodyOp.getNumOperands() == 0) {
        return emitOpError(
            "handshake.join inside fabric.function_unit requires at least one input");
      }
      if (bodyOp.getNumOperands() > 64) {
        return emitOpError(
            "handshake.join inside fabric.function_unit supports at most 64 inputs");
      }
    }

    if (isDedicatedDataflowFunctionUnitOp(opName)) {
      ++dedicatedDataflowCount;
      dedicatedDataflowOpName = opName;
    }
  }

  if (nonTerminatorCount == 0) {
    return emitOpError(
        "function_unit body must contain at least one non-terminator operation");
  }

  for (mlir::BlockArgument arg : body.getArguments()) {
    if (arg.use_empty()) {
      return emitOpError(
          "all function_unit arguments must be consumed by the body");
    }
  }

  if (dedicatedDataflowCount > 0) {
    if (nonTerminatorCount != 1 || dedicatedDataflowCount != 1) {
      return emitOpError()
             << "function_unit containing " << dedicatedDataflowOpName
             << " must contain exactly one non-terminator dataflow operation "
                "and no other body ops";
    }
    if (latency != -1 || interval != -1) {
      return emitOpError() << "function_unit containing "
                           << dedicatedDataflowOpName
                           << " must use latency = -1 and interval = -1";
    }
  } else if (latency == -1 || interval == -1) {
    return emitOpError(
        "latency = -1 and interval = -1 are reserved for dedicated dataflow function_units");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "loom/Dialect/Fabric/FabricOps.cpp.inc"
