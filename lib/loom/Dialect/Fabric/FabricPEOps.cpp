//===-- FabricPEOps.cpp - Fabric PE operation impls --------------*- C++ -*-===//
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
using dataflow_t = loom::dataflow::TaggedType;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract the value type from a potentially tagged type.
static Type getValueType(Type t) {
  if (auto tagged = dyn_cast<dataflow_t>(t))
    return tagged.getValueType();
  return t;
}

/// Parse [key = value, ...] hw params block for PE operations.
/// Populates latency, interval, output_tag, constant_value, cont_cond_sel,
/// lqDepth, sqDepth.
static ParseResult parsePEHwParams(OpAsmParser &parser,
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
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(PEOp::getLatencyAttrName(result.name), attr);
    } else if (keyword == "interval") {
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(PEOp::getIntervalAttrName(result.name), attr);
    } else if (keyword == "output_tag") {
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(PEOp::getOutputTagAttrName(result.name), attr);
    } else if (keyword == "constant_value") {
      Attribute attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(PEOp::getConstantValueAttrName(result.name), attr);
    } else if (keyword == "cont_cond_sel") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(PEOp::getContCondSelAttrName(result.name), attr);
    } else if (keyword == "lqDepth") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(PEOp::getLqDepthAttrName(result.name), attr);
    } else if (keyword == "sqDepth") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(PEOp::getSqDepthAttrName(result.name), attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in PE hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PEOp parse/print
//
// Named:  fabric.pe @name(%a: T0, %b: T1) -> (R0)
//           [latency = [...], interval = [...]] { body }
// Inline: %o = fabric.pe %i0, %i1 [hw_params]
//           : (T0, T1) -> (R0) { ^bb0(%a: VT0, %b: VT1): body }
//===----------------------------------------------------------------------===//

ParseResult PEOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  SmallVector<Type> inputTypes, outputTypes;
  SmallVector<OpAsmParser::Argument> bodyArgs;

  if (isNamed) {
    // Named form: parse (%arg0: T0, ...) -> (R0, ...) [hw_params] { body }.
    SmallVector<OpAsmParser::Argument> entryArgs;
    if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                                 /*allowType=*/true))
      return failure();

    for (auto &arg : entryArgs)
      inputTypes.push_back(arg.type);

    if (parser.parseArrow() || parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(outputTypes) || parser.parseRParen())
        return failure();
    }

    auto fnType =
        FunctionType::get(parser.getContext(), inputTypes, outputTypes);
    result.addAttribute(getFunctionTypeAttrName(result.name),
                        TypeAttr::get(fnType));

    // Parse [hw_params].
    if (parsePEHwParams(parser, result))
      return failure();

    // Derive body block argument types (strip tags).
    for (auto &arg : entryArgs)
      arg.type = getValueType(arg.type);
    bodyArgs.append(entryArgs.begin(), entryArgs.end());
  } else {
    // Inline form: parse operands, [hw_params], then : (types) -> (types).
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (parser.parseOperandList(operands))
      return failure();

    // Parse [hw_params].
    if (parsePEHwParams(parser, result))
      return failure();

    if (parser.parseColon())
      return failure();

    // Parse (input_types) -> (output_types).
    FunctionType fnType;
    if (parser.parseType(fnType))
      return failure();

    inputTypes.assign(fnType.getInputs().begin(), fnType.getInputs().end());
    outputTypes.assign(fnType.getResults().begin(), fnType.getResults().end());

    if (parser.resolveOperands(operands, inputTypes, parser.getNameLoc(),
                               result.operands))
      return failure();
    result.addTypes(outputTypes);
  }

  // Parse region body.
  auto *body = result.addRegion();
  if (bodyArgs.empty() && !isNamed) {
    // Inline form: parse region with block arguments.
    // Block args have value types (tags stripped).
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (parser.parseRegion(*body, regionArgs, /*enableNameShadowing=*/false))
      return failure();
  } else {
    if (parser.parseRegion(*body, bodyArgs))
      return failure();
  }

  if (body->empty())
    body->emplaceBlock();

  return success();
}

void PEOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  // Determine interface types.
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

  if (isNamed) {
    // Named form: print (%arg: T, ...) -> (R, ...).
    Block &entryBlock = getBody().front();
    p << "(";
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      if (i > 0)
        p << ", ";
      p.printOperand(entryBlock.getArgument(i));
      p << ": ";
      p.printType(inputTypes[i]);
    }
    p << ") -> (";
    llvm::interleaveComma(outputTypes, p,
                          [&](Type t) { p.printType(t); });
    p << ")";
  }

  // Print [hw_params].
  SmallVector<std::pair<StringRef, Attribute>> hwParams;
  if (auto a = getLatencyAttr())
    hwParams.push_back({"latency", a});
  if (auto a = getIntervalAttr())
    hwParams.push_back({"interval", a});
  if (auto a = getOutputTagAttr())
    hwParams.push_back({"output_tag", a});
  if (auto a = getConstantValueAttr())
    hwParams.push_back({"constant_value", a});
  if (getContCondSel())
    hwParams.push_back({"cont_cond_sel",
                         IntegerAttr::get(
                             IntegerType::get(getContext(), 64),
                             *getContCondSel())});
  if (getLqDepth())
    hwParams.push_back({"lqDepth",
                         IntegerAttr::get(
                             IntegerType::get(getContext(), 64),
                             *getLqDepth())});
  if (getSqDepth())
    hwParams.push_back({"sqDepth",
                         IntegerAttr::get(
                             IntegerType::get(getContext(), 64),
                             *getSqDepth())});

  if (!hwParams.empty()) {
    p << " [";
    for (unsigned i = 0; i < hwParams.size(); ++i) {
      if (i > 0)
        p << ", ";
      p << hwParams[i].first << " = ";
      p.printAttribute(hwParams[i].second);
    }
    p << "]";
  }

  if (!isNamed) {
    // Inline form: print operands and type signature.
    if (!getInputs().empty()) {
      p << " ";
      p.printOperands(getInputs());
    }
    p << " : (";
    llvm::interleaveComma(inputTypes, p, [&](Type t) { p.printType(t); });
    p << ") -> (";
    llvm::interleaveComma(outputTypes, p, [&](Type t) { p.printType(t); });
    p << ")";
  }

  // Print region body.
  p << " ";
  bool printBlockArgs = !isNamed;
  p.printRegion(getBody(), /*printEntryBlockArgs=*/printBlockArgs,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// PEOp verify
//===----------------------------------------------------------------------===//

LogicalResult PEOp::verify() {
  bool isNamed = getSymName().has_value();
  unsigned numInputs, numOutputs;

  SmallVector<Type> inputTypes, outputTypes;
  if (isNamed) {
    if (!getFunctionType())
      return emitOpError("named PE requires function_type attribute");
    auto fnType = *getFunctionType();
    numInputs = fnType.getNumInputs();
    numOutputs = fnType.getNumResults();
    inputTypes.assign(fnType.getInputs().begin(), fnType.getInputs().end());
    outputTypes.assign(fnType.getResults().begin(), fnType.getResults().end());
    if (!getInputs().empty() || !getOutputs().empty())
      return emitOpError("named PE must not have SSA operands or results");
  } else {
    numInputs = getInputs().size();
    numOutputs = getOutputs().size();
    for (auto v : getInputs())
      inputTypes.push_back(v.getType());
    for (auto v : getOutputs())
      outputTypes.push_back(v.getType());
  }

  // Check interface category: all native or all tagged.
  bool hasTagged = false, hasNative = false;
  for (Type t : inputTypes) {
    if (isa<dataflow_t>(t))
      hasTagged = true;
    else
      hasNative = true;
  }
  for (Type t : outputTypes) {
    if (isa<dataflow_t>(t))
      hasTagged = true;
    else
      hasNative = true;
  }
  if (hasTagged && hasNative)
    return emitOpError("[COMP_PE_MIXED_INTERFACE] "
                       "all ports must be either native or tagged; "
                       "mixed interface not allowed");

  // Validate latency array if present.
  if (auto lat = getLatency()) {
    if (lat->size() != 3)
      return emitOpError("latency must be a 3-element array [min, typ, max]");
  }

  // Validate interval array if present.
  if (auto intv = getInterval()) {
    if (intv->size() != 3)
      return emitOpError("interval must be a 3-element array [min, typ, max]");
  }

  // Native interface: output_tag must be absent.
  if (hasNative && getOutputTag())
    return emitOpError("[COMP_PE_OUTPUT_TAG_NATIVE] "
                       "native PE must not have output_tag");

  // Body must have at least one non-terminator.
  Block &body = getBody().front();
  bool hasOp = false;
  for (auto &op : body) {
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      hasOp = true;
      break;
    }
  }
  if (!hasOp)
    return emitOpError("[COMP_PE_EMPTY_BODY] "
        "PE body must contain at least one non-terminator operation");

  // Classify body operations for PE body constraints.
  bool hasLoadStore = false;
  bool hasConstant = false;
  bool hasDataflow = false;
  bool hasInstance = false;
  bool hasArithMath = false;
  bool hasMux = false;      // partial-consume: handshake.mux, handshake.cmerge
  bool hasFullConsume = false; // full-consume: arith.*, math.*, arith.cmpi etc
  unsigned nonTermCount = 0;
  unsigned instanceCount = 0;
  unsigned dataflowCount = 0;

  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    nonTermCount++;

    StringRef dialectName = op.getDialect()
        ? op.getDialect()->getNamespace() : "";
    StringRef opName = op.getName().getStringRef();

    if (opName == "handshake.load" || opName == "handshake.store") {
      hasLoadStore = true;
    } else if (opName == "handshake.constant") {
      hasConstant = true;
    } else if (opName == "handshake.mux" || opName == "handshake.cmerge") {
      hasMux = true;
    } else if (dialectName == "arith" || dialectName == "math") {
      hasArithMath = true;
      hasFullConsume = true;
    }

    if (dialectName == "dataflow") {
      hasDataflow = true;
      dataflowCount++;
    }

    if (isa<InstanceOp>(&op))
      instanceCount++;

    if (isa<InstanceOp>(&op))
      hasInstance = true;
  }

  // COMP_PE_INSTANCE_ONLY_BODY: exactly one non-terminator and it's instance.
  if (nonTermCount == 1 && instanceCount == 1)
    return emitOpError("[COMP_PE_INSTANCE_ONLY_BODY] "
                       "PE body must not contain only a single "
                       "fabric.instance with no other operations");

  // COMP_PE_LOADSTORE_BODY: load/store PE body must contain exactly one
  // handshake.load or handshake.store and no other non-terminator ops.
  if (hasLoadStore) {
    unsigned lsCount = 0;
    for (auto &op : body) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      StringRef opName = op.getName().getStringRef();
      if (opName == "handshake.load" || opName == "handshake.store")
        lsCount++;
    }
    if (lsCount != 1 || nonTermCount != lsCount)
      return emitOpError("[COMP_PE_LOADSTORE_BODY] "
                         "load/store PE must contain exactly one "
                         "handshake.load or handshake.store; found ")
             << lsCount << " load/store ops and " << nonTermCount
             << " non-terminator ops";
  }

  // COMP_PE_LOADSTORE_TAG_MODE: TagOverwrite vs TagTransparent exclusivity.
  if (hasLoadStore) {
    bool hasOutputTag = getOutputTag().has_value();
    bool hasLqSq = getLqDepth().has_value() || getSqDepth().has_value();
    if (hasOutputTag && hasLqSq)
      return emitOpError("[COMP_PE_LOADSTORE_TAG_MODE] "
          "load/store PE cannot have both output_tag and lqDepth/sqDepth");
    if (hasLqSq && !hasTagged)
      return emitOpError("[COMP_PE_LOADSTORE_TAG_MODE] "
          "TagTransparent load/store PE requires all tagged ports");
    if (hasTagged && !hasOutputTag && !hasLqSq)
      return emitOpError("[COMP_PE_LOADSTORE_TAG_MODE] "
          "tagged load/store PE requires output_tag or lqDepth/sqDepth");
  }

  // COMP_PE_LOADSTORE_TAG_WIDTH: tag widths must match across all ports.
  if (hasLoadStore && hasTagged) {
    unsigned firstTagWidth = 0;
    bool tagWidthSet = false;
    auto checkTagWidth = [&](Type t) -> LogicalResult {
      if (auto tagged = dyn_cast<dataflow_t>(t)) {
        unsigned tw = tagged.getTagType().getWidth();
        if (!tagWidthSet) {
          firstTagWidth = tw;
          tagWidthSet = true;
        } else if (tw != firstTagWidth) {
          return emitOpError("[COMP_PE_LOADSTORE_TAG_WIDTH] "
              "tag widths must match across all ports; got ")
                 << firstTagWidth << " and " << tw;
        }
      }
      return success();
    };
    for (Type t : inputTypes)
      if (failed(checkTagWidth(t)))
        return failure();
    for (Type t : outputTypes)
      if (failed(checkTagWidth(t)))
        return failure();
  }

  // Tagged interface: output_tag should be present.
  // Allow missing output_tag for load/store PEs (detected by body content).
  if (hasTagged && !getOutputTag() && !hasLoadStore)
    return emitOpError("[COMP_PE_OUTPUT_TAG_MISSING] "
        "tagged PE requires output_tag (unless load/store PE)");

  // COMP_PE_CONSTANT_BODY: constant PE has no other ops.
  if (hasConstant && nonTermCount > 1)
    return emitOpError("[COMP_PE_CONSTANT_BODY] "
                       "constant PE must contain only a single "
                       "handshake.constant; found ")
           << nonTermCount << " non-terminator operations";

  // COMP_PE_DATAFLOW_BODY: dataflow exclusivity.
  if (hasDataflow) {
    if (hasArithMath)
      return emitOpError("[COMP_PE_DATAFLOW_BODY] "
                         "dataflow PE body must not contain arith/math ops");
    if (dataflowCount > 1)
      return emitOpError("[COMP_PE_DATAFLOW_BODY] "
                         "dataflow PE body must contain at most one "
                         "dataflow operation; found ")
             << dataflowCount;
    if (hasInstance)
      return emitOpError("[COMP_PE_DATAFLOW_BODY] "
                         "dataflow PE body must not contain fabric.instance");
  }

  // COMP_PE_MIXED_CONSUMPTION: full-consume vs partial-consume.
  if (hasFullConsume && hasMux)
    return emitOpError("[COMP_PE_MIXED_CONSUMPTION] "
                       "PE body must not mix full-consume (arith/math) "
                       "and partial-consume (handshake.mux/cmerge) operations");

  return success();
}

//===----------------------------------------------------------------------===//
// TemporalPEOp parse/print
//
// Always named:
//   fabric.temporal_pe @name(%in: T, ...) -> (T, ...)
//     [num_register = R, num_instruction = I, reg_fifo_depth = F,
//      enable_share_operand_buffer = B, operand_buffer_size = S,
//      instruction_mem = [...]]
//     { body }
//===----------------------------------------------------------------------===//

/// Parse [hw_params] for temporal_pe.
static ParseResult parseTemporalPEHwParams(OpAsmParser &parser,
                                           OperationState &result) {
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

    if (keyword == "num_register") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(TemporalPEOp::getNumRegisterAttrName(result.name),
                          attr);
    } else if (keyword == "num_instruction") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(
          TemporalPEOp::getNumInstructionAttrName(result.name), attr);
    } else if (keyword == "reg_fifo_depth") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(TemporalPEOp::getRegFifoDepthAttrName(result.name),
                          attr);
    } else if (keyword == "enable_share_operand_buffer") {
      BoolAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(
          TemporalPEOp::getEnableShareOperandBufferAttrName(result.name), attr);
    } else if (keyword == "operand_buffer_size") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(
          TemporalPEOp::getOperandBufferSizeAttrName(result.name), attr);
    } else if (keyword == "instruction_mem") {
      ArrayAttr attr;
      if (parser.parseAttribute(attr))
        return failure();
      result.addAttribute(
          TemporalPEOp::getInstructionMemAttrName(result.name), attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in temporal_pe hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

ParseResult TemporalPEOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse @sym_name (optional).
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse argument list: (%arg0: T0, ...).
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  // Parse -> (result_types).
  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }

  auto fnType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Parse [hw_params] (required).
  if (parseTemporalPEHwParams(parser, result))
    return failure();

  // Parse region body. Block args use value types (tags stripped).
  for (auto &arg : entryArgs)
    arg.type = getValueType(arg.type);

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs))
    return failure();
  if (body->empty())
    body->emplaceBlock();

  return success();
}

void TemporalPEOp::print(OpAsmPrinter &p) {
  auto fnType = getFunctionType();

  if (getSymName())
    p << " @" << *getSymName();
  p << "(";
  Block &entryBlock = getBody().front();
  auto inputTypes = fnType.getInputs();
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    if (i > 0)
      p << ", ";
    p.printOperand(entryBlock.getArgument(i));
    p << ": ";
    p.printType(inputTypes[i]);
  }
  p << ") -> (";
  llvm::interleaveComma(fnType.getResults(), p,
                        [&](Type t) { p.printType(t); });
  p << ")";

  // Print [hw_params].
  p << " [num_register = " << getNumRegister();
  p << ", num_instruction = " << getNumInstruction();
  p << ", reg_fifo_depth = " << getRegFifoDepth();
  if (getEnableShareOperandBuffer()) {
    p << ", enable_share_operand_buffer = true";
  }
  if (auto obs = getOperandBufferSize()) {
    p << ", operand_buffer_size = " << *obs;
  }
  if (auto im = getInstructionMem()) {
    p << ", instruction_mem = ";
    p.printAttribute(*im);
  }
  p << "]";

  // Print region body (block args not printed since they differ from interface).
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// TemporalPEOp verify
//===----------------------------------------------------------------------===//

LogicalResult TemporalPEOp::verify() {
  auto fnType = getFunctionType();

  // All ports must be !dataflow.tagged.
  for (Type t : fnType.getInputs()) {
    if (!isa<dataflow_t>(t))
      return emitOpError("all ports must be !dataflow.tagged; got ") << t;
  }
  for (Type t : fnType.getResults()) {
    if (!isa<dataflow_t>(t))
      return emitOpError("all ports must be !dataflow.tagged; got ") << t;
  }

  // All ports must use the same tagged type.
  SmallVector<Type> allTypes;
  allTypes.append(fnType.getInputs().begin(), fnType.getInputs().end());
  allTypes.append(fnType.getResults().begin(), fnType.getResults().end());
  if (!allTypes.empty()) {
    Type first = allTypes.front();
    for (Type t : allTypes) {
      if (t != first)
        return emitOpError("[COMP_TEMPORAL_PE_TAG_WIDTH] "
                           "all ports must use the same tagged type; got ")
               << first << " and " << t;
    }
  }

  // num_instruction must be >= 1.
  if (getNumInstruction() < 1)
    return emitOpError("[COMP_TEMPORAL_PE_NUM_INSTRUCTION] "
                       "num_instruction must be >= 1");

  // reg_fifo_depth constraints.
  if (getNumRegister() == 0 && getRegFifoDepth() != 0)
    return emitOpError("[COMP_TEMPORAL_PE_REG_FIFO_DEPTH] "
        "reg_fifo_depth must be 0 when num_register is 0");
  if (getNumRegister() > 0 && getRegFifoDepth() < 1)
    return emitOpError("[COMP_TEMPORAL_PE_REG_FIFO_DEPTH] "
        "reg_fifo_depth must be >= 1 when num_register > 0");

  // operand_buffer_size constraints.
  if (getEnableShareOperandBuffer()) {
    if (!getOperandBufferSize())
      return emitOpError("[COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING] "
          "operand_buffer_size required when enable_share_operand_buffer");
    int64_t obs = *getOperandBufferSize();
    if (obs < 1 || obs > 8192)
      return emitOpError(
          "[COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE] "
          "operand_buffer_size must be in [1, 8192]; got ")
             << obs;
  }

  // COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE
  if (!getEnableShareOperandBuffer() && getOperandBufferSize())
    return emitOpError("[COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE] "
        "operand_buffer_size must not be set when "
        "enable_share_operand_buffer is false");

  // Body must have at least one non-terminator.
  Block &body = getBody().front();
  bool hasOp = false;
  for (auto &op : body) {
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      hasOp = true;
      break;
    }
  }
  if (!hasOp)
    return emitOpError("[COMP_TEMPORAL_PE_EMPTY_BODY] "
                       "body must contain at least one FU definition");

  // COMP_TEMPORAL_PE_TAGGED_PE: no tagged PEs inside temporal_pe.
  for (auto &op : body) {
    auto pe = dyn_cast<PEOp>(&op);
    if (!pe)
      continue;
    if (auto peFnType = pe.getFunctionType()) {
      for (Type t : peFnType->getInputs()) {
        if (isa<dataflow_t>(t))
          return emitOpError("[COMP_TEMPORAL_PE_TAGGED_PE] "
                             "temporal_pe must not contain tagged fabric.pe");
      }
      for (Type t : peFnType->getResults()) {
        if (isa<dataflow_t>(t))
          return emitOpError("[COMP_TEMPORAL_PE_TAGGED_PE] "
                             "temporal_pe must not contain tagged fabric.pe");
      }
    }
  }

  // COMP_TEMPORAL_PE_LOADSTORE: no load/store PEs inside temporal_pe.
  // Detect by body content (handshake.load/handshake.store), not attributes.
  for (auto &op : body) {
    auto pe = dyn_cast<PEOp>(&op);
    if (!pe)
      continue;
    Block &peBody = pe.getBody().front();
    for (auto &innerOp : peBody) {
      StringRef opName = innerOp.getName().getStringRef();
      if (opName == "handshake.load" || opName == "handshake.store")
        return emitOpError("[COMP_TEMPORAL_PE_LOADSTORE] "
                           "temporal_pe must not contain load/store PE");
    }
  }

  // COMP_TEMPORAL_PE_REG_DISABLED / COMP_TEMPORAL_PE_SRC_MISMATCH:
  // Validate instruction_mem entries if present.
  if (auto im = getInstructionMem()) {
    for (auto [instIdx, entry] : llvm::enumerate(*im)) {
      auto strAttr = dyn_cast<StringAttr>(entry);
      if (!strAttr)
        continue;
      StringRef inst = strAttr.getValue();
      if (inst.contains("invalid"))
        continue;

      // COMP_TEMPORAL_PE_REG_DISABLED: check for reg() usage.
      if (getNumRegister() == 0 && inst.contains("reg("))
        return emitOpError("[COMP_TEMPORAL_PE_REG_DISABLED] "
                           "instruction_mem entry ")
               << instIdx << " uses reg() when num_register is 0";

      // COMP_TEMPORAL_PE_SRC_MISMATCH: in(j) at operand position i must have
      // j == i. Extract operand portion after '= fuName(fuIdx)'.
      {
        size_t eqPos = inst.find(" = ");
        if (eqPos != StringRef::npos) {
          StringRef rhs = inst.substr(eqPos + 3).ltrim();
          // Skip FU call 'fuName(fuIdx)' to get operand list.
          size_t parenClose = rhs.find(')');
          if (parenClose != StringRef::npos) {
            StringRef operands = rhs.substr(parenClose + 1).ltrim();
            // Parse comma-separated operand tokens, track position.
            unsigned operandPos = 0;
            while (!operands.empty()) {
              // Trim leading whitespace and commas.
              operands = operands.ltrim();
              if (operands.starts_with(",")) {
                operands = operands.substr(1).ltrim();
              }
              if (operands.empty())
                break;
              // Check for in(N) pattern at current position.
              if (operands.starts_with("in(")) {
                StringRef rest = operands.substr(3);
                size_t closeParen = rest.find(')');
                if (closeParen != StringRef::npos) {
                  unsigned srcIdx;
                  if (!rest.substr(0, closeParen).getAsInteger(10, srcIdx)) {
                    if (srcIdx != operandPos)
                      return emitOpError(
                          "[COMP_TEMPORAL_PE_SRC_MISMATCH] "
                          "instruction_mem entry ")
                             << instIdx << ": in(" << srcIdx
                             << ") at operand position " << operandPos
                             << " (expected in(" << operandPos << "))";
                  }
                }
              }
              // Advance past current token (up to next ',' or end).
              size_t nextComma = operands.find(',');
              if (nextComma == StringRef::npos)
                break;
              operands = operands.substr(nextComma + 1);
              operandPos++;
            }
          }
        }
      }
    }
  }

  return success();
}
