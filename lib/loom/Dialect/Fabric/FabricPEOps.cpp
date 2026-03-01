//===-- FabricPEOps.cpp - Fabric PE operation impls --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricError.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom;
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

/// Verify sparse format rules for human-readable config entries.
/// Checks: mixed format, ascending slot order, implicit hole consistency.
static LogicalResult verifySparseFormat(Operation *op, ArrayAttr entries,
                                        StringRef prefix,
                                        StringRef mixedCode,
                                        StringRef orderCode,
                                        StringRef holeCode) {
  bool hasHex = false, hasHuman = false, hasExplicitInvalid = false;
  SmallVector<int64_t> slotIndices;

  for (auto entry : entries) {
    auto str = dyn_cast<StringAttr>(entry);
    if (!str)
      continue;
    StringRef s = str.getValue();
    if (s.starts_with("0x") || s.starts_with("0X")) {
      hasHex = true;
    } else {
      hasHuman = true;
      if (s.contains("invalid"))
        hasExplicitInvalid = true;
      // Parse slot index from "prefix[N]: ..."
      size_t lb = s.find('['), rb = s.find(']');
      if (lb != StringRef::npos && rb != StringRef::npos && rb > lb) {
        unsigned idx;
        if (!s.substr(lb + 1, rb - lb - 1).getAsInteger(10, idx))
          slotIndices.push_back(idx);
      }
    }
  }

  if (hasHex && hasHuman)
    return op->emitOpError(mixedCode)
           << " " << prefix
           << " entries mix human-readable and hex formats";

  for (unsigned i = 1; i < slotIndices.size(); ++i) {
    if (slotIndices[i] <= slotIndices[i - 1])
      return op->emitOpError(orderCode)
             << " " << prefix
             << " slot indices must be strictly ascending; got "
             << slotIndices[i - 1] << " followed by " << slotIndices[i];
  }

  if (hasExplicitInvalid) {
    for (unsigned i = 1; i < slotIndices.size(); ++i) {
      if (slotIndices[i] != slotIndices[i - 1] + 1)
        return op->emitOpError(holeCode)
               << " " << prefix << " has implicit hole between slot "
               << slotIndices[i - 1] << " and " << slotIndices[i]
               << "; all holes must be explicit when explicit invalid"
                  " entries exist";
    }
  }
  return success();
}

/// Parse [key = value, ...] hw params block for PE operations.
/// Populates latency, interval, lqDepth, sqDepth.
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

/// Parse {key = value, ...} runtime config block for PE operations.
/// Populates output_tag, constant_value, cont_cond_sel.
static ParseResult parsePERuntimeConfig(OpAsmParser &parser,
                                        OperationState &result) {
  if (failed(parser.parseOptionalLBrace()))
    return success(); // No braces.

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

    if (keyword == "output_tag") {
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
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in PE runtime configuration";
    }
  }

  if (parser.parseRBrace())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PEOp parse/print
//
// Named:  fabric.pe @name(%a: T0, %b: T1)
//           [hw_params] {runtime_config} -> (R0) { body }
// Inline: %o = fabric.pe %i0, %i1 [hw_params] {runtime_config}
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
    // Named form: (%arg0: T0, ...) [hw_params] {config} -> (R0, ...) { body }.
    SmallVector<OpAsmParser::Argument> entryArgs;
    if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                                 /*allowType=*/true))
      return failure();

    for (auto &arg : entryArgs)
      inputTypes.push_back(arg.type);

    // Parse [hw_params] {runtime_config} before -> (results).
    if (parsePEHwParams(parser, result))
      return failure();
    if (parsePERuntimeConfig(parser, result))
      return failure();

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

    // Check if interface uses bits-typed ports (inputs or outputs).
    auto isBitsValueType = [](Type t) {
      Type v = getValueType(t);
      return isa<dataflow::BitsType>(v);
    };
    bool hasBitsInterface = llvm::any_of(inputTypes, isBitsValueType) ||
                            llvm::any_of(outputTypes, isBitsValueType);
    if (!hasBitsInterface) {
      // Original path: derive body args from interface types (strip tags).
      for (auto &arg : entryArgs)
        arg.type = getValueType(arg.type);
      bodyArgs.append(entryArgs.begin(), entryArgs.end());
    }
    // When hasBitsInterface, bodyArgs stays empty so the parser reads
    // explicit ^bb0(...) from source (body types are native, not bits).
  } else {
    // Inline form: operands [hw_params] {config} : (types) -> (types) { body }.
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (parser.parseOperandList(operands))
      return failure();

    // Parse [hw_params] {runtime_config}.
    if (parsePEHwParams(parser, result))
      return failure();
    if (parsePERuntimeConfig(parser, result))
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
  if (bodyArgs.empty()) {
    // Inline form OR named-bits form: parse ^bb0 from source.
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (parser.parseRegion(*body, regionArgs, /*enableNameShadowing=*/false))
      return failure();
  } else {
    // Named non-bits form: body args pre-supplied.
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

  // Helper to print [hw_params] section.
  auto printHwParams = [&](OpAsmPrinter &p) {
    SmallVector<std::function<void()>> entries;
    if (auto a = getLatencyAttr())
      entries.push_back([&, a]() { p << "latency = "; p.printAttribute(a); });
    if (auto a = getIntervalAttr())
      entries.push_back([&, a]() { p << "interval = "; p.printAttribute(a); });
    if (getLqDepth())
      entries.push_back([&]() { p << "lqDepth = " << *getLqDepth(); });
    if (getSqDepth())
      entries.push_back([&]() { p << "sqDepth = " << *getSqDepth(); });
    if (entries.empty())
      return;
    p << " [";
    for (unsigned i = 0; i < entries.size(); ++i) {
      if (i > 0) p << ", ";
      entries[i]();
    }
    p << "]";
  };

  // Helper to print {runtime_config} section.
  auto printRuntimeConfig = [&](OpAsmPrinter &p) {
    SmallVector<std::function<void()>> entries;
    if (auto a = getOutputTagAttr())
      entries.push_back([&, a]() { p << "output_tag = "; p.printAttribute(a); });
    if (auto a = getConstantValueAttr())
      entries.push_back([&, a]() { p << "constant_value = "; p.printAttribute(a); });
    if (getContCondSel())
      entries.push_back([&]() { p << "cont_cond_sel = " << *getContCondSel(); });
    if (entries.empty())
      return;
    p << " {";
    for (unsigned i = 0; i < entries.size(); ++i) {
      if (i > 0) p << ", ";
      entries[i]();
    }
    p << "}";
  };

  if (isNamed) {
    // Named form: (%arg: T, ...) [hw] {config} -> (R, ...).
    Block &entryBlock = getBody().front();
    p << "(";
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      if (i > 0)
        p << ", ";
      p.printOperand(entryBlock.getArgument(i));
      p << ": ";
      p.printType(inputTypes[i]);
    }
    p << ")";
  }

  if (isNamed) {
    // Named form: (%args) [hw_params] {config} -> (results).
    printHwParams(p);
    printRuntimeConfig(p);
    p << " -> (";
    llvm::interleaveComma(outputTypes, p,
                          [&](Type t) { p.printType(t); });
    p << ")";
  } else {
    // Inline form: operands [hw_params] {config} : (types) -> (types).
    if (!getInputs().empty()) {
      p << " ";
      p.printOperands(getInputs());
    }
    printHwParams(p);
    printRuntimeConfig(p);
    p << " : (";
    llvm::interleaveComma(inputTypes, p, [&](Type t) { p.printType(t); });
    p << ") -> (";
    llvm::interleaveComma(outputTypes, p, [&](Type t) { p.printType(t); });
    p << ")";
  }

  // Print region body.
  // For bits-interface named PEs, always print block args since they
  // differ from interface types (body uses native types like i32, f32).
  auto isBitsValueType = [](Type t) {
    Type v = getValueType(t);
    return isa<dataflow::BitsType>(v);
  };
  bool hasBitsInterface = llvm::any_of(inputTypes, isBitsValueType) ||
                          llvm::any_of(outputTypes, isBitsValueType);
  p << " ";
  bool printBlockArgs = !isNamed || hasBitsInterface;
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

  // Check interface category: native, tagged, or bits.
  // NoneType ports (ctrl tokens) are compatible with any interface.
  // Tagged ports (tagged<X, iK>) count as tagged regardless of value type X.
  // Untagged bits<N> and untagged native (i32/f32/index) are distinct categories.
  bool hasTagged = false, hasNative = false, hasBits = false;
  bool hasBitsValue = false; // any port (tagged or not) with bits value type
  for (Type t : inputTypes) {
    if (isa<NoneType>(t)) continue;
    if (isa<dataflow_t>(t)) {
      hasTagged = true;
      if (isa<dataflow::BitsType>(getValueType(t))) hasBitsValue = true;
    } else if (isa<dataflow::BitsType>(t)) {
      hasBits = true;
      hasBitsValue = true;
    } else {
      hasNative = true;
    }
  }
  for (Type t : outputTypes) {
    if (isa<NoneType>(t)) continue;
    if (isa<dataflow_t>(t)) {
      hasTagged = true;
      if (isa<dataflow::BitsType>(getValueType(t))) hasBitsValue = true;
    } else if (isa<dataflow::BitsType>(t)) {
      hasBits = true;
      hasBitsValue = true;
    } else {
      hasNative = true;
    }
  }
  // Mixed tagged/non-tagged or mixed bits/native
  if ((hasTagged && (hasNative || hasBits)) ||
      (hasBits && hasNative))
    return emitOpError(cplErrMsg(CplError::PE_MIXED_INTERFACE,
                       "all ports must be either native, tagged, or bits; "
                       "mixed interface not allowed"));

  // For bits-interface PEs, validate body block args.
  if (hasBitsValue && isNamed) {
    Block &entryBlock = getBody().front();
    // Body arg count must match interface input count.
    if (entryBlock.getNumArguments() != numInputs)
      return emitOpError("bits-interface PE body must have ")
             << numInputs << " block arguments; got "
             << entryBlock.getNumArguments();
    // Body args must be concrete native types, not bits<N>.
    for (auto arg : entryBlock.getArguments()) {
      if (isa<dataflow::BitsType>(arg.getType()))
        return emitOpError("bits-interface PE body arguments must use "
                           "native types (i32, f32, index, ...), not bits<N>");
    }
  }

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

  // Non-tagged interface (native or bits): output_tag must be absent.
  if ((hasNative || hasBits) && getOutputTag())
    return emitOpError(cplErrMsg(CplError::PE_OUTPUT_TAG_NATIVE,
                       "native PE must not have output_tag"));

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
    return emitOpError(cplErrMsg(CplError::PE_EMPTY_BODY,
        "PE body must contain at least one non-terminator operation"));

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
    } else if (dialectName == "arith" || dialectName == "math" ||
               opName == "llvm.intr.bitreverse") {
      hasArithMath = true;
      hasFullConsume = true;
    }

    if (dialectName == "dataflow") {
      hasDataflow = true;
      dataflowCount++;
    }

    if (isa<InstanceOp>(&op)) {
      hasInstance = true;
      instanceCount++;
    }
  }

  // CPL_PE_INSTANCE_ONLY_BODY: exactly one non-terminator and it's instance.
  // Exception: inside temporal_pe, a PE wrapping a single fabric.instance is
  // the canonical way to reference an external PE as a FU type.
  if (nonTermCount == 1 && instanceCount == 1 &&
      !getOperation()->getParentOfType<TemporalPEOp>())
    return emitOpError(cplErrMsg(CplError::PE_INSTANCE_ONLY_BODY,
                       "PE body must not contain only a single "
                       "fabric.instance with no other operations"));

  // CPL_PE_LOADSTORE_BODY: load/store PE body must contain exactly one
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
      return emitOpError(cplErrMsg(CplError::PE_LOADSTORE_BODY,
                         "load/store PE must contain exactly one "
                         "handshake.load or handshake.store; found "))
             << lsCount << " load/store ops and " << nonTermCount
             << " non-terminator ops";
  }

  // CPL_PE_LOADSTORE_TAG_MODE: TagOverwrite vs TagTransparent exclusivity.
  if (hasLoadStore) {
    bool hasOutputTag = getOutputTag().has_value();
    bool hasLqSq = getLqDepth().has_value() || getSqDepth().has_value();
    if (hasOutputTag && hasLqSq)
      return emitOpError(cplErrMsg(CplError::PE_LOADSTORE_TAG_MODE,
          "load/store PE cannot have both output_tag and lqDepth/sqDepth"));
    if (hasLqSq && !hasTagged)
      return emitOpError(cplErrMsg(CplError::PE_LOADSTORE_TAG_MODE,
          "TagTransparent load/store PE requires all tagged ports"));
    if (hasTagged && !hasOutputTag && !hasLqSq)
      return emitOpError(cplErrMsg(CplError::PE_LOADSTORE_TAG_MODE,
          "tagged load/store PE requires output_tag or lqDepth/sqDepth"));
  }

  // CPL_PE_LOADSTORE_TAG_WIDTH: tag widths must match across all ports.
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
          return emitOpError(cplErrMsg(CplError::PE_LOADSTORE_TAG_WIDTH,
              "tag widths must match across all ports; got "))
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
    return emitOpError(cplErrMsg(CplError::PE_OUTPUT_TAG_MISSING,
        "tagged PE requires output_tag (unless load/store PE)"));

  // CPL_PE_CONSTANT_BODY: constant PE has no other ops.
  if (hasConstant && nonTermCount > 1)
    return emitOpError(cplErrMsg(CplError::PE_CONSTANT_BODY,
                       "constant PE must contain only a single "
                       "handshake.constant; found "))
           << nonTermCount << " non-terminator operations";

  // CPL_PE_DATAFLOW_BODY: dataflow exclusivity.
  if (hasDataflow) {
    if (hasArithMath)
      return emitOpError(cplErrMsg(CplError::PE_DATAFLOW_BODY,
                         "dataflow PE body must not contain arith/math ops"));
    if (dataflowCount > 1)
      return emitOpError(cplErrMsg(CplError::PE_DATAFLOW_BODY,
                         "dataflow PE body must contain at most one "
                         "dataflow operation; found "))
             << dataflowCount;
    if (hasInstance)
      return emitOpError(cplErrMsg(CplError::PE_DATAFLOW_BODY,
                         "dataflow PE body must not contain fabric.instance"));
  }

  // CPL_PE_MIXED_CONSUMPTION: full-consume vs partial-consume.
  if (hasFullConsume && hasMux)
    return emitOpError(cplErrMsg(CplError::PE_MIXED_CONSUMPTION,
                       "PE body must not mix full-consume (arith/math) "
                       "and partial-consume (handshake.mux/cmerge) operations"));

  return success();
}

//===----------------------------------------------------------------------===//
// TemporalPEOp parse/print
//
// Always named:
//   fabric.temporal_pe @name(%in: T, ...)
//     [num_register = R, num_instruction = I, reg_fifo_depth = F,
//      enable_share_operand_buffer = B, operand_buffer_size = S]
//     {instruction_mem = [...]}
//     -> (T, ...) { body }
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

/// Parse optional {instruction_mem = [...]} runtime config for temporal_pe.
static ParseResult parseTemporalPERuntimeConfig(OpAsmParser &parser,
                                                OperationState &result) {
  if (failed(parser.parseOptionalLBrace()))
    return success(); // No braces.

  if (parser.parseKeyword("instruction_mem") || parser.parseEqual())
    return failure();
  ArrayAttr attr;
  if (parser.parseAttribute(attr))
    return failure();
  result.addAttribute(TemporalPEOp::getInstructionMemAttrName(result.name),
                      attr);
  if (parser.parseRBrace())
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

  // Parse [hw_params] {runtime_config} before -> (results).
  if (parseTemporalPEHwParams(parser, result))
    return failure();
  if (parseTemporalPERuntimeConfig(parser, result))
    return failure();

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
  p << "]";

  // Print optional {instruction_mem = [...]}.
  if (auto im = getInstructionMem()) {
    p << " {instruction_mem = ";
    p.printAttribute(*im);
    p << "}";
  }

  // Print -> (results).
  p << " -> (";
  llvm::interleaveComma(fnType.getResults(), p,
                        [&](Type t) { p.printType(t); });
  p << ")";

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
        return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_TAG_WIDTH,
                           "all ports must use the same tagged type; got "))
               << first << " and " << t;
    }
  }

  // num_instruction must be >= 1.
  if (getNumInstruction() < 1)
    return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_NUM_INSTRUCTION,
                       "num_instruction must be >= 1"));

  // reg_fifo_depth constraints.
  if (getNumRegister() == 0 && getRegFifoDepth() != 0)
    return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_REG_FIFO_DEPTH,
        "reg_fifo_depth must be 0 when num_register is 0"));
  if (getNumRegister() > 0 && getRegFifoDepth() < 1)
    return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_REG_FIFO_DEPTH,
        "reg_fifo_depth must be >= 1 when num_register > 0"));

  // operand_buffer_size constraints.
  if (getEnableShareOperandBuffer()) {
    if (!getOperandBufferSize())
      return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING,
          "operand_buffer_size required when enable_share_operand_buffer"));
    int64_t obs = *getOperandBufferSize();
    if (obs < 1 || obs > 8192)
      return emitOpError(cplErrMsg(
          CplError::TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE,
          "operand_buffer_size must be in [1, 8192]; got "))
             << obs;
  }

  // CPL_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE
  if (!getEnableShareOperandBuffer() && getOperandBufferSize())
    return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE,
        "operand_buffer_size must not be set when "
        "enable_share_operand_buffer is false"));

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
    return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_EMPTY_BODY,
                       "body must contain at least one FU definition"));

  // CPL_TEMPORAL_PE_TAGGED_PE: no tagged PEs inside temporal_pe.
  for (auto &op : body) {
    auto pe = dyn_cast<PEOp>(&op);
    if (!pe)
      continue;
    if (auto peFnType = pe.getFunctionType()) {
      for (Type t : peFnType->getInputs()) {
        if (isa<dataflow_t>(t))
          return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_TAGGED_PE,
                             "temporal_pe must not contain tagged fabric.pe"));
      }
      for (Type t : peFnType->getResults()) {
        if (isa<dataflow_t>(t))
          return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_TAGGED_PE,
                             "temporal_pe must not contain tagged fabric.pe"));
      }
    }
  }

  // CPL_TEMPORAL_PE_LOADSTORE: no load/store PEs inside temporal_pe.
  // Detect by body content (handshake.load/handshake.store), not attributes.
  for (auto &op : body) {
    auto pe = dyn_cast<PEOp>(&op);
    if (!pe)
      continue;
    Block &peBody = pe.getBody().front();
    for (auto &innerOp : peBody) {
      StringRef opName = innerOp.getName().getStringRef();
      if (opName == "handshake.load" || opName == "handshake.store")
        return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_LOADSTORE,
                           "temporal_pe must not contain load/store PE"));
    }
  }

  // CPL_TEMPORAL_PE_DATAFLOW_INVALID: no dataflow PEs inside temporal_pe.
  // Also follows fabric.instance references to check the target PE body.
  for (auto &op : body) {
    auto pe = dyn_cast<PEOp>(&op);
    if (!pe)
      continue;
    // Collect the PE bodies to check: the inline body, plus any instance
    // targets referenced from it.
    SmallVector<Block *, 2> bodiesToCheck;
    bodiesToCheck.push_back(&pe.getBody().front());
    for (auto &innerOp : pe.getBody().front()) {
      auto inst = dyn_cast<InstanceOp>(&innerOp);
      if (!inst)
        continue;
      auto *target = SymbolTable::lookupNearestSymbolFrom(
          getOperation(), inst.getModuleAttr());
      if (auto targetPE = dyn_cast_or_null<PEOp>(target))
        bodiesToCheck.push_back(&targetPE.getBody().front());
    }
    for (Block *checkBody : bodiesToCheck) {
      for (auto &innerOp : *checkBody) {
        if (innerOp.hasTrait<OpTrait::IsTerminator>())
          continue;
        StringRef dialectName = innerOp.getDialect()
            ? innerOp.getDialect()->getNamespace() : "";
        if (dialectName == "dataflow")
          return emitOpError(cplErrMsg(
              CplError::TEMPORAL_PE_DATAFLOW_INVALID,
              "temporal_pe must not contain dataflow PE "
              "(carry/invariant/gate/stream)"));
      }
    }
  }

  // Validate instruction_mem entries if present.
  if (auto im = getInstructionMem()) {
    unsigned numInputs = fnType.getNumInputs();
    unsigned numOutputs = fnType.getNumResults();

    // CPL_TEMPORAL_PE_TOO_MANY_SLOTS
    if (static_cast<int64_t>(im->size()) > getNumInstruction())
      return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_TOO_MANY_SLOTS,
                         "instruction_mem slot count must be <= "
                         "num_instruction ("))
             << getNumInstruction() << "); got " << im->size();

    // Sparse format checks: mixed format, slot order, implicit hole.
    if (failed(verifySparseFormat(
            getOperation(), *im, "instruction_mem",
            cplErrCode(CplError::TEMPORAL_PE_MIXED_FORMAT),
            cplErrCode(CplError::TEMPORAL_PE_SLOT_ORDER),
            cplErrCode(CplError::TEMPORAL_PE_IMPLICIT_HOLE))))
      return failure();

    for (auto [instIdx, entry] : llvm::enumerate(*im)) {
      auto strAttr = dyn_cast<StringAttr>(entry);
      if (!strAttr)
        continue;
      StringRef inst = strAttr.getValue();
      // Skip hex entries.
      if (inst.starts_with("0x") || inst.starts_with("0X"))
        continue;
      if (inst.contains("invalid"))
        continue;

      // CPL_TEMPORAL_PE_REG_DISABLED: check for reg() usage.
      if (getNumRegister() == 0 && inst.contains("reg("))
        return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_REG_DISABLED,
                           "instruction_mem entry "))
               << instIdx << " uses reg() when num_register is 0";

      // Parse dest and src regions from human-readable format:
      //   inst[N]: when(tag=T) DESTS = fuName(opcode) SRCS
      size_t eqPos = inst.find(" = ");
      if (eqPos == StringRef::npos)
        continue;

      // Extract DESTS: text between ') ' (after when(...)) and ' = '
      StringRef beforeEq = inst.substr(0, eqPos);
      size_t whenClose = beforeEq.find(") ");
      StringRef destsStr;
      if (whenClose != StringRef::npos)
        destsStr = beforeEq.substr(whenClose + 2).trim();

      // Extract SRCS: text after the FU call 'fuName(opcode) '
      StringRef rhs = inst.substr(eqPos + 3).ltrim();
      size_t parenClose = rhs.find(')');
      StringRef srcsStr;
      if (parenClose != StringRef::npos)
        srcsStr = rhs.substr(parenClose + 1).ltrim();

      // Count destinations (paren-aware comma splitting).
      if (!destsStr.empty()) {
        unsigned destCount = 1;
        int depth = 0;
        for (char c : destsStr) {
          if (c == '(') ++depth;
          else if (c == ')') --depth;
          else if (c == ',' && depth == 0) ++destCount;
        }
        // CPL_TEMPORAL_PE_DEST_COUNT
        if (destCount != numOutputs)
          return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_DEST_COUNT,
                             "instruction_mem entry "))
                 << instIdx << " has " << destCount
                 << " destination(s) but num_outputs is " << numOutputs;
      }

      // Count sources (paren-aware comma splitting).
      if (parenClose != StringRef::npos && !srcsStr.empty()) {
        unsigned srcCount = 1;
        int depth = 0;
        for (char c : srcsStr) {
          if (c == '(') ++depth;
          else if (c == ')') --depth;
          else if (c == ',' && depth == 0) ++srcCount;
        }
        // CPL_TEMPORAL_PE_SRC_COUNT
        if (srcCount != numInputs)
          return emitOpError(cplErrMsg(CplError::TEMPORAL_PE_SRC_COUNT,
                             "instruction_mem entry "))
                 << instIdx << " has " << srcCount
                 << " source(s) but num_inputs is " << numInputs;
      }

      // CPL_TEMPORAL_PE_SRC_MISMATCH: in(j) at operand position i must have
      // j == i. Extract operand portion after '= fuName(fuIdx)'.
      if (parenClose != StringRef::npos) {
        StringRef operands = srcsStr;
        unsigned operandPos = 0;
        while (!operands.empty()) {
          operands = operands.ltrim();
          if (operands.starts_with(",")) {
            operands = operands.substr(1).ltrim();
          }
          if (operands.empty())
            break;
          if (operands.starts_with("in(")) {
            StringRef rest = operands.substr(3);
            size_t cp = rest.find(')');
            if (cp != StringRef::npos) {
              unsigned srcIdx;
              if (!rest.substr(0, cp).getAsInteger(10, srcIdx)) {
                if (srcIdx != operandPos)
                  return emitOpError(cplErrMsg(
                      CplError::TEMPORAL_PE_SRC_MISMATCH,
                      "instruction_mem entry "))
                         << instIdx << ": in(" << srcIdx
                         << ") at operand position " << operandPos
                         << " (expected in(" << operandPos << "))";
              }
            }
          }
          size_t nextComma = operands.find(',');
          if (nextComma == StringRef::npos)
            break;
          operands = operands.substr(nextComma + 1);
          operandPos++;
        }
      }
    }
  }

  return success();
}
