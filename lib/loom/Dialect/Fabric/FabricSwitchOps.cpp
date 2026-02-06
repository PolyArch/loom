//===-- FabricSwitchOps.cpp - Fabric switch operation verifiers --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace loom::fabric;

//===----------------------------------------------------------------------===//
// Shared verification helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyUniformType(Operation *op,
                                       OperandRange inputs,
                                       ResultRange outputs) {
  SmallVector<Type> allTypes;
  for (auto v : inputs)
    allTypes.push_back(v.getType());
  for (auto v : outputs)
    allTypes.push_back(v.getType());
  if (allTypes.empty())
    return success();
  Type first = allTypes.front();
  for (Type t : allTypes) {
    if (t != first)
      return op->emitOpError("all ports must have the same type; got ")
             << first << " and " << t;
  }
  return success();
}

static LogicalResult
verifyConnectivityTable(Operation *op, ArrayRef<int8_t> table,
                        unsigned numOutputs, unsigned numInputs) {
  if (table.size() != numOutputs * numInputs)
    return op->emitOpError("connectivity_table length must be ")
           << numOutputs * numInputs << "; got " << table.size();

  for (unsigned o = 0; o < numOutputs; ++o) {
    bool hasConn = false;
    for (unsigned i = 0; i < numInputs; ++i) {
      int8_t val = table[o * numInputs + i];
      if (val != 0 && val != 1)
        return op->emitOpError("connectivity_table values must be 0 or 1");
      if (val == 1)
        hasConn = true;
    }
    if (!hasConn)
      return op->emitOpError("output row ") << o << " has no connections";
  }

  for (unsigned i = 0; i < numInputs; ++i) {
    bool hasConn = false;
    for (unsigned o = 0; o < numOutputs; ++o) {
      if (table[o * numInputs + i] == 1)
        hasConn = true;
    }
    if (!hasConn)
      return op->emitOpError("input column ") << i << " has no connections";
  }

  return success();
}

/// Parse the [hw_param = val, ...] bracket section for fabric.switch.
/// Returns true if brackets were parsed. Populates connectivity_table
/// and route_table attributes.
static ParseResult parseSwitchHwParams(OpAsmParser &parser,
                                       OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success(); // No brackets, use defaults.

  // Parse key=value pairs inside [...].
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

    if (keyword == "connectivity_table") {
      DenseI8ArrayAttr attr;
      if (parser.parseCustomAttributeWithFallback(attr))
        return failure();
      result.addAttribute(
          SwitchOp::getConnectivityTableAttrName(result.name), attr);
    } else if (keyword == "route_table") {
      DenseI8ArrayAttr attr;
      if (parser.parseCustomAttributeWithFallback(attr))
        return failure();
      result.addAttribute(SwitchOp::getRouteTableAttrName(result.name), attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in switch hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// SwitchOp parse/print
//
// Named:  fabric.switch @name [hw_params] : (T,...) -> (T,...)
// Inline: %o = fabric.switch [hw_params] %i0, %i1 : T -> T, T
//===----------------------------------------------------------------------===//

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse optional [hw_params].
  if (parseSwitchHwParams(parser, result))
    return failure();

  if (isNamed) {
    // Named form: parse `: (types) -> (types)` and store as function_type.
    FunctionType fnType;
    if (parser.parseColonType(fnType))
      return failure();
    result.addAttribute(getFunctionTypeAttrName(result.name),
                        TypeAttr::get(fnType));
    // Named form has no SSA operands or results.
  } else {
    // Inline form: parse operands and type signature.
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (parser.parseOperandList(operands))
      return failure();

    if (parser.parseColon())
      return failure();

    // Parse uniform input type.
    Type inputType;
    if (parser.parseType(inputType))
      return failure();

    if (parser.parseArrow())
      return failure();

    // Parse output types (comma-separated).
    SmallVector<Type> outputTypes;
    if (parser.parseTypeList(outputTypes))
      return failure();

    // All inputs have the same type.
    SmallVector<Type> inputTypes(operands.size(), inputType);
    if (parser.resolveOperands(operands, inputTypes, parser.getNameLoc(),
                               result.operands))
      return failure();
    result.addTypes(outputTypes);
  }

  return success();
}

void SwitchOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  // Print [hw_params] if any are present.
  bool hasCt = getConnectivityTable().has_value();
  bool hasRt = getRouteTable().has_value();
  if (hasCt || hasRt) {
    p << " [";
    bool needComma = false;
    if (hasCt) {
      p << "connectivity_table = ";
      p.printAttribute(getConnectivityTableAttr());
      needComma = true;
    }
    if (hasRt) {
      if (needComma)
        p << ", ";
      p << "route_table = ";
      p.printAttribute(getRouteTableAttr());
    }
    p << "]";
  }

  if (isNamed) {
    // Named form: print `: (types) -> (types)`.
    p << " : ";
    p.printAttribute(getFunctionTypeAttr());
  } else {
    // Inline form: print operands and type signature.
    if (!getInputs().empty()) {
      p << " ";
      p.printOperands(getInputs());
    }
    p << " : ";
    if (!getInputs().empty())
      p.printType(getInputs().front().getType());
    p << " -> ";
    llvm::interleaveComma(getOutputs().getTypes(), p,
                          [&](Type t) { p.printType(t); });
  }
}

//===----------------------------------------------------------------------===//
// SwitchOp verify
//===----------------------------------------------------------------------===//

LogicalResult SwitchOp::verify() {
  bool isNamed = getSymName().has_value();
  unsigned numInputs, numOutputs;

  if (isNamed) {
    if (!getFunctionType())
      return emitOpError("named switch requires function_type attribute");
    auto fnType = *getFunctionType();
    numInputs = fnType.getNumInputs();
    numOutputs = fnType.getNumResults();
    // Named form should have no SSA operands/results.
    if (!getInputs().empty() || !getOutputs().empty())
      return emitOpError(
          "named switch must not have SSA operands or results");
  } else {
    numInputs = getInputs().size();
    numOutputs = getOutputs().size();
    if (failed(verifyUniformType(getOperation(), getInputs(), getOutputs())))
      return failure();
  }

  if (numInputs > 32)
    return emitOpError("number of inputs must be <= 32; got ") << numInputs;
  if (numOutputs > 32)
    return emitOpError("number of outputs must be <= 32; got ") << numOutputs;

  if (auto ct = getConnectivityTable()) {
    if (failed(verifyConnectivityTable(getOperation(), *ct, numOutputs,
                                       numInputs)))
      return failure();

    if (auto rt = getRouteTable()) {
      unsigned popcount = 0;
      for (int8_t v : *ct)
        if (v == 1)
          ++popcount;
      if (rt->size() != popcount)
        return emitOpError("route_table length must equal popcount of "
                           "connectivity_table (")
               << popcount << "); got " << rt->size();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TemporalSwOp parse/print
//
// Named:  fabric.temporal_sw @name
//           [num_route_table = N, connectivity_table = [...]]
//           {route_table = [...]}
//           : (tagged_types) -> (tagged_types)
//
// Inline: %o = fabric.temporal_sw
//           [num_route_table = N, connectivity_table = [...]]
//           {route_table = [...]}
//           %i0, %i1 : tagged_type -> tagged_type, tagged_type
//===----------------------------------------------------------------------===//

ParseResult TemporalSwOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try parsing @sym_name for named form.
  StringAttr symName;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(symName));
  if (isNamed)
    result.addAttribute(getSymNameAttrName(result.name), symName);

  // Parse required [hw_params].
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

    if (keyword == "num_route_table") {
      IntegerAttr attr;
      if (parser.parseAttribute(attr, parser.getBuilder().getIntegerType(64)))
        return failure();
      result.addAttribute(getNumRouteTableAttrName(result.name), attr);
    } else if (keyword == "connectivity_table") {
      DenseI8ArrayAttr attr;
      if (parser.parseCustomAttributeWithFallback(attr))
        return failure();
      result.addAttribute(getConnectivityTableAttrName(result.name), attr);
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unexpected keyword '")
             << keyword << "' in temporal_sw hardware parameters";
    }
  }

  if (parser.parseRSquare())
    return failure();

  // Parse optional {route_table = [...]}.
  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("route_table") || parser.parseEqual())
      return failure();
    ArrayAttr rtAttr;
    if (parser.parseAttribute(rtAttr))
      return failure();
    result.addAttribute(getRouteTableAttrName(result.name), rtAttr);
    if (parser.parseRBrace())
      return failure();
  }

  if (isNamed) {
    // Named form: parse `: (types) -> (types)`.
    FunctionType fnType;
    if (parser.parseColonType(fnType))
      return failure();
    result.addAttribute(getFunctionTypeAttrName(result.name),
                        TypeAttr::get(fnType));
  } else {
    // Inline form: parse operands and type signature.
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (parser.parseOperandList(operands))
      return failure();

    if (parser.parseColon())
      return failure();

    Type inputType;
    if (parser.parseType(inputType))
      return failure();

    if (parser.parseArrow())
      return failure();

    SmallVector<Type> outputTypes;
    if (parser.parseTypeList(outputTypes))
      return failure();

    SmallVector<Type> inputTypes(operands.size(), inputType);
    if (parser.resolveOperands(operands, inputTypes, parser.getNameLoc(),
                               result.operands))
      return failure();
    result.addTypes(outputTypes);
  }

  return success();
}

void TemporalSwOp::print(OpAsmPrinter &p) {
  bool isNamed = getSymName().has_value();

  if (isNamed)
    p << " @" << *getSymName();

  // Print [hw_params] (always present: num_route_table is required).
  p << " [num_route_table = " << getNumRouteTable();
  if (auto ct = getConnectivityTable()) {
    p << ", connectivity_table = ";
    p.printAttribute(getConnectivityTableAttr());
  }
  p << "]";

  // Print optional {route_table = [...]}.
  if (auto rt = getRouteTable()) {
    p << " {route_table = ";
    p.printAttribute(*rt);
    p << "}";
  }

  if (isNamed) {
    p << " : ";
    p.printAttribute(getFunctionTypeAttr());
  } else {
    if (!getInputs().empty()) {
      p << " ";
      p.printOperands(getInputs());
    }
    p << " : ";
    if (!getInputs().empty())
      p.printType(getInputs().front().getType());
    p << " -> ";
    llvm::interleaveComma(getOutputs().getTypes(), p,
                          [&](Type t) { p.printType(t); });
  }
}

//===----------------------------------------------------------------------===//
// TemporalSwOp verify
//===----------------------------------------------------------------------===//

LogicalResult TemporalSwOp::verify() {
  bool isNamed = getSymName().has_value();
  unsigned numInputs, numOutputs;

  if (isNamed) {
    if (!getFunctionType())
      return emitOpError(
          "named temporal_sw requires function_type attribute");
    auto fnType = *getFunctionType();
    numInputs = fnType.getNumInputs();
    numOutputs = fnType.getNumResults();
    // Verify all types are tagged.
    for (Type t : fnType.getInputs()) {
      if (!isa<dataflow::TaggedType>(t))
        return emitOpError("all ports must be !dataflow.tagged; got ") << t;
    }
    for (Type t : fnType.getResults()) {
      if (!isa<dataflow::TaggedType>(t))
        return emitOpError("all ports must be !dataflow.tagged; got ") << t;
    }
  } else {
    numInputs = getInputs().size();
    numOutputs = getOutputs().size();
    for (auto v : getInputs()) {
      if (!isa<dataflow::TaggedType>(v.getType()))
        return emitOpError("all ports must be !dataflow.tagged; got ")
               << v.getType();
    }
    for (auto v : getOutputs()) {
      if (!isa<dataflow::TaggedType>(v.getType()))
        return emitOpError("all ports must be !dataflow.tagged; got ")
               << v.getType();
    }
    if (failed(verifyUniformType(getOperation(), getInputs(), getOutputs())))
      return failure();
  }

  if (numInputs > 32)
    return emitOpError("number of inputs must be <= 32; got ") << numInputs;
  if (numOutputs > 32)
    return emitOpError("number of outputs must be <= 32; got ") << numOutputs;

  if (getNumRouteTable() < 1)
    return emitOpError("num_route_table must be >= 1");

  if (auto ct = getConnectivityTable()) {
    if (failed(verifyConnectivityTable(getOperation(), *ct, numOutputs,
                                       numInputs)))
      return failure();
  }

  if (auto rt = getRouteTable()) {
    if (static_cast<int64_t>(rt->size()) > getNumRouteTable())
      return emitOpError("route_table slot count must be <= num_route_table (")
             << getNumRouteTable() << "); got " << rt->size();
  }

  return success();
}
