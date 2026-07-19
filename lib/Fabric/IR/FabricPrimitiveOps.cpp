#include "Fabric/IR/FabricOps.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/StreamConfiguration.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;
using namespace fabric;

//===----------------------------------------------------------------------===//
// fabric.mux
//===----------------------------------------------------------------------===//

// Format:
//   fabric.mux %a, %b[, ...] [ {sel = N : i32, discard = B, disconnect = B} ]
//     : <fabric-type>
//
// Software parameters live in `{...}`. Fabric mux has no hardware parameters,
// so `[...]` never appears here.

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None))
    return failure();

  // Optional software-parameter dictionary in `{}`.
  if (succeeded(parser.parseOptionalLBrace())) {
    // Re-present the opening brace to the attribute-dict parser.
    // We parsed `{`, so rewind by manually parsing attributes until `}`.
    NamedAttrList attrs;
    // Parse attribute entries: name `=` attribute, comma-separated.
    if (failed(parser.parseOptionalRBrace())) {
      do {
        StringRef name;
        Attribute value;
        if (parser.parseKeyword(&name) || parser.parseEqual() ||
            parser.parseAttribute(value))
          return failure();
        attrs.append(name, value);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRBrace())
        return failure();
    }
    result.attributes.append(attrs);
  }

  Type elemType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(elemType))
    return failure();

  SmallVector<Type, 4> operandTypes(operands.size(), elemType);
  if (parser.resolveOperands(operands, operandTypes, typeLoc, result.operands))
    return failure();
  result.addTypes(elemType);
  return success();
}

void MuxOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getInputs());

  // Print software parameters in `{}`, if any is set.
  SmallVector<NamedAttribute, 3> swAttrs;
  if (auto a = getSelAttr())
    swAttrs.push_back(NamedAttribute(getSelAttrName(), a));
  if (auto a = getDiscardAttr())
    swAttrs.push_back(NamedAttribute(getDiscardAttrName(), a));
  if (auto a = getDisconnectAttr())
    swAttrs.push_back(NamedAttribute(getDisconnectAttrName(), a));
  if (!swAttrs.empty()) {
    p << " {";
    llvm::interleaveComma(swAttrs, p, [&](const NamedAttribute &na) {
      p << na.getName().getValue() << " = ";
      p.printAttribute(na.getValue());
    });
    p << "}";
  }

  p << " : " << getOutput().getType();
}

LogicalResult MuxOp::verify() {
  auto operands = getInputs();
  if (operands.size() < 2)
    return emitOpError("requires at least 2 inputs, got ") << operands.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError("software parameters must be all set or all unset "
                       "(sel, discard, disconnect)");

  if (present == 0)
    return success();

  bool discard = discardAttr.getValue();
  bool disconnect = disconnectAttr.getValue();
  int32_t sel = selAttr.getInt();
  int64_t n = static_cast<int64_t>(operands.size());

  if (discard && disconnect)
    return emitOpError("'discard' and 'disconnect' cannot both be true");

  if (disconnect) {
    if (sel != 0)
      return emitOpError("when 'disconnect' is true, 'sel' must be 0");
  } else {
    if (sel < 0 || sel >= n)
      return emitOpError("'sel' (") << sel << ") must be in [0, " << n << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.demux
//===----------------------------------------------------------------------===//
//
// Format:
//   fabric.demux %in [ {sel = N : i32, discard = B, disconnect = B} ]
//                : <fabric-type> -> N
//
// Software parameters live in `{...}`. Fabric demux has no hardware
// parameters, so `[...]` never appears. The trailing `-> N` records the
// number of output ports so the parser can recreate them all (they share
// the input's type via SameOperandsAndResultType).

ParseResult DemuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();

  if (succeeded(parser.parseOptionalLBrace())) {
    NamedAttrList attrs;
    if (failed(parser.parseOptionalRBrace())) {
      do {
        StringRef name;
        Attribute value;
        if (parser.parseKeyword(&name) || parser.parseEqual() ||
            parser.parseAttribute(value))
          return failure();
        attrs.append(name, value);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRBrace())
        return failure();
    }
    result.attributes.append(attrs);
  }

  Type elemType;
  unsigned numOuts = 0;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(elemType) ||
      parser.parseArrow() || parser.parseInteger(numOuts))
    return failure();
  if (parser.resolveOperand(input, elemType, result.operands))
    return failure();
  (void)typeLoc;
  SmallVector<Type, 4> outputTypes(numOuts, elemType);
  result.addTypes(outputTypes);
  return success();
}

void DemuxOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();

  SmallVector<NamedAttribute, 3> swAttrs;
  if (auto a = getSelAttr())
    swAttrs.push_back(NamedAttribute(getSelAttrName(), a));
  if (auto a = getDiscardAttr())
    swAttrs.push_back(NamedAttribute(getDiscardAttrName(), a));
  if (auto a = getDisconnectAttr())
    swAttrs.push_back(NamedAttribute(getDisconnectAttrName(), a));
  if (!swAttrs.empty()) {
    p << " {";
    llvm::interleaveComma(swAttrs, p, [&](const NamedAttribute &na) {
      p << na.getName().getValue() << " = ";
      p.printAttribute(na.getValue());
    });
    p << "}";
  }

  p << " : " << getInput().getType() << " -> " << getOutputs().size();
}

LogicalResult DemuxOp::verify() {
  auto outputs = getOutputs();
  if (outputs.size() < 2)
    return emitOpError("requires at least 2 outputs, got ") << outputs.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError("software parameters must be all set or all unset "
                       "(sel, discard, disconnect)");
  if (present == 0)
    return success();

  bool discard = discardAttr.getValue();
  bool disconnect = disconnectAttr.getValue();
  int32_t sel = selAttr.getInt();
  int64_t n = static_cast<int64_t>(outputs.size());

  if (discard && disconnect)
    return emitOpError("'discard' and 'disconnect' cannot both be true");

  if (disconnect) {
    if (sel != 0)
      return emitOpError("when 'disconnect' is true, 'sel' must be 0");
  } else {
    if (sel < 0 || sel >= n)
      return emitOpError("'sel' (") << sel << ") must be in [0, " << n << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.fifo
//===----------------------------------------------------------------------===//
//
// Format:
//   fabric.fifo %in [max_depth = N, bypassable = true|false]
//                   [ {bypassed = true|false} ]
//                   : <fabric-type>

static ParseResult parseBoolKeyword(OpAsmParser &parser, bool &out) {
  StringRef kw;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&kw))
    return failure();
  if (kw == "true") {
    out = true;
    return success();
  }
  if (kw == "false") {
    out = false;
    return success();
  }
  return parser.emitError(loc, "expected 'true' or 'false'");
}

// Canonical FIFO assembly:
//   fabric.fifo %src [max_depth = N, bypassable = B] [{bypassed = B}]
//       : <source-type> [to <storage-type>]
// The storage type defaults to the source type. A differing same-kind type
// uses the enclosing fabric.module connection semantics.
ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SMLoc operandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input))
    return failure();

  // Hardware params: [max_depth = N, bypassable = true|false]
  int32_t maxDepth = 0;
  bool bypassable = false;
  if (parser.parseLSquare() || parser.parseKeyword("max_depth") ||
      parser.parseEqual() || parser.parseInteger(maxDepth) ||
      parser.parseComma() || parser.parseKeyword("bypassable") ||
      parser.parseEqual() || parseBoolKeyword(parser, bypassable) ||
      parser.parseRSquare())
    return failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("max_depth", builder.getI32IntegerAttr(maxDepth));
  result.addAttribute("bypassable", builder.getBoolAttr(bypassable));

  // Optional software param: {bypassed = true|false}
  if (succeeded(parser.parseOptionalLBrace())) {
    bool bypassed = false;
    if (parser.parseKeyword("bypassed") || parser.parseEqual() ||
        parseBoolKeyword(parser, bypassed) || parser.parseRBrace())
      return failure();
    result.addAttribute("bypassed", builder.getBoolAttr(bypassed));
  }

  Type outerType;
  if (parser.parseColon() || parser.parseType(outerType))
    return failure();
  // Optional `to <inner-type>` after the FIFO's own type. When absent,
  // inner == outer.
  Type innerType = outerType;
  if (succeeded(parser.parseOptionalKeyword("to"))) {
    if (parser.parseType(innerType))
      return failure();
  }
  if (parser.resolveOperand(input, outerType, result.operands))
    return failure();
  (void)operandLoc;
  result.addTypes(innerType);
  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p << " [max_depth = " << getMaxDepth()
    << ", bypassable = " << (getBypassable() ? "true" : "false") << "]";
  if (auto a = getBypassedAttr())
    p << " {bypassed = " << (a.getValue() ? "true" : "false") << "}";
  Type outerTy = getInput().getType();
  Type innerTy = getOutput().getType();
  p << " : " << outerTy;
  if (outerTy != innerTy)
    p << " to " << innerTy;
}

LogicalResult FifoOp::verify() {
  if (getMaxDepth() <= 0)
    return emitOpError("'max_depth' must be > 0, got ") << getMaxDepth();
  if (!getBypassable() && getBypassedAttr())
    return emitOpError(
        "'bypassed' software parameter is only allowed when 'bypassable' is "
        "true");
  // Width-relaxation rule at the FIFO operand boundary. The outer SSA
  // source type may differ from the FIFO's inner type only in width, and
  // only for the same fabric kind (bits / bits_tag). memref operands
  // (not currently part of the SameOperandsAndResultType-constrained type)
  // are not legal here; the type constraint already rejects them, but the
  // explicit `to <T_inner>` clause is still rejected when the kinds
  // disagree. Emit a clear diagnostic when the kinds disagree.
  Type outerTy = getInput().getType();
  Type innerTy = getOutput().getType();
  if (outerTy != innerTy) {
    if (!haveSameFabricModulePortKind(outerTy, innerTy))
      return emitOpError(
                 "operand outer type and inner type must share the same "
                 "fabric kind (bits, bits_tag); got outer ")
             << outerTy << " and inner " << innerTy;
    if (isa<MemRefType>(outerTy))
      return emitOpError(
          "memref operands cannot use the 'to <inner-type>' clause: memref "
          "types must match exactly");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.op
//===----------------------------------------------------------------------===//

namespace {

// Logical port shape for one port of a software op:
//   - TypeParam(id): variable width tied to a per-op type-parameter id
//   - Fixed(w):      a known fixed bit width (e.g., i1 -> 1, none -> 0)
struct PortSpec {
  enum Kind { TypeParam, Fixed } kind;
  unsigned value; // type-param id, or fixed width
  static PortSpec param(unsigned id) { return {TypeParam, id}; }
  static PortSpec fixed(unsigned w) { return {Fixed, w}; }
};

struct OpSchema {
  enum Kind {
    Fixed,         // statically-known input / output port spec
    VariadicSync,  // N in / N out, paired widths, sw config: bitmask
    VariadicMux,   // 1 sel + N data inputs / 1 output (N >= 2)
    VariadicDemux, // 1 sel + 1 input / N data outputs (N >= 2)
  };
  Kind kind = Fixed;
  llvm::SmallVector<PortSpec, 4> inputs;
  llvm::SmallVector<PortSpec, 4> outputs;
};

// All ops listed below are accepted in `op_list`. Anything outside is
// rejected. The dataflow-graph allowlist's exclusions are applied here too:
// LLVM memory, dataflow.{load,store,graph,yield}, ub.poison, arith.constant
// are NOT in this table.
static const llvm::StringMap<OpSchema> &opSchemas() {
  static const llvm::StringMap<OpSchema> table = []() {
    llvm::StringMap<OpSchema> m;
    auto pT = PortSpec::param;
    auto pF = PortSpec::fixed;

    auto add = [&](StringRef name, llvm::SmallVector<PortSpec, 4> ins,
                   llvm::SmallVector<PortSpec, 4> outs,
                   OpSchema::Kind kind = OpSchema::Fixed) {
      OpSchema s;
      s.inputs = std::move(ins);
      s.outputs = std::move(outs);
      s.kind = kind;
      m.insert({name, std::move(s)});
    };

    // --- arith integer arithmetic / logic / comparison ---
    for (StringRef n :
         {"arith.addi", "arith.subi", "arith.muli", "arith.divsi",
          "arith.divui", "arith.remsi", "arith.remui", "arith.shli",
          "arith.shrsi", "arith.shrui", "arith.andi", "arith.ori", "arith.xori",
          "arith.minsi", "arith.maxsi", "arith.minui", "arith.maxui"}) {
      add(n, {pT(0), pT(0)}, {pT(0)});
    }
    add("arith.cmpi", {pT(0), pT(0)}, {pF(1)});
    add("llvm.icmp", {pT(0), pT(0)}, {pF(1)});

    // --- arith floating-point arithmetic / comparison ---
    for (StringRef n : {"arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                        "arith.remf", "arith.minimumf", "arith.maximumf"}) {
      add(n, {pT(0), pT(0)}, {pT(0)});
    }
    add("arith.cmpf", {pT(0), pT(0)}, {pF(1)});
    add("llvm.intr.fmuladd", {pT(0), pT(0), pT(0)}, {pT(0)});

    // --- arith int<->fp casts (independent in/out widths) ---
    for (StringRef n :
         {"arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui"}) {
      add(n, {pT(0)}, {pT(1)});
    }
    for (StringRef n :
         {"llvm.sitofp", "llvm.uitofp", "llvm.fptosi", "llvm.fptoui"}) {
      add(n, {pT(0)}, {pT(1)});
    }
    add("arith.extsi", {pT(0)}, {pT(1)});
    add("arith.extui", {pT(0)}, {pT(1)});
    add("arith.trunci", {pT(0)}, {pT(1)});
    add("arith.index_cast", {pT(0)}, {pT(1)});
    add("arith.index_castui", {pT(0)}, {pT(1)});
    add("llvm.trunc", {pT(0)}, {pT(1)});
    add("llvm.sext", {pT(0)}, {pT(1)});
    add("llvm.zext", {pT(0)}, {pT(1)});
    add("llvm.fneg", {pT(0)}, {pT(0)});
    add("llvm.intr.abs", {pT(0)}, {pT(0)});
    add("llvm.intr.fabs", {pT(0)}, {pT(0)});
    add("llvm.intr.umin", {pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.umax", {pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.usub.sat", {pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.smin", {pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.smax", {pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.ctlz", {pT(0)}, {pT(0)});
    add("llvm.intr.fshl", {pT(0), pT(0), pT(0)}, {pT(0)});
    add("llvm.intr.bswap", {pT(0)}, {pT(0)});
    add("llvm.arm.pkhbt", {pT(0), pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.pkhtb", {pT(0), pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.sxtab16", {pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.sxtb16", {pT(0)}, {pT(0)});
    add("llvm.arm.qadd16", {pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.sadd16", {pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.qsub8", {pT(0), pT(0)}, {pT(0)});
    add("llvm.arm.qsub16", {pT(0), pT(0)}, {pT(0)});

    // --- math unary ops: 1 in, 1 out, same width ---
    for (StringRef n :
         {"math.sin",   "math.cos",       "math.tan",  "math.sinh",
          "math.cosh",  "math.tanh",      "math.exp",  "math.exp2",
          "math.expm1", "math.log",       "math.log2", "math.log10",
          "math.log1p", "math.floor",     "math.ceil", "math.round",
          "math.trunc", "math.roundeven", "math.sqrt", "math.rsqrt",
          "math.absf",  "math.absi",      "math.erf"}) {
      add(n, {pT(0)}, {pT(0)});
    }

    // --- dataflow ops ---
    add("dataflow.stream", {pT(0), pT(0), pT(0)}, {pT(0), pF(1)});
    add("dataflow.constant", {pF(0)}, {pT(0)});
    add("dataflow.carry", {pF(1), pT(0), pT(0)}, {pT(0)});
    add("dataflow.invariant", {pF(1), pT(0)}, {pT(0)});
    add("dataflow.gate", {pF(1), pT(0)}, {pF(1), pT(0)});
    // Variadic ops: structural counts depend on sw_configs / fabric ports.
    add("dataflow.sync", {}, {}, OpSchema::VariadicSync);
    add("dataflow.mux", {}, {}, OpSchema::VariadicMux);
    add("dataflow.demux", {}, {}, OpSchema::VariadicDemux);

    // --- select: i1 sel + 2 data of type T -> T ---
    // Strict-SSA eager-evaluation semantics. Distinct from dataflow.mux
    // (which has data-dependent gating and is variadic). Each spelling belongs
    // to no hardware-share group, so it must occupy its fabric.op alone.
    add("arith.select", {pF(1), pT(0), pT(0)}, {pT(0)});
    add("llvm.select", {pF(1), pT(0), pT(0)}, {pT(0)});

    return m;
  }();
  return table;
}

} // namespace

namespace fabric {
bool isFabricOpSupported(llvm::StringRef name) {
  return opSchemas().count(name);
}
} // namespace fabric

LogicalResult OpOp::verify() {
  // 1. Operand and result types: all must be fabric.bits<N>. (Already enforced
  //    by the type constraint, but emit a clearer error if something slipped
  //    through, e.g. via the generic op syntax.)
  for (auto [i, t] : llvm::enumerate(getInputs().getTypes()))
    if (!getFabricBitsWidth(t))
      return emitOpError("input #") << i << " must be fabric.bits<N>";
  for (auto [i, t] : llvm::enumerate(getOutputs().getTypes()))
    if (!getFabricBitsWidth(t))
      return emitOpError("output #") << i << " must be fabric.bits<N>";

  // 2. op_list: non-empty array of FlatSymbolRefAttr; each entry refers to a
  //    schema-known op.
  ArrayAttr opList = getOpList();
  if (opList.empty())
    return emitOpError("'op_list' must be non-empty");
  llvm::SmallVector<StringRef, 4> opNames;
  opNames.reserve(opList.size());
  for (auto [i, attr] : llvm::enumerate(opList)) {
    auto sym = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!sym)
      return emitOpError("'op_list' entry #")
             << i << " must be a flat symbol reference";
    StringRef n = sym.getValue();
    if (!opSchemas().count(n))
      return emitOpError("'op_list' entry @")
             << n << " is not a fabric.op-supported software op";
    opNames.push_back(n);
  }

  // 3. Members of op_list must all share one hardware group (or there is just
  //    one entry).
  if (opNames.size() > 1) {
    auto firstGroup = ::loom::common::findShareGroup(opNames.front());
    if (!firstGroup)
      return emitOpError("op @")
             << opNames.front()
             << " is not in any multi-member hardware-share group; it must "
                "occupy fabric.op alone";
    for (size_t i = 1; i < opNames.size(); ++i) {
      auto g = ::loom::common::findShareGroup(opNames[i]);
      if (g != firstGroup)
        return emitOpError("ops @")
               << opNames.front() << " and @" << opNames[i]
               << " do not belong to the same hardware-share group";
    }
  }

  // 4. Identify the canonical typed-mode form before applying any legacy
  //    schema checks. A normalized mode owns its exact software type,
  //    attributes, and ordered software-to-physical port maps. The schema
  //    table remains only as a compatibility check for programmed forms that
  //    do not carry typed modes.
  ArrayAttr hwParams = getHwParamsAttr();
  FabricOpModeClassification modeClassification = classifyFabricOpModes(*this);
  if (modeClassification.kind == FabricOpModeKind::Malformed)
    return emitOpError(modeClassification.diagnostic);
  bool normalizedModes =
      modeClassification.kind == FabricOpModeKind::Normalized;
  std::string pairedLaneError;
  if (!normalizedModes && failed(preflightPairedLaneModes(
                              *this, modeClassification, pairedLaneError)))
    return emitOpError(pairedLaneError);

  DictionaryAttr swDict = getSwConfigsAttr();

  // 5. Legacy schema check (port count and per-port width). Canonical modes
  //    are checked below against their selected software operation instance.
  if (!normalizedModes) {
    unsigned numIns = getInputs().size();
    unsigned numOuts = getOutputs().size();
    SmallVector<unsigned, 4> inW, outW;
    for (Type t : getInputs().getTypes())
      inW.push_back(*getFabricBitsWidth(t));
    for (Type t : getOutputs().getTypes())
      outW.push_back(*getFabricBitsWidth(t));

    // Variadic ops are always in a singleton group; opNames.size() == 1.
    const OpSchema &schemaFront = opSchemas().lookup(opNames.front());
    bool isFixedKind = schemaFront.kind == OpSchema::Fixed;

    if (!isFixedKind) {
      // Variadic ops are singletons; ensure no group misuse already covered.
      switch (schemaFront.kind) {
      case OpSchema::Fixed:
        llvm_unreachable("handled above");
      case OpSchema::VariadicSync: {
        if (numIns != numOuts)
          return emitOpError(
                     "@dataflow.sync requires equal input/output counts, got ")
                 << numIns << " inputs and " << numOuts << " outputs";
        if (numIns < 1)
          return emitOpError("@dataflow.sync requires at least 1 port");
        for (unsigned p = 0; p < numIns; ++p)
          if (inW[p] != outW[p])
            return emitOpError("@dataflow.sync port #")
                   << p << " input width " << inW[p]
                   << " must match output width " << outW[p];
        if (swDict) {
          auto bm = swDict.get("bitmask");
          if (!bm)
            return emitOpError(
                "programmed @dataflow.sync requires sw_configs key 'bitmask'");
          auto bmStr = dyn_cast<StringAttr>(bm);
          if (!bmStr)
            return emitOpError(
                "'sw_configs.bitmask' must be a string attribute");
          StringRef s = bmStr.getValue();
          if (s.size() != numIns)
            return emitOpError("'sw_configs.bitmask' length (")
                   << s.size() << ") must equal port count (" << numIns << ")";
          for (char c : s)
            if (c != '0' && c != '1')
              return emitOpError(
                  "'sw_configs.bitmask' must contain only '0' and '1'");
        }
        break;
      }
      case OpSchema::VariadicMux: {
        if (numIns < 3)
          return emitOpError("@dataflow.mux requires at least 1 sel + 2 data "
                             "inputs, got ")
                 << numIns << " inputs";
        if (numOuts != 1)
          return emitOpError("@dataflow.mux requires exactly 1 output, got ")
                 << numOuts;
        unsigned fanIn = numIns - 1;
        unsigned wantSel = (fanIn == 2) ? 1u : loom::getIndexWidth();
        if (inW[0] != wantSel)
          return emitOpError("@dataflow.mux sel port (input #0) width ")
                 << inW[0] << " must be " << wantSel
                 << " (i1 for fan-in 2, index width " << loom::getIndexWidth()
                 << " otherwise)";
        unsigned dataW = outW[0];
        for (unsigned p = 1; p < numIns; ++p)
          if (inW[p] != dataW)
            return emitOpError("@dataflow.mux input #")
                   << p << " width " << inW[p] << " must match output width "
                   << dataW;
        break;
      }
      case OpSchema::VariadicDemux: {
        if (numIns != 2)
          return emitOpError("@dataflow.demux requires exactly 1 sel + 1 data "
                             "input, got ")
                 << numIns << " inputs";
        if (numOuts < 2)
          return emitOpError(
                     "@dataflow.demux requires at least 2 outputs, got ")
                 << numOuts;
        unsigned fanOut = numOuts;
        unsigned wantSel = (fanOut == 2) ? 1u : loom::getIndexWidth();
        if (inW[0] != wantSel)
          return emitOpError("@dataflow.demux sel port (input #0) width ")
                 << inW[0] << " must be " << wantSel
                 << " (i1 for fan-out 2, index width " << loom::getIndexWidth()
                 << " otherwise)";
        unsigned dataW = inW[1];
        for (unsigned p = 0; p < numOuts; ++p)
          if (outW[p] != dataW)
            return emitOpError("@dataflow.demux output #")
                   << p << " width " << outW[p]
                   << " must match data input width " << dataW;
        break;
      }
      }
    }

    if (isFixedKind) {
      // 4a. All members must agree on input/output count.
      const OpSchema &first = opSchemas().lookup(opNames.front());
      for (StringRef n : opNames) {
        const OpSchema &s = opSchemas().lookup(n);
        if (s.inputs.size() != first.inputs.size() ||
            s.outputs.size() != first.outputs.size())
          return emitOpError("ops in op_list must agree on input/output port "
                             "counts; @")
                 << opNames.front() << " has " << first.inputs.size() << "->"
                 << first.outputs.size() << " but @" << n << " has "
                 << s.inputs.size() << "->" << s.outputs.size();
      }
      if (numIns != first.inputs.size() || numOuts != first.outputs.size())
        return emitOpError("port count (")
               << numIns << "->" << numOuts
               << ") does not match the supported software ops ("
               << first.inputs.size() << "->" << first.outputs.size() << ")";

      // 4b. Per port: collect the required width across all members. For
      //     TypeParam ports the required width is taken from the fabric.op
      //     port itself; we then check consistency across all TypeParam ports
      //     sharing the same id within each member. For Fixed ports each
      //     member contributes its fixed width and we take the max.
      auto check = [&](ArrayRef<unsigned> portWidths, auto extractor,
                       StringRef portKind) -> LogicalResult {
        for (unsigned p = 0; p < portWidths.size(); ++p) {
          unsigned want = 0;
          for (StringRef n : opNames) {
            const OpSchema &s = opSchemas().lookup(n);
            PortSpec spec = extractor(s, p);
            unsigned needed;
            if (spec.kind == PortSpec::Fixed) {
              needed = spec.value;
            } else {
              // TypeParam: the width is whatever the fabric port has, but it
              // must agree with all other ports of the same param id within
              // the same member.
              needed = portWidths[p];
              for (unsigned q = 0; q < s.inputs.size(); ++q) {
                if (s.inputs[q].kind == PortSpec::TypeParam &&
                    s.inputs[q].value == spec.value && q < portWidths.size()) {
                  // Skip cross-checking here; handled below.
                }
              }
            }
            want = std::max(want, needed);
          }
          if (portWidths[p] != want)
            return emitOpError()
                   << portKind << " port #" << p << " has width "
                   << portWidths[p] << " but software op(s) require width "
                   << want;
        }
        return success();
      };

      if (failed(check(
              inW, [](const OpSchema &s, unsigned p) { return s.inputs[p]; },
              "input")))
        return failure();
      if (failed(check(
              outW, [](const OpSchema &s, unsigned p) { return s.outputs[p]; },
              "output")))
        return failure();

      // 4c. Within each member, all TypeParam ports with the same param id
      //     must end up at the same width given the chosen fabric port widths.
      for (StringRef n : opNames) {
        const OpSchema &s = opSchemas().lookup(n);
        llvm::SmallDenseMap<unsigned, unsigned, 4> paramWidth;
        auto checkParam = [&](PortSpec spec, unsigned w) -> LogicalResult {
          if (spec.kind != PortSpec::TypeParam)
            return success();
          auto it = paramWidth.find(spec.value);
          if (it == paramWidth.end()) {
            paramWidth[spec.value] = w;
            return success();
          }
          if (it->second != w)
            return emitOpError("op @")
                   << n
                   << " requires the same width on all ports tied to its "
                      "type parameter T"
                   << spec.value << ", got " << it->second << " and " << w;
          return success();
        };
        for (unsigned p = 0; p < s.inputs.size(); ++p)
          if (failed(checkParam(s.inputs[p], inW[p])))
            return failure();
        for (unsigned p = 0; p < s.outputs.size(); ++p)
          if (failed(checkParam(s.outputs[p], outW[p])))
            return failure();
      }
    }
  }

  if (!normalizedModes && opNames.size() == 1 &&
      opNames.front() == "dataflow.stream") {
    std::string error;
    if (failed(parseStreamConfiguration(*this, error)))
      return emitOpError(error);
  }

  // 6. hw_params owns the concrete datapath capability. Canonical mode tuples
  //    are stored directly as array entries. The historical length-one
  //    allowed-set dictionary remains accepted only as a programmed adapter.
  if (normalizedModes && failed(verifyNormalizedHardwareModes(*this)))
    return failure();

  // 7. sw_configs selects one normalized mode index. Legacy allowed-set input
  //    retains its historical per-field checks.
  if (swDict) {
    if (normalizedModes) {
      if (swDict.size() != 1 || !swDict.get("mode"))
        return emitOpError(
            "normalized hw_params requires sw_configs = {mode = N}");
      auto selected = dyn_cast<IntegerAttr>(swDict.get("mode"));
      if (!selected || selected.getValue().isNegative() ||
          selected.getValue().getActiveBits() > 32)
        return emitOpError("'sw_configs.mode' must be a non-negative i32");
      uint64_t modeIndex = selected.getValue().getZExtValue();
      if (modeIndex >= hwParams.size())
        return emitOpError("'sw_configs.mode' is out of range for hw_params");
      auto mode = cast<DictionaryAttr>(hwParams[modeIndex]);
      auto selectedOp = mode.getAs<FlatSymbolRefAttr>("op");
      if (!selectedOp || !llvm::is_contained(opNames, selectedOp.getValue()))
        return emitOpError(
            "selected hw_params mode operation is not in op_list");
      return success();
    }

    if (opNames.size() > 1) {
      auto sel = swDict.get("op_sel");
      if (!sel)
        return emitOpError(
            "'sw_configs' must contain key 'op_sel' when 'op_list' has more "
            "than one entry");
      auto selStr = dyn_cast<StringAttr>(sel);
      if (!selStr)
        return emitOpError("'sw_configs.op_sel' must be a string attribute");
      bool found = false;
      for (StringRef n : opNames)
        if (selStr.getValue() == n) {
          found = true;
          break;
        }
      if (!found)
        return emitOpError("'sw_configs.op_sel' value \"")
               << selStr.getValue()
               << "\" is not one of the symbols listed in 'op_list'";
    }

    // hw_params allowed-set check: keys that appear on both sides must have
    // the sw value in the hw value set.
    DictionaryAttr hwDict;
    if (auto hp = getHwParamsAttr())
      if (hp.size() == 1)
        hwDict = dyn_cast<DictionaryAttr>(hp[0]);
    if (hwDict) {
      for (NamedAttribute na : swDict) {
        StringRef key = na.getName().getValue();
        if (key == "op_sel")
          continue;
        auto hwVal = hwDict.get(key);
        if (!hwVal)
          continue;
        auto allowed = dyn_cast<ArrayAttr>(hwVal);
        if (!allowed)
          return emitOpError("'hw_params[\"")
                 << key
                 << "\"]' must be an array of allowed values when the same "
                    "key is selected by sw_configs";
        bool found = false;
        for (Attribute v : allowed)
          if (v == na.getValue()) {
            found = true;
            break;
          }
        if (!found)
          return emitOpError("'sw_configs[\"")
                 << key << "\"]' value " << na.getValue()
                 << " is not in the 'hw_params[\"" << key << "\"]' allowed set";
      }
    }
  }
  return success();
}
