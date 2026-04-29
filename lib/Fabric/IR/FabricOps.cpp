#include "Fabric/IR/FabricOps.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace fabric;

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.cpp.inc"

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
    return emitOpError("requires at least 2 inputs, got ")
           << operands.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError(
        "software parameters must be all set or all unset "
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
      return emitOpError("'sel' (")
             << sel << ") must be in [0, " << n << ")";
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
    return emitOpError("requires at least 2 outputs, got ")
           << outputs.size();

  auto selAttr = getSelAttr();
  auto discardAttr = getDiscardAttr();
  auto disconnectAttr = getDisconnectAttr();

  unsigned present = (selAttr ? 1u : 0u) + (discardAttr ? 1u : 0u) +
                     (disconnectAttr ? 1u : 0u);
  if (present != 0 && present != 3)
    return emitOpError(
        "software parameters must be all set or all unset "
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
      return emitOpError("'sel' (")
             << sel << ") must be in [0, " << n << ")";
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

ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();

  // Hardware params: [max_depth = N, bypassable = true|false]
  int32_t maxDepth = 0;
  bool bypassable = false;
  if (parser.parseLSquare() ||
      parser.parseKeyword("max_depth") || parser.parseEqual() ||
      parser.parseInteger(maxDepth) ||
      parser.parseComma() ||
      parser.parseKeyword("bypassable") || parser.parseEqual() ||
      parseBoolKeyword(parser, bypassable) ||
      parser.parseRSquare())
    return failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("max_depth", builder.getI32IntegerAttr(maxDepth));
  result.addAttribute("bypassable", builder.getBoolAttr(bypassable));

  // Optional software param: {bypassed = true|false}
  if (succeeded(parser.parseOptionalLBrace())) {
    bool bypassed = false;
    if (parser.parseKeyword("bypassed") || parser.parseEqual() ||
        parseBoolKeyword(parser, bypassed) ||
        parser.parseRBrace())
      return failure();
    result.addAttribute("bypassed", builder.getBoolAttr(bypassed));
  }

  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseColon() || parser.parseType(type))
    return failure();
  if (parser.resolveOperand(input, type, result.operands))
    return failure();
  (void)typeLoc;
  result.addTypes(type);
  return success();
}

void FifoOp::print(OpAsmPrinter &p) {
  p << ' ' << getInput();
  p << " [max_depth = " << getMaxDepth()
    << ", bypassable = " << (getBypassable() ? "true" : "false") << "]";
  if (auto a = getBypassedAttr())
    p << " {bypassed = " << (a.getValue() ? "true" : "false") << "}";
  p << " : " << getOutput().getType();
}

LogicalResult FifoOp::verify() {
  if (getMaxDepth() <= 0)
    return emitOpError("'max_depth' must be > 0, got ") << getMaxDepth();
  if (!getBypassable() && getBypassedAttr())
    return emitOpError(
        "'bypassed' software parameter is only allowed when 'bypassable' is "
        "true");
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
// llvm.alloca, dataflow.{load,store,graph,yield}, ub.poison, arith.constant
// are NOT in this table.
static const llvm::StringMap<OpSchema> &opSchemas() {
  static const llvm::StringMap<OpSchema> table = []() {
    llvm::StringMap<OpSchema> m;
    auto pT = PortSpec::param;
    auto pF = PortSpec::fixed;

    auto add = [&](StringRef name,
                   llvm::SmallVector<PortSpec, 4> ins,
                   llvm::SmallVector<PortSpec, 4> outs,
                   OpSchema::Kind kind = OpSchema::Fixed) {
      OpSchema s;
      s.inputs = std::move(ins);
      s.outputs = std::move(outs);
      s.kind = kind;
      m.insert({name, std::move(s)});
    };

    // --- arith integer arithmetic / logic / comparison ---
    for (StringRef n : {"arith.addi", "arith.subi", "arith.muli",
                        "arith.divsi", "arith.divui",
                        "arith.remsi", "arith.remui",
                        "arith.shli", "arith.shrsi", "arith.shrui",
                        "arith.andi", "arith.ori", "arith.xori",
                        "arith.minsi", "arith.maxsi",
                        "arith.minui", "arith.maxui"}) {
      add(n, {pT(0), pT(0)}, {pT(0)});
    }
    add("arith.cmpi", {pT(0), pT(0)}, {pF(1)});

    // --- arith floating-point arithmetic / comparison ---
    for (StringRef n : {"arith.addf", "arith.subf", "arith.mulf",
                        "arith.divf", "arith.remf",
                        "arith.minimumf", "arith.maximumf"}) {
      add(n, {pT(0), pT(0)}, {pT(0)});
    }
    add("arith.cmpf", {pT(0), pT(0)}, {pF(1)});

    // --- arith int<->fp casts (independent in/out widths) ---
    for (StringRef n : {"arith.sitofp", "arith.uitofp",
                        "arith.fptosi", "arith.fptoui"}) {
      add(n, {pT(0)}, {pT(1)});
    }

    // --- math unary ops: 1 in, 1 out, same width ---
    for (StringRef n : {"math.sin", "math.cos", "math.tan",
                        "math.sinh", "math.cosh", "math.tanh",
                        "math.exp", "math.exp2", "math.expm1",
                        "math.log", "math.log2", "math.log10", "math.log1p",
                        "math.floor", "math.ceil", "math.round",
                        "math.trunc", "math.roundeven",
                        "math.sqrt", "math.rsqrt",
                        "math.absf", "math.absi",
                        "math.erf"}) {
      add(n, {pT(0)}, {pT(0)});
    }

    // --- dataflow ops ---
    add("dataflow.stream",
        {pT(0), pT(0), pT(0)}, {pT(0), pF(1)});
    add("dataflow.constant",
        {pF(0)}, {pT(0)});
    add("dataflow.carry",
        {pF(1), pT(0), pT(0)}, {pT(0)});
    add("dataflow.invariant",
        {pF(1), pT(0)}, {pT(0)});
    add("dataflow.gate",
        {pF(1), pT(0)}, {pF(1), pT(0)});
    // Variadic ops: structural counts depend on sw_configs / fabric ports.
    add("dataflow.sync", {}, {}, OpSchema::VariadicSync);
    add("dataflow.mux", {}, {}, OpSchema::VariadicMux);
    add("dataflow.demux", {}, {}, OpSchema::VariadicDemux);

    // --- arith.select: i1 sel + 2 data of type T -> T ---
    // Strict-SSA eager-evaluation semantics. Distinct from dataflow.mux
    // (which has data-dependent gating and is variadic). Belongs to no
    // hardware-share group, so it must occupy its fabric.op alone.
    add("arith.select", {pF(1), pT(0), pT(0)}, {pT(0)});

    return m;
  }();
  return table;
}

} // namespace

namespace fabric {
bool isFabricOpSupported(llvm::StringRef name) { return opSchemas().count(name); }
} // namespace fabric

namespace {

// Returns the bit width of a fabric.bits<N> type, or std::nullopt if `t` is
// not a fabric.bits.
static std::optional<unsigned> bitsWidth(Type t) {
  if (auto bt = dyn_cast<BitsType>(t))
    return bt.getWidth();
  return std::nullopt;
}

} // namespace

LogicalResult OpOp::verify() {
  // 1. Operand and result types: all must be fabric.bits<N>. (Already enforced
  //    by the type constraint, but emit a clearer error if something slipped
  //    through, e.g. via the generic op syntax.)
  for (auto [i, t] : llvm::enumerate(getInputs().getTypes()))
    if (!bitsWidth(t))
      return emitOpError("input #") << i << " must be fabric.bits<N>";
  for (auto [i, t] : llvm::enumerate(getOutputs().getTypes()))
    if (!bitsWidth(t))
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

  // 4. Schema check (port count and per-port width).
  unsigned numIns = getInputs().size();
  unsigned numOuts = getOutputs().size();
  SmallVector<unsigned, 4> inW, outW;
  for (Type t : getInputs().getTypes())
    inW.push_back(*bitsWidth(t));
  for (Type t : getOutputs().getTypes())
    outW.push_back(*bitsWidth(t));

  // Variadic ops are always in a singleton group; opNames.size() == 1.
  const OpSchema &schemaFront = opSchemas().lookup(opNames.front());
  bool isFixedKind = schemaFront.kind == OpSchema::Fixed;

  // Convenience for sw_configs lookups used by both variadic and shared
  // hw_params <- sw_configs cross-check below.
  DictionaryAttr swDict;
  if (auto sw = getSwConfigsAttr())
    swDict = sw;

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
          return emitOpError("'sw_configs.bitmask' must be a string attribute");
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
        return emitOpError(
                   "@dataflow.mux requires exactly 1 output, got ")
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
                 << p << " width " << inW[p]
                 << " must match output width " << dataW;
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
               << opNames.front() << " has "
               << first.inputs.size() << "->" << first.outputs.size()
               << " but @" << n << " has " << s.inputs.size() << "->"
               << s.outputs.size();
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
    auto check = [&](ArrayRef<unsigned> portWidths,
                     auto extractor,
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
                  s.inputs[q].value == spec.value &&
                  q < portWidths.size()) {
                // Skip cross-checking here; handled below.
              }
            }
          }
          want = std::max(want, needed);
        }
        if (portWidths[p] != want)
          return emitOpError() << portKind << " port #" << p << " has width "
                               << portWidths[p]
                               << " but software op(s) require width " << want;
      }
      return success();
    };

    if (failed(check(inW, [](const OpSchema &s, unsigned p) { return s.inputs[p]; },
                     "input")))
      return failure();
    if (failed(check(outW, [](const OpSchema &s, unsigned p) { return s.outputs[p]; },
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
                 << n << " requires the same width on all ports tied to its "
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

  // 5. hw_params (optional): must be ArrayAttr of length 1 wrapping a
  //    DictionaryAttr.
  if (auto hp = getHwParamsAttr()) {
    if (hp.size() != 1)
      return emitOpError("'hw_params' must be a length-1 array wrapping a "
                         "dictionary, got length ")
             << hp.size();
    if (!isa<DictionaryAttr>(hp[0]))
      return emitOpError(
          "'hw_params' inner element must be a dictionary attribute");
  }

  // 6. sw_configs (optional). When op_list has > 1 entry and sw_configs is
  //    present, it must contain `op_sel` whose StringAttr value matches one
  //    of the symbols in op_list. Additionally, every key in sw_configs that
  //    also appears in hw_params must take a value listed in the
  //    corresponding hw_params allowed set.
  if (swDict) {
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
        if (selStr.getValue() == n) { found = true; break; }
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
          if (v == na.getValue()) { found = true; break; }
        if (!found)
          return emitOpError("'sw_configs[\"")
                 << key << "\"]' value " << na.getValue()
                 << " is not in the 'hw_params[\"" << key << "\"]' allowed set";
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.fu / fabric.yield
//===----------------------------------------------------------------------===//
//
// Assembly format mirrors dataflow.graph: inline (block-arg = outer : T)
// pairs in `(...)`, then `-> result-types`, then optional attributes
// keyword + region body.

ParseResult FuOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  SMLoc operandsLoc = parser.getCurrentLocation();

  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    auto parseOne = [&]() -> ParseResult {
      OpAsmParser::Argument arg;
      OpAsmParser::UnresolvedOperand op;
      Type ty;
      if (parser.parseArgument(arg) || parser.parseEqual() ||
          parser.parseOperand(op) || parser.parseColon() ||
          parser.parseType(ty))
        return failure();
      arg.type = ty;
      blockArgs.push_back(arg);
      operands.push_back(op);
      operandTypes.push_back(ty);
      return success();
    };
    if (parseOne())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseOne())
        return failure();
    if (parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(operands, operandTypes, operandsLoc,
                             result.operands))
    return failure();

  if (parser.parseArrow())
    return failure();
  SmallVector<Type, 4> resultTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    }
  } else {
    Type ty;
    if (parser.parseType(ty))
      return failure();
    resultTypes.push_back(ty);
  }
  result.addTypes(resultTypes);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs, /*enableNameShadowing=*/false))
    return failure();
  FuOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void FuOp::print(OpAsmPrinter &p) {
  p << '(';
  Block &entry = getBody().front();
  llvm::interleaveComma(
      llvm::zip(entry.getArguments(), getInputs()), p, [&](auto pair) {
        BlockArgument bb;
        Value outer;
        std::tie(bb, outer) = pair;
        p.printRegionArgument(bb, /*argAttrs=*/{}, /*omitType=*/true);
        p << " = " << outer << " : " << outer.getType();
      });
  p << ") -> ";
  auto rTypes = getResultTypes();
  if (rTypes.size() == 1) {
    p << rTypes.front();
  } else {
    p << '(';
    llvm::interleaveComma(rTypes, p);
    p << ')';
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

RegionKind FuOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

LogicalResult FuOp::verify() {
  Block &entry = getBody().front();
  if (entry.getNumArguments() != getInputs().size())
    return emitOpError("region entry block argument count (")
           << entry.getNumArguments() << ") must equal operand count ("
           << getInputs().size() << ")";
  for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
    if (arg.getType() != getInputs()[i].getType())
      return emitOpError("region entry block argument #")
             << i << " type " << arg.getType() << " must match operand type "
             << getInputs()[i].getType();
  }

  unsigned numCompute = 0;
  for (Operation &op : entry.without_terminator()) {
    if (isa<OpOp>(op)) {
      ++numCompute;
      continue;
    }
    if (isa<MuxOp, DemuxOp>(op))
      continue;
    return op.emitOpError(
        "is not allowed inside fabric.fu; only fabric.op, fabric.mux, "
        "fabric.demux are permitted (no fabric.fu nesting, no fabric.fifo)");
  }
  if (numCompute < 1)
    return emitOpError(
        "fabric.fu body requires at least one fabric.op; got 0");
  return success();
}

LogicalResult YieldOp::verify() {
  auto fu = cast<FuOp>((*this)->getParentOp());
  if (getValues().size() != fu.getOutputs().size())
    return emitOpError("yield value count (")
           << getValues().size() << ") must match parent fabric.fu result "
                                    "count ("
           << fu.getOutputs().size() << ")";
  for (auto [i, v] : llvm::enumerate(getValues())) {
    Type expected = fu.getOutputs()[i].getType();
    if (v.getType() != expected)
      return emitOpError("yield value #")
             << i << " type " << v.getType()
             << " must match parent fabric.fu result type " << expected;
  }
  return success();
}
