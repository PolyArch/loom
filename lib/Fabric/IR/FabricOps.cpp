#include "Fabric/IR/FabricOps.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace mlir;
using namespace fabric;

#include "Fabric/IR/FabricEnums.cpp.inc"

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.cpp.inc"

namespace {

// True iff `t` is one of the allowed fabric.module port types
// (memref or one of the two fabric stream types).
static bool isFabricModulePortType(Type t) {
  return isa<BitsType, BitsTagType, MemRefType>(t);
}

// Returns true if both types have the same kind (both bits, both bits_tag,
// or both memref). For memref this also requires exact equality (no
// relaxation). Caller is expected to first ensure both are valid module
// port types.
static bool sameModulePortKind(Type src, Type dst) {
  if (isa<BitsType>(src))
    return isa<BitsType>(dst);
  if (isa<BitsTagType>(src))
    return isa<BitsTagType>(dst);
  if (isa<MemRefType>(src))
    return isa<MemRefType>(dst);
  return false;
}

struct SystemChannel {
  std::string port;
  std::string channel;
  std::string direction;
};

struct SystemPortProfile {
  bool awInput = false;
  bool awOutput = false;
  bool wInput = false;
  bool wOutput = false;
  bool bInput = false;
  bool bOutput = false;
  bool arInput = false;
  bool arOutput = false;
  bool rInput = false;
  bool rOutput = false;
};

static std::string endpointKey(llvm::StringRef owner, llvm::StringRef port,
                               llvm::StringRef channel) {
  return (owner + "." + port + "." + channel).str();
}

static FailureOr<SystemChannel> parseSystemChannel(Attribute attr,
                                                   Operation *owner) {
  auto stringAttr = dyn_cast<StringAttr>(attr);
  if (!stringAttr) {
    owner->emitOpError("port descriptors must be string attributes");
    return failure();
  }

  StringRef raw = stringAttr.getValue();
  auto [qualified, direction] = raw.split(':');
  auto [port, channel] = qualified.split('.');
  if (qualified.empty() || port.empty() || channel.empty() ||
      direction.empty() || qualified == raw || channel.contains('.') ||
      direction.contains(':')) {
    owner->emitOpError("port descriptor '")
        << raw
        << "' must use the form 'port.channel:input' or "
           "'port.channel:output'";
    return failure();
  }
  if (direction != "input" && direction != "output") {
    owner->emitOpError("port descriptor '")
        << raw << "' has direction '" << direction
        << "', expected 'input' or 'output'";
    return failure();
  }
  return SystemChannel{port.str(), channel.str(), direction.str()};
}

static LogicalResult verifySystemPortArray(Operation *op, ArrayAttr ports) {
  if (!ports || ports.empty())
    return op->emitOpError("requires at least one system port channel");
  llvm::StringSet<> seenChannels;
  for (Attribute attr : ports) {
    FailureOr<SystemChannel> channel = parseSystemChannel(attr, op);
    if (failed(channel))
      return failure();
    std::string key = endpointKey("", channel->port, channel->channel);
    if (!seenChannels.insert(key).second)
      return op->emitOpError("duplicates port channel '")
             << channel->port << "." << channel->channel << "'";
  }
  return success();
}

static LogicalResult
collectSystemPortProfiles(Operation *op, ArrayAttr ports,
                          llvm::StringMap<SystemPortProfile> &profiles) {
  for (Attribute attr : ports) {
    FailureOr<SystemChannel> channel = parseSystemChannel(attr, op);
    if (failed(channel))
      return failure();
    SystemPortProfile &profile = profiles[channel->port];
    bool isInput = channel->direction == "input";
    bool isOutput = channel->direction == "output";
    if (channel->channel == "aw") {
      profile.awInput |= isInput;
      profile.awOutput |= isOutput;
    } else if (channel->channel == "w") {
      profile.wInput |= isInput;
      profile.wOutput |= isOutput;
    } else if (channel->channel == "b") {
      profile.bInput |= isInput;
      profile.bOutput |= isOutput;
    } else if (channel->channel == "ar") {
      profile.arInput |= isInput;
      profile.arOutput |= isOutput;
    } else if (channel->channel == "r") {
      profile.rInput |= isInput;
      profile.rOutput |= isOutput;
    }
  }
  return success();
}

static bool hasMemorySubordinateShape(const SystemPortProfile &profile) {
  return profile.awInput && profile.wInput && profile.bOutput &&
         profile.arInput && profile.rOutput;
}

static bool hasMemoryManagerShape(const SystemPortProfile &profile) {
  return profile.awOutput && profile.wOutput && profile.bInput &&
         profile.arOutput && profile.rInput;
}

static bool isDmaControlOrDescriptorPort(StringRef port,
                                         const SystemPortProfile &profile) {
  return hasMemorySubordinateShape(profile) &&
         (port == "ctrl" || port == "control" || port == "desc" ||
          port == "descriptor");
}

static std::optional<int64_t> getPositiveI64Param(DictionaryAttr params,
                                                  StringRef name) {
  if (!params)
    return std::nullopt;
  Attribute attr = params.get(name);
  if (!attr)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return std::nullopt;
  int64_t value = intAttr.getInt();
  if (value <= 0)
    return std::nullopt;
  return value;
}

static bool isPowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

static bool isBaselineSystemNodeKind(StringRef kind) {
  return kind == "host_core" || kind == "acc_core" ||
         kind == "fixed_accelerator" || kind == "memory" || kind == "cache" ||
         kind == "dma_engine";
}

static bool isValidMemoryModel(StringRef model) {
  return model == "sequential" || model == "tso" ||
         model == "release_acquire" || model == "weak" || model == "custom";
}

static Operation *lookupSymbolUpward(Operation *from, FlatSymbolRefAttr ref) {
  Operation *cursor = from;
  while (cursor) {
    Operation *symbolTable = SymbolTable::getNearestSymbolTable(cursor);
    if (!symbolTable)
      break;
    if (Operation *target = SymbolTable::lookupSymbolIn(symbolTable, ref))
      return target;
    cursor = symbolTable->getParentOp();
  }
  return nullptr;
}

} // namespace

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

// FIFO assembly form (with optional `to <inner-type>`):
//   fabric.fifo %src [to <inner-type>] [max_depth = N, bypassable = B]
//                    [{bypassed = B}] : <type>
//
// The trailing `: <type>` is the FIFO's own type (= inner type). The optional
// `to <inner-type>` after the source operand allows the SSA source value to
// have a different (same-kind) fabric type than the FIFO's internal type;
// at the FIFO boundary low-bit alignment with zero-fill applies (handled by
// the enclosing fabric.module connection-point rule).
ParseResult FifoOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SMLoc operandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input))
    return failure();

  // Optional `to <inner-type>` clause. When present, we record the inner
  // type and parse the outer type later (after the trailing `: <type>` is
  // wired to the inner side). Reuse a single optional approach: if `to`
  // appears here, the trailing `: <type>` describes the FIFO's inner type;
  // the SSA source needs an explicit outer type as well, so the form
  // becomes `%src : <outer> to <inner>`. Use that ordering instead, like
  // fabric.fu does, to keep the syntax aligned across the dialect.
  //
  // Concretely the form is one of:
  //   %src : <T>                           // outer == inner == T
  //   %src : <T_outer> to <T_inner>        // explicit relaxation
  // The trailing `[max_depth ...]` and `: <type>` of the legacy form are
  // replaced by inline operand types; the FIFO's own type matches the inner
  // type and is also printed in the trailing `: <type>` slot for backward
  // readability and so the round-trip is unambiguous.

  // Re-design: keep the legacy trailing `: <fabric-type>` as the canonical
  // FIFO type (inner) and add an optional inline `to <inner-type>` *after*
  // the bracket params and right before the trailing colon, mirroring the
  // existing print order. The new shape is:
  //   fabric.fifo %src [max_depth = N, bypassable = B] [{bypassed = B}]
  //               : <T_outer> [to <T_inner>]
  // where `<T_inner>` defaults to `<T_outer>` when the `to` clause is
  // absent. The FIFO's own type is `<T_inner>`.

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
    if (!sameModulePortKind(outerTy, innerTy))
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
        return emitOpError("@dataflow.demux requires at least 2 outputs, got ")
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
                 << p << " width " << outW[p] << " must match data input width "
                 << dataW;
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
                 << portKind << " port #" << p << " has width " << portWidths[p]
                 << " but software op(s) require width " << want;
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

//===----------------------------------------------------------------------===//
// fabric.fu / fabric.yield
//===----------------------------------------------------------------------===//
//
// Assembly format mirrors dataflow.graph: inline (block-arg = outer : T)
// pairs in `(...)`, then `-> result-types`, then optional attributes
// keyword + region body.
//
// FU boundary truncation:
// The optional `to <inner-type>` syntax lets a FU declare an inner block-arg
// type narrower than its outer operand type. When the SSA source is bits<W>
// and the inner block arg is bits<F> with F < W, hardware drops the high
// (W - F) bits at the FU boundary on each token. The outer operand type is
// always reported in the FU op's signature (visible to the enclosing PE);
// the inner type only governs the body block.

// fabric.fu has two disjoint syntactic forms by `sym_name` presence.
// Anonymous form (definition+use, original syntax):
//
//   fabric.fu (%fa = %src : <T_outer> [to <T_inner>], ...) -> (<T_res>, ...)
//             { ... fabric.yield %v : <T_res> ... }
//
// Named template form (template-only):
//
//   fabric.fu @F (<T_in0>, <T_in1>, ...) -> (<T_res0>, ...) {
//   ^bb0(%a0: <T_in0>, %a1: <T_in1>, ...):
//     ...
//     fabric.yield %v : <T_res0>
//   }
//
// In the named form the op carries zero SSA operands and zero SSA results;
// the function signature is recorded in the `function_type` attribute.
ParseResult FuOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional `@sym_name` immediately after the op keyword. When present,
  // the parser switches to the template form (no SSA operands/results).
  StringAttr nameAttr;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(
      nameAttr, ::mlir::SymbolTable::getSymbolAttrName(), result.attributes));

  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  SMLoc operandsLoc = parser.getCurrentLocation();

  if (parser.parseLParen())
    return failure();

  if (isNamed) {
    // Template signature: `(<T_in0>, <T_in1>, ...)` with optional empty list.
    SmallVector<Type, 4> argTypes;
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(argTypes) || parser.parseRParen())
        return failure();
    }
    for (Type t : argTypes) {
      OpAsmParser::Argument arg;
      arg.type = t;
      blockArgs.push_back(arg);
    }
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
    auto funcType =
        FunctionType::get(parser.getContext(), argTypes, resultTypes);
    result.addAttribute("function_type", TypeAttr::get(funcType));
  } else {
    // Anonymous form: `(%fa = %src : T [to T_inner], ...)`.
    if (failed(parser.parseOptionalRParen())) {
      auto parseOne = [&]() -> ParseResult {
        OpAsmParser::Argument arg;
        OpAsmParser::UnresolvedOperand op;
        Type outerTy;
        if (parser.parseArgument(arg) || parser.parseEqual() ||
            parser.parseOperand(op) || parser.parseColon() ||
            parser.parseType(outerTy))
          return failure();
        Type innerTy = outerTy;
        if (succeeded(parser.parseOptionalKeyword("to")))
          if (parser.parseType(innerTy))
            return failure();
        arg.type = innerTy;
        blockArgs.push_back(arg);
        operands.push_back(op);
        operandTypes.push_back(outerTy);
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
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs, /*enableNameShadowing=*/false))
    return failure();
  FuOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void FuOp::print(OpAsmPrinter &p) {
  bool isNamed = static_cast<bool>(getSymNameAttr());
  if (isNamed) {
    p << ' ';
    p.printSymbolName(getSymNameAttr().getValue());
    // Template signature: `(<T_in0>, ...) -> (<T_res0>, ...)`.
    FunctionType ft;
    if (auto fta = getFunctionTypeAttr())
      ft = cast<FunctionType>(fta.getValue());
    p << " (";
    if (ft)
      llvm::interleaveComma(ft.getInputs(), p);
    p << ") -> ";
    if (ft && ft.getNumResults() == 1) {
      p << ft.getResult(0);
    } else {
      p << '(';
      if (ft)
        llvm::interleaveComma(ft.getResults(), p);
      p << ')';
    }
  } else {
    p << '(';
    Block &entry = getBody().front();
    llvm::interleaveComma(
        llvm::zip(entry.getArguments(), getInputs()), p, [&](auto pair) {
          BlockArgument bb;
          Value outer;
          std::tie(bb, outer) = pair;
          p.printRegionArgument(bb, /*argAttrs=*/{}, /*omitType=*/true);
          p << " = " << outer << " : " << outer.getType();
          if (outer.getType() != bb.getType())
            p << " to " << bb.getType();
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
  }
  // Elide attributes already serialized inline.
  SmallVector<StringRef, 2> elided{::mlir::SymbolTable::getSymbolAttrName(),
                                   "function_type"};
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  p << ' ';
  p.printRegion(getBody(),
                /*printEntryBlockArgs=*/isNamed,
                /*printBlockTerminators=*/true);
}

RegionKind FuOp::getRegionKind(unsigned /*index*/) { return RegionKind::Graph; }

LogicalResult FuOp::verify() {
  // The op exists in two disjoint forms by `sym_name` presence:
  //   * Named form (template-only): no SSA operands, no SSA results; port
  //     signature comes from the `function_type` attribute. Body entry
  //     block arg types must equal `function_type.getInputs()`. Yield
  //     types must equal `function_type.getResults()`. Parent must be a
  //     fabric.module body (siblings host fabric.fu templates).
  //   * Anonymous form (definition + use): variadic SSA operands and SSA
  //     results; the FU input boundary supports high-bit truncation. Must
  //     live inside a fabric.pe.
  bool isNamed = static_cast<bool>(getSymNameAttr());

  if (isNamed) {
    // Cross-form rejections.
    if (!getInputs().empty())
      return emitOpError(
                 "named fabric.fu template must have zero SSA operands; got ")
             << getInputs().size();
    if (!getResultTypes().empty())
      return emitOpError(
                 "named fabric.fu template must have zero SSA results; got ")
             << getResultTypes().size();
    auto fta = getFunctionTypeAttr();
    if (!fta)
      return emitOpError(
          "named fabric.fu template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return emitOpError("'function_type' attribute must be a FunctionType");

    // Parent: a named fabric.fu lives in a fabric.module body or fabric.pe
    // body (the latter form mirrors the existing in-PE named-fu template).
    Operation *parent = (*this)->getParentOp();
    if (!isa_and_nonnull<fabric::ModuleOp, PeOp>(parent))
      return emitOpError(
          "named fabric.fu template must live inside a fabric.module or "
          "fabric.pe body");

    // Block-arg types match function_type inputs.
    Block &entry = getBody().front();
    if (entry.getNumArguments() != ft.getNumInputs())
      return emitOpError("entry block argument count (")
             << entry.getNumArguments() << ") must match declared input count ("
             << ft.getNumInputs() << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(entry.getArguments(), ft.getInputs()))) {
      BlockArgument bb;
      Type t;
      std::tie(bb, t) = pair;
      if (bb.getType() != t)
        return emitOpError("entry block argument #")
               << i << " type " << bb.getType()
               << " must equal declared input type " << t;
      if (!bitsWidth(t))
        return emitOpError("declared input #")
               << i << " must be fabric.bits<N>, got " << t;
    }
    for (auto [i, t] : llvm::enumerate(ft.getResults())) {
      if (!bitsWidth(t))
        return emitOpError("declared result #")
               << i << " must be fabric.bits<N>, got " << t;
    }

    // Body whitelist + at-least-one-fabric.op.
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

    // Yield types must equal function_type.getResults().
    auto yield = dyn_cast<YieldOp>(entry.getTerminator());
    if (!yield)
      return emitOpError(
          "named fabric.fu body must terminate with fabric.yield");
    if (yield.getValues().size() != ft.getNumResults())
      return emitOpError("yield value count (")
             << yield.getValues().size()
             << ") must match declared result count (" << ft.getNumResults()
             << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(yield.getValues(), ft.getResults()))) {
      Value v;
      Type t;
      std::tie(v, t) = pair;
      if (v.getType() != t)
        return emitOpError("yield value #")
               << i << " type " << v.getType()
               << " must equal declared result type " << t;
    }
    return success();
  }

  // Anonymous form. Reject stray function_type.
  if (getFunctionTypeAttr())
    return emitOpError(
        "anonymous fabric.fu must not carry a 'function_type' attribute");

  Operation *parent = (*this)->getParentOp();
  if (!isa_and_nonnull<PeOp>(parent))
    return emitOpError("must be inside a fabric.pe (parent must be fabric.pe)");

  Block &entry = getBody().front();
  if (entry.getNumArguments() != getInputs().size())
    return emitOpError("region entry block argument count (")
           << entry.getNumArguments() << ") must equal operand count ("
           << getInputs().size() << ")";
  for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
    Type outerTy = getInputs()[i].getType();
    Type innerTy = arg.getType();
    // FU outer ports are strict !fabric.bits<W>. Inner block-arg width
    // may be narrower (high-bit truncation), but cannot exceed outer.
    auto outerW = bitsWidth(outerTy);
    auto innerW = bitsWidth(innerTy);
    if (!outerW)
      return emitOpError("operand #")
             << i << " must be fabric.bits<N>, got " << outerTy;
    if (!innerW)
      return emitOpError("region entry block argument #")
             << i << " must be fabric.bits<N>, got " << innerTy;
    if (*outerW < *innerW)
      return emitOpError("operand #")
             << i << " bits-width " << *outerW
             << " is less than block-argument bits-width " << *innerW
             << "; the FU boundary only supports high-bit truncation "
                "(outer >= inner)";
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
    return emitOpError("fabric.fu body requires at least one fabric.op; got 0");
  return success();
}

bool FuOp::isOptionalSymbol() { return true; }

//===----------------------------------------------------------------------===//
// fabric.pe
//===----------------------------------------------------------------------===//
//
// Assembly form mirrors fabric.fu but with a mandatory schedule predicate
// in `[...]` immediately after the op keyword and no inner terminator:
//
//   fabric.pe [spatial] (%fa = %a : !fabric.bits<W>)
//                       -> !fabric.bits<W> { ... }
//
// Both spatial and temporal schedules have dedicated verifier branches.

// fabric.pe has two disjoint syntactic forms by `sym_name` presence.
// Anonymous form (definition+use, original syntax):
//
//   %r:N = fabric.pe [schedule]
//              (%pa = %a : <T_outer> [to <T_inner>], ...)
//              -> (<T_res>, ...) { ... }
//
// Named template form (template-only):
//
//   fabric.pe @S [schedule] (<T_in0>, <T_in1>, ...) -> (<T_res0>, ...) {
//   ^bb0(%a0: <T_in0>, %a1: <T_in1>, ...):
//     ...
//     fabric.yield %v0 : <T_res0>
//   }
ParseResult PeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional `@sym_name` immediately after the op keyword. When present
  // the parser switches to the template form (no SSA operands/results).
  StringAttr nameAttr;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(
      nameAttr, ::mlir::SymbolTable::getSymbolAttrName(), result.attributes));

  // Mandatory `[<schedule>]` predicate.
  StringRef scheduleKw;
  SMLoc scheduleLoc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseKeyword(&scheduleKw) ||
      parser.parseRSquare())
    return failure();
  auto sym = symbolizeSchedule(scheduleKw);
  if (!sym)
    return parser.emitError(scheduleLoc, "expected fabric pe schedule keyword "
                                         "'spatial' or 'temporal', got '")
           << scheduleKw << "'";
  result.addAttribute("schedule", ScheduleAttr::get(parser.getContext(), *sym));

  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  SMLoc operandsLoc = parser.getCurrentLocation();

  if (parser.parseLParen())
    return failure();

  if (isNamed) {
    SmallVector<Type, 4> argTypes;
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(argTypes) || parser.parseRParen())
        return failure();
    }
    // Spatial named form: PE ports are !fabric.bits<W>, and the entry
    // block args are required to match the function_type inputs. Reuse
    // the function-type inputs as the pre-declared block-arg types.
    //
    // Temporal named form: PE ports are !fabric.bits_tag<W, T>, but the
    // body sees auto-tag-stripped !fabric.bits<W'> (W' <= W). Do not
    // pre-fill blockArgs here; let parseRegion read the entry block
    // arg types from the user-written `^bb0(...)` line. The verifier
    // enforces the (W' = W) match (or rejects bits_tag inner types).
    bool isTemporal = (*sym == Schedule::Temporal);
    if (!isTemporal) {
      for (Type t : argTypes) {
        OpAsmParser::Argument arg;
        arg.type = t;
        blockArgs.push_back(arg);
      }
    }
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
    auto funcType =
        FunctionType::get(parser.getContext(), argTypes, resultTypes);
    result.addAttribute("function_type", TypeAttr::get(funcType));
  } else {
    if (failed(parser.parseOptionalRParen())) {
      // Temporal-PE boundary auto-strip: when an outer port is
      // !fabric.bits_tag<W, T> and the user does not write
      // 'to <T_inner>', the inner block-arg defaults to !fabric.bits<W>
      // (the bits-data part). When the user writes 'to bits<F>' with
      // F < W, the override path narrows further (high-bit truncation).
      // Spatial PE outer types are bits<W>; the default is unchanged.
      bool isTemporal = (*sym == Schedule::Temporal);
      auto parseOne = [&]() -> ParseResult {
        OpAsmParser::Argument arg;
        OpAsmParser::UnresolvedOperand op;
        Type outerTy;
        if (parser.parseArgument(arg) || parser.parseEqual() ||
            parser.parseOperand(op) || parser.parseColon() ||
            parser.parseType(outerTy))
          return failure();
        Type innerTy = outerTy;
        if (isTemporal) {
          if (auto tag = dyn_cast<BitsTagType>(outerTy))
            innerTy = BitsType::get(parser.getContext(), tag.getWidth());
        }
        if (succeeded(parser.parseOptionalKeyword("to")))
          if (parser.parseType(innerTy))
            return failure();
        arg.type = innerTy;
        blockArgs.push_back(arg);
        operands.push_back(op);
        operandTypes.push_back(outerTy);
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
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs, /*enableNameShadowing=*/false))
    return failure();
  return success();
}

void PeOp::print(OpAsmPrinter &p) {
  bool isNamed = static_cast<bool>(getSymNameAttr());
  if (isNamed) {
    p << ' ';
    p.printSymbolName(getSymNameAttr().getValue());
  }
  p << " [" << stringifySchedule(getSchedule()) << "]";
  if (isNamed) {
    FunctionType ft;
    if (auto fta = getFunctionTypeAttr())
      ft = cast<FunctionType>(fta.getValue());
    p << " (";
    if (ft)
      llvm::interleaveComma(ft.getInputs(), p);
    p << ") -> ";
    if (ft && ft.getNumResults() == 1) {
      p << ft.getResult(0);
    } else {
      p << '(';
      if (ft)
        llvm::interleaveComma(ft.getResults(), p);
      p << ')';
    }
  } else {
    p << " (";
    Block &entry = getBody().front();
    bool isTemporal = (getSchedule() == Schedule::Temporal);
    llvm::interleaveComma(
        llvm::zip(entry.getArguments(), getInputs()), p, [&](auto pair) {
          BlockArgument bb;
          Value outer;
          std::tie(bb, outer) = pair;
          p.printRegionArgument(bb, /*argAttrs=*/{}, /*omitType=*/true);
          p << " = " << outer << " : " << outer.getType();
          // Temporal-PE auto-strip: when outer is bits_tag<W, T> and the
          // inner block-arg is exactly bits<W>, the 'to' clause is the
          // implicit default and need not be printed. Only print 'to'
          // when inner differs from that implicit default.
          Type outerTy = outer.getType();
          Type innerTy = bb.getType();
          bool isImplicitStrip = false;
          if (isTemporal) {
            if (auto tag = dyn_cast<BitsTagType>(outerTy)) {
              if (auto bits = dyn_cast<BitsType>(innerTy))
                isImplicitStrip = (tag.getWidth() == bits.getWidth());
            }
          }
          if (outerTy != innerTy && !isImplicitStrip)
            p << " to " << innerTy;
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
  }
  // Elide attributes already serialized inline.
  SmallVector<StringRef, 3> elided{
      "schedule", ::mlir::SymbolTable::getSymbolAttrName(), "function_type"};
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  p << ' ';
  p.printRegion(getBody(),
                /*printEntryBlockArgs=*/isNamed,
                /*printBlockTerminators=*/isNamed);
}

RegionKind PeOp::getRegionKind(unsigned /*index*/) { return RegionKind::Graph; }

// Helpers for the temporal branch live in FabricPeTemporalOps.cpp.
namespace fabric {
LogicalResult verifyPeTemporal(PeOp op);
LogicalResult verifyPeSpatialNoTemporalAttrs(PeOp op);
} // namespace fabric

LogicalResult PeOp::verify() {
  // Schedule predicate dispatch.
  if (getSchedule() == Schedule::Temporal)
    return ::fabric::verifyPeTemporal(*this);

  // Spatial branch: reject any temporal-only attribute, then run the
  // spatial verifier.
  if (failed(::fabric::verifyPeSpatialNoTemporalAttrs(*this)))
    return failure();

  bool isNamed = static_cast<bool>(getSymNameAttr());
  Block &entry = getBody().front();

  // Resolve the per-form (input, output) type lists.
  SmallVector<Type, 4> declaredIns;
  SmallVector<Type, 4> declaredOuts;

  if (isNamed) {
    // Cross-form rejections.
    if (!getInputs().empty())
      return emitOpError(
                 "named fabric.pe template must have zero SSA operands; got ")
             << getInputs().size();
    if (!getResultTypes().empty())
      return emitOpError(
                 "named fabric.pe template must have zero SSA results; got ")
             << getResultTypes().size();
    auto fta = getFunctionTypeAttr();
    if (!fta)
      return emitOpError(
          "named fabric.pe template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return emitOpError("'function_type' attribute must be a FunctionType");
    declaredIns.assign(ft.getInputs().begin(), ft.getInputs().end());
    declaredOuts.assign(ft.getResults().begin(), ft.getResults().end());

    // Block-arg types must equal declaredIns.
    if (entry.getNumArguments() != declaredIns.size())
      return emitOpError("entry block argument count (")
             << entry.getNumArguments() << ") must match declared input count ("
             << declaredIns.size() << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(entry.getArguments(), declaredIns))) {
      BlockArgument bb;
      Type t;
      std::tie(bb, t) = pair;
      if (bb.getType() != t)
        return emitOpError("entry block argument #")
               << i << " type " << bb.getType()
               << " must equal declared input type " << t;
    }
  } else {
    // Anonymous form. Reject stray function_type.
    if (getFunctionTypeAttr())
      return emitOpError(
          "anonymous fabric.pe must not carry a 'function_type' attribute");

    if (entry.getNumArguments() != getInputs().size())
      return emitOpError("region entry block argument count (")
             << entry.getNumArguments() << ") must equal operand count ("
             << getInputs().size() << ")";
    for (BlockArgument arg : entry.getArguments())
      declaredIns.push_back(arg.getType());
    for (Type t : getOutputs().getTypes())
      declaredOuts.push_back(t);
  }

  // 1. K >= 1.
  if (declaredIns.empty())
    return emitOpError("requires at least 1 input port (K >= 1)");
  // 2. L >= 1.
  if (declaredOuts.empty())
    return emitOpError("requires at least 1 output port (L >= 1)");

  // 3. Uniform W on all PE ports (the inner / PE-side types).
  auto firstW = bitsWidth(declaredIns[0]);
  if (!firstW)
    return emitOpError(
               "requires uniform 'bits<W>' on all PE ports; PE input #0 has "
               "type ")
           << declaredIns[0];
  unsigned W = *firstW;
  for (auto [i, t] : llvm::enumerate(declaredIns)) {
    auto w = bitsWidth(t);
    if (!w || *w != W)
      return emitOpError("requires uniform 'bits<W>' on all PE ports; PE "
                         "input #")
             << i << " has type " << t << " (expected '!fabric.bits<" << W
             << ">')";
  }
  for (auto [i, t] : llvm::enumerate(declaredOuts)) {
    auto w = bitsWidth(t);
    if (!w || *w != W)
      return emitOpError("requires uniform 'bits<W>' on all PE ports; PE "
                         "output #")
             << i << " has type " << t << " (expected '!fabric.bits<" << W
             << ">')";
  }

  // 4. Outer-vs-inner per operand (anonymous form only): outer SSA type
  // may differ from inner block-arg type only in width and only for the
  // same fabric kind. The named form has no SSA operands.
  if (!isNamed) {
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      Type outerTy = getInputs()[i].getType();
      Type innerTy = arg.getType();
      if (outerTy == innerTy)
        continue;
      if (!sameModulePortKind(outerTy, innerTy))
        return emitOpError("operand #")
               << i << " outer type " << outerTy
               << " and PE block-arg inner type " << innerTy
               << " must share the same fabric kind (bits, bits_tag)";
      if (isa<MemRefType>(outerTy))
        return emitOpError("operand #")
               << i
               << ": memref operands cannot use the 'to <inner-type>' clause; "
                  "memref types must match exactly";
    }
  }

  // 5./6. Body whitelist: fabric.fu plus fabric.instantiate. Named form
  // additionally permits fabric.yield as terminator.
  unsigned numCompute = 0;
  for (Operation &op : entry) {
    if (isa<InstantiateOp>(op)) {
      ++numCompute;
      continue;
    }
    if (isa<YieldOp>(op)) {
      if (!isNamed)
        return op.emitOpError(
            "fabric.yield is not allowed in an anonymous fabric.pe body");
      continue;
    }
    auto fu = dyn_cast<FuOp>(op);
    if (!fu)
      return op.emitOpError("'fabric.pe' op body may only contain "
                            "fabric.fu and fabric.instantiate; got '")
             << op.getName().getStringRef() << "'";
    ++numCompute;

    // 7. Per-FU constraints.
    unsigned fuNumIns;
    unsigned fuNumOuts;
    SmallVector<Type, 4> fuIns;
    SmallVector<Type, 4> fuOuts;
    if (fu.getSymNameAttr()) {
      auto fta = fu.getFunctionTypeAttr();
      if (!fta)
        continue; // FU verifier flags the missing attr.
      auto ft = dyn_cast<FunctionType>(fta.getValue());
      if (!ft)
        continue;
      fuNumIns = ft.getNumInputs();
      fuNumOuts = ft.getNumResults();
      fuIns.assign(ft.getInputs().begin(), ft.getInputs().end());
      fuOuts.assign(ft.getResults().begin(), ft.getResults().end());
    } else {
      fuNumIns = fu.getInputs().size();
      fuNumOuts = fu.getOutputs().size();
      for (Type t : fu.getInputs().getTypes())
        fuIns.push_back(t);
      for (Type t : fu.getOutputs().getTypes())
        fuOuts.push_back(t);
    }
    if (fuNumIns > declaredIns.size())
      return fu.emitOpError("inner fabric.fu has ")
             << fuNumIns << " inputs which exceeds fabric.pe input count K="
             << declaredIns.size();
    if (fuNumOuts > declaredOuts.size())
      return fu.emitOpError("inner fabric.fu has ")
             << fuNumOuts << " outputs which exceeds fabric.pe output count L="
             << declaredOuts.size();
    for (auto [i, t] : llvm::enumerate(fuIns)) {
      auto w = bitsWidth(t);
      if (!w || *w != W)
        return fu.emitOpError(
                   "inner fabric.fu boundary width must equal fabric.pe "
                   "width W=")
               << W << "; FU input #" << i << " has type " << t;
    }
    for (auto [i, t] : llvm::enumerate(fuOuts)) {
      auto w = bitsWidth(t);
      if (!w || *w != W)
        return fu.emitOpError(
                   "inner fabric.fu boundary width must equal fabric.pe "
                   "width W=")
               << W << "; FU output #" << i << " has type " << t;
    }
  }
  if (numCompute < 1)
    return emitOpError(
        "body requires at least one fabric.fu or fabric.instantiate");

  // Named-form yield must close the body and match function_type results.
  if (isNamed) {
    if (entry.empty() || !isa<YieldOp>(entry.back()))
      return emitOpError(
          "named fabric.pe body must terminate with fabric.yield");
    auto yield = cast<YieldOp>(entry.back());
    if (yield.getValues().size() != declaredOuts.size())
      return emitOpError("yield value count (")
             << yield.getValues().size()
             << ") must match declared result count (" << declaredOuts.size()
             << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(yield.getValues(), declaredOuts))) {
      Value v;
      Type t;
      std::tie(v, t) = pair;
      if (v.getType() != t)
        return emitOpError("yield value #")
               << i << " type " << v.getType()
               << " must equal declared result type " << t;
    }
  }

  return success();
}

bool PeOp::isOptionalSymbol() { return true; }

//===----------------------------------------------------------------------===//
// fabric.instantiate (parser/printer/verifier defined in
// FabricInstantiateOp.cpp; kept in a separate translation unit to keep
// each file under the dialect's file-size guideline).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// fabric.module
//===----------------------------------------------------------------------===//
//
// Assembly form (mirrors func.func for inputs/outputs but with the fabric
// dialect's restricted port-type union):
//
//   fabric.module @sym(%a : !fabric.bits<32>,
//                      %b : memref<8xi32>) -> (!fabric.bits<32>) {
//     ...
//     fabric.yield %r : !fabric.bits<32>
//   }
//
// Both input list and result list may be empty.

ParseResult fabric::ModuleOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Symbol name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Argument list: `(` (%name : T (`,` ...)*)? `)`
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<Type, 4> argTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    auto parseOne = [&]() -> ParseResult {
      OpAsmParser::Argument arg;
      Type ty;
      if (parser.parseArgument(arg) || parser.parseColon() ||
          parser.parseType(ty))
        return failure();
      arg.type = ty;
      entryArgs.push_back(arg);
      argTypes.push_back(ty);
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

  // Result list: `->` ( `(` types? `)` | type )?  -- support the empty
  // form (no `->`) for backward compatibility with body-less modules,
  // an explicit `-> ()` for zero results, or `-> T` / `-> (T0, T1, ...)`.
  SmallVector<Type, 4> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
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
  }

  // Build the function_type attribute.
  auto funcType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(funcType));

  // Optional attribute dictionary keyword.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Region body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/false))
    return failure();

  // Materialize the implicit terminator (fabric.yield with zero operands)
  // when the body is empty -- mirrors SingleBlockImplicitTerminator.
  fabric::ModuleOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

void fabric::ModuleOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  Block &entry = getBody().front();
  // Inputs.
  p << '(';
  llvm::interleaveComma(entry.getArguments(), p, [&](BlockArgument bb) {
    p.printRegionArgument(bb, /*argAttrs=*/{}, /*omitType=*/true);
    p << " : " << bb.getType();
  });
  p << ')';
  // Outputs.
  ArrayRef<Type> resultTypes = getFunctionType().getResults();
  if (!resultTypes.empty()) {
    p << " -> ";
    if (resultTypes.size() == 1) {
      p << resultTypes.front();
    } else {
      p << '(';
      llvm::interleaveComma(resultTypes, p);
      p << ')';
    }
  }
  // Attribute dict (skip sym_name and function_type which we just printed).
  SmallVector<StringRef, 2> elided{"sym_name", "function_type"};
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  p << ' ';
  // Elide the implicit terminator (a fabric.yield with no operands and no
  // attributes) so an empty module body round-trips as `{ }`. When the
  // module declares results, the yield carries operands and is printed.
  bool printTerm = false;
  if (auto y = dyn_cast<YieldOp>(getBody().front().getTerminator())) {
    if (y.getValues().size() != 0)
      printTerm = true;
    else if (y->getAttrs().size() != 0)
      printTerm = true;
  } else {
    printTerm = true;
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printTerm);
}

RegionKind fabric::ModuleOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

LogicalResult fabric::ModuleOp::verify() {
  // Validate declared input/output types: each must be one of the four
  // allowed module port types (bits, bits_tag, memref). The
  // declarative type constraint Variadic<Fabric_ModulePortType> on the
  // results is checked by the auto-generated verifier; we re-check here
  // to give clean diagnostics for the input side, which is encoded as
  // entry-block arguments rather than ODS-typed operands.
  FunctionType ft = getFunctionType();
  for (auto [i, t] : llvm::enumerate(ft.getInputs())) {
    if (!isFabricModulePortType(t))
      return emitOpError("input #")
             << i << " type " << t
             << " is not an allowed fabric.module port type "
                "(allowed: !fabric.bits<W>, !fabric.bits_tag<W,T>, "
                "memref<...>)";
  }
  for (auto [i, t] : llvm::enumerate(ft.getResults())) {
    if (!isFabricModulePortType(t))
      return emitOpError("result #")
             << i << " type " << t
             << " is not an allowed fabric.module port type "
                "(allowed: !fabric.bits<W>, !fabric.bits_tag<W,T>, "
                "memref<...>)";
  }

  // Block argument count + types must match the declared inputs.
  Block &entry = getBody().front();
  if (entry.getNumArguments() != ft.getNumInputs())
    return emitOpError("entry block argument count (")
           << entry.getNumArguments() << ") must match declared input count ("
           << ft.getNumInputs() << ")";
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(entry.getArguments(), ft.getInputs()))) {
    BlockArgument bb;
    Type declared;
    std::tie(bb, declared) = pair;
    if (bb.getType() != declared)
      return emitOpError("entry block argument #")
             << i << " type " << bb.getType()
             << " must equal declared input type " << declared;
  }

  // Body whitelist: only fabric.pe, fabric.switch, fabric.mem,
  // fabric.fifo, fabric.module (nested), fabric.instantiate,
  // fabric.boundary, and the implicit fabric.yield terminator may
  // appear directly in the module body.
  for (Operation &op : entry) {
    if (isa<PeOp, SwitchOp, MemOp, FifoOp, fabric::ModuleOp, InstantiateOp,
            BoundaryOp, YieldOp>(op))
      continue;
    return op.emitOpError(
        "is not allowed inside fabric.module; only fabric.pe, "
        "fabric.switch, fabric.mem, fabric.fifo, fabric.module, "
        "fabric.instantiate, and fabric.boundary are permitted (plus the "
        "implicit terminator fabric.yield)");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.system, fabric.node, fabric.external_port, fabric.link
//===----------------------------------------------------------------------===//

LogicalResult NodeOp::verify() {
  StringRef kind = getKindAttr().getValue();
  if (!isBaselineSystemNodeKind(kind))
    return emitOpError("kind '")
           << kind
           << "' is not supported by the baseline fabric.system verifier";
  if (failed(verifySystemPortArray(getOperation(), getPortsAttr())))
    return failure();

  if (kind == "acc_core") {
    if (!getSpatialAttr())
      return emitOpError(
          "kind 'acc_core' requires a spatial fabric.module reference");
    if (!getScalarAttr())
      return emitOpError("kind 'acc_core' requires scalar metadata");
  }
  if (kind == "fixed_accelerator") {
    if (!getFunctionAttr() || getFunctionAttr().getValue().empty())
      return emitOpError(
          "kind 'fixed_accelerator' requires a non-empty function attribute");
  }
  if (kind == "memory") {
    auto bytes = getBytesAttr();
    if (!bytes)
      return emitOpError("kind 'memory' requires bytes metadata");
    if (bytes.getInt() <= 0)
      return emitOpError("kind 'memory' requires positive bytes metadata");
  }
  if (kind == "cache") {
    std::optional<int64_t> lineBytes =
        getPositiveI64Param(getParamsAttr(), "line_bytes");
    std::optional<int64_t> capacityBytes =
        getPositiveI64Param(getParamsAttr(), "capacity_bytes");
    if (!lineBytes || !capacityBytes || !isPowerOfTwo(*lineBytes) ||
        *capacityBytes < *lineBytes)
      return emitOpError("kind 'cache' requires positive power-of-two "
                         "line_bytes and positive capacity_bytes of at "
                         "least one line");

    llvm::StringMap<SystemPortProfile> profiles;
    if (failed(collectSystemPortProfiles(getOperation(), getPortsAttr(),
                                         profiles)))
      return failure();
    bool hasSubordinate = false;
    bool hasManager = false;
    for (const auto &entry : profiles) {
      hasSubordinate |= hasMemorySubordinateShape(entry.getValue());
      hasManager |= hasMemoryManagerShape(entry.getValue());
    }
    if (!hasSubordinate || !hasManager)
      return emitOpError("kind 'cache' requires at least one subordinate "
                         "memory port and one manager memory port");
  }
  if (kind == "dma_engine") {
    std::optional<int64_t> queueDepth =
        getPositiveI64Param(getParamsAttr(), "queue_depth");
    if (!queueDepth)
      return emitOpError("kind 'dma_engine' requires positive queue_depth");

    llvm::StringMap<SystemPortProfile> profiles;
    if (failed(collectSystemPortProfiles(getOperation(), getPortsAttr(),
                                         profiles)))
      return failure();
    bool hasControlOrDescriptor = false;
    bool hasMemoryManager = false;
    for (const auto &entry : profiles) {
      hasControlOrDescriptor |=
          isDmaControlOrDescriptorPort(entry.getKey(), entry.getValue());
      hasMemoryManager |= hasMemoryManagerShape(entry.getValue());
    }
    if (!hasControlOrDescriptor || !hasMemoryManager)
      return emitOpError("kind 'dma_engine' requires at least one control or "
                         "descriptor port and one memory-capable manager port");
  }
  return success();
}

LogicalResult ExternalPortOp::verify() {
  return verifySystemPortArray(getOperation(), getPortsAttr());
}

LogicalResult LinkOp::verify() {
  if (getSrc() == getDst() && getSrcPort() == getDstPort() &&
      getSrcChannel() == getDstChannel())
    return emitOpError("source and destination endpoints must be distinct");
  return success();
}

LogicalResult SystemOp::verify() {
  StringRef memoryModel = getMemoryModelAttr().getValue();
  if (!isValidMemoryModel(memoryModel))
    return emitOpError("memory_model '")
           << memoryModel
           << "' is not one of sequential, tso, release_acquire, weak, custom";
  if (memoryModel == "custom") {
    bool hasModelName =
        getModelNameAttr() && !getModelNameAttr().getValue().empty();
    bool hasParams = getParamsAttr() && !getParamsAttr().empty();
    if (!hasModelName && !hasParams)
      return emitOpError(
          "memory_model 'custom' requires model_name or non-empty params");
  }

  struct EndpointInfo {
    std::string direction;
    Operation *owner = nullptr;
  };

  llvm::StringMap<EndpointInfo> endpoints;
  llvm::StringSet<> nodeSymbols;
  llvm::StringSet<> externalSymbols;
  llvm::SmallVector<LinkOp> links;

  auto collectEndpointOwner =
      [&](Operation *op, StringRef symbol, ArrayAttr ports,
          llvm::StringSet<> &symbolSet) -> LogicalResult {
    if (!symbolSet.insert(symbol).second)
      return op->emitOpError("duplicates system symbol @") << symbol;
    for (Attribute attr : ports) {
      FailureOr<SystemChannel> channel = parseSystemChannel(attr, op);
      if (failed(channel))
        return failure();
      std::string key = endpointKey(symbol, channel->port, channel->channel);
      if (endpoints.contains(key))
        return op->emitOpError("duplicates endpoint @")
               << symbol << " " << channel->port << "." << channel->channel;
      endpoints[key] = EndpointInfo{channel->direction, op};
    }
    return success();
  };

  for (Operation &op : getBody().front()) {
    if (auto node = dyn_cast<NodeOp>(&op)) {
      StringRef symbol = node.getSymNameAttr().getValue();
      if (failed(collectEndpointOwner(&op, symbol, node.getPortsAttr(),
                                      nodeSymbols)))
        return failure();
      if (node.getKindAttr().getValue() == "acc_core") {
        FlatSymbolRefAttr spatial = node.getSpatialAttr();
        Operation *target =
            spatial ? lookupSymbolUpward(node.getOperation(), spatial)
                    : nullptr;
        if (!target)
          return node.emitOpError("acc_core spatial reference @")
                 << (spatial ? spatial.getValue() : StringRef(""))
                 << " does not resolve to a fabric.module";
        if (!isa<fabric::ModuleOp>(target))
          return node.emitOpError("acc_core spatial reference @")
                 << spatial.getValue() << " resolved to "
                 << target->getName().getStringRef()
                 << ", expected fabric.module";
      }
      continue;
    }
    if (auto external = dyn_cast<ExternalPortOp>(&op)) {
      StringRef symbol = external.getSymNameAttr().getValue();
      if (failed(collectEndpointOwner(&op, symbol, external.getPortsAttr(),
                                      externalSymbols)))
        return failure();
      continue;
    }
    if (auto link = dyn_cast<LinkOp>(&op)) {
      links.push_back(link);
      continue;
    }
    if (isa<YieldOp>(op))
      continue;
    return op.emitOpError(
        "is not allowed inside fabric.system; only fabric.node, "
        "fabric.external_port, fabric.link, and fabric.yield are permitted");
  }

  llvm::StringSet<> usedSources;
  llvm::StringSet<> usedDests;
  auto verifyEndpoint = [&](LinkOp link, FlatSymbolRefAttr ownerAttr,
                            StringRef port, StringRef channel,
                            StringRef expectedDirection,
                            llvm::StringSet<> &used) -> LogicalResult {
    StringRef owner = ownerAttr.getValue();
    std::string key = endpointKey(owner, port, channel);
    auto it = endpoints.find(key);
    if (it == endpoints.end())
      return link.emitOpError("endpoint @")
             << owner
             << " does not refer to a fabric.node or fabric.external_port in "
                "this fabric.system";
    if (StringRef(it->second.direction) != expectedDirection)
      return link.emitOpError("endpoint @")
             << owner << " " << port << "." << channel << " is "
             << it->second.direction << ", expected " << expectedDirection;
    if (!used.insert(key).second)
      return link.emitOpError("endpoint @")
             << owner << " " << port << "." << channel
             << " is used by more than one fabric.link";
    return success();
  };

  for (LinkOp link : links) {
    if (failed(verifyEndpoint(link, link.getSrcAttr(), link.getSrcPort(),
                              link.getSrcChannel(), "output", usedSources)))
      return failure();
    if (failed(verifyEndpoint(link, link.getDstAttr(), link.getDstPort(),
                              link.getDstChannel(), "input", usedDests)))
      return failure();
  }

  return success();
}

// fabric.yield assembly format. Two forms are accepted; the printer picks
// the compact form when no per-value `to` clause is needed.
//
//   fabric.yield                                            // empty
//   fabric.yield %v0, %v1 : T0, T1                          // compact
//   fabric.yield %v0 : T0 [to TR0], %v1 : T1 [to TR1] ...   // per-value
//
// The optional `to <module-result-type>` clause is only meaningful inside
// fabric.module (it expresses the connection-point width relaxation
// against the module's declared result type). Inside fabric.fu the
// clause must not appear; the verifier rejects it.

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  // Allow the empty form: `fabric.yield` followed only by attr-dict.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> sourceTypes;   // types of the SSA source values
  SmallVector<Type, 4> declaredTypes; // declared destination types (with `to`)

  OpAsmParser::UnresolvedOperand first;
  OptionalParseResult firstParse = parser.parseOptionalOperand(first);
  if (firstParse.has_value()) {
    if (failed(*firstParse))
      return failure();
    operands.push_back(first);

    // Decide between compact form (next token is `,` or `:` followed by a
    // type list) and per-value form (next token is `:` followed by a single
    // type that may be paired with `to` and another `,` operand).
    //
    // Strategy: collect operands greedily while we see `,`; after the
    // operand list, consume `:` and parse types matching the operand count.
    // If during that phase we see `to` (per-value form starting with one
    // operand), fall through and re-route. We use a simpler split: peek at
    // the next token; if it's `,` we're in compact form's operand list, if
    // it's `:` we proceed with type parsing.
    bool isCompact = false;
    if (succeeded(parser.parseOptionalComma())) {
      isCompact = true;
      auto parseMoreOperand = [&]() -> ParseResult {
        OpAsmParser::UnresolvedOperand op;
        return parser.parseOperand(op).failed()
                   ? failure()
                   : (operands.push_back(op), success());
      };
      if (parseMoreOperand())
        return failure();
      while (succeeded(parser.parseOptionalComma()))
        if (parseMoreOperand())
          return failure();
    }

    if (parser.parseColon())
      return failure();

    if (isCompact) {
      // Compact: parse a type list with N entries.
      SmallVector<Type, 4> types;
      if (parser.parseTypeList(types))
        return failure();
      if (types.size() != operands.size())
        return parser.emitError(parser.getCurrentLocation(),
                                "yield operand count and type count differ");
      sourceTypes = std::move(types);
      declaredTypes = sourceTypes;
    } else {
      // Per-value: we already have the first operand; parse `T [to TR]`,
      // then optional `, %op : T [to TR]` repeated.
      auto parseTypePair = [&]() -> ParseResult {
        Type t;
        if (parser.parseType(t))
          return failure();
        sourceTypes.push_back(t);
        Type r = t;
        if (succeeded(parser.parseOptionalKeyword("to")))
          if (parser.parseType(r))
            return failure();
        declaredTypes.push_back(r);
        return success();
      };
      if (parseTypePair())
        return failure();
      while (succeeded(parser.parseOptionalComma())) {
        OpAsmParser::UnresolvedOperand op;
        if (parser.parseOperand(op) || parser.parseColon())
          return failure();
        operands.push_back(op);
        if (parseTypePair())
          return failure();
      }
    }

    // Resolve operands at their declared source types.
    if (parser.resolveOperands(operands, sourceTypes,
                               parser.getCurrentLocation(), result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Stash the declared destination types as an attribute so the printer can
  // emit per-value `to` clauses on round-trip and the verifier can compare
  // them against the parent's declared result types.
  if (!operands.empty()) {
    SmallVector<Attribute, 4> typeAttrs;
    typeAttrs.reserve(declaredTypes.size());
    for (Type t : declaredTypes)
      typeAttrs.push_back(TypeAttr::get(t));
    result.addAttribute("declared_types",
                        ArrayAttr::get(parser.getContext(), typeAttrs));
  }
  return success();
}

void YieldOp::print(OpAsmPrinter &p) {
  auto values = getValues();
  if (values.empty()) {
    p.printOptionalAttrDict(getOperation()->getAttrs());
    return;
  }

  // Read the per-value declared destination types out of the
  // `declared_types` attribute when present; otherwise the declared
  // type equals the source type (no relaxation).
  auto declaredArr = (*this)->getAttrOfType<ArrayAttr>("declared_types");
  SmallVector<Type, 4> declared;
  declared.reserve(values.size());
  if (declaredArr && declaredArr.size() == values.size()) {
    for (Attribute a : declaredArr) {
      if (auto ta = dyn_cast<TypeAttr>(a))
        declared.push_back(ta.getValue());
      else
        declared.push_back(Type{});
    }
  }
  if (declared.size() != values.size())
    declared.assign(values.size(), Type{});

  bool needsPerValue = false;
  for (auto [v, d] : llvm::zip(values, declared))
    if (d && d != v.getType()) {
      needsPerValue = true;
      break;
    }

  p << ' ';
  if (!needsPerValue) {
    // Compact form: %v0, %v1 : T0, T1
    llvm::interleaveComma(values, p, [&](Value v) { p << v; });
    p << " : ";
    llvm::interleaveComma(values, p, [&](Value v) { p << v.getType(); });
  } else {
    // Per-value form: %v0 : T0 [to TR0], %v1 : T1 [to TR1]
    llvm::interleaveComma(llvm::zip(values, declared), p, [&](auto pair) {
      Value v;
      Type d;
      std::tie(v, d) = pair;
      p << v << " : " << v.getType();
      if (d && d != v.getType())
        p << " to " << d;
    });
  }

  // Print attr-dict but elide `declared_types` (handled above).
  SmallVector<StringRef, 1> elided{"declared_types"};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elided);
}

LogicalResult YieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  // Recover the per-value declared destination types if any (defaults to
  // each operand's source type when unset).
  auto declaredArr = (*this)->getAttrOfType<ArrayAttr>("declared_types");
  SmallVector<Type, 4> declared;
  declared.reserve(getValues().size());
  if (declaredArr && declaredArr.size() == getValues().size()) {
    for (Attribute a : declaredArr) {
      auto ta = dyn_cast<TypeAttr>(a);
      declared.push_back(ta ? ta.getValue() : Type{});
    }
  } else {
    for (Value v : getValues())
      declared.push_back(v.getType());
  }

  if (auto fu = dyn_cast_or_null<FuOp>(parent)) {
    // Resolve expected result types per fu form: named (function_type) or
    // anonymous (op.results).
    SmallVector<Type, 4> expectedResults;
    if (fu.getSymNameAttr()) {
      if (auto fta = fu.getFunctionTypeAttr())
        if (auto ft = dyn_cast<FunctionType>(fta.getValue()))
          for (Type t : ft.getResults())
            expectedResults.push_back(t);
    } else {
      for (Type t : fu.getOutputs().getTypes())
        expectedResults.push_back(t);
    }
    if (getValues().size() != expectedResults.size())
      return emitOpError("yield value count (")
             << getValues().size()
             << ") must match parent fabric.fu result "
                "count ("
             << expectedResults.size() << ")";
    for (auto [i, v] : llvm::enumerate(getValues())) {
      Type expected = expectedResults[i];
      Type innerTy = v.getType();
      // Output-side boundary widening (symmetric to the FU input-side
      // `to <inner-type>` truncation): when the per-value `to <type>`
      // clause is present, it names the FU's OUTER result type. Inner
      // value width must be <= outer width; the high (outer - inner)
      // bits are zero-filled at the FU boundary on each token.
      if (declared[i] && declared[i] != innerTy) {
        if (declared[i] != expected)
          return emitOpError("yield value #")
                 << i << ": declared destination type " << declared[i]
                 << " does not match parent fabric.fu result type " << expected;
        auto innerW = bitsWidth(innerTy);
        auto outerW = bitsWidth(expected);
        if (!innerW)
          return emitOpError("yield value #")
                 << i << " inner type " << innerTy
                 << " must be fabric.bits<N> for FU output boundary widening";
        if (!outerW)
          return emitOpError("yield value #")
                 << i << " outer type " << expected
                 << " must be fabric.bits<N> for FU output boundary widening";
        if (*innerW > *outerW)
          return emitOpError("yield value #")
                 << i << " inner bits-width " << *innerW
                 << " is greater than outer bits-width " << *outerW
                 << "; the FU output boundary only supports low-bit-aligned "
                    "widening (inner <= outer, high bits zero-filled)";
        continue;
      }
      if (innerTy != expected)
        return emitOpError("yield value #")
               << i << " type " << innerTy
               << " must match parent fabric.fu result type " << expected;
    }
    return success();
  }
  if (auto pe = dyn_cast_or_null<PeOp>(parent)) {
    // fabric.yield inside a fabric.pe body is only legal when the PE is in
    // named template form (signature carried in `function_type`).
    auto fta = pe.getFunctionTypeAttr();
    if (!pe.getSymNameAttr() || !fta)
      return emitOpError(
          "fabric.yield is only legal inside a named fabric.pe template "
          "(anonymous fabric.pe has no terminator)");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    SmallVector<Type, 4> expectedResults;
    if (ft)
      for (Type t : ft.getResults())
        expectedResults.push_back(t);
    if (getValues().size() != expectedResults.size())
      return emitOpError("yield value count (")
             << getValues().size()
             << ") must match parent fabric.pe result count ("
             << expectedResults.size() << ")";
    bool isTemporal = (pe.getSchedule() == Schedule::Temporal);
    for (auto [i, v] : llvm::enumerate(getValues())) {
      Type expected = expectedResults[i];
      if (declared[i] && declared[i] != v.getType())
        return emitOpError("yield value #")
               << i << ": 'to <type>' clause is not allowed inside fabric.pe";
      // Temporal PE: yield carries !fabric.bits<W> matching the bits-data
      // part of the declared bits_tag<W, T> port; tag is reattached at
      // the PE boundary by hardware. The detailed bits<W'>/bits_tag<W,T>
      // shape check lives in verifyPeTemporal; here we only need to
      // accept the bits-vs-bits_tag mismatch without rejecting it.
      if (isTemporal) {
        if (isa<BitsType>(v.getType()) && isa<BitsTagType>(expected))
          continue;
      }
      if (v.getType() != expected)
        return emitOpError("yield value #")
               << i << " type " << v.getType()
               << " must match parent fabric.pe result type " << expected;
    }
    return success();
  }
  if (auto mod = dyn_cast_or_null<fabric::ModuleOp>(parent)) {
    ArrayRef<Type> resultTypes = mod.getFunctionType().getResults();
    if (getValues().size() != resultTypes.size())
      return emitOpError("yield value count (")
             << getValues().size()
             << ") must match parent fabric.module result count ("
             << resultTypes.size() << ")";
    for (auto [i, v] : llvm::enumerate(getValues())) {
      Type srcTy = v.getType();
      Type modResultTy = resultTypes[i];
      // Apply the connection-point rule. Order: validate the source type
      // is a legal fabric.module port type, then check kind agreement, then
      // check the per-value `to <type>` clause (when present) matches the
      // module's declared result type, then enforce memref's exact-match
      // rule.
      if (!isFabricModulePortType(srcTy))
        return emitOpError("yield value #")
               << i << " has type " << srcTy
               << " which is not an allowed fabric.module port type";
      if (!sameModulePortKind(srcTy, modResultTy))
        return emitOpError("yield value #")
               << i << " type " << srcTy
               << " has a different fabric kind than the module result type "
               << modResultTy
               << "; type-kind must match (bits/bits_tag/memref)";
      if (declared[i] && declared[i] != modResultTy)
        return emitOpError("yield value #")
               << i << ": declared destination type " << declared[i]
               << " does not match the module's result type " << modResultTy;
      if (isa<MemRefType>(srcTy) && srcTy != modResultTy)
        return emitOpError("yield value #")
               << i << " memref type " << srcTy
               << " must match the module result type " << modResultTy
               << " exactly (no width/shape relaxation on memref)";
    }
    return success();
  }
  if (isa_and_nonnull<SystemOp>(parent)) {
    if (!getValues().empty())
      return emitOpError("inside fabric.system must not carry values");
    if (declaredArr)
      return emitOpError(
          "inside fabric.system must not carry declared destination types");
    return success();
  }
  return emitOpError("expects parent op 'fabric.fu', 'fabric.pe' (named), "
                     "'fabric.module', or 'fabric.system'");
}
