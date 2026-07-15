#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/ConfiguredFunction.h"

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
      if (!getFabricBitsWidth(t))
        return emitOpError("declared input #")
               << i << " must be fabric.bits<N>, got " << t;
    }
    for (auto [i, t] : llvm::enumerate(ft.getResults())) {
      if (!getFabricBitsWidth(t))
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
    return verifyValidSemanticEncodings(*this);
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
    auto outerW = getFabricBitsWidth(outerTy);
    auto innerW = getFabricBitsWidth(innerTy);
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
  return verifyValidSemanticEncodings(*this);
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
  auto firstW = getFabricBitsWidth(declaredIns[0]);
  if (!firstW)
    return emitOpError(
               "requires uniform 'bits<W>' on all PE ports; PE input #0 has "
               "type ")
           << declaredIns[0];
  unsigned W = *firstW;
  for (auto [i, t] : llvm::enumerate(declaredIns)) {
    auto w = getFabricBitsWidth(t);
    if (!w || *w != W)
      return emitOpError("requires uniform 'bits<W>' on all PE ports; PE "
                         "input #")
             << i << " has type " << t << " (expected '!fabric.bits<" << W
             << ">')";
  }
  for (auto [i, t] : llvm::enumerate(declaredOuts)) {
    auto w = getFabricBitsWidth(t);
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
      if (!haveSameFabricModulePortKind(outerTy, innerTy))
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
      auto w = getFabricBitsWidth(t);
      if (!w || *w != W)
        return fu.emitOpError(
                   "inner fabric.fu boundary width must equal fabric.pe "
                   "width W=")
               << W << "; FU input #" << i << " has type " << t;
    }
    for (auto [i, t] : llvm::enumerate(fuOuts)) {
      auto w = getFabricBitsWidth(t);
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

  // Module-body stream transports are point-to-point. Count only uses owned
  // by direct body operations; nested PE/FU regions define their own
  // transport semantics.
  auto countBodyConsumers = [&](Value value) {
    unsigned count = 0;
    for (OpOperand &use : value.getUses())
      if (use.getOwner()->getBlock() == &entry)
        ++count;
    return count;
  };

  for (auto [i, argument] : llvm::enumerate(entry.getArguments())) {
    if (!isa<BitsType, BitsTagType>(argument.getType()))
      continue;
    unsigned consumerCount = countBodyConsumers(argument);
    if (consumerCount > 1)
      return emitOpError(
                 "transport source is used by more than one consumer in this "
                 "fabric.module body: block argument #")
             << i << " of type " << argument.getType() << " has "
             << consumerCount << " consuming uses";
  }

  for (Operation &sourceOp : entry) {
    for (auto [i, result] : llvm::enumerate(sourceOp.getResults())) {
      if (!isa<BitsType, BitsTagType>(result.getType()))
        continue;
      unsigned consumerCount = countBodyConsumers(result);
      if (consumerCount > 1)
        return sourceOp.emitOpError(
                   "transport source is used by more than one consumer in "
                   "this fabric.module body: result #")
               << i << " of type " << result.getType() << " has "
               << consumerCount << " consuming uses";
    }
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
        auto innerW = getFabricBitsWidth(innerTy);
        auto outerW = getFabricBitsWidth(expected);
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
      if (!haveSameFabricModulePortKind(srcTy, modResultTy))
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
