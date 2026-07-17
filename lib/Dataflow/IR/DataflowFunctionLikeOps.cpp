// Implementation of the symbol-bearing function-like ops added to the
// dataflow dialect for SCF-to-DFG lowering: dataflow.thread (def),
// dataflow.thread.launch (async launcher), dataflow.thread.yield
// (terminator), dataflow.graph.func (def), dataflow.graph.launch (sync
// launcher), and dataflow.graph.return (terminator).
//
// The regional dataflow.graph and dataflow.subgraph ops, plus their
// dataflow.yield terminator, remain implemented in DataflowOps.cpp.

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <array>

using namespace mlir;
using namespace dataflow;

//===----------------------------------------------------------------------===//
// Common helpers used by both function-like ops.
//===----------------------------------------------------------------------===//

namespace {

static bool isSupportedArmInlineAsm(Operation *op) {
  if (op->getName().getStringRef() != "llvm.inline_asm")
    return false;
  auto asmString = op->getAttrOfType<StringAttr>("asm_string");
  if (!asmString)
    return false;
  StringRef text = asmString.getValue();
  return text == "pkhbt $0, $1, $2, lsl $3" ||
         text == "pkhtb $0, $1, $2, asr $3" ||
         text == "sxtab16 $0, $1, $2" || text == "sxtb16 $0, $1";
}

static bool isGraphMemoryCapabilityType(Type type) {
  return isa<MemRefType, UnrankedMemRefType, LLVM::LLVMPointerType>(type);
}

static bool containsGraphMemoryCapability(Type type) {
  if (isGraphMemoryCapabilityType(type))
    return true;
  if (auto tuple = dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsGraphMemoryCapability);
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type))
    return !structure.isOpaque() &&
           llvm::any_of(structure.getBody(), containsGraphMemoryCapability);
  if (auto shaped = dyn_cast<ShapedType>(type))
    return containsGraphMemoryCapability(shaped.getElementType());
  if (auto complex = dyn_cast<ComplexType>(type))
    return containsGraphMemoryCapability(complex.getElementType());
  return false;
}

static std::array<int32_t, 3> defaultGraphSegments(size_t count) {
  return {static_cast<int32_t>(count), 0, 0};
}

static GraphPortKind graphPortKindAt(ArrayRef<int32_t> segments,
                                     unsigned index) {
  if (index < static_cast<unsigned>(segments[0]))
    return GraphPortKind::Value;
  if (index < static_cast<unsigned>(segments[0] + segments[1]))
    return GraphPortKind::Stream;
  return GraphPortKind::Memory;
}

static StringRef graphPortKindName(GraphPortKind kind) {
  switch (kind) {
  case GraphPortKind::Value:
    return "value";
  case GraphPortKind::Stream:
    return "stream";
  case GraphPortKind::Memory:
    return "memory";
  }
  llvm_unreachable("unknown graph port kind");
}

static LogicalResult verifyGraphPortType(Operation *op, Type type,
                                         GraphPortKind kind,
                                         StringRef direction,
                                         unsigned kindIndex) {
  if (kind == GraphPortKind::Memory) {
    if (!isGraphMemoryCapabilityType(type))
      return op->emitOpError()
             << "memory " << direction << " #" << kindIndex
             << " has non-capability type " << type;
    return success();
  }
  if (containsGraphMemoryCapability(type))
    return op->emitOpError()
           << graphPortKindName(kind) << " " << direction << " #"
           << kindIndex << " contains memory capability type " << type;
  if (kind == GraphPortKind::Value && isa<NoneType>(type))
    return op->emitOpError()
           << graphPortKindName(kind) << " " << direction << " #"
           << kindIndex << " must not use protocol type none";
  return success();
}

// Build a generic `func.func`-shaped op state for our function-like
// ops. Used by both ThreadOp and GraphFuncOp.
template <typename Op>
void buildFunctionLike(OpBuilder &builder, OperationState &state,
                       StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(Op::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

template <typename Op>
ParseResult parseFunctionLike(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      Op::getFunctionTypeAttrName(result.name), buildFuncType,
      Op::getArgAttrsAttrName(result.name),
      Op::getResAttrsAttrName(result.name));
}

template <typename Op>
void printFunctionLike(OpAsmPrinter &p, Op op) {
  function_interface_impl::printFunctionOp(
      p, op, /*isVariadic=*/false, op.getFunctionTypeAttrName(),
      op.getArgAttrsAttrName(), op.getResAttrsAttrName());
}

} // namespace

//===----------------------------------------------------------------------===//
// dataflow.thread (definition)
//===----------------------------------------------------------------------===//

void ThreadOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs) {
  buildFunctionLike<ThreadOp>(builder, state, name, type, attrs);
}

// Custom assembly format for dataflow.thread:
//
//   dataflow.thread [visibility] @sym (T0, T1, ...)
//                   ( `ctrl` `(` ctrlArg : none `)` )?
//                   ( `iv`   `(` ivArgs... : index `)` )?
//                   attributes? body
//
// Function-signature args print just like func.func. The trailing
// `ctrl(...)` clause carries the body's thread_ctrl block arg (per
// spec section 5.4.1); the `iv(...)` clause carries the grid index
// block args. Both clauses are optional in the parser surface so
// that purely external thread declarations (no body) round-trip
// cleanly. Body-carrying threads, however, are required by the
// verifier to have a thread_ctrl slot.

ParseResult ThreadOp::parse(OpAsmParser &parser, OperationState &result) {
  ::mlir::Builder &builder = parser.getBuilder();
  ::mlir::StringAttr visibilityAttr;
  StringRef visibility;
  if (succeeded(parser.parseOptionalKeyword(&visibility, {"private"}))) {
    visibilityAttr = builder.getStringAttr(visibility);
  }

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Function arguments + (empty) results.
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes,
          resultAttrs))
    return failure();
  if (!resultTypes.empty())
    return parser.emitError(parser.getNameLoc(),
                            "dataflow.thread does not have function results");

  SmallVector<Type, 4> argTypes;
  for (auto &arg : arguments)
    argTypes.push_back(arg.type);
  auto funcType = builder.getFunctionType(argTypes, /*results=*/{});
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcType));

  // Helper: parse a parenthesized comma-separated list of typed
  // arguments into the given vector. Allows an empty list `()`.
  auto parseTypedArgList =
      [&](SmallVectorImpl<OpAsmParser::Argument> &out) -> ParseResult {
    if (parser.parseLParen())
      return failure();
    if (succeeded(parser.parseOptionalRParen()))
      return success();
    auto parseOne = [&]() -> ParseResult {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, /*allowType=*/true))
        return failure();
      out.push_back(arg);
      return success();
    };
    if (parseOne())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseOne())
        return failure();
    return parser.parseRParen();
  };

  // Optional `ctrl ( <name> : <type> )` for the thread_ctrl slot.
  SmallVector<OpAsmParser::Argument> ctrlArgs;
  if (succeeded(parser.parseOptionalKeyword("ctrl"))) {
    if (parseTypedArgList(ctrlArgs))
      return failure();
  }

  // Optional `iv ( <name> : <type> [, ...] )` for grid IV slots.
  SmallVector<OpAsmParser::Argument> ivArgs;
  if (succeeded(parser.parseOptionalKeyword("iv"))) {
    if (parseTypedArgList(ivArgs))
      return failure();
  }

  if (visibilityAttr) {
    result.addAttribute(getSymVisibilityAttrName(result.name), visibilityAttr);
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> allArgs(arguments);
  for (auto &a : ctrlArgs)
    allArgs.push_back(a);
  for (auto &a : ivArgs)
    allArgs.push_back(a);
  if (parser.parseRegion(*body, allArgs, /*enableNameShadowing=*/false))
    return failure();
  ThreadOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void ThreadOp::print(OpAsmPrinter &p) {
  if (auto vis = getSymVisibility())
    p << ' ' << *vis;
  p << ' ';
  p.printSymbolName(getSymName());
  ArrayRef<Type> argTypes = getArgumentTypes();
  Block *entry = getBody().empty() ? nullptr : &getBody().front();
  // Print function-signature arguments inline as the function's
  // argument list. When the op is external (no body) we have to
  // synthesize them from `argTypes`.
  p << '(';
  for (size_t i = 0, e = argTypes.size(); i < e; ++i) {
    if (i)
      p << ", ";
    if (entry) {
      p.printRegionArgument(entry->getArgument(i));
    } else {
      p.printType(argTypes[i]);
    }
  }
  p << ')';

  // Print the optional `ctrl ( ... )` and `iv ( ... )` clauses.
  if (entry && entry->getNumArguments() > argTypes.size()) {
    const size_t N = argTypes.size();
    // Entry block args at indices [N .. end). By layout convention,
    // the first one is the `none`-typed thread_ctrl, the rest are
    // `index`-typed ivs. The verifier enforces this; the printer
    // simply reflects whatever shape is present.
    if (entry->getNumArguments() >= N + 1) {
      p << " ctrl (";
      p.printRegionArgument(entry->getArgument(N));
      p << ')';
    }
    if (entry->getNumArguments() > N + 1) {
      p << " iv (";
      for (size_t i = N + 1, e = entry->getNumArguments(); i < e; ++i) {
        if (i > N + 1)
          p << ", ";
        p.printRegionArgument(entry->getArgument(i));
      }
      p << ')';
    }
  }

  ::llvm::SmallVector<::llvm::StringRef, 4> elidedAttrs = {
      SymbolTable::getSymbolAttrName(),
      getFunctionTypeAttrName(),
      getSymVisibilityAttrName(),
      getArgAttrsAttrName(),
      getResAttrsAttrName(),
  };
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult ThreadOp::verify() {
  // Symbol visibility, when set, must be private. The entry-block and
  // function-type consistency check lives in verifyBody.
  if (auto vis = getSymVisibility()) {
    if (*vis != "private" && *vis != "")
      return emitOpError("sym_visibility must be 'private'; got \"")
             << *vis << "\"";
  }
  return success();
}

// CallableOpInterface methods come from extraClassDeclaration in
// DataflowOps.td. ArgAndResultAttrsOpInterface auto-generates the
// arg/res attr getters/setters/removers from the tablegen field
// declarations.

//===----------------------------------------------------------------------===//
// dataflow.thread.yield
//===----------------------------------------------------------------------===//

// No verifier body needed; ODS enforces the optional variadic `none`
// completion frontier and terminator placement.

//===----------------------------------------------------------------------===//
// dataflow.thread.launch
//===----------------------------------------------------------------------===//

LogicalResult ThreadLaunchOp::verifySymbolUses(SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<ThreadOp>(*this, getCalleeAttr());
  if (!callee)
    return emitOpError("'")
           << getCallee()
           << "' does not reference a valid 'dataflow.thread' op";

  // Body operand types must equal callee.function_type.inputs
  // position-by-position.
  ArrayRef<Type> calleeInputs = callee.getFunctionType().getInputs();
  if (getBodyOperands().size() != calleeInputs.size())
    return emitOpError("body operand count (")
           << getBodyOperands().size()
           << ") does not match callee input count ("
           << calleeInputs.size() << ")";
  for (size_t i = 0, e = calleeInputs.size(); i < e; ++i) {
    Type expected = calleeInputs[i];
    Type actual = getBodyOperands()[i].getType();
    if (actual != expected)
      return emitOpError("body operand #")
             << i << " type " << actual
             << " does not match callee input type " << expected;
  }
  return success();
}

LogicalResult ThreadLaunchOp::verify() {
  if ((*this)->getParentOfType<ThreadOp>() ||
      (*this)->getParentOfType<GraphFuncOp>() ||
      (*this)->getParentOfType<GraphOp>())
    return emitOpError(
        "must appear outside any dataflow.thread or dataflow.graph "
        "definition");

  // The op result is the required thread token. Tablegen enforces its type.
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.thread.wait
//===----------------------------------------------------------------------===//

LogicalResult ThreadWaitOp::verify() {
  if ((*this)->getParentOfType<ThreadOp>() ||
      (*this)->getParentOfType<GraphFuncOp>() ||
      (*this)->getParentOfType<GraphOp>())
    return emitOpError(
        "must appear outside any dataflow.thread or dataflow.graph "
        "definition");
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.graph.func (definition)
//===----------------------------------------------------------------------===//

void GraphFuncOp::build(OpBuilder &builder, OperationState &state,
                        StringRef name, FunctionType type,
                        ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute, 8> normalizedAttrs(attrs.begin(), attrs.end());
  auto hasAttr = [&](StringRef name) {
    return llvm::any_of(normalizedAttrs, [&](NamedAttribute attr) {
      return attr.getName().strref() == name;
    });
  };
  if (!hasAttr("input_segments")) {
    auto segments = defaultGraphSegments(type.getNumInputs());
    normalizedAttrs.push_back(builder.getNamedAttr(
        "input_segments", builder.getDenseI32ArrayAttr(segments)));
  }
  if (!hasAttr("result_segments")) {
    auto segments = defaultGraphSegments(type.getNumResults());
    normalizedAttrs.push_back(builder.getNamedAttr(
        "result_segments", builder.getDenseI32ArrayAttr(segments)));
  }
  buildFunctionLike<GraphFuncOp>(builder, state, name, type, normalizedAttrs);
}

ParseResult GraphFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  StringAttr visibilityAttr;
  StringRef visibility;
  if (succeeded(parser.parseOptionalKeyword(&visibility, {"private"})))
    visibilityAttr = builder.getStringAttr(visibility);

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes,
          resultAttrs))
    return failure();
  if (arguments.empty() || !isa<NoneType>(arguments.front().type))
    return parser.emitError(parser.getNameLoc(),
                            "graph signature must begin with explicit "
                            "start argument of type none");
  if (resultTypes.empty() || !isa<NoneType>(resultTypes.front()))
    return parser.emitError(parser.getNameLoc(),
                            "graph signature must begin results with explicit "
                            "done protocol type none");

  if (visibilityAttr)
    result.addAttribute(getSymVisibilityAttrName(result.name), visibilityAttr);
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Attribute inputSegmentsAttr = result.attributes.get("input_segments");
  Attribute resultSegmentsAttr = result.attributes.get("result_segments");
  if (static_cast<bool>(inputSegmentsAttr) !=
      static_cast<bool>(resultSegmentsAttr))
    return parser.emitError(parser.getNameLoc(),
                            "input_segments and result_segments must be "
                            "specified together");

  SmallVector<OpAsmParser::Argument> appArguments(arguments.begin() + 1,
                                                   arguments.end());
  SmallVector<Type> appResults(resultTypes.begin() + 1, resultTypes.end());
  if (!inputSegmentsAttr) {
    auto inputSegments = defaultGraphSegments(appArguments.size());
    auto resultSegments = defaultGraphSegments(appResults.size());
    result.addAttribute("input_segments",
                        builder.getDenseI32ArrayAttr(inputSegments));
    result.addAttribute("result_segments",
                        builder.getDenseI32ArrayAttr(resultSegments));
  }

  SmallVector<Type> inputTypes;
  llvm::transform(appArguments, std::back_inserter(inputTypes),
                  [](const OpAsmParser::Argument &arg) { return arg.type; });
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(builder.getFunctionType(inputTypes,
                                                            appResults)));

  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> bodyArguments;
  bodyArguments.push_back(arguments.front());
  bodyArguments.append(appArguments.begin(), appArguments.end());
  if (parser.parseRegion(*body, bodyArguments,
                         /*enableNameShadowing=*/false))
    return failure();
  return success();
}

void GraphFuncOp::print(OpAsmPrinter &p) {
  if (auto vis = getSymVisibility())
    p << ' ' << *vis;
  p << ' ';
  p.printSymbolName(getSymName());

  Block *entry = getBody().empty() ? nullptr : &getBody().front();
  p << '(';
  if (entry) {
    p.printRegionArgument(entry->getArgument(0));
    for (BlockArgument argument : entry->getArguments().drop_front()) {
      p << ", ";
      p.printRegionArgument(argument);
    }
  } else {
    p.printType(NoneType::get(getContext()));
    for (Type type : getFunctionType().getInputs()) {
      p << ", ";
      p.printType(type);
    }
  }
  p << ") -> ";
  ArrayRef<Type> results = getFunctionType().getResults();
  if (results.empty()) {
    p.printType(NoneType::get(getContext()));
  } else {
    p << '(';
    p.printType(NoneType::get(getContext()));
    for (Type type : results) {
      p << ", ";
      p.printType(type);
    }
    p << ')';
  }

  SmallVector<StringRef, 6> elidedAttrs = {
      SymbolTable::getSymbolAttrName(), getFunctionTypeAttrName(),
      getSymVisibilityAttrName(),      getArgAttrsAttrName(),
      getResAttrsAttrName(),
  };
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

BlockArgument GraphFuncOp::getStart() {
  assert(!isExternal() && "external graph has no start argument");
  return getBody().front().getArgument(0);
}

ArrayRef<int32_t> GraphFuncOp::getInputSegmentSizes() {
  return getInputSegments();
}

ArrayRef<int32_t> GraphFuncOp::getResultSegmentSizes() {
  return getResultSegments();
}

GraphPortKind GraphFuncOp::getInputPortKind(unsigned index) {
  return graphPortKindAt(getInputSegmentSizes(), index);
}

GraphPortKind GraphFuncOp::getResultPortKind(unsigned index) {
  return graphPortKindAt(getResultSegmentSizes(), index);
}

// `dataflow.graph.func` carries the SpatialCore body of a leaf
// dataflow graph. Per the streaming spec (see
// docs/spec-dataflow-part-1-streaming.md), graph regions allow
// feedback edges, i.e., values produced by an op can be referenced
// by an earlier op in program order. Mirror the regional
// `dataflow.graph` op's region kind so feedback-edge ops like
// `dataflow.carry` (whose `next` operand is computed downstream)
// verify cleanly inside the function body.
RegionKind GraphFuncOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

// Whitelist for ops permitted directly or transitively inside a
// `dataflow.graph.func` body. The body is the SpatialCore image of a
// leaf graph; the SCF-to-DFG frontend pipeline only ever emits ops
// from a tightly bounded set: the dataflow streaming/control
// primitives, residual SCF envelopes (for control shapes the lowering
// has not collapsed yet), arith/math computation, the LLVM
// computation/intrinsic ops we lift through `loom-cc`, and the
// `builtin.unrealized_conversion_cast` bridge between `!llvm.ptr`
// memory ops and the dataflow load/store memref shape. Anything else
// indicates the body has been polluted with content that does not
// belong on SpatialCore (e.g., a nested function symbol definition or
// a direct `func.call` into ScalarCore) and should be rejected at
// verifier time so regressions trip immediately rather than silently
// passing through `loom-raise-opt`.
static bool isAllowedInDataflowGraphFuncBody(::mlir::Operation *op) {
  // Symbol-defining ops (modules, nested functions, globals, etc.)
  // never belong inside a graph.func body. The graph.func itself is
  // module-level; its body is leaf compute, not a place to anchor
  // further symbol definitions.
  if (op->hasTrait<::mlir::OpTrait::SymbolTable>())
    return false;
  if (::llvm::isa<::mlir::FunctionOpInterface>(op))
    return false;

  ::mlir::StringRef dialect =
      op->getDialect() ? op->getDialect()->getNamespace() : ::mlir::StringRef{};
  ::mlir::StringRef name = op->getName().getStringRef();

  // dataflow.* is broadly allowed for leaf streaming primitives,
  // control routing, and the graph.return terminator. A graph launch is
  // never valid in another graph. ThreadLaunchOp owns its containment
  // invariant so every thread/graph surface reports one canonical error.
  if (::llvm::isa<GraphLaunchOp>(op))
    return false;
  if (dialect == "dataflow")
    return true;
  // arith.*, math.*, and memref.* are entire-dialect allowlists: every
  // scalar computation primitive plus the memref load/store/alloc surface
  // the SCF input uses before the graph-memory pass converts it into
  // dataflow ops. The corpus lit tests feed graph.func bodies that
  // still carry raw `memref.load` / `memref.store` ops, so the
  // verifier needs to admit them mid-pipeline.
  if (dialect == "arith" || dialect == "math" || dialect == "memref")
    return true;
  // ub.poison shows up as a none-typed placeholder in some lowering
  // residuals.
  if (dialect == "ub")
    return true;
  // SCF envelopes the SCF-to-DFG layer has not collapsed yet survive
  // here so the verifier admits the IR mid-pipeline. Only
  // structured-control-flow ops are listed -- ops that escape the
  // body (e.g., scf.execute_region.yield) are terminator-traited and
  // covered separately below.
  if (dialect == "scf")
    return true;
  // Plain CFG ops (cf.br/cond_br/switch) round-trip through some
  // late-stage IRs.
  if (dialect == "cf")
    return true;
  // Permit unrealized_conversion_cast: the !llvm.ptr -> memref<?xT>
  // bridge between LLVM-load/store and dataflow load/store ops.
  if (op->getName().getStringRef() == "builtin.unrealized_conversion_cast")
    return true;
  // LLVM dialect: allow the computation/intrinsic surface that
  // `loom-cc` lifts onto graph.func bodies. We list the ops we know
  // appear (computation, conversion, compare, intrinsics, GEP,
  // load/store, the call/call_intrinsic forms used for
  // CMSIS-NN-style shared subroutines and ARM SIMD intrinsics) and
  // permit `llvm.intr.*` permissively for forward-compat with new
  // intrinsics.
  if (dialect == "llvm") {
    if (isSupportedArmInlineAsm(op))
      return true;
    if (name.starts_with("llvm.intr."))
      return true;
    if (name.starts_with("llvm.mlir."))
      return true;
    static const ::llvm::StringSet<> llvmAllowed = {
        // Memory and address arithmetic.
        "llvm.getelementptr", "llvm.load", "llvm.store",
        "llvm.alloca",        "llvm.bitcast",
        // Calls (computation and intrinsics).
        "llvm.call", "llvm.call_intrinsic",
        // Computation: integer arithmetic and bitwise.
        "llvm.add",  "llvm.sub",  "llvm.mul",  "llvm.sdiv", "llvm.udiv",
        "llvm.srem", "llvm.urem", "llvm.and",  "llvm.or",   "llvm.xor",
        "llvm.shl",  "llvm.lshr", "llvm.ashr",
        // Floating-point arithmetic.
        "llvm.fadd", "llvm.fsub", "llvm.fmul", "llvm.fdiv", "llvm.frem",
        "llvm.fneg",
        // Compare.
        "llvm.icmp", "llvm.fcmp",
        // Conversions.
        "llvm.trunc",  "llvm.zext",        "llvm.sext",       "llvm.fpext",
        "llvm.fptrunc", "llvm.uitofp",     "llvm.sitofp",     "llvm.fptoui",
        "llvm.fptosi", "llvm.ptrtoint",    "llvm.inttoptr",   "llvm.addrspacecast",
        // Element-wise / select / freeze.
        "llvm.select", "llvm.freeze",
        // Vector and aggregate ops.
        "llvm.extractelement", "llvm.insertelement", "llvm.extractvalue",
        "llvm.insertvalue",    "llvm.shufflevector",
    };
    return llvmAllowed.contains(name);
  }
  return false;
}

LogicalResult GraphFuncOp::verify() {
  if (auto vis = getSymVisibility()) {
    if (*vis != "private" && *vis != "")
      return emitOpError("sym_visibility must be 'private'; got \"")
             << *vis << "\"";
  }

  ArrayRef<Type> inputs = getFunctionType().getInputs();
  ArrayRef<Type> results = getFunctionType().getResults();

  auto verifySegments = [&](ArrayRef<int32_t> segments, StringRef name,
                            size_t count) -> LogicalResult {
    int64_t sum = 0;
    bool nonnegative = segments.size() == 3;
    for (int32_t size : segments) {
      nonnegative &= size >= 0;
      sum += size;
    }
    if (!nonnegative || sum != static_cast<int64_t>(count))
      return emitOpError()
             << name
             << " must contain exactly three nonnegative sizes whose sum ("
             << sum << ") matches the function "
             << (name == "input_segments" ? "input" : "result")
             << " count (" << count << ")";
    return success();
  };
  if (failed(verifySegments(getInputSegmentSizes(), "input_segments",
                            inputs.size())) ||
      failed(verifySegments(getResultSegmentSizes(), "result_segments",
                            results.size())))
    return failure();

  auto verifyTypes = [&](ArrayRef<Type> types, ArrayRef<int32_t> segments,
                         StringRef direction) -> LogicalResult {
    unsigned kindIndices[] = {0, 0, 0};
    for (auto [index, type] : llvm::enumerate(types)) {
      GraphPortKind kind = graphPortKindAt(segments, index);
      unsigned kindOrdinal = static_cast<unsigned>(kind);
      if (failed(verifyGraphPortType(getOperation(), type, kind, direction,
                                     kindIndices[kindOrdinal]++)))
        return failure();
    }
    return success();
  };
  if (failed(verifyTypes(inputs, getInputSegmentSizes(), "input")) ||
      failed(verifyTypes(results, getResultSegmentSizes(), "result")))
    return failure();

  if (isExternal())
    return success();

  Block &entry = getBody().front();
  if (entry.getNumArguments() != inputs.size() + 1)
    return emitOpError("entry block argument count (")
           << entry.getNumArguments()
           << ") must equal one start argument plus function_type input count ("
           << inputs.size() << ")";
  if (!isa<NoneType>(entry.getArgument(0).getType()))
    return emitOpError("entry block argument #0 must be start type none");
  for (size_t i = 0, e = inputs.size(); i < e; ++i) {
    Type ty = inputs[i];
    if (entry.getArgument(i + 1).getType() != ty)
      return emitOpError("entry block argument #")
             << (i + 1) << " type " << entry.getArgument(i + 1).getType()
             << " must match function_type input type " << ty;
  }

  // Body content whitelist: walk every op transitively contained in
  // the body and reject anything outside the SCF-to-DFG residual
  // surface. The walk uses pre-order so a disallowed parent (e.g.,
  // a nested `func.func` symbol definition) is reported before the
  // verifier dives into its body and complains about the inner
  // `func.return` instead. Each op's own verifier remains
  // responsible for checking its own arguments and semantics; this
  // loop only enforces the dialect-membership policy.
  ::mlir::WalkResult contentResult = getBody().walk<::mlir::WalkOrder::PreOrder>(
      [](::mlir::Operation *op) -> ::mlir::WalkResult {
        if (!isAllowedInDataflowGraphFuncBody(op)) {
          op->emitOpError(
              "is not allowed inside a dataflow.graph.func body; permitted "
              "ops are leaf dataflow.* primitives, arith.*, math.*, ub.*, scf.*, cf.*, "
              "builtin.unrealized_conversion_cast, and a curated llvm.* "
              "computation/conversion/intrinsic surface");
          return ::mlir::WalkResult::interrupt();
        }
        return ::mlir::WalkResult::advance();
      });
  if (contentResult.wasInterrupted())
    return failure();
  return success();
}

// CallableOpInterface methods come from extraClassDeclaration in
// DataflowOps.td. ArgAndResultAttrsOpInterface auto-generates the
// arg/res attr getters/setters/removers from the tablegen field
// declarations.

//===----------------------------------------------------------------------===//
// dataflow.graph.return
//===----------------------------------------------------------------------===//

namespace {

ParseResult parseGraphReturnSegmentBody(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None) ||
      parser.parseColon() || parser.parseTypeList(types) ||
      parser.parseRParen())
    return failure();
  if (operands.size() != types.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "operand count and type count differ");
  return success();
}

void printGraphReturnSegment(OpAsmPrinter &printer, StringRef name,
                             ValueRange values) {
  printer << name << '(';
  if (!values.empty()) {
    printer.printOperands(values);
    printer << " : ";
    llvm::interleaveComma(values, printer, [&](Value value) {
      printer.printType(value.getType());
    });
  }
  printer << ')';
}

} // namespace

ParseResult GraphReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> values;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> streams;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> memories;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> complete;
  SmallVector<Type, 4> valueTypes;
  SmallVector<Type, 4> streamTypes;
  SmallVector<Type, 4> memoryTypes;
  SmallVector<Type, 4> completeTypes;
  SMLoc operandLoc = parser.getCurrentLocation();

  if (succeeded(parser.parseOptionalKeyword("values"))) {
    if (parseGraphReturnSegmentBody(parser, values, valueTypes) ||
        parser.parseKeyword("streams") ||
        parseGraphReturnSegmentBody(parser, streams, streamTypes) ||
        parser.parseKeyword("memories") ||
        parseGraphReturnSegmentBody(parser, memories, memoryTypes) ||
        parser.parseKeyword("complete") ||
        parseGraphReturnSegmentBody(parser, complete, completeTypes))
      return failure();
  } else {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> compactOperands;
    SmallVector<Type, 4> compactTypes;
    OpAsmParser::UnresolvedOperand first;
    OptionalParseResult firstResult = parser.parseOptionalOperand(first);
    if (firstResult.has_value()) {
      if (failed(*firstResult))
        return failure();
      compactOperands.push_back(first);
      while (succeeded(parser.parseOptionalComma())) {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand))
          return failure();
        compactOperands.push_back(operand);
      }
      if (parser.parseColon() || parser.parseTypeList(compactTypes))
        return failure();
      if (compactOperands.size() != compactTypes.size())
        return parser.emitError(parser.getCurrentLocation(),
                                "operand count and type count differ");
    }
    if (!compactOperands.empty()) {
      complete.push_back(compactOperands.front());
      completeTypes.push_back(compactTypes.front());
      for (size_t i = 1, e = compactOperands.size(); i < e; ++i) {
        values.push_back(compactOperands[i]);
        valueTypes.push_back(compactTypes[i]);
      }
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.resolveOperands(values, valueTypes, operandLoc, result.operands) ||
      parser.resolveOperands(streams, streamTypes, operandLoc,
                             result.operands) ||
      parser.resolveOperands(memories, memoryTypes, operandLoc,
                             result.operands) ||
      parser.resolveOperands(complete, completeTypes, operandLoc,
                             result.operands))
    return failure();

  auto &properties = result.getOrAddProperties<GraphReturnOp::Properties>();
  properties.operandSegmentSizes = {
      static_cast<int32_t>(values.size()),
      static_cast<int32_t>(streams.size()),
      static_cast<int32_t>(memories.size()),
      static_cast<int32_t>(complete.size())};
  return success();
}

void GraphReturnOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  if (getStreams().empty() && getMemories().empty() &&
      getComplete().size() == 1) {
    SmallVector<Value, 4> operands{getComplete().front()};
    operands.append(getValues().begin(), getValues().end());
    printer.printOperands(operands);
    printer << " : ";
    llvm::interleaveComma(operands, printer, [&](Value value) {
      printer.printType(value.getType());
    });
  } else {
    printGraphReturnSegment(printer, "values", getValues());
    printer << ' ';
    printGraphReturnSegment(printer, "streams", getStreams());
    printer << ' ';
    printGraphReturnSegment(printer, "memories", getMemories());
    printer << ' ';
    printGraphReturnSegment(printer, "complete", getComplete());
  }
  printer.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
}

LogicalResult GraphReturnOp::verify() {
  auto parent = (*this)->getParentOfType<GraphFuncOp>();
  if (!parent)
    return emitOpError("must be inside a dataflow.graph.func op");
  if (getComplete().empty())
    return emitOpError("complete segment must not be empty");

  ArrayRef<int32_t> segments = parent.getResultSegmentSizes();
  ValueRange ranges[] = {getValues(), getStreams(), getMemories()};
  StringRef names[] = {"values", "streams", "memories"};
  for (unsigned segment = 0; segment < 3; ++segment) {
    if (ranges[segment].size() != static_cast<size_t>(segments[segment]))
      return emitOpError()
             << names[segment] << " segment count (" << ranges[segment].size()
             << ") must match parent result segment size ("
             << segments[segment] << ")";
  }

  ArrayRef<Type> expectedResults = parent.getFunctionType().getResults();
  unsigned resultIndex = 0;
  for (unsigned segment = 0; segment < 3; ++segment) {
    GraphPortKind kind = static_cast<GraphPortKind>(segment);
    for (auto [kindIndex, value] : llvm::enumerate(ranges[segment])) {
      Type expected = expectedResults[resultIndex++];
      Type actual = value.getType();
      if (actual != expected)
        return emitOpError()
               << graphPortKindName(kind) << " output #" << kindIndex
               << " type " << actual
               << " must match parent result type " << expected;
      if (failed(verifyGraphPortType(getOperation(), actual, kind, "output",
                                     kindIndex)))
        return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.graph.launch
//===----------------------------------------------------------------------===//

LogicalResult GraphLaunchOp::verifySymbolUses(SymbolTableCollection &symbols) {
  auto callee =
      symbols.lookupNearestSymbolFrom<GraphFuncOp>(*this, getCalleeAttr());
  if (!callee)
    return emitOpError("'")
           << getCallee()
           << "' does not reference a valid 'dataflow.graph.func' op";

  // Start is an explicit launch operand, not part of the callee FunctionType.
  ArrayRef<Type> calleeInputs = callee.getFunctionType().getInputs();
  if (calleeInputs.size() != getBodyOperands().size())
    return emitOpError("application operand count (")
           << getBodyOperands().size() << ") does not match callee input count ("
           << calleeInputs.size() << ")";
  for (size_t i = 0; i < calleeInputs.size(); ++i) {
    Type expected = calleeInputs[i];
    Type actual = getBodyOperands()[i].getType();
    if (actual != expected)
      return emitOpError("body operand #")
             << i << " type " << actual
             << " does not match callee input type " << expected;
  }

  // Done is an explicit launch result, not part of the callee FunctionType.
  ArrayRef<Type> calleeResults = callee.getFunctionType().getResults();
  if (calleeResults.size() != getResults().size())
    return emitOpError("application result count (")
           << getResults().size() << ") does not match callee result count ("
           << calleeResults.size() << ")";
  for (size_t i = 0; i < calleeResults.size(); ++i) {
    Type expected = calleeResults[i];
    Type actual = getResults()[i].getType();
    if (actual != expected)
      return emitOpError("body result #")
             << i << " type " << actual
             << " does not match callee result type " << expected;
  }

  // The op must appear inside a dataflow.thread definition's body.
  if (!(*this)->getParentOfType<ThreadOp>())
    return emitOpError(
        "must appear inside a dataflow.thread body (per spec section 5.5)");
  return success();
}

LogicalResult GraphLaunchOp::verify() { return success(); }
