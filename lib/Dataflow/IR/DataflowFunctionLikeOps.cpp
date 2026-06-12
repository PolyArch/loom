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

using namespace mlir;
using namespace dataflow;

//===----------------------------------------------------------------------===//
// Common helpers used by both function-like ops.
//===----------------------------------------------------------------------===//

namespace {

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
  // Symbol visibility, when set, must be "private" in the first
  // milestone (the spec rejects "public" / "nested" until cross-module
  // linkage is specified). The entry-block / function-type
  // consistency check lives in verifyBody (which the
  // FunctionOpInterface default verifier calls).
  if (auto vis = getSymVisibility()) {
    if (*vis != "private" && *vis != "")
      return emitOpError(
                 "first-milestone sym_visibility must be 'private'; got \"")
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

// No verifier body needed; the assembly format is empty and the
// terminator status is trait-enforced.

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
  // The op result, if present, must be a thread token. Tablegen's
  // Optional<Dataflow_ThreadTokenType> already enforces this; nothing
  // additional to check at the smoke level.
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.graph.func (definition)
//===----------------------------------------------------------------------===//

void GraphFuncOp::build(OpBuilder &builder, OperationState &state,
                        StringRef name, FunctionType type,
                        ArrayRef<NamedAttribute> attrs) {
  buildFunctionLike<GraphFuncOp>(builder, state, name, type, attrs);
}

ParseResult GraphFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFunctionLike<GraphFuncOp>(parser, result);
}

void GraphFuncOp::print(OpAsmPrinter &p) { printFunctionLike(p, *this); }

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
  // control routing, and the graph.return terminator. Launch ops
  // belong in the host/thread orchestration layer, not inside a leaf
  // graph body.
  if (::llvm::isa<ThreadLaunchOp, GraphLaunchOp>(op))
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
      return emitOpError(
                 "first-milestone sym_visibility must be 'private'; got \"")
             << *vis << "\"";
  }

  if (isExternal())
    return success();

  // The function type must lead with `none` ctrl_in and `none`
  // done_out per spec section 5.5.1, and the entry block / return
  // operand list must mirror that.
  ArrayRef<Type> inputs = getFunctionType().getInputs();
  ArrayRef<Type> results = getFunctionType().getResults();
  if (inputs.empty() || !isa<NoneType>(inputs.front()))
    return emitOpError(
        "function_type inputs must lead with a `none` ctrl_in slot");
  if (results.empty() || !isa<NoneType>(results.front()))
    return emitOpError(
        "function_type results must lead with a `none` done_out slot");

  Block &entry = getBody().front();
  if (entry.getNumArguments() != inputs.size())
    return emitOpError("entry block argument count (")
           << entry.getNumArguments() << ") must equal function_type input count ("
           << inputs.size() << ")";
  for (size_t i = 0, e = inputs.size(); i < e; ++i) {
    Type ty = inputs[i];
    if (entry.getArgument(i).getType() != ty)
      return emitOpError("entry block argument #")
             << i << " type " << entry.getArgument(i).getType()
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

LogicalResult GraphReturnOp::verify() {
  auto parent = (*this)->getParentOfType<GraphFuncOp>();
  if (!parent)
    return emitOpError("must be inside a dataflow.graph.func op");
  ArrayRef<Type> results = parent.getFunctionType().getResults();
  if (getValues().size() != results.size())
    return emitOpError("return value count (")
           << getValues().size() << ") must match parent dataflow.graph.func "
           << "result count (" << results.size() << ")";
  for (size_t i = 0, e = results.size(); i < e; ++i) {
    Type expected = results[i];
    Type actual = getValues()[i].getType();
    if (actual != expected)
      return emitOpError("return value #")
             << i << " type " << actual
             << " must match parent dataflow.graph.func result type "
             << expected;
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

  // (none, type(bodyOperands)) must equal callee.function_type.inputs
  ArrayRef<Type> calleeInputs = callee.getFunctionType().getInputs();
  size_t expectedOperands = getBodyOperands().size() + 1;
  if (calleeInputs.size() != expectedOperands)
    return emitOpError("operand count (ctrl_in + body operands = ")
           << expectedOperands << ") does not match callee input count ("
           << calleeInputs.size() << ")";
  // Slot 0 is ctrl_in : none on both sides; tablegen already enforces
  // ctrl_in's type. Skip slot 0 and check the rest.
  for (size_t i = 1; i < calleeInputs.size(); ++i) {
    Type expected = calleeInputs[i];
    Type actual = getBodyOperands()[i - 1].getType();
    if (actual != expected)
      return emitOpError("body operand #")
             << (i - 1) << " type " << actual
             << " does not match callee input type " << expected;
  }

  // (none, type(results)) must equal callee.function_type.results
  ArrayRef<Type> calleeResults = callee.getFunctionType().getResults();
  size_t expectedResults = getResults().size() + 1;
  if (calleeResults.size() != expectedResults)
    return emitOpError("result count (done_out + body results = ")
           << expectedResults << ") does not match callee result count ("
           << calleeResults.size() << ")";
  for (size_t i = 1; i < calleeResults.size(); ++i) {
    Type expected = calleeResults[i];
    Type actual = getResults()[i - 1].getType();
    if (actual != expected)
      return emitOpError("body result #")
             << (i - 1) << " type " << actual
             << " does not match callee result type " << expected;
  }

  // The op must appear inside a dataflow.thread definition's body.
  if (!(*this)->getParentOfType<ThreadOp>())
    return emitOpError(
        "must appear inside a dataflow.thread body (per spec section 5.5)");
  return success();
}

LogicalResult GraphLaunchOp::verify() { return success(); }
