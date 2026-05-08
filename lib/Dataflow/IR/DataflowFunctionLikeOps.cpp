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
//                   ( `iv` `(` ivTypes... `)` )?
//                   attributes? body
//
// Function-signature args print just like func.func; the trailing
// `iv(...)` clause exposes any extra block arguments after the
// function inputs (per spec section 5.4.1). This is purely a custom
// surface; the underlying op state still tracks the entry block
// arguments directly.

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

  // Optional `iv (` extraArg : T, extraArg : T `)` clause.
  SmallVector<OpAsmParser::Argument> extraArgs;
  if (succeeded(parser.parseOptionalKeyword("iv"))) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      auto parseExtra = [&]() -> ParseResult {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg, /*allowType=*/true))
          return failure();
        extraArgs.push_back(arg);
        return success();
      };
      if (parseExtra())
        return failure();
      while (succeeded(parser.parseOptionalComma()))
        if (parseExtra())
          return failure();
      if (parser.parseRParen())
        return failure();
    }
  }

  if (visibilityAttr) {
    result.addAttribute(getSymVisibilityAttrName(result.name), visibilityAttr);
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> allArgs(arguments);
  for (auto &a : extraArgs)
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
  Block &entry = getBody().front();
  // Print function-signature arguments inline as the function's
  // argument list.
  p << '(';
  for (size_t i = 0, e = argTypes.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p.printRegionArgument(entry.getArgument(i));
  }
  p << ')';

  // Print optional `iv ( ... )` for any extra block arguments.
  if (entry.getNumArguments() > argTypes.size()) {
    p << " iv (";
    for (size_t i = argTypes.size(), e = entry.getNumArguments(); i < e; ++i) {
      if (i > argTypes.size())
        p << ", ";
      p.printRegionArgument(entry.getArgument(i));
    }
    p << ')';
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
