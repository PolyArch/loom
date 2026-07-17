// Implementation of the symbol-bearing function-like ops used by
// SCF-to-DFG lowering: dataflow.thread (def),
// dataflow.thread.launch (async launcher), dataflow.thread.yield
// (terminator), dataflow.graph (def), dataflow.graph.launch (async
// launcher), and dataflow.graph.return (terminator).

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <array>

using namespace mlir;
using namespace dataflow;

//===----------------------------------------------------------------------===//
// Common helpers used by both function-like ops.
//===----------------------------------------------------------------------===//

namespace {

static bool isGraphMemoryCapabilityType(Type type) {
  return DataflowDialect::isMemoryCapabilityType(type);
}

static bool containsGraphMemoryCapability(Type type) {
  return DataflowDialect::containsMemoryCapability(type);
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
      return op->emitOpError() << "memory " << direction << " #" << kindIndex
                               << " has non-capability type " << type;
    return success();
  }
  if (containsGraphMemoryCapability(type))
    return op->emitOpError()
           << graphPortKindName(kind) << " " << direction << " #" << kindIndex
           << " contains memory capability type " << type;
  if (kind == GraphPortKind::Value && isa<NoneType>(type))
    return op->emitOpError()
           << graphPortKindName(kind) << " " << direction << " #" << kindIndex
           << " must not use protocol type none";
  return success();
}

// Build a generic `func.func`-shaped op state for our function-like
// ops. Used by both ThreadOp and GraphOp.
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

template <typename Op> void printFunctionLike(OpAsmPrinter &p, Op op) {
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
  auto callee =
      symbols.lookupNearestSymbolFrom<ThreadOp>(*this, getCalleeAttr());
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
           << ") does not match callee input count (" << calleeInputs.size()
           << ")";
  for (size_t i = 0, e = calleeInputs.size(); i < e; ++i) {
    Type expected = calleeInputs[i];
    Type actual = getBodyOperands()[i].getType();
    if (actual != expected)
      return emitOpError("body operand #")
             << i << " type " << actual << " does not match callee input type "
             << expected;
  }
  return success();
}

LogicalResult ThreadLaunchOp::verify() {
  if ((*this)->getParentOfType<ThreadOp>() ||
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
      (*this)->getParentOfType<GraphOp>())
    return emitOpError(
        "must appear outside any dataflow.thread or dataflow.graph "
        "definition");
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.graph (definition)
//===----------------------------------------------------------------------===//

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs) {
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
  buildFunctionLike<GraphOp>(builder, state, name, type, normalizedAttrs);
}

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
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
  if (arguments.front().attrs && !arguments.front().attrs.empty())
    return parser.emitError(
        arguments.front().ssaName.location,
        "graph start argument is a protocol endpoint and cannot carry "
        "application interface attributes");
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
  SmallVector<Type> appResults(resultTypes.begin(), resultTypes.end());
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
  result.addAttribute(
      getFunctionTypeAttrName(result.name),
      TypeAttr::get(builder.getFunctionType(inputTypes, appResults)));
  call_interface_impl::addArgAndResultAttrs(
      builder, result, appArguments, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> bodyArguments;
  bodyArguments.push_back(arguments.front());
  bodyArguments.append(appArguments.begin(), appArguments.end());
  if (parser.parseRegion(*body, bodyArguments,
                         /*enableNameShadowing=*/false))
    return failure();
  return success();
}

void GraphOp::print(OpAsmPrinter &p) {
  if (auto vis = getSymVisibility())
    p << ' ' << *vis;
  p << ' ';
  p.printSymbolName(getSymName());

  SmallVector<Type> signatureInputs{NoneType::get(getContext())};
  signatureInputs.append(getFunctionType().getInputs().begin(),
                         getFunctionType().getInputs().end());
  ArrayAttr signatureArgAttrs;
  if (ArrayAttr appArgAttrs = getArgAttrsAttr()) {
    SmallVector<Attribute> attrs{
        DictionaryAttr::get(getContext(), ArrayRef<NamedAttribute>{})};
    attrs.append(appArgAttrs.begin(), appArgAttrs.end());
    signatureArgAttrs = ArrayAttr::get(getContext(), attrs);
  }
  call_interface_impl::printFunctionSignature(
      p, signatureInputs, signatureArgAttrs, /*isVariadic=*/false,
      getFunctionType().getResults(), getResAttrsAttr(), &getBody(),
      /*printEmptyResult=*/false);
  if (getFunctionType().getResults().empty())
    p << " -> ()";

  SmallVector<StringRef, 6> elidedAttrs = {
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

BlockArgument GraphOp::getStart() {
  assert(!isExternal() && "external graph has no start argument");
  return getBody().front().getArgument(0);
}

ArrayRef<int32_t> GraphOp::getInputSegmentSizes() { return getInputSegments(); }

ArrayRef<int32_t> GraphOp::getResultSegmentSizes() {
  return getResultSegments();
}

GraphPortKind GraphOp::getInputPortKind(unsigned index) {
  return graphPortKindAt(getInputSegmentSizes(), index);
}

GraphPortKind GraphOp::getResultPortKind(unsigned index) {
  return graphPortKindAt(getResultSegmentSizes(), index);
}

// Graph regions permit explicit feedback edges without CFG dominance.
RegionKind GraphOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

LogicalResult GraphOp::verify() {
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
             << (name == "input_segments" ? "input" : "result") << " count ("
             << count << ")";
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

  for (auto [index, type] : llvm::enumerate(inputs)) {
    if (!DataflowDialect::containsChannelOrThreadToken(type))
      continue;
    if (isa<ChannelType>(type))
      return emitOpError("function_type input #")
             << index << " must not be a dataflow channel type";
    return emitOpError("function_type input #")
           << index
           << " must not contain !dataflow.channel or "
              "!dataflow.thread_token";
  }
  for (auto [index, type] : llvm::enumerate(results)) {
    if (!DataflowDialect::containsChannelOrThreadToken(type))
      continue;
    if (isa<ChannelType>(type))
      return emitOpError("function_type result #")
             << index << " must not be a dataflow channel type";
    return emitOpError("function_type result #")
           << index
           << " must not contain !dataflow.channel or "
              "!dataflow.thread_token";
  }

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
  properties.operandSegmentSizes = {static_cast<int32_t>(values.size()),
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
  auto parent = (*this)->getParentOfType<GraphOp>();
  if (!parent)
    return emitOpError("must be inside a dataflow.graph op");
  if (getComplete().empty())
    return emitOpError("complete segment must not be empty");

  ArrayRef<int32_t> segments = parent.getResultSegmentSizes();
  ValueRange ranges[] = {getValues(), getStreams(), getMemories()};
  StringRef names[] = {"values", "streams", "memories"};
  for (unsigned segment = 0; segment < 3; ++segment) {
    if (ranges[segment].size() != static_cast<size_t>(segments[segment]))
      return emitOpError() << names[segment] << " segment count ("
                           << ranges[segment].size()
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
        return emitOpError() << graphPortKindName(kind) << " output #"
                             << kindIndex << " type " << actual
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

namespace {

ParseResult parseGraphLaunchOperandSegment(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  if (parser.parseKeyword(keyword) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();
  return success();
}

void printGraphLaunchOperandSegment(OpAsmPrinter &printer, StringRef keyword,
                                    ValueRange operands) {
  printer << ' ' << keyword << '(';
  printer.printOperands(operands);
  printer << ')';
}

ParseResult parseGraphLaunchStreamInputs(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<AffineMap> &sourceMaps) {
  if (parser.parseKeyword("stream_inputs") || parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    if (parser.parseOperand(operands.emplace_back()))
      return failure();
    if (failed(parser.parseOptionalKeyword("source_map")))
      return parser.emitError(
          parser.getCurrentLocation(),
          "expected 'source_map' after stream input binding");
    AffineMapAttr sourceMap;
    if (parser.parseAttribute(sourceMap))
      return failure();
    sourceMaps.push_back(sourceMap.getValue());
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

void printGraphLaunchStreamInputs(OpAsmPrinter &printer, ValueRange operands,
                                  ArrayAttr sourceMaps) {
  printer << " stream_inputs(";
  for (auto [index, operand] : llvm::enumerate(operands)) {
    if (index != 0)
      printer << ", ";
    printer.printOperand(operand);
    printer << " source_map ";
    printer.printAttributeWithoutType(
        llvm::cast<AffineMapAttr>(sourceMaps[index]));
  }
  printer << ')';
}

} // namespace

ParseResult GraphLaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr callee;
  if (parser.parseAttribute(callee, getCalleeAttrName(result.name),
                            result.attributes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> dependencies;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> valueInputs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> streamInputs;
  SmallVector<AffineMap, 4> sourceMaps;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> memoryInputs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> streamOutputs;
  SMLoc operandsLoc = parser.getCurrentLocation();
  if (parseGraphLaunchOperandSegment(parser, "deps", dependencies) ||
      parseGraphLaunchOperandSegment(parser, "values", valueInputs) ||
      parseGraphLaunchStreamInputs(parser, streamInputs, sourceMaps) ||
      parseGraphLaunchOperandSegment(parser, "memories", memoryInputs) ||
      parseGraphLaunchOperandSegment(parser, "stream_outputs", streamOutputs) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  FunctionType type;
  if (parser.parseColonType(type))
    return failure();
  size_t operandCount = dependencies.size() + valueInputs.size() +
                        streamInputs.size() + memoryInputs.size() +
                        streamOutputs.size();
  if (type.getNumInputs() != operandCount)
    return parser.emitError(parser.getCurrentLocation())
           << "operand count (" << operandCount
           << ") does not match function type input count ("
           << type.getNumInputs() << ")";

  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  operands.append(dependencies);
  operands.append(valueInputs);
  operands.append(streamInputs);
  operands.append(memoryInputs);
  operands.append(streamOutputs);
  if (parser.resolveOperands(operands, type.getInputs(), operandsLoc,
                             result.operands))
    return failure();

  ArrayRef<Type> resultTypes = type.getResults();
  if (resultTypes.empty() || !isa<NoneType>(resultTypes.back()))
    return parser.emitError(parser.getCurrentLocation(),
                            "graph launch requires a trailing none result");
  unsigned valueResultCount = 0;
  unsigned memoryResultCount = 0;
  bool sawMemory = false;
  for (Type resultType : resultTypes.drop_back()) {
    if (isGraphMemoryCapabilityType(resultType)) {
      sawMemory = true;
      ++memoryResultCount;
      continue;
    }
    if (sawMemory)
      return parser.emitError(
          parser.getCurrentLocation(),
          "value result must not follow a memory capability result");
    ++valueResultCount;
  }
  result.addTypes(resultTypes);

  auto &properties = result.getOrAddProperties<GraphLaunchOp::Properties>();
  properties.operandSegmentSizes = {static_cast<int32_t>(dependencies.size()),
                                    static_cast<int32_t>(valueInputs.size()),
                                    static_cast<int32_t>(streamInputs.size()),
                                    static_cast<int32_t>(memoryInputs.size()),
                                    static_cast<int32_t>(streamOutputs.size())};
  properties.resultSegmentSizes = {static_cast<int32_t>(valueResultCount),
                                   static_cast<int32_t>(memoryResultCount), 1};
  result.addAttribute("source_maps",
                      parser.getBuilder().getAffineMapArrayAttr(sourceMaps));
  return success();
}

void GraphLaunchOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttributeWithoutType(getCalleeAttr());
  printGraphLaunchOperandSegment(printer, "deps", getDependencies());
  printGraphLaunchOperandSegment(printer, "values", getValueInputs());
  printGraphLaunchStreamInputs(printer, getStreamInputs(), getSourceMaps());
  printGraphLaunchOperandSegment(printer, "memories", getMemoryInputs());
  printGraphLaunchOperandSegment(printer, "stream_outputs", getStreamOutputs());
  printer.printOptionalAttrDict(
      (*this)->getAttrs(),
      {getCalleeAttrName(), getSourceMapsAttrName(), "operandSegmentSizes",
       "resultSegmentSizes"});
  printer << " : ";
  printer.printFunctionalType(getOperandTypes(), getResultTypes());
}

LogicalResult GraphLaunchOp::verifySymbolUses(SymbolTableCollection &symbols) {
  auto callee =
      symbols.lookupNearestSymbolFrom<GraphOp>(*this, getCalleeAttr());
  if (!callee)
    return emitOpError("'")
           << getCallee() << "' does not reference a valid 'dataflow.graph' op";

  ArrayRef<int32_t> inputSegments = callee.getInputSegmentSizes();
  ArrayRef<int32_t> resultSegments = callee.getResultSegmentSizes();
  ArrayRef<Type> inputs = callee.getFunctionType().getInputs();
  ArrayRef<Type> results = callee.getFunctionType().getResults();

  auto verifyCount = [&](size_t actual, int32_t expected,
                         StringRef label) -> LogicalResult {
    if (actual == static_cast<size_t>(expected))
      return success();
    return emitOpError() << label << " count (" << actual
                         << ") does not match callee segment size (" << expected
                         << ")";
  };
  if (failed(verifyCount(getValueInputs().size(), inputSegments[0],
                         "value input")) ||
      failed(verifyCount(getStreamInputs().size(), inputSegments[1],
                         "stream input binding")) ||
      failed(verifyCount(getMemoryInputs().size(), inputSegments[2],
                         "memory input")) ||
      failed(verifyCount(getValueResults().size(), resultSegments[0],
                         "value result")) ||
      failed(verifyCount(getStreamOutputs().size(), resultSegments[1],
                         "stream output binding")) ||
      failed(verifyCount(getMemoryResults().size(), resultSegments[2],
                         "memory result")))
    return failure();

  unsigned inputIndex = 0;
  for (auto [index, value] : llvm::enumerate(getValueInputs())) {
    if (value.getType() != inputs[inputIndex])
      return emitOpError("value input #")
             << index << " type " << value.getType()
             << " does not match callee payload type " << inputs[inputIndex];
    ++inputIndex;
  }
  for (auto [index, channel] : llvm::enumerate(getStreamInputs())) {
    Type payload = cast<ChannelType>(channel.getType()).getElementType();
    if (payload != inputs[inputIndex])
      return emitOpError("stream input binding #")
             << index << " payload type " << payload
             << " does not match callee payload type " << inputs[inputIndex];
    ++inputIndex;
  }
  for (auto [index, memory] : llvm::enumerate(getMemoryInputs())) {
    if (memory.getType() != inputs[inputIndex])
      return emitOpError("memory input #")
             << index << " type " << memory.getType()
             << " does not match callee payload type " << inputs[inputIndex];
    ++inputIndex;
  }

  unsigned resultIndex = 0;
  for (auto [index, value] : llvm::enumerate(getValueResults())) {
    if (value.getType() != results[resultIndex])
      return emitOpError("value result #")
             << index << " type " << value.getType()
             << " does not match callee payload type " << results[resultIndex];
    ++resultIndex;
  }
  for (auto [index, channel] : llvm::enumerate(getStreamOutputs())) {
    Type payload = cast<ChannelType>(channel.getType()).getElementType();
    if (payload != results[resultIndex])
      return emitOpError("stream output binding #")
             << index << " payload type " << payload
             << " does not match callee payload type " << results[resultIndex];
    ++resultIndex;
  }
  for (auto [index, memory] : llvm::enumerate(getMemoryResults())) {
    if (memory.getType() != results[resultIndex])
      return emitOpError("memory result #")
             << index << " type " << memory.getType()
             << " does not match callee payload type " << results[resultIndex];
    ++resultIndex;
  }

  llvm::SmallDenseSet<Value, 4> producerBindings;
  for (Value channel : getStreamOutputs()) {
    if (!producerBindings.insert(channel).second)
      return emitOpError(
          "the same channel cannot bind more than one stream output port");
  }
  return success();
}

LogicalResult GraphLaunchOp::verify() {
  auto thread = (*this)->getParentOfType<ThreadOp>();
  if (!thread)
    return emitOpError("must appear inside a dataflow.thread body");

  ArrayAttr sourceMaps = getSourceMaps();
  if (sourceMaps.size() != getStreamInputs().size())
    return emitOpError("source_maps count (")
           << sourceMaps.size() << ") must match stream input binding count ("
           << getStreamInputs().size() << ')';

  Block &entry = thread.getBody().front();
  unsigned consumerRank =
      entry.getNumArguments() - thread.getFunctionType().getNumInputs() - 1;
  for (auto [index, attr] : llvm::enumerate(sourceMaps)) {
    AffineMap map = cast<AffineMapAttr>(attr).getValue();
    if (map.getNumDims() != consumerRank)
      return emitOpError("stream input source_map #")
             << index << " has " << map.getNumDims()
             << " dimensions but consumer thread domain has rank "
             << consumerRank;
    if (map.getNumSymbols() != 0)
      return emitOpError("stream input source_map #")
             << index << " must not contain symbols";
  }
  return success();
}
