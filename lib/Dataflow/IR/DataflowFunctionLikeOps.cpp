// Implementation of the symbol-bearing function-like ops used by
// SCF-to-DFG lowering: dataflow.thread (def),
// dataflow.thread.launch (async launcher), dataflow.thread.yield
// (terminator), dataflow.graph (def), dataflow.graph.launch (async
// launcher), dataflow.graph.return (terminator), and
// dataflow.graph.wait (stored-program retirement wait).

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
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
  return isa<MemRefType, UnrankedMemRefType>(type);
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
    if (!isGraphMemoryCapabilityType(type)) {
      if (DataflowDialect::isMemoryCapabilityType(type))
        return op->emitOpError()
               << "memory " << direction << " #" << kindIndex
               << " must be a memref capability, but got " << type;
      return op->emitOpError() << "memory " << direction << " #" << kindIndex
                               << " has non-capability type " << type;
    }
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

enum class ExtentExprKind { Unsupported, Constant, AddI, IndexCast };

static bool isScalarNonzeroSignlessIntegerOrIndex(Type type) {
  if (isa<IndexType>(type))
    return true;
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isSignless() && integerType.getWidth() != 0;
}

static unsigned getScalarIntegerBitWidth(Type type) {
  if (isa<IndexType>(type))
    return IndexType::kInternalStorageBitWidth;
  return cast<IntegerType>(type).getWidth();
}

static ExtentExprKind classifyExtentExpr(Operation *op) {
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
    return ExtentExprKind::Unsupported;

  if (isa<arith::ConstantOp>(op)) {
    if (op->getNumOperands() != 0 || op->getNumResults() != 1)
      return ExtentExprKind::Unsupported;
    Type resultType = op->getResult(0).getType();
    auto value = dyn_cast_or_null<IntegerAttr>(
        cast<arith::ConstantOp>(op).getProperties().value);
    if (!isScalarNonzeroSignlessIntegerOrIndex(resultType) || !value ||
        value.getType() != resultType)
      return ExtentExprKind::Unsupported;
    return ExtentExprKind::Constant;
  }

  if (isa<arith::AddIOp>(op)) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return ExtentExprKind::Unsupported;
    Type resultType = op->getResult(0).getType();
    if (!isScalarNonzeroSignlessIntegerOrIndex(resultType) ||
        op->getOperand(0).getType() != resultType ||
        op->getOperand(1).getType() != resultType)
      return ExtentExprKind::Unsupported;
    return ExtentExprKind::AddI;
  }

  if (isa<arith::IndexCastOp>(op)) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return ExtentExprKind::Unsupported;
    Type inputType = op->getOperand(0).getType();
    Type resultType = op->getResult(0).getType();
    auto inputInteger = dyn_cast<IntegerType>(inputType);
    auto resultInteger = dyn_cast<IntegerType>(resultType);
    bool validPair =
        (isa<IndexType>(inputType) && resultInteger &&
         resultInteger.isSignless() && resultInteger.getWidth() != 0) ||
        (inputInteger && inputInteger.isSignless() &&
         inputInteger.getWidth() != 0 && isa<IndexType>(resultType));
    if (!validPair)
      return ExtentExprKind::Unsupported;
    return ExtentExprKind::IndexCast;
  }

  return ExtentExprKind::Unsupported;
}

class ExtentConstantEvaluator {
public:
  Attribute evaluate(Value value) {
    if (auto cached = constants.find(value); cached != constants.end())
      return cached->second;

    SmallVector<Frame> stack;
    schedule(value, stack);
    while (!stack.empty()) {
      Frame &frame = stack.back();
      if (constants.contains(frame.value)) {
        active.erase(frame.value);
        stack.pop_back();
        continue;
      }

      Operation *definingOp = cast<OpResult>(frame.value).getOwner();
      bool descended = false;
      while (frame.nextOperand < definingOp->getNumOperands()) {
        Value operand = definingOp->getOperand(frame.nextOperand++);
        if (constants.contains(operand))
          continue;
        if (schedule(operand, stack)) {
          descended = true;
          break;
        }
      }
      if (descended)
        continue;

      evaluateOperation(definingOp, frame.kind);
      active.erase(frame.value);
      stack.pop_back();
    }

    return constants.lookup(value);
  }

private:
  struct Frame {
    Value value;
    unsigned nextOperand;
    ExtentExprKind kind;
  };

  void cacheUnknown(Operation *op) {
    for (Value result : op->getResults())
      constants.try_emplace(result, Attribute{});
  }

  bool schedule(Value value, SmallVectorImpl<Frame> &stack) {
    if (constants.contains(value))
      return false;

    auto result = dyn_cast<OpResult>(value);
    if (!result) {
      constants.try_emplace(value, Attribute{});
      return false;
    }

    Operation *definingOp = result.getOwner();
    ExtentExprKind kind = classifyExtentExpr(definingOp);
    if (kind == ExtentExprKind::Unsupported) {
      cacheUnknown(definingOp);
      return false;
    }
    if (!active.insert(value).second) {
      cacheUnknown(definingOp);
      return false;
    }

    stack.push_back({value, 0, kind});
    return true;
  }

  void evaluateOperation(Operation *op, ExtentExprKind kind) {
    if (kind == ExtentExprKind::Constant) {
      constants.try_emplace(
          op->getResult(0),
          cast<IntegerAttr>(cast<arith::ConstantOp>(op).getProperties().value));
      return;
    }

    auto operandConstant = [&](unsigned index) {
      return dyn_cast_or_null<IntegerAttr>(
          constants.lookup(op->getOperand(index)));
    };
    IntegerAttr input = operandConstant(0);
    if (!input || input.getType() != op->getOperand(0).getType()) {
      cacheUnknown(op);
      return;
    }

    Type resultType = op->getResult(0).getType();
    if (kind == ExtentExprKind::AddI) {
      IntegerAttr rhs = operandConstant(1);
      if (!rhs || rhs.getType() != op->getOperand(1).getType()) {
        cacheUnknown(op);
        return;
      }
      // Addition that breaks the no-overflow promise of its nsw/nuw flags
      // is poison, and poison is no extent at all. Unflagged addition wraps
      // and still folds.
      arith::IntegerOverflowFlags flags =
          cast<arith::AddIOp>(op).getOverflowFlags();
      bool poison = false;
      if (arith::bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nsw))
        (void)input.getValue().sadd_ov(rhs.getValue(), poison);
      if (!poison &&
          arith::bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nuw))
        (void)input.getValue().uadd_ov(rhs.getValue(), poison);
      if (poison) {
        cacheUnknown(op);
        return;
      }
      constants.try_emplace(
          op->getResult(0),
          IntegerAttr::get(resultType, input.getValue() + rhs.getValue()));
      return;
    }

    assert(kind == ExtentExprKind::IndexCast);
    unsigned resultBitWidth = getScalarIntegerBitWidth(resultType);
    constants.try_emplace(
        op->getResult(0),
        IntegerAttr::get(resultType,
                         input.getValue().sextOrTrunc(resultBitWidth)));
  }

  DenseMap<Value, Attribute> constants;
  DenseSet<Value> active;
};

// Segmented operand accessors index the operand list through the raw
// segment property, and the whole-module extent analysis below runs from a
// thread definition's verifier, before a launch's own ODS invariants hold.
// This is a readability predicate for that raw property, not a second
// verifier: it reports nothing, and a launch it rejects is left to its own
// ODS verification, which stays the only authority on segmentation.
static bool hasReadableOperandSegments(ThreadLaunchOp launch) {
  int64_t total = 0;
  for (int32_t size : launch.getProperties().operandSegmentSizes) {
    if (size < 0)
      return false;
    total += size;
  }
  return total == static_cast<int64_t>(launch->getNumOperands());
}

static LogicalResult verifyThreadLaunchExtents(ModuleOp module) {
  ExtentConstantEvaluator evaluator;
  WalkResult result =
      module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (op != module.getOperation() && op->hasTrait<OpTrait::SymbolTable>())
          return WalkResult::skip();

        auto launch = dyn_cast<ThreadLaunchOp>(op);
        if (!launch)
          return WalkResult::advance();
        // A launch whose segmentation cannot be read safely is diagnosed by
        // its own verification; this analysis just leaves it alone.
        if (!hasReadableOperandSegments(launch))
          return WalkResult::advance();
        for (auto [index, extent] :
             llvm::enumerate(launch.getGridUpperBounds())) {
          auto constant =
              dyn_cast_or_null<IntegerAttr>(evaluator.evaluate(extent));
          if (constant && constant.getValue().isNegative()) {
            launch.emitOpError("grid upper bound #")
                << index << " must be nonnegative";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
  return success(!result.wasInterrupted());
}

static bool ownsThreadLaunchExtentAnalysis(ThreadOp thread) {
  for (Operation *previous = thread->getPrevNode(); previous;
       previous = previous->getPrevNode())
    if (isa<ThreadOp>(previous))
      return false;
  return true;
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
                     FunctionType type, ThreadDomainAttr domain,
                     ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(getDomainAttrName(state.name), domain);
  buildFunctionLike<ThreadOp>(builder, state, name, type, attrs);
}

// Custom assembly format for dataflow.thread:
//
//   dataflow.thread [visibility] @sym domain(#dataflow.thread_domain<...>)
//                   (T0, T1, ...)
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

  if (parser.parseKeyword("domain") || parser.parseLParen())
    return failure();
  ThreadDomainAttr domain;
  if (parser.parseAttribute(domain) || parser.parseRParen())
    return failure();
  result.addAttribute(getDomainAttrName(result.name), domain);

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
  call_interface_impl::addArgAndResultAttrs(
      builder, result, arguments, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

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
  p << " domain(";
  p.printAttribute(getDomainAttr());
  p << ')';
  ArrayRef<Type> argTypes = getArgumentTypes();
  Block *entry = getBody().empty() ? nullptr : &getBody().front();
  call_interface_impl::printFunctionSignature(
      p, argTypes, getArgAttrsAttr(), /*isVariadic=*/false, ::mlir::TypeRange{},
      getResAttrsAttr(), &getBody(),
      /*printEmptyResult=*/false);

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
      getDomainAttrName(),
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
  if (!getSymVisibility() || *getSymVisibility() != "private")
    return emitOpError("requires explicit 'private' visibility");
  if (getFunctionType().getNumResults() != 0)
    return emitOpError("must not declare function results");

  if (getDomain().getKind() == ThreadDomainKind::DynamicWork) {
    uint64_t ordinal = *getDomain().getWorkItemArgOrdinal();
    ArrayRef<Type> inputs = getFunctionType().getInputs();
    if (ordinal >= inputs.size())
      return emitOpError("dynamic-work item argument ordinal ")
             << ordinal << " is out of bounds for " << inputs.size()
             << " thread inputs";
    for (auto [index, type] : llvm::enumerate(inputs))
      if (DataflowDialect::containsChannelOrThreadToken(type))
        return emitOpError("dynamic-work thread input #")
               << index << " must not contain a channel or thread token";
  }
  if (!ownsThreadLaunchExtentAnalysis(*this))
    return success();
  return verifyThreadLaunchExtents(cast<ModuleOp>((*this)->getParentOp()));
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

  size_t calleeRank = 0;
  if (!callee.isExternal()) {
    size_t entryArgumentCount = callee.getBody().front().getNumArguments();
    size_t requiredArgumentCount = calleeInputs.size() + 1;
    if (entryArgumentCount >= requiredArgumentCount)
      calleeRank = entryArgumentCount - requiredArgumentCount;
  }
  if (getGridUpperBounds().size() != calleeRank)
    return emitOpError("grid upper bound count (")
           << getGridUpperBounds().size() << ") must match callee rank ("
           << calleeRank << ")";
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

  for (auto [index, token] : llvm::enumerate(getAsyncDependencies()))
    if (!token.getDefiningOp<ThreadLaunchOp>())
      return emitOpError("operand #") << index
                                      << " must be produced directly by "
                                         "dataflow.thread.launch";
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
  if (!getSymVisibility() || *getSymVisibility() != "private")
    return emitOpError("requires explicit 'private' visibility");

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
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {getCalleeAttrName(), getSourceMapsAttrName(),
                                 "operandSegmentSizes", "resultSegmentSizes"});
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
    Type actual = memory.getType();
    Type formal = inputs[inputIndex];
    if (actual == formal) {
      ++inputIndex;
      continue;
    }

    auto pointer = dyn_cast<LLVM::LLVMPointerType>(actual);
    if (!pointer)
      return emitOpError("memory input #")
             << index << " type " << actual
             << " does not match callee payload type " << formal;

    auto view = dyn_cast<MemRefType>(formal);
    if (pointer.getAddressSpace() != 0)
      return emitOpError("memory input #")
             << index << " cannot bind pointer address space "
             << pointer.getAddressSpace() << " to canonical graph memref "
             << formal;
    if (!view || view.getRank() != 1 || !view.isDynamicDim(0) ||
        !view.getLayout().isIdentity() || view.getMemorySpace())
      return emitOpError("memory input #")
             << index << " pointer view target " << formal
             << " must be a rank-one dynamic identity-layout memref in the "
                "default memory space";
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

namespace {

// The single structural ownership rule for stored-program ops that name a
// graph invocation: exactly one enclosing dataflow.thread definition and no
// enclosing dataflow.graph body. Making ownership total and unambiguous here
// is what lets the finalized-program validator key one deterministic
// per-thread completion index off the innermost enclosing thread.
FailureOr<ThreadOp> getOwningThread(Operation *op) {
  ThreadOp owner;
  unsigned enclosingThreads = 0;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<GraphOp>(parent))
      return op->emitOpError(
          "must not appear inside a dataflow.graph definition");
    if (auto thread = dyn_cast<ThreadOp>(parent)) {
      if (enclosingThreads++ == 0)
        owner = thread;
    }
  }
  if (enclosingThreads != 1)
    return op->emitOpError("must be transitively contained by exactly one "
                           "dataflow.thread definition");
  return owner;
}

} // namespace

LogicalResult GraphLaunchOp::verify() {
  FailureOr<ThreadOp> thread = getOwningThread(getOperation());
  if (failed(thread))
    return failure();

  if ((*thread).getDomain().getKind() == ThreadDomainKind::DynamicWork &&
      (!getStreamInputs().empty() || !getStreamOutputs().empty()))
    return emitOpError(
        "dynamic-work thread must not bind graph stream ports to channels");

  ArrayAttr sourceMaps = getSourceMaps();
  if (sourceMaps.size() != getStreamInputs().size())
    return emitOpError("source_maps count (")
           << sourceMaps.size() << ") must match stream input binding count ("
           << getStreamInputs().size() << ')';

  Block &entry = thread->getBody().front();
  unsigned consumerRank =
      entry.getNumArguments() - thread->getFunctionType().getNumInputs() - 1;
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

//===----------------------------------------------------------------------===//
// dataflow.graph.wait
//===----------------------------------------------------------------------===//

LogicalResult GraphWaitOp::verify() {
  // Placement is the only locally decidable ownership fact the wait owns;
  // completion-frontier coverage is whole-program causal analysis owned by
  // the finalized-program validator.
  return success(succeeded(getOwningThread(getOperation())));
}
