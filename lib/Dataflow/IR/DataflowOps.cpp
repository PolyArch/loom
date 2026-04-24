#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace dataflow;

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Streaming Ops
//===----------------------------------------------------------------------===//

// dataflow.stream

static bool isValidStepOp(llvm::StringRef s) {
  return llvm::StringSwitch<bool>(s)
      .Case("+=", true)
      .Case("*=", true)
      .Case("-=", true)
      .Case("/=", true)
      .Case("<<=", true)
      .Case(">>=", true)
      .Default(false);
}

static bool isValidContCond(llvm::StringRef s) {
  return llvm::StringSwitch<bool>(s)
      .Case("<", true)
      .Case("<=", true)
      .Case(">", true)
      .Case(">=", true)
      .Case("!=", true)
      .Default(false);
}

LogicalResult StreamOp::verify() {
  if (!isValidStepOp(getStepOp()))
    return emitOpError("'step_op' must be one of '+=', '*=', '-=', '/=', "
                       "'<<=', '>>='; got \"")
           << getStepOp() << "\"";
  if (!isValidContCond(getContCond()))
    return emitOpError(
               "'cont_cond' must be one of '<', '<=', '>', '>=', '!='; got \"")
           << getContCond() << "\"";
  return success();
}

//===----------------------------------------------------------------------===//
// Control Ops
//===----------------------------------------------------------------------===//

// dataflow.constant

LogicalResult ConstantOp::verify() {
  auto typed = llvm::dyn_cast<TypedAttr>(getConstValue());
  if (!typed)
    return emitOpError("'const_value' must be a typed attribute");
  if (typed.getType() != getValue().getType())
    return emitOpError("'const_value' type ")
           << typed.getType() << " must match result type "
           << getValue().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.sync
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// dataflow.mux / dataflow.demux
//===----------------------------------------------------------------------===//

static LogicalResult verifySelAgainstArity(Operation *op, Type selType,
                                           size_t n, StringRef fanName) {
  if (n < 2)
    return op->emitOpError()
           << "requires at least 2 " << fanName << ", got " << n;
  bool isI1 = selType.isInteger(1);
  bool isIndex = isa<IndexType>(selType);
  if (n == 2) {
    if (!isI1)
      return op->emitOpError()
             << "with 2 " << fanName << ", 'sel' must be 'i1', got "
             << selType;
  } else {
    if (!isIndex)
      return op->emitOpError()
             << "with more than 2 " << fanName
             << ", 'sel' must be 'index', got " << selType;
  }
  return success();
}

LogicalResult MuxOp::verify() {
  if (failed(verifySelAgainstArity(getOperation(), getSel().getType(),
                                   getInputs().size(), "inputs")))
    return failure();
  Type outTy = getOutput().getType();
  for (auto [i, in] : llvm::enumerate(getInputs())) {
    if (in.getType() != outTy)
      return emitOpError("input #")
             << i << " type " << in.getType() << " must match output type "
             << outTy;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Region Ops
//===----------------------------------------------------------------------===//

// dataflow.graph

RegionKind GraphOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

// Assembly format:
//   dataflow.graph(%bb_arg0 = %outer0 : T0, %bb_arg1 = %outer1 : T1, ...)
//                 -> ResultTypes [attributes {...}] { body; dataflow.yield ... }
//
// Block arguments are declared inline with their corresponding outer SSA
// operand, removing the need for an explicit `^bb0(...)` header.
ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
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
  GraphOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void GraphOp::print(OpAsmPrinter &p) {
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

// Ops allowed directly inside a `dataflow.graph` region.
//
// Policy:
//   * dataflow.*                        : all
//   * arith.*                           : all except arith.constant
//                                         (use dataflow.constant instead)
//   * math.*                            : all
//   * ub.*                              : all (poison generators)
//   * llvm.alloca                       : explicitly allowed
//   * llvm.intr.*                       : all intrinsics
//   * llvm.<computation ops>            : arithmetic / bitwise / compare /
//                                         conversions / element-wise /
//                                         select / freeze
//
// Everything else is rejected.
static bool isAllowedInDataflowGraph(Operation *op) {
  if (isa<YieldOp>(op))
    return true;
  StringRef dialect =
      op->getDialect() ? op->getDialect()->getNamespace() : StringRef{};
  StringRef name = op->getName().getStringRef();

  if (dialect == "dataflow")
    return true;
  if (dialect == "arith")
    return name != "arith.constant";
  if (dialect == "math")
    return true;
  if (dialect == "ub")
    return true;

  if (dialect == "llvm") {
    if (name == "llvm.alloca")
      return true;
    if (name.starts_with("llvm.intr."))
      return true;
    static const llvm::StringSet<> compute = {
        "llvm.add",           "llvm.sub",          "llvm.mul",
        "llvm.sdiv",          "llvm.udiv",         "llvm.srem",
        "llvm.urem",          "llvm.fadd",         "llvm.fsub",
        "llvm.fmul",          "llvm.fdiv",         "llvm.frem",
        "llvm.fneg",          "llvm.and",          "llvm.or",
        "llvm.xor",           "llvm.shl",          "llvm.lshr",
        "llvm.ashr",          "llvm.icmp",         "llvm.fcmp",
        "llvm.bitcast",       "llvm.trunc",        "llvm.zext",
        "llvm.sext",          "llvm.fptrunc",      "llvm.fpext",
        "llvm.sitofp",        "llvm.uitofp",       "llvm.fptosi",
        "llvm.fptoui",        "llvm.ptrtoint",     "llvm.inttoptr",
        "llvm.addrspacecast", "llvm.select",       "llvm.freeze",
        "llvm.extractelement","llvm.insertelement","llvm.extractvalue",
        "llvm.insertvalue",   "llvm.shufflevector",
    };
    return compute.contains(name);
  }

  return false;
}

LogicalResult GraphOp::verify() {
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
  for (Operation &op : entry.without_terminator()) {
    if (!isAllowedInDataflowGraph(&op))
      return op.emitOpError(
                 "is not allowed inside dataflow.graph; permitted ops are "
                 "dataflow.*, arith.* (except arith.constant), math.*, ub.*, "
                 "llvm.alloca, llvm.intr.*, and llvm computation ops");
  }
  return success();
}

// dataflow.yield

LogicalResult YieldOp::verify() {
  auto graph = cast<GraphOp>((*this)->getParentOp());
  if (getValues().size() != graph.getOutputs().size())
    return emitOpError("yield value count (")
           << getValues().size() << ") must match parent graph result count ("
           << graph.getOutputs().size() << ")";
  for (auto [i, v] : llvm::enumerate(getValues())) {
    Type expected = graph.getOutputs()[i].getType();
    if (v.getType() != expected)
      return emitOpError("yield value #")
             << i << " type " << v.getType()
             << " must match parent graph result type " << expected;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// dataflow.mux / dataflow.demux (continued)
//===----------------------------------------------------------------------===//

LogicalResult DemuxOp::verify() {
  if (failed(verifySelAgainstArity(getOperation(), getSel().getType(),
                                   getOutputs().size(), "outputs")))
    return failure();
  Type inTy = getInput().getType();
  for (auto [i, out] : llvm::enumerate(getOutputs())) {
    if (out.getType() != inTy)
      return emitOpError("output #")
             << i << " type " << out.getType() << " must match input type "
             << inTy;
  }
  return success();
}

LogicalResult SyncOp::verify() {
  auto ins = getInputs();
  auto outs = getOutputs();
  if (ins.size() != outs.size())
    return emitOpError("number of inputs (")
           << ins.size() << ") must equal number of outputs ("
           << outs.size() << ")";
  for (unsigned i = 0, e = ins.size(); i < e; ++i) {
    if (ins[i].getType() != outs[i].getType())
      return emitOpError("input #")
             << i << " type " << ins[i].getType() << " must match output #"
             << i << " type " << outs[i].getType();
  }
  return success();
}
