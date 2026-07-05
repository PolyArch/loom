#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace dataflow;

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.cpp.inc"

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

} // namespace

//===----------------------------------------------------------------------===//
// Streaming Ops
//===----------------------------------------------------------------------===//

// dataflow.stream

LogicalResult StreamOp::verify() {
  // Convert the textual attribute to the internal enum exactly once at the
  // verifier boundary; downstream code should consume the enum value rather
  // than recomparing strings.
  if (!symbolizeStepOp(getStepOp()))
    return emitOpError("'step_op' must be one of '+=', '-=', '*=', '/=', "
                       "'<<=', '>>='; got \"")
           << getStepOp() << "\"";
  if (!symbolizeContCond(getContCond()))
    return emitOpError(
               "'cont_cond' must be one of '<', '<=', '>', '>=', '!='; got \"")
           << getContCond() << "\"";
  return success();
}

// dataflow.parallelize / pack / unpack / serialize

static FailureOr<unsigned> verifyVecSizeAttr(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>("vec_size");
  if (!attr)
    return op->emitOpError("requires integer attribute 'vec_size'");
  int64_t value = attr.getInt();
  if (value < 1 || value > 64 || (value & (value - 1)) != 0)
    return op->emitOpError(
        "'vec_size' must be a power of two in the range [1, 64]");
  return static_cast<unsigned>(value);
}

static unsigned signlessIntegerWidth(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || !intType.isSignless())
    return 0;
  return intType.getWidth();
}

static LogicalResult verifyMaskWidth(Operation *op, Type maskType,
                                     unsigned vecSize) {
  unsigned maskWidth = signlessIntegerWidth(maskType);
  if (maskWidth != vecSize)
    return op->emitOpError("mask type width ")
           << maskWidth << " must match 'vec_size' " << vecSize;
  return success();
}

static LogicalResult verifyLaneTypes(Operation *op, TypeRange lanes,
                                     Type expected,
                                     unsigned vecSize) {
  if (lanes.size() != vecSize)
    return op->emitOpError("lane count ")
           << lanes.size() << " must match 'vec_size' " << vecSize;
  for (auto [i, laneType] : llvm::enumerate(lanes)) {
    if (laneType != expected)
      return op->emitOpError("lane #")
             << i << " type " << laneType << " must match lane #0 type "
             << expected;
  }
  return success();
}

static LogicalResult verifyPackedWidth(Operation *op, Type packedType,
                                       Type laneType, unsigned vecSize) {
  unsigned packedWidth = signlessIntegerWidth(packedType);
  unsigned laneWidth = signlessIntegerWidth(laneType);
  if (packedWidth != laneWidth * vecSize)
    return op->emitOpError("packed type width ")
           << packedWidth << " must equal lane width " << laneWidth
           << " times 'vec_size' " << vecSize;
  return success();
}

LogicalResult ParallelizeOp::verify() {
  FailureOr<unsigned> vecSize = verifyVecSizeAttr(getOperation());
  if (failed(vecSize))
    return failure();
  if (failed(verifyLaneTypes(getOperation(), getOutputs().getTypes(),
                             getData().getType(), *vecSize)))
    return failure();
  if (failed(verifyMaskWidth(getOperation(), getMask().getType(), *vecSize)))
    return failure();
  if (getStride() && getStride().getType() != getData().getType())
    return emitOpError("stride type ")
           << getStride().getType() << " must match data type "
           << getData().getType();
  return success();
}

LogicalResult PackOp::verify() {
  FailureOr<unsigned> vecSize = verifyVecSizeAttr(getOperation());
  if (failed(vecSize))
    return failure();
  if (getInputs().empty())
    return emitOpError("requires at least one lane input");
  Type laneType = getInputs().front().getType();
  if (failed(verifyLaneTypes(getOperation(), getInputs().getTypes(), laneType,
                             *vecSize)))
    return failure();
  if (failed(verifyMaskWidth(getOperation(), getMask().getType(), *vecSize)))
    return failure();
  return verifyPackedWidth(getOperation(), getPacked().getType(), laneType,
                           *vecSize);
}

LogicalResult UnpackOp::verify() {
  FailureOr<unsigned> vecSize = verifyVecSizeAttr(getOperation());
  if (failed(vecSize))
    return failure();
  if (getOutputs().empty())
    return emitOpError("requires at least one lane output");
  Type laneType = getOutputs().front().getType();
  if (failed(verifyLaneTypes(getOperation(), getOutputs().getTypes(), laneType,
                             *vecSize)))
    return failure();
  if (failed(verifyMaskWidth(getOperation(), getMask().getType(), *vecSize)))
    return failure();
  return verifyPackedWidth(getOperation(), getPacked().getType(), laneType,
                           *vecSize);
}

LogicalResult SerializeOp::verify() {
  FailureOr<unsigned> vecSize = verifyVecSizeAttr(getOperation());
  if (failed(vecSize))
    return failure();
  if (getInputs().empty())
    return emitOpError("requires at least one lane input");
  Type laneType = getInputs().front().getType();
  if (failed(verifyLaneTypes(getOperation(), getInputs().getTypes(), laneType,
                             *vecSize)))
    return failure();
  if (failed(verifyMaskWidth(getOperation(), getMask().getType(), *vecSize)))
    return failure();
  if (getData().getType() != laneType)
    return emitOpError("data result type ")
           << getData().getType() << " must match lane type " << laneType;
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

// dataflow.graph / dataflow.subgraph share assembly format and parser/printer:
//   <op>(%bb_arg0 = %outer0 : T0, %bb_arg1 = %outer1 : T1, ...)
//        -> ResultTypes [attributes {...}] { body; dataflow.yield ... }
//
// Block arguments are declared inline with their corresponding outer SSA
// operand, removing the need for an explicit `^bb0(...)` header.

template <typename OpT>
static ParseResult parseGraphLikeOp(OpAsmParser &parser,
                                    OperationState &result) {
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
  OpT::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

template <typename OpT>
static void printGraphLikeOp(OpAsmPrinter &p, OpT op) {
  p << '(';
  Block &entry = op.getBody().front();
  llvm::interleaveComma(
      llvm::zip(entry.getArguments(), op.getInputs()), p, [&](auto pair) {
        BlockArgument bb;
        Value outer;
        std::tie(bb, outer) = pair;
        p.printRegionArgument(bb, /*argAttrs=*/{}, /*omitType=*/true);
        p << " = " << outer << " : " << outer.getType();
      });
  p << ") -> ";
  auto rTypes = op.getResultTypes();
  if (rTypes.size() == 1) {
    p << rTypes.front();
  } else {
    p << '(';
    llvm::interleaveComma(rTypes, p);
    p << ')';
  }
  p.printOptionalAttrDictWithKeyword(op.getOperation()->getAttrs());
  p << ' ';
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

template <typename OpT>
static LogicalResult verifyGraphLikeStructure(OpT op) {
  Block &entry = op.getBody().front();
  if (entry.getNumArguments() != op.getInputs().size())
    return op.emitOpError("region entry block argument count (")
           << entry.getNumArguments() << ") must equal operand count ("
           << op.getInputs().size() << ")";
  for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
    if (arg.getType() != op.getInputs()[i].getType())
      return op.emitOpError("region entry block argument #")
             << i << " type " << arg.getType() << " must match operand type "
             << op.getInputs()[i].getType();
  }
  return success();
}

// dataflow.graph

RegionKind GraphOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGraphLikeOp<GraphOp>(parser, result);
}

void GraphOp::print(OpAsmPrinter &p) { printGraphLikeOp(p, *this); }

// dataflow.subgraph

RegionKind SubgraphOp::getRegionKind(unsigned /*index*/) {
  return RegionKind::Graph;
}

ParseResult SubgraphOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGraphLikeOp<SubgraphOp>(parser, result);
}

void SubgraphOp::print(OpAsmPrinter &p) { printGraphLikeOp(p, *this); }

// Ops allowed directly inside a `dataflow.graph` region.
//
// Policy:
//   * dataflow.*                        : all (including dataflow.subgraph),
//                                         except nested dataflow.graph (we
//                                         disallow graph-in-graph for clean
//                                         hierarchy analysis).
//   * arith.*                           : all except arith.constant
//                                         (use dataflow.constant instead).
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
  if (isa<GraphOp>(op))
    return false; // graph-in-graph is forbidden
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
    if (isSupportedArmInlineAsm(op))
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

// Ops allowed directly inside a `dataflow.subgraph` region.
//
// Policy:
//   * dataflow.yield                  : terminator
//   * any op named in fabric.op's allowlist (the canonical set of ops a
//     fabric tile can implement)
//
// Notably excluded: dataflow.graph, dataflow.subgraph, dataflow.load,
// dataflow.store, llvm.*, ub.poison, arith.constant.
static bool isAllowedInDataflowSubgraph(Operation *op) {
  if (isa<YieldOp>(op))
    return true;
  return fabric::isFabricOpSupported(op->getName().getStringRef());
}

LogicalResult GraphOp::verify() {
  if (failed(verifyGraphLikeStructure(*this)))
    return failure();
  for (Operation &op : getBody().front().without_terminator()) {
    if (isa<GraphOp>(op))
      return op.emitOpError(
          "dataflow.graph cannot be nested inside another dataflow.graph; use "
          "dataflow.subgraph for hierarchy");
    if (!isAllowedInDataflowGraph(&op))
      return op.emitOpError(
                 "is not allowed inside dataflow.graph; permitted ops are "
                 "dataflow.* (incl. dataflow.subgraph; not dataflow.graph), "
                 "arith.* (except arith.constant), math.*, ub.*, "
                 "llvm.alloca, llvm.intr.*, and llvm computation ops");
  }
  return success();
}

LogicalResult SubgraphOp::verify() {
  if (failed(verifyGraphLikeStructure(*this)))
    return failure();
  for (Operation &op : getBody().front().without_terminator()) {
    if (!isAllowedInDataflowSubgraph(&op))
      return op.emitOpError(
                 "is not allowed inside dataflow.subgraph; permitted ops are "
                 "those supported by fabric.op (and dataflow.yield)");
  }
  return success();
}

// dataflow.yield

LogicalResult YieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  TypeRange parentTypes;
  StringRef parentLabel;
  if (auto graph = dyn_cast<GraphOp>(parent)) {
    parentTypes = graph.getResultTypes();
    parentLabel = "graph";
  } else if (auto sg = dyn_cast<SubgraphOp>(parent)) {
    parentTypes = sg.getResultTypes();
    parentLabel = "subgraph";
  } else {
    return emitOpError("must be inside dataflow.graph or dataflow.subgraph");
  }
  if (getValues().size() != parentTypes.size())
    return emitOpError("yield value count (")
           << getValues().size() << ") must match parent " << parentLabel
           << " result count (" << parentTypes.size() << ")";
  for (auto [i, v] : llvm::enumerate(getValues())) {
    Type expected = parentTypes[i];
    if (v.getType() != expected)
      return emitOpError("yield value #")
             << i << " type " << v.getType() << " must match parent "
             << parentLabel << " result type " << expected;
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
