#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace dataflow;

#define GET_OP_CLASSES
#include "Dataflow/IR/DataflowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Streaming Ops
//===----------------------------------------------------------------------===//

// dataflow.stream

ParseResult StreamOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands(3);
  if (parser.parseOperand(operands[0]) || parser.parseComma() ||
      parser.parseOperand(operands[1]) || parser.parseComma() ||
      parser.parseOperand(operands[2]) || parser.parseKeyword("step"))
    return failure();

  StringRef stepKeyword;
  SMLoc stepLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&stepKeyword))
    return failure();
  std::optional<StreamStepKind> stepKind = symbolizeStreamStepKind(stepKeyword);
  if (!stepKind)
    return parser.emitError(stepLoc, "expected dataflow.stream step kind "
                                     "'add', 'sub', 'mul', 'sdiv', 'udiv', "
                                     "'shl', 'ashr', or 'lshr', got '")
           << stepKeyword << "'";

  if (parser.parseKeyword("while"))
    return failure();
  StringRef predicateKeyword;
  SMLoc predicateLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&predicateKeyword))
    return failure();
  std::optional<arith::CmpIPredicate> predicate =
      arith::symbolizeCmpIPredicate(predicateKeyword);
  if (!predicate)
    return parser.emitError(predicateLoc,
                            "expected integer comparison predicate, got '")
           << predicateKeyword << "'";

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();
  Type valueType;
  if (parser.parseType(valueType) ||
      parser.resolveOperands(operands, valueType, result.operands))
    return failure();

  result.addAttribute("step_kind",
                      StreamStepKindAttr::get(parser.getContext(), *stepKind));
  result.addAttribute("predicate", arith::CmpIPredicateAttr::get(
                                       parser.getContext(), *predicate));
  result.addTypes({valueType, parser.getBuilder().getI1Type()});
  return success();
}

void StreamOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInit() << ", " << getLimit() << ", " << getStep()
          << " step " << stringifyStreamStepKind(getStepKind()) << " while "
          << arith::stringifyCmpIPredicate(getPredicate());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {"step_kind", "predicate"});
  printer << " : " << getIv().getType();
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
                                     Type expected, unsigned vecSize) {
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
             << "with 2 " << fanName << ", 'sel' must be 'i1', got " << selType;
  } else {
    if (!isIndex)
      return op->emitOpError() << "with more than 2 " << fanName
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
      return emitOpError("input #") << i << " type " << in.getType()
                                    << " must match output type " << outTy;
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
      return emitOpError("output #") << i << " type " << out.getType()
                                     << " must match input type " << inTy;
  }
  return success();
}

LogicalResult SyncOp::verify() {
  auto ins = getInputs();
  auto outs = getOutputs();
  if (ins.size() != outs.size())
    return emitOpError("number of inputs (")
           << ins.size() << ") must equal number of outputs (" << outs.size()
           << ")";
  for (unsigned i = 0, e = ins.size(); i < e; ++i) {
    if (ins[i].getType() != outs[i].getType())
      return emitOpError("input #")
             << i << " type " << ins[i].getType() << " must match output #" << i
             << " type " << outs[i].getType();
  }
  return success();
}
