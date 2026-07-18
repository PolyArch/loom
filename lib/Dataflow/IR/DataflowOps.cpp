#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include <limits>

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

static bool isSupportedVectorElementType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() != 0;
  return isa<FloatType>(type);
}

static FailureOr<VectorType> verifyDataVector(Operation *op, Type type) {
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 1 || vector.isScalable())
    return op->emitOpError("data vector must be a fixed-size rank-1 vector");
  if (!isSupportedVectorElementType(vector.getElementType()))
    return op->emitOpError(
        "data vector element type must be a nonzero-width integer or "
        "floating-point type");
  return vector;
}

static LogicalResult verifyMaskVector(Operation *op, VectorType dataVector,
                                      Type type) {
  auto mask = dyn_cast<VectorType>(type);
  if (!mask || mask.getRank() != 1 || mask.isScalable())
    return op->emitOpError("mask vector must be a fixed-size rank-1 vector");
  if (!mask.getElementType().isInteger(1))
    return op->emitOpError("mask vector element type must be 'i1'");
  if (mask.getShape() != dataVector.getShape())
    return op->emitOpError("mask vector shape ")
           << mask << " must match data vector shape " << dataVector;
  return success();
}

static uint64_t getElementBitWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  return cast<FloatType>(type).getWidth();
}

static LogicalResult verifyPackedWidth(Operation *op, VectorType vector,
                                       Type packedType) {
  auto packed = dyn_cast<IntegerType>(packedType);
  if (!packed || !packed.isSignless())
    return op->emitOpError("packed type must be a signless integer");
  const uint64_t lanes = vector.getShape().front();
  const uint64_t elementWidth = getElementBitWidth(vector.getElementType());
  if (lanes > std::numeric_limits<unsigned>::max() / elementWidth)
    return op->emitOpError(
        "vector bit width exceeds the signless integer width limit");
  const unsigned vectorWidth = static_cast<unsigned>(lanes * elementWidth);
  if (packed.getWidth() != vectorWidth)
    return op->emitOpError("packed integer width ")
           << packed.getWidth() << " must equal vector bit width "
           << vectorWidth;
  return success();
}

LogicalResult ParallelizeOp::verify() {
  FailureOr<VectorType> vector =
      verifyDataVector(getOperation(), getVector().getType());
  if (failed(vector))
    return failure();
  if ((*vector).getElementType() != getData().getType())
    return emitOpError("data vector element type ")
           << (*vector).getElementType() << " must match scalar type "
           << getData().getType();
  return verifyMaskVector(getOperation(), *vector, getMask().getType());
}

LogicalResult PackOp::verify() {
  FailureOr<VectorType> vector =
      verifyDataVector(getOperation(), getVector().getType());
  if (failed(vector))
    return failure();
  return verifyPackedWidth(getOperation(), *vector, getPacked().getType());
}

LogicalResult UnpackOp::verify() {
  FailureOr<VectorType> vector =
      verifyDataVector(getOperation(), getVector().getType());
  if (failed(vector))
    return failure();
  return verifyPackedWidth(getOperation(), *vector, getPacked().getType());
}

LogicalResult SerializeOp::verify() {
  FailureOr<VectorType> vector =
      verifyDataVector(getOperation(), getVector().getType());
  if (failed(vector))
    return failure();
  if (failed(verifyMaskVector(getOperation(), *vector, getMask().getType())))
    return failure();
  if (getData().getType() != (*vector).getElementType())
    return emitOpError("scalar result type ")
           << getData().getType() << " must match data vector element type "
           << (*vector).getElementType();
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
