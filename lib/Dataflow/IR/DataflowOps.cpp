#include "Dataflow/IR/DataflowOps.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
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

static FailureOr<VectorType> verifyDataVector(Operation *op, Type type) {
  auto vector = semantics::analyzeFixedRankOneDataVector(type);
  if (!vector)
    return op->emitOpError(llvm::toString(vector.takeError()));
  return *vector;
}

static LogicalResult verifyMaskVector(Operation *op, VectorType dataVector,
                                      Type type) {
  if (llvm::Error error = semantics::validateVectorMaskType(dataVector, type))
    return op->emitOpError(llvm::toString(std::move(error)));
  return success();
}

static LogicalResult verifyPackedWidth(Operation *op, VectorType vectorType,
                                       Type packedType) {
  auto vectorWidth = semantics::getFlattenedVectorBitWidth(vectorType);
  if (!vectorWidth)
    return op->emitOpError(llvm::toString(vectorWidth.takeError()));
  const unsigned packedWidth = cast<IntegerType>(packedType).getWidth();
  if (packedWidth != *vectorWidth)
    return op->emitOpError("packed integer width ")
           << packedWidth << " must equal vector bit width " << *vectorWidth;
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
  return verifyPackedWidth(getOperation(), getVector().getType(),
                           getPacked().getType());
}

LogicalResult UnpackOp::verify() {
  return verifyPackedWidth(getOperation(), getVector().getType(),
                           getPacked().getType());
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
// Memory Ops
//===----------------------------------------------------------------------===//

namespace {

struct ParsedMemoryAccessTypes {
  MemRefType memoryType;
  Type addressType;
  Type dataType;
};

ParseResult parseMemoryAccessTypes(OpAsmParser &parser,
                                   ParsedMemoryAccessTypes &types) {
  Type parsedMemoryType;
  if (parser.parseColon() || parser.parseType(parsedMemoryType))
    return failure();
  types.memoryType = dyn_cast<MemRefType>(parsedMemoryType);
  if (!types.memoryType)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected memref memory type");
  types.addressType = parser.getBuilder().getIndexType();
  types.dataType = types.memoryType.getElementType();
  if (failed(parser.parseOptionalComma()))
    return success();

  Type firstExplicitType;
  if (parser.parseType(firstExplicitType))
    return failure();
  if (failed(parser.parseOptionalComma())) {
    types.dataType = firstExplicitType;
    return success();
  }
  if (!isa<VectorType>(firstExplicitType))
    return parser.emitError(parser.getCurrentLocation(),
                            "first explicit type must be a vector address type");

  types.addressType = firstExplicitType;
  return parser.parseType(types.dataType);
}

FailureOr<VectorType> getParsedDataVector(OpAsmParser &parser, Type dataType) {
  auto vector = dyn_cast<VectorType>(dataType);
  if (!vector)
    return parser.emitError(
        parser.getCurrentLocation(),
        "masked memory access requires an explicit vector data type");
  return vector;
}

Type getMaskType(OpAsmParser &parser, VectorType dataVector) {
  return VectorType::get(dataVector.getShape(), parser.getBuilder().getI1Type(),
                         dataVector.getScalableDims());
}

LogicalResult verifyMemoryAccess(Operation *op, Value memory, Value address,
                                 Type dataType, Value mask,
                                 bool allowVectorAddress) {
  auto access = semantics::analyzeMemoryAccessType(
      cast<MemRefType>(memory.getType()), dataType, address.getType(),
      mask ? mask.getType() : Type{});
  if (!access)
    return op->emitOpError(llvm::toString(access.takeError()));
  if (access->isGather() && !allowVectorAddress)
    return op->emitOpError("vector address is unsupported for dataflow.store");
  return success();
}

bool hasExplicitMemoryDataType(Value memory, Type dataType) {
  return dataType != cast<MemRefType>(memory.getType()).getElementType();
}

} // namespace

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand memory;
  OpAsmParser::UnresolvedOperand address;
  OpAsmParser::UnresolvedOperand control;
  OpAsmParser::UnresolvedOperand mask;
  if (parser.parseOperand(memory) || parser.parseLSquare() ||
      parser.parseOperand(address) || parser.parseRSquare() ||
      parser.parseOperand(control))
    return failure();

  bool hasMask = succeeded(parser.parseOptionalKeyword("mask"));
  if (hasMask && parser.parseOperand(mask))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  ParsedMemoryAccessTypes types;
  if (failed(parseMemoryAccessTypes(parser, types)))
    return failure();

  if (parser.resolveOperand(memory, types.memoryType, result.operands) ||
      parser.resolveOperand(address, types.addressType, result.operands) ||
      parser.resolveOperand(control, parser.getBuilder().getNoneType(),
                            result.operands))
    return failure();
  if (hasMask) {
    FailureOr<VectorType> vector = getParsedDataVector(parser, types.dataType);
    if (failed(vector) ||
        parser.resolveOperand(mask, getMaskType(parser, *vector),
                              result.operands))
      return failure();
  }
  result.addTypes({types.dataType, parser.getBuilder().getNoneType()});
  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getMem() << '[' << getAddr() << "] " << getCtrl();
  if (getMask())
    printer << " mask " << getMask();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getMem().getType();
  if (isa<VectorType>(getAddr().getType()))
    printer << ", " << getAddr().getType();
  if (hasExplicitMemoryDataType(getMem(), getData().getType()))
    printer << ", " << getData().getType();
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type data,
                   Type done, Value memory, Value address, Value control) {
  state.addOperands({memory, address, control});
  state.addTypes({data, done});
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value memory,
                   Value address, Value control) {
  Type data = cast<MemRefType>(memory.getType()).getElementType();
  build(builder, state, data, builder.getNoneType(), memory, address, control);
}

LogicalResult LoadOp::verify() {
  return verifyMemoryAccess(getOperation(), getMem(), getAddr(),
                            getData().getType(), getMask(),
                            /*allowVectorAddress=*/true);
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand memory;
  OpAsmParser::UnresolvedOperand address;
  OpAsmParser::UnresolvedOperand data;
  OpAsmParser::UnresolvedOperand control;
  OpAsmParser::UnresolvedOperand mask;
  if (parser.parseOperand(memory) || parser.parseLSquare() ||
      parser.parseOperand(address) || parser.parseRSquare() ||
      parser.parseOperand(data) || parser.parseOperand(control))
    return failure();

  bool hasMask = succeeded(parser.parseOptionalKeyword("mask"));
  if (hasMask && parser.parseOperand(mask))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  ParsedMemoryAccessTypes types;
  if (failed(parseMemoryAccessTypes(parser, types)))
    return failure();

  if (parser.resolveOperand(memory, types.memoryType, result.operands) ||
      parser.resolveOperand(address, types.addressType, result.operands) ||
      parser.resolveOperand(data, types.dataType, result.operands) ||
      parser.resolveOperand(control, parser.getBuilder().getNoneType(),
                            result.operands))
    return failure();
  if (hasMask) {
    FailureOr<VectorType> vector = getParsedDataVector(parser, types.dataType);
    if (failed(vector) ||
        parser.resolveOperand(mask, getMaskType(parser, *vector),
                              result.operands))
      return failure();
  }
  result.addTypes(parser.getBuilder().getNoneType());
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getMem() << '[' << getAddr() << "] " << getData() << ' '
          << getCtrl();
  if (getMask())
    printer << " mask " << getMask();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getMem().getType();
  if (hasExplicitMemoryDataType(getMem(), getData().getType()))
    printer << ", " << getData().getType();
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Type done,
                    Value memory, Value address, Value data, Value control) {
  state.addOperands({memory, address, data, control});
  state.addTypes(done);
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value memory,
                    Value address, Value data, Value control) {
  build(builder, state, builder.getNoneType(), memory, address, data, control);
}

LogicalResult StoreOp::verify() {
  return verifyMemoryAccess(getOperation(), getMem(), getAddr(),
                            getData().getType(), getMask(),
                            /*allowVectorAddress=*/false);
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
