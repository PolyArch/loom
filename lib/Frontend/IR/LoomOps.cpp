#include "Frontend/IR/LoomOps.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace loom;

namespace {

LogicalResult verifyValueCarrier(Operation *op, Type type, StringRef label,
                                 unsigned index) {
  if (dataflow::DataflowDialect::containsMemoryCapability(type))
    return op->emitOpError() << label << " #" << index
                             << " contains memory capability type " << type;
  if (dataflow::DataflowDialect::containsChannelOrThreadToken(type))
    return op->emitOpError() << label << " #" << index
                             << " contains channel or thread token type "
                             << type;
  if (isa<NoneType>(type))
    return op->emitOpError() << label << " #" << index
                             << " must not use protocol type none";
  return success();
}

bool containsScalableVector(Type type) {
  return type
      .walk<WalkOrder::PreOrder>([](Type nested) -> WalkResult {
        auto vector = dyn_cast<VectorType>(nested);
        return vector && vector.isScalable() ? WalkResult::interrupt()
                                             : WalkResult::advance();
      })
      .wasInterrupted();
}

// The first scalable vector type the selected boundary holds, on its own
// operands and results or on any value its body defines, or a null type.
//
// A scalable vector is a legal S0 value: its element count is a runtime
// `vscale` multiple, which the fixed-rank Canonical Dataflow contract has no
// meaning for. A selected SpatialCore boundary is therefore where it has to be
// gone: a typed structured transform must first materialize its semantics as
// fixed-width chunks, loops, and masks or tails, and until then the candidate
// that selected this region cannot finalize. Rejecting it here neither moves
// the code elsewhere nor presents `vscale` as a constant.
Type findScalableVector(Operation *op, Region &body) {
  for (Type type :
       llvm::concat<const Type>(op->getOperandTypes(), op->getResultTypes()))
    if (containsScalableVector(type))
      return type;

  // Every value inside the region is either a block argument or an operation
  // result, so walking the blocks reaches all of them, nested regions included.
  Type found;
  body.walk([&](Block *block) {
    for (BlockArgument argument : block->getArguments())
      if (containsScalableVector(argument.getType())) {
        found = argument.getType();
        return WalkResult::interrupt();
      }
    for (Operation &nested : *block)
      for (Value result : nested.getResults())
        if (containsScalableVector(result.getType())) {
          found = result.getType();
          return WalkResult::interrupt();
        }
    return WalkResult::advance();
  });
  return found;
}

LogicalResult verifyMemoryCarrier(Operation *op, Type type, StringRef label,
                                  unsigned index) {
  if (dataflow::DataflowDialect::isMemoryCapabilityType(type))
    return success();
  return op->emitOpError() << label << " #" << index
                           << " has non-capability type " << type;
}

LogicalResult verifyChannelUses(BlockArgument argument, bool consumer) {
  for (OpOperand &use : argument.getUses()) {
    Operation *owner = use.getOwner();
    if (consumer && isa<dataflow::ChannelReceiveOp>(owner) &&
        use.getOperandNumber() == 0)
      continue;
    if (!consumer && isa<dataflow::ChannelSendOp>(owner) &&
        use.getOperandNumber() == 0)
      continue;
    return owner->emitOpError()
           << "must not use a stream " << (consumer ? "input" : "output")
           << " binding except through dataflow.channel."
           << (consumer ? "receive" : "send");
  }
  return success();
}

} // namespace

LogicalResult SpatialRegionOp::verify() {
  auto thread = (*this)->getParentOfType<dataflow::ThreadOp>();
  if (!thread)
    return emitOpError("must appear inside a dataflow.thread body");
  if ((*this)->getParentOfType<SpatialRegionOp>())
    return emitOpError("must not be nested in another loom.spatial_region");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one block");

  Block &entry = getBody().front();
  if (entry.getNumArguments() != getNumOperands())
    return emitOpError("entry block argument count (")
           << entry.getNumArguments() << ") must match operand count ("
           << getNumOperands() << ')';
  for (auto [index, pair] : llvm::enumerate(
           llvm::zip_equal(entry.getArguments(), getOperands()))) {
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError("entry block argument #")
             << index << " type " << std::get<0>(pair).getType()
             << " must match operand type " << std::get<1>(pair).getType();
  }

  for (auto [index, value] : llvm::enumerate(getValueInputs()))
    if (failed(verifyValueCarrier(getOperation(), value.getType(),
                                  "value input", index)))
      return failure();
  for (auto [index, value] : llvm::enumerate(getValueResults()))
    if (failed(verifyValueCarrier(getOperation(), value.getType(),
                                  "value result", index)))
      return failure();
  for (auto [index, value] : llvm::enumerate(getMemoryInputs()))
    if (failed(verifyMemoryCarrier(getOperation(), value.getType(),
                                   "memory input", index)))
      return failure();
  for (auto [index, value] : llvm::enumerate(getMemoryResults()))
    if (failed(verifyMemoryCarrier(getOperation(), value.getType(),
                                   "memory result", index)))
      return failure();

  ArrayAttr sourceMaps = getSourceMaps();
  if (sourceMaps.size() != getStreamInputs().size())
    return emitOpError("source_maps count (")
           << sourceMaps.size() << ") must match stream input count ("
           << getStreamInputs().size() << ')';
  unsigned consumerRank = thread.getBody().front().getNumArguments() -
                          thread.getFunctionType().getNumInputs() - 1;
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

  unsigned argumentIndex = getValueInputs().size();
  for (unsigned index = 0; index < getStreamInputs().size(); ++index)
    if (failed(verifyChannelUses(entry.getArgument(argumentIndex++), true)))
      return failure();
  argumentIndex += getMemoryInputs().size();
  for (unsigned index = 0; index < getStreamOutputs().size(); ++index)
    if (failed(verifyChannelUses(entry.getArgument(argumentIndex++), false)))
      return failure();

  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(getBody(), captures);
  if (!captures.empty())
    return emitOpError("must not capture values implicitly");

  if (Type scalable = findScalableVector(getOperation(), getBody()))
    return emitOpError("holds scalable vector type ")
           << scalable
           << ", which must be materialized as fixed-width chunks, loops, and "
              "masks or tails before this candidate can finalize";

  Operation *forbidden = nullptr;
  getBody().walk([&](Operation *nested) {
    if (isa<SpatialRegionOp, dataflow::GraphOp, dataflow::GraphLaunchOp,
            dataflow::ThreadOp, dataflow::ThreadLaunchOp>(nested) ||
        nested->getName().getStringRef() == "dataflow.channel.create") {
      forbidden = nested;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (forbidden)
    return forbidden->emitOpError(
        "must not appear inside loom.spatial_region");

  auto yield = dyn_cast<SpatialYieldOp>(entry.getTerminator());
  if (!yield)
    return emitOpError("must terminate with loom.spatial_yield");
  return success();
}

LogicalResult SpatialYieldOp::verify() {
  auto parent = (*this)->getParentOfType<SpatialRegionOp>();
  if (!parent)
    return emitOpError("must appear inside loom.spatial_region");
  if (getValues().size() != parent.getValueResults().size())
    return emitOpError("value count (")
           << getValues().size() << ") must match parent value result count ("
           << parent.getValueResults().size() << ')';
  if (getMemories().size() != parent.getMemoryResults().size())
    return emitOpError("memory count (")
           << getMemories().size()
           << ") must match parent memory result count ("
           << parent.getMemoryResults().size() << ')';
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(getValues(), parent.getValueResults())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError("value #")
             << index << " type " << std::get<0>(pair).getType()
             << " must match parent result type "
             << std::get<1>(pair).getType();
  for (auto [index, pair] : llvm::enumerate(
           llvm::zip_equal(getMemories(), parent.getMemoryResults())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError("memory #")
             << index << " type " << std::get<0>(pair).getType()
             << " must match parent result type "
             << std::get<1>(pair).getType();
  return success();
}

#define GET_OP_CLASSES
#include "Frontend/IR/LoomOps.cpp.inc"
