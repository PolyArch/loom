#include "GraphPointerAddressing.h"

#include "Dataflow/IR/GepAddressPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Error.h"

namespace loom::lowering {
namespace {

mlir::arith::IntegerOverflowFlags
integerFlags(mlir::LLVM::GEPNoWrapFlags flags) {
  using IntegerFlags = mlir::arith::IntegerOverflowFlags;
  using PointerFlags = mlir::LLVM::GEPNoWrapFlags;
  IntegerFlags result = IntegerFlags::none;
  if (mlir::LLVM::bitEnumContainsAny(flags, PointerFlags::nusw))
    result = result | IntegerFlags::nsw;
  if (mlir::LLVM::bitEnumContainsAny(flags, PointerFlags::nuw))
    result = result | IntegerFlags::nuw;
  return result;
}

mlir::LogicalResult normalize(mlir::LLVM::GEPOp op, dataflow::GraphOp graph) {
  auto plan = dataflow::semantics::projectGepAddressPlan(op, graph);
  if (!plan)
    return op.emitError("cannot normalize pointer address: ")
           << llvm::toString(plan.takeError());
  const unsigned addressBits = plan->pointerLayout.addressBits;
  if (op.getElemType().isInteger(8) && plan->terms.size() == 1 &&
      plan->terms.front().dynamicOperandOrdinal &&
      op.getDynamicIndices().front().getType().isInteger(addressBits))
    return mlir::success();

  mlir::OpBuilder builder(op);
  const mlir::Location location = op.getLoc();
  const auto offsetType = builder.getIntegerType(addressBits);
  const auto overflow = integerFlags(plan->noWrapFlags);
  auto constant = [&](const llvm::APInt &bits) -> mlir::Value {
    return mlir::arith::ConstantOp::create(
        builder, location,
        builder.getIntegerAttr(builder.getIntegerType(bits.getBitWidth()),
                               bits));
  };

  mlir::Value pointer = op.getBase();
  mlir::Value accumulated;
  for (const auto &term : plan->terms) {
    mlir::Value index = term.dynamicOperandOrdinal
                            ? op->getOperand(*term.dynamicOperandOrdinal)
                            : constant(term.constantIndex);
    const unsigned sourceBits =
        mlir::cast<mlir::IntegerType>(index.getType()).getWidth();
    if (sourceBits < addressBits) {
      index =
          mlir::arith::ExtSIOp::create(builder, location, offsetType, index);
    } else if (sourceBits > addressBits) {
      auto truncated =
          mlir::arith::TruncIOp::create(builder, location, offsetType, index);
      truncated.setOverflowFlags(overflow);
      index = truncated;
    }
    mlir::Value offset = index;
    if (!term.scale.isOne()) {
      auto scaled = mlir::arith::MulIOp::create(builder, location, index,
                                                constant(term.scale));
      scaled.setOverflowFlags(overflow);
      offset = scaled;
    }

    // GEP checks both each address step and the accumulated offset. Preserve
    // the latter's poison through the step operand without adding the prefix
    // twice. The subtraction is intentionally wrapping.
    mlir::Value step = offset;
    if (accumulated && overflow != mlir::arith::IntegerOverflowFlags::none) {
      auto sum =
          mlir::arith::AddIOp::create(builder, location, accumulated, offset);
      sum.setOverflowFlags(overflow);
      step = mlir::arith::SubIOp::create(builder, location, sum, accumulated);
      accumulated = sum;
    } else if (!accumulated) {
      accumulated = offset;
    }
    pointer = mlir::LLVM::GEPOp::create(
        builder, location, op.getType(), builder.getI8Type(), pointer,
        mlir::ValueRange{step}, plan->noWrapFlags);
  }
  if (pointer != op.getBase())
    pointer.getDefiningOp()->setDiscardableAttrs(
        op->getDiscardableAttrDictionary());
  op.getResult().replaceAllUsesWith(pointer);
  op.erase();
  return mlir::success();
}

} // namespace

mlir::LogicalResult normalizeGraphPointerAddresses(dataflow::GraphOp graph) {
  llvm::SmallVector<mlir::LLVM::GEPOp, 8> addresses;
  graph.walk([&](mlir::LLVM::GEPOp op) { addresses.push_back(op); });
  for (mlir::LLVM::GEPOp address : addresses)
    if (mlir::failed(normalize(address, graph)))
      return mlir::failure();
  return mlir::success();
}

} // namespace loom::lowering
