#include "Frontend/Analysis/PointerLoopProjection.h"
#include "Common/PointerLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

namespace loom::frontend::analysis {
namespace {

bool outsideLoop(mlir::Value value, mlir::scf::WhileOp loop) {
  if (auto *definition = value.getDefiningOp())
    return !loop->isAncestor(definition);
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  auto *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  return owner && owner != loop && !loop->isAncestor(owner);
}

bool hasNonzeroConstantStride(const ResolvedLinearMemoryAddress &address) {
  llvm::APInt stride(address.addressBitWidth, address.byteBias, true);
  for (const auto &term : address.terms) {
    llvm::APInt value;
    if (!mlir::matchPattern(term.index, mlir::m_ConstantInt(&value)))
      return false;
    stride += value.sextOrTrunc(address.addressBitWidth) *
              llvm::APInt(address.addressBitWidth, term.byteStride, true);
  }
  return !stride.isZero();
}

bool inboundsAddress(const ResolvedLinearMemoryAddress &address,
                     mlir::scf::WhileOp loop) {
  if (!outsideLoop(address.root, loop))
    return false;
  for (const auto &term : address.terms)
    if (!outsideLoop(term.index, loop))
      return false;
  for (auto *operation : address.gepsLeafToRoot) {
    auto gep = llvm::cast<mlir::LLVM::GEPOp>(operation);
    if (!mlir::LLVM::bitEnumContainsAny(
            gep.getNoWrapFlags(), mlir::LLVM::GEPNoWrapFlags::inboundsFlag))
      return false;
  }
  return true;
}

} // namespace

std::optional<PointerLoopTerminationProjection>
projectPointerLoopTermination(mlir::scf::WhileOp loop) {
  if (!loop || !loop.getBefore().hasOneBlock() ||
      !loop.getAfter().hasOneBlock())
    return std::nullopt;
  auto condition = loop.getConditionOp();
  auto yield = loop.getYieldOp();
  auto *before = loop.getBeforeBody();
  auto *after = loop.getAfterBody();
  const unsigned lanes = before->getNumArguments();
  if (loop.getInits().size() != lanes || condition.getArgs().size() != lanes ||
      after->getNumArguments() != lanes || yield.getNumOperands() != lanes)
    return std::nullopt;
  for (unsigned lane = 0; lane != lanes; ++lane)
    if (yield.getOperand(lane) != after->getArgument(lane))
      return std::nullopt;

  mlir::Value predicate = condition.getCondition();
  if (auto invert = predicate.getDefiningOp<mlir::arith::XOrIOp>()) {
    llvm::APInt bits;
    if (mlir::matchPattern(invert.getRhs(), mlir::m_ConstantInt(&bits)) &&
        bits.isOne())
      predicate = invert.getLhs();
    else if (mlir::matchPattern(invert.getLhs(), mlir::m_ConstantInt(&bits)) &&
             bits.isOne())
      predicate = invert.getRhs();
    else
      return std::nullopt;
  }
  auto compare = predicate.getDefiningOp<mlir::LLVM::ICmpOp>();
  if (!compare || compare->getParentRegion() != &loop.getBefore() ||
      (compare.getPredicate() != mlir::LLVM::ICmpPredicate::eq &&
       compare.getPredicate() != mlir::LLVM::ICmpPredicate::ne))
    return std::nullopt;

  for (unsigned lane = 0; lane != lanes; ++lane) {
    auto pointer = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(
        before->getArgument(lane).getType());
    if (!pointer)
      continue;
    auto update = condition.getArgs()[lane].getDefiningOp<mlir::LLVM::GEPOp>();
    if (!update || update->getParentRegion() != &loop.getBefore() ||
        update.getBase() != before->getArgument(lane) ||
        !mlir::LLVM::bitEnumContainsAny(
            update.getNoWrapFlags(), mlir::LLVM::GEPNoWrapFlags::inboundsFlag))
      continue;
    mlir::Value end;
    if (compare.getLhs() == update.getResult())
      end = compare.getRhs();
    else if (compare.getRhs() == update.getResult())
      end = compare.getLhs();
    if (!end || !outsideLoop(end, loop) || end.getType() != pointer)
      continue;
    auto layout = resolvePointerLayout(loop, pointer.getAddressSpace());
    if (!layout) {
      llvm::consumeError(layout.takeError());
      continue;
    }
    if (layout->kind != PointerLayoutKind::StableIntegral ||
        layout->addressBits != layout->representationBits)
      continue;
    auto byte = mlir::IntegerType::get(loop.getContext(), 8);
    auto beginAddress = resolveLinearPointerAddress(loop.getInits()[lane], byte);
    auto endAddress = resolveLinearPointerAddress(end, byte);
    auto stride = resolveLinearPointerAddress(update.getResult(), byte);
    if (!beginAddress || !endAddress || !stride ||
        beginAddress->root != endAddress->root ||
        beginAddress->addressBitWidth != layout->addressBits ||
        endAddress->addressBitWidth != layout->addressBits ||
        !inboundsAddress(*beginAddress, loop) ||
        !inboundsAddress(*endAddress, loop) ||
        stride->root != before->getArgument(lane) ||
        !hasNonzeroConstantStride(*stride))
      continue;
    return PointerLoopTerminationProjection{loop, lane, compare, update,
                                            std::move(*beginAddress),
                                            std::move(*endAddress)};
  }
  return std::nullopt;
}

} // namespace loom::frontend::analysis
