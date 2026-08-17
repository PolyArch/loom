#include "Frontend/Raising/CountedLoopProjection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace loom::raising {
namespace {

mlir::IntegerAttr integerConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  return constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                  : mlir::IntegerAttr{};
}

bool hasOrdinalIdentityFeedback(mlir::scf::WhileOp loop) {
  mlir::Block *before = loop.getBeforeBody();
  mlir::Block *after = loop.getAfterBody();
  mlir::scf::ConditionOp condition = loop.getConditionOp();
  mlir::scf::YieldOp yield = loop.getYieldOp();
  const unsigned width = before->getNumArguments();
  if (loop.getInits().size() != width || condition.getArgs().size() != width ||
      after->getNumArguments() != width || yield.getResults().size() != width ||
      loop->getNumResults() != width || !after->without_terminator().empty())
    return false;
  for (unsigned lane = 0; lane < width; ++lane) {
    mlir::Type type = before->getArgument(lane).getType();
    if (loop.getInits()[lane].getType() != type ||
        condition.getArgs()[lane].getType() != type ||
        after->getArgument(lane).getType() != type ||
        loop->getResult(lane).getType() != type ||
        yield.getResults()[lane] != after->getArgument(lane))
      return false;
  }
  return true;
}

} // namespace

std::optional<ExactPostTestedCountedLoopProjection>
projectExactPostTestedCountedLoop(mlir::scf::WhileOp loop) {
  if (!loop || !hasOrdinalIdentityFeedback(loop))
    return std::nullopt;

  mlir::Block *before = loop.getBeforeBody();
  mlir::scf::ConditionOp condition = loop.getConditionOp();
  auto compare = condition.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare->getParentRegion() != &loop.getBefore() ||
      compare.getPredicate() != mlir::arith::CmpIPredicate::ne)
    return std::nullopt;

  std::optional<ExactPostTestedCountedLoopProjection> projection;
  for (unsigned lane = 0; lane < before->getNumArguments(); ++lane) {
    auto integer =
        llvm::dyn_cast<mlir::IntegerType>(before->getArgument(lane).getType());
    if (!integer)
      continue;
    mlir::IntegerAttr lower = integerConstant(loop.getInits()[lane]);
    if (!lower || lower.getType() != integer)
      continue;

    auto update =
        condition.getArgs()[lane].getDefiningOp<mlir::arith::AddIOp>();
    if (!update || update->getParentRegion() != &loop.getBefore())
      continue;
    mlir::Value step;
    if (update.getLhs() == before->getArgument(lane))
      step = update.getRhs();
    else if (update.getRhs() == before->getArgument(lane))
      step = update.getLhs();
    else
      continue;
    mlir::IntegerAttr stepAttr = integerConstant(step);
    if (!stepAttr || stepAttr.getType() != integer)
      continue;

    mlir::Value upperBound;
    if (compare.getLhs() == update.getResult())
      upperBound = compare.getRhs();
    else if (compare.getRhs() == update.getResult())
      upperBound = compare.getLhs();
    else
      continue;
    mlir::IntegerAttr upper = integerConstant(upperBound);
    if (!upper || upper.getType() != integer)
      continue;

    const unsigned arithmeticWidth = integer.getWidth() + 1;
    llvm::APInt lowerValue = lower.getValue().sext(arithmeticWidth);
    llvm::APInt upperValue = upper.getValue().sext(arithmeticWidth);
    llvm::APInt stepValue = stepAttr.getValue().sext(arithmeticWidth);
    if (lowerValue.isNegative() || !stepValue.isStrictlyPositive() ||
        !lowerValue.slt(upperValue))
      continue;
    llvm::APInt distance = upperValue - lowerValue;
    if (!distance.srem(stepValue).isZero())
      continue;

    ExactPostTestedCountedLoopProjection candidate{
        loop,       lane,     loop.getInits()[lane],
        upperBound, step,     lowerValue,
        upperValue, stepValue};
    if (projection)
      return std::nullopt;
    projection = std::move(candidate);
  }
  return projection;
}

} // namespace loom::raising
