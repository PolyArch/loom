#include "StructuredScheduleInternal.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_schedule_invalid: " + message);
}

bool operationTouchesVector(mlir::Operation *operation) {
  const auto isVector = [](mlir::Type type) {
    return llvm::isa<mlir::VectorType>(type);
  };
  return llvm::any_of(operation->getOperandTypes(), isVector) ||
         llvm::any_of(operation->getResultTypes(), isVector);
}

llvm::SmallVector<mlir::Operation *>
verificationClosure(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> result;
  llvm::SmallVector<mlir::Operation *> pending;
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  root->walk([&](mlir::Operation *operation) {
    if (seen.insert(operation).second) {
      result.push_back(operation);
      pending.push_back(operation);
    }
  });
  while (!pending.empty()) {
    mlir::Operation *operation = pending.pop_back_val();
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      if (definition && operationTouchesVector(definition) &&
          seen.insert(definition).second) {
        result.push_back(definition);
        pending.push_back(definition);
      }
    }
    for (mlir::Value resultValue : operation->getResults()) {
      for (mlir::Operation *user : resultValue.getUsers()) {
        if (seen.insert(user).second) {
          result.push_back(user);
          pending.push_back(user);
        }
      }
    }
  }
  return result;
}

llvm::Expected<mlir::Operation *>
resolveScheduledLoop(const ExactStructuredScopView &source,
                     mlir::ModuleOp materialized) {
  mlir::Operation *owner =
      mlir::SymbolTable::lookupSymbolIn(materialized, source.ownerSymbol);
  if (!owner)
    return invalid("materialized vector child lost its exact symbol owner");
  mlir::Operation *result = nullptr;
  std::uint64_t ordinal = 0;
  owner->walk([&](mlir::Operation *operation) {
    if (!llvm::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(operation))
      return mlir::WalkResult::advance();
    if (ordinal++ == source.loopOrdinalInOwner) {
      result = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!result)
    return invalid("materialized vector child lost its exact loop image");
  return result;
}

std::optional<std::uint64_t> localBoundaryArgument(mlir::Value value,
                                                   mlir::Operation *loop) {
  for (unsigned depth = 0; depth != 16; ++depth) {
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != loop->getBlock())
        return std::nullopt;
      return argument.getArgNumber();
    }
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    if (llvm::isa<mlir::memref::AssumeAlignmentOp>(result.getOwner())) {
      value = result.getOwner()->getOperand(0);
      continue;
    }
    auto distinct =
        llvm::dyn_cast<mlir::memref::DistinctObjectsOp>(result.getOwner());
    if (!distinct || result.getResultNumber() >= distinct.getNumOperands())
      return std::nullopt;
    value = distinct.getOperand(result.getResultNumber());
  }
  return std::nullopt;
}

mlir::Value inductionVariable(mlir::Operation *loop) {
  if (auto affine = llvm::dyn_cast<mlir::affine::AffineForOp>(loop))
    return affine.getInductionVar();
  if (auto scf = llvm::dyn_cast<mlir::scf::ForOp>(loop))
    return scf.getInductionVar();
  return {};
}

bool hasExactLoopDomain(mlir::Operation *loop, std::uint64_t upperBound,
                        std::uint64_t step) {
  constexpr std::uint64_t maximumSignedValue =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (upperBound > maximumSignedValue || step > maximumSignedValue)
    return false;
  const std::int64_t expectedUpper = static_cast<std::int64_t>(upperBound);
  const std::int64_t expectedStep = static_cast<std::int64_t>(step);
  if (auto affine = llvm::dyn_cast<mlir::affine::AffineForOp>(loop))
    return affine.hasConstantLowerBound() &&
           affine.getConstantLowerBound() == 0 &&
           affine.hasConstantUpperBound() &&
           affine.getConstantUpperBound() == expectedUpper &&
           affine.getStepAsInt() == expectedStep;
  auto scf = llvm::dyn_cast<mlir::scf::ForOp>(loop);
  if (!scf)
    return false;
  return mlir::getConstantIntValue(scf.getLowerBound()) == 0 &&
         mlir::getConstantIntValue(scf.getUpperBound()) == expectedUpper &&
         mlir::getConstantIntValue(scf.getStep()) == expectedStep;
}

bool allInBounds(mlir::ArrayAttr attribute) {
  return attribute && llvm::all_of(attribute, [](mlir::Attribute value) {
           return llvm::cast<mlir::BoolAttr>(value).getValue();
         });
}

bool payloadCorresponds(const dataflow::SemanticPayload &source,
                        const dataflow::SemanticPayload &child) {
  if (source == child)
    return true;
  auto *sourceConstant = std::get_if<dataflow::ConstantValuePayload>(&source);
  auto *childConstant = std::get_if<dataflow::ConstantValuePayload>(&child);
  if (!sourceConstant || !childConstant)
    return false;
  auto elements = llvm::dyn_cast<mlir::DenseElementsAttr>(childConstant->value);
  return elements && elements.isSplat() &&
         elements.getSplatValue<mlir::Attribute>() == sourceConstant->value;
}

bool vectorShapeMatches(mlir::Operation *operation, std::uint64_t factor) {
  const auto matches = [&](mlir::Type type) {
    auto vector = llvm::dyn_cast<mlir::VectorType>(type);
    return !vector ||
           (!vector.isScalable() && vector.getRank() == 1 &&
            static_cast<std::uint64_t>(vector.getDimSize(0)) == factor);
  };
  return llvm::all_of(operation->getOperandTypes(), matches) &&
         llvm::all_of(operation->getResultTypes(), matches);
}

bool isLoopIterArgument(mlir::Value value, mlir::Operation *loop) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument &&
         argument.getOwner() ==
             (llvm::isa<mlir::affine::AffineForOp>(loop)
                  ? llvm::cast<mlir::affine::AffineForOp>(loop).getBody()
                  : llvm::cast<mlir::scf::ForOp>(loop).getBody()) &&
         argument.getArgNumber() != 0;
}

bool isTransferPaddingProducer(mlir::Operation *operation) {
  if (operation->getNumOperands() != 0 || operation->getNumResults() != 1 ||
      operation->getResult(0).use_empty())
    return false;
  return llvm::all_of(
      operation->getResult(0).getUsers(), [&](mlir::Operation *user) {
        auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(user);
        return read && read.getPadding() == operation->getResult(0);
      });
}

bool isExactTailBoundProducer(mlir::Operation *operation, mlir::Value induction,
                              mlir::Value tailMask, std::uint64_t tripCount) {
  auto apply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(operation);
  auto mask = tailMask.getDefiningOp<mlir::vector::CreateMaskOp>();
  if (!apply || !mask || apply.getMapOperands().size() != 1 ||
      apply.getMapOperands().front() != induction ||
      !apply.getResult().hasOneUse() ||
      *apply.getResult().user_begin() != mask || mask->getNumOperands() != 1 ||
      mask->getOperand(0) != apply.getResult())
    return false;
  mlir::AffineExpr remaining =
      mlir::getAffineConstantExpr(static_cast<std::int64_t>(tripCount),
                                  operation->getContext()) -
      mlir::getAffineDimExpr(0, operation->getContext());
  return apply.getAffineMap() ==
         mlir::AffineMap::get(1, 0, remaining, operation->getContext());
}

bool isConstantInt(mlir::Value value, std::int64_t expected) {
  return mlir::getConstantIntValue(value) == expected;
}

bool collectExactLoweredTailMask(
    mlir::Value mask, mlir::Value induction, std::uint64_t tripCount,
    std::uint64_t factor, llvm::SmallPtrSetImpl<mlir::Operation *> &support) {
  if (tripCount > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max()) ||
      factor >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return false;

  mlir::Value bound;
  mlir::Value chain = mask;
  for (std::uint64_t reverseLane = factor; reverseLane != 0; --reverseLane) {
    const std::int64_t lane = static_cast<std::int64_t>(reverseLane - 1);
    auto insert = chain.getDefiningOp<mlir::vector::InsertOp>();
    if (!insert || !insert.getDynamicPosition().empty() ||
        insert.getStaticPosition() != llvm::ArrayRef<std::int64_t>{lane})
      return false;
    auto compare =
        insert.getValueToStore().getDefiningOp<mlir::arith::CmpIOp>();
    if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::sgt ||
        !isConstantInt(compare.getRhs(), lane))
      return false;
    if (bound && compare.getLhs() != bound)
      return false;
    bound = compare.getLhs();
    support.insert(insert);
    support.insert(compare);
    support.insert(compare.getRhs().getDefiningOp());
    chain = insert.getDest();
  }

  auto falseConstant = chain.getDefiningOp<mlir::arith::ConstantOp>();
  auto elements =
      falseConstant
          ? llvm::dyn_cast<mlir::DenseElementsAttr>(falseConstant.getValue())
          : mlir::DenseElementsAttr{};
  if (!elements || !elements.isSplat() ||
      elements.getSplatValue<mlir::BoolAttr>().getValue())
    return false;

  auto add = bound.getDefiningOp<mlir::arith::AddIOp>();
  if (!add)
    return false;
  mlir::Value negated;
  mlir::Value tripConstant;
  if (isConstantInt(add.getLhs(), static_cast<std::int64_t>(tripCount))) {
    tripConstant = add.getLhs();
    negated = add.getRhs();
  } else if (isConstantInt(add.getRhs(),
                           static_cast<std::int64_t>(tripCount))) {
    tripConstant = add.getRhs();
    negated = add.getLhs();
  } else {
    return false;
  }
  auto multiply = negated.getDefiningOp<mlir::arith::MulIOp>();
  if (!multiply)
    return false;
  const bool exactNegation =
      (multiply.getLhs() == induction &&
       isConstantInt(multiply.getRhs(), -1)) ||
      (multiply.getRhs() == induction && isConstantInt(multiply.getLhs(), -1));
  if (!exactNegation ||
      multiply.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::nsw)
    return false;

  support.insert(falseConstant);
  support.insert(add);
  support.insert(tripConstant.getDefiningOp());
  support.insert(multiply);
  support.insert(
      (multiply.getLhs() == induction ? multiply.getRhs() : multiply.getLhs())
          .getDefiningOp());
  return true;
}

bool matchesStatementValue(mlir::Value actual, mlir::Value expected,
                           mlir::Value tailMask, bool requireNeutralSelect,
                           mlir::Value neutral) {
  if (!requireNeutralSelect)
    return actual == expected;
  auto select = actual.getDefiningOp<mlir::arith::SelectOp>();
  return select && select.getCondition() == tailMask &&
         select.getTrueValue() == expected && select.getFalseValue() == neutral;
}

std::optional<mlir::vector::CombiningKind>
vectorReductionKind(mlir::arith::AtomicRMWKind kind) {
  using Atomic = mlir::arith::AtomicRMWKind;
  using Combining = mlir::vector::CombiningKind;
  switch (kind) {
  case Atomic::addf:
  case Atomic::addi:
    return Combining::ADD;
  case Atomic::mulf:
  case Atomic::muli:
    return Combining::MUL;
  case Atomic::minimumf:
    return Combining::MINIMUMF;
  case Atomic::minnumf:
    return Combining::MINNUMF;
  case Atomic::mins:
    return Combining::MINSI;
  case Atomic::minu:
    return Combining::MINUI;
  case Atomic::maximumf:
    return Combining::MAXIMUMF;
  case Atomic::maxnumf:
    return Combining::MAXNUMF;
  case Atomic::maxs:
    return Combining::MAXSI;
  case Atomic::maxu:
    return Combining::MAXUI;
  case Atomic::andi:
    return Combining::AND;
  case Atomic::ori:
    return Combining::OR;
  case Atomic::xori:
    return Combining::XOR;
  case Atomic::assign:
    return std::nullopt;
  }
  return std::nullopt;
}

bool operationMatchesReductionKind(mlir::Operation *operation,
                                   mlir::arith::AtomicRMWKind kind) {
  using Atomic = mlir::arith::AtomicRMWKind;
  const auto hasNoFastMath = [](mlir::Operation *candidate) {
    auto fastMath =
        llvm::dyn_cast<mlir::arith::ArithFastMathInterface>(candidate);
    if (!fastMath)
      return false;
    mlir::arith::FastMathFlagsAttr attribute = fastMath.getFastMathFlagsAttr();
    return !attribute ||
           attribute.getValue() == mlir::arith::FastMathFlags::none;
  };
  switch (kind) {
  case Atomic::addf:
    return llvm::isa<mlir::arith::AddFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::addi: {
    auto add = llvm::dyn_cast<mlir::arith::AddIOp>(operation);
    return add &&
           add.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none;
  }
  case Atomic::mulf:
    return llvm::isa<mlir::arith::MulFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::muli: {
    auto multiply = llvm::dyn_cast<mlir::arith::MulIOp>(operation);
    return multiply && multiply.getOverflowFlags() ==
                           mlir::arith::IntegerOverflowFlags::none;
  }
  case Atomic::minimumf:
    return llvm::isa<mlir::arith::MinimumFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::minnumf:
    return llvm::isa<mlir::arith::MinNumFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::mins:
    return llvm::isa<mlir::arith::MinSIOp>(operation);
  case Atomic::minu:
    return llvm::isa<mlir::arith::MinUIOp>(operation);
  case Atomic::maximumf:
    return llvm::isa<mlir::arith::MaximumFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::maxnumf:
    return llvm::isa<mlir::arith::MaxNumFOp>(operation) &&
           hasNoFastMath(operation);
  case Atomic::maxs:
    return llvm::isa<mlir::arith::MaxSIOp>(operation);
  case Atomic::maxu:
    return llvm::isa<mlir::arith::MaxUIOp>(operation);
  case Atomic::andi:
    return llvm::isa<mlir::arith::AndIOp>(operation);
  case Atomic::ori:
    return llvm::isa<mlir::arith::OrIOp>(operation);
  case Atomic::xori:
    return llvm::isa<mlir::arith::XOrIOp>(operation);
  case Atomic::assign:
    return false;
  }
  return false;
}

llvm::Expected<mlir::Value>
exactReductionNeutral(const ExactStructuredScopView &source,
                      mlir::Operation *loop) {
  if (source.reductionCount == 0)
    return mlir::Value{};
  if (source.reductionCount != 1 || !source.reductionKind ||
      loop->getNumResults() != 1)
    return invalid("exact SCoP has an ambiguous reduction identity");

  mlir::ValueRange inits =
      llvm::isa<mlir::affine::AffineForOp>(loop)
          ? llvm::cast<mlir::affine::AffineForOp>(loop).getInits()
          : llvm::cast<mlir::scf::ForOp>(loop).getInitArgs();
  if (inits.size() != 1)
    return invalid("materialized vector reduction has ambiguous inits");
  mlir::Value neutral = inits.front();
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(neutral.getType());
  auto constant = neutral.getDefiningOp<mlir::arith::ConstantOp>();
  auto elements =
      constant ? llvm::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())
               : mlir::DenseElementsAttr{};
  if (!vectorType || !elements || !elements.isSplat())
    return invalid("materialized vector reduction lost its neutral init");
  mlir::OpBuilder builder(loop);
  const mlir::TypedAttr identity = mlir::arith::getIdentityValueAttr(
      *source.reductionKind, vectorType.getElementType(), builder,
      loop->getLoc());
  if (elements.getSplatValue<mlir::Attribute>() != identity)
    return invalid("materialized vector reduction changed its neutral init");
  return neutral;
}

llvm::Error
verifyReductionImage(const ExactStructuredScopView &source,
                     const StructuredVectorScheduleCoordinate &coordinate,
                     mlir::Operation *loop) {
  llvm::SmallVector<mlir::vector::ReductionOp> reductions;
  llvm::SmallVector<mlir::vector::ExtractOp> extracts;
  llvm::SmallVector<mlir::Operation *> closure = verificationClosure(loop);
  for (mlir::Operation *operation : closure) {
    if (auto reduction = llvm::dyn_cast<mlir::vector::ReductionOp>(operation))
      reductions.push_back(reduction);
    if (auto extract = llvm::dyn_cast<mlir::vector::ExtractOp>(operation))
      extracts.push_back(extract);
  }
  if (source.reductionCount == 0)
    return reductions.empty() && extracts.empty()
               ? llvm::Error::success()
               : invalid("non-reduction vector child acquired a reduction");
  if (source.reductionCount != 1 || !source.reductionKind ||
      loop->getNumResults() != 1)
    return invalid("exact SCoP reduction identity is incomplete");
  const std::optional<mlir::vector::CombiningKind> expectedKind =
      vectorReductionKind(*source.reductionKind);
  if (!expectedKind)
    return invalid("exact SCoP reduction kind has no vector image");
  mlir::Value loopResult = loop->getResult(0);

  if (!reductions.empty()) {
    if (reductions.size() != 1 || !extracts.empty())
      return invalid("provider reduction image has ambiguous operations");
    mlir::vector::ReductionOp reduction = reductions.front();
    mlir::arith::FastMathFlagsAttr fastMath = reduction.getFastmathAttr();
    auto returned = reduction.getResult().hasOneUse()
                        ? llvm::dyn_cast<mlir::func::ReturnOp>(
                              *reduction.getResult().user_begin())
                        : mlir::func::ReturnOp{};
    if (reduction.getKind() != *expectedKind || reduction.getAcc() ||
        reduction.getVector() != loopResult ||
        !reduction.getResult().hasOneUse() ||
        (fastMath && fastMath.getValue() != mlir::arith::FastMathFlags::none) ||
        !returned || returned.getNumOperands() != 1 ||
        returned.getOperand(0) != reduction.getResult())
      return invalid("provider reduction kind or value relation changed");
    if (!loopResult.hasOneUse() || *loopResult.user_begin() != reduction)
      return invalid("provider reduction no longer uniquely consumes the loop");
    return llvm::Error::success();
  }

  const std::uint64_t factor = coordinate.shape.front();
  if (extracts.size() != factor)
    return invalid("lowered reduction has the wrong extract cardinality");
  std::vector<mlir::Value> lanes(factor);
  for (mlir::vector::ExtractOp extract : extracts) {
    llvm::ArrayRef<std::int64_t> position = extract.getStaticPosition();
    if (extract.getSource() != loopResult || position.size() != 1 ||
        position.front() < 0 ||
        static_cast<std::uint64_t>(position.front()) >= factor ||
        lanes[static_cast<std::size_t>(position.front())] ||
        !extract.getResult().hasOneUse())
      return invalid("lowered reduction changed an extract lane or source");
    lanes[static_cast<std::size_t>(position.front())] = extract.getResult();
  }
  std::size_t loopResultUsers = 0;
  for (mlir::Operation *user : loopResult.getUsers()) {
    ++loopResultUsers;
    if (!llvm::isa<mlir::vector::ExtractOp>(user))
      return invalid("lowered reduction loop result has a foreign user");
  }
  if (loopResultUsers != factor)
    return invalid("lowered reduction loop result has missing lane users");

  llvm::SmallVector<mlir::Operation *> combiners;
  llvm::SmallPtrSet<mlir::Operation *, 16> combinerSet;
  for (mlir::Operation *operation : closure) {
    if (!loop->isAncestor(operation) &&
        operationMatchesReductionKind(operation, *source.reductionKind)) {
      combiners.push_back(operation);
      combinerSet.insert(operation);
    }
  }
  if (combiners.size() != factor - 1)
    return invalid("lowered reduction has the wrong combiner cardinality");

  llvm::DenseMap<mlir::Value, std::uint64_t> laneUses;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> combinerUses;
  for (mlir::Value lane : lanes)
    laneUses.try_emplace(lane, 0);
  for (mlir::Operation *combiner : combiners) {
    if (combiner->getNumOperands() != 2 || combiner->getNumResults() != 1)
      return invalid("lowered reduction combiner has the wrong arity");
    for (mlir::Value operand : combiner->getOperands()) {
      auto lane = laneUses.find(operand);
      if (lane != laneUses.end()) {
        ++lane->second;
        continue;
      }
      mlir::Operation *definition = operand.getDefiningOp();
      if (!definition || !combinerSet.contains(definition))
        return invalid("lowered reduction combiner has a foreign operand");
      ++combinerUses[definition];
    }
  }
  if (llvm::any_of(laneUses,
                   [](const auto &entry) { return entry.second != 1; }))
    return invalid("lowered reduction does not consume every lane once");
  mlir::Operation *root = nullptr;
  for (mlir::Operation *combiner : combiners) {
    const std::uint64_t uses = combinerUses.lookup(combiner);
    if (uses == 0) {
      if (root)
        return invalid("lowered reduction has multiple scalar roots");
      root = combiner;
    } else if (uses != 1 || !combiner->getResult(0).hasOneUse()) {
      return invalid("lowered reduction combiner is reused");
    }
  }
  auto returned = root && root->getResult(0).hasOneUse()
                      ? llvm::dyn_cast<mlir::func::ReturnOp>(
                            *root->getResult(0).user_begin())
                      : mlir::func::ReturnOp{};
  if (!root || !root->getResult(0).hasOneUse() || !returned ||
      returned.getNumOperands() != 1 ||
      returned.getOperand(0) != root->getResult(0))
    return invalid("lowered reduction lost its externally visible result");
  return llvm::Error::success();
}

} // namespace

llvm::Error verifyStructuredVectorScheduleMaterialization(
    const ExactStructuredScopView &source,
    const StructuredVectorScheduleCoordinate &coordinate,
    mlir::ModuleOp materialized) {
  if (llvm::Error error =
          detail::validateStructuredVectorScheduleCoordinate(coordinate))
    return error;
  if (coordinate.reductionSchedule != source.reductionSchedule)
    return invalid("materialized vector reduction policy differs from source");
  const std::optional<std::uint64_t> requiredAlignment =
      llvm::checkedMulUnsigned(source.maximumElementBytes,
                               coordinate.shape.front());
  if (!requiredAlignment ||
      coordinate.requiredAlignmentBytes != *requiredAlignment ||
      llvm::any_of(source.accesses,
                   [&](const StructuredScopAccessView &access) {
                     return access.elementBytes != source.maximumElementBytes ||
                            access.alignmentBytes % *requiredAlignment != 0;
                   }))
    return invalid("materialized vector alignment proof is inconsistent");
  const bool divisible =
      source.constantTripCount &&
      *source.constantTripCount % coordinate.shape.front() == 0;
  const StructuredVectorTailPolicy expectedTail =
      divisible ? StructuredVectorTailPolicy::Exact
                : StructuredVectorTailPolicy::ReductionMask;
  if (coordinate.tailPolicy != expectedTail ||
      (!divisible &&
       source.reductionSchedule == StructuredReductionSchedule::None))
    return invalid("materialized vector tail policy is inconsistent");

  auto resolved = resolveScheduledLoop(source, materialized);
  if (!resolved)
    return resolved.takeError();
  mlir::Operation *loop = *resolved;
  if (!source.constantTripCount ||
      !hasExactLoopDomain(loop, *source.constantTripCount,
                          coordinate.shape.front()))
    return invalid("materialized vector loop has the wrong exact domain");
  mlir::Value induction = inductionVariable(loop);
  auto neutral = exactReductionNeutral(source, loop);
  if (!neutral)
    return neutral.takeError();

  llvm::SmallVector<mlir::Operation *> statements;
  for (mlir::Operation &operation : loop->getRegion(0).front()) {
    if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
      statements.push_back(&operation);
  }
  llvm::SmallVector<mlir::Operation *> transfers;
  mlir::Value tailMask;
  std::size_t createMasks = 0;
  for (mlir::Operation *statement : statements) {
    if (!vectorShapeMatches(statement, coordinate.shape.front()))
      return invalid("materialized vector statement has the wrong shape");
    if (auto mask = llvm::dyn_cast<mlir::vector::CreateMaskOp>(statement)) {
      tailMask = mask.getResult();
      ++createMasks;
    }
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(statement)) {
      transfers.push_back(statement);
      if (read.getMask())
        tailMask = read.getMask();
    } else if (auto write =
                   llvm::dyn_cast<mlir::vector::TransferWriteOp>(statement)) {
      transfers.push_back(statement);
      if (write.getMask())
        tailMask = write.getMask();
    }
  }
  if (transfers.size() != source.accesses.size())
    return invalid("materialized vector access cardinality changed");
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::Exact &&
      (tailMask || createMasks != 0))
    return invalid("exact vector coordinate acquired a tail mask");
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
      (!tailMask || createMasks > 1))
    return invalid("masked vector coordinate lost its unique tail mask");

  std::vector<mlir::Value> statementValues(source.statementCount);
  std::vector<std::pair<std::uint64_t, mlir::Value>> pendingStores;
  for (auto [access, operation] : llvm::zip_equal(source.accesses, transfers)) {
    mlir::Value base;
    mlir::ValueRange indices;
    mlir::AffineMap permutation;
    mlir::Value mask;
    bool inBounds = false;
    mlir::Value stored;
    mlir::Value loaded;
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(operation)) {
      if (access.kind != StructuredScopAccessKind::Read)
        return invalid("materialized vector access direction changed");
      base = read.getBase();
      indices = read.getIndices();
      permutation = read.getPermutationMap();
      mask = read.getMask();
      inBounds = allInBounds(read.getInBounds());
      loaded = read.getResult();
    } else {
      auto write = llvm::cast<mlir::vector::TransferWriteOp>(operation);
      if (access.kind != StructuredScopAccessKind::Write)
        return invalid("materialized vector access direction changed");
      base = write.getBase();
      indices = write.getIndices();
      permutation = write.getPermutationMap();
      mask = write.getMask();
      inBounds = allInBounds(write.getInBounds());
      stored = write.getValueToStore();
    }
    if (localBoundaryArgument(base, loop) != access.memoryBoundaryArgument ||
        indices.size() != 1 || indices.front() != induction ||
        permutation != mlir::AffineMap::getMultiDimIdentityMap(
                           1, materialized.getContext()))
      return invalid("materialized vector access relation changed");
    if (coordinate.tailPolicy == StructuredVectorTailPolicy::Exact) {
      if (mask || !inBounds)
        return invalid("exact vector transfer is not proven in-bounds");
    } else if (!mask || mask != tailMask) {
      return invalid("vector transfer uses a different tail mask");
    }
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(operation)) {
      if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
          (read.getResult().use_empty() ||
           !llvm::all_of(
               read.getResult().getUsers(), [&](mlir::Operation *user) {
                 auto select = llvm::dyn_cast<mlir::arith::SelectOp>(user);
                 return select && select.getCondition() == mask &&
                        select.getTrueValue() == read.getResult() &&
                        select.getFalseValue() == *neutral;
               })))
        return invalid("masked vector read lost its exact neutral guard");
    }
    if (loaded)
      statementValues[access.statementOrdinal] = loaded;
    if (stored) {
      if (!access.storedStatementOrdinal)
        return invalid("materialized vector store lost its source value");
      pendingStores.emplace_back(*access.storedStatementOrdinal, stored);
    }
  }

  llvm::SmallPtrSet<mlir::Operation *, 32> loweredTailSupport;
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask) {
    auto createMask = tailMask.getDefiningOp<mlir::vector::CreateMaskOp>();
    if (!source.constantTripCount)
      return invalid("masked vector coordinate lost its static trip count");
    if (createMask) {
      if (createMasks != 1 || createMask.getNumOperands() != 1 ||
          !isExactTailBoundProducer(createMask.getOperand(0).getDefiningOp(),
                                    induction, tailMask,
                                    *source.constantTripCount))
        return invalid("provider vector tail bound changed");
    } else if (createMasks != 0 ||
               !collectExactLoweredTailMask(
                   tailMask, induction, *source.constantTripCount,
                   coordinate.shape.front(), loweredTailSupport)) {
      return invalid("materialized vector tail mask changed");
    }
  }

  std::size_t nextCompute = 0;
  for (mlir::Operation *statement : statements) {
    if (llvm::isa<mlir::vector::TransferReadOp, mlir::vector::TransferWriteOp,
                  mlir::vector::CreateMaskOp>(statement))
      continue;
    if (isTransferPaddingProducer(statement))
      continue;
    if (loweredTailSupport.contains(statement))
      continue;
    if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
        source.constantTripCount &&
        isExactTailBoundProducer(statement, induction, tailMask,
                                 *source.constantTripCount))
      continue;
    auto projection =
        dataflow::projectRegisteredActorSchemaProjection(statement);
    if (!projection) {
      return invalid("materialized vector child has unregistered support: " +
                     llvm::toString(projection.takeError()));
    }
    if (nextCompute == source.computes.size() ||
        projection->schema != source.computes[nextCompute].schema ||
        !payloadCorresponds(source.computes[nextCompute].payload,
                            projection->payload)) {
      const bool generatedTailSupport =
          coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
          (projection->schema == dataflow::OperationSchemaId::ArithSelect ||
           projection->schema == dataflow::OperationSchemaId::ArithConstant ||
           projection->schema == dataflow::OperationSchemaId::ArithCmpI ||
           projection->schema == dataflow::OperationSchemaId::VectorInsert);
      if (!generatedTailSupport)
        return invalid("materialized vector compute semantics changed");
      continue;
    }
    const StructuredScopComputeView &expected = source.computes[nextCompute++];
    const bool maskedRecurrence =
        coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
        llvm::any_of(expected.operandStatements,
                     [](const std::optional<std::uint64_t> &operand) {
                       return !operand;
                     });
    if (statement->getNumResults() != 1 ||
        statement->getNumOperands() != expected.operandStatements.size())
      return invalid("materialized vector compute arity changed");
    for (auto [actual, sourceStatement] : llvm::zip_equal(
             statement->getOperands(), expected.operandStatements)) {
      if (sourceStatement) {
        if (!statementValues[*sourceStatement] ||
            !matchesStatementValue(actual, statementValues[*sourceStatement],
                                   tailMask, maskedRecurrence, *neutral))
          return invalid("materialized vector compute data dependence changed");
      } else if (!isLoopIterArgument(actual, loop)) {
        return invalid("materialized vector reduction recurrence changed");
      }
    }
    statementValues[expected.statementOrdinal] = statement->getResult(0);
  }
  if (nextCompute != source.computes.size())
    return invalid("materialized vector child lost a source computation");
  for (auto [sourceStatement, stored] : pendingStores) {
    if (sourceStatement >= statementValues.size() ||
        !statementValues[sourceStatement] ||
        stored != statementValues[sourceStatement])
      return invalid("materialized vector store data dependence changed");
  }

  if (llvm::Error error = verifyReductionImage(source, coordinate, loop))
    return error;
  if (mlir::failed(mlir::verify(materialized)))
    return invalid("materialized vector child does not verify");
  return llvm::Error::success();
}

} // namespace loom::frontend
