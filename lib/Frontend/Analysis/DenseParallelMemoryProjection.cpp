#include "Frontend/Analysis/DenseParallelMemoryProjection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace loom::frontend::analysis {
namespace {

std::optional<unsigned> integralWidth(mlir::Type type, unsigned indexWidth) {
  if (llvm::isa<mlir::IndexType>(type))
    return indexWidth;
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  return std::nullopt;
}

mlir::Value stripWidthPreservingIndexCast(mlir::Value value,
                                          unsigned indexWidth) {
  while (true) {
    mlir::Value input;
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
      input = cast.getIn();
    else if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
      input = cast.getIn();
    else
      return value;
    auto sourceWidth = integralWidth(input.getType(), indexWidth);
    auto resultWidth = integralWidth(value.getType(), indexWidth);
    if (!sourceWidth || !resultWidth || *sourceWidth != *resultWidth)
      return value;
    value = input;
  }
}

bool sameInvariantFactor(mlir::Value lhs, mlir::Value rhs,
                         unsigned indexWidth) {
  lhs = stripWidthPreservingIndexCast(lhs, indexWidth);
  rhs = stripWidthPreservingIndexCast(rhs, indexWidth);
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  if (auto lhsExtension = lhs.getDefiningOp<mlir::arith::ExtSIOp>()) {
    auto rhsExtension = rhs.getDefiningOp<mlir::arith::ExtSIOp>();
    return rhsExtension &&
           sameInvariantFactor(lhsExtension.getIn(), rhsExtension.getIn(),
                               indexWidth);
  }
  if (auto lhsExtension = lhs.getDefiningOp<mlir::arith::ExtUIOp>()) {
    auto rhsExtension = rhs.getDefiningOp<mlir::arith::ExtUIOp>();
    return rhsExtension &&
           sameInvariantFactor(lhsExtension.getIn(), rhsExtension.getIn(),
                               indexWidth);
  }

  return mlir::getConstantIntValue(lhs) == mlir::getConstantIntValue(rhs) &&
         mlir::getConstantIntValue(lhs).has_value();
}

bool sameInvariantFactor(mlir::Value actual, mlir::OpFoldResult expected,
                         unsigned indexWidth) {
  if (auto value = llvm::dyn_cast<mlir::Value>(expected))
    return sameInvariantFactor(actual, value, indexWidth);
  return mlir::getConstantIntValue(actual) ==
         mlir::getConstantIntValue(expected);
}

bool matchesIdentityConstant(mlir::Value value, std::int64_t expected,
                             llvm::DenseSet<mlir::Value> &active) {
  if (expected != 0 && expected != 1)
    return false;
  auto constant = mlir::getConstantIntValue(value);
  if (constant)
    return *constant == expected;
  if (!active.insert(value).second)
    return false;

  const auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return finish(matchesIdentityConstant(cast.getIn(), expected, active));
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return finish(matchesIdentityConstant(cast.getIn(), expected, active));
  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>())
    return finish(
        matchesIdentityConstant(extension.getIn(), expected, active));
  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>())
    return finish(
        matchesIdentityConstant(extension.getIn(), expected, active));
  return finish(false);
}

bool matchesIdentityConstant(mlir::Value value, std::int64_t expected) {
  llvm::DenseSet<mlir::Value> active;
  return matchesIdentityConstant(value, expected, active);
}

bool matchesZeroBasedCoordinateProjection(
    mlir::Value value, mlir::Value coordinate, unsigned indexWidth,
    llvm::DenseSet<mlir::Value> &active) {
  value = stripWidthPreservingIndexCast(value, indexWidth);
  coordinate = stripWidthPreservingIndexCast(coordinate, indexWidth);
  if (value == coordinate)
    return true;
  if (!active.insert(value).second)
    return false;

  const auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return finish(matchesZeroBasedCoordinateProjection(
        cast.getIn(), coordinate, indexWidth, active));
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return finish(matchesZeroBasedCoordinateProjection(
        cast.getIn(), coordinate, indexWidth, active));
  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>())
    return finish(matchesZeroBasedCoordinateProjection(
        extension.getIn(), coordinate, indexWidth, active));
  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>())
    return finish(matchesZeroBasedCoordinateProjection(
        extension.getIn(), coordinate, indexWidth, active));
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    if (matchesIdentityConstant(add.getLhs(), 0))
      return finish(matchesZeroBasedCoordinateProjection(
          add.getRhs(), coordinate, indexWidth, active));
    if (matchesIdentityConstant(add.getRhs(), 0))
      return finish(matchesZeroBasedCoordinateProjection(
          add.getLhs(), coordinate, indexWidth, active));
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (matchesIdentityConstant(multiply.getLhs(), 1))
      return finish(matchesZeroBasedCoordinateProjection(
          multiply.getRhs(), coordinate, indexWidth, active));
    if (matchesIdentityConstant(multiply.getRhs(), 1))
      return finish(matchesZeroBasedCoordinateProjection(
          multiply.getLhs(), coordinate, indexWidth, active));
  }
  return finish(false);
}

bool matchesZeroBasedCoordinateProjection(mlir::Value value,
                                          mlir::Value coordinate,
                                          unsigned indexWidth) {
  llvm::DenseSet<mlir::Value> active;
  return matchesZeroBasedCoordinateProjection(value, coordinate, indexWidth,
                                               active);
}

bool provesSignedIndexFit(mlir::Value value, unsigned indexWidth) {
  value = stripWidthPreservingIndexCast(value, indexWidth);
  if (auto constant = mlir::getConstantIntValue(value))
    return llvm::isIntN(indexWidth, *constant);
  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
    auto source = llvm::dyn_cast<mlir::IntegerType>(
        extension.getIn().getType());
    return source && source.getWidth() <= indexWidth;
  }
  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
    auto source = llvm::dyn_cast<mlir::IntegerType>(
        extension.getIn().getType());
    return source &&
           (source.getWidth() < indexWidth ||
            (source.getWidth() == indexWidth && extension.getNonNeg()));
  }
  return false;
}

bool matchesDenseBoundFactor(mlir::Value actual, mlir::OpFoldResult expected,
                             unsigned indexWidth) {
  if (sameInvariantFactor(actual, expected, indexWidth))
    return true;
  auto expectedValue = llvm::dyn_cast<mlir::Value>(expected);
  if (!expectedValue)
    return false;
  if (auto cast = expectedValue.getDefiningOp<mlir::arith::IndexCastOp>())
    return sameInvariantFactor(actual, cast.getIn(), indexWidth) &&
           provesSignedIndexFit(actual, indexWidth);
  if (auto cast =
          expectedValue.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return sameInvariantFactor(actual, cast.getIn(), indexWidth) &&
           provesSignedIndexFit(actual, indexWidth);
  return false;
}

bool collectDenseProductFactors(
    mlir::Value value, unsigned indexWidth,
    llvm::SmallVectorImpl<mlir::Value> &factors) {
  value = stripWidthPreservingIndexCast(value, indexWidth);
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (multiply.getOverflowFlags() ==
        mlir::arith::IntegerOverflowFlags::none)
      return false;
    return collectDenseProductFactors(multiply.getLhs(), indexWidth,
                                      factors) &&
           collectDenseProductFactors(multiply.getRhs(), indexWidth, factors);
  }
  factors.push_back(value);
  return true;
}

bool matchesDenseCoordinateFactor(
    mlir::Value expression, std::size_t dimension,
    llvm::ArrayRef<mlir::Value> coordinates,
    llvm::ArrayRef<mlir::OpFoldResult> upperBounds, unsigned indexWidth) {
  llvm::SmallVector<mlir::Value, 4> factors;
  if (!collectDenseProductFactors(expression, indexWidth, factors))
    return false;

  const auto consumeValue = [&](mlir::Value expected) {
    auto found = llvm::find_if(factors, [&](mlir::Value factor) {
      return matchesZeroBasedCoordinateProjection(factor, expected,
                                                  indexWidth);
    });
    if (found == factors.end())
      return false;
    factors.erase(found);
    return true;
  };
  const auto consumeBound = [&](mlir::OpFoldResult expected) {
    auto found = llvm::find_if(factors, [&](mlir::Value factor) {
      return matchesDenseBoundFactor(factor, expected, indexWidth);
    });
    if (found == factors.end())
      return false;
    factors.erase(found);
    return true;
  };

  if (!consumeValue(coordinates[dimension]))
    return false;
  for (std::size_t inner = dimension + 1; inner < coordinates.size(); ++inner)
    if (!consumeBound(upperBounds[inner]))
      return false;
  return factors.empty();
}

bool collectDenseGepCoordinates(
    mlir::Value address, llvm::SmallVectorImpl<mlir::Value> &coordinates,
    mlir::Type &elementType) {
  auto gep = address.getDefiningOp<mlir::LLVM::GEPOp>();
  if (!gep)
    return true;
  if (!mlir::LLVM::bitEnumContainsAny(
          gep.getNoWrapFlags(), mlir::LLVM::GEPNoWrapFlags::inboundsFlag) ||
      gep.getRawConstantIndices().size() != 1 ||
      gep.getRawConstantIndices().front() !=
          mlir::LLVM::GEPOp::kDynamicIndex ||
      gep.getDynamicIndices().size() != 1)
    return false;
  if (!collectDenseGepCoordinates(gep.getBase(), coordinates, elementType))
    return false;
  if (elementType && elementType != gep.getElemType())
    return false;
  elementType = gep.getElemType();
  coordinates.push_back(gep.getDynamicIndices().front());
  return true;
}

} // namespace

bool hasExactDenseCoordinateStoreProjection(
    mlir::Operation *store, llvm::ArrayRef<mlir::Value> coordinates,
    llvm::ArrayRef<mlir::OpFoldResult> upperBounds, unsigned indexWidth) {
  if (!store || coordinates.size() < 2 ||
      coordinates.size() != upperBounds.size() || indexWidth == 0)
    return false;

  if (auto memrefStore = mlir::dyn_cast<mlir::memref::StoreOp>(store)) {
    if (memrefStore.getIndices().size() != coordinates.size())
      return false;
    return llvm::all_of(llvm::enumerate(memrefStore.getIndices()),
                        [&](auto indexed) {
                          return sameInvariantFactor(
                              indexed.value(), coordinates[indexed.index()],
                              indexWidth);
                        });
  }

  auto llvmStore = mlir::dyn_cast<mlir::LLVM::StoreOp>(store);
  if (!llvmStore)
    return false;
  llvm::SmallVector<mlir::Value, 4> projectedCoordinates;
  mlir::Type elementType;
  if (!collectDenseGepCoordinates(llvmStore.getAddr(), projectedCoordinates,
                                  elementType) ||
      projectedCoordinates.size() != coordinates.size())
    return false;
  for (auto [dimension, expression] :
       llvm::enumerate(projectedCoordinates))
    if (!matchesDenseCoordinateFactor(expression, dimension, coordinates,
                                      upperBounds, indexWidth))
      return false;
  return true;
}

} // namespace loom::frontend::analysis
