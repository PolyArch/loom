#include "Arbitration.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace loom::hardware::rtl::hierarchy {
namespace {

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

} // namespace

mlir::Value packBits(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> lowToHigh) {
  assert(!lowToHigh.empty() && "packed bit domain must not be empty");
  if (lowToHigh.size() == 1)
    return lowToHigh.front();
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(lowToHigh.size());
  for (mlir::Value value : llvm::reverse(lowToHigh))
    highToLow.push_back(value);
  return circt::comb::ConcatOp::create(builder, location, highToLow);
}

mlir::Value roundRobinPackedSelection(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      mlir::Value packed, unsigned requestCount,
                                      mlir::Value cursor) {
  assert(requestCount != 0 && "round-robin domain must not be empty");
  if (requestCount == 1)
    return packed;
  const unsigned cursorWidth =
      mlir::cast<mlir::IntegerType>(cursor.getType()).getWidth();
  const unsigned distanceWidth = requestCount;
  assert(cursorWidth <= distanceWidth &&
         "round-robin cursor width exceeds its requester domain");
  mlir::Value extendedCursor = cursor;
  if (cursorWidth < distanceWidth)
    extendedCursor = circt::comb::ConcatOp::create(
        builder, location,
        llvm::ArrayRef<mlir::Value>{
            constant(builder, location, distanceWidth - cursorWidth, 0),
            cursor});
  mlir::Value inverseDistance = circt::comb::SubOp::create(
      builder, location,
      constant(builder, location, distanceWidth, requestCount), extendedCursor,
      true);

  mlir::Value rotatedRight = circt::comb::ShrUOp::create(
      builder, location, packed, extendedCursor, true);
  mlir::Value wrappedLeft = circt::comb::ShlOp::create(
      builder, location, packed, inverseDistance, true);
  mlir::Value rotated = circt::comb::OrOp::create(
      builder, location, rotatedRight, wrappedLeft, true);
  mlir::Value lowest = circt::comb::AndOp::create(
      builder, location, rotated,
      circt::comb::SubOp::create(builder, location,
                                 constant(builder, location, requestCount, 0),
                                 rotated, true),
      true);

  mlir::Value restoredLeft = circt::comb::ShlOp::create(
      builder, location, lowest, extendedCursor, true);
  mlir::Value restoredRight = circt::comb::ShrUOp::create(
      builder, location, lowest, inverseDistance, true);
  return circt::comb::OrOp::create(builder, location, restoredLeft,
                                   restoredRight, true);
}

mlir::Value roundRobinPackedSelection(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      llvm::ArrayRef<mlir::Value> requests,
                                      mlir::Value cursor) {
  assert(!requests.empty() && "round-robin domain must not be empty");
  return roundRobinPackedSelection(
      builder, location, packBits(builder, location, requests),
      static_cast<unsigned>(requests.size()), cursor);
}

std::vector<mlir::Value>
roundRobinSelection(mlir::OpBuilder &builder, mlir::Location location,
                    llvm::ArrayRef<mlir::Value> requests, mlir::Value cursor) {
  if (requests.empty())
    return {};
  mlir::Value restored =
      roundRobinPackedSelection(builder, location, requests, cursor);
  std::vector<mlir::Value> selected;
  selected.reserve(requests.size());
  for (std::size_t requester = 0; requester != requests.size(); ++requester)
    selected.push_back(circt::comb::ExtractOp::create(builder, location,
                                                      restored, requester, 1));
  return selected;
}

mlir::Value nextCursorFromPacked(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value current,
                                 mlir::Value packed,
                                 std::size_t requesterCount) {
  assert(requesterCount != 0 && "cursor domain must not be empty");
  const unsigned width =
      mlir::cast<mlir::IntegerType>(current.getType()).getWidth();
  const llvm::APInt zeroMask(requesterCount, 0);
  mlir::Value anyFired = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ne, packed,
      circt::hw::ConstantOp::create(builder, location, zeroMask), true);

  // Encode the granted successor with constant masks instead of expanding one
  // cursor-width mux for every requester.
  llvm::SmallVector<mlir::Value> encodedHighToLow;
  encodedHighToLow.reserve(width);
  for (unsigned bit = width; bit != 0; --bit) {
    llvm::APInt mask(requesterCount, 0);
    for (std::size_t requester = 0; requester != requesterCount; ++requester)
      if ((((requester + 1) % requesterCount) >> (bit - 1)) & 1U)
        mask.setBit(requester);
    if (mask.isZero()) {
      encodedHighToLow.push_back(bitConstant(builder, location, false));
      continue;
    }
    mlir::Value selected = circt::comb::AndOp::create(
        builder, location, packed,
        circt::hw::ConstantOp::create(builder, location, mask), true);
    encodedHighToLow.push_back(circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::ne, selected,
        circt::hw::ConstantOp::create(builder, location, zeroMask), true));
  }
  mlir::Value encoded =
      width == 1
          ? encodedHighToLow.front()
          : circt::comb::ConcatOp::create(builder, location, encodedHighToLow);
  return circt::comb::MuxOp::create(builder, location, anyFired, encoded,
                                    current, true);
}

mlir::Value nextCursor(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value current, llvm::ArrayRef<mlir::Value> fired) {
  if (fired.empty())
    return current;
  return nextCursorFromPacked(builder, location, current,
                              packBits(builder, location, fired), fired.size());
}

StatefulSelection makeStatefulSelection(mlir::OpBuilder &builder,
                                        mlir::Location location,
                                        circt::BackedgeBuilder &backedges,
                                        llvm::ArrayRef<mlir::Value> requests,
                                        mlir::Value clock, mlir::Value reset,
                                        llvm::StringRef name,
                                        const ClockResetPlan &clockReset) {
  const unsigned width = indexWidth(requests.size());
  if (requests.size() <= 1)
    return StatefulSelection{
        std::nullopt, constant(builder, location, width, 0),
        std::vector<mlir::Value>(requests.begin(), requests.end())};
  circt::Backedge next = backedges.get(builder.getIntegerType(width));
  mlir::Value cursor =
      createRegister(builder, location, next, clock, reset,
                     llvm::APInt(width, 0), name, clockReset.asynchronousReset);
  return StatefulSelection{
      std::optional<circt::Backedge>(std::move(next)), cursor,
      roundRobinSelection(builder, location, requests, cursor)};
}

void advanceStatefulSelection(mlir::OpBuilder &builder, mlir::Location location,
                              StatefulSelection &selection,
                              llvm::ArrayRef<mlir::Value> fired) {
  if (selection.next)
    selection.next->setValue(
        nextCursor(builder, location, selection.cursor, fired));
}

} // namespace loom::hardware::rtl::hierarchy
