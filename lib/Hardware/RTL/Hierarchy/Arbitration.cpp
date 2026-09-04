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
  assert(mlir::cast<mlir::IntegerType>(packed.getType()).getWidth() ==
             requestCount &&
         "packed request width disagrees with its round-robin domain");
  const unsigned cursorWidth =
      mlir::cast<mlir::IntegerType>(cursor.getType()).getWidth();
  assert(cursorWidth == indexWidth(requestCount) &&
         "round-robin cursor has the wrong width for its requester domain");
  if (requestCount == 1)
    return packed;
  mlir::Value extendedCursor = cursor;
  if (cursorWidth < requestCount)
    extendedCursor = circt::comb::ConcatOp::create(
        builder, location,
        llvm::ArrayRef<mlir::Value>{
            constant(builder, location, requestCount - cursorWidth, 0),
            cursor});
  mlir::Value cursorOneHot = circt::comb::ShlOp::create(
      builder, location, constant(builder, location, requestCount, 1),
      extendedCursor, true);
  // Subtracting the cursor bit isolates the first request at or after that
  // position. If that interval is empty, the ordinary lowest bit is the
  // wrapped selection.
  mlir::Value selectedAhead = circt::comb::AndOp::create(
      builder, location, packed,
      circt::comb::createOrFoldNot(
          builder, location,
          circt::comb::SubOp::create(builder, location, packed, cursorOneHot,
                                     true),
          true),
      true);
  mlir::Value selectedWrapped = circt::comb::AndOp::create(
      builder, location, packed,
      circt::comb::SubOp::create(builder, location,
                                 constant(builder, location, requestCount, 0),
                                 packed, true),
      true);
  mlir::Value hasAhead = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ne, selectedAhead,
      constant(builder, location, requestCount, 0), true);
  return circt::comb::MuxOp::create(builder, location, hasAhead, selectedAhead,
                                    selectedWrapped, true);
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
