#include "Arbitration.h"

#include "Fabric/IR/ResultPresentation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cassert>

namespace loom::hardware::rtl::hierarchy {
namespace {

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

/// Encodes the requester that a one-hot-or-zero word names, advanced `step`
/// positions in cyclic requester order, and falls back to `fallback` when no
/// bit is set. Constant masks keep this one comparison per encoded bit
/// instead of one fallback-width mux per requester.
mlir::Value encodeSelected(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value packed, std::size_t requesterCount,
                           std::size_t step, mlir::Value fallback) {
  assert(requesterCount != 0 && "encoded requester domain must not be empty");
  const unsigned width =
      mlir::cast<mlir::IntegerType>(fallback.getType()).getWidth();
  const llvm::APInt zeroMask(requesterCount, 0);
  mlir::Value any = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ne, packed,
      circt::hw::ConstantOp::create(builder, location, zeroMask), true);
  llvm::SmallVector<mlir::Value> encodedHighToLow;
  encodedHighToLow.reserve(width);
  for (unsigned bit = width; bit != 0; --bit) {
    llvm::APInt mask(requesterCount, 0);
    for (std::size_t requester = 0; requester != requesterCount; ++requester)
      if ((((requester + step) % requesterCount) >> (bit - 1)) & 1U)
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
  return circt::comb::MuxOp::create(builder, location, any, encoded, fallback,
                                    true);
}

/// The first requester at or after `start` in cyclic requester order whose
/// `requests` bit is set, and `fallback` when no bit is set. This is the scan
/// of the registered-grant next-state function; the caller separates `start`
/// from `fallback` so a scan that begins strictly after the pointer still
/// falls back to the pointer.
mlir::Value scanRequests(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value requests, std::size_t requesterCount,
                         mlir::Value start, mlir::Value fallback) {
  return encodeSelected(
      builder, location,
      roundRobinPackedSelection(builder, location, requests,
                                static_cast<unsigned>(requesterCount), start),
      requesterCount, 0, fallback);
}

/// The requester word that names `position` alone. One shift keeps this
/// independent of the requester count, where one comparator per requester
/// would not be.
mlir::Value oneHotAt(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Value position, unsigned requesterCount) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(position.getType()).getWidth();
  assert(width <= requesterCount && "position is wider than its domain");
  mlir::Value extended =
      width == requesterCount
          ? position
          : circt::comb::ConcatOp::create(
                builder, location,
                llvm::ArrayRef<mlir::Value>{
                    constant(builder, location, requesterCount - width, 0),
                    position});
  return circt::comb::ShlOp::create(
      builder, location, constant(builder, location, requesterCount, 1),
      extended, true);
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
  assert(mlir::cast<mlir::IntegerType>(cursor.getType()).getWidth() ==
             indexWidth(requestCount) &&
         "round-robin cursor has the wrong width for its requester domain");
  if (requestCount == 1)
    return packed;
  mlir::Value cursorOneHot =
      oneHotAt(builder, location, cursor, requestCount);
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
  return encodeSelected(builder, location, packed, requesterCount, 1, current);
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

RegisteredGrant makeRegisteredGrant(mlir::OpBuilder &builder,
                                    mlir::Location location,
                                    circt::BackedgeBuilder &backedges,
                                    std::size_t requesterCount,
                                    bool roundRobin, unsigned resetPosition,
                                    mlir::Value clock, mlir::Value reset,
                                    llvm::StringRef name,
                                    const ClockResetPlan &clockReset) {
  assert(requesterCount != 0 && "grant domain must not be empty");
  if (requesterCount == 1)
    return RegisteredGrant{std::nullopt,
                           {bitConstant(builder, location, true)},
                           constant(builder, location, 1, 1),
                           mlir::Value{},
                           roundRobin};
  const unsigned count = static_cast<unsigned>(requesterCount);
  const unsigned width = indexWidth(requesterCount);
  circt::Backedge next = backedges.get(builder.getIntegerType(width));
  mlir::Value pointer = createRegister(
      builder, location, next, clock, reset, llvm::APInt(width, resetPosition),
      (name + "_pointer_reg").str(), clockReset.asynchronousReset);
  mlir::Value oneHot = oneHotAt(builder, location, pointer, count);
  std::vector<mlir::Value> pointed;
  pointed.reserve(requesterCount);
  for (unsigned position = 0; position != count; ++position)
    pointed.push_back(
        circt::comb::ExtractOp::create(builder, location, oneHot, position, 1));
  return RegisteredGrant{std::optional<circt::Backedge>(std::move(next)),
                         std::move(pointed), oneHot, pointer, roundRobin};
}

void advanceRegisteredGrant(mlir::OpBuilder &builder, mlir::Location location,
                            RegisteredGrant &grant, mlir::Value requests,
                            mlir::Value fired) {
  if (!grant.next)
    return;
  const unsigned count = static_cast<unsigned>(grant.pointed.size());
  assert(mlir::cast<mlir::IntegerType>(requests.getType()).getWidth() ==
             count &&
         "packed request width disagrees with its grant domain");
  const unsigned width =
      mlir::cast<mlir::IntegerType>(grant.pointer.getType()).getWidth();
  if (!grant.roundRobin) {
    // FixedPriority names the highest-priority requester of this cycle, and
    // keeps the pointer when none requests.
    grant.next->setValue(scanRequests(builder, location, requests, count,
                                      constant(builder, location, width, 0),
                                      grant.pointer));
    return;
  }
  // RoundRobin passes its turn on once the pointed requester has proceeded or
  // has nothing to offer, and otherwise keeps it, so a requester whose
  // service refuses holds its turn. The scan begins strictly after the
  // pointer and ends at the pointer, so a lone continuous requester keeps its
  // own turn every cycle.
  mlir::Value requestedAtPointer = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ne,
      circt::comb::AndOp::create(builder, location, requests, grant.oneHot,
                                 true),
      constant(builder, location, count, 0), true);
  mlir::Value blocked =
      andValues(builder, location,
                {requestedAtPointer,
                 circt::comb::createOrFoldNot(builder, location, fired)});
  grant.next->setValue(circt::comb::MuxOp::create(
      builder, location, blocked, grant.pointer,
      scanRequests(builder, location, requests, count,
                   incrementModulo(builder, location, grant.pointer, count),
                   grant.pointer),
      true));
}

std::vector<mlir::Value> selectResultPresentation(
    mlir::OpBuilder &builder, mlir::Location location,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> requestedLaneDestinations,
    mlir::Value priority) {
  assert(!requestedLaneDestinations.empty() &&
         !requestedLaneDestinations.front().empty());
  const auto count =
      static_cast<std::uint32_t>(requestedLaneDestinations.size());
  const unsigned destinationCount =
      mlir::cast<mlir::IntegerType>(
          requestedLaneDestinations.front().front().getType())
          .getWidth();
  mlir::Value empty = constant(builder, location, destinationCount, 0);
  llvm::SmallVector<mlir::Value> requests;
  llvm::SmallVector<mlir::Value> requestedDestinations;
  for (const auto &lanes : requestedLaneDestinations) {
    mlir::Value claims = empty;
    llvm::SmallVector<mlir::Value> admissible;
    for (mlir::Value lane : lanes) {
      mlir::Value overlap =
          circt::comb::AndOp::create(builder, location, claims, lane, true);
      admissible.push_back(circt::comb::ICmpOp::create(
          builder, location, circt::comb::ICmpPredicate::eq, overlap, empty,
          true));
      claims = circt::comb::OrOp::create(builder, location, claims, lane, true);
    }
    requestedDestinations.push_back(claims);
    admissible.push_back(circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::ne, claims, empty,
        true));
    requests.push_back(andValues(builder, location, admissible));
  }
  if (count == 1) {
    mlir::Value active = circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::eq, priority,
        constant(builder, location,
                 mlir::cast<mlir::IntegerType>(priority.getType()).getWidth(),
                 0),
        true);
    return {andValues(builder, location, {requests.front(), active})};
  }
  std::vector<mlir::Value> result(count, bitConstant(builder, location, false));
  for (std::uint32_t head = 0; head != count; ++head) {
    mlir::Value thisHead = circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::eq, priority,
        constant(builder, location,
                 mlir::cast<mlir::IntegerType>(priority.getType()).getWidth(),
                 head),
        true);
    mlir::Value occupied = empty;
    for (std::uint32_t requester :
         ::fabric::resultPresentationOrder(count, head)) {
      mlir::Value claims = requestedDestinations[requester];
      mlir::Value overlap =
          circt::comb::AndOp::create(builder, location, claims, occupied, true);
      mlir::Value fits = circt::comb::ICmpOp::create(
          builder, location, circt::comb::ICmpPredicate::eq, overlap, empty,
          true);
      mlir::Value grant =
          andValues(builder, location, {requests[requester], fits});
      result[requester] = orValues(
          builder, location,
          {result[requester], andValues(builder, location, {thisHead, grant})});
      occupied = circt::comb::OrOp::create(
          builder, location, occupied,
          circt::comb::MuxOp::create(builder, location, grant, claims, empty,
                                     true),
          true);
    }
  }
  return result;
}

mlir::Value makeResultPresentationPriority(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::BackedgeBuilder &backedges,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> eligible,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> evaluated, mlir::Value clock,
    mlir::Value reset, llvm::StringRef name, const ClockResetPlan &clockReset) {
  llvm::SmallVector<std::uint32_t> evaluationCounts;
  for (const auto &positions : eligible)
    evaluationCounts.push_back(positions.size());
  const auto positions =
      ::fabric::resultPresentationPositions(evaluationCounts);
  llvm::SmallVector<mlir::Value> requests;
  for (const auto position : positions)
    requests.push_back(eligible[position.requester][position.evaluation]);
  auto selection = makeStatefulSelection(builder, location, backedges, requests,
                                         clock, reset, name, clockReset);
  // The additional code denotes no eligible position, so every selector
  // stays inactive without inventing a requester in the canonical domain.
  const unsigned focusWidth = indexWidth(eligible.size() + 1);
  mlir::Value focus = constant(builder, location, focusWidth, eligible.size());
  llvm::SmallVector<mlir::Value> completed;
  for (auto [ordinal, position] : llvm::enumerate(positions)) {
    focus = circt::comb::MuxOp::create(
        builder, location, selection.selected[ordinal],
        constant(builder, location, focusWidth, position.requester), focus,
        true);
    completed.push_back(
        andValues(builder, location,
                  {selection.selected[ordinal],
                   evaluated[position.requester][position.evaluation]}));
  }
  advanceStatefulSelection(builder, location, selection, completed);
  return focus;
}

} // namespace loom::hardware::rtl::hierarchy
