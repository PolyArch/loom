//===- DFGSimulatorVectorActors.cpp - fixed-vector structure ------------===//
//
// Exact functional execution of standard vector.extract, vector.insert, and
// vector.shuffle actors. OperationSchema owns positions and masks; this file
// only evaluates that projection over execution-local lane values. CGRA
// compute commits through the same provider.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

enum class PositionState : std::uint8_t { Valid, Poison, Undef };

struct ResolvedVectorPosition {
  PositionState state = PositionState::Valid;
  std::size_t firstLane = 0;
  std::size_t laneCount = 0;
};

llvm::Expected<std::size_t> checkedProduct(llvm::ArrayRef<std::int64_t> dims,
                                           llvm::StringRef what) {
  std::size_t product = 1;
  for (std::int64_t dim : dims) {
    if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                        std::numeric_limits<std::size_t>::max() / product)
      return llvm::createStringError(std::errc::value_too_large,
                                     "%s exceeds the host size domain",
                                     what.str().c_str());
    product *= static_cast<std::size_t>(dim);
  }
  return product;
}

llvm::Expected<ResolvedVectorPosition>
resolvePosition(const dataflow::VectorStaticPositionPayload &payload,
                llvm::ArrayRef<Token> dynamicTokens,
                mlir::VectorType containerType, mlir::Operation *scope) {
  llvm::ArrayRef<std::int64_t> shape = containerType.getShape();
  if (payload.position.size() > shape.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector position rank exceeds the container rank");

  std::optional<unsigned> indexBitWidth;
  std::size_t dynamicOrdinal = 0;
  std::size_t blockOrdinal = 0;
  const std::size_t expectedDynamicCount =
      llvm::count_if(payload.position, [](std::int64_t component) {
        return component == mlir::ShapedType::kDynamic;
      });
  if (dynamicTokens.size() != expectedDynamicCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector position dynamic operand cardinality does not match its "
        "static projection");

  PositionState state = PositionState::Valid;
  for (auto [axis, component] : llvm::enumerate(payload.position)) {
    std::uint64_t selected = 0;
    if (component == mlir::ShapedType::kDynamic) {
      const Token &token = dynamicTokens[dynamicOrdinal++];
      if (token.valueState == PrimitiveValueState::Poison) {
        state = PositionState::Poison;
      } else if (token.valueState == PrimitiveValueState::Undef) {
        if (state == PositionState::Valid)
          state = PositionState::Undef;
      } else {
        if (!indexBitWidth) {
          auto resolved = loom::getIndexBitWidth(scope);
          if (!resolved)
            return resolved.takeError();
          indexBitWidth = *resolved;
        }
        auto bits = indexTokenBitPattern(token, *indexBitWidth);
        if (!bits)
          return bits.takeError();
        if (bits->isNegative() ||
            bits->uge(static_cast<std::uint64_t>(shape[axis]))) {
          if (state == PositionState::Valid)
            state = PositionState::Undef;
        } else {
          selected = bits->getZExtValue();
        }
      }
    } else {
      if (component < 0 || component >= shape[axis])
        return llvm::createStringError(
            std::errc::invalid_argument,
            "canonical vector position is outside its static dimension");
      selected = static_cast<std::uint64_t>(component);
    }

    if (static_cast<std::uint64_t>(shape[axis]) >
        std::numeric_limits<std::size_t>::max())
      return llvm::createStringError(
          std::errc::value_too_large,
          "vector position dimension exceeds the host size domain");
    const std::size_t dimension = static_cast<std::size_t>(shape[axis]);
    const std::size_t selectedOrdinal = static_cast<std::size_t>(selected);
    if (blockOrdinal >
        (std::numeric_limits<std::size_t>::max() - selectedOrdinal) / dimension)
      return llvm::createStringError(
          std::errc::value_too_large,
          "vector position offset exceeds the host size domain");
    blockOrdinal = blockOrdinal * dimension + selectedOrdinal;
  }
  if (state != PositionState::Valid)
    return ResolvedVectorPosition{state};

  auto laneCount = checkedProduct(shape.drop_front(payload.position.size()),
                                  "vector trailing block");
  if (!laneCount)
    return laneCount.takeError();
  if (blockOrdinal > std::numeric_limits<std::size_t>::max() / *laneCount)
    return llvm::createStringError(
        std::errc::value_too_large,
        "vector position lane offset exceeds the host size domain");
  return ResolvedVectorPosition{PositionState::Valid, blockOrdinal * *laneCount,
                                *laneCount};
}

llvm::Expected<llvm::SmallVector<PrimitiveValue, 8>>
primitiveValues(const Token &token, mlir::Type type, mlir::Operation *scope) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return vectorPrimitiveValues(token, vector, scope);
  unsigned indexWidth = 0;
  if (mlir::isa<mlir::IndexType>(type)) {
    auto resolved = loom::getIndexBitWidth(scope);
    if (!resolved)
      return resolved.takeError();
    indexWidth = *resolved;
  }
  auto value = primitiveValueFromToken(token, type, indexWidth);
  if (!value)
    return value.takeError();
  return llvm::SmallVector<PrimitiveValue, 8>{std::move(*value)};
}

llvm::Expected<Token>
tokenFromPrimitiveValues(llvm::ArrayRef<PrimitiveValue> values, mlir::Type type,
                         mlir::Operation *scope) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return tokenFromVectorPrimitiveValues(values, vector, scope);
  if (values.size() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "scalar structural-vector result has more than one lane");
  return tokenFromPrimitiveValue(values.front(), type);
}

llvm::Expected<Token> exceptionalResult(PositionState state, mlir::Type type) {
  if (state == PositionState::Poison)
    return exceptionalValueToken(PrimitiveValueState::Poison, type);
  if (state == PositionState::Undef)
    return exceptionalValueToken(PrimitiveValueState::Undef, type);
  return llvm::createStringError(
      std::errc::invalid_argument,
      "valid vector position has no exceptional result");
}

llvm::Expected<Token>
evaluateExtract(mlir::vector::ExtractOp op,
                const dataflow::CanonicalActorSchemaProjection &projection,
                llvm::ArrayRef<Token> inputs) {
  const auto *payload =
      std::get_if<dataflow::VectorStaticPositionPayload>(&projection.payload);
  if (!payload || inputs.size() != op->getNumOperands())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector.extract provider received an incompatible actor projection");
  auto position = resolvePosition(*payload, inputs.drop_front(),
                                  op.getSourceVectorType(), op);
  if (!position)
    return position.takeError();
  if (position->state != PositionState::Valid)
    return exceptionalResult(position->state, op.getResult().getType());

  auto source = vectorPrimitiveValues(inputs.front(), op.getSourceVectorType(),
                                      op.getOperation());
  if (!source)
    return source.takeError();
  if (position->firstLane > source->size() ||
      position->laneCount > source->size() - position->firstLane)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector.extract selected block exceeds its source lanes");
  return tokenFromPrimitiveValues(
      llvm::ArrayRef(*source).slice(position->firstLane, position->laneCount),
      op.getResult().getType(), op.getOperation());
}

llvm::Expected<Token>
evaluateInsert(mlir::vector::InsertOp op,
               const dataflow::CanonicalActorSchemaProjection &projection,
               llvm::ArrayRef<Token> inputs) {
  const auto *payload =
      std::get_if<dataflow::VectorStaticPositionPayload>(&projection.payload);
  if (!payload || inputs.size() != op->getNumOperands())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector.insert provider received an incompatible actor projection");
  auto position = resolvePosition(*payload, inputs.drop_front(2),
                                  op.getDestVectorType(), op);
  if (!position)
    return position.takeError();
  if (position->state != PositionState::Valid)
    return exceptionalResult(position->state, op.getResult().getType());

  auto inserted = primitiveValues(inputs[0], op.getValueToStoreType(), op);
  if (!inserted)
    return inserted.takeError();
  auto destination =
      vectorPrimitiveValues(inputs[1], op.getDestVectorType(), op);
  if (!destination)
    return destination.takeError();
  if (inserted->size() != position->laneCount ||
      position->firstLane > destination->size() ||
      position->laneCount > destination->size() - position->firstLane)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector.insert selected block does not match its inserted value");
  std::copy(inserted->begin(), inserted->end(),
            destination->begin() + position->firstLane);
  return tokenFromVectorPrimitiveValues(*destination, op.getDestVectorType(),
                                        op.getOperation());
}

llvm::Expected<std::size_t> leadingBlockLaneCount(mlir::VectorType type) {
  llvm::ArrayRef<std::int64_t> shape = type.getShape();
  return checkedProduct(shape.empty() ? shape : shape.drop_front(),
                        "vector.shuffle leading block");
}

llvm::Expected<Token>
evaluateShuffle(mlir::vector::ShuffleOp op,
                const dataflow::CanonicalActorSchemaProjection &projection,
                llvm::ArrayRef<Token> inputs) {
  const auto *payload =
      std::get_if<dataflow::VectorShuffleMaskPayload>(&projection.payload);
  if (!payload || inputs.size() != 2)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector.shuffle provider received an incompatible actor projection");
  auto lhs = vectorPrimitiveValues(inputs[0], op.getV1VectorType(), op);
  if (!lhs)
    return lhs.takeError();
  auto rhs = vectorPrimitiveValues(inputs[1], op.getV2VectorType(), op);
  if (!rhs)
    return rhs.takeError();
  auto blockLaneCount = leadingBlockLaneCount(op.getV1VectorType());
  if (!blockLaneCount)
    return blockLaneCount.takeError();
  const std::size_t lhsBlocks =
      op.getV1VectorType().getRank() == 0
          ? 1
          : static_cast<std::size_t>(op.getV1VectorType().getShape().front());
  const std::size_t rhsBlocks =
      op.getV2VectorType().getRank() == 0
          ? 1
          : static_cast<std::size_t>(op.getV2VectorType().getShape().front());

  llvm::SmallVector<PrimitiveValue, 8> result;
  if (payload->mask.size() >
      std::numeric_limits<std::size_t>::max() / *blockLaneCount)
    return llvm::createStringError(
        std::errc::value_too_large,
        "vector.shuffle result exceeds the host size domain");
  result.reserve(payload->mask.size() * *blockLaneCount);
  for (std::int64_t selected : payload->mask) {
    if (selected == -1) {
      result.append(*blockLaneCount, PrimitiveValue::poison());
      continue;
    }
    if (selected < 0 ||
        static_cast<std::uint64_t>(selected) >= lhsBlocks + rhsBlocks)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "canonical vector.shuffle mask is outside its source blocks");
    const bool fromLhs = static_cast<std::size_t>(selected) < lhsBlocks;
    const std::size_t block =
        fromLhs ? static_cast<std::size_t>(selected)
                : static_cast<std::size_t>(selected) - lhsBlocks;
    llvm::ArrayRef<PrimitiveValue> source =
        fromLhs ? llvm::ArrayRef(*lhs) : llvm::ArrayRef(*rhs);
    llvm::ArrayRef<PrimitiveValue> selectedBlock =
        source.slice(block * *blockLaneCount, *blockLaneCount);
    result.append(selectedBlock.begin(), selectedBlock.end());
  }
  return tokenFromVectorPrimitiveValues(result, op.getResultVectorType(),
                                        op.getOperation());
}

llvm::Expected<Token> evaluateVectorStructuralActor(
    mlir::Operation *operation,
    const dataflow::CanonicalActorSchemaProjection &projection,
    llvm::ArrayRef<Token> inputs) {
  if (auto extract = mlir::dyn_cast<mlir::vector::ExtractOp>(operation))
    return evaluateExtract(extract, projection, inputs);
  if (auto insert = mlir::dyn_cast<mlir::vector::InsertOp>(operation))
    return evaluateInsert(insert, projection, inputs);
  if (auto shuffle = mlir::dyn_cast<mlir::vector::ShuffleOp>(operation))
    return evaluateShuffle(shuffle, projection, inputs);
  return llvm::createStringError(
      std::errc::invalid_argument,
      "structural-vector provider received the wrong operation class");
}

} // namespace

bool fireVectorStructuralActor(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state) {
  if (state.terminalComputeOps.contains(op))
    return false;
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    if (!hasInputToken(state, operand))
      return false;

  llvm::SmallVector<Token, 4> inputs;
  inputs.reserve(op->getNumOperands());
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    inputs.push_back(peekInputToken(state, operand));
  auto result = evaluateVectorStructuralActor(op, projection, inputs);
  if (!result) {
    state.diagnostics.push_back(llvm::toString(result.takeError()));
    state.terminalComputeOps.insert(op);
    state.failure = RunFailure::ProviderInvariant;
    return false;
  }
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    (void)popInputToken(state, operand);
  emitResultToken(state, 0, std::move(*result));
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
