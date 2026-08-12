#include "Dataflow/IR/DataflowActorSemantics.h"

#include "Common/VectorWidth.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>

namespace {

llvm::Error validateVectorElementRepresentation(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (llvm::isa<mlir::FloatType>(type) || (integer && integer.getWidth() != 0))
    return llvm::Error::success();
  return llvm::createStringError(
      std::errc::invalid_argument,
      "data vector element type must be a nonzero-width integer or "
      "floating-point type");
}

std::uint64_t vectorElementBitWidth(mlir::Type type) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  return llvm::cast<mlir::FloatType>(type).getWidth();
}

std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  type.print(stream);
  return storage;
}

} // namespace

llvm::Expected<mlir::VectorType>
dataflow::semantics::analyzeFixedRankDataVector(mlir::Type type,
                                                VectorRank rank) {
  auto vector = llvm::dyn_cast<mlir::VectorType>(type);
  const bool admitted = vector && !vector.isScalable() &&
                        (rank == VectorRank::AnyFixed ? vector.getRank() > 0
                                                      : vector.getRank() == 1);
  if (!admitted)
    return llvm::createStringError(
        std::errc::invalid_argument, "data vector must be a fixed-size %s",
        rank == VectorRank::AnyFixed ? "vector" : "rank-1 vector");
  if (llvm::Error error =
          validateVectorElementRepresentation(vector.getElementType()))
    return std::move(error);
  return vector;
}

llvm::Expected<std::uint64_t>
dataflow::semantics::getFlattenedVectorBitWidth(mlir::VectorType vector) {
  if (llvm::Error error =
          validateVectorElementRepresentation(vector.getElementType()))
    return std::move(error);
  return loom::getFixedVectorBitWidth(
      vector, vectorElementBitWidth(vector.getElementType()));
}

llvm::Error
dataflow::semantics::validateVectorMaskType(mlir::VectorType dataVector,
                                            mlir::Type maskType) {
  auto mask = llvm::dyn_cast<mlir::VectorType>(maskType);
  if (!mask || mask.isScalable())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mask vector must be a fixed-size vector");
  if (!mask.getElementType().isInteger(1))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mask vector element type must be 'i1'");
  if (mask.getShape() != dataVector.getShape())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mask vector shape '%s' must match data vector shape '%s'",
        typeToString(mask).c_str(), typeToString(dataVector).c_str());
  return llvm::Error::success();
}

bool dataflow::semantics::isStatelessOneTokenVectorBoundary(
    mlir::Operation *op) {
  return op && llvm::isa<dataflow::PackOp, dataflow::UnpackOp>(op);
}

std::optional<mlir::Value>
dataflow::semantics::getVectorBoundaryInputPhase(mlir::Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getScalarPhase();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getGroupPhase();
  return std::nullopt;
}

std::optional<mlir::Value>
dataflow::semantics::getVectorBoundaryOutputPhase(mlir::Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getGroupPhase();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getScalarPhase();
  return std::nullopt;
}

mlir::ValueRange dataflow::semantics::getVectorBoundaryTruePhaseInputPayloads(
    mlir::Operation *op) {
  if (!op || !llvm::isa<dataflow::ParallelizeOp, dataflow::SerializeOp>(op))
    return {};
  return op->getOperands().drop_back();
}

bool dataflow::semantics::isVectorBoundaryTruePhaseOutputPayload(
    mlir::Value value, mlir::Value phase) {
  mlir::Operation *def = value.getDefiningOp();
  if (auto parallelize = llvm::dyn_cast_or_null<dataflow::ParallelizeOp>(def))
    return phase == parallelize.getGroupPhase() &&
           (value == parallelize.getVector() || value == parallelize.getMask());
  if (auto serialize = llvm::dyn_cast_or_null<dataflow::SerializeOp>(def))
    return phase == serialize.getScalarPhase() && value == serialize.getData();
  return false;
}

bool dataflow::semantics::haveEquivalentOrderedCardinality(
    mlir::Value lhsPhase, mlir::Value rhsPhase) {
  if (lhsPhase == rhsPhase)
    return true;
  auto lhsParallelize = lhsPhase.getDefiningOp<dataflow::ParallelizeOp>();
  auto rhsParallelize = rhsPhase.getDefiningOp<dataflow::ParallelizeOp>();
  if (lhsParallelize && rhsParallelize &&
      lhsPhase == lhsParallelize.getGroupPhase() &&
      rhsPhase == rhsParallelize.getGroupPhase()) {
    auto lhsType =
        llvm::dyn_cast<mlir::VectorType>(lhsParallelize.getVector().getType());
    auto rhsType =
        llvm::dyn_cast<mlir::VectorType>(rhsParallelize.getVector().getType());
    return lhsType && rhsType && lhsType.getShape() == rhsType.getShape() &&
           lhsParallelize.getScalarPhase() == rhsParallelize.getScalarPhase();
  }

  auto lhsSerialize = lhsPhase.getDefiningOp<dataflow::SerializeOp>();
  auto rhsSerialize = rhsPhase.getDefiningOp<dataflow::SerializeOp>();
  if (lhsSerialize && rhsSerialize &&
      lhsPhase == lhsSerialize.getScalarPhase() &&
      rhsPhase == rhsSerialize.getScalarPhase()) {
    auto lhsType =
        llvm::dyn_cast<mlir::VectorType>(lhsSerialize.getVector().getType());
    auto rhsType =
        llvm::dyn_cast<mlir::VectorType>(rhsSerialize.getVector().getType());
    return lhsType && rhsType && lhsType.getShape() == rhsType.getShape() &&
           lhsSerialize.getMask() == rhsSerialize.getMask() &&
           lhsSerialize.getGroupPhase() == rhsSerialize.getGroupPhase();
  }
  return false;
}
