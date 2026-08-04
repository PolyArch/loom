//===- ImplementationFamilyVectorStructure.cpp --------------------------===//

#include "ImplementationFamilyVectorStructure.h"

#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstdint>
#include <limits>

namespace {

using dataflow::CanonicalActorSchemaProjection;
using dataflow::OperationSchemaId;
using fabric::FixedVectorShuffleParams;
using fabric::FixedVectorSliceAlignMergeParams;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<mlir::VectorType>
fixedVector(mlir::Type type, std::uint32_t capacity, llvm::StringRef relation) {
  auto vector = dataflow::semantics::analyzeFixedRankDataVector(
      type, dataflow::semantics::VectorRank::AnyFixed);
  if (!vector)
    return reject(relation + " requires a fixed vector: " +
                  llvm::toString(vector.takeError()));
  auto width = dataflow::semantics::getFlattenedVectorBitWidth(*vector);
  if (!width)
    return reject(relation + " has no finite payload width: " +
                  llvm::toString(width.takeError()));
  if (*width > capacity)
    return reject(relation + " exceeds payload capacity");
  return *vector;
}

llvm::Error admitElement(mlir::Type element,
                         fabric::IntegerWidthSet integerWidths,
                         fabric::FloatFormatSet floatFormats,
                         llvm::StringRef relation) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(element)) {
    std::optional<fabric::IntegerWidth> width;
    switch (integer.getWidth()) {
    case 1:
      width = fabric::IntegerWidth::I1;
      break;
    case 8:
      width = fabric::IntegerWidth::I8;
      break;
    case 16:
      width = fabric::IntegerWidth::I16;
      break;
    case 32:
      width = fabric::IntegerWidth::I32;
      break;
    case 64:
      width = fabric::IntegerWidth::I64;
      break;
    default:
      break;
    }
    if (!integer.isSignless() || !width || !integerWidths.contains(*width))
      return reject(relation + " integer element width is not admitted");
    return llvm::Error::success();
  }

  std::optional<fabric::FloatFormat> format;
  if (element.isF16())
    format = fabric::FloatFormat::F16;
  else if (element.isBF16())
    format = fabric::FloatFormat::BF16;
  else if (element.isF32())
    format = fabric::FloatFormat::F32;
  else if (element.isF64())
    format = fabric::FloatFormat::F64;
  if (!format || !floatFormats.contains(*format))
    return reject(relation + " floating element format is not admitted");
  return llvm::Error::success();
}

llvm::Error verifyElementDomain(fabric::IntegerWidthSet integerWidths,
                                fabric::FloatFormatSet floatFormats,
                                llvm::StringRef relation) {
  if (!integerWidths.valid() || !floatFormats.valid())
    return reject(relation + " element domain is invalid");
  if (integerWidths.empty() && floatFormats.empty())
    return reject(relation + " element domain must not be empty");
  return llvm::Error::success();
}

struct SliceGeometry final {
  mlir::VectorType container;
  mlir::Type slice;
  llvm::ArrayRef<std::int64_t> position;
  unsigned dynamicInputBegin = 0;
};

llvm::Expected<SliceGeometry>
resolveSliceGeometry(const CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<dataflow::VectorStaticPositionPayload>(&actor.payload);
  if (!payload)
    return reject("vector slice actor has the wrong semantic payload");
  if (actor.type.getNumResults() != 1)
    return reject("vector slice actor has the wrong result arity");

  mlir::VectorType container;
  mlir::Type slice;
  unsigned dynamicInputBegin = 0;
  if (actor.schema == OperationSchemaId::VectorExtract) {
    if (actor.type.getNumInputs() == 0)
      return reject("vector extract has no source operand");
    auto resolved = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(0));
    if (!resolved)
      return reject("vector extract source is not a fixed vector");
    container = resolved;
    slice = actor.type.getResult(0);
    dynamicInputBegin = 1;
  } else if (actor.schema == OperationSchemaId::VectorInsert) {
    if (actor.type.getNumInputs() < 2)
      return reject("vector insert has incomplete value operands");
    auto resolved = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(1));
    if (!resolved || actor.type.getResult(0) != resolved)
      return reject("vector insert destination and result do not match");
    container = resolved;
    slice = actor.type.getInput(0);
    dynamicInputBegin = 2;
  } else {
    return reject("slice/align/merge family received a different schema");
  }

  const auto sliceVector = llvm::dyn_cast<mlir::VectorType>(slice);
  const unsigned sliceRank = sliceVector ? sliceVector.getRank() : 0;
  if ((!sliceVector && slice != container.getElementType()) ||
      (sliceVector &&
       sliceVector.getElementType() != container.getElementType()))
    return reject("vector slice element type does not match its container");
  if (sliceRank > container.getRank() ||
      payload->position.size() !=
          static_cast<std::size_t>(container.getRank() - sliceRank))
    return reject("vector slice position does not select the leading rank");
  if (sliceVector && !llvm::equal(sliceVector.getShape(),
                                  container.getShape().take_back(sliceRank)))
    return reject("vector slice shape is not the container trailing shape");

  unsigned dynamicCount = 0;
  for (auto [dimension, position] : llvm::enumerate(payload->position)) {
    if (position == mlir::ShapedType::kDynamic) {
      ++dynamicCount;
      continue;
    }
    if (position < 0 || position >= container.getDimSize(dimension))
      return reject("vector slice static position is out of bounds");
  }
  if (actor.type.getNumInputs() != dynamicInputBegin + dynamicCount)
    return reject("vector slice dynamic position arity is inconsistent");
  return SliceGeometry{container, slice, payload->position, dynamicInputBegin};
}

llvm::Expected<std::uint64_t> semanticWidth(mlir::Type type) {
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type))
    return dataflow::semantics::getFlattenedVectorBitWidth(vector);
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = llvm::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return reject("vector structural slice has an unsupported value type");
}

llvm::Error admitSlice(const FixedVectorSliceAlignMergeParams &params,
                       const CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error =
          verifyElementDomain(params.integerElementWidths,
                              params.floatElementFormats, "vector slice"))
    return error;
  if (params.maxContainerPayloadBits == 0 || params.maxSlicePayloadBits == 0 ||
      params.maxSlicePayloadBits > params.maxContainerPayloadBits)
    return reject("vector slice payload capacities are invalid");
  if (!params.resolvedIndexWidths.valid())
    return reject("vector slice resolved index domain is invalid");
  if ((params.maxDynamicPositionRank == 0) !=
      params.resolvedIndexWidths.empty())
    return reject("vector slice dynamic rank and index domain disagree");

  auto geometry = resolveSliceGeometry(actor);
  if (!geometry)
    return geometry.takeError();
  auto container =
      fixedVector(geometry->container, params.maxContainerPayloadBits,
                  "vector slice container");
  if (!container)
    return container.takeError();
  auto width = semanticWidth(geometry->slice);
  if (!width)
    return width.takeError();
  if (*width > params.maxSlicePayloadBits)
    return reject("vector slice exceeds slice payload capacity");
  if (llvm::Error error = admitElement(
          geometry->container.getElementType(), params.integerElementWidths,
          params.floatElementFormats, "vector slice"))
    return error;

  const unsigned dynamicCount =
      actor.type.getNumInputs() - geometry->dynamicInputBegin;
  if (dynamicCount > params.maxDynamicPositionRank)
    return reject("vector slice exceeds dynamic-position rank capacity");
  for (mlir::Type type :
       actor.type.getInputs().drop_front(geometry->dynamicInputBegin)) {
    auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
    auto resolved =
        integer ? fabric::symbolizeResolvedIndexWidth(integer.getWidth())
                : std::nullopt;
    if (!integer || !integer.isSignless() || !resolved ||
        !params.resolvedIndexWidths.contains(*resolved))
      return reject("vector slice dynamic position width is not admitted");
  }
  return llvm::Error::success();
}

llvm::Error admitShuffle(const FixedVectorShuffleParams &params,
                         const CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error =
          verifyElementDomain(params.integerElementWidths,
                              params.floatElementFormats, "vector shuffle"))
    return error;
  if (params.maxOperandPayloadBits == 0 || params.maxResultPayloadBits == 0 ||
      params.maxBlockPayloadBits == 0 || params.maxSourceBlocks == 0 ||
      params.maxResultBlocks == 0 ||
      params.maxBlockPayloadBits > params.maxOperandPayloadBits ||
      params.maxBlockPayloadBits > params.maxResultPayloadBits ||
      params.maxSourceBlocks < 2)
    return reject("vector shuffle capacities are invalid");
  if (actor.schema != OperationSchemaId::VectorShuffle ||
      actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return reject("shuffle family received a non-shuffle actor");
  const auto *payload =
      std::get_if<dataflow::VectorShuffleMaskPayload>(&actor.payload);
  if (!payload)
    return reject("vector shuffle has the wrong semantic payload");
  auto left = fixedVector(actor.type.getInput(0), params.maxOperandPayloadBits,
                          "left shuffle operand");
  auto right = fixedVector(actor.type.getInput(1), params.maxOperandPayloadBits,
                           "right shuffle operand");
  auto result = fixedVector(actor.type.getResult(0),
                            params.maxResultPayloadBits, "shuffle result");
  if (!left)
    return left.takeError();
  if (!right)
    return right.takeError();
  if (!result)
    return result.takeError();
  if (left->getRank() != right->getRank() ||
      left->getRank() != result->getRank() || left->getRank() == 0 ||
      left->getElementType() != right->getElementType() ||
      left->getElementType() != result->getElementType() ||
      !llvm::equal(left->getShape().drop_front(),
                   right->getShape().drop_front()) ||
      !llvm::equal(left->getShape().drop_front(),
                   result->getShape().drop_front()))
    return reject("vector shuffle trailing block geometry does not match");
  if (llvm::Error error =
          admitElement(left->getElementType(), params.integerElementWidths,
                       params.floatElementFormats, "vector shuffle"))
    return error;

  const auto sourceBlocks = llvm::checkedAddUnsigned<std::uint64_t>(
      left->getDimSize(0), right->getDimSize(0));
  if (!sourceBlocks || *sourceBlocks > params.maxSourceBlocks)
    return reject("vector shuffle exceeds source-block capacity");
  const std::uint64_t resultBlocks = result->getDimSize(0);
  if (resultBlocks > params.maxResultBlocks ||
      payload->mask.size() != resultBlocks)
    return reject("vector shuffle exceeds result-block capacity");
  auto resultWidth = dataflow::semantics::getFlattenedVectorBitWidth(*result);
  if (!resultWidth || resultBlocks == 0)
    return reject("vector shuffle has no finite block geometry");
  const std::uint64_t blockWidth = *resultWidth / resultBlocks;
  if (blockWidth > params.maxBlockPayloadBits)
    return reject("vector shuffle exceeds block payload capacity");
  for (std::int64_t selector : payload->mask)
    if (selector < -1 || (selector >= 0 && static_cast<std::uint64_t>(
                                               selector) >= *sourceBlocks))
      return reject("vector shuffle selector is outside the source domain");
  return llvm::Error::success();
}

} // namespace

llvm::Error fabric::detail::admitFixedVectorSliceAlignMergeAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  return admitSlice(std::get<FixedVectorSliceAlignMergeParams>(capability),
                    actor);
}

llvm::Error fabric::detail::admitFixedVectorShuffleAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  return admitShuffle(std::get<FixedVectorShuffleParams>(capability), actor);
}
