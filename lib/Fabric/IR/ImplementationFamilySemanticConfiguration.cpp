//===- ImplementationFamilySemanticConfiguration.cpp ---------------------===//
//
// Owns direct semantic-field layouts for Fabric's sealed operation
// semantic-field relation.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilyBehaviorInternal.h"

#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

std::uint32_t bitsForCardinality(std::uint32_t cardinality) {
  return cardinality <= 1 ? 0 : llvm::Log2_32_Ceil(cardinality);
}

std::uint32_t bitsForInclusiveMaximum(std::uint32_t maximum) {
  return maximum == std::numeric_limits<std::uint32_t>::max()
             ? 32
             : bitsForCardinality(maximum + 1);
}

llvm::Error appendField(std::uint32_t &cursor, std::uint32_t width) {
  auto next = llvm::checkedAddUnsigned(cursor, width);
  if (!next)
    return reject("semantic configuration layout exceeds uint32");
  cursor = *next;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<fabric::FixedVectorSliceAlignMergeConfigurationLayout>
fabric::resolveFixedVectorSliceAlignMergeConfigurationLayout(
    const FixedVectorSliceAlignMergeParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas) {
  if (params.maxContainerPayloadBits == 0 || params.maxSlicePayloadBits == 0 ||
      params.maxSlicePayloadBits > params.maxContainerPayloadBits)
    return reject("vector slice payload capacities are invalid");
  if (!params.integerElementWidths.valid() ||
      !params.floatElementFormats.valid() ||
      (params.integerElementWidths.empty() &&
       params.floatElementFormats.empty()))
    return reject("vector slice element domain is invalid");
  if (!params.resolvedIndexWidths.valid() ||
      ((params.maxDynamicPositionRank == 0) !=
       params.resolvedIndexWidths.empty()))
    return reject("vector slice dynamic rank and index domain disagree");

  bool extract = false;
  bool insert = false;
  for (::dataflow::OperationSchemaId schema : enabledSchemas) {
    bool *selected = nullptr;
    if (schema == ::dataflow::OperationSchemaId::VectorExtract)
      selected = &extract;
    else if (schema == ::dataflow::OperationSchemaId::VectorInsert)
      selected = &insert;
    else
      return reject("vector slice capability enables a foreign schema");
    if (*selected)
      return reject("vector slice capability enables a schema twice");
    *selected = true;
  }
  if (!extract && !insert)
    return reject("vector slice capability has no enabled schema");

  FixedVectorSliceAlignMergeConfigurationLayout layout;
  std::uint32_t cursor = 0;
  layout.encodesMode = extract && insert;
  layout.modeBitOffset = cursor;
  if (layout.encodesMode)
    if (llvm::Error error = appendField(cursor, 1))
      return std::move(error);
  layout.staticOffsetBitOffset = cursor;
  layout.offsetBitCount = bitsForCardinality(params.maxContainerPayloadBits);
  if (llvm::Error error = appendField(cursor, layout.offsetBitCount))
    return std::move(error);
  layout.sliceWidthBitOffset = cursor;
  layout.sliceWidthBitCount = bitsForCardinality(params.maxSlicePayloadBits);
  if (llvm::Error error = appendField(cursor, layout.sliceWidthBitCount))
    return std::move(error);
  layout.dynamicStrideBitOffset = cursor;
  layout.dynamicStrideCount = params.maxDynamicPositionRank;
  layout.dynamicStrideBitCount =
      bitsForInclusiveMaximum(params.maxContainerPayloadBits);
  auto strideFieldBits = llvm::checkedMulUnsigned(
      layout.dynamicStrideBitCount, params.maxDynamicPositionRank);
  if (!strideFieldBits)
    return reject("vector slice stride layout exceeds uint32");
  if (llvm::Error error = appendField(cursor, *strideFieldBits))
    return std::move(error);
  layout.encodedBitCount = cursor;
  return layout;
}

llvm::Expected<fabric::FixedVectorShuffleConfigurationLayout>
fabric::resolveFixedVectorShuffleConfigurationLayout(
    const FixedVectorShuffleParams &params) {
  if (!params.integerElementWidths.valid() ||
      !params.floatElementFormats.valid() ||
      (params.integerElementWidths.empty() &&
       params.floatElementFormats.empty()))
    return reject("vector shuffle element domain is invalid");
  if (params.maxOperandPayloadBits == 0 || params.maxResultPayloadBits == 0 ||
      params.maxBlockPayloadBits == 0 ||
      params.maxBlockPayloadBits > params.maxOperandPayloadBits ||
      params.maxBlockPayloadBits > params.maxResultPayloadBits ||
      params.maxSourceBlocks < 2 || params.maxResultBlocks == 0)
    return reject("vector shuffle capacities are invalid");

  FixedVectorShuffleConfigurationLayout layout;
  std::uint32_t cursor = 0;
  layout.blockWidthBitOffset = cursor;
  layout.blockWidthBitCount = bitsForCardinality(params.maxBlockPayloadBits);
  if (llvm::Error error = appendField(cursor, layout.blockWidthBitCount))
    return std::move(error);
  layout.leftBlockCountBitOffset = cursor;
  layout.blockCountBitCount = bitsForCardinality(params.maxSourceBlocks);
  if (llvm::Error error = appendField(cursor, layout.blockCountBitCount))
    return std::move(error);
  layout.resultBlockCountBitOffset = cursor;
  layout.resultBlockCountBitCount = bitsForCardinality(params.maxResultBlocks);
  if (llvm::Error error = appendField(cursor, layout.resultBlockCountBitCount))
    return std::move(error);
  layout.selectorBitOffset = cursor;
  layout.selectorBitCount = bitsForCardinality(params.maxSourceBlocks);
  layout.selectorCount = params.maxResultBlocks;
  auto selectorFieldBits =
      llvm::checkedMulUnsigned(layout.selectorBitCount, layout.selectorCount);
  if (!selectorFieldBits)
    return reject("vector shuffle selector layout exceeds uint32");
  if (llvm::Error error = appendField(cursor, *selectorFieldBits))
    return std::move(error);
  layout.encodedBitCount = cursor;
  return layout;
}
