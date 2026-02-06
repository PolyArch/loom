//===-- FabricTagOps.cpp - Fabric tag operation verifiers -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace loom::fabric;

/// Verify tag type width is in [1, 16].
static LogicalResult verifyTagWidthRange(Operation *op, IntegerType tagType) {
  unsigned w = tagType.getWidth();
  if (w < 1 || w > 16)
    return op->emitOpError("[COMP_TAG_WIDTH_RANGE] tag type width must be "
                           "in [1, 16]; got i")
           << w;
  return success();
}

LogicalResult AddTagOp::verify() {
  auto resultType = dyn_cast<dataflow::TaggedType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !dataflow.tagged type");

  // COMP_TAG_WIDTH_RANGE
  if (failed(verifyTagWidthRange(getOperation(), resultType.getTagType())))
    return failure();

  Type inputType = getValue().getType();
  if (inputType != resultType.getValueType())
    return emitOpError("[COMP_ADD_TAG_VALUE_TYPE_MISMATCH] "
                       "input value type must match result value type; got ")
           << inputType << " vs " << resultType.getValueType();

  // Verify tag attr if present.
  if (auto tagAttr = getTagAttr()) {
    unsigned tagWidth = resultType.getTagType().getWidth();
    // Tag attr type width must match the result tag type width.
    if (tagAttr.getType().getIntOrFloatBitWidth() != tagWidth)
      return emitOpError("[COMP_ADD_TAG_VALUE_OVERFLOW] "
                         "tag attribute type width must match result tag "
                         "type width; got i")
             << tagAttr.getType().getIntOrFloatBitWidth() << " vs i"
             << tagWidth;
    // Tag value must fit in tag width.
    APInt val = tagAttr.getValue();
    if (val.getActiveBits() > tagWidth)
      return emitOpError("[COMP_ADD_TAG_VALUE_OVERFLOW] "
                         "tag value does not fit in i")
             << tagWidth;
  }

  return success();
}

LogicalResult DelTagOp::verify() {
  auto inputType = dyn_cast<dataflow::TaggedType>(getTagged().getType());
  if (!inputType)
    return emitOpError("input must be !dataflow.tagged type");

  // COMP_TAG_WIDTH_RANGE
  if (failed(verifyTagWidthRange(getOperation(), inputType.getTagType())))
    return failure();

  Type resultType = getResult().getType();
  if (resultType != inputType.getValueType())
    return emitOpError("[COMP_DEL_TAG_VALUE_TYPE_MISMATCH] "
                       "result type must match input value type; got ")
           << resultType << " vs " << inputType.getValueType();

  return success();
}

LogicalResult MapTagOp::verify() {
  auto inputType = dyn_cast<dataflow::TaggedType>(getTagged().getType());
  if (!inputType)
    return emitOpError("input must be !dataflow.tagged type");

  auto resultType = dyn_cast<dataflow::TaggedType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !dataflow.tagged type");

  // COMP_TAG_WIDTH_RANGE
  if (failed(verifyTagWidthRange(getOperation(), inputType.getTagType())))
    return failure();
  if (failed(verifyTagWidthRange(getOperation(), resultType.getTagType())))
    return failure();

  if (inputType.getValueType() != resultType.getValueType())
    return emitOpError("[COMP_MAP_TAG_VALUE_TYPE_MISMATCH] "
                       "input and output value types must match; got ")
           << inputType.getValueType() << " vs " << resultType.getValueType();

  int64_t tableSize = getTableSize();
  if (tableSize < 1 || tableSize > 256)
    return emitOpError("[COMP_MAP_TAG_TABLE_SIZE] "
                       "table_size must be in [1, 256]; got ")
           << tableSize;

  if (auto table = getTable()) {
    if (static_cast<int64_t>(table->size()) != tableSize)
      return emitOpError("[COMP_MAP_TAG_TABLE_LENGTH] "
                         "table length must equal table_size; got ")
             << table->size() << " vs " << tableSize;
  }

  return success();
}
