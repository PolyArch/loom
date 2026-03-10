//===-- TypeCompat.h - MLIR type width-compatibility helpers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Shared type-compatibility utilities used across the mapper pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_TYPECOMPAT_H
#define LOOM_MAPPER_TYPECOMPAT_H

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace loom {

// Forward declaration for isTemporalPEFU.
class Node;

/// Check if two MLIR types are width-compatible.
/// Allows matching native types (i32, f32) with bits<N> types of the same
/// width. IndexType maps to ADDR_BIT_WIDTH (57) for width comparison.
/// For tagged types, tag types must match AND value widths must match.
inline bool isTypeWidthCompatible(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;

  auto getWidth = [](mlir::Type t) -> std::optional<unsigned> {
    if (auto bits = mlir::dyn_cast<loom::dataflow::BitsType>(t))
      return bits.getWidth();
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(t))
      return intTy.getWidth();
    if (mlir::isa<mlir::Float32Type>(t))
      return 32u;
    if (mlir::isa<mlir::Float64Type>(t))
      return 64u;
    if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(t))
      return 16u;
    if (t.isIndex())
      return static_cast<unsigned>(loom::ADDR_BIT_WIDTH);
    if (mlir::isa<mlir::NoneType>(t))
      return 0u;
    return std::nullopt;
  };

  auto tagA = mlir::dyn_cast<loom::dataflow::TaggedType>(a);
  auto tagB = mlir::dyn_cast<loom::dataflow::TaggedType>(b);
  if (tagA && tagB) {
    if (tagA.getTagType() != tagB.getTagType())
      return false;
    auto wA = getWidth(tagA.getValueType());
    auto wB = getWidth(tagB.getValueType());
    return wA && wB && *wA == *wB;
  }
  if (tagA || tagB)
    return false;

  auto wA = getWidth(a);
  auto wB = getWidth(b);
  return wA && wB && *wA == *wB;
}

/// Check if an ADG node is a temporal PE FU sub-node.
/// Temporal PE FU nodes have a "parent_temporal_pe" attribute set during
/// ADG flattening Phase C.
inline bool isTemporalPEFU(const Node *node) {
  if (!node)
    return false;
  for (auto &attr : node->attributes) {
    if (attr.getName().getValue() == "parent_temporal_pe")
      return true;
  }
  return false;
}

/// Type width check with tagged-type unwrapping for temporal PE FU boundary.
/// If HW type is tagged and SW type is native, unwraps the tagged value type
/// before comparing widths. Per spec: "Each FU type operates on value-only
/// data. Tags are stripped at the boundary."
inline bool isTypeWidthCompatibleForTemporalFU(mlir::Type swType,
                                               mlir::Type hwType) {
  if (isTypeWidthCompatible(swType, hwType))
    return true;

  // Unwrap tagged HW type for comparison with native SW type.
  auto tagHw = mlir::dyn_cast<loom::dataflow::TaggedType>(hwType);
  if (tagHw && !mlir::isa<loom::dataflow::TaggedType>(swType))
    return isTypeWidthCompatible(swType, tagHw.getValueType());

  // Symmetric case: native HW with tagged SW (defensive).
  auto tagSw = mlir::dyn_cast<loom::dataflow::TaggedType>(swType);
  if (tagSw && !tagHw)
    return isTypeWidthCompatible(tagSw.getValueType(), hwType);

  return false;
}

} // namespace loom

#endif // LOOM_MAPPER_TYPECOMPAT_H
